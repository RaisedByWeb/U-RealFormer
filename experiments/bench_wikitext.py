"""
Benchmark 2: WikiText-2 character-level language modelling.

Downloads WikiText-2 via HuggingFace ``datasets``, tokenises at
character level (vocab ~200, no external tokeniser), and trains a
causal decoder on next-character prediction.

Design:
  - Character-level, seq_len=128
  - 6L/256H/4heads/64d, 5000 steps, batch=4
  - Linear warmup 5% of steps, then linear decay
  - AdamW, grad clip 1.0
  - Eval on validation split every 500 steps

Usage:
    python experiments/bench_wikitext.py
    python experiments/bench_wikitext.py --steps 2000 --eval_every 200
"""

import argparse
import math
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from realformer_evo import RealFormerConfig, RealFormerDecoder
from realformer_evo.attention import score_norm


# ── data ──────────────────────────────────────────────────────────────────────


def load_wikitext2_chars():
    """Load WikiText-2 and return (train_text, val_text) as strings."""
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", trust_remote_code=True)
    train_text = "\n".join(ds["train"]["text"])
    val_text = "\n".join(ds["validation"]["text"])
    return train_text, val_text


def build_char_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    itos = {i: c for c, i in stoi.items()}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    return torch.tensor([stoi[c] for c in text if c in stoi], dtype=torch.long)


def get_batch(data: torch.Tensor, batch: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(data) - seq_len - 1
    starts = torch.randint(0, max_start, (batch,))
    x = torch.stack([data[s : s + seq_len] for s in starts])
    y = torch.stack([data[s + 1 : s + seq_len + 1] for s in starts])
    return x, y


# ── lr schedule ───────────────────────────────────────────────────────────────


def warmup_then_decay(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return step / max(warmup_steps, 1)
    return max(0.0, 1.0 - (step - warmup_steps) / max(total_steps - warmup_steps, 1))


# ── diagnostics ───────────────────────────────────────────────────────────────


@dataclass
class Snapshot:
    step: int
    train_loss: float
    eval_loss: float
    perplexity: float
    gate_per_layer: list[float]
    gamma_per_layer: list[float]
    residual_norm_per_layer: list[float]
    score_cosine_per_pair: list[float]


@torch.no_grad()
def evaluate(model, data, cfg, batch, seq_len, n_batches=10):
    model.eval()
    total_loss = 0.0
    for _ in range(n_batches):
        ids, labels = get_batch(data, batch, seq_len)
        logits = model(ids)
        total_loss += F.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1)).item()
    model.train()
    avg_loss = total_loss / n_batches
    return avg_loss, math.exp(avg_loss)


@torch.no_grad()
def collect_diagnostics(model, ids, cfg):
    model.eval()
    B, T = ids.shape
    x = model.drop(model.embed(ids) + model.pos(torch.arange(T, device=ids.device).unsqueeze(0)))

    gates, gammas, rnorms, scores_flat = [], [], [], []
    s = None
    for i, layer in enumerate(model.layers):
        s_in = s if i in cfg.residual_layers else None
        gates.append(torch.sigmoid(layer.attn.alpha).mean().item())
        gammas.append(layer.attn.gamma.abs().mean().item())

        if s_in is not None:
            g = torch.sigmoid(layer.attn.alpha).view(1, cfg.heads, 1, 1)
            sc = layer.attn.gamma.view(1, cfg.heads, 1, 1)
            rnorms.append((g * sc * score_norm(s_in)).norm().item())
        else:
            rnorms.append(0.0)

        h, s = layer.attn(layer.norm1(x), s_in)
        x = x + h
        x = x + layer.ff(layer.norm2(x))
        scores_flat.append(s.flatten(1))

    cosines = []
    for i in range(len(scores_flat) - 1):
        cos = F.cosine_similarity(scores_flat[i].flatten(1), scores_flat[i + 1].flatten(1), dim=-1)
        cosines.append(cos.mean().item())

    model.train()
    return gates, gammas, rnorms, cosines


# ── training ──────────────────────────────────────────────────────────────────


def train_model(name, cfg, train_data, val_data, args):
    torch.manual_seed(42)
    model = RealFormerDecoder(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_steps = int(args.steps * 0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_then_decay(s, warmup_steps, args.steps)
    )

    snapshots: list[Snapshot] = []
    model.train()
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        ids, labels = get_batch(train_data, args.batch, args.seq_len)
        logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, cfg.vocab_size), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % args.eval_every == 0 or step == 1:
            eval_loss, ppl = evaluate(model, val_data, cfg, args.batch, args.seq_len)
            diag_ids, _ = get_batch(val_data, 2, args.seq_len)
            gates, gammas, rnorms, cosines = collect_diagnostics(model, diag_ids, cfg)
            snapshots.append(Snapshot(
                step=step, train_loss=loss.item(), eval_loss=eval_loss,
                perplexity=ppl, gate_per_layer=gates, gamma_per_layer=gammas,
                residual_norm_per_layer=rnorms, score_cosine_per_pair=cosines,
            ))
            print(f"  [{name}] step {step:5d}  train={loss.item():.4f}  "
                  f"eval={eval_loss:.4f}  ppl={ppl:.1f}")

    dt = time.perf_counter() - t0
    return snapshots, dt


# ── reporting ─────────────────────────────────────────────────────────────────


def print_report(name, snapshots, dt):
    last = snapshots[-1]
    gate_avg = sum(last.gate_per_layer) / len(last.gate_per_layer)
    cos_avg = sum(last.score_cosine_per_pair) / max(len(last.score_cosine_per_pair), 1)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Final train loss:   {last.train_loss:.4f}")
    print(f"  Final eval loss:    {last.eval_loss:.4f}")
    print(f"  Final perplexity:   {last.perplexity:.1f}")
    print(f"  Avg gate (sigmoid): {gate_avg:.4f}")
    print(f"  Avg score cosine:   {cos_avg:.4f}")
    print(f"  Wall time:          {dt:.1f}s")

    print(f"\n  Convergence curve:")
    for s in snapshots:
        print(f"    step {s.step:5d}  train={s.train_loss:.4f}  eval={s.eval_loss:.4f}  ppl={s.perplexity:.1f}")

    print(f"\n  Per-layer (final):")
    for i, (g, gm, rn) in enumerate(zip(last.gate_per_layer, last.gamma_per_layer, last.residual_norm_per_layer)):
        print(f"    L{i}: gate={g:.4f}  |gamma|={gm:.4f}  ||res||={rn:.4f}")

    if last.score_cosine_per_pair:
        print(f"\n  Score cosine (final):")
        for i, c in enumerate(last.score_cosine_per_pair):
            print(f"    S_{i}<->S_{i+1}: {c:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark: WikiText-2 char-level LM")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()

    print("Loading WikiText-2...")
    train_text, val_text = load_wikitext2_chars()
    stoi, itos = build_char_vocab(train_text + val_text)
    V = len(stoi)
    print(f"  Vocab size: {V} characters")
    print(f"  Train: {len(train_text):,} chars  Val: {len(val_text):,} chars")

    train_data = encode(train_text, stoi)
    val_data = encode(val_text, stoi)

    base_kw = dict(
        hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
        layers=args.layers, seq_len=args.seq_len, vocab_size=V, dropout=0.1,
    )
    cfg_evo = RealFormerConfig(**base_kw)
    cfg_base = RealFormerConfig(**base_kw, residual_layers=set())

    print(f"\nWikiText-2 char-LM  L={args.layers} H={args.hidden} T={args.seq_len} "
          f"steps={args.steps}  warmup={int(args.steps*0.05)}")
    print(f"Baseline: residual_layers={cfg_base.residual_layers}")
    print(f"Evo:      residual_layers={cfg_evo.residual_layers}\n")

    snaps_base, dt_base = train_model("Baseline", cfg_base, train_data, val_data, args)
    snaps_evo, dt_evo = train_model("Evo", cfg_evo, train_data, val_data, args)

    print_report("Baseline (no residual)", snaps_base, dt_base)
    print_report("RealFormer-Evo (residual)", snaps_evo, dt_evo)

    base_final = snaps_base[-1].eval_loss
    evo_final = snaps_evo[-1].eval_loss
    delta = base_final - evo_final
    gate_alive = all(g > 0.1 for g in snaps_evo[-1].gate_per_layer)
    cos_evo = sum(snaps_evo[-1].score_cosine_per_pair) / max(len(snaps_evo[-1].score_cosine_per_pair), 1)
    cos_base = sum(snaps_base[-1].score_cosine_per_pair) / max(len(snaps_base[-1].score_cosine_per_pair), 1)

    print(f"\n{'=' * 60}")
    print(f"  VERDICT")
    print(f"{'=' * 60}")
    print(f"  Eval loss delta (base - evo): {delta:+.4f}")
    print(f"  Perplexity (base vs evo):     {snaps_base[-1].perplexity:.1f} vs {snaps_evo[-1].perplexity:.1f}")
    print(f"  Gate alive (all > 0.1):       {gate_alive}")
    print(f"  Score cosine (evo vs base):   {cos_evo:.4f} vs {cos_base:.4f}")
    if delta > 0.01 and gate_alive:
        print(f"  --> PASSED: residual attention improves on real language data.")
    elif delta > 0:
        print(f"  --> MARGINAL: small advantage.")
    else:
        print(f"  --> INCONCLUSIVE or FAILED: review diagnostics.")
    print("\ndone.")


if __name__ == "__main__":
    main()
