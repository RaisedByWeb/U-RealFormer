"""
Benchmark 1: Synthetic copy-shift task.

The second half of each sequence is a copy of the first half.  The model
must learn the relational pattern "position T/2+i attends to position i"
— a structure that is identical across depth.  Residual attention should
preserve this pattern and converge faster than baseline.

Design:
  - Vocab 32, seq_len 32 (first 16 = source, last 16 = target)
  - Loss on target half only
  - 6L/256H/4heads/64d, 5000 steps, batch=8
  - Linear warmup 5% of steps, then linear decay
  - AdamW, grad clip 1.0

Usage:
    python experiments/bench_structured.py
    python experiments/bench_structured.py --steps 2000 --eval_every 200
"""

import argparse
import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from realformer_evo import RealFormerConfig, RealFormerDecoder
from realformer_evo.attention import score_norm


# ── data generation ───────────────────────────────────────────────────────────


def generate_copy_batch(batch: int, half_len: int, vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Generate (input_ids, labels) where the second half copies the first."""
    src = torch.randint(0, vocab, (batch, half_len))
    ids = torch.cat([src, src], dim=1)
    labels = ids.clone()
    labels[:, :half_len] = -100
    return ids, labels


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
    gate_per_layer: list[float]
    gamma_per_layer: list[float]
    residual_norm_per_layer: list[float]
    score_cosine_per_pair: list[float]


@torch.no_grad()
def collect_diagnostics(model, ids, labels, cfg):
    model.eval()
    B, T = ids.shape

    x = model.drop(model.embed(ids) + model.pos(torch.arange(T, device=ids.device).unsqueeze(0)))

    gates, gammas, rnorms, scores_flat = [], [], [], []
    s = None
    for i, layer in enumerate(model.layers):
        s_in = s if i in cfg.residual_layers else None

        gate_val = torch.sigmoid(layer.attn.alpha).mean().item()
        gamma_val = layer.attn.gamma.abs().mean().item()
        gates.append(gate_val)
        gammas.append(gamma_val)

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

    logits = model.head(model.norm(x))
    mask = labels != -100
    loss = F.cross_entropy(logits[mask], labels[mask])
    model.train()
    return loss.item(), gates, gammas, rnorms, cosines


# ── training ──────────────────────────────────────────────────────────────────


def train_model(name, cfg, args):
    torch.manual_seed(42)
    model = RealFormerDecoder(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    warmup_steps = int(args.steps * 0.05)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: warmup_then_decay(s, warmup_steps, args.steps)
    )

    half = args.seq_len // 2
    eval_ids, eval_labels = generate_copy_batch(args.batch, half, cfg.vocab_size)

    snapshots: list[Snapshot] = []
    model.train()
    t0 = time.perf_counter()

    for step in range(1, args.steps + 1):
        ids, labels = generate_copy_batch(args.batch, half, cfg.vocab_size)
        logits = model(ids)
        mask = labels != -100
        loss = F.cross_entropy(logits[mask], labels[mask])
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()

        if step % args.eval_every == 0 or step == 1:
            eval_loss, gates, gammas, rnorms, cosines = collect_diagnostics(
                model, eval_ids, eval_labels, cfg
            )
            snapshots.append(Snapshot(
                step=step, train_loss=loss.item(), eval_loss=eval_loss,
                gate_per_layer=gates, gamma_per_layer=gammas,
                residual_norm_per_layer=rnorms, score_cosine_per_pair=cosines,
            ))

    dt = time.perf_counter() - t0
    return snapshots, dt


# ── reporting ─────────────────────────────────────────────────────────────────


def print_report(name, snapshots, dt):
    last = snapshots[-1]
    gate_avg = sum(last.gate_per_layer) / len(last.gate_per_layer)
    gamma_avg = sum(last.gamma_per_layer) / len(last.gamma_per_layer)
    rnorm_avg = sum(last.residual_norm_per_layer) / max(len(last.residual_norm_per_layer), 1)
    cos_avg = sum(last.score_cosine_per_pair) / max(len(last.score_cosine_per_pair), 1)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Final train loss:   {last.train_loss:.4f}")
    print(f"  Final eval loss:    {last.eval_loss:.4f}")
    print(f"  Avg gate (sigmoid): {gate_avg:.4f}")
    print(f"  Avg |gamma|:        {gamma_avg:.4f}")
    print(f"  Avg residual norm:  {rnorm_avg:.4f}")
    print(f"  Avg score cosine:   {cos_avg:.4f}")
    print(f"  Wall time:          {dt:.1f}s")

    print(f"\n  Convergence curve:")
    for s in snapshots:
        print(f"    step {s.step:5d}  train={s.train_loss:.4f}  eval={s.eval_loss:.4f}")

    print(f"\n  Per-layer (final):")
    for i, (g, gm, rn) in enumerate(zip(last.gate_per_layer, last.gamma_per_layer, last.residual_norm_per_layer)):
        print(f"    L{i}: gate={g:.4f}  |gamma|={gm:.4f}  ||res||={rn:.4f}")

    if last.score_cosine_per_pair:
        print(f"\n  Score cosine (final):")
        for i, c in enumerate(last.score_cosine_per_pair):
            print(f"    S_{i}<->S_{i+1}: {c:.4f}")


# ── main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Benchmark: copy-shift task")
    parser.add_argument("--steps", type=int, default=5000)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=500)
    args = parser.parse_args()

    V = 32
    base_kw = dict(
        hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
        layers=args.layers, seq_len=args.seq_len, vocab_size=V, dropout=0.0,
    )
    cfg_evo = RealFormerConfig(**base_kw)
    cfg_base = RealFormerConfig(**base_kw, residual_layers=set())

    print(f"Copy-shift benchmark  L={args.layers} H={args.hidden} T={args.seq_len} "
          f"steps={args.steps}  warmup={int(args.steps*0.05)}")
    print(f"Baseline: residual_layers={cfg_base.residual_layers}")
    print(f"Evo:      residual_layers={cfg_evo.residual_layers}")

    snaps_base, dt_base = train_model("Baseline", cfg_base, args)
    snaps_evo, dt_evo = train_model("RealFormer-Evo", cfg_evo, args)

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
    print(f"  Gate alive (all > 0.1):       {gate_alive}")
    print(f"  Score cosine (evo vs base):   {cos_evo:.4f} vs {cos_base:.4f}")
    if delta > 0.01 and gate_alive:
        print(f"  --> PASSED: residual attention converges faster on structured data.")
    elif delta > 0:
        print(f"  --> MARGINAL: small advantage, may need more steps.")
    else:
        print(f"  --> INCONCLUSIVE or FAILED: review diagnostics.")
    print("\ndone.")


if __name__ == "__main__":
    main()
