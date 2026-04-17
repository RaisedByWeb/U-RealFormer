"""
Phase 0 — Prove A: Does residual attention earn its keep?

Three diagnostics in one script:
  0a. Convergence Race   — RealFormer-Evo (skip_k=1) vs baseline (skip_k=L+1)
  0b. Gate & Residual    — sigmoid(alpha), gamma, injected residual norm per layer
  0c. Score Cosine Sim   — cross-layer score similarity (the "smoking gun")

Usage:
    python experiments/prove_residual.py
    python experiments/prove_residual.py --steps 500 --layers 12
"""

import argparse
import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from u_realformer import RealFormerConfig, RealFormerDecoder
from u_realformer.attention import score_norm


@dataclass
class EvalSnapshot:
    step: int
    train_loss: float
    eval_loss: float
    gate_per_layer: list[float]
    gamma_per_layer: list[float]
    residual_norm_per_layer: list[float]
    score_cosine_per_pair: list[float]


def make_model(cfg: RealFormerConfig) -> RealFormerDecoder:
    return RealFormerDecoder(cfg)


@torch.no_grad()
def collect_diagnostics(
    model: RealFormerDecoder, ids: torch.Tensor, labels: torch.Tensor
) -> dict:
    """Run a forward pass and collect gate values, residual norms, and
    cross-layer score cosine similarities."""
    model.eval()
    cfg = model.cfg
    B, T = ids.shape

    x = model.drop(model.embed(ids) + model.pos(torch.arange(T, device=ids.device).unsqueeze(0)))

    gates, gammas, residual_norms, scores_flat = [], [], [], []
    s = None
    for i, layer in enumerate(model.layers):
        gate_val = torch.sigmoid(layer.attn.alpha).mean().item()
        gamma_val = layer.attn.gamma.abs().mean().item()
        gates.append(gate_val)
        gammas.append(gamma_val)

        s_in = s if i in cfg.residual_layers else None
        if s_in is not None:
            g = torch.sigmoid(layer.attn.alpha).view(1, cfg.heads, 1, 1)
            sc = layer.attn.gamma.view(1, cfg.heads, 1, 1)
            r_norm = (g * sc * score_norm(s_in)).norm().item()
        else:
            r_norm = 0.0
        residual_norms.append(r_norm)

        h_normed = layer.norm1(x)
        h, s = layer.attn(h_normed, s_in)
        x = x + h
        x = x + layer.ff(layer.norm2(x))

        scores_flat.append(s.flatten(1))

    cosines = []
    for i in range(len(scores_flat) - 1):
        a = scores_flat[i].flatten(1)
        b = scores_flat[i + 1].flatten(1)
        cos = F.cosine_similarity(a, b, dim=-1).mean().item()
        cosines.append(cos)

    logits = model.head(model.norm(x))
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))

    model.train()
    return {
        "eval_loss": loss.item(),
        "gates": gates,
        "gammas": gammas,
        "residual_norms": residual_norms,
        "cosines": cosines,
    }


def train_and_evaluate(name, cfg, args, eval_ids, eval_labels):
    torch.manual_seed(42)
    model = make_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()
    V = cfg.vocab_size

    snapshots: list[EvalSnapshot] = []

    model.train()
    t0 = time.perf_counter()
    for step in range(1, args.steps + 1):
        ids = torch.randint(0, V, (args.batch, args.seq_len))
        labels = torch.randint(0, V, (args.batch, args.seq_len))
        logits = model(ids)
        loss = loss_fn(logits.reshape(-1, V), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if step % args.eval_every == 0 or step == 1:
            diag = collect_diagnostics(model, eval_ids, eval_labels)
            snap = EvalSnapshot(
                step=step,
                train_loss=loss.item(),
                eval_loss=diag["eval_loss"],
                gate_per_layer=diag["gates"],
                gamma_per_layer=diag["gammas"],
                residual_norm_per_layer=diag["residual_norms"],
                score_cosine_per_pair=diag["cosines"],
            )
            snapshots.append(snap)

    dt = time.perf_counter() - t0
    return snapshots, dt


def print_report(name, snapshots, dt, n_layers):
    last = snapshots[-1]
    gate_avg = sum(last.gate_per_layer) / len(last.gate_per_layer)
    gamma_avg = sum(last.gamma_per_layer) / len(last.gamma_per_layer)
    rnorm_avg = sum(last.residual_norm_per_layer) / max(len(last.residual_norm_per_layer), 1)
    cos_avg = sum(last.score_cosine_per_pair) / max(len(last.score_cosine_per_pair), 1)

    print(f"\n{'=' * 60}")
    print(f"  {name}")
    print(f"{'=' * 60}")
    print(f"  Final train loss:  {last.train_loss:.4f}")
    print(f"  Final eval loss:   {last.eval_loss:.4f}")
    print(f"  Avg gate (sigmoid): {gate_avg:.4f}")
    print(f"  Avg |gamma|:        {gamma_avg:.4f}")
    print(f"  Avg residual norm:  {rnorm_avg:.4f}")
    print(f"  Avg score cosine:   {cos_avg:.4f}")
    print(f"  Wall time:          {dt:.2f}s")

    print(f"\n  Convergence curve:")
    for s in snapshots:
        print(f"    step {s.step:5d}  train={s.train_loss:.4f}  eval={s.eval_loss:.4f}")

    print(f"\n  Per-layer gate openness (final):")
    for i, g in enumerate(last.gate_per_layer):
        print(f"    L{i:2d}: sigmoid(alpha)={g:.4f}  |gamma|={last.gamma_per_layer[i]:.4f}  "
              f"||residual||={last.residual_norm_per_layer[i]:.4f}")

    if last.score_cosine_per_pair:
        print(f"\n  Cross-layer score cosine similarity (final):")
        for i, c in enumerate(last.score_cosine_per_pair):
            print(f"    S_{i} <-> S_{i+1}: {c:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Phase 0: Prove residual attention works")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval_every", type=int, default=50)
    args = parser.parse_args()

    V = 512
    cfg_residual = RealFormerConfig(
        hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
        layers=args.layers, seq_len=args.seq_len, vocab_size=V,
        skip_k=1,
    )
    cfg_baseline = RealFormerConfig(
        hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
        layers=args.layers, seq_len=args.seq_len, vocab_size=V,
        residual_layers=set(),
    )

    torch.manual_seed(99)
    eval_ids = torch.randint(0, V, (args.batch, args.seq_len))
    eval_labels = torch.randint(0, V, (args.batch, args.seq_len))

    print(f"Phase 0: Prove A  |  L={args.layers} H={args.hidden} T={args.seq_len} "
          f"steps={args.steps}")
    print(f"Baseline: residual_layers={cfg_baseline.residual_layers} (no residual)")
    print(f"RealFormer-Evo: residual_layers={cfg_residual.residual_layers} (full residual)")

    snaps_base, dt_base = train_and_evaluate(
        "Baseline (no residual)", cfg_baseline, args, eval_ids, eval_labels
    )
    snaps_real, dt_real = train_and_evaluate(
        "RealFormer-Evo (residual)", cfg_residual, args, eval_ids, eval_labels
    )

    print_report("Baseline (no residual)", snaps_base, dt_base, args.layers)
    print_report("RealFormer-Evo (residual)", snaps_real, dt_real, args.layers)

    # Summary verdict
    base_final = snaps_base[-1].eval_loss
    real_final = snaps_real[-1].eval_loss
    delta = base_final - real_final
    gate_alive = all(g > 0.1 for g in snaps_real[-1].gate_per_layer)
    cos_avg = sum(snaps_real[-1].score_cosine_per_pair) / max(len(snaps_real[-1].score_cosine_per_pair), 1)
    cos_base = sum(snaps_base[-1].score_cosine_per_pair) / max(len(snaps_base[-1].score_cosine_per_pair), 1)

    print(f"\n{'=' * 60}")
    print(f"  VERDICT")
    print(f"{'=' * 60}")
    print(f"  Eval loss delta (base - evo): {delta:+.4f}")
    print(f"  Gate alive (all > 0.1):       {gate_alive}")
    print(f"  Score cosine (evo vs base):   {cos_avg:.4f} vs {cos_base:.4f}")
    if delta > 0 and gate_alive:
        print(f"  --> Problem A PASSED: residual attention earns its keep.")
    else:
        print(f"  --> Problem A INCONCLUSIVE: review diagnostics above.")

    print("\ndone.")


if __name__ == "__main__":
    main()
