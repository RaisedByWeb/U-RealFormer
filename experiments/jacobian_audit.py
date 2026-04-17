"""
Phase 2 prerequisite — The Jacobian Audit.

Instruments the REAL forward path (via ``_audit_hook`` on each layer)
and measures dL/dS_l at every layer in both modes:

  - ``residual_grad_flow=False``  (detached — Phase 1)
  - ``residual_grad_flow=True``   (coupled  — Phase 2)

Per-layer reports:
  - gradient norm (absolute and depth-normalised)
  - per-head max/min gradient ratio (variance explosion detector)
  - gradient norm ratio (coupled / detached)

Global decision metrics:
  - log-norm slope across depth (vanishing < -0.2, exploding > 0.2)
  - worst-case per-head variance ratio

Usage:
    python experiments/jacobian_audit.py
    python experiments/jacobian_audit.py --layers 24 --heads 8
    python experiments/jacobian_audit.py --layers 48 --hidden 128
"""

import argparse
import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from u_realformer import RealFormerConfig, RealFormerEncoder


@dataclass
class LayerGradInfo:
    layer: int
    grad_norm: float
    grad_norm_normalised: float
    per_head_norms: list[float]
    head_max_min_ratio: float


def _linear_slope(ys: list[float]) -> float:
    """Slope of a least-squares line through (i, y_i)."""
    n = len(ys)
    if n < 2:
        return 0.0
    sx = sum(range(n))
    sy = sum(ys)
    sxx = sum(i * i for i in range(n))
    sxy = sum(i * y for i, y in enumerate(ys))
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-30:
        return 0.0
    return (n * sxy - sx * sy) / denom


def run_audit(cfg: RealFormerConfig, seq_len: int, batch: int) -> list[LayerGradInfo]:
    """Instrument the real forward path via _audit_hook and capture dL/dS_l."""
    torch.manual_seed(42)
    model = RealFormerEncoder(cfg)
    model.train()

    grad_store: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(idx: int, s: torch.Tensor):
            assert s.dim() == 4, f"Expected (B,H,Q,KV), got {s.shape}"
            assert s.size(1) == cfg.heads, f"Head dim mismatch: {s.size(1)} vs {cfg.heads}"
            s.retain_grad()
            def grad_hook(grad):
                grad_store[layer_idx] = grad.detach().clone()
            s.register_hook(grad_hook)
        return hook

    for i, layer in enumerate(model.layers):
        layer._audit_hook = make_hook(i)

    ids = torch.randint(0, cfg.vocab_size, (batch, seq_len))
    out = model(ids)

    probe = torch.randn(out.size(-1), 128)
    logits = out @ probe
    labels = torch.randint(0, 128, (batch, seq_len))
    loss = F.cross_entropy(logits.view(-1, 128), labels.view(-1))
    loss.backward()

    for layer in model.layers:
        layer._audit_hook = None

    raw_norms = []
    for i in range(cfg.layers):
        if i in grad_store:
            raw_norms.append(grad_store[i].norm().item())
        else:
            raw_norms.append(0.0)
    max_norm = max(raw_norms) if raw_norms else 1.0

    results = []
    for i in range(cfg.layers):
        if i in grad_store:
            g = grad_store[i]
            total_norm = g.norm().item()
            normed = total_norm / (max_norm + 1e-12)

            per_head = [g[:, h].norm().item() for h in range(cfg.heads)]
            h_max = max(per_head)
            h_min = min(per_head)
            ratio = h_max / max(h_min, 1e-12)

            results.append(LayerGradInfo(
                layer=i,
                grad_norm=total_norm,
                grad_norm_normalised=normed,
                per_head_norms=per_head,
                head_max_min_ratio=ratio,
            ))
        else:
            results.append(LayerGradInfo(
                layer=i, grad_norm=0.0, grad_norm_normalised=0.0,
                per_head_norms=[], head_max_min_ratio=0.0,
            ))

    return results


def compute_slope(results: list[LayerGradInfo]) -> float:
    """Log-norm slope across depth.  ~0 = stable, <-0.2 = vanishing, >0.2 = exploding."""
    log_norms = [math.log(max(r.grad_norm, 1e-30)) for r in results]
    return _linear_slope(log_norms)


def main():
    parser = argparse.ArgumentParser(description="Jacobian Audit")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    print(f"Jacobian Audit  L={args.layers} H={args.hidden} heads={args.heads}\n")

    cfg_detach = RealFormerConfig(
        hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
        layers=args.layers, seq_len=args.seq_len, vocab_size=256,
        residual_grad_flow=False, dropout=0.0,
    )
    results_detach = run_audit(cfg_detach, args.seq_len, args.batch)
    slope_det = compute_slope(results_detach)

    cfg_coupled = RealFormerConfig(
        hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
        layers=args.layers, seq_len=args.seq_len, vocab_size=256,
        residual_grad_flow=True, residual_grad_clip=1.0, dropout=0.0,
    )
    results_coupled = run_audit(cfg_coupled, args.seq_len, args.batch)
    slope_cpl = compute_slope(results_coupled)

    header = (f"{'Layer':>5}  {'||dL/dS|| det':>14}  {'norm_d':>7}  "
              f"{'||dL/dS|| cpl':>14}  {'norm_c':>7}  "
              f"{'c/d ratio':>10}  {'h-var det':>10}  {'h-var cpl':>10}")
    print(header)
    print("-" * len(header))

    worst_hvar_det = 0.0
    worst_hvar_cpl = 0.0
    for d, c in zip(results_detach, results_coupled):
        ratio = c.grad_norm / max(d.grad_norm, 1e-12)
        worst_hvar_det = max(worst_hvar_det, d.head_max_min_ratio)
        worst_hvar_cpl = max(worst_hvar_cpl, c.head_max_min_ratio)
        print(f"  L{d.layer:2d}   {d.grad_norm:14.6f}  {d.grad_norm_normalised:7.4f}  "
              f"{c.grad_norm:14.6f}  {c.grad_norm_normalised:7.4f}  "
              f"{ratio:10.4f}  {d.head_max_min_ratio:10.2f}  {c.head_max_min_ratio:10.2f}")

    print(f"\n{'=' * 70}")
    print(f"  SLOPE (log-norm decay across depth)")
    print(f"{'=' * 70}")
    print(f"  Detached: {slope_det:+.4f}")
    print(f"  Coupled:  {slope_cpl:+.4f}")
    print(f"  Interpretation:  ~0 = stable,  < -0.2 = vanishing,  > 0.2 = exploding")

    print(f"\n{'=' * 70}")
    print(f"  PER-HEAD VARIANCE (worst-case max/min ratio)")
    print(f"{'=' * 70}")
    print(f"  Detached: {worst_hvar_det:.2f}")
    print(f"  Coupled:  {worst_hvar_cpl:.2f}")
    print(f"  Threshold: > 100 = head-level variance explosion")

    print(f"\n{'=' * 70}")
    print(f"  VERDICT")
    print(f"{'=' * 70}")

    if slope_cpl < -0.2:
        print(f"  Coupled log-norm slope = {slope_cpl:+.4f} → gradients VANISH.")
        print(f"  --> Phase 1 (detach) = production standard.")
    elif slope_cpl > 0.2:
        print(f"  Coupled log-norm slope = {slope_cpl:+.4f} → gradients EXPLODE.")
        print(f"  --> Phase 2 needs stronger clipping.")
    elif worst_hvar_cpl > 100:
        print(f"  Per-head variance ratio = {worst_hvar_cpl:.1f} → head-level EXPLOSION.")
        print(f"  --> Phase 2 needs per-head clipping or head dropout.")
    else:
        print(f"  Coupled gradients are STABLE (slope={slope_cpl:+.4f}, h-var={worst_hvar_cpl:.1f}).")
        print(f"  --> Phase 2 (co-evolution) is viable.")

    print("\ndone.")


if __name__ == "__main__":
    main()
