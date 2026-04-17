"""
ARCH-021 — Ablation over gate initialisation (alpha_init).

Sweeps alpha_init ∈ {-2, -1, 0, 1, 2} and measures:
  - final training loss after N steps
  - gradient norm of alpha at step 1
  - effective gate value (sigmoid(alpha)) across depth

Usage:
    python experiments/ablation_alpha.py --steps 500 --layers 6
"""

import argparse
import math

import torch
import torch.nn as nn

from realformer_evo import RealFormerConfig, RealFormerEncoder


def run_alpha_sweep(alpha_values, args):
    results = []
    for alpha_init in alpha_values:
        cfg = RealFormerConfig(
            hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
            layers=args.layers, alpha_init=alpha_init,
        )
        model = RealFormerEncoder(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for step in range(args.steps):
            ids = torch.randint(0, cfg.vocab_size, (args.batch, 64))
            h = model(ids)
            logits = h[:, 0, :args.heads]  # tiny proxy task
            labels = torch.zeros(args.batch, dtype=torch.long)
            loss = loss_fn(logits, labels)
            loss.backward()

            if step == 0:
                grad_norm = torch.stack([
                    layer.attn.alpha.grad.norm()
                    for layer in model.layers
                ]).mean().item()

            optimizer.step()
            optimizer.zero_grad()

        gate_values = [
            torch.sigmoid(layer.attn.alpha).mean().item()
            for layer in model.layers
        ]

        results.append({
            "alpha_init": alpha_init,
            "sigmoid_init": 1 / (1 + math.exp(-alpha_init)),
            "final_loss": loss.item(),
            "grad_norm_step0": grad_norm,
            "gate_mean": sum(gate_values) / len(gate_values),
            "gate_per_layer": gate_values,
        })
        print(f"  alpha_init={alpha_init:+.1f}  "
              f"sigmoid={results[-1]['sigmoid_init']:.3f}  "
              f"loss={loss.item():.4f}  "
              f"gate_mean={results[-1]['gate_mean']:.3f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="ARCH-021 alpha ablation")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    print(f"ARCH-021  alpha ablation  layers={args.layers}  steps={args.steps}\n")
    run_alpha_sweep([-2.0, -1.0, 0.0, 1.0, 2.0], args)
    print("\ndone.")


if __name__ == "__main__":
    main()
