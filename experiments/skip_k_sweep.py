"""
skip_k sweep — measure the effect of propagating score residuals every k layers.

When skip_k=1 every layer receives the residual; higher values create
sparser "score highways" and reduce memory at the cost of less frequent
relational feedback.

Usage:
    python experiments/skip_k_sweep.py --layers 12 --steps 300
"""

import argparse
import time

import torch
import torch.nn as nn

from realformer_evo import RealFormerConfig, RealFormerEncoder


def run_skip_sweep(skip_values, args):
    results = []
    for k in skip_values:
        cfg = RealFormerConfig(
            hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
            layers=args.layers, skip_k=k,
        )
        model = RealFormerEncoder(cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
        loss_fn = nn.CrossEntropyLoss()

        t0 = time.perf_counter()
        model.train()
        for step in range(args.steps):
            ids = torch.randint(0, cfg.vocab_size, (args.batch, 64))
            h = model(ids)
            logits = h[:, 0, :args.heads]
            labels = torch.zeros(args.batch, dtype=torch.long)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        dt = time.perf_counter() - t0
        results.append({"skip_k": k, "final_loss": loss.item(), "time": dt})
        print(f"  skip_k={k:2d}  loss={loss.item():.4f}  time={dt:.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="skip_k sweep")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    skip_values = [k for k in [1, 2, 3, 4, 6] if k <= args.layers]
    print(f"skip_k sweep  layers={args.layers}  steps={args.steps}\n")
    run_skip_sweep(skip_values, args)
    print("\ndone.")


if __name__ == "__main__":
    main()
