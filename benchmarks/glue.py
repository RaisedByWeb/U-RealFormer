"""
GLUE benchmark runner for RealFormer-Evo encoder.

Usage:
    python benchmarks/glue.py --task sst2 --layers 12 --epochs 3

Requires: datasets, evaluate  (pip install datasets evaluate)
"""

import argparse
import time

import torch
import torch.nn as nn

from u_realformer import RealFormerConfig, RealFormerEncoder


class GlueClassifier(nn.Module):
    def __init__(self, cfg: RealFormerConfig, num_labels: int):
        super().__init__()
        self.encoder = RealFormerEncoder(cfg)
        self.classifier = nn.Linear(cfg.hidden, num_labels)

    def forward(self, input_ids, attention_mask=None):
        h = self.encoder(input_ids, attention_mask=attention_mask)
        return self.classifier(h[:, 0])


def main():
    parser = argparse.ArgumentParser(description="GLUE benchmark")
    parser.add_argument("--task", default="sst2", choices=["sst2", "mnli", "qqp", "mrpc"])
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--skip_k", type=int, default=1)
    args = parser.parse_args()

    cfg = RealFormerConfig(
        hidden=args.hidden,
        heads=args.heads,
        layers=args.layers,
        skip_k=args.skip_k,
    )

    num_labels = {"sst2": 2, "mnli": 3, "qqp": 2, "mrpc": 2}[args.task]
    model = GlueClassifier(cfg, num_labels)
    print(f"[glue] task={args.task}  params={sum(p.numel() for p in model.parameters()):,}")

    # Placeholder training loop — real data loading via HuggingFace `datasets`
    # is left as a TODO so the benchmark file stays dependency-light.
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        dummy_ids = torch.randint(0, cfg.vocab_size, (args.batch_size, 128))
        dummy_mask = torch.ones_like(dummy_ids)
        dummy_labels = torch.randint(0, num_labels, (args.batch_size,))

        logits = model(dummy_ids, dummy_mask)
        loss = loss_fn(logits, dummy_labels)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"  epoch {epoch+1}/{args.epochs}  loss={loss.item():.4f}  "
              f"time={time.perf_counter()-t0:.2f}s")

    print("[glue] done — replace dummy data with HF datasets for real results.")


if __name__ == "__main__":
    main()
