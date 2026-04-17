"""
SQuAD v2 span-extraction benchmark for RealFormer-Evo encoder.

Usage:
    python benchmarks/squad_v2.py --layers 12 --epochs 2
"""

import argparse
import time

import torch
import torch.nn as nn

from realformer_evo import RealFormerConfig, RealFormerEncoder


class SpanExtractor(nn.Module):
    def __init__(self, cfg: RealFormerConfig):
        super().__init__()
        self.encoder = RealFormerEncoder(cfg)
        self.qa_head = nn.Linear(cfg.hidden, 2)  # start / end logits

    def forward(self, input_ids, attention_mask=None):
        h = self.encoder(input_ids, attention_mask=attention_mask)
        logits = self.qa_head(h)
        return logits[:, :, 0], logits[:, :, 1]


def main():
    parser = argparse.ArgumentParser(description="SQuAD v2 benchmark")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=768)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()

    cfg = RealFormerConfig(hidden=args.hidden, layers=args.layers)
    model = SpanExtractor(cfg)
    print(f"[squad_v2] params={sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        t0 = time.perf_counter()
        seq_len = 384
        ids = torch.randint(0, cfg.vocab_size, (args.batch_size, seq_len))
        mask = torch.ones_like(ids)
        start_pos = torch.randint(0, seq_len, (args.batch_size,))
        end_pos = torch.clamp(start_pos + torch.randint(1, 20, (args.batch_size,)), max=seq_len - 1)

        start_logits, end_logits = model(ids, mask)
        loss = loss_fn(start_logits, start_pos) + loss_fn(end_logits, end_pos)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"  epoch {epoch+1}/{args.epochs}  loss={loss.item():.4f}  "
              f"time={time.perf_counter()-t0:.2f}s")

    print("[squad_v2] done.")


if __name__ == "__main__":
    main()
