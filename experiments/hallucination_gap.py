"""
Hallucination-gap diagnostic experiment.

Trains a small decoder under four regimes — baseline (no strategy),
Strategy A (segmented BPTT), Strategy A+B (segmented + cache dropout),
and Strategy C (self-distillation) — then measures the train/inference
gap when KV + s_row cache is enabled at generation time.

Usage:
    python experiments/hallucination_gap.py
    python experiments/hallucination_gap.py --steps 500 --layers 6
"""

import argparse
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from realformer_evo import (
    RealFormerConfig,
    TrainingConfig,
    RealFormerDecoder,
    DecoderCache,
    segmented_step,
    distillation_step,
    CacheDropoutSchedule,
    set_cache_dropout,
)


@dataclass
class GapMetrics:
    strategy: str
    final_loss: float
    loss_nocache: float
    loss_cached: float
    gap: float
    s_row_norm: float
    gate_effective: list[float]
    train_time: float


def make_model(cfg: RealFormerConfig) -> RealFormerDecoder:
    return RealFormerDecoder(cfg)


@torch.no_grad()
def eval_nocache(model: RealFormerDecoder, ids: torch.Tensor, labels: torch.Tensor) -> float:
    model.eval()
    logits = model(ids)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1)
    )
    model.train()
    return loss.item()


@torch.no_grad()
def eval_cached(model: RealFormerDecoder, ids: torch.Tensor, labels: torch.Tensor) -> tuple[float, float]:
    """Token-by-token generation with DecoderCache.  Returns (loss, mean_s_row_norm)."""
    model.eval()
    cfg = model.cfg
    B, T = ids.shape
    cache = DecoderCache.empty(cfg.layers)

    total_loss = 0.0
    s_row_norms = []
    for t in range(T):
        logits = model(ids[:, t : t + 1], cache=cache)
        total_loss += nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            labels[:, t].reshape(-1),
        ).item()
        for lc in cache.layers:
            if lc.s_row is not None:
                s_row_norms.append(lc.s_row.norm().item())

    model.train()
    avg_loss = total_loss / T
    avg_norm = sum(s_row_norms) / max(len(s_row_norms), 1)
    return avg_loss, avg_norm


def gate_values(model: RealFormerDecoder) -> list[float]:
    return [
        torch.sigmoid(layer.attn.alpha).mean().item()
        for layer in model.layers
    ]


def train_baseline(model, optimizer, loss_fn, cfg, args):
    for step in range(args.steps):
        ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        labels = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        logits = model(ids)
        loss = loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def train_strategy_a(model, optimizer, loss_fn, cfg, tcfg, args):
    for step in range(args.steps):
        ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        labels = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        loss = segmented_step(model, ids, labels, loss_fn, tcfg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def train_strategy_ab(model, optimizer, loss_fn, cfg, tcfg, args):
    schedule = CacheDropoutSchedule(
        model, p_start=1.0, p_end=0.0, total_steps=args.steps
    )
    for step in range(args.steps):
        schedule.step()
        ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        labels = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        loss = segmented_step(model, ids, labels, loss_fn, tcfg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    set_cache_dropout(model, 0.0)
    return loss.item()


def train_strategy_c(model, optimizer, loss_fn, cfg, tcfg, args):
    for step in range(args.steps):
        ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        labels = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
        loss = distillation_step(model, ids, labels, loss_fn, tcfg)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()


def run_experiment(strategy_name, train_fn, cfg, args, tcfg=None):
    torch.manual_seed(42)
    model = make_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    t0 = time.perf_counter()
    if tcfg is not None:
        final_loss = train_fn(model, optimizer, loss_fn, cfg, tcfg, args)
    else:
        final_loss = train_fn(model, optimizer, loss_fn, cfg, args)
    dt = time.perf_counter() - t0

    eval_ids = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))
    eval_labels = torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len))

    loss_nc = eval_nocache(model, eval_ids, eval_labels)
    loss_c, s_norm = eval_cached(model, eval_ids, eval_labels)

    return GapMetrics(
        strategy=strategy_name,
        final_loss=final_loss,
        loss_nocache=loss_nc,
        loss_cached=loss_c,
        gap=loss_c - loss_nc,
        s_row_norm=s_norm,
        gate_effective=gate_values(model),
        train_time=dt,
    )


def main():
    parser = argparse.ArgumentParser(description="Hallucination-gap diagnostic")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    cfg = RealFormerConfig(
        hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
        layers=args.layers, seq_len=args.seq_len, vocab_size=512,
    )

    tcfg = TrainingConfig(segment_ratio=0.5, segment_min_len=8)
    tcfg_ab = TrainingConfig(segment_ratio=0.5, segment_min_len=8, cache_dropout_p=0.5)
    tcfg_c = TrainingConfig(
        segment_ratio=0.5, segment_min_len=8,
        distill_weight=1.0, distill_temperature=2.0,
    )

    print(f"Hallucination-gap diagnostic  L={args.layers} H={args.hidden} "
          f"T={args.seq_len} steps={args.steps}\n")
    header = f"{'Strategy':<16} {'Loss(nc)':>9} {'Loss(c)':>9} {'Gap':>9} "
    header += f"{'s_row ‖·‖':>10} {'Gate(avg)':>10} {'Time(s)':>8}"
    print(header)
    print("-" * len(header))

    experiments = [
        ("baseline", train_baseline, None),
        ("A: seg-BPTT", train_strategy_a, tcfg),
        ("A+B: +dropout", train_strategy_ab, tcfg_ab),
        ("C: distill", train_strategy_c, tcfg_c),
    ]

    for name, fn, tc in experiments:
        m = run_experiment(name, fn, cfg, args, tcfg=tc)
        g_avg = sum(m.gate_effective) / len(m.gate_effective)
        print(f"{m.strategy:<16} {m.loss_nocache:9.4f} {m.loss_cached:9.4f} "
              f"{m.gap:+9.4f} {m.s_row_norm:10.4f} {g_avg:10.4f} {m.train_time:8.2f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
