"""
WikiText-103 GPU benchmark — the credibility test.

Trains a 12L/512H (~85M param) causal decoder on WikiText-103 with GPT-2
BPE tokenisation and compares U-RealFormer (full residual) against a
matched baseline (no residual).  Designed for a single A100 GPU.

Output: perplexity comparison table at every eval checkpoint, plus
per-layer gate/gamma diagnostics for the U-RealFormer model.

Usage:
    # Full run (~6-8h on A100)
    python experiments/bench_wikitext103.py

    # Quick smoke test
    python experiments/bench_wikitext103.py --steps 200 --eval_every 100

    # Custom config
    python experiments/bench_wikitext103.py --layers 6 --hidden 256 --steps 5000

Requirements:
    pip install -e ".[bench]"   # installs datasets, tiktoken
"""

from __future__ import annotations

import argparse
import math
import time
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from u_realformer import RealFormerConfig, RealFormerDecoder

# ══════════════════════════════════════════════════════════════════════════════
#  Data
# ══════════════════════════════════════════════════════════════════════════════


def _load_and_tokenise(split: str, seq_len: int) -> torch.Tensor:
    """Load a WikiText-103 split, tokenise with GPT-2 BPE, return token tensor."""
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)

    tokens: list[int] = []
    for row in ds:
        text = row["text"]
        if text.strip():
            tokens.extend(enc.encode_ordinary(text))

    n = len(tokens) // (seq_len + 1) * (seq_len + 1)
    return torch.tensor(tokens[:n], dtype=torch.long)


class ChunkedLMDataset(Dataset):
    """Pre-chunked language-modelling dataset.  Each item is (input, target)."""

    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        self.n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[: self.n_chunks * (seq_len + 1)].view(self.n_chunks, seq_len + 1)

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, idx):
        chunk = self.data[idx]
        return chunk[:-1], chunk[1:]


# ══════════════════════════════════════════════════════════════════════════════
#  LR schedule
# ══════════════════════════════════════════════════════════════════════════════


def cosine_with_warmup(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return step / max(warmup, 1)
    progress = (step - warmup) / max(total - warmup, 1)
    return 0.5 * (1.0 + math.cos(math.pi * progress))


# ══════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class EvalResult:
    step: int
    loss: float
    ppl: float
    tokens_per_sec: float
    gate_per_layer: list[float]
    gamma_per_layer: list[float]


@torch.no_grad()
def evaluate(
    model: RealFormerDecoder,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> tuple[float, float]:
    """Return (avg_loss, perplexity) over up to max_batches."""
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (ids, labels) in enumerate(loader):
        if i >= max_batches:
            break
        ids, labels = ids.to(device), labels.to(device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda" else nullcontext()
        )
        with amp_ctx:
            logits = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
        total_loss += loss.item()
        count += 1
    avg = total_loss / max(count, 1)
    model.train()
    return avg, math.exp(avg)


def gate_diagnostics(model: RealFormerDecoder) -> tuple[list[float], list[float]]:
    gates = [torch.sigmoid(layer.attn.alpha).mean().item() for layer in model.layers]
    gammas = [layer.attn.gamma.abs().mean().item() for layer in model.layers]
    return gates, gammas


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop
# ══════════════════════════════════════════════════════════════════════════════


def train_model(
    name: str,
    cfg: RealFormerConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
) -> list[EvalResult]:
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    model = RealFormerDecoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n[{name}] Parameters: {n_params:,}")

    if hasattr(torch, "compile") and args.compile and device.type == "cuda":
        model = torch.compile(model)
        print(f"[{name}] torch.compile enabled")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, args.warmup, args.steps),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    results: list[EvalResult] = []
    model.train()
    train_iter = iter(train_loader)
    t0 = time.perf_counter()
    tokens_seen = 0

    for step in range(1, args.steps + 1):
        try:
            ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            ids, labels = next(train_iter)

        ids, labels = ids.to(device), labels.to(device)
        tokens_seen += ids.numel()

        ctx = torch.amp.autocast("cuda", dtype=torch.float16) if use_amp else nullcontext()
        with ctx:
            logits = model(ids)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss = loss / args.grad_accum

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if step % args.grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Eval
        if step % args.eval_every == 0 or step == 1:
            dt = time.perf_counter() - t0
            tps = tokens_seen / max(dt, 1e-6)
            val_loss, val_ppl = evaluate(model, val_loader, device, args.eval_batches)
            gates, gammas = gate_diagnostics(model)
            lr = scheduler.get_last_lr()[0]

            result = EvalResult(
                step=step, loss=val_loss, ppl=val_ppl,
                tokens_per_sec=tps, gate_per_layer=gates, gamma_per_layer=gammas,
            )
            results.append(result)
            print(f"  [{name}] step {step:6d}  loss={val_loss:.4f}  ppl={val_ppl:7.1f}  "
                  f"lr={lr:.2e}  tok/s={tps:,.0f}")

    total_time = time.perf_counter() - t0
    print(f"  [{name}] Done. {total_time:.0f}s total, {tokens_seen:,} tokens")
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════════


def print_comparison(base_results: list[EvalResult], evo_results: list[EvalResult]):
    print(f"\n{'=' * 70}")
    print("  PERPLEXITY COMPARISON")
    print(f"{'=' * 70}")
    header = f"{'Step':>8}  {'Baseline PPL':>14}  {'Evo PPL':>10}  {'Delta':>8}  {'% Change':>10}"
    print(header)
    print("-" * len(header))

    for b, e in zip(base_results, evo_results):
        delta = b.ppl - e.ppl
        pct = (delta / b.ppl * 100) if b.ppl > 0 else 0
        marker = " <--" if delta > 0 else ""
        print(f"{b.step:8d}  {b.ppl:14.1f}  {e.ppl:10.1f}  {delta:+8.1f}  {pct:+9.1f}%{marker}")


def print_gate_report(results: list[EvalResult], n_layers: int):
    last = results[-1]
    print(f"\n{'=' * 70}")
    print("  U-REALFORMER GATE DIAGNOSTICS (final checkpoint)")
    print(f"{'=' * 70}")
    for i in range(n_layers):
        g = last.gate_per_layer[i]
        gm = last.gamma_per_layer[i]
        bar = "#" * int(g * 40)
        print(f"  L{i:2d}  gate={g:.4f}  |gamma|={gm:.4f}  [{bar}]")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="WikiText-103 GPU benchmark")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=2000)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_false", dest="compile")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name()}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU (no GPU detected -- this will be slow)")

    # Data
    print("\nLoading and tokenising WikiText-103...")
    train_tokens = _load_and_tokenise("train", args.seq_len)
    val_tokens = _load_and_tokenise("validation", args.seq_len)
    n_train = len(train_tokens) // (args.seq_len + 1)
    print(f"  Train: {len(train_tokens):,} tokens ({n_train:,} chunks)")
    print(f"  Val:   {len(val_tokens):,} tokens ({len(val_tokens) // (args.seq_len + 1):,} chunks)")

    train_ds = ChunkedLMDataset(train_tokens, args.seq_len)
    val_ds = ChunkedLMDataset(val_tokens, args.seq_len)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    # Vocab size from tiktoken
    import tiktoken
    VOCAB = tiktoken.get_encoding("gpt2").n_vocab

    # Configs
    base_kw = dict(
        hidden=args.hidden, heads=args.heads, head_dim=args.head_dim,
        layers=args.layers, seq_len=args.seq_len, vocab_size=VOCAB, dropout=0.1,
    )
    cfg_base = RealFormerConfig(**base_kw, residual_layers=set())
    cfg_evo = RealFormerConfig(**base_kw)

    print(f"\nModel: {args.layers}L/{args.hidden}H/{args.heads}heads  seq_len={args.seq_len}")
    print(f"Training: {args.steps} steps, batch {args.batch_size}x{args.grad_accum}="
          f"{args.batch_size * args.grad_accum}, lr={args.lr}")
    print("Baseline: residual_layers=set()")
    print(f"Evo:      residual_layers={cfg_evo.residual_layers}")

    # Train baseline
    print(f"\n{'=' * 70}")
    print("  TRAINING BASELINE")
    print(f"{'=' * 70}")
    base_results = train_model("Baseline", cfg_base, train_loader, val_loader, args, device)

    # Clear GPU memory
    torch.cuda.empty_cache()

    # Train Evo
    print(f"\n{'=' * 70}")
    print("  TRAINING U-REALFORMER")
    print(f"{'=' * 70}")
    evo_results = train_model("Evo", cfg_evo, train_loader, val_loader, args, device)

    # Report
    print_comparison(base_results, evo_results)
    print_gate_report(evo_results, args.layers)

    # Verdict
    base_final = base_results[-1].ppl
    evo_final = evo_results[-1].ppl
    delta = base_final - evo_final
    pct = delta / base_final * 100 if base_final > 0 else 0

    print(f"\n{'=' * 70}")
    print("  VERDICT")
    print(f"{'=' * 70}")
    print(f"  Final perplexity:  Baseline={base_final:.1f}  Evo={evo_final:.1f}")
    print(f"  Delta:             {delta:+.1f} ({pct:+.1f}%)")
    print(f"  Evo throughput:    {evo_results[-1].tokens_per_sec:,.0f} tok/s")
    print(f"  Base throughput:   {base_results[-1].tokens_per_sec:,.0f} tok/s")

    if delta > 0.5:
        print("\n  --> PASSED: U-RealFormer improves perplexity at scale.")
    elif delta > 0:
        print("\n  --> MARGINAL: small improvement, may need more steps or larger model.")
    else:
        print("\n  --> NO IMPROVEMENT at this scale. Review architecture.")

    print("\ndone.")


if __name__ == "__main__":
    main()
