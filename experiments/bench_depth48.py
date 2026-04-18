"""
Deep-stack benchmark: 48L/512H on WikiText-103 with multi-seed evaluation.

Tests the core thesis: residual attention should matter more where depth
matters more.  Runs 3 seeds for both baseline and U-RealFormer, reporting:

  - Val perplexity (mean +/- std across seeds)
  - Training loss curves
  - Gradient norm per step (training stability)
  - Per-layer gate diagnostics
  - Early-step sample efficiency (steps to reach target PPL)
  - Fraction of diverged runs

Designed for a single A100 GPU.  Expected runtime: ~18-24h total
(6 runs x 3-4h each).

Usage:
    # Full run (3 seeds, 20K steps each)
    python experiments/bench_depth48.py

    # Quick smoke test
    python experiments/bench_depth48.py --steps 200 --eval_every 100 --seeds 1

    # Custom
    python experiments/bench_depth48.py --layers 24 --seeds 5
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from u_realformer import RealFormerConfig, RealFormerDecoder

# ══════════════════════════════════════════════════════════════════════════════
#  Data (shared with bench_wikitext103.py)
# ══════════════════════════════════════════════════════════════════════════════


def _load_and_tokenise(split: str, seq_len: int) -> torch.Tensor:
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
    def __init__(self, tokens: torch.Tensor, seq_len: int):
        self.seq_len = seq_len
        self.n_chunks = len(tokens) // (seq_len + 1)
        self.data = tokens[: self.n_chunks * (seq_len + 1)].view(
            self.n_chunks, seq_len + 1
        )

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
#  Eval + diagnostics
# ══════════════════════════════════════════════════════════════════════════════


@dataclass
class StepLog:
    step: int
    val_loss: float
    val_ppl: float
    train_loss: float
    grad_norm: float
    lr: float
    tokens_per_sec: float
    gate_per_layer: list[float]
    gamma_per_layer: list[float]


@dataclass
class RunResult:
    name: str
    seed: int
    n_params: int
    total_time: float
    total_tokens: int
    logs: list[StepLog] = field(default_factory=list)
    diverged: bool = False


@torch.no_grad()
def evaluate(
    model: RealFormerDecoder,
    loader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    count = 0
    for i, (ids, labels) in enumerate(loader):
        if i >= max_batches:
            break
        ids, labels = ids.to(device), labels.to(device)
        amp_ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if device.type == "cuda"
            else nullcontext()
        )
        with amp_ctx:
            logits = model(ids)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1)
        )
        total_loss += loss.item()
        count += 1
    avg = total_loss / max(count, 1)
    model.train()
    return avg, math.exp(min(avg, 20.0))


def gate_diagnostics(
    model: RealFormerDecoder,
) -> tuple[list[float], list[float]]:
    gates = [
        torch.sigmoid(layer.attn.alpha).mean().item()
        for layer in model.layers
    ]
    gammas = [
        layer.attn.gamma.abs().mean().item() for layer in model.layers
    ]
    return gates, gammas


# ══════════════════════════════════════════════════════════════════════════════
#  Single training run
# ══════════════════════════════════════════════════════════════════════════════


def train_single(
    name: str,
    cfg: RealFormerConfig,
    seed: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
) -> RunResult:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)

    model = RealFormerDecoder(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    if hasattr(torch, "compile") and args.compile and device.type == "cuda":
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: cosine_with_warmup(s, args.warmup, args.steps),
    )
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    result = RunResult(
        name=name, seed=seed, n_params=n_params, total_time=0, total_tokens=0
    )
    model.train()
    train_iter = iter(train_loader)
    t0 = time.perf_counter()
    tokens_seen = 0
    last_train_loss = 0.0

    print(f"\n  [{name} seed={seed}] {n_params:,} params, training...")

    for step in range(1, args.steps + 1):
        try:
            ids, labels = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            ids, labels = next(train_iter)

        ids, labels = ids.to(device), labels.to(device)
        tokens_seen += ids.numel()

        ctx = (
            torch.amp.autocast("cuda", dtype=torch.float16)
            if use_amp
            else nullcontext()
        )
        with ctx:
            logits = model(ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), labels.view(-1)
            )
            scaled_loss = loss / args.grad_accum

        if scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        last_train_loss = loss.item()

        # Check for divergence
        if math.isnan(last_train_loss) or last_train_loss > 100:
            print(f"  [{name} seed={seed}] DIVERGED at step {step} "
                  f"(loss={last_train_loss})")
            result.diverged = True
            break

        grad_norm_val = 0.0
        if step % args.grad_accum == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            grad_norm_val = nn.utils.clip_grad_norm_(
                model.parameters(), args.grad_clip
            ).item()
            if scaler is not None:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            scheduler.step()

        # Eval
        if step % args.eval_every == 0 or step == 1:
            dt = time.perf_counter() - t0
            tps = tokens_seen / max(dt, 1e-6)
            val_loss, val_ppl = evaluate(
                model, val_loader, device, args.eval_batches
            )
            gates, gammas = gate_diagnostics(model)
            lr = scheduler.get_last_lr()[0]

            log = StepLog(
                step=step,
                val_loss=val_loss,
                val_ppl=val_ppl,
                train_loss=last_train_loss,
                grad_norm=grad_norm_val,
                lr=lr,
                tokens_per_sec=tps,
                gate_per_layer=gates,
                gamma_per_layer=gammas,
            )
            result.logs.append(log)
            print(
                f"    step {step:6d}  val_ppl={val_ppl:7.1f}  "
                f"train={last_train_loss:.4f}  "
                f"gnorm={grad_norm_val:.2f}  tok/s={tps:,.0f}"
            )

    result.total_time = time.perf_counter() - t0
    result.total_tokens = tokens_seen
    return result


# ══════════════════════════════════════════════════════════════════════════════
#  Multi-seed runner
# ══════════════════════════════════════════════════════════════════════════════


def run_multi_seed(
    label: str,
    cfg: RealFormerConfig,
    seeds: list[int],
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
) -> list[RunResult]:
    results = []
    for seed in seeds:
        if device.type == "cuda":
            torch.cuda.empty_cache()
        r = train_single(
            label, cfg, seed, train_loader, val_loader, args, device
        )
        results.append(r)
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  Reporting
# ══════════════════════════════════════════════════════════════════════════════


def summarise(label: str, runs: list[RunResult], args):
    valid = [r for r in runs if not r.diverged]
    diverged = len(runs) - len(valid)

    print(f"\n{'=' * 70}")
    print(f"  {label}  ({len(valid)}/{len(runs)} seeds converged)")
    print(f"{'=' * 70}")

    if not valid:
        print("  ALL RUNS DIVERGED.")
        return

    final_ppls = [r.logs[-1].val_ppl for r in valid]
    final_losses = [r.logs[-1].val_loss for r in valid]
    mean_ppl = statistics.mean(final_ppls)
    std_ppl = statistics.stdev(final_ppls) if len(final_ppls) > 1 else 0.0

    print(f"  Final PPL:        {mean_ppl:.1f} +/- {std_ppl:.1f}")
    print(f"  Final loss:       {statistics.mean(final_losses):.4f}")
    print(f"  Diverged:         {diverged}/{len(runs)}")

    # Convergence curve (mean across seeds at each eval step)
    n_logs = min(len(r.logs) for r in valid)
    print(f"\n  Convergence curve (mean of {len(valid)} seeds):")
    for i in range(n_logs):
        step = valid[0].logs[i].step
        ppls = [r.logs[i].val_ppl for r in valid]
        gnorms = [r.logs[i].grad_norm for r in valid]
        m_ppl = statistics.mean(ppls)
        m_gnorm = statistics.mean(gnorms)
        print(f"    step {step:6d}  ppl={m_ppl:7.1f}  gnorm={m_gnorm:.2f}")

    # Gate diagnostics from first valid run
    if valid[0].logs:
        last = valid[0].logs[-1]
        print(f"\n  Gate diagnostics (seed {valid[0].seed}, final):")
        for i, (g, gm) in enumerate(
            zip(last.gate_per_layer, last.gamma_per_layer)
        ):
            if i < 5 or i >= len(last.gate_per_layer) - 3:
                bar = "#" * int(g * 40)
                print(f"    L{i:2d}  gate={g:.4f}  |gamma|={gm:.4f}  [{bar}]")
            elif i == 5:
                print(f"    ... ({len(last.gate_per_layer) - 8} layers omitted)")

    # Early-step efficiency
    print("\n  Early-step sample efficiency:")
    for target_ppl in [500, 200, 100]:
        steps_to_target = []
        for r in valid:
            for log in r.logs:
                if log.val_ppl <= target_ppl:
                    steps_to_target.append(log.step)
                    break
        if steps_to_target:
            m = statistics.mean(steps_to_target)
            print(f"    PPL <= {target_ppl}: reached at step {m:.0f} (mean)")
        else:
            print(f"    PPL <= {target_ppl}: not reached")

    return mean_ppl, std_ppl


def print_verdict(
    base_runs: list[RunResult],
    evo_runs: list[RunResult],
    base_summary,
    evo_summary,
):
    print(f"\n{'=' * 70}")
    print("  VERDICT")
    print(f"{'=' * 70}")

    if base_summary is None or evo_summary is None:
        print("  Cannot compare — one or both groups fully diverged.")
        return

    b_mean, b_std = base_summary
    e_mean, e_std = evo_summary
    delta = b_mean - e_mean
    pct = delta / b_mean * 100 if b_mean > 0 else 0

    b_div = sum(1 for r in base_runs if r.diverged)
    e_div = sum(1 for r in evo_runs if r.diverged)

    print(f"  Baseline PPL:  {b_mean:.1f} +/- {b_std:.1f}  "
          f"({b_div}/{len(base_runs)} diverged)")
    print(f"  Evo PPL:       {e_mean:.1f} +/- {e_std:.1f}  "
          f"({e_div}/{len(evo_runs)} diverged)")
    print(f"  Delta:         {delta:+.1f} ({pct:+.1f}%)")

    # Throughput
    b_tps = statistics.mean(
        [r.logs[-1].tokens_per_sec for r in base_runs if not r.diverged]
    )
    e_tps = statistics.mean(
        [r.logs[-1].tokens_per_sec for r in evo_runs if not r.diverged]
    )
    overhead = (1 - e_tps / b_tps) * 100
    print(f"  Throughput:    Base={b_tps:,.0f}  Evo={e_tps:,.0f}  "
          f"(overhead={overhead:.1f}%)")

    # Stability
    b_gnorms = [
        r.logs[-1].grad_norm
        for r in base_runs
        if not r.diverged and r.logs
    ]
    e_gnorms = [
        r.logs[-1].grad_norm
        for r in evo_runs
        if not r.diverged and r.logs
    ]
    if b_gnorms and e_gnorms:
        print(f"  Final gnorm:   Base={statistics.mean(b_gnorms):.2f}  "
              f"Evo={statistics.mean(e_gnorms):.2f}")

    if delta > 1.0 and e_div <= b_div:
        print("\n  --> PASSED: U-RealFormer improves PPL at depth.")
    elif delta > 0 and e_div <= b_div:
        print("\n  --> MARGINAL: small improvement.")
    elif e_div < b_div:
        print("\n  --> STABILITY WIN: Evo has fewer diverged runs.")
    else:
        print("\n  --> NO IMPROVEMENT at this depth. "
              "Review architecture or increase steps.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Deep-stack benchmark: 48L multi-seed"
    )
    parser.add_argument("--layers", type=int, default=48)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--steps", type=int, default=20000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad_accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_every", type=int, default=1000)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true", default=True)
    parser.add_argument("--no_compile", action="store_false", dest="compile")
    parser.add_argument("--save_json", type=str, default="")
    args = parser.parse_args()

    seeds = list(range(42, 42 + args.seeds))

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Device: {torch.cuda.get_device_name()}")
        props = torch.cuda.get_device_properties(0)
        mem = getattr(props, "total_memory", None) or getattr(
            props, "total_mem", 0
        )
        print(f"Memory: {mem / 1024**3:.1f} GB")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Device: Apple MPS")
    else:
        device = torch.device("cpu")
        print("Device: CPU")

    print("\nLoading and tokenising WikiText-103...")
    train_tokens = _load_and_tokenise("train", args.seq_len)
    val_tokens = _load_and_tokenise("validation", args.seq_len)
    print(f"  Train: {len(train_tokens):,} tokens")
    print(f"  Val:   {len(val_tokens):,} tokens")

    train_ds = ChunkedLMDataset(train_tokens, args.seq_len)
    val_ds = ChunkedLMDataset(val_tokens, args.seq_len)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    import tiktoken

    VOCAB = tiktoken.get_encoding("gpt2").n_vocab

    base_kw = dict(
        hidden=args.hidden,
        heads=args.heads,
        head_dim=args.head_dim,
        layers=args.layers,
        seq_len=args.seq_len,
        vocab_size=VOCAB,
        dropout=0.1,
    )
    cfg_base = RealFormerConfig(**base_kw, residual_layers=set())
    cfg_evo = RealFormerConfig(**base_kw)

    eff_batch = args.batch_size * args.grad_accum
    print(f"\nExperiment: {args.layers}L/{args.hidden}H  "
          f"{args.steps} steps  batch={eff_batch}  "
          f"seeds={seeds}")
    print("Baseline: residual_layers=set()")
    print(f"Evo:      residual_layers={cfg_evo.residual_layers}")

    # Run baseline seeds
    print(f"\n{'=' * 70}")
    print(f"  BASELINE ({args.seeds} seeds)")
    print(f"{'=' * 70}")
    base_runs = run_multi_seed(
        "Baseline", cfg_base, seeds, train_loader, val_loader, args, device
    )

    # Run Evo seeds
    print(f"\n{'=' * 70}")
    print(f"  U-REALFORMER ({args.seeds} seeds)")
    print(f"{'=' * 70}")
    evo_runs = run_multi_seed(
        "Evo", cfg_evo, seeds, train_loader, val_loader, args, device
    )

    # Summarise
    base_summary = summarise("BASELINE SUMMARY", base_runs, args)
    evo_summary = summarise("U-REALFORMER SUMMARY", evo_runs, args)
    print_verdict(base_runs, evo_runs, base_summary, evo_summary)

    # Save raw results as JSON
    if args.save_json:
        data = {
            "config": {
                "layers": args.layers,
                "hidden": args.hidden,
                "steps": args.steps,
                "seeds": seeds,
            },
            "baseline": [asdict(r) for r in base_runs],
            "evo": [asdict(r) for r in evo_runs],
        }
        with open(args.save_json, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nRaw results saved to {args.save_json}")

    print("\ndone.")


if __name__ == "__main__":
    main()
