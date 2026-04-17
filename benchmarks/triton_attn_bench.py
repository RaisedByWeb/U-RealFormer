"""
Benchmark: Fused Triton residual attention vs PyTorch reference.

Compares wall-clock time and peak GPU memory for the forward pass at
various sequence lengths. Also includes standard (no-residual) attention
as a reference ceiling.

Usage (requires CUDA GPU):
    python benchmarks/triton_attn_bench.py
    python benchmarks/triton_attn_bench.py --seq_lens 512 1024 2048
"""

import argparse
import gc
import time

import torch
import torch.nn.functional as F

from u_realformer.attention import score_norm
from u_realformer.triton_kernels import (
    is_triton_available,
    fused_residual_attention,
    reference_residual_attention,
)


def bench_reference(Q, K, V, s_prev, gate, scale, shift, sm_scale, causal, warmup=3, rep=10):
    """PyTorch reference path."""
    for _ in range(warmup):
        reference_residual_attention(Q, K, V, s_prev, gate, scale, shift, sm_scale, causal)
    if Q.is_cuda:
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(rep):
        reference_residual_attention(Q, K, V, s_prev, gate, scale, shift, sm_scale, causal)
    if Q.is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / rep
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return dt, peak


def bench_triton(Q, K, V, s_prev, gate, scale, shift, sm_scale, causal, warmup=3, rep=10):
    """Fused Triton path."""
    for _ in range(warmup):
        fused_residual_attention(Q, K, V, s_prev, gate, scale, shift, sm_scale, causal)
    if Q.is_cuda:
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(rep):
        fused_residual_attention(Q, K, V, s_prev, gate, scale, shift, sm_scale, causal)
    if Q.is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / rep
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return dt, peak


def bench_no_residual(Q, K, V, sm_scale, causal, warmup=3, rep=10):
    """Standard attention (no residual) — reference ceiling."""
    def run():
        scores = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale
        if causal:
            mask = torch.triu(torch.full((scores.size(2), scores.size(3)), float("-inf"),
                              device=scores.device), diagonal=1)[None, None]
            scores = scores + mask
        w = F.softmax(scores, dim=-1)
        return torch.matmul(w, V)

    for _ in range(warmup):
        run()
    if Q.is_cuda:
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    for _ in range(rep):
        run()
    if Q.is_cuda:
        torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / rep
    peak = torch.cuda.max_memory_allocated() / 1024**2
    return dt, peak


def main():
    parser = argparse.ArgumentParser(description="Triton attention benchmark")
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[256, 512, 1024, 2048])
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--head_dim", type=int, default=64)
    parser.add_argument("--causal", action="store_true")
    parser.add_argument("--dtype", choices=["fp32", "fp16"], default="fp16")
    args = parser.parse_args()

    if not is_triton_available():
        print("Triton + CUDA not available. Exiting.")
        return

    device = "cuda"
    dtype = torch.float16 if args.dtype == "fp16" else torch.float32
    sm_scale = 1.0 / (args.head_dim ** 0.5)

    header = (f"{'T':>6}  {'No-Resid (ms)':>14}  {'PyTorch (ms)':>13}  "
              f"{'Triton (ms)':>12}  {'Speedup':>8}  "
              f"{'Mem PT (MB)':>12}  {'Mem Tri (MB)':>13}")
    print(f"Triton Attention Benchmark  B={args.batch} H={args.heads} "
          f"D={args.head_dim} causal={args.causal} dtype={args.dtype}\n")
    print(header)
    print("-" * len(header))

    for T in args.seq_lens:
        gc.collect()
        torch.cuda.empty_cache()

        Q = torch.randn(args.batch, args.heads, T, args.head_dim, device=device, dtype=dtype)
        K = torch.randn(args.batch, args.heads, T, args.head_dim, device=device, dtype=dtype)
        V = torch.randn(args.batch, args.heads, T, args.head_dim, device=device, dtype=dtype)
        s_prev = torch.randn(args.batch, args.heads, T, T, device=device, dtype=dtype)
        gate = torch.sigmoid(torch.zeros(args.heads, device=device, dtype=dtype))
        scale = torch.full((args.heads,), 0.577, device=device, dtype=dtype)
        shift = torch.zeros(args.heads, device=device, dtype=dtype)

        dt_nr, _ = bench_no_residual(Q, K, V, sm_scale, args.causal)
        dt_pt, mem_pt = bench_reference(Q, K, V, s_prev, gate, scale, shift, sm_scale, args.causal)
        dt_tr, mem_tr = bench_triton(Q, K, V, s_prev, gate, scale, shift, sm_scale, args.causal)

        speedup = dt_pt / max(dt_tr, 1e-9)
        print(f"{T:6d}  {dt_nr*1000:14.2f}  {dt_pt*1000:13.2f}  "
              f"{dt_tr*1000:12.2f}  {speedup:8.2f}x  "
              f"{mem_pt:12.1f}  {mem_tr:13.1f}")

    print("\ndone.")


if __name__ == "__main__":
    main()
