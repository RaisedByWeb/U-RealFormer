"""
Memory profiler for RealFormer-Evo (ARCH-010 diagnostics).

Reports peak GPU/CPU memory for encoder and decoder at various depths
and sequence lengths, with and without low-rank compression.

Usage:
    python benchmarks/memory_profile.py
    python benchmarks/memory_profile.py --device cuda --depths 6 12 24
"""

import argparse
import gc
import time

import torch

from realformer_evo import RealFormerConfig, RealFormerEncoder, RealFormerDecoder


def profile_forward(model, input_ids, device, label=""):
    gc.collect()
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(input_ids)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    if device == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
    else:
        peak_mb = float("nan")

    params = sum(p.numel() for p in model.parameters())
    print(f"  {label:30s}  params={params:>12,}  peak={peak_mb:8.1f} MB  time={dt:.3f}s")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--depths", nargs="+", type=int, default=[6, 12])
    parser.add_argument("--seq_lens", nargs="+", type=int, default=[128, 512])
    parser.add_argument("--batch", type=int, default=2)
    args = parser.parse_args()

    device = args.device
    print(f"Memory profile — device={device}\n")

    for depth in args.depths:
        for seq_len in args.seq_lens:
            cfg = RealFormerConfig(
                hidden=256, heads=4, head_dim=64,
                layers=depth, seq_len=seq_len,
            )
            ids = torch.randint(0, cfg.vocab_size, (args.batch, seq_len), device=device)

            enc = RealFormerEncoder(cfg).to(device).eval()
            profile_forward(enc, ids, device, f"encoder L={depth} T={seq_len}")
            del enc

            dec = RealFormerDecoder(cfg).to(device).eval()
            profile_forward(dec, ids, device, f"decoder L={depth} T={seq_len}")
            del dec


if __name__ == "__main__":
    main()
