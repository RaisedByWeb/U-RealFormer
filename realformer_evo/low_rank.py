"""
Low-rank score compression for memory-efficient residual attention.

When ``rank > 0``, the full (T × T) score matrix is factored as:

    S_approx = P @ Q^T          P: (B, H, T, rank)   Q: (B, H, T, rank)

The factored form is propagated instead of the full matrix, cutting
cross-layer memory from O(T²) to O(T·rank).

This module also supports temporal stride: only every `stride`-th
position is stored, further reducing the cost for long sequences.
"""

import torch
import torch.nn as nn

from .config import RealFormerConfig


class LowRankProjector(nn.Module):
    """Projects full score matrices into a rank-r factored form."""

    def __init__(self, cfg: RealFormerConfig):
        super().__init__()
        self.rank = cfg.rank
        self.stride = cfg.stride
        self.heads = cfg.heads

        if self.rank > 0:
            self.down = nn.Linear(cfg.seq_len, cfg.rank, bias=False)
            self.up = nn.Linear(cfg.rank, cfg.seq_len, bias=False)

    def compress(self, S: torch.Tensor) -> torch.Tensor:
        """(B, H, T, T) → (B, H, T, rank)"""
        if self.rank <= 0:
            return S
        if self.stride > 1:
            S = S[:, :, ::self.stride, :]
        return self.down(S)

    def decompress(self, P: torch.Tensor, T: int) -> torch.Tensor:
        """(B, H, ?, rank) → (B, H, T, T)"""
        if self.rank <= 0:
            return P
        S = self.up(P)
        if self.stride > 1 and S.size(2) != T:
            S = torch.nn.functional.interpolate(
                S.flatten(0, 1), size=(T, S.size(-1)), mode="nearest"
            ).view(S.size(0), S.size(1), T, -1)
        return S[:, :, :T, :T]
