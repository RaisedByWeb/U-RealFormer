"""
RealFormer-Evo causal decoder.

Autoregressive variant with:
  - causal masking
  - KV cache for incremental generation
  - cross-layer *and* cross-step score residuals
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GatedResidualAttention, score_norm, _clip_residual_grad
from .triton_kernels import is_triton_available, fused_residual_attention
from .config import RealFormerConfig


# ── per-layer KV + score-row cache ────────────────────────────────────────────


@dataclass
class LayerCache:
    k: Optional[torch.Tensor] = None       # (B, H, t, D)
    v: Optional[torch.Tensor] = None       # (B, H, t, D)
    s_row: Optional[torch.Tensor] = None   # (B, H, 1, t) last score row

    def detach_(self) -> "LayerCache":
        """Detach all stored tensors in-place (truncated BPTT boundary)."""
        if self.k is not None:
            self.k = self.k.detach()
        if self.v is not None:
            self.v = self.v.detach()
        if self.s_row is not None:
            self.s_row = self.s_row.detach()
        return self


@dataclass
class DecoderCache:
    layers: list[LayerCache] = field(default_factory=list)

    @classmethod
    def empty(cls, n: int) -> "DecoderCache":
        return cls(layers=[LayerCache() for _ in range(n)])

    def detach_(self) -> "DecoderCache":
        """Detach all cached tensors across every layer (BPTT boundary)."""
        for lc in self.layers:
            lc.detach_()
        return self


# ── causal attention (extends GatedResidualAttention with KV cache) ───────────


class CausalAttention(nn.Module):
    """Causal multi-head attention with gated score residuals and KV cache."""

    def __init__(self, cfg: RealFormerConfig):
        super().__init__()
        H, D = cfg.heads, cfg.head_dim
        self.H, self.D = H, D

        self.q = nn.Linear(cfg.hidden, H * D, bias=False)
        self.k = nn.Linear(cfg.hidden, H * D, bias=False)
        self.v = nn.Linear(cfg.hidden, H * D, bias=False)
        self.o = nn.Linear(H * D, cfg.hidden, bias=False)

        self.alpha = nn.Parameter(torch.full((H,), cfg.alpha_init))
        self.gamma = nn.Parameter(torch.full((H,), cfg.gamma_init))
        self.beta = nn.Parameter(torch.full((H,), cfg.beta_init))
        self.drop = cfg.dropout
        self.cache_dropout_p: float = 0.0  # set by TrainingConfig / CacheDropoutSchedule

        self.residual_grad_flow = cfg.residual_grad_flow
        self.residual_grad_clip = cfg.residual_grad_clip

        causal = torch.triu(
            torch.full((cfg.seq_len, cfg.seq_len), float("-inf")), diagonal=1
        )
        self.register_buffer("causal_mask", causal[None, None])

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, t, _ = x.shape
        return x.view(B, t, self.H, self.D).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)

    def forward(
        self,
        x: torch.Tensor,
        s_prev: Optional[torch.Tensor] = None,
        cache: Optional[LayerCache] = None,
    ):
        B, t, _ = x.shape
        Q = self._split(self.q(x))
        K = self._split(self.k(x))
        V = self._split(self.v(x))

        if cache is not None:
            if cache.k is not None:
                K = torch.cat([cache.k, K], dim=2)
                V = torch.cat([cache.v, V], dim=2)
            cache.k, cache.v = K, V

        T = K.size(2)

        # Fused Triton path: Phase 1 (detached), no dropout, no KV cache,
        # full-sequence causal self-attention only (M == N, zero-based positions).
        # Inference-optimised — not used during training with dropout > 0.
        _use_triton = (
            is_triton_available()
            and x.is_cuda
            and not self.residual_grad_flow
            and s_prev is not None
            and cache is None
            and not (self.training and self.drop > 0)
        )

        if _use_triton:
            assert s_prev.dim() == 4 and s_prev.size(1) == self.H
            s_prev_d = s_prev.detach()
            gate = torch.sigmoid(self.alpha)
            out_t, raw = fused_residual_attention(
                Q, K, V, s_prev_d,
                gate, self.gamma, self.beta,
                sm_scale=1.0 / math.sqrt(self.D),
                causal=True,
            )
            return self.o(self._merge(out_t)), raw

        raw = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.D)

        gate = torch.sigmoid(self.alpha).view(1, self.H, 1, 1)
        scale = self.gamma.view(1, self.H, 1, 1)
        shift = self.beta.view(1, self.H, 1, 1)
        r = None

        # Cross-layer residual (vertical path: previous layer's scores)
        if s_prev is not None:
            if not self.residual_grad_flow:
                s_prev = s_prev.detach()
            elif self.residual_grad_clip > 0 and s_prev.requires_grad:
                s_prev = _clip_residual_grad(s_prev, self.residual_grad_clip)
            r = gate * (scale * score_norm(s_prev) + shift)

        # Cross-step residual (temporal path: previous decode step's scores)
        # Independent of cross-layer — fires whenever cache has s_row.
        if cache is not None and cache.s_row is not None:
            drop_s_row = (
                self.training
                and self.cache_dropout_p > 0.0
                and torch.rand(1).item() < self.cache_dropout_p
            )
            if not drop_s_row:
                pad = T - cache.s_row.size(-1)
                prev_row = F.pad(cache.s_row, (0, pad))
                cross_step = (1 - gate) * scale * score_norm(prev_row.detach())
                r = cross_step if r is None else r + cross_step

        if r is not None:
            raw = raw + r

        scores = raw + self.causal_mask[:, :, :t, :T] if t > 1 else raw
        w = F.softmax(scores, dim=-1)
        if self.training:
            w = F.dropout(w, p=self.drop)

        if cache is not None:
            cache.s_row = raw[:, :, -1:, :].detach()

        return self.o(self._merge(torch.matmul(w, V))), raw


# ── decoder layer ─────────────────────────────────────────────────────────────


class DecoderLayer(nn.Module):
    def __init__(self, cfg: RealFormerConfig, idx: int):
        super().__init__()
        self.idx = idx
        self.receives_residual = idx in cfg.residual_layers
        self.attn = CausalAttention(cfg)
        self.norm1 = nn.LayerNorm(cfg.hidden)
        self.norm2 = nn.LayerNorm(cfg.hidden)
        ff = cfg.hidden * 4
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden, ff),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ff, cfg.hidden),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, s_prev=None, cache=None):
        s_in = s_prev if self.receives_residual else None
        h, s = self.attn(self.norm1(x), s_in, cache)
        if hasattr(self, "_audit_hook") and self._audit_hook is not None:
            self._audit_hook(self.idx, s)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x, s


# ── full decoder ──────────────────────────────────────────────────────────────


class RealFormerDecoder(nn.Module):
    """Causal Transformer decoder with gated residual attention."""

    def __init__(self, cfg: RealFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.pos = nn.Embedding(cfg.seq_len, cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i) for i in range(cfg.layers)]
        )
        self.norm = nn.LayerNorm(cfg.hidden)
        self.head = nn.Linear(cfg.hidden, cfg.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        cache: Optional[DecoderCache] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        offset = cache.layers[0].k.size(2) if cache and cache.layers[0].k is not None else 0
        pos_ids = torch.arange(offset, offset + T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos(pos_ids))

        s = None
        for i, layer in enumerate(self.layers):
            lc = cache.layers[i] if cache else None
            x, s = layer(x, s, lc)
        return self.head(self.norm(x))

    def forward_segmented(
        self,
        input_ids: torch.Tensor,
        split_pos: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-segment forward for truncated BPTT training.

        Segment A (tokens ``0..split_pos``) runs without gradients and
        populates a KV + score-row cache.  The cache is then detached
        (truncated BPTT boundary) and segment B (tokens ``split_pos..T``)
        runs with gradients, receiving the cross-step ``s_row`` signal
        that is otherwise absent during standard training.

        Returns ``(logits_a, logits_b)`` so the caller can decide which
        segments to include in the loss.
        """
        seg_a = input_ids[:, :split_pos]
        seg_b = input_ids[:, split_pos:]
        cache = DecoderCache.empty(self.cfg.layers)

        with torch.no_grad():
            logits_a = self.forward(seg_a, cache=cache)

        cache.detach_()
        logits_b = self.forward(seg_b, cache=cache)

        return logits_a, logits_b
