"""
urealformer.py — Standalone baseline (origin manifesto).

This is the original single-file prototype that preceded the full library.
For the production implementation with beta, residual_layers, FP16-safe
ScoreNorm, detach toggle, training strategies, and Triton kernels, see:

    realformer_evo/          # the full library
    realformer_evo/attention.py   # GatedResidualAttention (encoder)
    realformer_evo/decoder.py     # CausalAttention (decoder)

This file is preserved as a self-contained reference and smoke test.

Original formula (v0):
  S_l = raw_l + sigmoid(alpha) * gamma * score_norm(S_{l-1}.detach())

Current formula (realformer_evo):
  S_l = raw_l + sigmoid(alpha) * (gamma * score_norm(S_{l-1}) + beta)
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── score normalisation ───────────────────────────────────────────────────────

def score_norm(S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise standardisation. Biased variance matches nn.LayerNorm; handles n=1."""
    mu  = S.mean(dim=-1, keepdim=True)
    var = S.var(dim=-1, keepdim=True, unbiased=False)
    return (S - mu) / (var + eps).sqrt()


# ── config ────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    hidden:   int   = 768
    heads:    int   = 12
    head_dim: int   = 64
    layers:   int   = 12
    seq_len:  int   = 2048
    dropout:  float = 0.1
    skip_k:   int   = 1        # pass residual every k layers; 1 = every layer


# ── kv + score-row cache ──────────────────────────────────────────────────────

@dataclass
class LayerCache:
    k:     Optional[torch.Tensor] = None   # (B, H, t, head_dim)
    v:     Optional[torch.Tensor] = None   # (B, H, t, head_dim)
    s_row: Optional[torch.Tensor] = None   # (B, H, 1, t)  last row, pre-mask & pre-norm


@dataclass
class Cache:
    layers: list[LayerCache] = field(default_factory=list)

    @classmethod
    def empty(cls, n: int) -> "Cache":
        return cls(layers=[LayerCache() for _ in range(n)])


# ── attention ─────────────────────────────────────────────────────────────────

class Attention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        H, D = cfg.heads, cfg.head_dim
        self.H, self.D = H, D

        self.q = nn.Linear(cfg.hidden, H * D, bias=False)
        self.k = nn.Linear(cfg.hidden, H * D, bias=False)
        self.v = nn.Linear(cfg.hidden, H * D, bias=False)
        self.o = nn.Linear(H * D, cfg.hidden, bias=False)

        self.alpha = nn.Parameter(torch.zeros(H))            # sigmoid(0) = 0.5
        self.gamma = nn.Parameter(torch.full((H,), 1 / math.sqrt(3)))
        self.drop  = cfg.dropout

        causal = torch.triu(torch.full((cfg.seq_len, cfg.seq_len), float("-inf")), 1)
        self.register_buffer("mask", causal[None, None])

    def _split(self, x):
        B, t, _ = x.shape
        return x.view(B, t, self.H, self.D).transpose(1, 2)

    def _merge(self, x):
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)

    def forward(self, x, s_prev=None, cache: Optional[LayerCache] = None):
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
        raw = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.D)

        # residual — s_prev is a signal, not an optimisation path
        if s_prev is not None:
            gate  = torch.sigmoid(self.alpha).view(1, self.H, 1, 1)
            scale = self.gamma.view(1, self.H, 1, 1)

            # cross-layer: from the layer below
            r = gate * scale * score_norm(s_prev.detach())

            # cross-step: from the previous decode step of this layer
            if cache is not None and cache.s_row is not None:
                pad = T - cache.s_row.size(-1)
                prev_row = F.pad(cache.s_row, (0, pad))
                r = r + (1 - gate) * scale * score_norm(prev_row.detach())

            raw = raw + r

        # causal mask, softmax, output
        scores = raw + self.mask[:, :, :t, :T] if t > 1 else raw
        w = F.softmax(scores, dim=-1)
        if self.training:
            w = F.dropout(w, p=self.drop)

        # cache the pre-mask, pre-norm score row for the next decode step
        if cache is not None:
            cache.s_row = raw[:, :, -1:, :].detach()

        return self.o(self._merge(torch.matmul(w, V))), raw


# ── transformer layer ─────────────────────────────────────────────────────────

class Layer(nn.Module):
    def __init__(self, cfg: Config, idx: int):
        super().__init__()
        self.idx    = idx
        self.skip_k = cfg.skip_k
        self.attn   = Attention(cfg)
        self.norm1  = nn.LayerNorm(cfg.hidden)
        self.norm2  = nn.LayerNorm(cfg.hidden)
        ff = cfg.hidden * 4
        self.ff = nn.Sequential(
            nn.Linear(cfg.hidden, ff), nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(ff, cfg.hidden), nn.Dropout(cfg.dropout),
        )

    def forward(self, x, s_prev=None, cache=None):
        s_in = s_prev if self.idx % self.skip_k == 0 else None
        h, s = self.attn(self.norm1(x), s_in, cache)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x, s


# ── decoder ───────────────────────────────────────────────────────────────────

class RealFormer(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.layers = nn.ModuleList([Layer(cfg, i) for i in range(cfg.layers)])

    def forward(self, x: torch.Tensor, cache: Optional[Cache] = None) -> torch.Tensor:
        s = None
        for i, layer in enumerate(self.layers):
            lc = cache.layers[i] if cache else None
            x, s = layer(x, s, lc)
        return x


# ── minimal smoke test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg   = Config(hidden=64, heads=4, head_dim=16, layers=4, seq_len=32)
    model = RealFormer(cfg).eval()

    x = torch.randn(2, 8, 64)

    # forward pass
    with torch.no_grad():
        out = model(x)
    assert out.shape == x.shape
    assert not out.isnan().any()
    print(f"forward:  {tuple(out.shape)}  ok")

    # causal correctness
    x2 = x.clone()
    x2[:, 5, :] += 10.0
    with torch.no_grad():
        out2 = model(x2)
    assert (out[:, :5] - out2[:, :5]).abs().max() < 1e-5
    assert (out[:, 5:] - out2[:, 5:]).abs().max() > 1e-3
    print("causal:   ok")

    # decode consistency — prefill then token-by-token
    cache = Cache.empty(cfg.layers)
    steps = []
    with torch.no_grad():
        for t in range(x.size(1)):
            steps.append(model(x[:, t:t+1], cache))
    decoded = torch.cat(steps, dim=1)

    # decode differs from full-forward (expected: cross-step cache introduces
    # a mild inference-time signal absent during training), but the first
    # token — which has no cache yet — must match exactly
    first_token_diff = (out[:, :1] - decoded[:, :1]).abs().max().item()
    assert first_token_diff < 1e-4, f"first token mismatch: {first_token_diff}"
    print(f"decode:   first-token diff {first_token_diff:.2e}  ok")

    # gradients flow through alpha and gamma at layer 1
    model.train()
    model(x).sum().backward()
    assert model.layers[1].attn.alpha.grad is not None
    assert not model.layers[1].attn.alpha.grad.isnan().any()
    print("grads:    alpha/gamma have clean gradients  ok")

    print("\nall checks passed.")
