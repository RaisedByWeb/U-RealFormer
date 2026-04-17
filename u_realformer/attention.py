"""
Gated residual attention with ScoreNorm.

Core formula (layer l, Phase 1 — detached):
    S_l = raw_l + sigmoid(alpha) * (gamma * score_norm(S_{l-1}) + beta)

Four per-head learnable parameters:
  alpha  — gate logit (sigmoid controls residual openness)
  gamma  — scale (row-wise contrast of the normalised residual)
  beta   — per-head constant shift.  Since softmax is shift-invariant,
            beta does NOT affect the current layer's attention weights.
            Its effect is on s_out — the raw scores propagated to the
            next layer's residual input — where it acts as a learnable
            bias in the optimisation geometry of cross-layer score flow.
  score_norm  — row-wise standardisation (fp32-safe, biased variance)

Phase 2 (``residual_grad_flow=True``) removes the ``.detach()`` and
applies gradient clipping to the residual path instead.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import RealFormerConfig
from .triton_kernels import fused_residual_attention, is_triton_available

# ── ScoreNorm (FP16-safe) ────────────────────────────────────────────────────


def score_norm(S: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Row-wise zero-mean / unit-variance standardisation.

    Always computes statistics in fp32 to avoid precision loss in fp16.
    Biased variance (matching ``nn.LayerNorm`` behaviour) so we don't
    blow up when the key dimension is 1.
    """
    dtype = S.dtype
    S = S.float()
    mu = S.mean(dim=-1, keepdim=True)
    var = S.var(dim=-1, keepdim=True, unbiased=False)
    out = (S - mu) / (var + eps).sqrt()
    return out.to(dtype)


# ── Residual-path gradient clamp (Phase 2) ───────────────────────────────────


class _ClipResidualGrad(torch.autograd.Function):
    """Identity in the forward pass; clips gradient norm in the backward pass.

    Used on the score residual path when ``residual_grad_flow=True`` to
    prevent the Jacobian product through stacked ScoreNorm layers from
    exploding.  This is a safety clamp, not a learned scaler (Guardrail 1).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, max_norm: float) -> torch.Tensor:
        ctx.max_norm = max_norm
        return x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        max_norm = ctx.max_norm
        norm = grad.norm()
        if norm > max_norm:
            grad = grad * (max_norm / norm)
        return grad, None


def _clip_residual_grad(x: torch.Tensor, max_norm: float) -> torch.Tensor:
    return _ClipResidualGrad.apply(x, max_norm)


# ── Multi-Head Attention with gated score residual ────────────────────────────


class GatedResidualAttention(nn.Module):
    """Multi-head attention with per-head gated score residuals and ScoreNorm."""

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

        self.residual_grad_flow = cfg.residual_grad_flow
        self.residual_grad_clip = cfg.residual_grad_clip

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, t, _ = x.shape
        return x.view(B, t, self.H, self.D).transpose(1, 2)

    def _merge(self, x: torch.Tensor) -> torch.Tensor:
        return x.transpose(1, 2).contiguous().view(x.shape[0], x.shape[2], -1)

    def forward(
        self,
        x: torch.Tensor,
        s_prev: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, H*D)
        s_prev : optional (B, H, T, T) score tensor from the previous layer
        mask : optional additive mask (broadcastable to (B, H, T, T))

        Returns
        -------
        out : (B, T, H*D)
        raw : (B, H, T, T) -- raw scores *before* masking, for the next layer
        """
        Q = self._split(self.q(x))
        K = self._split(self.k(x))
        V = self._split(self.v(x))

        # Fused Triton path: Phase 1 (detached), no dropout, no arbitrary mask,
        # full-sequence self-attention only.  Inference-optimised — not used
        # during training with dropout > 0.
        _use_triton = (
            is_triton_available()
            and x.is_cuda
            and not self.residual_grad_flow
            and s_prev is not None
            and mask is None
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
                causal=False,
            )
            out = self.o(self._merge(out_t))
            return out, raw

        raw = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.D)

        if s_prev is not None:
            if not self.residual_grad_flow:
                s_prev = s_prev.detach()
            elif self.residual_grad_clip > 0 and s_prev.requires_grad:
                s_prev = _clip_residual_grad(s_prev, self.residual_grad_clip)

            gate = torch.sigmoid(self.alpha).view(1, self.H, 1, 1)
            scale = self.gamma.view(1, self.H, 1, 1)
            shift = self.beta.view(1, self.H, 1, 1)
            raw = raw + gate * (scale * score_norm(s_prev) + shift)

        scores = raw + mask if mask is not None else raw
        w = F.softmax(scores, dim=-1)
        if self.training:
            w = F.dropout(w, p=self.drop)

        out = self.o(self._merge(torch.matmul(w, V)))
        return out, raw
