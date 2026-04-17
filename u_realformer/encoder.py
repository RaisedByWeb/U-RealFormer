"""
RealFormer-Evo encoder — bidirectional (non-causal) variant.

Suitable for GLUE-style classification, span extraction, etc.
Score residuals flow to layers listed in ``cfg.residual_layers``; no causal mask.
"""

from typing import Optional

import torch
import torch.nn as nn

from .attention import GatedResidualAttention
from .config import RealFormerConfig


class EncoderLayer(nn.Module):
    def __init__(self, cfg: RealFormerConfig, idx: int):
        super().__init__()
        self.idx = idx
        self.receives_residual = idx in cfg.residual_layers
        self.attn = GatedResidualAttention(cfg)
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

    def forward(
        self,
        x: torch.Tensor,
        s_prev: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        s_in = s_prev if self.receives_residual else None
        h, s = self.attn(self.norm1(x), s_in, mask=mask)
        if hasattr(self, "_audit_hook") and self._audit_hook is not None:
            self._audit_hook(self.idx, s)
        x = x + h
        x = x + self.ff(self.norm2(x))
        return x, s


class RealFormerEncoder(nn.Module):
    """Bidirectional encoder with gated residual attention."""

    def __init__(self, cfg: RealFormerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.hidden)
        self.pos = nn.Embedding(cfg.seq_len, cfg.hidden)
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            [EncoderLayer(cfg, i) for i in range(cfg.layers)]
        )
        self.norm = nn.LayerNorm(cfg.hidden)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        pos_ids = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.drop(self.embed(input_ids) + self.pos(pos_ids))

        additive_mask = None
        if attention_mask is not None:
            additive_mask = (1.0 - attention_mask[:, None, None, :].float()) * -1e9

        s = None
        for layer in self.layers:
            x, s = layer(x, s, mask=additive_mask)
        return self.norm(x)
