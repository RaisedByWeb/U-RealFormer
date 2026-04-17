"""
Configuration for RealFormer-Evo models.

Covers encoder, decoder, and shared residual-attention parameters.
TrainingConfig holds training-regime knobs (segmented BPTT, cache dropout,
self-distillation) that are separate from the model architecture.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RealFormerConfig:
    hidden: int = 768
    heads: int = 12
    head_dim: int = 64
    layers: int = 12
    seq_len: int = 2048
    vocab_size: int = 30_522
    dropout: float = 0.1

    # ── residual-attention knobs ──────────────────────────────────────────
    skip_k: int = 1               # convenience stride; auto-generates residual_layers if not set
    residual_layers: Optional[set[int]] = None  # explicit set of layer indices that receive s_prev
    alpha_init: float = 0.0       # sigmoid(0) ≈ 0.5; controls initial gate openness
    gamma_init: Optional[float] = None  # defaults to 1/sqrt(3) when None
    beta_init: float = 0.0        # per-head learnable shift after ScoreNorm (0 = no shift)

    # ── detach roadmap (Phase 1 vs Phase 2) ───────────────────────────────
    residual_grad_flow: bool = False  # False = detach (Phase 1); True = coupled (Phase 2)
    residual_grad_clip: float = 1.0   # max norm for residual-path gradient (Phase 2 safety clamp)

    # ── low-rank compression ──────────────────────────────────────────────
    rank: int = 0                 # 0 = full-rank (no compression); >0 = rank-r factored scores
    stride: int = 1               # temporal stride for compressed score history

    @property
    def inner_dim(self) -> int:
        return self.heads * self.head_dim

    def __post_init__(self):
        if self.gamma_init is None:
            self.gamma_init = 1.0 / (3 ** 0.5)

        if self.skip_k <= 0:
            raise ValueError("skip_k must be >= 1")

        if self.residual_layers is None:
            self.residual_layers = {i for i in range(self.layers) if i % self.skip_k == 0}
        else:
            self.residual_layers = set(self.residual_layers)

        invalid = [i for i in self.residual_layers if i < 0 or i >= self.layers]
        if invalid:
            raise ValueError(f"residual_layers contains invalid layer indices: {invalid}")


@dataclass
class TrainingConfig:
    """Training-regime knobs for closing the hallucination gap.

    These are *not* architectural parameters — they control how the model
    is exposed to cross-step score state (s_row) during training so that
    the gate learns a policy robust to cached generation.
    """

    # ── Strategy A: segmented BPTT ────────────────────────────────────────
    segment_ratio: float = 0.5    # where to split (fraction of seq_len); 0 = disabled
    segment_min_len: int = 16     # minimum tokens per segment (avoids degenerate splits)

    # ── Strategy B: stochastic cache dropout ──────────────────────────────
    cache_dropout_p: float = 0.0  # prob of zeroing s_row at segment boundary (0 = disabled)

    # ── Strategy C: online self-distillation ──────────────────────────────
    distill_weight: float = 0.0   # KL coefficient for self-distillation (0 = disabled)
    distill_temperature: float = 2.0  # softmax temperature for distillation targets
