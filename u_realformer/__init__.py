"""
RealFormer-Evo — Persistent Relational Attention for Deep Transformers.
"""

__version__ = "0.1.0"

from .attention import GatedResidualAttention, score_norm
from .config import RealFormerConfig, TrainingConfig
from .decoder import DecoderCache, LayerCache, RealFormerDecoder
from .encoder import RealFormerEncoder
from .low_rank import LowRankProjector
from .training import (
    CacheDropoutSchedule,
    distillation_step,
    pick_split,
    segmented_step,
    set_cache_dropout,
)

__all__ = [
    "RealFormerConfig",
    "TrainingConfig",
    "GatedResidualAttention",
    "score_norm",
    "LowRankProjector",
    "RealFormerEncoder",
    "RealFormerDecoder",
    "DecoderCache",
    "LayerCache",
    "segmented_step",
    "distillation_step",
    "CacheDropoutSchedule",
    "set_cache_dropout",
    "pick_split",
]
