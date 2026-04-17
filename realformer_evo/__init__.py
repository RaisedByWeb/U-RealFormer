"""
RealFormer-Evo — Persistent Relational Attention for Deep Transformers.
"""

__version__ = "0.1.0"

from .config import RealFormerConfig, TrainingConfig
from .attention import GatedResidualAttention, score_norm
from .low_rank import LowRankProjector
from .encoder import RealFormerEncoder
from .decoder import RealFormerDecoder, DecoderCache, LayerCache
from .training import (
    segmented_step,
    distillation_step,
    CacheDropoutSchedule,
    set_cache_dropout,
    pick_split,
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
