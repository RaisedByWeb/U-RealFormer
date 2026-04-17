"""
Training utilities for closing the hallucination gap.

Three composable strategies that expose the model to cross-step score
state (s_row) during training so the gate learns a policy robust to
cached generation:

  A. Segmented BPTT   — ``segmented_step``
  B. Cache dropout     — ``CacheDropoutSchedule``
  C. Self-distillation — ``distillation_step``
"""

from __future__ import annotations

import random
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import TrainingConfig
from .decoder import CausalAttention, RealFormerDecoder

# ── helpers ───────────────────────────────────────────────────────────────────


def pick_split(
    seq_len: int,
    ratio: float = 0.5,
    min_len: int = 16,
    randomise: bool = True,
) -> int:
    """Choose a split position for segmented training.

    When *randomise* is True the split is sampled uniformly from
    ``[min_len, seq_len - min_len]``.  Otherwise it is deterministic
    at ``round(seq_len * ratio)``, clamped to the same bounds.
    """
    lo = min(min_len, seq_len // 2)
    hi = max(seq_len - lo, lo + 1)
    if randomise:
        return random.randint(lo, hi - 1)
    return max(lo, min(hi - 1, round(seq_len * ratio)))


# ── Strategy A: segmented BPTT ────────────────────────────────────────────────


def segmented_step(
    model: RealFormerDecoder,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    cfg: TrainingConfig,
    *,
    include_seg_a_loss: bool = False,
) -> torch.Tensor:
    """One training step with truncated-BPTT segmented forward.

    Splits the sequence at a (randomised) position, runs segment A
    without gradients to populate the cache, detaches, then runs
    segment B **with** the cross-step ``s_row`` signal active.

    Loss is computed on segment B by default.  Set *include_seg_a_loss*
    to ``True`` to add the segment-A loss (no gradients flow through A;
    this just provides a monitoring signal).
    """
    T = input_ids.size(1)
    split = pick_split(T, cfg.segment_ratio, cfg.segment_min_len)
    logits_a, logits_b = model.forward_segmented(input_ids, split)

    V = logits_b.size(-1)
    loss_b = loss_fn(logits_b.reshape(-1, V), labels[:, split:].reshape(-1))

    if include_seg_a_loss:
        loss_a = loss_fn(logits_a.reshape(-1, V), labels[:, :split].reshape(-1))
        return loss_b + loss_a.detach()

    return loss_b


# ── Strategy B: stochastic cache dropout schedule ────────────────────────────


class CacheDropoutSchedule:
    """Linear annealing schedule for cache-dropout probability.

    Updates ``cache_dropout_p`` on every ``CausalAttention`` module in
    the model.  Typical usage: anneal from ``p_start=1.0`` (never see
    cache) down to ``p_end=0.0`` (always see cache) over training.
    """

    def __init__(
        self,
        model: RealFormerDecoder,
        p_start: float = 1.0,
        p_end: float = 0.0,
        total_steps: int = 1000,
    ):
        self.model = model
        self.p_start = p_start
        self.p_end = p_end
        self.total_steps = max(total_steps, 1)
        self._step = 0

    @property
    def current_p(self) -> float:
        t = min(self._step / self.total_steps, 1.0)
        return self.p_start + (self.p_end - self.p_start) * t

    def step(self) -> float:
        """Advance one step and update all CausalAttention modules.

        After calling ``step()``, ``current_p`` equals the value that
        was just written to all modules.
        """
        self._step += 1
        p = self.current_p
        self._set_p(p)
        return p

    def _set_p(self, p: float) -> None:
        for module in self.model.modules():
            if isinstance(module, CausalAttention):
                module.cache_dropout_p = p


def set_cache_dropout(model: nn.Module, p: float) -> None:
    """Set cache_dropout_p on all CausalAttention modules in *model*."""
    for module in model.modules():
        if isinstance(module, CausalAttention):
            module.cache_dropout_p = p


# ── Strategy C: online self-distillation ──────────────────────────────────────


def _kl_div(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """KL(teacher || student) with temperature scaling."""
    s = F.log_softmax(student_logits / temperature, dim=-1)
    t = F.softmax(teacher_logits / temperature, dim=-1)
    return F.kl_div(s, t, reduction="batchmean") * (temperature ** 2)


def distillation_step(
    model: RealFormerDecoder,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    loss_fn: Callable[..., torch.Tensor],
    cfg: TrainingConfig,
) -> torch.Tensor:
    """One training step with online self-distillation.

    The *student* pass is a standard full-sequence forward (no cache).
    The *teacher* pass uses segmented forward with cache, producing
    soft targets that encode the cross-step ``s_row`` signal.  A KL
    divergence loss aligns the student to the teacher on segment B.
    """
    T = input_ids.size(1)
    V = model.cfg.vocab_size

    student_logits = model(input_ids)
    hard_loss = loss_fn(student_logits.reshape(-1, V), labels.reshape(-1))

    if cfg.distill_weight <= 0.0:
        return hard_loss

    split = pick_split(T, cfg.segment_ratio, cfg.segment_min_len)

    with torch.no_grad():
        _, teacher_logits_b = model.forward_segmented(input_ids, split)

    soft_loss = _kl_div(
        student_logits[:, split:],
        teacher_logits_b,
        cfg.distill_temperature,
    )

    return hard_loss + cfg.distill_weight * soft_loss
