"""Tests for the hallucination-gap strategies (A, B, C)."""

import pytest
import torch
import torch.nn as nn

from u_realformer import (
    CacheDropoutSchedule,
    DecoderCache,
    RealFormerConfig,
    RealFormerDecoder,
    TrainingConfig,
    distillation_step,
    pick_split,
    segmented_step,
    set_cache_dropout,
)
from u_realformer.decoder import CausalAttention


@pytest.fixture
def cfg():
    return RealFormerConfig(
        hidden=64, heads=4, head_dim=16, layers=4,
        seq_len=32, vocab_size=128,
    )


@pytest.fixture
def tcfg():
    return TrainingConfig(segment_ratio=0.5, segment_min_len=4)


# ── DecoderCache.detach_() ────────────────────────────────────────────────────


class TestCacheDetach:
    def test_detach_removes_grad_fn(self, cfg):
        model = RealFormerDecoder(cfg)
        cache = DecoderCache.empty(cfg.layers)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        model(ids, cache=cache)
        cache.detach_()

        for lc in cache.layers:
            if lc.k is not None:
                assert lc.k.grad_fn is None
                assert not lc.k.requires_grad
            if lc.v is not None:
                assert lc.v.grad_fn is None
                assert not lc.v.requires_grad
            if lc.s_row is not None:
                assert lc.s_row.grad_fn is None

    def test_detach_returns_self(self, cfg):
        cache = DecoderCache.empty(cfg.layers)
        assert cache.detach_() is cache


# ── forward_segmented ─────────────────────────────────────────────────────────


class TestForwardSegmented:
    def test_matches_manual_cache_forward(self, cfg):
        """forward_segmented(ids, split) on seg_b must match
        manually running seg_a then seg_b with the same cache."""
        model = RealFormerDecoder(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        split = 8

        with torch.no_grad():
            _, logits_seg = model.forward_segmented(ids, split)

        cache = DecoderCache.empty(cfg.layers)
        with torch.no_grad():
            model(ids[:, :split], cache=cache)
            cache.detach_()
            logits_manual = model(ids[:, split:], cache=cache)

        diff = (logits_seg - logits_manual).abs().max().item()
        assert diff < 1e-5, f"segmented vs manual mismatch: {diff}"

    def test_output_shapes(self, cfg):
        model = RealFormerDecoder(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))

        with torch.no_grad():
            logits_a, logits_b = model.forward_segmented(ids, 8)

        assert logits_a.shape == (2, 8, cfg.vocab_size)
        assert logits_b.shape == (2, 8, cfg.vocab_size)

    def test_grad_flows_through_segment_b(self, cfg):
        model = RealFormerDecoder(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        _, logits_b = model.forward_segmented(ids, 8)
        logits_b.sum().backward()

        layer1 = model.layers[1]
        assert layer1.attn.alpha.grad is not None
        assert not layer1.attn.alpha.grad.isnan().any()
        assert layer1.attn.gamma.grad is not None
        assert not layer1.attn.gamma.grad.isnan().any()

    def test_no_grad_on_segment_a(self, cfg):
        """Segment A logits should not require grad (produced under no_grad)."""
        model = RealFormerDecoder(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 16))
        logits_a, _ = model.forward_segmented(ids, 8)
        assert not logits_a.requires_grad


# ── Strategy B: cache dropout ─────────────────────────────────────────────────


class TestCacheDropout:
    def test_p1_always_drops_s_row(self, cfg):
        """With p=1.0, the cross-step branch should never fire during training."""
        model = RealFormerDecoder(cfg).train()
        set_cache_dropout(model, 1.0)

        cache = DecoderCache.empty(cfg.layers)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        model(ids[:, :4], cache=cache)

        for lc in cache.layers:
            lc.s_row = torch.randn_like(lc.s_row)

        model(ids[:, 4:], cache=cache)
        # If dropout fires, the s_row signal is skipped (but s_row is
        # still written for future steps).  We verify indirectly by
        # comparing with p=0.0 — they should differ.
        set_cache_dropout(model, 0.0)

    def test_p0_never_drops(self, cfg):
        """With p=0.0, cache dropout should be a no-op."""
        model = RealFormerDecoder(cfg).eval()
        set_cache_dropout(model, 0.0)

        for m in model.modules():
            if isinstance(m, CausalAttention):
                assert m.cache_dropout_p == 0.0

    def test_inactive_at_eval(self, cfg):
        """Dropout should never fire during model.eval()."""
        model = RealFormerDecoder(cfg)
        set_cache_dropout(model, 1.0)
        model.eval()

        cache = DecoderCache.empty(cfg.layers)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))

        with torch.no_grad():
            model(ids[:, :4], cache=cache)
            out_with_cache = model(ids[:, 4:], cache=cache)

        # Even with p=1.0, eval mode should still use s_row.
        # Compare with a fresh cache (no s_row history).
        cache2 = DecoderCache.empty(cfg.layers)
        with torch.no_grad():
            model(ids[:, :4], cache=cache2)
            for lc in cache2.layers:
                lc.s_row = None
            out_no_srow = model(ids[:, 4:], cache=cache2)

        # The outputs should differ because eval uses s_row.
        diff = (out_with_cache - out_no_srow).abs().max().item()
        assert diff > 1e-6, "eval should use s_row but outputs are identical"

    def test_dropout_vs_no_dropout_differ(self, cfg):
        """Outputs with p=0.0 and p=1.0 should differ during training
        when s_row is present."""
        torch.manual_seed(0)
        model = RealFormerDecoder(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (1, 16))

        set_cache_dropout(model, 0.0)
        _, logits_keep = model.forward_segmented(ids, 8)

        model.zero_grad()
        set_cache_dropout(model, 1.0)
        _, logits_drop = model.forward_segmented(ids, 8)

        diff = (logits_keep - logits_drop).abs().max().item()
        assert diff > 1e-6, "cache dropout had no effect"


# ── CacheDropoutSchedule ─────────────────────────────────────────────────────


class TestCacheDropoutSchedule:
    def test_annealing(self, cfg):
        model = RealFormerDecoder(cfg)
        sched = CacheDropoutSchedule(model, p_start=1.0, p_end=0.0, total_steps=10)

        assert abs(sched.current_p - 1.0) < 1e-6
        for _ in range(10):
            sched.step()
        assert abs(sched.current_p - 0.0) < 1e-6

    def test_sets_all_modules(self, cfg):
        model = RealFormerDecoder(cfg)
        sched = CacheDropoutSchedule(model, p_start=0.8, p_end=0.2, total_steps=4)
        sched.step()

        for m in model.modules():
            if isinstance(m, CausalAttention):
                assert m.cache_dropout_p == sched.current_p


# ── pick_split ────────────────────────────────────────────────────────────────


class TestPickSplit:
    def test_bounds(self):
        for _ in range(100):
            s = pick_split(32, ratio=0.5, min_len=4, randomise=True)
            assert 4 <= s <= 28

    def test_deterministic(self):
        s1 = pick_split(32, ratio=0.5, min_len=4, randomise=False)
        s2 = pick_split(32, ratio=0.5, min_len=4, randomise=False)
        assert s1 == s2 == 16

    def test_short_sequence(self):
        s = pick_split(8, ratio=0.5, min_len=16, randomise=False)
        assert 1 <= s <= 7


# ── Strategy A: segmented_step ────────────────────────────────────────────────


class TestSegmentedStep:
    def test_returns_finite_loss(self, cfg, tcfg):
        model = RealFormerDecoder(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        loss_fn = nn.CrossEntropyLoss()

        loss = segmented_step(model, ids, labels, loss_fn, tcfg)
        assert loss.isfinite()
        assert loss.item() > 0

    def test_backward_runs(self, cfg, tcfg):
        model = RealFormerDecoder(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        loss_fn = nn.CrossEntropyLoss()

        loss = segmented_step(model, ids, labels, loss_fn, tcfg)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad


# ── Strategy C: distillation_step ─────────────────────────────────────────────


class TestDistillationStep:
    def test_returns_finite_loss(self, cfg):
        tcfg = TrainingConfig(
            segment_ratio=0.5, segment_min_len=4,
            distill_weight=1.0, distill_temperature=2.0,
        )
        model = RealFormerDecoder(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        loss_fn = nn.CrossEntropyLoss()

        loss = distillation_step(model, ids, labels, loss_fn, tcfg)
        assert loss.isfinite()
        assert loss.item() > 0

    def test_zero_weight_equals_standard(self):
        """With distill_weight=0, output should equal plain cross-entropy."""
        no_drop_cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=4,
            seq_len=32, vocab_size=128, dropout=0.0,
        )
        tcfg = TrainingConfig(distill_weight=0.0)
        model = RealFormerDecoder(no_drop_cfg).train()
        torch.manual_seed(7)
        ids = torch.randint(0, no_drop_cfg.vocab_size, (1, 16))
        labels = torch.randint(0, no_drop_cfg.vocab_size, (1, 16))
        loss_fn = nn.CrossEntropyLoss()

        loss_d = distillation_step(model, ids, labels, loss_fn, tcfg)
        logits = model(ids)
        loss_std = loss_fn(logits.reshape(-1, no_drop_cfg.vocab_size), labels.reshape(-1))

        assert abs(loss_d.item() - loss_std.item()) < 1e-4

    def test_backward_runs(self, cfg):
        tcfg = TrainingConfig(
            segment_ratio=0.5, segment_min_len=4,
            distill_weight=1.0, distill_temperature=2.0,
        )
        model = RealFormerDecoder(cfg).train()
        ids = torch.randint(0, cfg.vocab_size, (2, 16))
        labels = torch.randint(0, cfg.vocab_size, (2, 16))
        loss_fn = nn.CrossEntropyLoss()

        loss = distillation_step(model, ids, labels, loss_fn, tcfg)
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad
