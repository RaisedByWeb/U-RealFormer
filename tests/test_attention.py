"""Tests for gated residual attention and ScoreNorm."""

import math

import pytest
import torch

from u_realformer import RealFormerConfig, GatedResidualAttention, score_norm


@pytest.fixture
def cfg():
    return RealFormerConfig(hidden=64, heads=4, head_dim=16, layers=2, seq_len=32)


# ── ScoreNorm ─────────────────────────────────────────────────────────────────

class TestScoreNorm:
    def test_zero_mean(self):
        S = torch.randn(2, 4, 8, 8)
        out = score_norm(S)
        assert out.mean(dim=-1).abs().max() < 1e-5

    def test_unit_variance(self):
        S = torch.randn(2, 4, 8, 8)
        out = score_norm(S)
        var = out.var(dim=-1, unbiased=False)
        assert (var - 1.0).abs().max() < 1e-4

    def test_single_element(self):
        """n=1 key dim should not produce NaN."""
        S = torch.tensor([[[[5.0]]]])
        out = score_norm(S)
        assert not out.isnan().any()

    def test_constant_row(self):
        """Constant rows → zero output (zero variance)."""
        S = torch.full((1, 1, 1, 4), 3.0)
        out = score_norm(S)
        assert out.abs().max() < 1e-5


# ── GatedResidualAttention ────────────────────────────────────────────────────

class TestGatedResidualAttention:
    def test_output_shape(self, cfg):
        attn = GatedResidualAttention(cfg)
        x = torch.randn(2, 8, cfg.hidden)
        out, raw = attn(x)
        assert out.shape == x.shape
        assert raw.shape == (2, cfg.heads, 8, 8)

    def test_with_score_residual(self, cfg):
        attn = GatedResidualAttention(cfg)
        x = torch.randn(2, 8, cfg.hidden)
        s_prev = torch.randn(2, cfg.heads, 8, 8)
        out, raw = attn(x, s_prev=s_prev)
        assert out.shape == x.shape

    def test_no_nan_in_output(self, cfg):
        attn = GatedResidualAttention(cfg).eval()
        x = torch.randn(2, 8, cfg.hidden)
        with torch.no_grad():
            out, raw = attn(x)
        assert not out.isnan().any()
        assert not raw.isnan().any()

    def test_gate_gradient_flows(self, cfg):
        attn = GatedResidualAttention(cfg)
        x = torch.randn(2, 8, cfg.hidden)
        s_prev = torch.randn(2, cfg.heads, 8, 8)
        out, _ = attn(x, s_prev=s_prev)
        out.sum().backward()
        assert attn.alpha.grad is not None
        assert attn.gamma.grad is not None
        assert not attn.alpha.grad.isnan().any()

    def test_additive_mask(self, cfg):
        attn = GatedResidualAttention(cfg).eval()
        x = torch.randn(1, 4, cfg.hidden)
        mask = torch.zeros(1, 1, 4, 4)
        mask[:, :, :, 3] = float("-inf")
        with torch.no_grad():
            out_masked, _ = attn(x, mask=mask)
            out_full, _ = attn(x)
        assert not torch.allclose(out_masked, out_full)
