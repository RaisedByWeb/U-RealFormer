"""Memory and parameter-count sanity checks."""

import pytest
import torch

from realformer_evo import (
    RealFormerConfig,
    RealFormerEncoder,
    RealFormerDecoder,
    LowRankProjector,
)


@pytest.fixture
def small_cfg():
    return RealFormerConfig(
        hidden=64, heads=4, head_dim=16, layers=4,
        seq_len=32, vocab_size=128,
    )


class TestParameterCounts:
    def test_encoder_param_count(self, small_cfg):
        model = RealFormerEncoder(small_cfg)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0
        assert n < 10_000_000, "small config should be well under 10M params"

    def test_decoder_param_count(self, small_cfg):
        model = RealFormerDecoder(small_cfg)
        n = sum(p.numel() for p in model.parameters())
        assert n > 0

    def test_alpha_gamma_per_head(self, small_cfg):
        """Each layer should have H alpha + H gamma parameters."""
        enc = RealFormerEncoder(small_cfg)
        for layer in enc.layers:
            assert layer.attn.alpha.shape == (small_cfg.heads,)
            assert layer.attn.gamma.shape == (small_cfg.heads,)


class TestLowRankProjector:
    def test_full_rank_passthrough(self, small_cfg):
        small_cfg.rank = 0
        proj = LowRankProjector(small_cfg)
        S = torch.randn(1, 4, 8, 8)
        assert torch.equal(proj.compress(S), S)
        assert torch.equal(proj.decompress(S, 8), S)

    def test_compressed_shape(self):
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=2,
            seq_len=16, rank=4,
        )
        proj = LowRankProjector(cfg)
        S = torch.randn(1, 4, 16, 16)
        compressed = proj.compress(S)
        assert compressed.shape == (1, 4, 16, 4)


class TestForwardNoOOM:
    """Ensure forward passes complete without OOM on small configs."""

    def test_encoder_forward(self, small_cfg):
        model = RealFormerEncoder(small_cfg).eval()
        ids = torch.randint(0, small_cfg.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids)
        assert out.shape == (2, 16, small_cfg.hidden)

    def test_decoder_forward(self, small_cfg):
        model = RealFormerDecoder(small_cfg).eval()
        ids = torch.randint(0, small_cfg.vocab_size, (2, 16))
        with torch.no_grad():
            out = model(ids)
        assert out.shape == (2, 16, small_cfg.vocab_size)
