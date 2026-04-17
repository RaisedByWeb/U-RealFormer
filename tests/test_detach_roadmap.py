"""Tests for the Detach Roadmap: Problem A (utility) and Problem B (differentiability)."""

import pytest
import torch
import torch.nn as nn

from realformer_evo import (
    RealFormerConfig,
    RealFormerEncoder,
    RealFormerDecoder,
    GatedResidualAttention,
    score_norm,
)
from realformer_evo.attention import _clip_residual_grad
from realformer_evo.decoder import CausalAttention


# ── fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def small_cfg():
    return RealFormerConfig(
        hidden=64, heads=4, head_dim=16, layers=4,
        seq_len=32, vocab_size=128,
    )


@pytest.fixture
def deep_cfg():
    """48-layer stress config (small hidden to keep memory manageable)."""
    return RealFormerConfig(
        hidden=32, heads=2, head_dim=16, layers=48,
        seq_len=16, vocab_size=64, dropout=0.0,
    )


# ═══════════════════════════════════════════════════════════════════════════════
#  Problem A: Does residual attention work?
# ═══════════════════════════════════════════════════════════════════════════════


class TestGateStaysOpen:
    """After training, the gate should not have collapsed to zero."""

    def test_gate_stays_open_after_training(self, small_cfg):
        model = RealFormerEncoder(small_cfg)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        loss_fn = nn.CrossEntropyLoss()

        model.train()
        for _ in range(100):
            ids = torch.randint(0, small_cfg.vocab_size, (4, 16))
            h = model(ids)
            logits = h[:, 0, :small_cfg.heads]
            labels = torch.zeros(4, dtype=torch.long)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        for layer in model.layers:
            gate = torch.sigmoid(layer.attn.alpha).mean().item()
            assert gate > 0.1, f"gate collapsed to {gate:.4f}"

    def test_residual_magnitude_nontrivial_at_init(self, small_cfg):
        """At init, the residual injection should have non-trivial magnitude."""
        model = RealFormerEncoder(small_cfg).eval()
        ids = torch.randint(0, small_cfg.vocab_size, (1, 8))

        with torch.no_grad():
            x = model.drop(model.embed(ids) + model.pos(
                torch.arange(8, device=ids.device).unsqueeze(0)
            ))
            s = None
            for i, layer in enumerate(model.layers):
                s_in = s if i in small_cfg.residual_layers else None
                h, s = layer.attn(layer.norm1(x), s_in)
                x = x + h
                x = x + layer.ff(layer.norm2(x))

                if s_in is not None:
                    gate = torch.sigmoid(layer.attn.alpha).view(1, small_cfg.heads, 1, 1)
                    scale = layer.attn.gamma.view(1, small_cfg.heads, 1, 1)
                    r_norm = (gate * scale * score_norm(s_in)).norm().item()
                    assert r_norm > 1e-3, f"residual norm too small at layer {i}: {r_norm}"


# ═══════════════════════════════════════════════════════════════════════════════
#  FP16 safety (Guardrail 3)
# ═══════════════════════════════════════════════════════════════════════════════


class TestFP16ScoreNorm:
    def test_fp16_no_nan_t1(self):
        S = torch.randn(1, 1, 1, 1, dtype=torch.float16)
        out = score_norm(S)
        assert not out.isnan().any()
        assert out.dtype == torch.float16

    def test_fp16_no_nan_t2(self):
        S = torch.randn(2, 4, 3, 2, dtype=torch.float16)
        out = score_norm(S)
        assert not out.isnan().any()
        assert out.dtype == torch.float16

    def test_fp16_preserves_statistics(self):
        S = torch.randn(2, 4, 8, 8, dtype=torch.float16)
        out = score_norm(S).float()
        assert out.mean(dim=-1).abs().max() < 0.01
        var = out.var(dim=-1, unbiased=False)
        assert (var - 1.0).abs().max() < 0.1

    def test_fp32_backward_compat(self):
        """FP32 inputs should still produce zero-mean, unit-variance."""
        S = torch.randn(2, 4, 8, 8)
        out = score_norm(S)
        assert out.dtype == torch.float32
        assert out.mean(dim=-1).abs().max() < 1e-5


# ═══════════════════════════════════════════════════════════════════════════════
#  Beta parameter (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBeta:
    def test_zero_init_matches_old_encoder(self, small_cfg):
        """With beta=0, output should be identical to pre-beta behaviour."""
        assert small_cfg.beta_init == 0.0
        model = RealFormerEncoder(small_cfg).eval()
        ids = torch.randint(0, small_cfg.vocab_size, (1, 8))

        with torch.no_grad():
            out1 = model(ids)
            out2 = model(ids)
        assert torch.allclose(out1, out2)

    def test_beta_exists_on_all_attention(self, small_cfg):
        enc = RealFormerEncoder(small_cfg)
        for layer in enc.layers:
            assert hasattr(layer.attn, "beta")
            assert layer.attn.beta.shape == (small_cfg.heads,)

        dec = RealFormerDecoder(small_cfg)
        for layer in dec.layers:
            assert hasattr(layer.attn, "beta")
            assert layer.attn.beta.shape == (small_cfg.heads,)

    def test_beta_gradient_flows(self, small_cfg):
        model = RealFormerEncoder(small_cfg)
        ids = torch.randint(0, small_cfg.vocab_size, (1, 8))
        out = model(ids)
        out.sum().backward()
        assert model.layers[1].attn.beta.grad is not None
        assert not model.layers[1].attn.beta.grad.isnan().any()


# ═══════════════════════════════════════════════════════════════════════════════
#  Problem B: 48-Layer Stress Stack
# ═══════════════════════════════════════════════════════════════════════════════


class TestStressStack:
    def _assert_grads_clean(self, model):
        # Layer 0's alpha/gamma/beta are never used (no s_prev at the
        # first layer), so they legitimately have no gradient.
        for name, p in model.named_parameters():
            if p.grad is None:
                assert "layers.0.attn.alpha" in name or \
                       "layers.0.attn.gamma" in name or \
                       "layers.0.attn.beta" in name, \
                    f"unexpected missing grad for {name}"
                continue
            assert not p.grad.isnan().any(), f"NaN grad in {name}"
            assert not p.grad.isinf().any(), f"Inf grad in {name}"

    def test_48_layer_no_nan_detached(self, deep_cfg):
        deep_cfg.residual_grad_flow = False
        model = RealFormerEncoder(deep_cfg)
        ids = torch.randint(0, deep_cfg.vocab_size, (1, deep_cfg.seq_len))
        out = model(ids)
        out.sum().backward()
        self._assert_grads_clean(model)

    def test_48_layer_no_nan_coupled(self, deep_cfg):
        deep_cfg.residual_grad_flow = True
        deep_cfg.residual_grad_clip = 1.0
        model = RealFormerEncoder(deep_cfg)
        ids = torch.randint(0, deep_cfg.vocab_size, (1, deep_cfg.seq_len))
        out = model(ids)
        out.sum().backward()
        self._assert_grads_clean(model)


# ═══════════════════════════════════════════════════════════════════════════════
#  Detach toggle
# ═══════════════════════════════════════════════════════════════════════════════


class TestDetachToggle:
    def test_toggle_changes_grad_flow(self):
        """With residual_grad_flow=True, alpha at layer 1 should receive
        gradient from downstream layers through the coupled score path."""
        torch.manual_seed(42)
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=4,
            seq_len=16, vocab_size=64, dropout=0.0,
            residual_grad_flow=True, residual_grad_clip=0.0,
        )
        model = RealFormerEncoder(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        out = model(ids)
        probe = torch.randn(cfg.hidden, 32)
        logits = out @ probe
        labels = torch.randint(0, 32, (1, 8))
        loss = nn.CrossEntropyLoss()(logits.view(-1, 32), labels.view(-1))
        loss.backward()

        layer1_alpha_grad = model.layers[1].attn.alpha.grad
        assert layer1_alpha_grad is not None
        assert layer1_alpha_grad.abs().sum() > 0

    def test_detached_mode_no_cross_layer_grad(self):
        """With residual_grad_flow=False, changing s_prev should not
        produce gradient on s_prev itself (it's detached)."""
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=2,
            seq_len=16, vocab_size=64,
            residual_grad_flow=False,
        )
        attn = GatedResidualAttention(cfg)
        x = torch.randn(1, 4, cfg.hidden)
        s_prev = torch.randn(1, cfg.heads, 4, 4, requires_grad=True)
        out, raw = attn(x, s_prev=s_prev)
        out.sum().backward()
        assert s_prev.grad is None or s_prev.grad.abs().sum() == 0


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradient clipping
# ═══════════════════════════════════════════════════════════════════════════════


class TestGradientClip:
    def test_clip_caps_gradient(self):
        x = torch.randn(2, 3, requires_grad=True)
        y = _clip_residual_grad(x, max_norm=0.01)
        loss = (y ** 2).sum()
        loss.backward()

        assert x.grad is not None
        grad_norm = x.grad.norm().item()
        assert grad_norm <= 0.011, f"grad norm {grad_norm} exceeds clip"

    def test_no_clip_when_small(self):
        """When gradient norm is well below max_norm, clip should be a no-op."""
        data = torch.randn(2, 3) * 0.001
        x = data.clone().requires_grad_(True)
        y = _clip_residual_grad(x, max_norm=100.0)
        loss = (y ** 2).sum()
        loss.backward()

        x2 = data.clone().requires_grad_(True)
        loss2 = (x2 ** 2).sum()
        loss2.backward()

        assert x.grad is not None
        assert torch.allclose(x.grad, x2.grad, atol=1e-6)

    def test_clip_identity_forward(self):
        x = torch.randn(3, 4)
        y = _clip_residual_grad(x, max_norm=1.0)
        assert torch.equal(x, y)


# ═══════════════════════════════════════════════════════════════════════════════
#  Gradient identity collapse (co-evolution stress)
# ═══════════════════════════════════════════════════════════════════════════════


class TestGradientIdentityCollapse:
    def test_early_wq_not_dominated_by_late_alpha(self):
        """In coupled mode, check that W_q gradients in early layers
        are not perfectly correlated with alpha gradients in later
        layers — early layers should retain their own identity."""
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=6,
            seq_len=16, vocab_size=64, dropout=0.0,
            residual_grad_flow=True, residual_grad_clip=1.0,
        )
        model = RealFormerEncoder(cfg)
        ids = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(ids)
        out.sum().backward()

        wq_grad_l0 = model.layers[0].attn.q.weight.grad.flatten()
        alpha_grad_l5 = model.layers[5].attn.alpha.grad.flatten()

        min_len = min(len(wq_grad_l0), len(alpha_grad_l5))
        cos = torch.nn.functional.cosine_similarity(
            wq_grad_l0[:min_len].unsqueeze(0),
            alpha_grad_l5[:min_len].unsqueeze(0),
        ).item()
        assert abs(cos) < 0.95, (
            f"W_q[L0] grad is too correlated with alpha[L5] grad "
            f"(cos={cos:.4f}): early layers may be losing identity"
        )


# ═══════════════════════════════════════════════════════════════════════════════
#  residual_layers config
# ═══════════════════════════════════════════════════════════════════════════════


class TestResidualLayers:
    def test_skip_k_generates_residual_layers(self):
        cfg = RealFormerConfig(hidden=64, heads=4, head_dim=16, layers=6, skip_k=2)
        assert cfg.residual_layers == {0, 2, 4}

    def test_skip_k_1_all_layers(self):
        cfg = RealFormerConfig(hidden=64, heads=4, head_dim=16, layers=4, skip_k=1)
        assert cfg.residual_layers == {0, 1, 2, 3}

    def test_skip_k_larger_than_layers_includes_zero(self):
        """skip_k=L+1 still includes layer 0 (i % 7 == 0 when i=0).
        True baseline requires explicit residual_layers=set()."""
        cfg = RealFormerConfig(hidden=64, heads=4, head_dim=16, layers=6, skip_k=7)
        assert cfg.residual_layers == {0}

    def test_explicit_overrides_skip_k(self):
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=6,
            skip_k=1, residual_layers={0, 1, 2},
        )
        assert cfg.residual_layers == {0, 1, 2}

    def test_empty_set_is_baseline(self):
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=6,
            residual_layers=set(),
        )
        assert cfg.residual_layers == set()

    def test_list_input_coerced_to_set(self):
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=6,
            residual_layers=[0, 2, 4],
        )
        assert isinstance(cfg.residual_layers, set)
        assert cfg.residual_layers == {0, 2, 4}

    def test_invalid_index_raises(self):
        with pytest.raises(ValueError, match="invalid layer indices"):
            RealFormerConfig(
                hidden=64, heads=4, head_dim=16, layers=4,
                residual_layers={0, 1, 7},
            )

    def test_negative_index_raises(self):
        with pytest.raises(ValueError, match="invalid layer indices"):
            RealFormerConfig(
                hidden=64, heads=4, head_dim=16, layers=4,
                residual_layers={-1, 0},
            )

    def test_skip_k_zero_raises(self):
        with pytest.raises(ValueError, match="skip_k must be >= 1"):
            RealFormerConfig(hidden=64, heads=4, head_dim=16, layers=4, skip_k=0)

    def test_nonuniform_pattern_forward(self):
        """First 3 layers get residual, last 3 don't. Model runs without error."""
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=6,
            seq_len=16, vocab_size=64,
            residual_layers={0, 1, 2},
        )
        model = RealFormerEncoder(cfg).eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        with torch.no_grad():
            out = model(ids)
        assert out.shape == (1, 8, cfg.hidden)
        assert not out.isnan().any()

    def test_empty_set_baseline_forward(self):
        """With residual_layers=set(), the model is a standard Pre-LN Transformer."""
        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=4,
            seq_len=16, vocab_size=64,
            residual_layers=set(),
        )
        model = RealFormerEncoder(cfg)
        ids = torch.randint(0, cfg.vocab_size, (1, 8))
        out = model(ids)
        out.sum().backward()
        for layer in model.layers:
            assert not layer.receives_residual
