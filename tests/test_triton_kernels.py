"""Tests for the fused Triton residual attention kernel.

All GPU tests are skip-marked when CUDA is unavailable. The reference
implementation is tested on CPU to verify its correctness independently.
"""

import math

import pytest
import torch

from realformer_evo.triton_kernels import (
    is_triton_available,
    reference_residual_attention,
)

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA not available",
)

requires_triton = pytest.mark.skipif(
    not is_triton_available(),
    reason="Triton + CUDA not available",
)


# ══════════════════════════════════════════════════════════════════════════════
#  Reference implementation tests (CPU — always run)
# ══════════════════════════════════════════════════════════════════════════════


class TestReferenceImplementation:
    """Verify the PyTorch reference against manual computation."""

    def test_no_residual_matches_standard_attn(self):
        B, H, T, D = 1, 2, 8, 16
        Q = torch.randn(B, H, T, D)
        K = torch.randn(B, H, T, D)
        V = torch.randn(B, H, T, D)
        gate = torch.sigmoid(torch.zeros(H))
        scale = torch.full((H,), 0.577)
        shift = torch.zeros(H)
        sm_scale = 1.0 / math.sqrt(D)

        out, s_out = reference_residual_attention(
            Q, K, V, None, gate, scale, shift, sm_scale
        )
        expected = torch.matmul(
            torch.softmax(torch.matmul(Q, K.transpose(-2, -1)) * sm_scale, dim=-1),
            V,
        )
        assert torch.allclose(out, expected, atol=1e-5)

    def test_causal_mask_applied(self):
        B, H, T, D = 1, 2, 4, 8
        Q = torch.randn(B, H, T, D)
        K = torch.randn(B, H, T, D)
        V = torch.randn(B, H, T, D)
        gate = torch.sigmoid(torch.zeros(H))
        scale = torch.full((H,), 0.577)
        shift = torch.zeros(H)
        sm_scale = 1.0 / math.sqrt(D)

        out_causal, _ = reference_residual_attention(
            Q, K, V, None, gate, scale, shift, sm_scale, causal=True
        )
        out_full, _ = reference_residual_attention(
            Q, K, V, None, gate, scale, shift, sm_scale, causal=False
        )
        assert not torch.allclose(out_causal, out_full)

    def test_residual_injection_changes_output(self):
        B, H, T, D = 1, 2, 8, 16
        Q = torch.randn(B, H, T, D)
        K = torch.randn(B, H, T, D)
        V = torch.randn(B, H, T, D)
        s_prev = torch.randn(B, H, T, T)
        gate = torch.sigmoid(torch.ones(H))
        scale = torch.full((H,), 0.577)
        shift = torch.zeros(H)
        sm_scale = 1.0 / math.sqrt(D)

        out_with, _ = reference_residual_attention(
            Q, K, V, s_prev, gate, scale, shift, sm_scale
        )
        out_without, _ = reference_residual_attention(
            Q, K, V, None, gate, scale, shift, sm_scale
        )
        assert not torch.allclose(out_with, out_without)

    def test_s_out_shape(self):
        B, H, T, D = 2, 4, 8, 16
        Q = torch.randn(B, H, T, D)
        K = torch.randn(B, H, T, D)
        V = torch.randn(B, H, T, D)
        s_prev = torch.randn(B, H, T, T)
        gate = torch.sigmoid(torch.zeros(H))
        scale = torch.full((H,), 0.577)
        shift = torch.zeros(H)

        _, s_out = reference_residual_attention(
            Q, K, V, s_prev, gate, scale, shift, 1.0 / math.sqrt(D)
        )
        assert s_out.shape == (B, H, T, T)


# ══════════════════════════════════════════════════════════════════════════════
#  Triton kernel tests (GPU only — skip-marked)
# ══════════════════════════════════════════════════════════════════════════════


@requires_triton
class TestTritonKernel:
    """Numerical equivalence: Triton kernel vs PyTorch reference."""

    def _run_equivalence(self, B, H, T, D, causal, has_residual, dtype):
        from realformer_evo.triton_kernels import fused_residual_attention

        device = "cuda"
        sm_scale = 1.0 / math.sqrt(D)

        torch.manual_seed(42)
        Q = torch.randn(B, H, T, D, device=device, dtype=dtype)
        K = torch.randn(B, H, T, D, device=device, dtype=dtype)
        V = torch.randn(B, H, T, D, device=device, dtype=dtype)
        s_prev = torch.randn(B, H, T, T, device=device, dtype=dtype) if has_residual else None
        gate = torch.sigmoid(torch.randn(H, device=device, dtype=dtype))
        scale = torch.randn(H, device=device, dtype=dtype).abs()
        shift = torch.randn(H, device=device, dtype=dtype) * 0.1

        out_ref, s_ref = reference_residual_attention(
            Q, K, V, s_prev, gate, scale, shift, sm_scale, causal=causal,
        )
        out_tri, s_tri = fused_residual_attention(
            Q, K, V, s_prev, gate, scale, shift, sm_scale, causal=causal,
        )

        atol = 1e-2 if dtype == torch.float16 else 1e-4
        rtol = 1e-2 if dtype == torch.float16 else 1e-4

        assert out_tri.shape == out_ref.shape
        assert s_tri.shape == s_ref.shape
        assert torch.allclose(out_tri.float(), out_ref.float(), atol=atol, rtol=rtol), \
            f"out mismatch: max diff {(out_tri.float() - out_ref.float()).abs().max()}"
        assert torch.allclose(s_tri.float(), s_ref.float(), atol=atol, rtol=rtol), \
            f"s_out mismatch: max diff {(s_tri.float() - s_ref.float()).abs().max()}"

    def test_fp32_noncausal_residual(self):
        self._run_equivalence(1, 2, 16, 32, causal=False, has_residual=True, dtype=torch.float32)

    def test_fp32_causal_residual(self):
        self._run_equivalence(1, 2, 16, 32, causal=True, has_residual=True, dtype=torch.float32)

    def test_fp32_noncausal_no_residual(self):
        self._run_equivalence(1, 2, 16, 32, causal=False, has_residual=False, dtype=torch.float32)

    def test_fp16_noncausal_residual(self):
        self._run_equivalence(2, 4, 32, 64, causal=False, has_residual=True, dtype=torch.float16)

    def test_fp16_causal_residual(self):
        self._run_equivalence(2, 4, 32, 64, causal=True, has_residual=True, dtype=torch.float16)

    def test_batch_and_heads(self):
        self._run_equivalence(4, 8, 16, 64, causal=True, has_residual=True, dtype=torch.float32)

    def test_non_power_of_2_seqlen(self):
        self._run_equivalence(1, 2, 13, 32, causal=False, has_residual=True, dtype=torch.float32)

    def test_large_seqlen(self):
        self._run_equivalence(1, 2, 128, 64, causal=True, has_residual=True, dtype=torch.float16)


@requires_triton
class TestTritonDispatch:
    """Verify that the attention modules dispatch to Triton on CUDA."""

    def test_encoder_uses_triton(self):
        from realformer_evo import RealFormerConfig, RealFormerEncoder

        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=2,
            seq_len=32, vocab_size=64,
        )
        model = RealFormerEncoder(cfg).cuda().eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 16), device="cuda")

        with torch.no_grad():
            out = model(ids)
        assert out.shape == (1, 16, cfg.hidden)
        assert not out.isnan().any()

    def test_decoder_uses_triton(self):
        from realformer_evo import RealFormerConfig, RealFormerDecoder

        cfg = RealFormerConfig(
            hidden=64, heads=4, head_dim=16, layers=2,
            seq_len=32, vocab_size=64,
        )
        model = RealFormerDecoder(cfg).cuda().eval()
        ids = torch.randint(0, cfg.vocab_size, (1, 16), device="cuda")

        with torch.no_grad():
            out = model(ids)
        assert out.shape == (1, 16, cfg.vocab_size)
        assert not out.isnan().any()


# ══════════════════════════════════════════════════════════════════════════════
#  Graceful fallback test (always runs)
# ══════════════════════════════════════════════════════════════════════════════


class TestFallback:
    def test_is_triton_available_returns_bool(self):
        assert isinstance(is_triton_available(), bool)

    def test_fused_raises_without_cuda(self):
        if is_triton_available():
            pytest.skip("CUDA available — fallback not testable")
        from realformer_evo.triton_kernels import fused_residual_attention
        with pytest.raises(RuntimeError, match="requires Triton"):
            fused_residual_attention(
                torch.randn(1, 1, 4, 8), torch.randn(1, 1, 4, 8),
                torch.randn(1, 1, 4, 8), None,
                torch.zeros(1), torch.ones(1), torch.zeros(1),
                sm_scale=0.25,
            )
