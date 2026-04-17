"""
Fused residual attention Triton kernel.

Fuses QK^T + ScoreNorm-based residual injection + causal masking +
online softmax + V accumulation into a single kernel launch, eliminating
one T-squared intermediate tensor and reducing kernel launch overhead
from ~5 launches to 1 per attention layer.

Scope constraints:
  - Phase 1 (detached s_prev) only — no gradient through s_prev.
  - Full-sequence self-attention only (no KV cache; causal mask
    assumes M == N with zero-based aligned positions).
  - No stochastic dropout — eval / dropout=0 only.
  - Backward falls back to PyTorch ops (recompute from saved tensors).

Requires: ``triton`` package and a CUDA device.
Falls back gracefully when unavailable — see ``fused_residual_attention()``.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _TRITON_AVAILABLE = True
except ImportError:
    _TRITON_AVAILABLE = False


def is_triton_available() -> bool:
    return _TRITON_AVAILABLE and torch.cuda.is_available()


# ══════════════════════════════════════════════════════════════════════════════
#  ScoreNorm stats precompute kernel
# ══════════════════════════════════════════════════════════════════════════════

if _TRITON_AVAILABLE:

    @triton.jit
    def _score_norm_stats_kernel(
        S_ptr, Mu_ptr, Rstd_ptr,
        stride_row,    # stride between consecutive rows (= stride(2) for contiguous (B,H,Q,K))
        stride_col,    # stride between consecutive columns (= stride(3), typically 1)
        T: tl.constexpr,
        eps: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        """Compute row-wise mean and reciprocal-stddev of s_prev.

        Grid: ``(B * H * M,)`` — one program per row.  The (B, H, Q)
        dimensions are flattened into a single grid axis; ``stride_row``
        and ``stride_col`` are the only strides the kernel needs.
        """
        row_id = tl.program_id(0)
        row_ptr = S_ptr + row_id * stride_row

        col_offsets = tl.arange(0, BLOCK_K)
        acc_sum = tl.zeros([BLOCK_K], dtype=tl.float32)
        acc_sq = tl.zeros([BLOCK_K], dtype=tl.float32)
        count = 0

        for start in range(0, T, BLOCK_K):
            cols = start + col_offsets
            mask = cols < T
            vals = tl.load(row_ptr + cols * stride_col, mask=mask, other=0.0).to(tl.float32)
            acc_sum += tl.where(mask, vals, 0.0)
            acc_sq += tl.where(mask, vals * vals, 0.0)
            count += tl.sum(mask.to(tl.int32))

        total_sum = tl.sum(acc_sum)
        total_sq = tl.sum(acc_sq)
        mean = total_sum / count
        var = total_sq / count - mean * mean
        rstd = 1.0 / tl.sqrt(var + eps)

        tl.store(Mu_ptr + row_id, mean)
        tl.store(Rstd_ptr + row_id, rstd)


    # ══════════════════════════════════════════════════════════════════════════
    #  Fused forward kernel
    # ══════════════════════════════════════════════════════════════════════════

    @triton.jit
    def _fused_residual_attn_fwd(
        Q_ptr, K_ptr, V_ptr,
        S_prev_ptr, S_out_ptr, Out_ptr,
        Mu_ptr, Rstd_ptr,
        Gate_ptr, Scale_ptr, Shift_ptr,
        sm_scale,
        stride_qb, stride_qh, stride_qm, stride_qd,
        stride_kb, stride_kh, stride_kn, stride_kd,
        stride_vb, stride_vh, stride_vn, stride_vd,
        stride_sb, stride_sh, stride_sm, stride_sn,
        stride_ob, stride_oh, stride_om, stride_od,
        B: tl.constexpr, H: tl.constexpr,
        M: tl.constexpr, N: tl.constexpr, D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        HAS_RESIDUAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Fused residual attention forward.

        Grid: ``(cdiv(M, BLOCK_M), B * H)``

        Invariants (enforced by dispatch guards, not checked here):
          - M == N (full-sequence self-attention)
          - Causal mask assumes zero-based aligned positions
            (not valid for cached decoding with offset queries)
          - No dropout (deterministic forward)
        """
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        batch_id = pid_bh // H
        head_id = pid_bh % H

        q_offset = batch_id * stride_qb + head_id * stride_qh
        k_offset = batch_id * stride_kb + head_id * stride_kh
        v_offset = batch_id * stride_vb + head_id * stride_vh
        s_offset = batch_id * stride_sb + head_id * stride_sh
        o_offset = batch_id * stride_ob + head_id * stride_oh

        gate_h = tl.load(Gate_ptr + head_id).to(tl.float32)
        scale_h = tl.load(Scale_ptr + head_id).to(tl.float32)
        shift_h = tl.load(Shift_ptr + head_id).to(tl.float32)

        m_offsets = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        d_offsets = tl.arange(0, BLOCK_D)

        # Load Q tile: (BLOCK_M, D)
        q_ptrs = Q_ptr + q_offset + m_offsets[:, None] * stride_qm + d_offsets[None, :] * stride_qd
        q_mask = (m_offsets[:, None] < M) & (d_offsets[None, :] < D)
        q_tile = tl.load(q_ptrs, mask=q_mask, other=0.0)

        # Online softmax accumulators
        m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_D], dtype=tl.float32)

        # Mu/Rstd for ScoreNorm (per query row in this batch-head)
        mu_offset = (batch_id * H + head_id) * M
        rstd_offset = mu_offset

        for start_n in range(0, N, BLOCK_N):
            n_offsets = start_n + tl.arange(0, BLOCK_N)

            # Load K tile: (BLOCK_N, D)
            k_ptrs = (K_ptr + k_offset
                      + n_offsets[:, None] * stride_kn
                      + d_offsets[None, :] * stride_kd)
            k_mask = (n_offsets[:, None] < N) & (d_offsets[None, :] < D)
            k_tile = tl.load(k_ptrs, mask=k_mask, other=0.0)

            # QK^T: (BLOCK_M, BLOCK_N)
            qk = tl.dot(q_tile, tl.trans(k_tile)) * sm_scale

            # Residual injection
            if HAS_RESIDUAL:
                s_ptrs = (S_prev_ptr + s_offset
                          + m_offsets[:, None] * stride_sm
                          + n_offsets[None, :] * stride_sn)
                s_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
                s_tile = tl.load(s_ptrs, mask=s_mask, other=0.0).to(tl.float32)

                mu_vals = tl.load(Mu_ptr + mu_offset + m_offsets, mask=m_offsets < M, other=0.0)
                rstd_vals = tl.load(
                    Rstd_ptr + rstd_offset + m_offsets,
                    mask=m_offsets < M, other=1.0,
                )

                norm_tile = (s_tile - mu_vals[:, None]) * rstd_vals[:, None]
                qk += gate_h * (scale_h * norm_tile + shift_h)

            # Store s_out (raw scores after injection, for next layer)
            sout_ptrs = (S_out_ptr + s_offset
                        + m_offsets[:, None] * stride_sm
                        + n_offsets[None, :] * stride_sn)
            sout_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
            tl.store(sout_ptrs, qk.to(S_out_ptr.dtype.element_ty), mask=sout_mask)

            # Causal mask
            if IS_CAUSAL:
                causal_mask = m_offsets[:, None] < n_offsets[None, :]
                qk = tl.where(causal_mask, float("-inf"), qk)

            # Validity mask for padding
            valid_mask = (m_offsets[:, None] < M) & (n_offsets[None, :] < N)
            qk = tl.where(valid_mask, qk, float("-inf"))

            # Online softmax update (Milakov-Gimelshein)
            m_ij = tl.max(qk, axis=1)
            m_new = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_new)
            p = tl.exp(qk - m_new[:, None])
            l_i = l_i * alpha + tl.sum(p, axis=1)
            acc = acc * alpha[:, None]

            # Load V tile and accumulate
            v_ptrs = (V_ptr + v_offset
                      + n_offsets[:, None] * stride_vn
                      + d_offsets[None, :] * stride_vd)
            v_mask = (n_offsets[:, None] < N) & (d_offsets[None, :] < D)
            v_tile = tl.load(v_ptrs, mask=v_mask, other=0.0)
            acc += tl.dot(p.to(v_tile.dtype), v_tile)

            m_i = m_new

        # Final rescale
        acc = acc / l_i[:, None]

        # Write output
        out_ptrs = (Out_ptr + o_offset
                    + m_offsets[:, None] * stride_om
                    + d_offsets[None, :] * stride_od)
        out_mask = (m_offsets[:, None] < M) & (d_offsets[None, :] < D)
        tl.store(out_ptrs, acc.to(Out_ptr.dtype.element_ty), mask=out_mask)


# ══════════════════════════════════════════════════════════════════════════════
#  Python wrapper + autograd
# ══════════════════════════════════════════════════════════════════════════════


def _compute_score_norm_stats(
    s_prev: torch.Tensor, eps: float = 1e-6
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Precompute row-wise mean and reciprocal-stddev for s_prev using Triton."""
    B, H, M, N = s_prev.shape
    mu = torch.empty(B, H, M, device=s_prev.device, dtype=torch.float32)
    rstd = torch.empty(B, H, M, device=s_prev.device, dtype=torch.float32)

    s_flat = s_prev.contiguous()
    total_rows = B * H * M
    BLOCK_K = triton.next_power_of_2(N)
    if BLOCK_K > 4096:
        BLOCK_K = 4096

    _score_norm_stats_kernel[(total_rows,)](
        s_flat, mu, rstd,
        s_flat.stride(2),    # stride_row: distance between consecutive rows in the key dim
        s_flat.stride(3),    # stride_col: distance between consecutive columns (1 for contiguous)
        T=N, eps=eps, BLOCK_K=BLOCK_K,
    )
    return mu, rstd


def _triton_fused_fwd(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    s_prev: Optional[torch.Tensor],
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    sm_scale: float,
    causal: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Launch the fused Triton kernel."""
    B, H, M, D = Q.shape
    _, _, N, _ = K.shape

    Out = torch.empty(B, H, M, D, device=Q.device, dtype=Q.dtype)
    S_out = torch.empty(B, H, M, N, device=Q.device, dtype=Q.dtype)

    has_residual = s_prev is not None
    if has_residual:
        mu, rstd = _compute_score_norm_stats(s_prev)
        s_prev_c = s_prev.contiguous()
    else:
        mu = torch.empty(0, device=Q.device, dtype=torch.float32)
        rstd = torch.empty(0, device=Q.device, dtype=torch.float32)
        s_prev_c = torch.empty(0, device=Q.device, dtype=Q.dtype)

    gate_flat = gate.flatten().contiguous()
    scale_flat = scale.flatten().contiguous()
    shift_flat = shift.flatten().contiguous()

    BLOCK_M = 128 if M >= 128 else triton.next_power_of_2(M)
    BLOCK_N = 128 if N >= 128 else triton.next_power_of_2(N)
    BLOCK_D = triton.next_power_of_2(D)

    grid = (triton.cdiv(M, BLOCK_M), B * H)

    Q_c = Q.contiguous()
    K_c = K.contiguous()
    V_c = V.contiguous()

    _fused_residual_attn_fwd[grid](
        Q_c, K_c, V_c,
        s_prev_c, S_out, Out,
        mu, rstd,
        gate_flat, scale_flat, shift_flat,
        sm_scale,
        Q_c.stride(0), Q_c.stride(1), Q_c.stride(2), Q_c.stride(3),
        K_c.stride(0), K_c.stride(1), K_c.stride(2), K_c.stride(3),
        V_c.stride(0), V_c.stride(1), V_c.stride(2), V_c.stride(3),
        S_out.stride(0), S_out.stride(1), S_out.stride(2), S_out.stride(3),
        Out.stride(0), Out.stride(1), Out.stride(2), Out.stride(3),
        B=B, H=H, M=M, N=N, D=D,
        IS_CAUSAL=causal,
        HAS_RESIDUAL=has_residual,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_D=BLOCK_D,
    )

    return Out, S_out


class _FusedResidualAttentionFn(torch.autograd.Function):
    """Triton forward, PyTorch backward fallback.

    The fused kernel does NOT support dropout.  The dispatch guards in
    ``GatedResidualAttention`` and ``CausalAttention`` ensure this
    function is only called at eval or when ``dropout == 0``.

    Output shapes are always ``(B, H, M, D)`` for ``out`` and
    ``(B, H, M, N)`` for ``s_out``.
    """

    @staticmethod
    def forward(
        ctx,
        Q, K, V, s_prev, gate, scale, shift,
        sm_scale: float, causal: bool,
    ):
        out, s_out = _triton_fused_fwd(Q, K, V, s_prev, gate, scale, shift, sm_scale, causal)

        ctx.save_for_backward(Q, K, V, s_prev, gate, scale, shift, s_out)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        return out, s_out

    @staticmethod
    def backward(ctx, grad_out, grad_s_out):
        # Fused path is only dispatched in Phase 1 (detach=True) and
        # without dropout.  s_prev gradient is always None.
        from .attention import score_norm as _sn
        Q, K, V, s_prev, gate, scale, shift, s_out = ctx.saved_tensors

        with torch.enable_grad():
            Q = Q.detach().requires_grad_(True)
            K = K.detach().requires_grad_(True)
            V = V.detach().requires_grad_(True)
            gate = gate.detach().requires_grad_(True)
            scale = scale.detach().requires_grad_(True)
            shift = shift.detach().requires_grad_(True)

            raw = torch.matmul(Q, K.transpose(-2, -1)) * ctx.sm_scale
            if s_prev is not None:
                g = gate.view(1, -1, 1, 1)
                sc = scale.view(1, -1, 1, 1)
                sh = shift.view(1, -1, 1, 1)
                raw = raw + g * (sc * _sn(s_prev.detach()) + sh)

            if ctx.causal:
                mask = torch.triu(
                    torch.full((raw.size(2), raw.size(3)), float("-inf"), device=raw.device),
                    diagonal=1,
                )[None, None]
                scores = raw + mask
            else:
                scores = raw

            w = F.softmax(scores, dim=-1)
            out_recomp = torch.matmul(w, V)

        out_recomp.backward(grad_out)

        return (
            Q.grad, K.grad, V.grad,
            None,  # s_prev (always detached — fused path is Phase 1 only)
            gate.grad, scale.grad, shift.grad,
            None, None,
        )


def fused_residual_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    s_prev: Optional[torch.Tensor],
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    sm_scale: float,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused residual attention with Triton.

    Returns ``(out, s_out)`` where ``out`` is ``(B, H, M, D)`` and
    ``s_out`` is ``(B, H, M, N)`` — the raw scores for the next layer.

    Scope constraints (enforced by dispatch guards in the attention modules):

    - **Phase 1 only** — ``s_prev`` is always detached before being passed
      in.  The backward does not propagate gradients through ``s_prev``.
    - **No dropout** — the fused kernel does not support stochastic dropout.
      Callers must only dispatch here at eval or when ``dropout == 0``.
    - **Full-sequence self-attention only** — the causal mask assumes M == N
      with zero-based aligned positions.  Not valid for cached decoding
      where query positions are offset.
    - **Backward falls back to PyTorch** — the forward is fused, the
      backward recomputes attention from saved tensors using PyTorch ops.

    Raises ``RuntimeError`` when Triton is unavailable or no CUDA device.
    """
    if not is_triton_available():
        raise RuntimeError("fused_residual_attention requires Triton + CUDA")

    return _FusedResidualAttentionFn.apply(
        Q, K, V, s_prev, gate, scale, shift,
        sm_scale, causal,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  PyTorch reference implementation (for testing)
# ══════════════════════════════════════════════════════════════════════════════


def reference_residual_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    s_prev: Optional[torch.Tensor],
    gate: torch.Tensor,
    scale: torch.Tensor,
    shift: torch.Tensor,
    sm_scale: float,
    causal: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch reference for numerical equivalence testing."""
    from .attention import score_norm as _sn

    raw = torch.matmul(Q, K.transpose(-2, -1)) * sm_scale

    if s_prev is not None:
        g = gate.view(1, -1, 1, 1)
        sc = scale.view(1, -1, 1, 1)
        sh = shift.view(1, -1, 1, 1)
        raw = raw + g * (sc * _sn(s_prev) + sh)

    s_out = raw.clone()

    if causal:
        mask = torch.triu(
            torch.full((raw.size(2), raw.size(3)), float("-inf"), device=raw.device),
            diagonal=1,
        )[None, None]
        raw = raw + mask

    w = F.softmax(raw, dim=-1)
    out = torch.matmul(w, V)
    return out, s_out
