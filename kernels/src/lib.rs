//! RealFormer-Evo fused kernels (v4 roadmap).
//!
//! Planned operations:
//!   1. Fused low-rank matmul: P @ Q^T with online ScoreNorm
//!   2. Gated residual injection: raw + gate * scale * score_norm(S_prev)
//!   3. Strided score compression for long sequences
//!
//! For now this crate is a scaffold.  The first kernel will target
//! the low-rank matmul path (see `realformer_evo/low_rank.py`).

/// Placeholder: fused low-rank score reconstruction.
///
/// Given P (B*H, T, R) and Q (B*H, T, R), computes P @ Q^T with
/// row-wise score normalisation fused into the same pass.
pub fn fused_low_rank_matmul(
    _p: &[f32],
    _q: &[f32],
    _batch: usize,
    _t: usize,
    _rank: usize,
) -> Vec<f32> {
    todo!("implement in v4 — see DESIGN_SPEC_ARCH030.md")
}

#[cfg(test)]
mod tests {
    #[test]
    fn placeholder() {
        assert_eq!(2 + 2, 4);
    }
}
