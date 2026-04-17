# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Hallucination gap strategies** (`TrainingConfig`, `realformer_evo/training.py`):
  - Strategy A: segmented BPTT (`segmented_step`) -- truncated forward with cache handoff
  - Strategy B: stochastic cache dropout (`CacheDropoutSchedule`, `set_cache_dropout`)
  - Strategy C: online self-distillation (`distillation_step`)
  - `DecoderCache.detach_()` and `LayerCache.detach_()` for BPTT boundaries
  - `RealFormerDecoder.forward_segmented()` for two-segment training
- **Detach roadmap** (Phase 1 / Phase 2 toggle):
  - `beta` parameter -- per-head learnable shift completing the ScoreNorm affine
  - `residual_grad_flow` config toggle (detach vs coupled gradients)
  - `residual_grad_clip` -- safety clamp for Phase 2 gradient flow
  - `_ClipResidualGrad` autograd function for residual-path gradient clipping
  - FP16-safe `score_norm` (fp32 upcast for variance computation)
- **`residual_layers: set[int]`** -- per-layer control of which layers receive the score residual, replacing the uniform `skip_k` modular arithmetic. `skip_k` remains as a convenience shorthand that auto-generates the set.
- **Decoupled cross-layer and cross-step residual paths** in `CausalAttention` -- the temporal `s_row` path now fires independently of whether the layer receives cross-layer `s_prev`.
- **Audit hooks** (`_audit_hook`) on `EncoderLayer` and `DecoderLayer` for instrumenting the real forward path.
- **Fused Triton kernel** (`realformer_evo/triton_kernels.py`):
  - `_score_norm_stats_kernel` -- row-wise mean/rstd precomputation
  - `_fused_residual_attn_fwd` -- tiled QK^T + residual injection + online softmax + V accumulation
  - `FusedResidualAttentionFn` autograd wrapper (Triton forward, PyTorch backward)
  - Automatic dispatch in `GatedResidualAttention` and `CausalAttention` when Triton + CUDA available
  - `reference_residual_attention` for numerical equivalence testing
- **Experiments**:
  - `prove_residual.py` -- convergence race (baseline vs Evo) with gate/residual diagnostics
  - `jacobian_audit.py` -- gradient stability across depth (slope metric, per-head variance)
  - `hallucination_gap.py` -- gap measurement across training strategies
  - `bench_structured.py` -- copy-shift synthetic benchmark
  - `bench_wikitext.py` -- WikiText-2 character-level LM benchmark
- **Tests**: `test_hallucination_gap.py`, `test_detach_roadmap.py`, `test_triton_kernels.py` (85 total tests)
- `benchmarks/triton_attn_bench.py` -- wall-clock + memory comparison

### Changed

- `score_norm` now always computes statistics in fp32 (FP16 safety).
- `EncoderLayer` and `DecoderLayer` use `self.receives_residual` (bool) instead of `self.idx % self.skip_k == 0`.
- Residual injection formula updated: `gate * (scale * score_norm(s_prev) + shift)` (was `gate * scale * score_norm(s_prev)`).
- Config validation: `skip_k >= 1`, `residual_layers` indices validated against layer count.

## [0.1.0] -- 2025-04-17

### Added

- `RealFormerConfig` with residual-attention, low-rank, and stride parameters.
- `GatedResidualAttention` with per-head alpha/gamma gates and ScoreNorm.
- `score_norm` -- row-wise standardisation for score tensors.
- `LowRankProjector` -- rank-r score compression and decompression.
- `RealFormerEncoder` -- bidirectional encoder with gated score residuals.
- `RealFormerDecoder` -- causal decoder with KV cache and cross-step score residuals.
- GLUE and SQuAD v2 benchmark scaffolds.
- Memory profiler (`benchmarks/memory_profile.py`).
- Alpha ablation and skip_k sweep experiments.
- Test suite: attention, causal correctness, and memory sanity checks.
- CI workflow for Python 3.10-3.12 and Rust kernel checks.
- Architecture docs and ARCH-030 causal decoder spec.
- Contributing guide and issue templates.
