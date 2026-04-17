# Architecture Overview

## Core idea

Standard Transformers rebuild attention structure from scratch at every layer.
RealFormer-Evo treats **attention scores as persistent state** — propagating
relational structure across depth through gated, normalised residual paths.

## The score residual

At layer *l*, the raw attention scores are augmented:

```
S_l = raw_l + sigmoid(α) · γ · ScoreNorm(S_{l-1})
```

| Component      | Role                                                       |
| -------------- | ---------------------------------------------------------- |
| `raw_l`        | Fresh Q·Kᵀ / √d scores from the current layer             |
| `α` (alpha)    | Per-head gate (logit-space); sigmoid controls openness     |
| `γ` (gamma)    | Per-head scale; init 1/√3 for stable early training        |
| `ScoreNorm`    | Row-wise zero-mean / unit-variance standardisation         |
| `.detach()`    | Residual is treated as signal, not an optimisation path    |

## ScoreNorm

Row-wise standardisation with biased variance (matching `nn.LayerNorm`):

```
ScoreNorm(S)_ij = (S_ij − μ_i) / √(σ²_i + ε)
```

This ensures the residual shifts the softmax distribution in a controlled way,
rather than overwhelming current-layer evidence with raw magnitudes.

## skip_k

When `skip_k > 1`, the score residual is injected only every *k* layers:

```
layer_idx % skip_k == 0  →  receive residual
```

This creates sparser "score highways" that trade relational frequency for
memory and compute savings in very deep stacks.

## Low-rank compression (planned)

When `rank > 0`, the full T×T score matrix is factored as P @ Qᵀ where
P, Q ∈ ℝ^{T × rank}, reducing cross-layer memory from O(T²) to O(T·rank).

## Module map

```
realformer_evo/
├── config.py       RealFormerConfig dataclass
├── attention.py    GatedResidualAttention + score_norm
├── low_rank.py     LowRankProjector (compress / decompress)
├── encoder.py      RealFormerEncoder (bidirectional)
└── decoder.py      RealFormerDecoder (causal + KV cache)
```

## Research ladder

| Version | Codename           | Focus                                    |
| ------- | ------------------ | ---------------------------------------- |
| v1      | Vertical Highway   | Encoder-first gated score residuals      |
| v2      | Per-head Scaling   | Learnable γ separation (gate vs. scale)  |
| v3      | Temporal Bridge    | Decoder cached score-state               |
| v4      | Systems            | Fused Rust/CUDA kernels                  |
