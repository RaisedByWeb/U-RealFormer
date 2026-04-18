<p align="center">
  <strong>U-RealFormer</strong><br>
  <em>Universal RealFormer — Persistent Relational Attention for Deep Transformers</em>
</p>

<p align="center">
  <a href="#background">Background</a> &bull;
  <a href="#installation">Install</a> &bull;
  <a href="#quickstart">Quickstart</a> &bull;
  <a href="#v1-api">v1 API</a> &bull;
  <a href="#experimental-results">Results</a> &bull;
  <a href="#repo-structure">Repo structure</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

---

## Background

### The original RealFormer

[RealFormer](https://arxiv.org/abs/2012.11747) (He et al., Google Research, 2020)
introduced **residual attention** -- the idea of adding the raw attention scores
from layer *l-1* directly into layer *l* before softmax:

```
S_l = raw_l + S_{l-1}
```

This simple addition showed consistent improvements on GLUE, SQuAD, and
image classification, demonstrating that **preserving relational structure
across depth** is a valuable inductive bias. The key insight: attention
scores carry useful information that should not be discarded between layers.

### Limitations of naive residual attention

The original formulation has three practical problems at scale:

1. **Instability in deep stacks.** Raw score accumulation across many layers
   causes magnitude drift -- softmax is sensitive to row-wise contrast, not
   absolute magnitude, so uncalibrated residuals overwhelm current-layer evidence.
2. **No per-head control.** All heads receive the same residual signal with
   equal weight. Different heads specialise in different relational types
   (local syntax vs global context) and need independent control.
3. **No normalisation.** The residual is injected in raw score space, making
   it sensitive to the scale of scores at different depths.

### What U-RealFormer adds

**U-RealFormer** (Universal RealFormer) is an extension of RealFormer focused on
making residual attention stable and controllable at scale. It builds on the
original idea with three stabilisation mechanisms that make residual attention
practical for deep (48+ layer) models:

```
S_l = raw_l + sigmoid(alpha) * (gamma * ScoreNorm(S_{l-1}) + beta)
```

| Component     | What it does                                                     | Why it matters                        |
| ------------- | ---------------------------------------------------------------- | ------------------------------------- |
| **ScoreNorm** | Row-wise standardisation in attention space (FP16-safe)          | Prevents magnitude drift across depth |
| **alpha** gate | Per-head learned control of residual openness                   | Heads choose their own relational half-life |
| **gamma** scale | Per-head contrast of the normalised residual                   | Calibrates signal strength per head   |
| **beta** shift | Per-head bias in cross-layer score flow                         | Completes the learnable affine transform |

Additionally, this library provides:
- A **Phase 1 / Phase 2 detach toggle** for controlling gradient flow through the residual path
- **Per-layer residual control** (`residual_layers`) instead of uniform stride
- **Hallucination gap strategies** (segmented BPTT, cache dropout, self-distillation) for decoder training
- A **fused Triton kernel** for inference
- Empirical evidence: stable gradients at 48 layers, and lower loss than baseline on WikiText-2

---

## Installation

```bash
git clone https://github.com/RaisedByWeb/U-RealFormer.git
cd U-RealFormer
pip install -e ".[dev]"

# Verify
pytest tests/ -v
```

**Requirements:** Python >= 3.10, PyTorch >= 2.1

---

## Quickstart

### Encoder (bidirectional)

```python
from u_realformer import RealFormerConfig, RealFormerEncoder

cfg = RealFormerConfig(hidden=768, heads=12, layers=12)
encoder = RealFormerEncoder(cfg)

import torch
ids  = torch.randint(0, cfg.vocab_size, (1, 128))
mask = torch.ones_like(ids)

hidden = encoder(ids, attention_mask=mask)   # (1, 128, 768)
```

### Decoder (causal, autoregressive)

```python
from u_realformer import RealFormerConfig, RealFormerDecoder, DecoderCache

cfg = RealFormerConfig(hidden=768, heads=12, layers=12, vocab_size=32000)
decoder = RealFormerDecoder(cfg)

ids = torch.randint(0, cfg.vocab_size, (1, 64))
logits = decoder(ids)   # (1, 64, 32000)

# Incremental generation with KV + score cache
cache = DecoderCache.empty(cfg.layers)
for token_id in ids[0]:
    logits = decoder(token_id.view(1, 1), cache=cache)
next_token = logits[0, -1].argmax()
```

### Configuring the residual path

```python
cfg = RealFormerConfig(
    hidden=768, heads=12, layers=24,

    # Per-layer control: which layers receive the score residual
    residual_layers={0, 1, 2, 3, 4, 5, 6, 7},  # first 8 of 24
    # Or use the convenience stride:
    # skip_k=2,  # every other layer (auto-generates residual_layers)

    alpha_init=-1.0,     # sigmoid(-1) ~ 0.27 -- conservative gate
    gamma_init=0.4,      # scale magnitude
    beta_init=0.0,       # shift (0 = no shift at init)

    # Detach roadmap: Phase 1 (detach) vs Phase 2 (coupled)
    residual_grad_flow=False,  # True enables end-to-end gradient through scores
    residual_grad_clip=1.0,    # safety clamp for Phase 2
)
```

### Hallucination gap training strategies

```python
from u_realformer import TrainingConfig, segmented_step, CacheDropoutSchedule

tcfg = TrainingConfig(
    segment_ratio=0.5,       # Strategy A: segmented BPTT
    cache_dropout_p=0.3,     # Strategy B: stochastic s_row dropout
    distill_weight=1.0,      # Strategy C: online self-distillation
    distill_temperature=2.0,
)

# In your training loop:
loss = segmented_step(model, input_ids, labels, loss_fn, tcfg)
```

---

## v1 API

### `RealFormerConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `hidden` | `int` | 768 | Hidden dimension |
| `heads` | `int` | 12 | Number of attention heads |
| `head_dim` | `int` | 64 | Dimension per head |
| `layers` | `int` | 12 | Number of Transformer layers |
| `seq_len` | `int` | 2048 | Maximum sequence length |
| `vocab_size` | `int` | 30522 | Vocabulary size |
| `dropout` | `float` | 0.1 | Dropout rate |
| `skip_k` | `int` | 1 | Convenience stride for `residual_layers` |
| `residual_layers` | `set[int]` | auto | Explicit set of layer indices that receive score residual |
| `alpha_init` | `float` | 0.0 | Gate init (logit-space; sigmoid(0) = 0.5) |
| `gamma_init` | `float` | 1/sqrt(3) | Per-head scale init |
| `beta_init` | `float` | 0.0 | Per-head shift init |
| `residual_grad_flow` | `bool` | False | Phase 1 (detach) vs Phase 2 (coupled gradients) |
| `residual_grad_clip` | `float` | 1.0 | Max gradient norm on residual path (Phase 2) |
| `rank` | `int` | 0 | Low-rank factorisation rank (0 = disabled) |
| `stride` | `int` | 1 | Temporal stride for compressed scores |

### `TrainingConfig`

| Parameter | Type | Default | Description |
|---|---|---|---|
| `segment_ratio` | `float` | 0.5 | Where to split for segmented BPTT |
| `segment_min_len` | `int` | 16 | Minimum tokens per segment |
| `cache_dropout_p` | `float` | 0.0 | Probability of zeroing s_row (0 = disabled) |
| `distill_weight` | `float` | 0.0 | KL coefficient for self-distillation (0 = disabled) |
| `distill_temperature` | `float` | 2.0 | Softmax temperature for distillation |

### Core modules

| Module | Signature | Description |
|---|---|---|
| `RealFormerEncoder` | `.forward(input_ids, attention_mask=None) -> (B, T, H)` | Bidirectional encoder |
| `RealFormerDecoder` | `.forward(input_ids, cache=None) -> (B, T, V)` | Causal decoder with KV + score cache |
| `GatedResidualAttention` | `.forward(x, s_prev=None, mask=None) -> (out, raw)` | Core attention with score residual |
| `score_norm` | `(S, eps=1e-6) -> Tensor` | FP16-safe row-wise standardisation |
| `DecoderCache` | `.empty(n_layers) -> DecoderCache` | KV + score-row cache for generation |

### Training utilities

| Function | Description |
|---|---|
| `segmented_step()` | Strategy A: truncated-BPTT segmented training |
| `distillation_step()` | Strategy C: online self-distillation |
| `CacheDropoutSchedule` | Strategy B: annealed cache dropout schedule |
| `set_cache_dropout()` | Set dropout probability on all attention modules |
| `pick_split()` | Random or deterministic sequence split point |

---

## Experimental Results

### Jacobian Audit (gradient stability across depth)

Tested at 12, 24, and 48 layers. Both detached (Phase 1) and coupled (Phase 2) modes are stable.

| Depth | Slope (detached) | Slope (coupled) | Head variance | Verdict |
|-------|-----------------|-----------------|---------------|---------|
| 12L | -0.014 | -0.036 | 1.13 | STABLE |
| 24L | -0.013 | -0.018 | 1.12 | STABLE |
| 48L | -0.012 | -0.013 | 1.15 | STABLE |

Log-norm slope near 0 = stable. Threshold: < -0.2 = vanishing, > 0.2 = exploding.
Per-head variance max/min ratio near 1 = balanced. Threshold: > 100 = explosion.

**Decision:** Phase 2 (co-evolution with coupled gradients) is promoted from theory to candidate.

### WikiText-2 Character-Level LM (6L/256H, 5000 steps)

| Metric | Baseline | RealFormer-Evo |
|--------|----------|----------------|
| Final eval loss | 1.634 | **1.620** |
| Final perplexity | 5.1 | **5.1** |
| Gate (L1) | 0.500 (fixed) | **0.702** (learned) |
| Gamma (L1) | 0.577 (fixed) | **1.414** (learned) |

The model learned to open the gate at early layers (L1: 0.70, L2: 0.63) and keep it
near default at late layers (L4-L5: 0.50). Early layers amplified the residual signal
(gamma grew from 0.577 to 1.414 at L1). This matches the theoretical prior: early
layers benefit most from relational persistence.

### Copy-Shift Synthetic Task (6L/256H, 5000 steps)

Both models solve the task perfectly (loss -> 0). The structural diagnostic shows
residual attention preserves cross-layer score similarity 6x better than baseline
(cosine 0.655 vs 0.109).

---

## Practical Impact

U-RealFormer is motivated by a simple inefficiency in deep Transformers:
models repeatedly recompute relational structure that was already discovered
in earlier layers.

By preserving and stabilising attention scores across depth, U-RealFormer
reduces the need for this re-discovery. This has implications in settings
where relational consistency and long-context reasoning are critical:

### Long-context reasoning

Tasks such as legal analysis, policy interpretation, and technical documentation
require maintaining relationships across long sequences. Persistent attention
improves stability of these relationships without relying solely on depth.

### Multi-step decision systems

In structured decision pipelines (e.g. underwriting, claims triage, fraud
detection), early relational signals influence downstream decisions. Preserving
attention structure allows later layers to build on prior reasoning instead of
reconstructing it.

### Training efficiency and depth scaling

Repeated relational discovery increases both compute cost and optimisation
difficulty. By reusing attention structure, U-RealFormer provides a path toward
deeper models with more stable gradients and potentially improved compute
efficiency.

---

## Repo structure

```
U-RealFormer/
├── u_realformer/               # core library
│   ├── __init__.py
│   ├── attention.py              # GatedResidualAttention + ScoreNorm + grad clip
│   ├── config.py                 # RealFormerConfig + TrainingConfig
│   ├── decoder.py                # causal decoder + KV/score cache
│   ├── encoder.py                # bidirectional encoder
│   ├── low_rank.py               # rank-r score compression
│   ├── training.py               # hallucination gap strategies (A, B, C)
│   └── triton_kernels.py         # fused Triton attention kernel
├── kernels/                      # Rust/CUDA fused ops (v4 roadmap)
│   ├── src/lib.rs
│   └── Cargo.toml
├── benchmarks/
│   ├── glue.py
│   ├── squad_v2.py
│   ├── memory_profile.py
│   └── triton_attn_bench.py      # Triton vs PyTorch wall-clock comparison
├── experiments/
│   ├── prove_residual.py         # Phase 0: convergence race + diagnostics
│   ├── jacobian_audit.py         # Phase 2: gradient stability across depth
│   ├── hallucination_gap.py      # hallucination gap measurement
│   ├── bench_structured.py       # copy-shift synthetic benchmark
│   ├── bench_wikitext.py         # WikiText-2 real corpus benchmark
│   ├── ablation_alpha.py         # gate init sweep
│   └── skip_k_sweep.py           # residual sparsity sweep
├── tests/                        # 85 tests (75 CPU, 10 GPU-only)
│   ├── test_attention.py
│   ├── test_causal.py
│   ├── test_memory.py
│   ├── test_hallucination_gap.py
│   ├── test_detach_roadmap.py
│   └── test_triton_kernels.py
├── docs/
│   ├── architecture.md
│   └── DESIGN_SPEC_ARCH030.md
├── .github/
│   ├── workflows/ci.yml
│   └── ISSUE_TEMPLATE/
├── pyproject.toml
├── CONTRIBUTING.md
├── CHANGELOG.md
├── LICENSE
└── README.md
```

---

## Research ladder

### v1 -- The Vertical Highway (current)

- Cross-layer gated score propagation with ScoreNorm
- Per-head alpha/gamma/beta affine
- FP16-safe ScoreNorm
- Detach toggle (Phase 1 vs Phase 2)
- Residual gradient clipping
- Non-uniform `residual_layers` control
- Decoupled cross-layer and cross-step paths
- Hallucination gap strategies (segmented BPTT, cache dropout, self-distillation)
- Fused Triton kernel (inference)
- Jacobian Audit: stable at 48 layers
- WikiText-2: lower loss than baseline, gate learns meaningful policy

### v2 -- Co-Evolution

End-to-end relational learning via the ScoreNorm Jacobian. The Jacobian Audit
confirms this is viable (stable gradients at 48L). Next: train with
`residual_grad_flow=True` and measure whether early layers adapt their attention
to serve downstream layers.

### v3 -- The Temporal Bridge

Extend residual attention into decoder settings with cached score-state.
The hallucination gap infrastructure is built; needs GPU-scale training.

### v4 -- Systems Optimisation

Fuse core operations into efficient kernels. The Triton forward kernel is
implemented; backward kernel and production benchmarks are next.

---

## The thesis

The original Transformer showed that attention could replace recurrence.
[RealFormer](https://arxiv.org/abs/2012.11747) showed that attention scores
carry useful structure worth preserving across depth.

U-RealFormer takes that insight and asks: **how do we make it stable, controllable,
and practical at scale?**

> **Deep attention should not have to rediscover the same relational
> structure over and over again.**

---

## Contributing

We welcome contributions. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

We are especially interested in:

- Residual attention variants
- Normalisation strategies for score propagation
- Decoder-safe persistent score state
- Efficient fused kernels (Triton / CUDA)
- Benchmark and ablation design

---

## References

- **RealFormer:** He, P., Liu, X., Gao, J., & Chen, W. (2020).
  *RealFormer: Transformer Likes Residual Attention.*
  [arXiv:2012.11747](https://arxiv.org/abs/2012.11747) -- The original
  paper that proposed residual attention across Transformer layers.

- **Attention Is All You Need:** Vaswani, A., et al. (2017).
  [arXiv:1706.03762](https://arxiv.org/abs/1706.03762) -- The Transformer
  architecture that U-RealFormer extends.

---

## Author

**Uriel Aharoni** -- Tech entrepreneur and PhD student, Golden Gate University.

---

## License

Apache 2.0 -- see [LICENSE](LICENSE) for details.
