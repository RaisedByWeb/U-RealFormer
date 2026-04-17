# ARCH-030: Causal Decoder Design Specification

**Status:** Active  
**Authors:** U-RealFormer contributors  
**Created:** 2025-04-17

---

## Motivation

The v1 encoder demonstrates that gated score residuals improve optimisation
stability in bidirectional settings. ARCH-030 extends this to **autoregressive
decoding** where two additional constraints apply:

1. **Causal masking** — position *i* must not attend to positions *j > i*.
2. **Incremental generation** — at inference time, only the last token is
   processed per step; KV caches must grow without recomputing prefixes.

## Design

### Cross-layer residual (same as encoder)

```
S_l = raw_l + gate · scale · ScoreNorm(S_{l-1}.detach())
```

During training (full-sequence forward), this is identical to the encoder path.

### Cross-step residual (decoder-only)

During cached generation, the model also has access to the score row from the
**previous decode step of the same layer**:

```
r = gate · scale · ScoreNorm(S_{l-1})
    + (1 − gate) · scale · ScoreNorm(pad(s_row_prev))
```

Where `s_row_prev` is the last row of the raw score matrix from step *t − 1*,
zero-padded to match the new key length.

### Cache structure

```python
@dataclass
class LayerCache:
    k:     Tensor | None   # (B, H, t, D)
    v:     Tensor | None   # (B, H, t, D)
    s_row: Tensor | None   # (B, H, 1, t)
```

Each layer stores its own cache. `DecoderCache` holds a list of `LayerCache`
objects, one per layer.

## Invariants

1. **First-token consistency** — the first decode step (no prior cache) must
   produce the same output as the corresponding position in a full prefill.
2. **Causal isolation** — modifying token *j* must not affect outputs at
   positions *i < j*.
3. **Monotonic cache growth** — `cache.layers[l].k.size(2)` must equal the
   total number of tokens processed so far.

## Open questions

- Should `s_row` be normalised with ScoreNorm before storage, or after retrieval?
  Current implementation: after retrieval (at injection time).
- Should the cross-step gate share `alpha` with the cross-layer gate, or have
  its own parameter? Current implementation: shared, with `(1 − gate)` weighting.

## Test coverage

See `tests/test_causal.py`:
- `test_future_token_independence` — causal isolation
- `test_first_token_matches` — first-token consistency
- `test_cache_grows` — monotonic cache growth
