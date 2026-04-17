# Contributing to RealFormer-Evo

Thank you for your interest in contributing! This project is research-grade
and we value clean, well-tested, well-motivated changes.

## Getting started

```bash
git clone https://github.com/urielaharoni/realformer-evo.git
cd realformer-evo
pip install -e ".[dev]"
pytest tests/ -v
```

## What we're looking for

We are especially interested in contributions around:

- **Residual attention variants** — new gating strategies, alternative
  propagation schedules, cross-attention residuals.
- **Normalisation strategies** — alternatives to row-wise ScoreNorm,
  learned normalisation, attention-space calibration.
- **Decoder-safe persistent score state** — extensions to the cached
  generation path, long-context handling, speculative decoding compat.
- **Efficient fused kernels** — Rust/CUDA implementations of the
  low-rank matmul, gated injection, or strided compression paths.
- **Benchmark and ablation design** — new evaluation tasks, better
  ablation controls, training stability diagnostics.

## Submitting a change

1. **Open an issue first** for non-trivial changes. Use the `arch_proposal`
   template for architectural changes.
2. **Fork & branch** from `main`.
3. **Write tests.** Every new module needs coverage in `tests/`.
4. **Run the checks:**
   ```bash
   ruff check realformer_evo/ tests/
   pytest tests/ -v
   ```
5. **Open a PR** with a clear description of what changed and why.

## Code style

- Python 3.10+, type hints encouraged.
- Ruff for linting (config in `pyproject.toml`).
- Prefer clarity over cleverness. This is research code that people
  need to read, understand, and modify.

## Commit messages

Use short, descriptive messages:
```
Add skip_k sweep experiment (ARCH-021)
Fix ScoreNorm variance for n=1 edge case
```

## Architecture proposals

Major changes to the residual path, normalisation, or cache design
should be discussed in an issue using the **ARCH proposal** template
before implementation begins.

## License

By contributing, you agree that your contributions will be licensed
under the Apache 2.0 License.
