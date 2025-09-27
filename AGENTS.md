# Titanax Development Guide

## Project Overview
Titanax is a lightweight JAX training framework that brings Hugging Face Accelerate/TorchTitan ergonomics to JAX with **explicit** parallelization. Users must declare meshes, sharding rules, and collectives - no XLA auto-sharding.


## Essential Commands

```bash
# Setup and install
uv pip install -e .

# fix python version mismatch
uv venv --python=3.13
uv sync

# Run tests
uv run python -m pytest tests/
uv run python -m pytest tests/unit/
uv run python -m pytest tests/integration/

# Type checking and formatting
uv run mypy src/titanax/
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/

# Run examples
uv run python examples/minimal_dp.py
uv run python examples/mnist_dp/train.py --steps=100 --eval-every=25
uv run python examples/gpt_small_tp/train.py
```
## Package Management
- Uses `uv` (not pip/conda)
- Dependencies in `pyproject.toml`
- Lock file: `uv.lock`

## Pre-commit Checks
**ALWAYS run these before committing:**
```bash
uv run mypy src/titanax/
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/
uv run python -m pytest tests/
```

## Code Style & Conventions

### Prime Directives
- **Clarity over cleverness**: Do not code-golf. Prefer simple, readable implementations
- **Small, surgical diffs**: Break big changes into a series of obviouswins. Avoid sprawling refactors.
- **Tests with every change**: Bug fixes come with a regression test; new features come with non-brittle tests.
- Don't churn docs/whitespace**: No cosmetic or doc-only edits unless explicitly requested

### Writing Code
- **Keep it small & explicit**: Short functions; return early; no hidden side-effects
- **No magic imports**: Absolute imports only; no `from x import *`
- **Limit dependencies**: Use the standard library unless a dependency removes substantial complexity
- **Comments explain "why"**: Avoid narrating the obvious "what"
- **Error messages are actionable**: Tell the user what went wrong and how to fix it

### Formatting
- **Type hints are pragmatic**: Prioritize public APIs and tricky internals; don't block progress on typing every corner
- **Pre-commit always passes locally**: Run linter, mypy, and a fast test subset before proposing a change
- **Format/Lint with Ruff**: `uv run ruff format .` and `uv run ruff check mlx_baselines3/ tests/` (add `--fix` for autofixes)
- **NO EMOJIS**: Never use any emojis

### Testing Policy
- **Every change includes tests**.
  -Bug fix → regression test that fails before, passes after.
  - Feature → unit tests + shape/dtype edge cases.
- **Avoid brittleness**. No tests that depend on timing, randomness (without seeding), or device-specific quirks unless the feature is device-specific.
- Property/fuzz tests welcome when invariants matter (e.g., algebraic rewrites, shape rules).
- Refactors with “no behavior change” should demonstrate equivalence (e.g., replay/process-comparison or identical outputs on a golden set).

### Change-scoping rules
- **Touch as little as possible**: Minimize blast radius
- **Keep "core" clean"**: Don't churn peripheral or poorly-tested areas without necessity
- **Split PRs logically**: Land enabling refactors first; then the 3-line feature that becomes obvious after

### Commit/PR hygiene
- **Title**: concise, imperative ("Fuse X into Y for simpler kernel schedule")
- **Body**: problem -> approach -> why it's simpler -> test/bench evidence -> risk & rollout plan
- **Scope**: one theme per PR; follow-ups for anything orthogonal


## Project Structure
```
src/titanax/
├── __init__.py
├── types.py            # Common type aliases and protocols
├── exceptions.py       # Exception hierarchy with suggestions
├── compat.py           # JAX version compatibility layer - IMPLEMENTED
├── runtime/            # Mesh, distributed init
├── parallel/           # Plans (DP/TP/PP), sharding rules, TP helpers - ENHANCED
│   ├── plan.py         # DP/TP/PP plan definitions with validation - ENHANCED
│   ├── tp_helpers.py   # Tensor parallel rule helpers for common patterns - NEW
│   └── pp.py           # Pipeline parallel Stage and PipelineSchedule - NEW
├── exec/               # Engine, collectives, step functions
├── optim/              # Optax adapters
├── io/                 # Checkpointing (checkpoint.py, orbax_io.py)
├── data/               # Dataloaders
├── logging/            # Observability
└── launch/             # CLI launcher

tests/
├── unit/            # Component validation
│   ├── test_plan.py      # Enhanced with all parallelism combinations
│   ├── test_tp_helpers.py # Golden spec validation tests - NEW
│   └── test_pp.py        # Stage and schedule validation tests - NEW
├── integration/     # End-to-end workflows
└── benchmarks/      # Performance validation

examples/
├── mnist_dp/        # MNIST Data Parallel example
├── gpt_small_tp/    # GPT Tensor Parallel example
├── gpt_small_pp/    # GPT Pipeline Parallel example
└── configs/         # YAML configuration examples
```
