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

## Feature Workflow
- Create a dedicated git worktree for each feature before making changes, e.g. `git worktree add ../titanax-myfeature myfeature-branch`.
- Perform all feature development (code edits, tests, commits) from within that feature-specific worktree.
- After the feature is merged, remove the worktree with `git worktree remove ../titanax-myfeature` and delete the branch if no longer needed.

## Pre-commit Checks
**ALWAYS run these before committing:**
```bash
uv run mypy src/titanax/
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/
uv run python -m pytest tests/
```

## Code Style Guidelines
- **Type annotations**: Heavy type annotation on all public APIs
- **Data structures**: Prefer `dataclasses` over `pydantic` for simplicity
- **Naming**: Use snake_case for functions/variables, PascalCase for classes
- **Imports**: Group stdlib, third-party, local imports with blank lines between
- **JAX Compatibility**: Always use `from titanax.compat import ...` for JAX APIs that change between versions (pjit, shard_map, sharding types, collectives)
- **Static shapes**: Use static shapes to avoid JAX recompiles
- **Formatting**: Use `black` for code formatting, `ruff` for linting
- **Comments**: Focus on why, not what. Document complex JAX transformations
- **Functions**: Keep functions focused and testable, avoid side effects
- **Style**: Don't ever use emojis in any code
- **Types**: Use common types from `titanax.types` (PyTree, Array, etc.)
- **Exceptions**: Raise specific Titanax exceptions with helpful suggestions
- **Protocols**: Use `@runtime_checkable` protocols for interfaces

## Testing Instructions
- **Unit tests**: Test individual components in isolation with clear assertions
- **Integration tests**: Test end-to-end workflows and training parity
- **Benchmarks**: Validate scaling performance and memory usage
- Use `pytest` fixtures for common test setup (devices, meshes, data)
- Mock external dependencies (checkpointing, logging) in unit tests
- Test both single and multi-device scenarios where applicable

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
