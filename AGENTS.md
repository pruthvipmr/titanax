# Titanax Development Guide

## Project Overview
Titanax is a lightweight JAX training framework that brings Hugging Face Accelerate/TorchTitan ergonomics to JAX with **explicit** parallelization. Users must declare meshes, sharding rules, and collectives - no XLA auto-sharding.


## Build & Test Commands
```bash
# Install in development mode
uv pip install -e .

# Run all tests
uv run python -m pytest tests/

# Run unit tests only
uv run python -m pytest tests/unit/

# Run integration tests
uv run python -m pytest tests/integration/

# Run benchmarks
uv run python -m pytest tests/benchmarks/

# Run comprehensive test suite with coverage summary
uv run python -m pytest tests/ -v --tb=short --durations=10

# Type checking
uv run mypy src/titanax/

# Format code
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/

# Run examples
uv run python examples/minimal_dp.py                                      # 10-line CPU-only example
uv run python examples/mnist_dp/train.py --steps=100 --eval-every=25
uv run python examples/mnist_dp/train.py --model=cnn --steps=50 --batch-size=128
uv run python examples/mnist_dp/download_mnist.py
uv run python examples/mnist_dp/test_synthetic.py
uv run python examples/gpt_small_tp/train.py

# Validate package structure and syntax (works without dependencies)
tools/quick_validation.py
```

## Pre-commit Checks
**ALWAYS run these commands before creating any commit:**
```bash
# 1. Type checking - must pass
uv run mypy src/titanax/

# 2. Code formatting - auto-fix
uv run black src/ tests/ examples/

# 3. Linting - must pass  
uv run ruff check src/ tests/ examples/

# 4. All tests - must pass
uv run python -m pytest tests/ -v --tb=short
```

## Code Style Guidelines
- **Type annotations**: Heavy type annotation on all public APIs
- **Data structures**: Prefer `dataclasses` over `pydantic` for simplicity
- **Naming**: Use snake_case for functions/variables, PascalCase for classes
- **Imports**: Group stdlib, third-party, local imports with blank lines between
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
├── types.py         # Common type aliases and protocols
├── exceptions.py    # Exception hierarchy with suggestions
├── runtime/         # Mesh, distributed init
├── parallel/        # Plans (DP/TP/PP), sharding rules
├── exec/            # Engine, collectives, step functions
├── optim/           # Optax adapters
├── io/              # Checkpointing (checkpoint.py, orbax_io.py) - IMPLEMENTED
├── data/            # Dataloaders
├── logging/         # Observability (base.py, basic.py, csv.py, tensorboard.py)
└── launch/          # CLI launcher

tests/
├── unit/            # Component validation
├── integration/     # End-to-end workflows
└── benchmarks/      # Performance validation

examples/
├── mnist_dp/        # MNIST Data Parallel example
├── gpt_small_tp/    # GPT Tensor Parallel example
├── gpt_small_pp/    # GPT Pipeline Parallel example
└── configs/         # YAML configuration examples
```
