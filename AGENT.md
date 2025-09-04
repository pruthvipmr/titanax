# Titanax Development Guide

## Project Overview
Titanax is a lightweight JAX training framework that brings Hugging Face Accelerate/TorchTitan ergonomics to JAX with **explicit** parallelization. Users must declare meshes, sharding rules, and collectives - no XLA auto-sharding.

## Key Technologies
- **JAX**: Core numerical computing with SPMD
- **Optax**: Optimizer integration
- **Orbax**: Checkpointing backend
- **Python**: uv for package management
- **XLA**: Compilation (but no auto-sharding)

## Package Management
Using `uv` for Python package management:
```bash
# Install dependencies
uv pip install -e .

# Add new dependency
uv add <package>

# Development dependencies
uv add --group dev <package>

# Sync environment
uv sync
```

## Development Commands
```bash
# Install in development mode
uv pip install -e .

# Run tests
uv run python -m pytest tests/

# Run unit tests only
uv run python -m pytest tests/unit/

# Run integration tests
uv run python -m pytest tests/integration/

# Run benchmarks
uv run python -m pytest tests/benchmarks/

# Type checking
mypy titanax/

# Format code
black titanax/ tests/ examples/
ruff check titanax/ tests/ examples/

# Run examples
uv run python examples/mnist_dp/train.py
uv run python examples/gpt_small_tp/train.py
```

## Project Structure
```
titanax/
├── __init__.py
├── runtime/          # Mesh, distributed init
├── parallel/         # Plans (DP/TP/PP), sharding rules
├── exec/            # Engine, collectives, step functions
├── optim/           # Optax adapters
├── io/              # Checkpointing (Orbax)
├── data/            # Dataloaders
├── logging/         # Observability
└── launch/          # CLI launcher
```

## Key Concepts
- **Mesh**: Logical device grid with named axes (e.g., "data", "model")
- **Plan**: Composition of DP/TP/PP strategies
- **PartitionSpec**: Maps array dimensions to mesh axes
- **Collectives**: Explicit cross-replica ops (psum, all_gather, etc.)
- **SPMD**: Single Program, Multiple Data execution


## Testing Strategy
- **Unit tests**: Component validation, type checks
- **Integration tests**: End-to-end training parity
- **Benchmarks**: Scaling performance validation

## Implementation Guidelines
- Use `pjit` with `NamedSharding` for TP
- Use `pmap` for pure DP where appropriate
- Prefer `dataclasses` over `pydantic`
- Heavy type annotation on public APIs
- Static shapes to avoid recompiles
- Keep examples runnable <10 minutes

## Milestones
- **P0**: DP Core (MeshSpec, DP plan, Engine.fit)
- **P1**: TP Core (TP rules, DP×TP composition)
- **P2**: PP Core (Stage API, 1F1B scheduler)
- **P3**: DX & Docs (CLI, YAML config, docs)

## Dependencies (Core)
- jax
- optax
- orbax-checkpoint
- numpy
- pyyaml (for config)
- typer (for CLI)

## Dependencies (Optional)
- tensorboard (logging)
- wandb (logging)
- flax (model adapters - v0.2+)

## Error Handling Focus
- Mesh/Plan validation with clear messages
- Collective guards with axis existence checks
- Shape compatibility validation
- Clean multi-host failure handling

## Performance Targets
- MNIST-DP: 1→8 GPU scaling with <1e-4 loss parity
- GPT-small TP: Reference perplexity within tolerance
- Checkpoint round-trip with resharding support
