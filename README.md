# Titanax

**Explicit-Parallel JAX Training Framework**

Titanax is a lightweight training framework that brings the user ergonomics of Hugging Face Accelerate / TorchTitan to **JAX**, while requiring **explicit** data/tensor/pipeline parallelization (no XLA auto-sharding). Users declare meshes, sharding rules, and collectives; Titanax wires the training loop, checkpointing, and observability around those choices.

## Quick Start

```python
import jax
import jax.numpy as jnp
import titanax as tx

# Simple model and data
def model_apply(params, x):
    return x @ params["w"] + params["b"]

params = {"w": jax.random.normal(jax.random.PRNGKey(42), (2, 1)) * 0.1, "b": jnp.zeros((1,))}
batch = {"x": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "y": jnp.array([[1.5], [3.5]])}

# Set up Titanax engine  
mesh = tx.MeshSpec(devices=[jax.devices("cpu")[0]], axes=("data",))
plan = tx.Plan(data_parallel=tx.DP(axis="data"))
engine = tx.Engine(mesh=mesh, plan=plan, optimizer=tx.optim.sgd(0.01), loggers=[tx.loggers.Basic()])

# Train with explicit step function
@tx.step_fn
def train_step(state, batch):
    loss, grads = jax.value_and_grad(lambda p: jnp.mean((model_apply(p, batch["x"]) - batch["y"]) ** 2))(state.params)
    return state.apply_gradients(grads=grads), {"loss": loss}

# Train for 3 steps
state = engine.create_state(params)
for step in range(3):
    state, metrics = train_step(state, batch)
    print(f"Step {step + 1}: loss={metrics['loss']:.6f}")
```

## Features

- **Explicit Parallelism**: Users must declare how arrays and computations are partitioned (DP/TP/PP)
- **Ergonomic API**: Run the same training script on 1 GPU, multi-GPU, or multi-host with minimal code changes
- **Production Ready**: Built-in logging, checkpointing, mixed precision, and reproducible randomness
- **Composable**: Parallel plans (DP/TP/PP) can be composed and validated before compilation

## Parallelization Roadmap

| Feature | Status | Description |
|---------|--------|-------------|
| **Data Parallel (DP)** | ✅ Complete | Gradient synchronization across data-parallel replicas |
| **Tensor Parallel (TP)** | 🚧 In Progress | Parameter sharding within model layers (P1 milestone) |  
| **Pipeline Parallel (PP)** | 🚧 Planned | Model stage parallelization with 1F1B scheduling (P2 milestone) |
| **DP×TP Composition** | 🚧 In Progress | Combined data and tensor parallelism |
| **DP×PP Composition** | 🚧 Planned | Combined data and pipeline parallelism |
| **TP×PP Composition** | 🚧 Planned | Combined tensor and pipeline parallelism |

**Note**: Titanax requires explicit mesh and collectives declarations - no XLA auto-sharding.

## Installation

```bash
# Install from source (development)
git clone https://github.com/pruthvipmr/titanax.git
cd titanax
uv pip install -e .

# Or directly from GitHub
pip install git+https://github.com/pruthvipmr/titanax.git
```

**Requirements**: Python ≥3.11, JAX ≥0.7.1

## Data Parallel Quickstart

For a complete MNIST data parallel example:

```bash
# Download MNIST data
uv run python examples/mnist_dp/download_mnist.py

# Train MNIST with data parallelism
uv run python examples/mnist_dp/train.py --steps=100 --eval-every=25

# Train with CNN model
uv run python examples/mnist_dp/train.py --model=cnn --steps=50 --batch-size=128
```

## Examples

- **[Minimal DP](examples/minimal_dp.py)**: 10-line CPU-only example demonstrating core concepts
- **[MNIST DP](examples/mnist_dp/)**: Complete MNIST training with data parallelism, checkpointing, and evaluation
- **[GPT Small TP](examples/gpt_small_tp/)**: GPT model with tensor parallelism (coming in P1)

## Contributing

```bash
# Install development dependencies
uv sync

# Run tests
uv run python -m pytest tests/ -v

# Format and lint
uv run black src/ tests/ examples/
uv run ruff check src/ tests/ examples/

# Type checking
uv run mypy src/titanax/
```

## Philosophy

Titanax embraces **explicit over implicit** parallelization:

- **Explicit meshes**: Users declare device topologies and axis names
- **Explicit sharding**: Users specify how parameters and data are partitioned  
- **Explicit collectives**: Users control when and how gradients/activations are synchronized

This design trades some convenience for predictability, performance control, and easier debugging in distributed training scenarios.
