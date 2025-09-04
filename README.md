# Titanax

**Explicit-Parallel JAX Training Framework**

Titanax is a lightweight training framework that brings the user ergonomics of Hugging Face Accelerate / TorchTitan to **JAX**, while requiring **explicit** data/tensor/pipeline parallelization (no XLA auto-sharding). Users declare meshes, sharding rules, and collectives; Titanax wires the training loop, checkpointing, and observability around those choices.

## Features

- **Explicit Parallelism**: Users must declare how arrays and computations are partitioned (DP/TP/PP)
- **Ergonomic API**: Run the same training script on 1 GPU, multi-GPU, or multi-host with minimal code changes
- **Production Ready**: Built-in logging, checkpointing, mixed precision, and reproducible randomness
- **Composable**: Parallel plans (DP/TP/PP) can be composed and validated before compilation

## Quick Start

```python
import jax, jax.numpy as jnp
import titanax as tx

# Define mesh and parallelization strategy
mesh = tx.MeshSpec(devices="all", axes=("data",))
plan = tx.Plan(data_parallel=tx.DP(axis="data"))

# Create training engine
engine = tx.Engine(
    mesh=mesh, 
    plan=plan,
    optimizer=tx.optax.adamw(3e-4),
    precision=tx.Precision(bfloat16=True)
)

# Define training step with explicit collectives
@tx.step_fn
def train_step(state, batch):
    def loss_fn(p):
        logits = model_apply(p, batch["x"])
        return cross_entropy(logits, batch["y"])
    
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = tx.collectives.psum(grads, axis="data")  # explicit collective
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss}

# Train the model
engine.fit(train_step, data=train_data, steps=10_000)
```

## Installation

```bash
# Using uv (recommended)
uv pip install titanax

# Using pip
pip install titanax
```

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/titanax.git
cd titanax

# Install in development mode
uv pip install -e .

# Run tests
python -m pytest tests/
```

## Documentation

- [Getting Started Guide](docs/quickstart.md)
- [Core Concepts](docs/concepts.md)
- [API Reference](docs/api/)
- [Examples](examples/)

## Examples

- [MNIST Data Parallel](examples/mnist_dp/) - Single/multi-host DP training
- [GPT-Small Tensor Parallel](examples/gpt_small_tp/) - DP×TP composition  
- [GPT-Small Pipeline Parallel](examples/gpt_small_pp/) - 1F1B scheduling

## Requirements

- Python 3.9+
- JAX 0.4.20+
- Optax 0.1.7+
- Orbax-checkpoint 0.4.0+

## License

Apache 2.0 - See [LICENSE](LICENSE) file for details.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

---

**Status**: 🚧 Under Development - See [plan.md](plan.md) for implementation roadmap.
