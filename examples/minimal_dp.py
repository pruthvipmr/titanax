#!/usr/bin/env python3
"""Minimal Titanax Data Parallel example - runs on CPU in <30s."""
import jax
import jax.numpy as jnp
import titanax as tx


# Simple linear model: y = Wx + b
def model_apply(params, x):
    return x @ params["w"] + params["b"]


def loss_fn(params, batch):
    pred = model_apply(params, batch["x"])
    return jnp.mean((pred - batch["y"]) ** 2)


# Initialize model parameters
key = jax.random.PRNGKey(42)
params = {"w": jax.random.normal(key, (2, 1)) * 0.1, "b": jnp.zeros((1,))}

# Create synthetic data
batch = {"x": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "y": jnp.array([[1.5], [3.5]])}

# Set up Titanax engine
mesh = tx.MeshSpec(devices=[jax.devices("cpu")[0]], axes=("data",))
plan = tx.Plan(data_parallel=tx.DP(axis="data"))
engine = tx.Engine(
    mesh=mesh, plan=plan, optimizer=tx.optim.sgd(0.01), loggers=[tx.loggers.Basic()]
)


# Define training step
@tx.step_fn
def train_step(state, batch):
    loss, grads = jax.value_and_grad(loss_fn)(state.params, batch)
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss}


# Train for 3 steps
print("=== Minimal Titanax DP Example ===")
state = engine.create_state(params)
for step in range(3):
    state, metrics = train_step(state, batch)
    print(f"Step {step + 1}: loss={metrics['loss']:.6f}")

print("Example completed successfully!")
