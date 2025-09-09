#!/usr/bin/env python3
"""Minimal test of MNIST DP training without downloading data."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp

import src.titanax as tx

# Import local modules
from model import create_model, cross_entropy_loss, accuracy


def create_fake_data_loader(batch_size: int, num_batches: int = 10):
    """Create a simple fake data loader for testing."""

    class FakeDataLoader:
        def __init__(self, batch_size, num_batches):
            self.batch_size = batch_size
            self.num_batches = num_batches
            self.current_batch = 0

        def __len__(self):
            return self.num_batches

        def __iter__(self):
            self.current_batch = 0
            return self

        def __next__(self):
            if self.current_batch >= self.num_batches:
                raise StopIteration

            # Generate random batch with different patterns per batch
            rng = jax.random.PRNGKey(self.current_batch)
            x = jax.random.normal(rng, (self.batch_size, 28, 28, 1)) * 0.1 + 0.5
            y = jax.random.randint(rng, (self.batch_size,), 0, 10)

            self.current_batch += 1
            return {"x": x, "y": y}

    return FakeDataLoader(batch_size, num_batches)


def main():
    print("Minimal MNIST-MLP test with Titanax")
    print(f"JAX devices: {jax.devices()}")

    # Simple single-device configuration
    mesh_spec = tx.MeshSpec(devices=None, axes=("data",))
    plan = tx.Plan(data_parallel=tx.DP(axis="data", accumulate_steps=1))

    # Create model and optimizer
    model_fn, init_params_fn = create_model("mlp")
    rng = jax.random.PRNGKey(42)
    init_params = init_params_fn(rng)

    optimizer = tx.optim.adamw(learning_rate=1e-3)
    precision = tx.Precision()
    loggers = [tx.loggers.Basic()]

    # Create engine
    engine = tx.Engine(
        mesh=mesh_spec,
        plan=plan,
        optimizer=optimizer,
        precision=precision,
        loggers=loggers,
    )

    # Create fake data
    train_loader = create_fake_data_loader(batch_size=8, num_batches=20)
    test_loader = create_fake_data_loader(batch_size=8, num_batches=5)

    # Get mesh info
    mesh = mesh_spec.build()
    dp_size = mesh.shape["data"]

    # Create training step
    @tx.step_fn
    def train_step(state: tx.TrainState, batch) -> tuple:
        def loss_fn(params):
            logits = model_fn(params, batch["x"])
            loss = cross_entropy_loss(logits, batch["y"])
            return loss, logits

        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # Only use collectives if we have multiple devices
        if dp_size > 1:
            grads = tx.collectives.psum(grads, axis="data")

        state = state.apply_gradients(grads=grads)

        acc = accuracy(logits, batch["y"])

        # Only aggregate metrics if multiple devices
        if dp_size > 1:
            loss = tx.collectives.pmean(loss, axis="data")
            acc = tx.collectives.pmean(acc, axis="data")

        return state, {"loss": loss, "accuracy": acc}

    # Create eval step
    @tx.step_fn
    def eval_step(state: tx.TrainState, batch):
        logits = model_fn(state.params, batch["x"])
        loss = cross_entropy_loss(logits, batch["y"])
        acc = accuracy(logits, batch["y"])

        # Only aggregate metrics if multiple devices
        if dp_size > 1:
            loss = tx.collectives.pmean(loss, axis="data")
            acc = tx.collectives.pmean(acc, axis="data")

        return state, {"eval_loss": loss, "eval_accuracy": acc}

    # Initialize state
    state = engine.create_state(init_params, rngs={"dropout": rng})

    print("Starting training...")

    # Training loop
    for step in range(1, 21):  # 20 steps
        # Get training batch
        batch = next(iter(train_loader))

        # Training step
        state, train_metrics = train_step(state, batch)

        # Log every 5 steps
        if step % 5 == 0 or step == 1:
            # Run evaluation
            eval_metrics_list = []
            for eval_batch in test_loader:
                _, metrics = eval_step(state, eval_batch)
                eval_metrics_list.append(metrics)

            # Average evaluation metrics
            eval_loss = jnp.mean(jnp.array([m["eval_loss"] for m in eval_metrics_list]))
            eval_acc = jnp.mean(
                jnp.array([m["eval_accuracy"] for m in eval_metrics_list])
            )

            print(
                f"Step {step:2d}: "
                f"train_loss={float(train_metrics['loss']):.4f}, "
                f"train_acc={float(train_metrics['accuracy']):.4f}, "
                f"eval_loss={float(eval_loss):.4f}, "
                f"eval_acc={float(eval_acc):.4f}"
            )

    print("Training completed successfully!")
    return state


if __name__ == "__main__":
    main()
