#!/usr/bin/env python3
"""MNIST Data Parallel Training Example with Titanax.

This script demonstrates how to train a simple MNIST classifier using
Titanax's data parallel capabilities.

Usage:
    # Single device
    python examples/mnist_dp/train.py

    # Multi-device (if available)
    python examples/mnist_dp/train.py --devices=all
"""

import sys
import os
import argparse
from typing import Dict

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp

import src.titanax as tx
from src.titanax.types import Array

# Import local modules
from model import create_model, cross_entropy_loss, accuracy
from data import create_data_loaders


def create_train_step(model_fn, dp_axis: str = "data", dp_size: int = 1):
    """Create training step function with data parallel gradient aggregation.

    Args:
        model_fn: Model forward function
        dp_axis: Data parallel axis name
        dp_size: Data parallel world size

    Returns:
        Decorated training step function
    """

    @tx.step_fn
    def train_step(state: tx.TrainState, batch: Dict[str, Array]) -> tuple:
        """Single training step with DP gradient synchronization.

        Args:
            state: Current training state
            batch: Batch dict with 'x' (images) and 'y' (labels)

        Returns:
            (updated_state, metrics)
        """

        def loss_fn(params):
            logits = model_fn(params, batch["x"])
            loss = cross_entropy_loss(logits, batch["y"])
            return loss, logits

        # Compute loss and gradients
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)

        # Only aggregate gradients across data parallel devices if multi-device
        if dp_size > 1:
            grads = tx.collectives.psum(grads, axis=dp_axis)

        # Apply optimizer update
        state = state.apply_gradients(grads=grads)

        # Compute metrics
        acc = accuracy(logits, batch["y"])

        # Only aggregate metrics across devices for logging if multi-device
        if dp_size > 1:
            loss = tx.collectives.pmean(loss, axis=dp_axis)
            acc = tx.collectives.pmean(acc, axis=dp_axis)

        metrics = {"loss": loss, "accuracy": acc}

        return state, metrics

    return train_step


def create_eval_step(model_fn, dp_axis: str = "data", dp_size: int = 1):
    """Create evaluation step function.

    Args:
        model_fn: Model forward function
        dp_axis: Data parallel axis name
        dp_size: Data parallel world size

    Returns:
        Evaluation step function
    """

    @tx.step_fn
    def eval_step(state: tx.TrainState, batch: Dict[str, Array]) -> tuple:
        """Single evaluation step.

        Args:
            state: Current training state
            batch: Batch dict with 'x' (images) and 'y' (labels)

        Returns:
            (state, metrics): Unchanged state and evaluation metrics dict
        """
        logits = model_fn(state.params, batch["x"])
        loss = cross_entropy_loss(logits, batch["y"])
        acc = accuracy(logits, batch["y"])

        # Only aggregate metrics across devices if multi-device
        if dp_size > 1:
            loss = tx.collectives.pmean(loss, axis=dp_axis)
            acc = tx.collectives.pmean(acc, axis=dp_axis)

        return state, {"eval_loss": loss, "eval_accuracy": acc}

    return eval_step


def evaluate_model(
    eval_step_fn, state: tx.TrainState, eval_loader, max_eval_steps: int = None
):
    """Run evaluation on the test set.

    Args:
        eval_step_fn: Evaluation step function
        state: Current training state
        eval_loader: Test data loader
        max_eval_steps: Maximum evaluation steps (None for full dataset)

    Returns:
        Dict of aggregated evaluation metrics
    """
    total_loss = 0.0
    total_acc = 0.0
    num_steps = 0

    for step, batch in enumerate(eval_loader):
        if max_eval_steps and step >= max_eval_steps:
            break

        _, metrics = eval_step_fn(state, batch)
        total_loss += float(metrics["eval_loss"])
        total_acc += float(metrics["eval_accuracy"])
        num_steps += 1

    if num_steps == 0:
        return {"eval_loss": 0.0, "eval_accuracy": 0.0}

    return {"eval_loss": total_loss / num_steps, "eval_accuracy": total_acc / num_steps}


def main():
    parser = argparse.ArgumentParser(description="MNIST Data Parallel Training")
    parser.add_argument(
        "--model",
        type=str,
        default="mlp",
        choices=["mlp", "cnn"],
        help="Model type: mlp or cnn",
    )
    parser.add_argument(
        "--devices",
        type=str,
        default="auto",
        help="Device specification: 'auto', 'all', or device count",
    )
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size per device"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-3, help="Learning rate"
    )
    parser.add_argument("--steps", type=int, default=1000, help="Training steps")
    parser.add_argument(
        "--eval-every", type=int, default=200, help="Evaluation frequency"
    )
    parser.add_argument(
        "--data-dir", type=str, default="data/mnist", help="MNIST data directory"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Checkpoint directory (optional)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "fp16"],
        help="Training precision",
    )

    args = parser.parse_args()

    print(f"Starting MNIST-{args.model.upper()} training with Titanax")
    print(f"JAX devices: {jax.devices()}")
    print(f"JAX device count: {jax.device_count()}")

    # Determine device configuration
    if args.devices == "auto":
        # Use single device for simplicity or all if multiple available
        if jax.device_count() == 1:
            mesh_spec = tx.MeshSpec(
                devices=None, axes=("data",)
            )  # None means use all available
        else:
            mesh_spec = tx.MeshSpec(devices="all", axes=("data",))
    elif args.devices == "all":
        mesh_spec = tx.MeshSpec(devices="all", axes=("data",))
    else:
        try:
            device_count = int(args.devices)
            if device_count == 1:
                devices = [jax.devices()[0]]
            else:
                devices = jax.devices()[:device_count]
            mesh_spec = tx.MeshSpec(devices=devices, axes=("data",))
        except ValueError:
            # Try to parse as literal device specification
            mesh_spec = tx.MeshSpec(
                devices=args.devices if args.devices != "auto" else None, axes=("data",)
            )

    # Create data parallel plan
    plan = tx.Plan(data_parallel=tx.DP(axis="data", accumulate_steps=1))

    # Create model
    model_fn, init_params_fn = create_model(args.model)

    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    init_params = init_params_fn(rng)

    print(f"Model parameters shape: {jax.tree.map(jnp.shape, init_params)}")

    # Setup precision
    precision_config = {
        "float32": tx.Precision(),
        "bfloat16": tx.Precision(bfloat16=True),
        "fp16": tx.Precision(fp16=True),
    }
    precision = precision_config[args.precision]

    # Setup optimizer
    optimizer = tx.optim.adamw(learning_rate=args.learning_rate)

    # Setup checkpoint (optional)
    checkpoint = None
    checkpoint_interval = None
    if args.checkpoint_dir:
        checkpoint_interval = 200
        checkpoint = tx.OrbaxCheckpoint(args.checkpoint_dir)

    # Setup logging
    loggers = [tx.loggers.Basic()]

    # Create engine
    engine = tx.Engine(
        mesh=mesh_spec,
        plan=plan,
        optimizer=optimizer,
        precision=precision,
        checkpoint=checkpoint,
        checkpoint_interval=checkpoint_interval,
        loggers=loggers,
    )

    # Get mesh info for data loading
    mesh = mesh_spec.build()
    process_groups = tx.ProcessGroups(mesh)
    dp_rank = process_groups.rank("data") if "data" in mesh.axis_names else 0
    dp_size = process_groups.size("data") if "data" in mesh.axis_names else 1

    print(f"Data parallel: rank={dp_rank}, size={dp_size}")

    # Create data loaders
    train_loader, test_loader = create_data_loaders(
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
    )

    print(f"Training batches per epoch: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Create step functions
    train_step = create_train_step(model_fn, dp_axis="data", dp_size=dp_size)
    eval_step = create_eval_step(model_fn, dp_axis="data", dp_size=dp_size)

    # Initialize training state
    state = engine.create_state(init_params, rngs={"dropout": rng})

    print("Starting training...")

    # Custom training loop with evaluation
    for step in range(1, args.steps + 1):
        # Get next training batch
        try:
            batch = next(iter(train_loader))
        except StopIteration:
            # Reset data loader if we've gone through all data
            train_loader = create_data_loaders(
                batch_size=args.batch_size,
                data_dir=args.data_dir,
                data_parallel_rank=dp_rank,
                data_parallel_size=dp_size,
            )[0]
            batch = next(iter(train_loader))

        # Training step
        state, train_metrics = train_step(state, batch)

        # Log training metrics
        for logger in engine.loggers:
            logger.log_dict(train_metrics, step)

        # Evaluation
        if step % args.eval_every == 0 or step == args.steps:
            eval_metrics = evaluate_model(
                eval_step, state, test_loader, max_eval_steps=50
            )

            print(
                f"Step {step}: "
                f"train_loss={train_metrics['loss']:.4f}, "
                f"train_acc={train_metrics['accuracy']:.4f}, "
                f"eval_loss={eval_metrics['eval_loss']:.4f}, "
                f"eval_acc={eval_metrics['eval_accuracy']:.4f}"
            )

            # Log evaluation metrics
            for logger in engine.loggers:
                logger.log_dict(eval_metrics, step)

        # Checkpoint saving
        if checkpoint and checkpoint_interval and step % checkpoint_interval == 0:
            checkpoint.save(state)

    print("Training completed!")

    # Final evaluation
    final_eval_metrics = evaluate_model(eval_step, state, test_loader)
    print(
        f"Final evaluation: "
        f"loss={final_eval_metrics['eval_loss']:.4f}, "
        f"accuracy={final_eval_metrics['eval_accuracy']:.4f}"
    )

    return state, final_eval_metrics


if __name__ == "__main__":
    main()
