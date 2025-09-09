#!/usr/bin/env python3
"""Test MNIST training with synthetic data."""

import sys
import os
import argparse

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import jax
import jax.numpy as jnp

import src.titanax as tx

# Import local modules
from train import create_train_step, create_eval_step, evaluate_model
from model import create_model


class SyntheticMNISTLoader:
    """Create synthetic MNIST-like data for testing."""
    
    def __init__(self, batch_size: int, num_batches: int, seed: int = 42):
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.seed = seed
        self.current_batch = 0
        
    def __len__(self):
        return self.num_batches
        
    def __iter__(self):
        self.current_batch = 0
        return self
        
    def __next__(self):
        if self.current_batch >= self.num_batches:
            raise StopIteration
            
        # Generate structured synthetic data that's learnable
        rng = jax.random.PRNGKey(self.seed + self.current_batch)
        rng1, rng2 = jax.random.split(rng)
        
        # Create patterns that correlate with labels
        labels = jax.random.randint(rng1, (self.batch_size,), 0, 10)
        
        # Generate images with some structure based on labels
        images = jax.random.normal(rng2, (self.batch_size, 28, 28, 1)) * 0.1 + 0.5
        
        # Add label-dependent patterns to make it learnable
        for i in range(self.batch_size):
            label = labels[i]
            # Add some structured noise based on the label
            pattern = jnp.sin(jnp.arange(28) * jnp.pi * label / 10.0).reshape(1, -1, 1)
            images = images.at[i, :, :, :].add(pattern * 0.2)
        
        self.current_batch += 1
        return {"x": images, "y": labels}


def main():
    parser = argparse.ArgumentParser(description="Test MNIST with synthetic data")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "cnn"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--eval-every", type=int, default=25)
    
    args = parser.parse_args()
    
    print(f"Testing MNIST-{args.model.upper()} with synthetic data")
    print(f"JAX devices: {jax.devices()}")
    
    # Simple single-device configuration
    mesh_spec = tx.MeshSpec(devices=None, axes=("data",))
    plan = tx.Plan(data_parallel=tx.DP(axis="data", accumulate_steps=1))
    
    # Create model
    model_fn, init_params_fn = create_model(args.model)
    rng = jax.random.PRNGKey(42)
    init_params = init_params_fn(rng)
    
    print(f"Model parameters: {jax.tree.map(jnp.shape, init_params)}")
    
    # Setup engine
    optimizer = tx.optim.adamw(learning_rate=args.learning_rate)
    precision = tx.Precision()
    loggers = [tx.loggers.Basic()]
    
    engine = tx.Engine(
        mesh=mesh_spec,
        plan=plan,
        optimizer=optimizer,
        precision=precision,
        loggers=loggers
    )
    
    # Get mesh info
    mesh = mesh_spec.build()
    dp_size = mesh.shape["data"]
    print(f"Data parallel size: {dp_size}")
    
    # Create synthetic data
    train_loader = SyntheticMNISTLoader(batch_size=args.batch_size, num_batches=200, seed=42)
    test_loader = SyntheticMNISTLoader(batch_size=args.batch_size, num_batches=20, seed=123)
    
    # Create step functions
    train_step = create_train_step(model_fn, dp_axis="data", dp_size=dp_size)
    eval_step = create_eval_step(model_fn, dp_axis="data", dp_size=dp_size)
    
    # Initialize state
    state = engine.create_state(init_params, rngs={"dropout": rng})
    
    print("Starting training...")
    
    # Training loop
    step = 0
    for epoch in range((args.steps + len(train_loader) - 1) // len(train_loader)):
        for batch in train_loader:
            step += 1
            if step > args.steps:
                break
                
            # Training step
            state, train_metrics = train_step(state, batch)
            
            # Log training metrics
            for logger in engine.loggers:
                logger.log_dict(train_metrics, step)
                
            # Evaluation
            if step % args.eval_every == 0 or step == args.steps:
                eval_metrics = evaluate_model(eval_step, state, test_loader, max_eval_steps=10)
                
                print(f"Step {step:3d}: "
                      f"train_loss={float(train_metrics['loss']):.4f}, "
                      f"train_acc={float(train_metrics['accuracy']):.4f}, "
                      f"eval_loss={eval_metrics['eval_loss']:.4f}, "
                      f"eval_acc={eval_metrics['eval_accuracy']:.4f}")
        
        if step >= args.steps:
            break
    
    # Final evaluation
    final_eval_metrics = evaluate_model(eval_step, state, test_loader)
    print(f"\nFinal evaluation: "
          f"loss={final_eval_metrics['eval_loss']:.4f}, "
          f"accuracy={final_eval_metrics['eval_accuracy']:.4f}")
    
    print("Training completed successfully!")
    return state, final_eval_metrics


if __name__ == "__main__":
    main()
