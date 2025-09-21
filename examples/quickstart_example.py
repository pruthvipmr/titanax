#!/usr/bin/env python3
"""Minimal example using the Titanax quickstart API."""

import jax
import jax.numpy as jnp
import titanax as tx


def create_toy_model():
    """Create a simple toy model for testing."""

    def init(rng, input_shape):
        return {"w": jax.random.normal(rng, (input_shape[-1], 10)), "b": jnp.zeros(10)}

    def forward(params, x):
        return jnp.dot(x, params["w"]) + params["b"]

    return init, forward


def create_toy_data(batch_size: int = 32):
    """Create toy training data."""
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, 5))
    y = jax.random.randint(key, (batch_size,), 0, 10)
    return x, y


def main():
    """Demonstrate quickstart API usage."""
    print("Testing Titanax quickstart API...")

    # Test simple_data_parallel
    try:
        engine = tx.quickstart.simple_data_parallel(
            batch_size=32,
            learning_rate=1e-3,
            precision="bf16",
            checkpoint_dir="./test_checkpoints",
        )
        print("✅ simple_data_parallel() created engine successfully")

        # Validate the engine
        diagnostics = tx.quickstart.validate_setup(engine)
        print(f"✅ Engine validation passed: {diagnostics['validation_status']}")
        print(f"   - Device count: {diagnostics['device_count']}")
        print(f"   - Plan: {diagnostics['plan_info']}")

    except Exception as e:
        print(f"❌ simple_data_parallel() failed: {e}")
        return

    # Test simple_tensor_parallel (should raise NotImplementedError)
    try:
        tx.quickstart.simple_tensor_parallel(
            batch_size=32, model_parallel_size=2, sharding_rules={}, learning_rate=1e-3
        )
        print("❌ simple_tensor_parallel() should have raised NotImplementedError")
    except NotImplementedError as e:
        print("✅ simple_tensor_parallel() correctly raised NotImplementedError")
        print(f"   Message: {e}")
    except Exception as e:
        print(f"❌ Unexpected error in simple_tensor_parallel(): {e}")

    print("\nQuickstart API test completed!")


if __name__ == "__main__":
    main()
