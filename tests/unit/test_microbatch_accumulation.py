"""Tests for microbatch gradient accumulation functionality."""

import pytest
import jax
import jax.numpy as jnp
from unittest.mock import Mock

from src.titanax.exec import step_fn
from src.titanax.exec.step_fn import (
    gradient_accumulation_step,
    create_gradient_accumulation_step_fn,
)
from src.titanax.exec import Engine, TrainState
from src.titanax.runtime import MeshSpec
from src.titanax.parallel import Plan, DP
from src.titanax.exceptions import EngineError


class TestGradientAccumulation:
    """Test gradient accumulation with JAX lax.scan."""

    def setup_method(self):
        """Set up test fixtures."""
        # Simple linear model parameters
        self.params = {
            "weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "bias": jnp.array([0.1, 0.2]),
        }

        # Mock optimizer state
        self.opt_state = {"momentum": jnp.zeros_like(self.params["weight"])}

        # Create training state
        self.state = TrainState(
            params=self.params,
            opt_state=self.opt_state,
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        # Sample microbatches
        self.microbatches = [
            {"x": jnp.array([[1.0, 0.0]]), "y": jnp.array([1.0])},
            {"x": jnp.array([[0.0, 1.0]]), "y": jnp.array([0.0])},
            {"x": jnp.array([[1.0, 1.0]]), "y": jnp.array([1.0])},
            {"x": jnp.array([[0.5, 0.5]]), "y": jnp.array([0.5])},
        ]

    def simple_loss_fn(self, params, batch):
        """Simple MSE loss function for testing."""
        x, y = batch["x"], batch["y"]
        pred = (
            jnp.dot(x, params["weight"]) + params["bias"][0]
        )  # Use first bias element
        return jnp.mean((pred - y) ** 2)

    def simple_grad_fn(self, params, batch):
        """Compute gradients for a single microbatch."""
        loss = self.simple_loss_fn(params, batch)
        grads = jax.grad(self.simple_loss_fn)(params, batch)
        return loss, grads

    def simple_apply_fn(self, state, grads):
        """Apply gradients (simplified - just subtract)."""
        new_params = jax.tree_util.tree_map(
            lambda p, g: p - 0.01 * g, state.params, grads
        )
        return state.replace(params=new_params, step=state.step + 1)

    def test_single_step_no_accumulation(self):
        """Test gradient accumulation with accumulate_steps=1."""
        result_state, metrics = gradient_accumulation_step(
            self.simple_grad_fn,
            self.simple_apply_fn,
            self.state,
            self.microbatches,
            accumulate_steps=1,
        )

        # Should have executed exactly one step
        assert result_state.step == 1
        assert "loss" in metrics
        assert isinstance(metrics["loss"], (float, jax.Array))

    def test_multiple_steps_accumulation(self):
        """Test gradient accumulation with multiple microbatches."""
        accumulate_steps = 4

        result_state, metrics = gradient_accumulation_step(
            self.simple_grad_fn,
            self.simple_apply_fn,
            self.state,
            self.microbatches,
            accumulate_steps=accumulate_steps,
        )

        # Should have executed one accumulated step
        assert result_state.step == 1
        assert "loss" in metrics
        assert "accumulate_steps" in metrics
        assert metrics["accumulate_steps"] == float(accumulate_steps)

        # Parameters should have changed
        assert not jnp.allclose(
            result_state.params["weight"], self.state.params["weight"]
        )
        assert not jnp.allclose(result_state.params["bias"], self.state.params["bias"])

    def test_gradient_accumulation_mathematical_correctness(self):
        """Test that accumulated gradients are mathematically equivalent to averaged gradients."""
        accumulate_steps = 3

        # Compute accumulated gradients using lax.scan
        accumulated_state, accumulated_metrics = gradient_accumulation_step(
            self.simple_grad_fn,
            self.simple_apply_fn,
            self.state,
            self.microbatches,
            accumulate_steps=accumulate_steps,
        )

        # Compute manual average gradients
        total_grads = None
        total_loss = 0.0

        for i in range(accumulate_steps):
            loss, grads = self.simple_grad_fn(self.state.params, self.microbatches[i])
            total_loss += loss

            if total_grads is None:
                total_grads = grads
            else:
                total_grads = jax.tree_util.tree_map(
                    lambda acc, new: acc + new, total_grads, grads
                )

        # Average the gradients
        avg_grads = jax.tree_util.tree_map(lambda g: g / accumulate_steps, total_grads)
        avg_loss = total_loss / accumulate_steps

        # Apply averaged gradients manually
        manual_state = self.simple_apply_fn(self.state, avg_grads)

        # Results should be very close
        assert jnp.allclose(
            accumulated_state.params["weight"], manual_state.params["weight"], atol=1e-6
        )
        assert jnp.allclose(
            accumulated_state.params["bias"], manual_state.params["bias"], atol=1e-6
        )
        assert abs(accumulated_metrics["loss"] - avg_loss) < 1e-6

    def test_insufficient_microbatches(self):
        """Test handling when there are fewer microbatches than accumulate_steps."""
        # This should still work - it will just use available batches
        short_batches = self.microbatches[:2]

        result_state, metrics = gradient_accumulation_step(
            self.simple_grad_fn,
            self.simple_apply_fn,
            self.state,
            short_batches,
            accumulate_steps=4,  # More than available
        )

        # Should still complete (using only available batches)
        assert result_state.step == 1
        assert "loss" in metrics

    def test_gradient_accumulation_with_loss_scale(self):
        """Loss scaling should be unscaled before applying gradients."""
        accumulate_steps = 2
        loss_scale = 128.0

        def scaled_grad_fn(params, batch):
            def scaled_loss(p):
                return self.simple_loss_fn(p, batch) * loss_scale

            loss, grads = jax.value_and_grad(scaled_loss)(params)
            return loss, grads

        result_state, metrics = gradient_accumulation_step(
            scaled_grad_fn,
            self.simple_apply_fn,
            self.state,
            self.microbatches,
            accumulate_steps=accumulate_steps,
            loss_scale=loss_scale,
        )

        # Manually unscale to compare
        total_grads = None
        total_loss = 0.0
        for i in range(accumulate_steps):
            loss, grads = scaled_grad_fn(self.state.params, self.microbatches[i])
            total_loss += loss / loss_scale
            grads = jax.tree_util.tree_map(lambda g: g / loss_scale, grads)
            if total_grads is None:
                total_grads = grads
            else:
                total_grads = jax.tree_util.tree_map(
                    lambda acc, new: acc + new, total_grads, grads
                )

        avg_grads = jax.tree_util.tree_map(lambda g: g / accumulate_steps, total_grads)
        manual_state = self.simple_apply_fn(self.state, avg_grads)

        assert jnp.allclose(
            result_state.params["weight"], manual_state.params["weight"], atol=1e-6
        )
        assert jnp.allclose(
            result_state.params["bias"], manual_state.params["bias"], atol=1e-6
        )
        assert metrics["loss_scale"] == pytest.approx(loss_scale)


class TestStepFnCreation:
    """Test the create_gradient_accumulation_step_fn function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mesh_spec = MeshSpec(devices="all", axes=("data",))
        self.plan = Plan(data_parallel=DP(axis="data"))
        self.optimizer = Mock()
        self.optimizer.get_learning_rate.return_value = 0.01

        self.params = {
            "weight": jnp.array([[1.0, 2.0], [3.0, 4.0]]),
            "bias": jnp.array([0.1, 0.2]),
        }

    def simple_loss_fn(self, params, batch):
        """Simple loss function."""
        x, y = batch["x"], batch["y"]
        pred = jnp.dot(x, params["weight"]) + params["bias"][0]
        return jnp.mean((pred - y) ** 2)

    def test_create_step_fn_no_accumulation(self):
        """Test creating step function without accumulation."""
        step_fn = create_gradient_accumulation_step_fn(
            self.simple_loss_fn, accumulate_steps=1
        )

        # Should be marked as a step function
        assert hasattr(step_fn, "_is_step_fn")
        assert step_fn._is_step_fn

        # Test execution
        state = TrainState(
            params=self.params,
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
            _optimizer=Mock(),  # Add mock optimizer for apply_gradients
        )

        # Mock the apply_gradients method
        def mock_apply_gradients(grads=None, **kwargs):
            new_params = jax.tree_util.tree_map(
                lambda p, g: p - 0.01 * g, state.params, grads
            )
            return state.replace(params=new_params, step=state.step + 1)

        state.apply_gradients = mock_apply_gradients

        batch = {"x": jnp.array([[1.0, 0.0]]), "y": jnp.array([1.0])}

        new_state, metrics = step_fn(state, batch)

        assert new_state.step == 1
        assert "loss" in metrics

    def test_create_step_fn_with_accumulation(self):
        """Test creating step function with accumulation."""
        accumulate_steps = 3
        step_fn = create_gradient_accumulation_step_fn(
            self.simple_loss_fn, accumulate_steps=accumulate_steps
        )

        # Should be marked as a step function
        assert hasattr(step_fn, "_is_step_fn")
        assert step_fn._is_step_fn

        # Test that it requires microbatches
        state = TrainState(
            params=self.params,
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
            _optimizer=Mock(),
        )

        # Should fail without microbatches
        batch_without_microbatches = {"x": jnp.ones((1, 2)), "y": jnp.ones(1)}

        with pytest.raises(
            EngineError,
            match="Gradient accumulation requires batch to contain 'microbatches'",
        ):
            step_fn(state, batch_without_microbatches)

        # Should fail with insufficient microbatches
        batch_insufficient = {
            "microbatches": [
                {"x": jnp.ones((1, 2)), "y": jnp.ones(1)}  # Only 1, need 3
            ]
        }

        with pytest.raises(EngineError, match="Not enough microbatches"):
            step_fn(state, batch_insufficient)

    def test_create_step_fn_with_loss_scale(self):
        """Loss scaling should be reported in metrics for accumulate_steps=1."""
        loss_scale = 32.0
        step = create_gradient_accumulation_step_fn(
            self.simple_loss_fn, accumulate_steps=1, loss_scale=loss_scale
        )

        state = TrainState(
            params=self.params,
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
            _optimizer=Mock(),
        )

        def mock_apply_gradients(grads=None, **kwargs):
            new_params = jax.tree_util.tree_map(
                lambda p, g: p - 0.01 * g, state.params, grads
            )
            return state.replace(params=new_params, step=state.step + 1)

        state.apply_gradients = mock_apply_gradients

        batch = {"x": jnp.array([[1.0, 0.0]]), "y": jnp.array([1.0])}
        _, metrics = step(state, batch)

        assert metrics["loss_scale"] == pytest.approx(loss_scale)


class TestEngineIntegration:
    """Test Engine integration with microbatch accumulation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mesh_spec = MeshSpec(devices="all", axes=("data",))

        # Create DP plan with microbatching
        self.plan_with_accumulation = Plan(
            data_parallel=DP(axis="data", accumulate_steps=2)
        )

        self.plan_no_accumulation = Plan(
            data_parallel=DP(axis="data", accumulate_steps=1)
        )

        self.optimizer = Mock()
        self.optimizer.get_learning_rate.return_value = 0.01

    def test_engine_validates_microbatch_requirements(self):
        """Test that Engine validates microbatch requirements for DP plans."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan_with_accumulation,
            optimizer=self.optimizer,
        )

        @step_fn()
        def simple_step(state, batch):
            return state.replace(step=state.step + 1), {"loss": 1.0}

        state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        # Batch without microbatches should fail
        data_without_microbatches = [{"x": jnp.ones((1, 2))}]

        with pytest.raises(EngineError, match="DP plan requires microbatching"):
            engine.fit(simple_step, data_without_microbatches, state=state, steps=1)

    def test_engine_accepts_proper_microbatch_data(self):
        """Test that Engine accepts data with proper microbatch structure."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan_with_accumulation,
            optimizer=self.optimizer,
        )

        @step_fn()
        def simple_step(state, batch):
            return state.replace(step=state.step + 1), {"loss": 1.0}

        state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        # Proper microbatch data structure
        data_with_microbatches = [
            {
                "microbatches": [
                    {"x": jnp.ones((1, 2)), "y": jnp.ones(1)},
                    {"x": jnp.ones((1, 2)), "y": jnp.ones(1)},
                ]
            }
        ]

        # Should not raise validation error (though execution might fail due to mocks)
        try:
            engine.fit(simple_step, data_with_microbatches, state=state, steps=1)
        except Exception as e:
            # We expect some error due to mocking, but not the validation error
            assert "DP plan requires microbatching" not in str(e)

    def test_engine_no_validation_for_single_accumulation(self):
        """Test that Engine doesn't validate microbatches when accumulate_steps=1."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan_no_accumulation,
            optimizer=self.optimizer,
        )

        @step_fn()
        def simple_step(state, batch):
            return state.replace(step=state.step + 1), {"loss": 1.0}

        state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        # Regular batch data should be fine
        data_regular = [{"x": jnp.ones((1, 2))}]

        # Should not raise validation error
        try:
            engine.fit(simple_step, data_regular, state=state, steps=1)
        except Exception as e:
            # We expect some error due to mocking, but not the validation error
            assert "DP plan requires microbatching" not in str(e)


if __name__ == "__main__":
    pytest.main([__file__])
