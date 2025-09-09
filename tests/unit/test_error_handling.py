"""Tests for Engine error handling and exception management."""

import pytest
import jax
import jax.numpy as jnp
from unittest.mock import Mock

from src.titanax.exec import Engine, TrainState, step_fn
from src.titanax.runtime import MeshSpec
from src.titanax.parallel import Plan, DP
from src.titanax.exceptions import EngineError


class FailingLogger:
    """Mock logger that always fails for testing error handling."""

    def __init__(self, fail_on="both"):
        self.fail_on = fail_on  # "both", "scalar", "dict"
        self.call_count = 0

    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.call_count += 1
        if self.fail_on in ["both", "scalar"]:
            raise Exception(f"Logging failed for scalar {name}")

    def log_dict(self, metrics: dict, step: int) -> None:
        self.call_count += 1
        if self.fail_on in ["both", "dict"]:
            raise Exception("Logging failed for dict")


class TestErrorHandling:
    """Test error handling in Engine.fit method."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mesh_spec = MeshSpec(devices="all", axes=("data",))
        self.plan = Plan(data_parallel=DP(axis="data"))
        self.optimizer = Mock()  # Mock optimizer
        self.optimizer.get_learning_rate.return_value = 0.01

    def test_continue_on_error_false_reraises_step_errors(self):
        """Test that step errors are re-raised when continue_on_error=False."""
        failing_logger = FailingLogger()
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            loggers=[failing_logger],
        )

        @step_fn()
        def failing_step(state, batch):
            # This will fail
            raise ValueError("Intentional step failure")

        initial_state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        data = [{"x": jnp.ones((1, 2))} for _ in range(3)]

        # Should re-raise the step error (default behavior)
        with pytest.raises(EngineError, match="Step execution failed"):
            engine.fit(failing_step, data, state=initial_state, continue_on_error=False)

    def test_continue_on_error_true_continues_training(self):
        """Test that training continues when continue_on_error=True."""
        logger = Mock()
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            loggers=[logger],
        )

        call_count = 0

        @step_fn()
        def sometimes_failing_step(state, batch):
            nonlocal call_count
            call_count += 1

            if call_count == 2:  # Fail on second step
                raise ValueError("Intentional step failure")

            return state.replace(step=state.step + 1), {"loss": 1.0}

        initial_state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        data = [{"x": jnp.ones((1, 2))} for _ in range(3)]

        # Should complete training despite one failure
        final_state = engine.fit(
            sometimes_failing_step, data, state=initial_state, continue_on_error=True
        )

        # Should have executed steps 1 and 3 successfully
        # Step 2 failed but was skipped
        assert final_state.step == 2  # Two successful steps

        # Logger should have been called for successful steps
        assert logger.log_dict.call_count == 2

    def test_logging_error_handling_continue_by_default(self):
        """Test that logging errors are handled gracefully by default."""
        failing_logger = FailingLogger()
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            loggers=[failing_logger],
        )

        @step_fn()
        def simple_step(state, batch):
            return state.replace(step=state.step + 1), {"loss": 1.0}

        initial_state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        data = [{"x": jnp.ones((1, 2))} for _ in range(2)]

        # Should complete training despite logging failures
        final_state = engine.fit(simple_step, data, state=initial_state)

        # Training should complete successfully
        assert final_state.step == 2

        # Logger should have been called (and failed)
        assert failing_logger.call_count > 0

    def test_logging_error_reraise_with_continue_on_error_false(self):
        """Test that logging errors can be configured to re-raise."""
        # This tests the logging methods directly since they have continue_on_error parameter
        failing_logger = FailingLogger()
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            loggers=[failing_logger],
        )

        # Test _log_scalar with continue_on_error=False
        with pytest.raises(EngineError, match="Logging failed"):
            engine._log_scalar("test", 1.0, 0, continue_on_error=False)

        # Test _log_metrics with continue_on_error=False
        with pytest.raises(EngineError, match="Logging failed"):
            engine._log_metrics({"loss": 1.0}, 0, continue_on_error=False)

    def test_step_error_logged_before_reraise(self):
        """Test that step errors are logged before being re-raised."""
        logger = Mock()
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            loggers=[logger],
        )

        @step_fn()
        def failing_step(state, batch):
            raise ValueError("Intentional step failure")

        initial_state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        data = [{"x": jnp.ones((1, 2))}]

        with pytest.raises(EngineError, match="Step execution failed"):
            engine.fit(failing_step, data, state=initial_state)

        # Check that error was logged via _log_scalar
        logger.log_scalar.assert_called_with("training/step_error", 0.0, 0)

    def test_keyboard_interrupt_handling(self):
        """Test that KeyboardInterrupt is properly handled with checkpointing."""
        logger = Mock()
        checkpoint = Mock()
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            checkpoint=checkpoint,
            loggers=[logger],
        )

        @step_fn()
        def interrupting_step(state, batch):
            # Simulate KeyboardInterrupt
            raise KeyboardInterrupt("User interrupted")

        initial_state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=5,
            rngs={"dropout": jax.random.PRNGKey(0)},
        )

        data = [{"x": jnp.ones((1, 2))}]

        with pytest.raises(KeyboardInterrupt):
            engine.fit(interrupting_step, data, state=initial_state)

        # Check that interrupt was logged (check if the call was made at all)
        interrupt_calls = [
            call
            for call in logger.log_scalar.call_args_list
            if call[0][0] == "training/interrupted"
        ]
        assert (
            len(interrupt_calls) > 0
        ), f"Expected interrupted log, got calls: {logger.log_scalar.call_args_list}"

        # Check that checkpoint was saved
        checkpoint.save.assert_called_with(initial_state, 5)


if __name__ == "__main__":
    pytest.main([__file__])
