"""Unit tests for Titanax execution engine components."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import Mock, patch

from src.titanax.exec import Engine, Precision, TrainState, step_fn
from src.titanax.runtime import MeshSpec
from src.titanax.parallel import Plan, DP
from src.titanax.types import Logger, CheckpointStrategy
from src.titanax.exceptions import EngineError


class TestPrecision:
    """Test Precision dataclass configuration."""
    
    def test_default_precision(self):
        """Test default precision configuration."""
        precision = Precision()
        assert not precision.bfloat16
        assert not precision.fp16
        assert not precision.loss_scaling
        assert not precision.enable_x32_params
        assert precision.dtype == jnp.float32
        assert precision.param_dtype == jnp.float32
    
    def test_bfloat16_precision(self):
        """Test bfloat16 precision configuration."""
        precision = Precision(bfloat16=True)
        assert precision.dtype == jnp.bfloat16
        assert precision.param_dtype == jnp.bfloat16
        assert "bfloat16" in precision.describe()
    
    def test_fp16_precision(self):
        """Test float16 precision configuration.""" 
        precision = Precision(fp16=True)
        assert precision.dtype == jnp.float16
        assert precision.param_dtype == jnp.float16
        assert "float16" in precision.describe()
    
    def test_x32_params(self):
        """Test keeping parameters in float32."""
        precision = Precision(bfloat16=True, enable_x32_params=True)
        assert precision.dtype == jnp.bfloat16
        assert precision.param_dtype == jnp.float32
        assert "x32_params" in precision.describe()
    
    def test_loss_scaling(self):
        """Test loss scaling configuration."""
        precision = Precision(fp16=True, loss_scaling=True)
        assert precision.loss_scaling
        assert "loss_scaling" in precision.describe()
    
    def test_invalid_both_precisions(self):
        """Test that both bfloat16 and fp16 cannot be enabled."""
        with pytest.raises(EngineError, match="Cannot enable both"):
            Precision(bfloat16=True, fp16=True)
    
    def test_invalid_loss_scaling_without_fp16(self):
        """Test that loss scaling requires fp16."""
        with pytest.raises(EngineError, match="Loss scaling requires fp16"):
            Precision(bfloat16=True, loss_scaling=True)


class TestTrainState:
    """Test TrainState dataclass."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.params = {"weight": jnp.ones((4, 4)), "bias": jnp.zeros(4)}
        self.opt_state = {"momentum": jnp.zeros((4, 4))}
        self.rngs = {"dropout": jax.random.PRNGKey(0)}
    
    def test_create_train_state(self):
        """Test creating TrainState."""
        state = TrainState(
            params=self.params,
            opt_state=self.opt_state,
            step=0,
            rngs=self.rngs
        )
        assert state.step == 0
        assert state.params == self.params
        assert state.opt_state == self.opt_state
        assert state.rngs == self.rngs
    
    def test_replace_method(self):
        """Test replace method."""
        state = TrainState(
            params=self.params,
            opt_state=self.opt_state,
            step=0,
            rngs=self.rngs
        )
        
        new_state = state.replace(step=5)
        assert new_state.step == 5
        assert new_state.params == self.params  # unchanged
        assert state.step == 0  # original unchanged
    
    def test_apply_gradients_placeholder(self):
        """Test apply_gradients placeholder method."""
        state = TrainState(
            params=self.params,
            opt_state=self.opt_state,
            step=10,
            rngs=self.rngs
        )
        
        grads = {"weight": jnp.ones((4, 4)), "bias": jnp.ones(4)}
        new_state = state.apply_gradients(grads=grads)
        
        # Should increment step
        assert new_state.step == 11
        # Other fields should be preserved
        assert new_state.params == self.params
        assert new_state.opt_state == self.opt_state


class TestStepFnDecorator:
    """Test @step_fn decorator functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={"momentum": jnp.zeros((2, 2))},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)}
        )
        self.batch = {"x": jnp.ones((4, 2)), "y": jnp.array([0, 1, 0, 1])}
    
    def test_step_fn_decoration(self):
        """Test basic step function decoration."""
        @step_fn()
        def simple_step(state, batch):
            return state.replace(step=state.step + 1), {"loss": 1.5}
        
        # Check it's marked as step function
        assert hasattr(simple_step, '_is_step_fn')
        assert simple_step._is_step_fn
        
        # Test execution
        new_state, metrics = simple_step(self.state, self.batch)
        assert new_state.step == 1
        assert metrics["loss"] == 1.5
    
    def test_step_fn_with_jax_array_metrics(self):
        """Test step function with JAX array metrics."""
        @step_fn()
        def array_metrics_step(state, batch):
            loss = jnp.array(2.5)
            return state, {"loss": loss, "accuracy": jnp.array(0.8)}
        
        new_state, metrics = array_metrics_step(self.state, self.batch)
        assert abs(metrics["loss"] - 2.5) < 1e-6
        assert abs(metrics["accuracy"] - 0.8) < 1e-6
    
    def test_invalid_batch_type(self):
        """Test error handling for invalid batch type."""
        @step_fn()
        def step(state, batch):
            return state, {}
        
        with pytest.raises(EngineError, match="Batch must be a dictionary"):
            step(self.state, [1, 2, 3])  # List instead of dict
    
    def test_invalid_state_type(self):
        """Test error handling for invalid state type."""
        @step_fn()
        def step(state, batch):
            return state, {}
        
        invalid_state = {"not": "trainstate"}
        with pytest.raises(EngineError, match="State must be a TrainState"):
            step(invalid_state, self.batch)
    
    def test_invalid_metrics_type(self):
        """Test error handling for invalid metrics return type."""
        @step_fn()
        def step(state, batch):
            return state, "not_a_dict"
        
        with pytest.raises(EngineError, match="Step function execution failed"):
            step(self.state, self.batch)
    
    def test_non_scalar_metrics_filtering(self):
        """Test filtering of non-scalar metrics."""
        @step_fn()
        def step(state, batch):
            return state, {
                "loss": 1.5,  # Valid scalar
                "array_metric": jnp.ones((3, 3)),  # Invalid - will be filtered
                "valid_array": jnp.array(2.0),  # Valid - single element
            }
        
        new_state, metrics = step(self.state, self.batch)
        assert "loss" in metrics
        assert "valid_array" in metrics
        assert "array_metric" not in metrics  # Should be filtered


class MockLogger:
    """Mock logger for testing."""
    
    def __init__(self):
        self.logged_scalars = []
        self.logged_dicts = []
    
    def log_scalar(self, name: str, value: float, step: int) -> None:
        self.logged_scalars.append((name, value, step))
    
    def log_dict(self, metrics: dict, step: int) -> None:
        self.logged_dicts.append((dict(metrics), step))


class MockCheckpoint:
    """Mock checkpoint strategy for testing."""
    
    def __init__(self):
        self.saved_states = {}
        self.load_should_fail = False
    
    def save(self, state, step: int) -> None:
        self.saved_states[step] = state
    
    def load(self, step=None):
        if self.load_should_fail:
            raise Exception("Checkpoint load failed")
        if not self.saved_states:
            raise Exception("No checkpoint available")
        # Return the latest checkpoint
        latest_step = max(self.saved_states.keys())
        return self.saved_states[latest_step]
    
    def restore(self, state, step=None):
        return self.load(step)


class TestEngine:
    """Test Engine class functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mesh_spec = MeshSpec(devices="all", axes=("data",))
        self.plan = Plan(data_parallel=DP(axis="data"))
        self.optimizer = Mock()  # Mock optimizer
        self.precision = Precision()
        
        # Mock logger and checkpoint
        self.logger = MockLogger()
        self.checkpoint = MockCheckpoint()
    
    def test_engine_initialization(self):
        """Test basic engine initialization."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            precision=self.precision,
            loggers=[self.logger]
        )
        
        assert engine.mesh_spec == self.mesh_spec
        assert engine.plan == self.plan
        assert engine.optimizer == self.optimizer
        assert engine.precision == self.precision
        assert len(engine.loggers) == 1
    
    def test_engine_with_checkpoint(self):
        """Test engine with checkpoint strategy."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            checkpoint=self.checkpoint
        )
        
        assert engine.checkpoint == self.checkpoint
    
    def test_create_state(self):
        """Test creating initial training state."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer
        )
        
        params = {"weight": jnp.ones((2, 2))}
        state = engine.create_state(params)
        
        assert state.params == params
        assert state.step == 0
        assert "dropout" in state.rngs
    
    def test_create_state_with_custom_rngs(self):
        """Test creating state with custom RNG keys."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer
        )
        
        params = {"weight": jnp.ones((2, 2))}
        custom_rngs = {"custom": jax.random.PRNGKey(123)}
        state = engine.create_state(params, rngs=custom_rngs)
        
        assert state.rngs == custom_rngs
    
    def test_describe_method(self):
        """Test engine description method."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            precision=self.precision,
            checkpoint=self.checkpoint,
            loggers=[self.logger]
        )
        
        description = engine.describe()
        assert "Titanax Engine Configuration" in description
        assert "Mesh:" in description
        assert "Plan:" in description
        assert "Precision:" in description
        assert "Checkpoint: enabled" in description
        assert "Loggers: 1 configured" in description
    
    @patch('src.titanax.exec.engine.set_current_mesh')
    def test_mesh_context_set(self, mock_set_mesh):
        """Test that mesh is set in collectives context."""
        Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer
        )
        
        # Verify set_current_mesh was called
        mock_set_mesh.assert_called_once()
    
    def test_invalid_mesh_handling(self):
        """Test handling of invalid mesh specifications."""
        invalid_mesh = MeshSpec(devices=[], axes=("nonexistent",))
        
        with pytest.raises(EngineError, match="Failed to build mesh"):
            Engine(
                mesh=invalid_mesh,
                plan=self.plan,
                optimizer=self.optimizer
            )
    
    def test_plan_validation_failure(self):
        """Test handling of plan validation failures."""
        # Create a plan that doesn't match the mesh
        invalid_plan = Plan(data_parallel=DP(axis="nonexistent"))
        
        with pytest.raises(EngineError, match="Plan validation failed"):
            Engine(
                mesh=self.mesh_spec,
                plan=invalid_plan,
                optimizer=self.optimizer
            )
    
    def test_fit_without_state_or_checkpoint(self):
        """Test fit method error handling when no state provided."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer
        )
        
        @step_fn()
        def step(state, batch):
            return state, {}
        
        data = [{"x": jnp.ones((1, 2))}]
        
        with pytest.raises(EngineError, match="No initial state provided"):
            engine.fit(step, data)
    
    def test_fit_with_checkpoint_load_failure(self):
        """Test fit method when checkpoint load fails."""
        self.checkpoint.load_should_fail = True
        
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            checkpoint=self.checkpoint
        )
        
        @step_fn()
        def step(state, batch):
            return state, {}
        
        data = [{"x": jnp.ones((1, 2))}]
        
        with pytest.raises(EngineError, match="No initial state provided"):
            engine.fit(step, data)
    
    def test_fit_with_provided_state(self):
        """Test successful fit execution with provided state."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            loggers=[self.logger]
        )
        
        @step_fn()
        def step(state, batch):
            return state.replace(step=state.step + 1), {"loss": 1.0}
        
        initial_state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)}
        )
        
        data = [{"x": jnp.ones((1, 2))} for _ in range(3)]
        
        final_state = engine.fit(step, data, steps=2, state=initial_state)
        
        # Should have run 2 steps
        assert final_state.step == 2
        
        # Check that metrics were logged
        assert len(self.logger.logged_dicts) == 2
        assert all(metrics["loss"] == 1.0 for metrics, step in self.logger.logged_dicts)
    
    def test_fit_with_checkpoint_saving(self):
        """Test checkpoint saving during training."""
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            checkpoint=self.checkpoint
        )
        
        @step_fn()
        def step(state, batch):
            return state.replace(step=state.step + 1000), {"loss": 1.0}  # Jump by 1000 to trigger save
        
        initial_state = TrainState(
            params={"weight": jnp.ones((2, 2))},
            opt_state={},
            step=0,
            rngs={"dropout": jax.random.PRNGKey(0)}
        )
        
        data = [{"x": jnp.ones((1, 2))} for _ in range(2)]
        
        final_state = engine.fit(step, data, state=initial_state)
        
        # Check that checkpoints were saved (at step 1000 and final)
        assert len(self.checkpoint.saved_states) >= 1


if __name__ == "__main__":
    pytest.main([__file__])
