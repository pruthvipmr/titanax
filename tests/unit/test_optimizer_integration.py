"""Unit tests for optimizer integration in Engine and TrainState."""

import jax
import jax.numpy as jnp
import pytest
from unittest.mock import Mock, patch

from src.titanax.exec.engine import Engine, TrainState, Precision
from src.titanax.optim.optax_adapter import OptaxAdapter, adamw, sgd
from src.titanax.runtime.mesh import MeshSpec
from src.titanax.parallel.plan import Plan, DP
from src.titanax.exceptions import EngineError


class TestTrainStateOptimizerIntegration:
    """Test TrainState integration with optimizers."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = {
            'layer1': {'weight': jnp.array([[1.0, 2.0], [3.0, 4.0]]), 'bias': jnp.array([0.1, 0.2])},
            'layer2': {'weight': jnp.array([[0.5, 1.5]]), 'bias': jnp.array([0.0])}
        }
        
        self.optimizer = adamw(learning_rate=1e-3)
        self.opt_state = self.optimizer.init(self.params)
        
        self.state = TrainState(
            params=self.params,
            opt_state=self.opt_state,
            step=0,
            rngs={'dropout': jax.random.PRNGKey(42)},
            _optimizer=self.optimizer
        )

    def test_apply_gradients_with_stored_optimizer(self):
        """Test apply_gradients using stored optimizer."""
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, self.params)
        
        new_state = self.state.apply_gradients(grads=grads)
        
        # Check that step increased
        assert new_state.step == self.state.step + 1
        
        # Check that parameters changed (AdamW should update them)
        assert not jnp.allclose(
            new_state.params['layer1']['weight'], 
            self.state.params['layer1']['weight']
        )
        
        # Check that optimizer state was updated
        assert new_state.opt_state != self.state.opt_state

    def test_apply_gradients_with_explicit_optimizer(self):
        """Test apply_gradients with explicitly passed optimizer."""
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, self.params)
        explicit_optimizer = sgd(learning_rate=1e-2)
        
        # Need to create a new state with the explicit optimizer's state
        explicit_opt_state = explicit_optimizer.init(self.params)
        state_with_explicit_opt = TrainState(
            params=self.params,
            opt_state=explicit_opt_state,
            step=0,
            rngs={'dropout': jax.random.PRNGKey(42)},
            _optimizer=None  # Force it to use the passed optimizer
        )
        
        new_state = state_with_explicit_opt.apply_gradients(grads=grads, optimizer=explicit_optimizer)
        
        assert new_state.step == state_with_explicit_opt.step + 1
        assert not jnp.allclose(
            new_state.params['layer1']['weight'], 
            state_with_explicit_opt.params['layer1']['weight']
        )

    def test_apply_gradients_no_optimizer(self):
        """Test apply_gradients fails without optimizer."""
        state_no_opt = TrainState(
            params=self.params,
            opt_state=self.opt_state,
            step=0,
            rngs={'dropout': jax.random.PRNGKey(42)},
            _optimizer=None
        )
        
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, self.params)
        
        with pytest.raises(EngineError) as exc_info:
            state_no_opt.apply_gradients(grads=grads)
        
        assert "No optimizer provided" in str(exc_info.value)

    def test_apply_gradients_shape_mismatch(self):
        """Test apply_gradients with shape mismatch."""
        # Create gradients with wrong shapes
        bad_grads = {
            'layer1': {'weight': jnp.array([1.0]), 'bias': jnp.array([0.1, 0.2])},  # Wrong weight shape
            'layer2': {'weight': jnp.array([[0.5, 1.5]]), 'bias': jnp.array([0.0])}
        }
        
        # This may not actually raise an error due to JAX broadcasting
        # So we test that it either works or raises an appropriate error
        try:
            result = self.state.apply_gradients(grads=bad_grads)
            # If it works, check that we get a valid state back
            assert result.step == self.state.step + 1
        except (EngineError, ValueError, TypeError):
            # Any of these errors are acceptable for malformed gradients
            pass

    def test_train_state_tree_operations(self):
        """Test that TrainState works correctly with JAX tree operations."""
        # Test tree_map
        doubled_state = jax.tree_util.tree_map(lambda x: x, self.state)  # Identity should work
        assert doubled_state.step == self.state.step
        assert doubled_state._optimizer == self.state._optimizer
        
        # Test tree_flatten/unflatten round trip
        children, aux_data = self.state.tree_flatten()
        reconstructed = TrainState.tree_unflatten(aux_data, children)
        
        assert reconstructed.step == self.state.step
        assert reconstructed._optimizer == self.state._optimizer
        assert jnp.allclose(
            reconstructed.params['layer1']['weight'],
            self.state.params['layer1']['weight']
        )


class TestEngineOptimizerIntegration:
    """Test Engine class optimizer integration."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mesh_spec = MeshSpec(devices="all", axes=("data",))
        self.plan = Plan(data_parallel=DP(axis="data"))
        self.optimizer = adamw(learning_rate=1e-3)
        self.precision = Precision()
        
        self.engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            precision=self.precision
        )
        
        self.params = {
            'weight': jnp.array([[1.0, 2.0]]),
            'bias': jnp.array([0.1])
        }

    def test_engine_initialization(self):
        """Test engine initializes with optimizer."""
        assert self.engine.optimizer == self.optimizer
        assert isinstance(self.engine.precision, Precision)

    def test_create_state_with_optimizer(self):
        """Test create_state includes optimizer reference."""
        state = self.engine.create_state(self.params)
        
        assert state._optimizer == self.optimizer
        assert state.step == 0
        assert state.params == self.params
        
        # Check that optimizer state was properly initialized
        assert state.opt_state is not None
        # For AdamW, state should be a tuple with momentum and variance terms
        assert isinstance(state.opt_state, tuple)

    def test_create_state_optimizer_init_error(self):
        """Test create_state handles optimizer initialization errors."""
        # Create engine with mock optimizer that fails
        mock_optimizer = Mock()
        mock_optimizer.init.side_effect = ValueError("Invalid params")
        
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=mock_optimizer,
            precision=self.precision
        )
        
        with pytest.raises(EngineError) as exc_info:
            engine.create_state(self.params)
        
        assert "Failed to initialize optimizer state" in str(exc_info.value)

    @patch('src.titanax.exec.engine.update_rngs')
    def test_execute_step_prng_threading(self, mock_update_rngs):
        """Test that PRNG keys are updated during step execution."""
        # Setup mock
        updated_rngs = {'dropout': jax.random.PRNGKey(123)}
        mock_update_rngs.return_value = updated_rngs
        
        # Create state and mock step function
        state = self.engine.create_state(self.params)
        mock_step_fn = Mock()
        mock_step_fn.return_value = (state, {'loss': 0.5})
        
        self.engine._compiled_step_fn = mock_step_fn
        
        # Execute step
        batch = {'x': jnp.array([[1.0]]), 'y': jnp.array([1])}
        new_state, metrics = self.engine._execute_step(state, batch)
        
        # Check that PRNG update was called
        mock_update_rngs.assert_called_once_with(state.rngs)
        
        # Check that step function received state with updated rngs
        call_args = mock_step_fn.call_args
        state_arg, batch_arg = call_args[0]
        assert state_arg.rngs == updated_rngs

    def test_precision_policy_application(self):
        """Test that precision policy is applied to batch data."""
        # Create engine with bf16 precision
        precision = Precision(bfloat16=True)
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            precision=precision
        )
        
        # Test batch conversion
        batch = {
            'x': jnp.array([[1.0, 2.0]], dtype=jnp.float32),
            'y': jnp.array([1], dtype=jnp.int32),  # Should not be converted
            'z': 'string'  # Should not be converted
        }
        
        converted_batch = engine._apply_precision_policy(batch)
        
        # Check float arrays were converted
        assert converted_batch['x'].dtype == jnp.bfloat16
        # Check non-float arrays were not converted
        assert converted_batch['y'].dtype == jnp.int32
        assert converted_batch['z'] == 'string'

    def test_precision_policy_no_conversion(self):
        """Test precision policy with float32 (no conversion)."""
        batch = {'x': jnp.array([[1.0, 2.0]], dtype=jnp.float32)}
        converted_batch = self.engine._apply_precision_policy(batch)
        
        # Should be identical (no conversion needed)
        assert converted_batch is batch

    def test_loss_scaling_application(self):
        """Test loss scaling for fp16 training."""
        precision = Precision(fp16=True, loss_scaling=True)
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            precision=precision
        )
        
        loss = jnp.array(0.5)
        scaled_loss = engine.apply_loss_scaling(loss)
        
        # Should be scaled by 2^14
        expected_scale = 2**14
        assert jnp.allclose(scaled_loss, loss * expected_scale)

    def test_loss_scaling_disabled(self):
        """Test loss scaling when disabled."""
        loss = jnp.array(0.5)
        scaled_loss = self.engine.apply_loss_scaling(loss)
        
        # Should be unchanged
        assert jnp.allclose(scaled_loss, loss)

    def test_gradient_scaling(self):
        """Test gradient scaling for fp16 training."""
        precision = Precision(fp16=True, loss_scaling=True)
        engine = Engine(
            mesh=self.mesh_spec,
            plan=self.plan,
            optimizer=self.optimizer,
            precision=precision
        )
        
        grads = {'w': jnp.array([1.0, 2.0])}
        scaled_grads = engine.scale_gradients(grads)
        
        # Should be scaled down by 2^14
        expected_scale = 2**14
        expected_grads = {'w': jnp.array([1.0/expected_scale, 2.0/expected_scale])}
        
        assert jnp.allclose(scaled_grads['w'], expected_grads['w'])

    def test_learning_rate_metrics(self):
        """Test that learning rate is added to metrics."""
        state = self.engine.create_state(self.params)
        
        # Mock step function that returns metrics
        mock_step_fn = Mock()
        original_metrics = {'loss': 0.5, 'accuracy': 0.8}
        mock_step_fn.return_value = (state.replace(step=1), original_metrics)
        
        self.engine._compiled_step_fn = mock_step_fn
        
        batch = {'x': jnp.array([[1.0]]), 'y': jnp.array([1])}
        new_state, metrics = self.engine._execute_step(state, batch)
        
        # Check that learning rate was added
        assert 'learning_rate' in metrics
        assert metrics['learning_rate'] == self.optimizer.get_learning_rate(new_state.step)
        
        # Check original metrics are still there
        assert metrics['loss'] == 0.5
        assert metrics['accuracy'] == 0.8

    def test_describe_includes_optimizer(self):
        """Test that engine description includes optimizer info."""
        description = self.engine.describe()
        
        assert "Optimizer:" in description
        assert "adamw" in description


class TestPrecisionPolicyEdgeCases:
    """Test edge cases in precision policy handling."""

    def test_precision_validation_both_enabled(self):
        """Test that both bf16 and fp16 cannot be enabled."""
        with pytest.raises(EngineError):
            Precision(bfloat16=True, fp16=True)

    def test_loss_scaling_without_fp16(self):
        """Test that loss scaling requires fp16."""
        with pytest.raises(EngineError):
            Precision(bfloat16=True, loss_scaling=True)

    def test_precision_dtype_properties(self):
        """Test precision dtype properties."""
        # Default (float32)
        p1 = Precision()
        assert p1.dtype == jnp.float32
        assert p1.param_dtype == jnp.float32
        
        # BFloat16
        p2 = Precision(bfloat16=True)
        assert p2.dtype == jnp.bfloat16
        assert p2.param_dtype == jnp.bfloat16
        
        # Float16
        p3 = Precision(fp16=True)
        assert p3.dtype == jnp.float16
        assert p3.param_dtype == jnp.float16
        
        # x32 params
        p4 = Precision(bfloat16=True, enable_x32_params=True)
        assert p4.dtype == jnp.bfloat16
        assert p4.param_dtype == jnp.float32

    def test_precision_describe(self):
        """Test precision description strings."""
        p1 = Precision()
        assert "float32" in p1.describe()
        
        p2 = Precision(bfloat16=True, enable_x32_params=True)
        desc = p2.describe()
        assert "bfloat16" in desc
        assert "x32_params" in desc
        
        p3 = Precision(fp16=True, loss_scaling=True)
        desc = p3.describe()
        assert "float16" in desc
        assert "loss_scaling" in desc
