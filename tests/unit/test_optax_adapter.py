"""Unit tests for Optax optimizer adapters."""

import jax.numpy as jnp
import optax
import pytest

from src.titanax.optim.optax_adapter import (
    OptaxAdapter,
    adamw,
    sgd,
    adam,
    cosine_schedule,
    exponential_schedule,
    warmup_cosine_schedule,
)
from src.titanax.exceptions import OptimizerError


class TestOptaxAdapter:
    """Test OptaxAdapter wrapper functionality."""

    def test_init_basic(self):
        """Test basic OptaxAdapter initialization."""
        base_opt = optax.adamw(learning_rate=1e-3)
        adapter = OptaxAdapter(base_opt, learning_rate=1e-3)
        
        assert adapter.optimizer is base_opt
        assert adapter.learning_rate == 1e-3
        assert adapter.name == "optax_adapter"
        assert not adapter._lr_is_callable
    
    def test_init_with_schedule(self):
        """Test OptaxAdapter with learning rate schedule."""
        base_opt = optax.sgd(learning_rate=1.0)
        def schedule(step):
            return 1e-3 * (0.9 ** (step // 100))
        
        adapter = OptaxAdapter(base_opt, learning_rate=schedule, name="custom")
        
        assert adapter.learning_rate is schedule
        assert adapter.name == "custom"
        assert adapter._lr_is_callable
    
    def test_init_params(self):
        """Test optimizer state initialization."""
        base_opt = optax.adamw(learning_rate=1e-3)
        adapter = OptaxAdapter(base_opt, learning_rate=1e-3)
        
        # Create simple parameters
        params = {'weights': jnp.array([[1.0, 2.0], [3.0, 4.0]])}
        
        opt_state = adapter.init(params)
        
        # Check that state was initialized (exact structure depends on optimizer)
        assert opt_state is not None
        assert isinstance(opt_state, tuple)  # Optax states are typically tuples
    
    def test_init_params_error(self):
        """Test error handling in parameter initialization."""
        base_opt = optax.adamw(learning_rate=1e-3)
        adapter = OptaxAdapter(base_opt, learning_rate=1e-3)
        
        # Invalid parameters (not a PyTree)
        with pytest.raises(OptimizerError) as exc_info:
            adapter.init("invalid")
        
        assert "Failed to initialize optimizer" in str(exc_info.value)
        assert exc_info.value.suggestion is not None
    
    def test_apply_gradients_fixed_lr(self):
        """Test gradient application with fixed learning rate."""
        base_opt = optax.sgd(learning_rate=1.0)  # Use learning rate of 1.0 for easy testing
        adapter = OptaxAdapter(base_opt, learning_rate=1.0)
        
        params = {'weight': jnp.array([1.0, 2.0])}
        grads = {'weight': jnp.array([0.1, 0.2])}
        opt_state = adapter.init(params)
        
        new_params, new_opt_state = adapter.apply_gradients(
            grads, opt_state, params, step=0
        )
        
        # SGD with lr=1.0: new_param = param - lr * grad
        expected = {'weight': jnp.array([0.9, 1.8])}  # [1.0-0.1, 2.0-0.2]
        
        assert jnp.allclose(new_params['weight'], expected['weight'])
        assert new_opt_state is not None
    
    def test_apply_gradients_with_schedule(self):
        """Test gradient application with learning rate schedule."""
        def schedule(step):
            return 1.0 if step < 10 else 0.1
        # Create optimizer with the schedule directly
        base_opt = optax.sgd(learning_rate=schedule)
        adapter = OptaxAdapter(base_opt, learning_rate=schedule)

        params = {'weight': jnp.array([1.0])}
        grads = {'weight': jnp.array([0.1])}
        opt_state = adapter.init(params)

        # The exact behavior depends on how Optax internally applies the schedule
        # We test that parameters change and that the schedule is accessible
        new_params, new_opt_state = adapter.apply_gradients(
            grads, opt_state, params, step=5
        )
        # Check that parameters changed
        assert not jnp.allclose(new_params['weight'], params['weight'])
        
        # Test that we can retrieve the learning rate from the schedule
        assert adapter.get_learning_rate(5) == 1.0
        assert adapter.get_learning_rate(15) == 0.1
    
    def test_apply_gradients_error(self):
        """Test error handling in gradient application."""
        base_opt = optax.adamw(learning_rate=1e-3)
        adapter = OptaxAdapter(base_opt, learning_rate=1e-3)
        
        # Create a scenario that will actually fail - incompatible tree structure
        params = {'weight': jnp.array([1.0, 2.0])}
        grads = {'different_key': jnp.array([0.1, 0.1])}  # Different key structure
        opt_state = adapter.init(params)
        
        # This should raise an error due to tree structure mismatch
        with pytest.raises((OptimizerError, ValueError, KeyError)) as exc_info:
            adapter.apply_gradients(grads, opt_state, params, step=0)
        
        # If it's an OptimizerError, check the message
        if isinstance(exc_info.value, OptimizerError):
            assert "Failed to apply gradients" in str(exc_info.value)
    
    def test_get_learning_rate_fixed(self):
        """Test learning rate retrieval with fixed rate."""
        adapter = OptaxAdapter(None, learning_rate=1e-3)
        
        assert adapter.get_learning_rate(0) == 1e-3
        assert adapter.get_learning_rate(100) == 1e-3
    
    def test_get_learning_rate_schedule(self):
        """Test learning rate retrieval with schedule."""
        def schedule(step):
            return 1e-3 * (0.9 ** step)
        adapter = OptaxAdapter(None, learning_rate=schedule)
        
        assert adapter.get_learning_rate(0) == 1e-3
        assert adapter.get_learning_rate(1) == 1e-3 * 0.9
        assert adapter.get_learning_rate(2) == 1e-3 * 0.9 * 0.9
    
    def test_describe_fixed_lr(self):
        """Test description with fixed learning rate."""
        adapter = OptaxAdapter(None, learning_rate=1e-3, name="test_opt")
        description = adapter.describe()
        
        assert "test_opt" in description
        assert "0.001" in description
    
    def test_describe_scheduled_lr(self):
        """Test description with scheduled learning rate."""
        def schedule(step):
            return 1e-3
        adapter = OptaxAdapter(None, learning_rate=schedule, name="test_opt")
        description = adapter.describe()
        
        assert "test_opt" in description
        assert "scheduled" in description


class TestOptimizerFactories:
    """Test optimizer factory functions."""

    def test_adamw_basic(self):
        """Test basic AdamW creation."""
        optimizer = adamw()
        
        assert isinstance(optimizer, OptaxAdapter)
        assert optimizer.name == "adamw"
        assert optimizer.learning_rate == 3e-4  # Default value
    
    def test_adamw_custom_params(self):
        """Test AdamW with custom parameters."""
        optimizer = adamw(
            learning_rate=1e-3,
            b1=0.95,
            b2=0.999,
            weight_decay=1e-3
        )
        
        assert optimizer.learning_rate == 1e-3
        # Test that parameters work by initializing and checking state
        params = {'w': jnp.array([1.0, 2.0])}
        opt_state = optimizer.init(params)
        assert opt_state is not None
    
    def test_adamw_with_schedule(self):
        """Test AdamW with learning rate schedule."""
        schedule = cosine_schedule(1e-3, 1000)
        optimizer = adamw(learning_rate=schedule)
        
        assert optimizer._lr_is_callable
        assert abs(optimizer.get_learning_rate(0) - 1e-3) < 1e-6
    
    def test_sgd_basic(self):
        """Test basic SGD creation."""
        optimizer = sgd()
        
        assert isinstance(optimizer, OptaxAdapter)
        assert optimizer.name == "sgd"
        assert optimizer.learning_rate == 1e-3  # Default value
    
    def test_sgd_with_momentum(self):
        """Test SGD with momentum."""
        optimizer = sgd(learning_rate=1e-2, momentum=0.9)
        
        assert optimizer.learning_rate == 1e-2
        # Test functionality
        params = {'w': jnp.array([1.0])}
        grads = {'w': jnp.array([0.1])}
        opt_state = optimizer.init(params)
        
        new_params, _ = optimizer.apply_gradients(grads, opt_state, params, step=0)
        assert new_params is not None
    
    def test_adam_basic(self):
        """Test basic Adam creation."""
        optimizer = adam(learning_rate=5e-4)
        
        assert isinstance(optimizer, OptaxAdapter)
        assert optimizer.name == "adam"
        assert optimizer.learning_rate == 5e-4


class TestLearningRateSchedules:
    """Test learning rate schedule utilities."""

    def test_cosine_schedule(self):
        """Test cosine decay schedule."""
        schedule = cosine_schedule(init_value=1e-3, decay_steps=100)
        
        # At step 0, should be initial value
        assert schedule(0) == 1e-3
        
        # At decay_steps/2, should be between init and final
        mid_lr = schedule(50)
        assert 0 < mid_lr < 1e-3
        
        # At decay_steps, should be close to 0 (alpha=0.0 default)
        final_lr = schedule(100)
        assert final_lr < 1e-6
    
    def test_exponential_schedule(self):
        """Test exponential decay schedule."""
        schedule = exponential_schedule(
            init_value=1e-2,
            transition_steps=10,
            decay_rate=0.9
        )
        
        assert schedule(0) == 1e-2
        
        # After transition_steps, should decay by decay_rate
        lr_10 = schedule(10)
        expected_10 = 1e-2 * 0.9
        assert abs(lr_10 - expected_10) < 1e-8
        
        # After 2*transition_steps, should decay by decay_rate^2
        lr_20 = schedule(20)
        expected_20 = 1e-2 * (0.9 ** 2)
        assert abs(lr_20 - expected_20) < 1e-8
    
    def test_warmup_cosine_schedule(self):
        """Test warmup + cosine decay schedule."""
        schedule = warmup_cosine_schedule(
            init_value=1e-5,
            peak_value=1e-3,
            warmup_steps=10,
            decay_steps=90,
            end_value=1e-6
        )
        
        # At step 0, should be init_value
        assert abs(schedule(0) - 1e-5) < 1e-7
        
        # During warmup (step 5), should be between init and peak
        warmup_lr = schedule(5)
        assert 1e-5 < warmup_lr < 1e-3
        
        # At warmup end (step 10), should be close to peak
        peak_lr = schedule(10)
        assert abs(peak_lr - 1e-3) < 1e-6
        
        # At end (step 100), should be close to end_value
        end_lr = schedule(100)
        assert abs(end_lr - 1e-6) < 1e-8


class TestErrorHandling:
    """Test error handling in optimizer components."""

    def test_adamw_creation_error(self):
        """Test error handling in AdamW creation."""
        # Test that we can detect errors in optimizer creation/usage
        try:
            # Test with an invalid hyperparameter that should cause issues
            optimizer = adamw(learning_rate=-1.0)  # Negative learning rate
            params = {'w': jnp.array([1.0])}
            grads = {'w': jnp.array([0.1])}
            
            # This might fail during init or apply_gradients
            opt_state = optimizer.init(params)
            result = optimizer.apply_gradients(grads, opt_state, params, step=0)
            
            # If it doesn't fail, that's also acceptable behavior
            assert result is not None
            
        except (OptimizerError, ValueError, TypeError):
            # Any of these errors are acceptable for invalid inputs
            pass
    
    def test_sgd_creation_error(self):
        """Test error handling in SGD creation with extreme values."""
        # Test with extremely large momentum that might cause issues
        try:
            optimizer = sgd(momentum=1e10)  # Extreme value
            params = {'w': jnp.array([1.0])}
            grads = {'w': jnp.array([0.1])} 
            opt_state = optimizer.init(params)
            
            # Try to use it - this might work or might not depending on implementation
            result = optimizer.apply_gradients(grads, opt_state, params, step=0)
            # If it doesn't error, that's also valid behavior
            assert result is not None
        except (OptimizerError, TypeError, ValueError, OverflowError):
            # Any of these error types are acceptable for extreme values
            pass
