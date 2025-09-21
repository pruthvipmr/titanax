"""Unit tests for optimizer recipes and enhanced OptaxAdapter functionality."""

import jax
import jax.numpy as jnp
import optax  # type: ignore

from src.titanax.optim.optax_adapter import (
    OptaxAdapter,
    adamw as base_adamw,
    sgd as base_sgd,
)
from src.titanax.optim import recipes
from src.titanax.exceptions import OptimizerError


class TestOptaxAdapterEnhancements:
    """Test the enhanced OptaxAdapter functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = {
            "layer1": {"weight": jnp.array([[1.0, 2.0]]), "bias": jnp.array([0.1])},
            "layer2": {"weight": jnp.array([[0.5]]), "bias": jnp.array([0.0])},
        }

    def test_current_lr_alias(self):
        """Test that current_lr() is an alias for get_learning_rate()."""
        optimizer = base_adamw(learning_rate=1e-3)

        # Should be identical results
        step = 5
        lr1 = optimizer.get_learning_rate(step)
        lr2 = optimizer.current_lr(step)

        assert lr1 == lr2 == 1e-3

    def test_current_lr_with_schedule(self):
        """Test current_lr with learning rate schedule."""
        schedule = optax.cosine_decay_schedule(1e-3, 100)
        optimizer = base_adamw(learning_rate=schedule)

        step = 50
        expected_lr = schedule(step)
        actual_lr = optimizer.current_lr(step)

        assert jnp.allclose(actual_lr, expected_lr)

    def test_lr_methods_consistency(self):
        """Test that both LR methods give consistent results across steps."""
        schedule = optax.exponential_decay(1e-2, 10, 0.9)
        optimizer = base_adamw(learning_rate=schedule)

        for step in [0, 1, 5, 10, 20, 100]:
            lr1 = optimizer.get_learning_rate(step)
            lr2 = optimizer.current_lr(step)
            assert jnp.allclose(lr1, lr2), f"Mismatch at step {step}: {lr1} vs {lr2}"

    def test_constant_lr_both_methods(self):
        """Test both LR methods with constant learning rate."""
        lr = 5e-4
        optimizer = base_adamw(learning_rate=lr)

        for step in [0, 1, 100, 1000]:
            assert optimizer.get_learning_rate(step) == lr
            assert optimizer.current_lr(step) == lr

    def test_describe_with_schedule(self):
        """Test describe method with scheduled learning rate."""
        schedule = optax.linear_schedule(1e-3, 1e-4, 1000)
        optimizer = base_adamw(learning_rate=schedule)

        description = optimizer.describe()
        assert "adamw" in description
        assert "scheduled" in description

    def test_describe_with_constant_lr(self):
        """Test describe method with constant learning rate."""
        optimizer = base_sgd(learning_rate=1e-2)

        description = optimizer.describe()
        assert "sgd" in description
        assert "1e-2" in description or "0.01" in description


class TestRecipesModule:
    """Test the recipes module functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.params = {
            "weight": jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]),
            "bias": jnp.array([0.1, 0.2, 0.3]),
        }

    def test_adamw_recipe_basic(self):
        """Test basic AdamW recipe creation."""
        optimizer = recipes.adamw(learning_rate=1e-4, weight_decay=1e-2)

        assert isinstance(optimizer, OptaxAdapter)
        assert optimizer.name == "adamw"

        # Test initialization
        opt_state = optimizer.init(self.params)
        assert opt_state is not None

        # Test gradient application
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, self.params)
        new_params, new_opt_state = optimizer.apply_gradients(
            grads, opt_state, self.params
        )

        # Parameters should have changed
        assert not jnp.allclose(new_params["weight"], self.params["weight"])

    def test_adamw_recipe_with_warmup_and_decay(self):
        """Test AdamW recipe with warmup and cosine decay."""
        optimizer = recipes.adamw(
            learning_rate=1e-3,
            warmup_steps=100,
            decay_steps=1000,
            weight_decay=1e-2,
            end_value=1e-5,
        )

        # Test that schedule is created properly
        assert callable(optimizer.learning_rate)

        # Test learning rate schedule behavior
        lr_initial = optimizer.current_lr(0)
        lr_warmup_mid = optimizer.current_lr(50)
        lr_peak = optimizer.current_lr(100)
        lr_decay_mid = optimizer.current_lr(500)
        lr_end = optimizer.current_lr(1000)

        # During warmup, LR should increase
        assert lr_initial < lr_warmup_mid < lr_peak
        # During decay, LR should decrease
        assert lr_peak > lr_decay_mid > lr_end

    def test_adamw_recipe_warmup_only(self):
        """Test AdamW recipe with warmup only."""
        peak_lr = 2e-3
        warmup_steps = 50

        optimizer = recipes.adamw(
            learning_rate=peak_lr, warmup_steps=warmup_steps, weight_decay=1e-3
        )

        # Should start low and reach peak
        lr_start = optimizer.current_lr(0)
        lr_peak = optimizer.current_lr(warmup_steps)
        lr_after = optimizer.current_lr(warmup_steps + 10)

        assert lr_start < lr_peak
        assert jnp.allclose(lr_peak, peak_lr, atol=1e-6)
        assert jnp.allclose(lr_after, peak_lr, atol=1e-6)  # Constant after warmup

    def test_adamw_recipe_decay_only(self):
        """Test AdamW recipe with cosine decay only."""
        init_lr = 1e-3
        decay_steps = 200
        end_lr = 1e-5

        optimizer = recipes.adamw(
            learning_rate=init_lr, decay_steps=decay_steps, end_value=end_lr
        )

        lr_start = optimizer.current_lr(0)
        lr_mid = optimizer.current_lr(100)
        lr_end = optimizer.current_lr(decay_steps)

        assert jnp.allclose(lr_start, init_lr, atol=1e-6)
        assert lr_start > lr_mid > lr_end
        assert jnp.allclose(lr_end, end_lr, atol=1e-6)

    def test_sgd_recipe_basic(self):
        """Test basic SGD recipe creation."""
        optimizer = recipes.sgd(learning_rate=1e-2, momentum=0.9, nesterov=True)

        assert isinstance(optimizer, OptaxAdapter)
        assert optimizer.name == "sgd_recipe"

        # Test that it works
        opt_state = optimizer.init(self.params)
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.1, self.params)
        new_params, new_opt_state = optimizer.apply_gradients(
            grads, opt_state, self.params
        )

        assert not jnp.allclose(new_params["weight"], self.params["weight"])

    def test_sgd_recipe_with_weight_decay(self):
        """Test SGD recipe with weight decay."""
        optimizer = recipes.sgd(
            learning_rate=1e-2, momentum=0.9, weight_decay=1e-4, nesterov=False
        )

        opt_state = optimizer.init(self.params)
        grads = jax.tree_util.tree_map(lambda x: jnp.zeros_like(x), self.params)

        # Even with zero gradients, weight decay should change parameters
        new_params, _ = optimizer.apply_gradients(grads, opt_state, self.params)

        # Parameters should change (weight decay applied)
        assert not jnp.allclose(new_params["weight"], self.params["weight"])

        # With multiple steps, the effect should accumulate
        new_params2, _ = optimizer.apply_gradients(grads, opt_state, new_params)
        assert not jnp.allclose(new_params2["weight"], new_params["weight"])

    def test_sgd_recipe_with_warmup_and_decay(self):
        """Test SGD recipe with warmup and exponential decay."""
        optimizer = recipes.sgd(
            learning_rate=1e-1,
            momentum=0.9,
            warmup_steps=20,
            decay_steps=50,
            decay_rate=0.5,
        )

        # Test schedule behavior
        lr_start = optimizer.current_lr(0)
        lr_warmup_end = optimizer.current_lr(20)
        lr_after_decay = optimizer.current_lr(70)

        assert lr_start < lr_warmup_end
        assert lr_after_decay < lr_warmup_end

    def test_adam_with_cosine_schedule(self):
        """Test Adam with cosine schedule recipe."""
        optimizer = recipes.adam_with_cosine_schedule(
            learning_rate=1e-3,
            decay_steps=1000,
            warmup_steps=100,
            b1=0.9,
            b2=0.999,
            alpha=0.1,
        )

        assert isinstance(optimizer, OptaxAdapter)
        assert optimizer.name == "adam_cosine"

        # Test schedule properties
        lr_start = optimizer.current_lr(0)
        lr_peak = optimizer.current_lr(100)
        lr_end = optimizer.current_lr(1000)

        # Should start low, peak at warmup end, decay to alpha * peak
        assert lr_start < lr_peak
        assert lr_end < lr_peak
        assert jnp.allclose(lr_end, 1e-3 * 0.1, atol=1e-6)

    def test_recipes_parameter_registry(self):
        """Test that recipes preserve parameter configurations correctly."""
        # Test AdamW hyperparameters
        adamw_opt = recipes.adamw(
            learning_rate=5e-4, weight_decay=2e-2, b1=0.95, b2=0.995, eps=1e-6
        )

        # Should be able to initialize and apply
        opt_state = adamw_opt.init(self.params)
        grads = jax.tree_util.tree_map(lambda x: jnp.ones_like(x) * 0.01, self.params)

        # Should not raise errors
        new_params, new_opt_state = adamw_opt.apply_gradients(
            grads, opt_state, self.params
        )
        assert new_params is not None
        assert new_opt_state is not None

        # Test SGD hyperparameters
        sgd_opt = recipes.sgd(
            learning_rate=1e-1, momentum=0.95, nesterov=False, weight_decay=5e-5
        )

        opt_state = sgd_opt.init(self.params)
        new_params, new_opt_state = sgd_opt.apply_gradients(
            grads, opt_state, self.params
        )
        assert new_params is not None
        assert new_opt_state is not None


class TestLRSchedules:
    """Test the LRSchedules utility class."""

    def test_cosine_with_warmup(self):
        """Test cosine schedule with warmup."""
        schedule = recipes.LRSchedules.cosine_with_warmup(
            peak_lr=1e-3, warmup_steps=100, decay_steps=1000, end_lr=1e-5
        )

        # Test key points
        lr_start = schedule(0)
        lr_warmup = schedule(100)
        lr_mid = schedule(500)
        lr_end = schedule(1000)

        assert lr_start < lr_warmup  # Should increase during warmup
        assert lr_warmup > lr_mid > lr_end  # Should decay after warmup
        assert jnp.allclose(lr_end, 1e-5, atol=1e-7)

    def test_step_decay(self):
        """Test step decay schedule."""
        schedule = recipes.LRSchedules.step_decay(
            init_lr=1e-2, decay_factor=0.5, step_size=100
        )

        lr_0 = schedule(0)
        lr_99 = schedule(99)
        lr_100 = schedule(100)
        lr_200 = schedule(200)

        # Should be constant within step
        assert jnp.allclose(lr_0, lr_99)
        assert jnp.allclose(lr_0, 1e-2)

        # Should decay at boundaries
        assert lr_100 < lr_0
        assert lr_200 < lr_100
        assert jnp.allclose(lr_100, 1e-2 * 0.5, atol=1e-6)
        assert jnp.allclose(lr_200, 1e-2 * 0.5 * 0.5, atol=1e-6)

    def test_linear_warmup(self):
        """Test linear warmup schedule."""
        schedule = recipes.LRSchedules.linear_warmup(
            init_lr=1e-4, peak_lr=1e-3, warmup_steps=50
        )

        lr_start = schedule(0)
        lr_mid = schedule(25)
        lr_end = schedule(50)
        lr_after = schedule(100)

        assert jnp.allclose(lr_start, 1e-4, atol=1e-6)
        assert lr_start < lr_mid < lr_end
        assert jnp.allclose(lr_end, 1e-3, atol=1e-6)
        assert jnp.allclose(lr_after, 1e-3, atol=1e-6)  # Constant after warmup


class TestOptimizerErrorHandling:
    """Test optimizer error handling and edge cases."""

    def test_optimizer_creation_error_handling(self):
        """Test that optimizer creation errors are handled properly."""
        # This should work fine
        optimizer = recipes.adamw(learning_rate=1e-3)
        assert optimizer is not None

        # Test invalid parameters (if Optax raises errors)
        try:
            # Very large learning rate might cause issues
            recipes.adamw(learning_rate=1e10, weight_decay=-1.0)
            # If this doesn't raise, that's fine too - depends on Optax validation
        except OptimizerError:
            # This is the expected behavior for invalid parameters
            pass

    def test_schedule_edge_cases(self):
        """Test learning rate schedule edge cases."""
        # Zero steps
        optimizer = recipes.adamw(learning_rate=1e-3, warmup_steps=0, decay_steps=100)
        assert optimizer.current_lr(0) > 0

        # Same warmup and decay steps
        optimizer = recipes.adamw(learning_rate=1e-3, warmup_steps=100, decay_steps=100)
        assert optimizer.current_lr(50) > 0

    def test_gradient_application_consistency(self):
        """Test that recipes produce consistent gradient applications."""
        params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])}
        grads = {"w": jnp.array([0.1, -0.1]), "b": jnp.array([0.05])}

        # Test multiple optimizers
        optimizers = [
            recipes.adamw(learning_rate=1e-3),
            recipes.sgd(learning_rate=1e-2, momentum=0.9),
            recipes.adam_with_cosine_schedule(learning_rate=1e-3, decay_steps=100),
        ]

        for optimizer in optimizers:
            opt_state = optimizer.init(params)
            new_params, new_opt_state = optimizer.apply_gradients(
                grads, opt_state, params
            )

            # Should update parameters
            assert not jnp.allclose(new_params["w"], params["w"])
            assert not jnp.allclose(new_params["b"], params["b"])

            # Should return valid states
            assert new_opt_state is not None

            # Second step should also work
            new_params2, new_opt_state2 = optimizer.apply_gradients(
                grads, new_opt_state, new_params
            )
            assert new_params2 is not None
            assert new_opt_state2 is not None
