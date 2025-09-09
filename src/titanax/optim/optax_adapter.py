"""Optax optimizer adapters for Titanax training.

This module provides wrapper classes and factory functions for integrating
Optax optimizers with Titanax's training system, including support for
learning rate scheduling and sharded parameters.
"""

import functools
from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
import optax  # type: ignore
try:
    from optax.typing import GradientTransformation  # type: ignore
except ImportError:
    # Fallback for older versions
    from optax._src.base import GradientTransformation  # type: ignore

from ..types import PyTree, Array, Params, OptState
from ..exceptions import OptimizerError


class OptaxAdapter:
    """Adapter for Optax optimizers with Titanax integration.
    
    This class wraps Optax optimizers to provide consistent interfaces
    for parameter updates, learning rate scheduling, and state management
    compatible with Titanax's distributed training system.
    
    Args:
        optimizer: The underlying Optax optimizer
        learning_rate: Learning rate (can be callable for scheduling)
        name: Optional name for the optimizer
    """
    
    def __init__(
        self,
        optimizer: GradientTransformation,
        learning_rate: Union[float, Callable[[int], float]],
        name: Optional[str] = None
    ):
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.name = name or "optax_adapter"
        
        # Store whether learning rate is callable for scheduling
        # Note: If learning_rate is callable, it should already be integrated
        # into the optimizer via optax scheduling transforms
        self._lr_is_callable = callable(learning_rate)
    
    def init(self, params: Params) -> OptState:
        """Initialize optimizer state.
        
        Args:
            params: Model parameters to optimize
            
        Returns:
            Initial optimizer state
        """
        try:
            return self.optimizer.init(params)
        except Exception as e:
            raise OptimizerError(
                f"Failed to initialize optimizer {self.name}: {e}",
                suggestion="Check that parameters are valid JAX PyTrees"
            ) from e
    
    def apply_gradients(
        self,
        grads: PyTree,
        opt_state: OptState,
        params: Params,
        step: Optional[int] = None,
        **kwargs
    ) -> tuple[Params, OptState]:
        """Apply gradients to parameters.
        
        Args:
            grads: Gradients to apply
            opt_state: Current optimizer state
            params: Current parameters
            step: Current step number (for learning rate scheduling)
            **kwargs: Additional arguments (ignored for compatibility)
            
        Returns:
            Tuple of (updated_params, updated_opt_state)
        """
        try:
            # Compute parameter updates using the optimizer
            # NOTE: Learning rate scheduling should be handled by the optimizer itself,
            # not by scaling gradients here. Gradient scaling breaks adaptive optimizers
            # like Adam/AdamW by affecting the running moment statistics.
            updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
            
            # Apply updates to parameters
            new_params = optax.apply_updates(params, updates)
            
            return new_params, new_opt_state
            
        except Exception as e:
            raise OptimizerError(
                f"Failed to apply gradients with optimizer {self.name}: {e}",
                suggestion="Check gradient shapes match parameter shapes"
            ) from e
    
    def get_learning_rate(self, step: int) -> float:
        """Get current learning rate for a given step.
        
        Args:
            step: Training step number
            
        Returns:
            Current learning rate value
            
        Note:
            When using schedules, this method evaluates the schedule function.
            The actual LR used by the optimizer may differ slightly due to
            Optax's internal transformations.
        """
        if self._lr_is_callable:
            return self.learning_rate(step)  # type: ignore
        else:
            return self.learning_rate  # type: ignore
    
    def describe(self) -> str:
        """Return a human-readable description of the optimizer."""
        lr_desc = "scheduled" if self._lr_is_callable else f"{self.learning_rate}"
        return f"{self.name}(lr={lr_desc})"


def adamw(
    learning_rate: Union[float, Callable[[int], float]] = 3e-4,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    weight_decay: float = 1e-2,
    mask: Optional[Union[Any, Callable[[PyTree], Any]]] = None,
    **kwargs
) -> OptaxAdapter:
    """Create an AdamW optimizer adapter.
    
    Args:
        learning_rate: Learning rate (can be a schedule function)
        b1: Coefficient for computing running averages of gradient
        b2: Coefficient for computing running averages of squared gradient
        eps: Term added to denominator to improve numerical stability
        weight_decay: Weight decay coefficient
        mask: Optional mask for applying weight decay selectively
        **kwargs: Additional arguments passed to optax.adamw
        
    Returns:
        OptaxAdapter wrapping the AdamW optimizer
        
    Example:
        ```python
        # Fixed learning rate
        optimizer = tx.optim.adamw(learning_rate=3e-4)
        
        # With learning rate schedule
        schedule = optax.cosine_decay_schedule(
            init_value=1e-3, decay_steps=10000
        )
        optimizer = tx.optim.adamw(learning_rate=schedule)
        ```
    """
    try:
        # Let Optax handle learning rate scheduling properly
        # This ensures Adam/AdamW statistics are computed correctly
        base_optimizer = optax.adamw(
            learning_rate=learning_rate,  # Pass schedule directly to Optax
            b1=b1,
            b2=b2,
            eps=eps,
            weight_decay=weight_decay,
            mask=mask,
            **kwargs
        )
        
        return OptaxAdapter(
            optimizer=base_optimizer,
            learning_rate=learning_rate,
            name="adamw"
        )
        
    except Exception as e:
        raise OptimizerError(
            f"Failed to create AdamW optimizer: {e}",
            suggestion="Check optimizer hyperparameters are valid"
        ) from e


def sgd(
    learning_rate: Union[float, Callable[[int], float]] = 1e-3,
    momentum: Optional[float] = None,
    nesterov: bool = False,
    **kwargs
) -> OptaxAdapter:
    """Create an SGD optimizer adapter.
    
    Args:
        learning_rate: Learning rate (can be a schedule function)
        momentum: Optional momentum coefficient
        nesterov: Whether to use Nesterov momentum
        **kwargs: Additional arguments passed to optax.sgd
        
    Returns:
        OptaxAdapter wrapping the SGD optimizer
        
    Example:
        ```python
        # Basic SGD
        optimizer = tx.optim.sgd(learning_rate=1e-3)
        
        # SGD with momentum
        optimizer = tx.optim.sgd(learning_rate=1e-3, momentum=0.9)
        
        # With learning rate schedule  
        schedule = optax.exponential_decay(
            init_value=1e-2, transition_steps=1000, decay_rate=0.9
        )
        optimizer = tx.optim.sgd(learning_rate=schedule, momentum=0.9)
        ```
    """
    try:
        # Let Optax handle learning rate scheduling properly
        base_optimizer = optax.sgd(
            learning_rate=learning_rate,  # Pass schedule directly to Optax
            momentum=momentum,
            nesterov=nesterov,
            **kwargs
        )
        
        return OptaxAdapter(
            optimizer=base_optimizer,
            learning_rate=learning_rate,
            name="sgd"
        )
        
    except Exception as e:
        raise OptimizerError(
            f"Failed to create SGD optimizer: {e}",
            suggestion="Check optimizer hyperparameters are valid"
        ) from e


def adam(
    learning_rate: Union[float, Callable[[int], float]] = 1e-3,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    **kwargs
) -> OptaxAdapter:
    """Create an Adam optimizer adapter.
    
    Args:
        learning_rate: Learning rate (can be a schedule function)
        b1: Coefficient for computing running averages of gradient
        b2: Coefficient for computing running averages of squared gradient
        eps: Term added to denominator to improve numerical stability
        **kwargs: Additional arguments passed to optax.adam
        
    Returns:
        OptaxAdapter wrapping the Adam optimizer
    """
    try:
        # Let Optax handle learning rate scheduling properly
        base_optimizer = optax.adam(
            learning_rate=learning_rate,  # Pass schedule directly to Optax
            b1=b1,
            b2=b2,
            eps=eps,
            **kwargs
        )
        
        return OptaxAdapter(
            optimizer=base_optimizer,
            learning_rate=learning_rate,
            name="adam"
        )
        
    except Exception as e:
        raise OptimizerError(
            f"Failed to create Adam optimizer: {e}",
            suggestion="Check optimizer hyperparameters are valid"
        ) from e


# Learning rate schedule utilities
def cosine_schedule(
    init_value: float,
    decay_steps: int,
    alpha: float = 0.0
) -> Callable[[int], float]:
    """Create a cosine decay learning rate schedule.
    
    Args:
        init_value: Initial learning rate
        decay_steps: Number of steps to decay over
        alpha: Minimum learning rate as a fraction of init_value
        
    Returns:
        Learning rate schedule function
    """
    return optax.cosine_decay_schedule(init_value, decay_steps, alpha)


def exponential_schedule(
    init_value: float,
    transition_steps: int,
    decay_rate: float,
    staircase: bool = False
) -> Callable[[int], float]:
    """Create an exponential decay learning rate schedule.
    
    Args:
        init_value: Initial learning rate
        transition_steps: Number of steps between decay events
        decay_rate: Multiplicative decay factor
        staircase: Whether to apply decay in discrete intervals
        
    Returns:
        Learning rate schedule function
    """
    return optax.exponential_decay(
        init_value, transition_steps, decay_rate, staircase
    )


def warmup_cosine_schedule(
    init_value: float,
    peak_value: float,
    warmup_steps: int,
    decay_steps: int,
    end_value: float = 0.0
) -> Callable[[int], float]:
    """Create a warmup + cosine decay learning rate schedule.
    
    Args:
        init_value: Initial learning rate (typically small)
        peak_value: Peak learning rate after warmup
        warmup_steps: Number of warmup steps
        decay_steps: Number of decay steps after warmup
        end_value: Final learning rate
        
    Returns:
        Learning rate schedule function
    """
    return optax.warmup_cosine_decay_schedule(
        init_value, peak_value, warmup_steps, decay_steps, end_value
    )
