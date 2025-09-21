"""Common optimizer recipes and configurations.

This module provides convenient factory functions for creating optimizers with
typical defaults and learning rate schedules. These recipes are designed to work
out-of-the-box for most training scenarios while remaining easily customizable.
"""

from typing import Callable, Optional, Union

import optax  # type: ignore

from .optax_adapter import OptaxAdapter, adamw as _adamw


def adamw(
    learning_rate: Union[float, Callable[[int], float]] = 3e-4,
    weight_decay: float = 1e-2,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    warmup_steps: Optional[int] = None,
    decay_steps: Optional[int] = None,
    end_value: float = 0.0,
    **kwargs,
) -> OptaxAdapter:
    """Create an AdamW optimizer with optional warmup and cosine decay.

    This recipe provides a complete AdamW setup with optional warmup and
    cosine decay scheduling, commonly used in modern deep learning.

    Args:
        learning_rate: Peak learning rate (used as constant if no scheduling)
        weight_decay: Weight decay coefficient
        b1: Adam beta1 parameter (momentum)
        b2: Adam beta2 parameter (second moment decay)
        eps: Epsilon for numerical stability
        warmup_steps: Optional number of warmup steps
        decay_steps: Optional number of decay steps after warmup
        end_value: Final learning rate for cosine decay
        **kwargs: Additional arguments passed to base adamw

    Returns:
        OptaxAdapter with AdamW optimizer and optional scheduling

    Example:
        ```python
        # Basic AdamW
        opt = tx.optim.recipes.adamw(learning_rate=1e-4)

        # With warmup and cosine decay
        opt = tx.optim.recipes.adamw(
            learning_rate=1e-3,
            warmup_steps=1000,
            decay_steps=10000,
            weight_decay=1e-2
        )
        ```
    """
    # Create learning rate schedule if warmup/decay specified
    if warmup_steps is not None and decay_steps is not None:
        # For schedules, learning_rate should be a float (peak value)
        peak_lr = learning_rate if isinstance(learning_rate, (int, float)) else 1e-3

        if decay_steps <= warmup_steps:
            # If decay_steps is too small, just do warmup then constant
            schedule = optax.linear_schedule(
                init_value=peak_lr * 0.1,
                end_value=peak_lr,
                transition_steps=warmup_steps,
            )
        else:
            # Normal warmup + cosine decay
            effective_decay_steps = decay_steps - warmup_steps
            schedule = optax.warmup_cosine_decay_schedule(
                init_value=peak_lr * 0.1,  # Start at 10% of peak
                peak_value=peak_lr,
                warmup_steps=warmup_steps,
                decay_steps=effective_decay_steps,
                end_value=end_value,
            )
        effective_lr = schedule
    elif warmup_steps is not None:
        # Warmup only, then constant
        peak_lr = learning_rate if isinstance(learning_rate, (int, float)) else 1e-3
        schedule = optax.linear_schedule(
            init_value=peak_lr * 0.1,
            end_value=peak_lr,
            transition_steps=warmup_steps,
        )
        effective_lr = schedule
    elif decay_steps is not None:
        # Cosine decay only, no warmup
        peak_lr = learning_rate if isinstance(learning_rate, (int, float)) else 1e-3
        schedule = optax.cosine_decay_schedule(
            init_value=peak_lr,
            decay_steps=decay_steps,
            alpha=end_value / peak_lr if peak_lr > 0 else 0.0,
        )
        effective_lr = schedule
    else:
        # Constant learning rate
        effective_lr = learning_rate

    return _adamw(
        learning_rate=effective_lr,
        weight_decay=weight_decay,
        b1=b1,
        b2=b2,
        eps=eps,
        **kwargs,
    )


def sgd(
    learning_rate: Union[float, Callable[[int], float]] = 1e-3,
    momentum: float = 0.9,
    nesterov: bool = True,
    weight_decay: Optional[float] = None,
    warmup_steps: Optional[int] = None,
    decay_steps: Optional[int] = None,
    decay_rate: float = 0.1,
    **kwargs,
) -> OptaxAdapter:
    """Create an SGD optimizer with optional momentum and scheduling.

    This recipe provides SGD with momentum, optional weight decay, and
    learning rate scheduling commonly used in computer vision training.

    Args:
        learning_rate: Initial learning rate
        momentum: Momentum coefficient (0.9 is typical for most tasks)
        nesterov: Whether to use Nesterov momentum
        weight_decay: Optional weight decay coefficient
        warmup_steps: Optional number of linear warmup steps
        decay_steps: Optional number of steps for exponential decay
        decay_rate: Decay rate for exponential scheduling
        **kwargs: Additional arguments passed to base sgd

    Returns:
        OptaxAdapter with SGD optimizer and optional scheduling

    Example:
        ```python
        # Basic SGD with momentum
        opt = tx.optim.recipes.sgd(learning_rate=1e-2, momentum=0.9)

        # With weight decay and step decay
        opt = tx.optim.recipes.sgd(
            learning_rate=1e-1,
            momentum=0.9,
            weight_decay=5e-4,
            decay_steps=2000,
            decay_rate=0.5
        )
        ```
    """
    # Create learning rate schedule if specified
    if warmup_steps is not None and decay_steps is not None:
        # Linear warmup followed by exponential decay
        peak_lr = learning_rate if isinstance(learning_rate, (int, float)) else 1e-3
        warmup_schedule = optax.linear_schedule(
            init_value=peak_lr * 0.1,
            end_value=peak_lr,
            transition_steps=warmup_steps,
        )
        decay_schedule = optax.exponential_decay(
            init_value=peak_lr,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
        )
        schedule = optax.join_schedules(
            schedules=[warmup_schedule, decay_schedule],
            boundaries=[warmup_steps],
        )
        effective_lr = schedule
    elif warmup_steps is not None:
        # Warmup only
        peak_lr = learning_rate if isinstance(learning_rate, (int, float)) else 1e-3
        schedule = optax.linear_schedule(
            init_value=peak_lr * 0.1,
            end_value=peak_lr,
            transition_steps=warmup_steps,
        )
        effective_lr = schedule
    elif decay_steps is not None:
        # Exponential decay only
        schedule = optax.exponential_decay(
            init_value=learning_rate,
            transition_steps=decay_steps,
            decay_rate=decay_rate,
        )
        effective_lr = schedule
    else:
        # Constant learning rate
        effective_lr = learning_rate

    # Build optimizer chain
    transforms = []

    # Add momentum
    if momentum > 0:
        transforms.append(
            optax.sgd(learning_rate=effective_lr, momentum=momentum, nesterov=nesterov)
        )
    else:
        transforms.append(optax.sgd(learning_rate=effective_lr))

    # Add weight decay if specified
    if weight_decay is not None:
        transforms.append(optax.add_decayed_weights(weight_decay))

    # Chain transforms
    if len(transforms) == 1:
        optimizer = transforms[0]
    else:
        optimizer = optax.chain(*transforms)

    return OptaxAdapter(
        optimizer=optimizer, learning_rate=effective_lr, name="sgd_recipe"
    )


def adam_with_cosine_schedule(
    learning_rate: float = 1e-3,
    decay_steps: int = 10000,
    warmup_steps: int = 1000,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    alpha: float = 0.0,
    **kwargs,
) -> OptaxAdapter:
    """Create Adam optimizer with warmup + cosine decay schedule.

    This is a common recipe for transformer training and other modern architectures.

    Args:
        learning_rate: Peak learning rate
        decay_steps: Total steps for cosine decay (after warmup)
        warmup_steps: Number of linear warmup steps
        b1: Adam beta1 parameter
        b2: Adam beta2 parameter
        eps: Epsilon for numerical stability
        alpha: Minimum learning rate as fraction of peak
        **kwargs: Additional arguments

    Returns:
        OptaxAdapter with Adam + warmup cosine schedule
    """
    if decay_steps <= warmup_steps:
        # If decay_steps is too small, just do warmup then constant
        schedule = optax.linear_schedule(
            init_value=learning_rate * 0.1,
            end_value=learning_rate,
            transition_steps=warmup_steps,
        )
    else:
        # Normal warmup + cosine decay
        effective_decay_steps = decay_steps - warmup_steps
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=learning_rate * 0.1,  # Start at 10% of peak
            peak_value=learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=effective_decay_steps,
            end_value=learning_rate * alpha,
        )

    optimizer = optax.adam(learning_rate=schedule, b1=b1, b2=b2, eps=eps, **kwargs)

    return OptaxAdapter(optimizer=optimizer, learning_rate=schedule, name="adam_cosine")


# Learning rate schedule presets
class LRSchedules:
    """Collection of common learning rate schedules."""

    @staticmethod
    def cosine_with_warmup(
        peak_lr: float, warmup_steps: int, decay_steps: int, end_lr: float = 0.0
    ) -> Callable[[int], float]:
        """Warmup followed by cosine decay."""
        return optax.warmup_cosine_decay_schedule(
            init_value=peak_lr * 0.1,
            peak_value=peak_lr,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=end_lr,
        )

    @staticmethod
    def step_decay(
        init_lr: float, decay_factor: float = 0.1, step_size: int = 2000
    ) -> Callable[[int], float]:
        """Step decay schedule (piecewise constant)."""
        return optax.exponential_decay(
            init_value=init_lr,
            transition_steps=step_size,
            decay_rate=decay_factor,
            staircase=True,  # Makes it step-wise instead of continuous
        )

    @staticmethod
    def linear_warmup(
        init_lr: float, peak_lr: float, warmup_steps: int
    ) -> Callable[[int], float]:
        """Linear warmup to peak learning rate."""
        return optax.linear_schedule(
            init_value=init_lr, end_value=peak_lr, transition_steps=warmup_steps
        )
