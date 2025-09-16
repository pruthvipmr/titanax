"""Step function decoration and helper utilities for Titanax training.

This module provides the :func:`@step_fn` decorator as well as gradient
accumulation helpers used by the execution engine. The decorator is responsible
for enforcing basic contracts (state/batch types, metric structure) and for
providing metadata required during compilation.
"""

import functools
import inspect
from collections.abc import Mapping
from typing import Any, Callable, Dict, Optional, Union, TYPE_CHECKING

import jax
import jax.numpy as jnp

from ..types import StepFunction, PyTree, Array, BatchData, StepOutput
from ..exceptions import EngineError

if TYPE_CHECKING:  # pragma: no cover - circular import guard
    from .engine import TrainState


def _get_train_state_type() -> type:
    """Return the TrainState type without causing circular imports."""

    from .engine import TrainState as _TrainState  # Local import to avoid cycles

    return _TrainState


def _validate_train_state(state: Any, func_name: str) -> None:
    """Ensure the first argument passed to a step function is a TrainState."""

    train_state_type = _get_train_state_type()
    if not isinstance(state, train_state_type):
        raise ValueError(
            (
                f"Step function '{func_name}' must receive a TrainState as its first argument, "
                f"got {type(state).__name__}."
            )
            + " Fix: update the signature to `def {func_name}(state: TrainState, batch: Mapping[str, Array], ...)`."
        )


def _validate_batch(batch: Any, func_name: str) -> Mapping[str, Any]:
    """Validate that the batch argument is mapping-like."""

    if not isinstance(batch, Mapping):
        raise ValueError(
            (
                f"Step function '{func_name}' expects batch to be a mapping of arrays, "
                f"got {type(batch).__name__}."
            )
            + " Fix: return dict-like batches from your dataloader (e.g. {'x': ..., 'y': ...})."
        )
    return batch


def _ensure_scalar_metric(key: str, value: Any, func_name: str) -> float:
    """Convert a metric value to a scalar float, enforcing shape and dtype rules."""

    # Fast-path for python scalars
    if isinstance(value, (int, float)):
        return float(value)

    array_value = jnp.asarray(value)

    if array_value.size != 1:
        raise ValueError(
            (
                f"Metric '{key}' returned from '{func_name}' must be a scalar. "
                f"Observed shape {array_value.shape}."
            )
            + " Fix: reduce the metric before returning it (e.g. jnp.mean or jnp.squeeze)."
        )

    # Convert to float for logging friendliness
    return float(array_value.reshape(()))


def _validate_metrics_host(metrics: Any, func_name: str) -> Dict[str, float]:
    """Validate metrics outside of JIT execution and convert to floats."""

    if not isinstance(metrics, Mapping):
        raise ValueError(
            (
                f"Step function '{func_name}' must return a tuple '(state, metrics_dict)'. "
                f"Got metrics of type {type(metrics).__name__}."
            )
            + " Fix: return a dictionary mapping metric names to scalar values."
        )

    validated: Dict[str, float] = {}
    for key, value in metrics.items():
        validated[key] = _ensure_scalar_metric(key, value, func_name)

    return validated


def _validate_metrics_tree(metrics: Any, func_name: str) -> Any:
    """Validate metrics structure during JIT tracing.

    Returns the metrics unchanged but raises ValueError when the metrics cannot
    be reduced to scalars (e.g. multi-dimensional arrays).
    """

    if not isinstance(metrics, Mapping):
        raise ValueError(
            (
                f"Step function '{func_name}' must return metrics as a mapping. "
                f"Got {type(metrics).__name__}."
            )
            + " Fix: return a dict like {'loss': loss_value}."
        )

    for key, value in metrics.items():
        array_value = jnp.asarray(value)
        if array_value.size != 1:
            raise ValueError(
                (
                    f"Metric '{key}' returned from '{func_name}' must be reducible to a scalar. "
                    f"Observed shape {array_value.shape}."
                )
                + " Fix: compute a scalar summary before returning it."
            )

    return metrics


def step_fn(
    func: Optional[StepFunction] = None,
    *,
    donate_argnums: tuple[int, ...] = (0,),
    static_argnums: tuple[int, ...] = (),
    device: Optional[jax.Device] = None,
) -> Union[StepFunction, Callable[[StepFunction], StepFunction]]:
    """Decorator to mark a function as a training step function.

    This decorator marks the function for compilation by the Engine.
    The actual compilation with mesh context happens when the Engine
    calls register_step_fn().

    This decorator handles:
    - Marking function as a step function
    - Input/output validation
    - Error handling with helpful messages

    Args:
        donate_argnums: Argument indices to donate (default: donate state)
        static_argnums: Argument indices that are static (compile-time constants)
        device: Target device for compilation (None for automatic placement)

    Returns:
        Decorator function that wraps step functions with validation

    Example:
        ```python
        @tx.step_fn
        def train_step(state, batch):
            def loss_fn(params):
                logits = model_apply(params, batch['x'])
                return cross_entropy(logits, batch['y'])

            loss, grads = jax.value_and_grad(loss_fn)(state.params)
            grads = tx.collectives.psum(grads, axis='data')
            state = state.apply_gradients(grads=grads)
            return state, {'loss': loss}
        ```
    """

    def decorator(func: StepFunction) -> StepFunction:
        """Apply the step function decoration with validation hooks."""

        signature = inspect.signature(func)
        parameter_names = list(signature.parameters.keys())
        if len(parameter_names) < 2:
            raise ValueError(
                (
                    f"Step function '{func.__name__}' must accept at least two arguments (state, batch)."
                )
                + " Fix: define the function as `def step(state: TrainState, batch: Mapping[str, Array], ...)`."
            )

        first_param, second_param = parameter_names[0], parameter_names[1]

        def _extract_state_batch(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
            try:
                bound = signature.bind_partial(*args, **kwargs)
            except TypeError as exc:  # Missing required parameters
                raise ValueError(
                    (
                        f"Step function '{func.__name__}' must be called with arguments `(state, batch, ...)`."
                    )
                    + f" Fix: {exc}."
                ) from exc
            if (
                first_param not in bound.arguments
                or second_param not in bound.arguments
            ):
                raise ValueError(
                    (
                        f"Step function '{func.__name__}' must be called with positional arguments for state and batch."
                    )
                    + " Fix: call it as `step(state, batch, ...)`."
                )
            return bound.arguments[first_param], bound.arguments[second_param]

        def _validated_body(*args: Any, **kwargs: Any) -> StepOutput:
            state, batch = _extract_state_batch(*args, **kwargs)
            _validate_train_state(state, func.__name__)
            _validate_batch(batch, func.__name__)

            new_state, metrics = func(*args, **kwargs)

            _validate_train_state(new_state, func.__name__)
            metrics = _validate_metrics_tree(metrics, func.__name__)
            return new_state, metrics

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> StepOutput:
            state, batch = _extract_state_batch(*args, **kwargs)
            _validate_train_state(state, func.__name__)
            _validate_batch(batch, func.__name__)

            new_state, metrics = func(*args, **kwargs)

            _validate_train_state(new_state, func.__name__)
            validated_metrics = _validate_metrics_host(metrics, func.__name__)
            return new_state, validated_metrics

        # Store compilation parameters and validation metadata
        wrapper._original_fn = func  # type: ignore[attr-defined]
        wrapper._validated_fn = _validated_body  # type: ignore[attr-defined]
        wrapper._is_step_fn = True  # type: ignore[attr-defined]
        wrapper._compile_params = {  # type: ignore[attr-defined]
            "donate_argnums": donate_argnums,
            "static_argnums": static_argnums,
            "device": device,
        }
        wrapper._parameter_names = tuple(parameter_names)  # type: ignore[attr-defined]

        return wrapper

    # If func is provided, apply decoration directly (used as @step_fn)
    if func is not None:
        return decorator(func)

    # Otherwise, return decorator (used as @step_fn(...))
    return decorator


def update_rngs(
    rngs: Dict[str, Array], keys: list[str] | None = None, axis: str = "batch"
) -> Dict[str, Array]:
    """Update PRNG keys for the next step with proper per-device handling.

    This function updates PRNG keys while maintaining device uniqueness in
    multi-device scenarios. Each device gets deterministic but unique RNG streams.

    Args:
        rngs: Current PRNG keys dictionary
        keys: List of keys to update (None to update all)
        axis: Mesh axis name to use for device indexing

    Returns:
        Dictionary with updated per-device PRNG keys

    Note:
        This function should be called inside a jitted function with mesh context
        to ensure proper device-specific key generation.
    """
    if keys is None:
        keys = list(rngs.keys())

    new_rngs = {}
    for key, rng in rngs.items():
        if key in keys:
            # Split and make device-unique using axis_index
            new_rng, _ = jax.random.split(rng)
            # Fold in device index to ensure per-device uniqueness
            try:
                device_unique_rng = jax.random.fold_in(
                    new_rng, jax.lax.axis_index(axis)
                )
                new_rngs[key] = device_unique_rng
            except NameError:
                # Fallback if not in mesh context (single device or initialization)
                new_rngs[key] = new_rng
        else:
            new_rngs[key] = rng

    return new_rngs


def split_rng(rng: Array, num: int = 2, axis: str = "batch") -> tuple[Array, ...]:
    """Split a PRNG key into multiple per-device unique keys.

    This function splits a PRNG key while ensuring device uniqueness in
    multi-device scenarios using axis_index and fold_in.

    Args:
        rng: Source PRNG key
        num: Number of keys to generate
        axis: Mesh axis name for device indexing

    Returns:
        Tuple of split per-device PRNG keys
    """
    # Split the key first
    split_keys = jax.random.split(rng, num)

    # Make each split device-unique
    try:
        device_index = jax.lax.axis_index(axis)
        device_keys = []
        for i, key in enumerate(split_keys):
            # Fold in both split index and device index for uniqueness
            unique_key = jax.random.fold_in(key, device_index + i * 1000)
            device_keys.append(unique_key)
        return tuple(device_keys)
    except NameError:
        # Fallback if not in mesh context (single device or initialization)
        return tuple(split_keys)


def gradient_accumulation_step(
    grad_fn: Callable,
    apply_fn: Callable,
    state: PyTree,
    batches: list[BatchData],
    accumulate_steps: int,
    loss_scale: float | None = None,
) -> StepOutput:
    """Execute gradient accumulation across multiple microbatches using JAX control flow.

    This function uses jax.lax.scan to properly accumulate gradients across microbatches
    within JIT-compiled code, avoiding Python loops that can't be optimized.

    Args:
        grad_fn: Function that computes gradients for a single microbatch: (params, batch) -> (loss, grads)
        apply_fn: Function that applies gradients: (state, grads) -> new_state
        state: Current training state
        batches: List of microbatch data (should have length >= accumulate_steps)
        accumulate_steps: Number of accumulation steps
        loss_scale: Optional scaling factor used to unscale gradients that were
            computed with loss scaling (e.g. for mixed precision training).

    Returns:
        Tuple of (final_state, aggregated_metrics)

    Example:
        ```python
        def loss_and_grad_fn(params, batch):
            loss = compute_loss(params, batch)
            grads = jax.grad(compute_loss)(params, batch)
            return loss, grads

        def apply_gradients_fn(state, accumulated_grads):
            return state.apply_gradients(grads=accumulated_grads)

        state, metrics = gradient_accumulation_step(
            loss_and_grad_fn, apply_gradients_fn, state, batches, 4
        )
        ```
    """
    if accumulate_steps == 1 or len(batches) == 1:
        # No accumulation needed - single step
        loss, grads = grad_fn(state.params, batches[0])
        if loss_scale is not None:
            grads = jax.tree_util.tree_map(lambda g: g / loss_scale, grads)
            loss = loss / loss_scale
        new_state = apply_fn(state, grads)
        return new_state, {"loss": loss}

    # Take only the required number of microbatches
    microbatches = batches[:accumulate_steps]

    # Convert to JAX arrays for scan
    batch_array = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *microbatches)

    # Get initial gradients to establish the structure
    _, init_grads = grad_fn(state.params, microbatches[0])
    # Initialize with zeros to maintain structure
    init_accumulated_grads = jax.tree_util.tree_map(jnp.zeros_like, init_grads)

    # Use JAX scan for proper gradient accumulation
    def scan_fn(carry, batch):
        """Inner function for lax.scan that accumulates gradients."""
        accumulated_grads, total_loss, count = carry

        # Compute loss and gradients for this microbatch
        loss, grads = grad_fn(state.params, batch)
        if loss_scale is not None:
            grads = jax.tree_util.tree_map(lambda g: g / loss_scale, grads)
            loss = loss / loss_scale

        # Accumulate gradients (tree_map handles PyTree structure)
        accumulated_grads = jax.tree_util.tree_map(
            lambda acc, new: acc + new, accumulated_grads, grads
        )

        # Accumulate loss for averaging
        total_loss += loss
        count += 1

        return (accumulated_grads, total_loss, count), None

    # Initialize carry state with proper gradient structure
    init_carry = (init_accumulated_grads, 0.0, 0)

    # Run scan to accumulate gradients
    (accumulated_grads, total_loss, count), _ = jax.lax.scan(
        scan_fn, init_carry, batch_array
    )

    # Average the accumulated gradients
    averaged_grads = jax.tree_util.tree_map(
        lambda g: g / accumulate_steps, accumulated_grads
    )

    # Apply the averaged gradients
    new_state = apply_fn(state, averaged_grads)

    # Compute average loss
    avg_loss = total_loss / accumulate_steps

    metrics = {
        "loss": avg_loss,
        "accumulate_steps": float(accumulate_steps),
    }
    if loss_scale is not None:
        metrics["loss_scale"] = float(loss_scale)

    return new_state, metrics


def create_gradient_accumulation_step_fn(
    loss_fn: Callable,
    accumulate_steps: int = 1,
    *,
    loss_scale: float | None = None,
) -> StepFunction:
    """Create a step function that performs gradient accumulation using JAX scan.

    This is a convenience function that creates the gradient and apply functions
    and returns a complete step function ready for use with Engine.

    Args:
        loss_fn: Function that computes loss: (params, batch) -> loss
        accumulate_steps: Number of microbatches to accumulate over
        loss_scale: Optional scaling factor to unscale gradients when using
            mixed precision loss scaling.

    Returns:
        Step function that can be used with Engine.fit()

    Example:
        ```python
        def loss_fn(params, batch):
            logits = model_apply(params, batch['x'])
            return cross_entropy(logits, batch['y'])

        # Create accumulating step function
        step_fn = create_gradient_accumulation_step_fn(loss_fn, accumulate_steps=4)

        # Use with engine
        engine.fit(step_fn, dataloader, state=state)
        ```
    """
    if accumulate_steps == 1:
        # No accumulation needed - return standard step function
        @step_fn()  # type: ignore
        def simple_step(state, batch):
            def compute_loss(params):
                base_loss = loss_fn(params, batch)
                if loss_scale is not None:
                    return base_loss * loss_scale
                return base_loss

            loss, grads = jax.value_and_grad(compute_loss)(state.params)
            if loss_scale is not None:
                grads = jax.tree_util.tree_map(lambda g: g / loss_scale, grads)
                loss = loss / loss_scale

            new_state = state.apply_gradients(grads=grads)
            metrics = {"loss": loss}
            if loss_scale is not None:
                metrics["loss_scale"] = float(loss_scale)
            return new_state, metrics

        return simple_step  # type: ignore

    # Create accumulating step function
    def grad_fn(params, batch):
        """Compute loss and gradients for a single microbatch."""

        def scaled_loss(p):
            base_loss = loss_fn(p, batch)
            if loss_scale is not None:
                return base_loss * loss_scale
            return base_loss

        loss, grads = jax.value_and_grad(scaled_loss)(params)
        return loss, grads

    def apply_fn(state, grads):
        """Apply accumulated gradients to state."""
        return state.apply_gradients(grads=grads)

    @step_fn()  # type: ignore
    def accumulating_step(state, batch):
        """Step function that accumulates gradients across microbatches."""
        # For now, we need the batches to be provided as a list in batch['microbatches']
        # This is a design choice - the dataloader should provide microbatches
        if "microbatches" not in batch:
            raise EngineError(
                "Gradient accumulation requires batch to contain 'microbatches' key",
                suggestion="Modify your dataloader to provide microbatches or use accumulate_steps=1",
            )

        microbatches = batch["microbatches"]
        if len(microbatches) < accumulate_steps:
            raise EngineError(
                f"Not enough microbatches: got {len(microbatches)}, need {accumulate_steps}",
                suggestion="Ensure your dataloader provides enough microbatches per step",
            )

        return gradient_accumulation_step(
            grad_fn,
            apply_fn,
            state,
            microbatches,
            accumulate_steps,
            loss_scale=loss_scale,
        )

    return accumulating_step  # type: ignore


def is_step_fn(func: Any) -> bool:
    """Check if a function is a decorated step function.

    Args:
        func: Function to check

    Returns:
        True if function was decorated with @step_fn
    """
    return hasattr(func, "_is_step_fn") and func._is_step_fn


def get_original_step_fn(func: Any) -> Optional[StepFunction]:
    """Get the original uncompiled step function.

    Args:
        func: Compiled step function

    Returns:
        Original step function or None if not available
    """
    return getattr(func, "_original_fn", None)
