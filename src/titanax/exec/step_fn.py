"""Step function decoration and compilation for training.

This module provides the @step_fn decorator that handles JIT compilation,
PRNG management, gradient accumulation, and other training utilities.
"""

import functools
from typing import Any, Callable, Dict, Optional, Union

import jax
import jax.numpy as jnp
from jax import lax
try:
    from jax.experimental import pjit
    from jax.experimental.shard_map import shard_map
except (ImportError, AttributeError):
    # In newer JAX versions, pjit is in the main module
    try:
        from jax import pjit  # type: ignore
        from jax.experimental.shard_map import shard_map
    except (ImportError, AttributeError):
        # Fallback if neither location works
        pjit = None  # type: ignore
        from jax.experimental.shard_map import shard_map

from ..types import StepFunction, PyTree, Array, BatchData, LogDict, StepOutput
from ..exceptions import EngineError


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
        """Apply the step function decoration."""
        
        @functools.wraps(func)
        def wrapper(state: PyTree, batch: BatchData) -> StepOutput:
            """Wrapped step function with validation and error handling."""
            
            # Input validation (outside JIT)
            if not isinstance(batch, dict):
                raise EngineError(
                    f"Batch must be a dictionary, got {type(batch)}",
                    suggestion="Ensure your dataloader returns dict-like batches"
                )
            
            # Check that state has required fields (basic validation)
            if not hasattr(state, 'step') or not hasattr(state, 'params'):
                raise EngineError(
                    "State must be a TrainState with 'step' and 'params' attributes",
                    suggestion="Use Engine.create_state() to create proper TrainState"
                )
            
            try:
                # Execute the original step function
                new_state, metrics = func(state, batch)
                
                # Output validation and metric processing (outside JIT)
                if not isinstance(metrics, dict):
                    raise EngineError(
                        f"Step function must return (state, metrics_dict), got metrics type {type(metrics)}",
                        suggestion="Return a dictionary of metrics as the second element"
                    )
                
                # Ensure metrics are scalar values
                validated_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, (int, float)):
                        validated_metrics[key] = float(value)
                    elif hasattr(value, 'item'):  # JAX array with single value
                        try:
                            validated_metrics[key] = float(value.item())
                        except Exception:
                            # Skip if can't convert to scalar
                            print(f"Warning: Skipping non-scalar metric '{key}' of type {type(value)}")
                            continue
                    else:
                        # Convert to scalar if possible
                        try:
                            validated_metrics[key] = float(jnp.asarray(value).item())
                        except Exception:
                            # Skip non-scalar metrics with a warning
                            print(f"Warning: Skipping non-scalar metric '{key}' of type {type(value)}")
                            continue
                
                return new_state, validated_metrics
                
            except Exception as e:
                raise EngineError(
                    f"Step function execution failed: {e}",
                    suggestion="Check your step function implementation for errors"
                ) from e
        
        # Store compilation parameters and original function for Engine to use
        wrapper._original_fn = func  # type: ignore
        wrapper._is_step_fn = True  # type: ignore
        wrapper._compile_params = {  # type: ignore
            'donate_argnums': donate_argnums,
            'static_argnums': static_argnums,
            'device': device
        }
        
        return wrapper
    
    # If func is provided, apply decoration directly (used as @step_fn)
    if func is not None:
        return decorator(func)
    
    # Otherwise, return decorator (used as @step_fn(...))
    return decorator


def compile_step_fn_with_mesh(
    step_fn: StepFunction, 
    mesh: jax.sharding.Mesh,
    donate_argnums: tuple[int, ...] = (0,),
    static_argnums: tuple[int, ...] = (),
    device: Optional[jax.Device] = None,
) -> StepFunction:
    """Compile a step function with proper mesh context for collectives.
    
    This function compiles a step function using shard_map with the provided mesh,
    enabling collective operations to work correctly.
    
    Args:
        step_fn: The step function to compile
        mesh: JAX mesh for distributed execution
        donate_argnums: Argument indices to donate
        static_argnums: Argument indices that are static
        device: Target device for compilation
        
    Returns:
        Compiled step function that works with mesh context
    """
    # Get the original function if it was decorated
    original_fn = getattr(step_fn, '_original_fn', step_fn)
    
    # Use shard_map to enable collective operations
    @functools.wraps(original_fn)
    def shard_mapped_fn(state: PyTree, batch: BatchData) -> StepOutput:
        """Execute step function with shard_map for collective support."""
        
        # Create a sharded function that allows collective operations
        def sharded_step(state, batch):
            return original_fn(state, batch)
        
        # Apply shard_map with the mesh
        with mesh:
            mapped_fn = shard_map(
                sharded_step,
                mesh=mesh,
                in_specs=(jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec()),
                out_specs=(jax.sharding.PartitionSpec(), jax.sharding.PartitionSpec())
            )
            return mapped_fn(state, batch)
    
    # JIT the shard_mapped function for performance
    compiled_fn = jax.jit(
        shard_mapped_fn,
        donate_argnums=donate_argnums,
        static_argnums=static_argnums,
    )
    
    return compiled_fn


def update_rngs(
    rngs: Dict[str, Array], 
    keys: list[str] | None = None,
    axis: str = 'batch'
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
                device_unique_rng = jax.random.fold_in(new_rng, jax.lax.axis_index(axis))
                new_rngs[key] = device_unique_rng
            except NameError:
                # Fallback if not in mesh context (single device or initialization)
                new_rngs[key] = new_rng
        else:
            new_rngs[key] = rng
    
    return new_rngs


def split_rng(rng: Array, num: int = 2, axis: str = 'batch') -> tuple[Array, ...]:
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
    accumulate_steps: int
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
        new_state = apply_fn(state, grads)
        return new_state, {"loss": loss}
    
    # Take only the required number of microbatches
    microbatches = batches[:accumulate_steps]
    
    # Convert to JAX arrays for scan
    batch_array = jax.tree_util.tree_map(
        lambda *xs: jnp.stack(xs), *microbatches
    )
    
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
    
    return new_state, {"loss": avg_loss, "accumulate_steps": float(accumulate_steps)}


def create_gradient_accumulation_step_fn(
    loss_fn: Callable,
    accumulate_steps: int = 1
) -> StepFunction:
    """Create a step function that performs gradient accumulation using JAX scan.
    
    This is a convenience function that creates the gradient and apply functions
    and returns a complete step function ready for use with Engine.
    
    Args:
        loss_fn: Function that computes loss: (params, batch) -> loss
        accumulate_steps: Number of microbatches to accumulate over
        
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
                return loss_fn(params, batch)
            
            loss, grads = jax.value_and_grad(compute_loss)(state.params)
            new_state = state.apply_gradients(grads=grads)
            return new_state, {"loss": loss}
        
        return simple_step  # type: ignore
    
    # Create accumulating step function
    def grad_fn(params, batch):
        """Compute loss and gradients for a single microbatch."""
        loss = loss_fn(params, batch)
        grads = jax.grad(loss_fn)(params, batch)
        return loss, grads
    
    def apply_fn(state, grads):
        """Apply accumulated gradients to state."""
        return state.apply_gradients(grads=grads)
    
    @step_fn()  # type: ignore
    def accumulating_step(state, batch):
        """Step function that accumulates gradients across microbatches."""
        # For now, we need the batches to be provided as a list in batch['microbatches']
        # This is a design choice - the dataloader should provide microbatches
        if 'microbatches' not in batch:
            raise EngineError(
                "Gradient accumulation requires batch to contain 'microbatches' key",
                suggestion="Modify your dataloader to provide microbatches or use accumulate_steps=1"
            )
        
        microbatches = batch['microbatches']
        if len(microbatches) < accumulate_steps:
            raise EngineError(
                f"Not enough microbatches: got {len(microbatches)}, need {accumulate_steps}",
                suggestion="Ensure your dataloader provides enough microbatches per step"
            )
        
        return gradient_accumulation_step(grad_fn, apply_fn, state, microbatches, accumulate_steps)
    
    return accumulating_step  # type: ignore


def is_step_fn(func: Any) -> bool:
    """Check if a function is a decorated step function.
    
    Args:
        func: Function to check
        
    Returns:
        True if function was decorated with @step_fn
    """
    return hasattr(func, '_is_step_fn') and func._is_step_fn


def get_original_step_fn(func: Any) -> Optional[StepFunction]:
    """Get the original uncompiled step function.
    
    Args:
        func: Compiled step function
        
    Returns:
        Original step function or None if not available
    """
    return getattr(func, '_original_fn', None)
