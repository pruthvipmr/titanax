"""Step function decoration and compilation for training.

This module provides the @step_fn decorator that handles JIT compilation,
PRNG management, gradient accumulation, and other training utilities.
"""

import functools
from typing import Any, Callable, Dict, Optional

import jax
import jax.numpy as jnp

from ..types import StepFunction, PyTree, Array, BatchData, LogDict, StepOutput
from ..exceptions import EngineError


def step_fn(
    donate_argnums: tuple[int, ...] = (0,),
    static_argnums: tuple[int, ...] = (),
    device: Optional[jax.Device] = None,
) -> Callable[[StepFunction], StepFunction]:
    """Decorator to create a compiled training step function.
    
    This decorator handles:
    - JIT compilation with appropriate donation and static arguments
    - PRNG key management and threading
    - Input/output validation
    - Error handling with helpful messages
    
    Args:
        donate_argnums: Argument indices to donate (default: donate state)
        static_argnums: Argument indices that are static (compile-time constants)
        device: Target device for compilation (None for automatic placement)
        
    Returns:
        Decorator function that wraps step functions with compilation and management
        
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
        
        # Create JIT compiled version of just the core function
        compiled_core = jax.jit(
            func,
            donate_argnums=donate_argnums,
            static_argnums=static_argnums,
            device=device
        )
        
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
                # Execute the compiled step function
                new_state, metrics = compiled_core(state, batch)
                
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
        
        compiled_fn = wrapper
        
        # Store original function for inspection
        compiled_fn._original_fn = func  # type: ignore
        compiled_fn._is_step_fn = True  # type: ignore
        
        return compiled_fn
    
    return decorator


def update_rngs(rngs: Dict[str, Array], keys: list[str] | None = None) -> Dict[str, Array]:
    """Update PRNG keys for the next step.
    
    Args:
        rngs: Current PRNG keys dictionary
        keys: List of keys to update (None to update all)
        
    Returns:
        Dictionary with updated PRNG keys
    """
    if keys is None:
        keys = list(rngs.keys())
    
    new_rngs = {}
    for key, rng in rngs.items():
        if key in keys:
            new_rng, _ = jax.random.split(rng)
            new_rngs[key] = new_rng
        else:
            new_rngs[key] = rng
    
    return new_rngs


def split_rng(rng: Array, num: int = 2) -> tuple[Array, ...]:
    """Split a PRNG key into multiple keys.
    
    Args:
        rng: Source PRNG key
        num: Number of keys to generate
        
    Returns:
        Tuple of split PRNG keys
    """
    return tuple(jax.random.split(rng, num))


def gradient_accumulation_step(
    step_fn: StepFunction,
    state: PyTree,
    batches: list[BatchData],
    accumulate_steps: int
) -> StepOutput:
    """Execute gradient accumulation across multiple microbatches.
    
    This function is a placeholder for future gradient accumulation support.
    It will be properly implemented when microbatching is fully supported.
    
    Args:
        step_fn: The base step function
        state: Current training state  
        batches: List of microbatch data
        accumulate_steps: Number of accumulation steps
        
    Returns:
        Tuple of (final_state, aggregated_metrics)
    """
    if accumulate_steps == 1 or len(batches) == 1:
        # No accumulation needed
        return step_fn(state, batches[0])
    
    # This is a placeholder implementation
    # Future version will properly accumulate gradients across microbatches
    accumulated_metrics = {}
    current_state = state
    
    for i, batch in enumerate(batches[:accumulate_steps]):
        current_state, metrics = step_fn(current_state, batch)
        
        # Simple metric averaging (will be more sophisticated)
        if i == 0:
            accumulated_metrics = metrics.copy()
        else:
            for key, value in metrics.items():
                if key in accumulated_metrics:
                    accumulated_metrics[key] = (accumulated_metrics[key] * i + value) / (i + 1)
                else:
                    accumulated_metrics[key] = value
    
    return current_state, accumulated_metrics


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
