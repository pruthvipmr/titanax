"""Titanax execution engine for distributed training.

This module provides the core Engine class, TrainState, and Precision configuration
for managing training loops with explicit parallelization.
"""

import dataclasses
from typing import Any, Dict, List, Optional, Union, Iterable

import jax
import jax.numpy as jnp
from jax import tree_util

from ..types import (
    PyTree, Array, BatchData, LogDict, StepFunction, Logger, CheckpointStrategy,
    Params, OptState
)
from ..runtime.mesh import MeshSpec
from ..parallel.plan import Plan
from ..exceptions import EngineError
from .collectives import set_current_mesh


@dataclasses.dataclass(frozen=True)
class Precision:
    """Precision policy configuration for training.
    
    Controls numeric precision, mixed precision training, and loss scaling.
    
    Args:
        bfloat16: Use bfloat16 for activations and gradients
        fp16: Use float16 for activations and gradients (alternative to bfloat16)
        loss_scaling: Enable automatic loss scaling for fp16 training
        enable_x32_params: Keep parameters in float32 while using lower precision for activations
    """
    bfloat16: bool = False
    fp16: bool = False
    loss_scaling: bool = False
    enable_x32_params: bool = False
    
    def __post_init__(self) -> None:
        """Validate precision configuration."""
        if self.bfloat16 and self.fp16:
            raise EngineError("Cannot enable both bfloat16 and fp16 precision")
        
        if self.loss_scaling and not self.fp16:
            raise EngineError("Loss scaling requires fp16=True")
    
    @property
    def dtype(self) -> jnp.dtype:
        """Get the target dtype for activations."""
        if self.bfloat16:
            return jnp.bfloat16
        elif self.fp16:
            return jnp.float16
        else:
            return jnp.float32
    
    @property
    def param_dtype(self) -> jnp.dtype:
        """Get the target dtype for parameters."""
        if self.enable_x32_params:
            return jnp.float32
        return self.dtype
    
    def describe(self) -> str:
        """Return a human-readable description of the precision policy."""
        parts = []
        if self.bfloat16:
            parts.append("bfloat16")
        elif self.fp16:
            parts.append("float16")
        else:
            parts.append("float32")
            
        if self.enable_x32_params:
            parts.append("x32_params")
        if self.loss_scaling:
            parts.append("loss_scaling")
            
        return " + ".join(parts)


@dataclasses.dataclass
class TrainState:
    """Training state containing parameters, optimizer state, and metadata.
    
    This class holds all the mutable state needed for training, including
    model parameters, optimizer state, current step count, and PRNG keys.
    
    Args:
        params: Model parameters as a PyTree
        opt_state: Optimizer state as a PyTree
        step: Current training step number
        rngs: Dictionary of named PRNG keys for different uses
    """
    params: Params
    opt_state: OptState
    step: int
    rngs: Dict[str, Array]
    
    def apply_gradients(self, *, grads: PyTree, **kwargs) -> 'TrainState':
        """Apply gradients and return new training state.
        
        This is a placeholder method that will be properly implemented
        when the optimizer integration is added.
        
        Args:
            grads: Gradients to apply
            **kwargs: Additional arguments for optimizer
            
        Returns:
            New TrainState with updated parameters and optimizer state
        """
        # This is a placeholder - will be implemented with optimizer integration
        return dataclasses.replace(
            self,
            step=self.step + 1,
            **kwargs
        )
    
    def replace(self, **kwargs) -> 'TrainState':
        """Return a copy of the state with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)
    
    def tree_flatten(self):
        """Flatten TrainState for JAX PyTree operations."""
        children = (self.params, self.opt_state, self.rngs)
        aux_data = (self.step,)
        return children, aux_data
    
    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten TrainState from JAX PyTree operations."""
        step, = aux_data
        params, opt_state, rngs = children
        return cls(params=params, opt_state=opt_state, step=step, rngs=rngs)


# Register TrainState as a JAX PyTree
tree_util.register_pytree_node(
    TrainState,
    TrainState.tree_flatten,
    TrainState.tree_unflatten
)


class Engine:
    """Core training engine for distributed JAX training.
    
    The Engine orchestrates training loops, manages state, handles checkpointing,
    and coordinates logging across distributed devices.
    
    Args:
        mesh: Mesh specification for device layout
        plan: Parallel plan (DP/TP/PP configuration)
        optimizer: Optimizer instance (placeholder for now)
        precision: Precision policy configuration
        checkpoint: Checkpoint strategy for save/load operations
        loggers: List of logger instances for metrics
    """
    
    def __init__(
        self,
        mesh: MeshSpec,
        plan: Plan,
        optimizer: Any,  # Will be typed properly with optimizer integration
        precision: Precision = Precision(),
        checkpoint: Optional[CheckpointStrategy] = None,
        loggers: Optional[List[Logger]] = None,
    ):
        self.mesh_spec = mesh
        self.plan = plan
        self.optimizer = optimizer
        self.precision = precision
        self.checkpoint = checkpoint
        self.loggers = loggers or []
        
        # Build and validate the mesh
        self._mesh = self._validate_and_build_mesh()
        
        # Set the mesh in the collectives context
        set_current_mesh(self._mesh)
        
        # Validate plan compatibility with mesh
        self._validate_plan()
        
        # Initialize compiled step function (will be set when step_fn is registered)
        self._compiled_step_fn: Optional[StepFunction] = None
    
    def _validate_and_build_mesh(self) -> jax.sharding.Mesh:
        """Validate and build the JAX mesh from specification."""
        try:
            mesh = self.mesh_spec.build()
            return mesh
        except Exception as e:
            raise EngineError(
                f"Failed to build mesh from specification: {e}",
                suggestion="Check mesh axes and device availability"
            ) from e
    
    def _validate_plan(self) -> None:
        """Validate the parallel plan against the mesh specification."""
        try:
            self.plan.validate(self.mesh_spec)
        except Exception as e:
            raise EngineError(
                f"Plan validation failed: {e}",
                suggestion="Check that plan axes match mesh axes"
            ) from e
    
    def register_step_fn(self, step_fn: StepFunction) -> None:
        """Register and compile a step function for training.
        
        Args:
            step_fn: The step function to compile and use for training
        """
        # This will be properly implemented when step_fn decorator is ready
        self._compiled_step_fn = step_fn
    
    def create_state(
        self,
        params: Params,
        rngs: Optional[Dict[str, Array]] = None
    ) -> TrainState:
        """Create initial training state.
        
        Args:
            params: Initial model parameters
            rngs: Named PRNG keys, will create default 'dropout' key if None
            
        Returns:
            Initialized TrainState
        """
        if rngs is None:
            # Create default PRNG keys
            rng = jax.random.PRNGKey(42)
            rngs = {'dropout': rng}
        
        # Initialize optimizer state (placeholder)
        opt_state = params  # This will be replaced with proper optimizer init
        
        return TrainState(
            params=params,
            opt_state=opt_state,
            step=0,
            rngs=rngs
        )
    
    def fit(
        self,
        step_fn: StepFunction,
        data: Iterable[BatchData],
        steps: Optional[int] = None,
        state: Optional[TrainState] = None
    ) -> TrainState:
        """Run the training loop.
        
        Args:
            step_fn: The step function to execute each training step
            data: Iterable of training data batches
            steps: Maximum number of steps to train (None for unlimited)
            state: Initial training state (None to create from checkpoint or error)
            
        Returns:
            Final training state after training
        """
        if self._compiled_step_fn is None:
            self.register_step_fn(step_fn)
        
        if state is None:
            if self.checkpoint is not None:
                try:
                    state = self.checkpoint.load()
                    self._log_scalar("checkpoint/loaded_step", float(state.step), state.step)
                except Exception:
                    # No checkpoint available or failed to load
                    raise EngineError(
                        "No initial state provided and no checkpoint available",
                        suggestion="Either provide state parameter or ensure checkpoint can be loaded"
                    )
            else:
                raise EngineError(
                    "No initial state provided and no checkpoint strategy configured",
                    suggestion="Either provide state parameter or configure checkpoint strategy"
                )
        
        current_step = 0
        try:
            for batch in data:
                if steps is not None and current_step >= steps:
                    break
                
                # Execute one training step
                state, metrics = self._execute_step(state, batch)
                current_step += 1
                
                # Log metrics
                self._log_metrics(metrics, state.step)
                
                # Save checkpoint periodically (simplified logic)
                if self.checkpoint is not None and state.step % 1000 == 0:
                    self.checkpoint.save(state, state.step)
                    self._log_scalar("checkpoint/saved_step", float(state.step), state.step)
        
        except KeyboardInterrupt:
            self._log_scalar("training/interrupted", float(state.step), state.step)
            if self.checkpoint is not None:
                self.checkpoint.save(state, state.step)
                self._log_scalar("checkpoint/saved_step", float(state.step), state.step)
            raise
        except Exception as e:
            raise EngineError(f"Training failed at step {state.step}: {e}") from e
        
        # Final checkpoint save
        if self.checkpoint is not None:
            self.checkpoint.save(state, state.step)
            self._log_scalar("checkpoint/final_step", float(state.step), state.step)
        
        return state
    
    def _execute_step(self, state: TrainState, batch: BatchData) -> tuple[TrainState, LogDict]:
        """Execute a single training step.
        
        Args:
            state: Current training state
            batch: Input batch data
            
        Returns:
            Tuple of (updated_state, metrics)
        """
        if self._compiled_step_fn is None:
            raise EngineError("No step function registered")
        
        # Execute the step function
        new_state, metrics = self._compiled_step_fn(state, batch)
        
        return new_state, metrics
    
    def _log_metrics(self, metrics: LogDict, step: int) -> None:
        """Log metrics to all registered loggers."""
        for logger in self.loggers:
            try:
                logger.log_dict(metrics, step)
            except Exception as e:
                # Don't fail training due to logging errors
                print(f"Warning: Logger failed: {e}")
    
    def _log_scalar(self, name: str, value: float, step: int) -> None:
        """Log a scalar value to all registered loggers."""
        for logger in self.loggers:
            try:
                logger.log_scalar(name, value, step)
            except Exception as e:
                # Don't fail training due to logging errors
                print(f"Warning: Logger failed: {e}")
    
    def describe(self) -> str:
        """Return a human-readable description of the engine configuration."""
        lines = [
            "Titanax Engine Configuration:",
            f"  Mesh: {self.mesh_spec.describe()}",
            f"  Plan: {self.plan.describe()}",
            f"  Precision: {self.precision.describe()}",
            f"  Checkpoint: {'enabled' if self.checkpoint else 'disabled'}",
            f"  Loggers: {len(self.loggers)} configured",
        ]
        return "\n".join(lines)
