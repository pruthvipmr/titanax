"""Common type definitions for Titanax.

This module provides type aliases and protocols used throughout the Titanax framework.
"""

from typing import Any, Dict, Tuple, Union, Protocol, runtime_checkable
import jax

# JAX/PyTree type aliases
PyTree = Any  # JAX PyTree - nested structure of arrays
Array = jax.Array  # JAX array type
ArrayTree = PyTree  # PyTree containing JAX arrays

# Shape and axis types
Shape = Tuple[int, ...]
Axis = Union[str, int]
AxisName = str
PartitionSpec = jax.sharding.PartitionSpec

# Device and sharding types
Device = jax.Device
Mesh = jax.sharding.Mesh
Sharding = jax.sharding.Sharding
NamedSharding = jax.sharding.NamedSharding

# Parameter and gradient types
Params = PyTree
Grads = PyTree
OptState = PyTree
BatchData = Dict[str, Array]

# Configuration types
ConfigDict = Dict[str, Any]
PathPattern = str

# Step function types
StepOutput = Tuple[PyTree, Dict[str, Any]]  # (new_state, metrics)

# Logging types
LogValue = Union[float, int, str]
LogDict = Dict[str, LogValue]

# Process group types
ProcessRank = int
WorldSize = int


@runtime_checkable
class StepFunction(Protocol):
    """Protocol for training step functions."""

    def __call__(self, state: PyTree, batch: BatchData) -> StepOutput:
        """Execute one training step.

        Args:
            state: Current training state (TrainState)
            batch: Input batch data

        Returns:
            Tuple of (updated_state, metrics_dict)
        """
        ...


@runtime_checkable
class Logger(Protocol):
    """Protocol for logging implementations."""

    def log_scalar(self, name: str, value: LogValue, step: int) -> None:
        """Log a scalar value."""
        ...

    def log_dict(self, metrics: LogDict, step: int) -> None:
        """Log a dictionary of metrics."""
        ...


@runtime_checkable
class CheckpointStrategy(Protocol):
    """Protocol for checkpoint implementations."""

    def save(self, state: PyTree, step: int) -> None:
        """Save training state to checkpoint."""
        ...

    def load(self, step: int | None = None) -> PyTree:
        """Load training state from checkpoint."""
        ...

    def restore(self, state: PyTree, step: int | None = None) -> PyTree:
        """Restore training state from checkpoint."""
        ...
