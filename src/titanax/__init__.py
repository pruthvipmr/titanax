"""Titanax: Explicit-Parallel JAX Training Framework

Titanax is a lightweight JAX training framework that brings Hugging Face Accelerate/TorchTitan
ergonomics to JAX with explicit parallelization. Users must declare meshes, sharding rules,
and collectives - no XLA auto-sharding.

Key Features:
- Explicit data/tensor/pipeline parallelization
- Production scaffolding (checkpointing, logging, mixed precision)
- Composable parallel plans (DP, TP, PP)
- Multi-host training support
- JAX/Optax integration

Example Usage:
    ```python
    import titanax as tx

    # Create mesh and data parallel plan
    mesh = tx.MeshSpec(devices="all", axes=("data",))
    plan = tx.Plan(data_parallel=tx.DP(axis="data"))

    # Set up engine with optimizer and precision
    engine = tx.Engine(
        mesh=mesh,
        plan=plan,
        optimizer=tx.optim.adamw(3e-4),
        precision=tx.Precision(bfloat16=True),
        checkpoint=tx.OrbaxCheckpoint("ckpts/run1"),
        loggers=[tx.loggers.Basic()]
    )

    # Define training step
    @tx.step_fn
    def train_step(state, batch):
        def loss_fn(p):
            logits = model_apply(p, batch["x"])
            return cross_entropy(logits, batch["y"])
        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        grads = tx.collectives.psum(grads, axis="data")
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss}

    # Train the model
    engine.fit(train_step, data=train_data, steps=10_000)
    ```
"""

# Version and metadata
from ._version import (
    __version__,
    __author__,
    __author_email__ as __email__,
    __project_description__ as __description__,
    __homepage__ as __url__,
)

# Core runtime components
from .runtime import (
    MeshSpec,
    ProcessGroups,
    detect_distributed_env,
    is_distributed_env,
    initialize_distributed,
    auto_initialize,
)

# Parallel plans
from .parallel import (
    Plan,
    DP,
    TP,  # Note: stub implementation
    PP,  # Note: stub implementation
)

# Execution engine
from .exec import (
    Engine,
    TrainState,
    Precision,
    step_fn,
    collectives,
    update_rngs,
    split_rng,
    create_per_device_rngs,
    update_per_device_rngs,
    split_per_device_rng,
    validate_rng_keys,
    create_host_device_rngs,
)

# Type system and exceptions
from .types import (
    Array,
    PyTree,
    Mesh,
    PartitionSpec,
    NamedSharding,
    Logger,
    CheckpointStrategy,
    StepFunction,
)

from .exceptions import (
    TitanaxError,
    MeshError,
    PlanError,
    CollectiveError,
    EngineError,
    CheckpointError,
)

# Convenience namespace imports
from . import optim
from . import logging as loggers
from . import io
from . import quickstart

# Checkpointing shortcuts
from .io import (
    OrbaxCheckpoint,
    CheckpointMetadata,
)

# Public API - organized for user convenience
__all__ = [
    # Version info
    "__version__",
    "__author__",
    "__email__",
    "__description__",
    "__url__",
    # Core runtime
    "MeshSpec",
    "ProcessGroups",
    "detect_distributed_env",
    "is_distributed_env",
    "initialize_distributed",
    "auto_initialize",
    # Parallel plans
    "Plan",
    "DP",
    "TP",
    "PP",
    # Execution
    "Engine",
    "TrainState",
    "Precision",
    "step_fn",
    "collectives",
    # PRNG utilities
    "update_rngs",
    "split_rng",
    "create_per_device_rngs",
    "update_per_device_rngs",
    "split_per_device_rng",
    "validate_rng_keys",
    "create_host_device_rngs",
    # Type system
    "Array",
    "PyTree",
    "Mesh",
    "PartitionSpec",
    "NamedSharding",
    "Logger",
    "CheckpointStrategy",
    "StepFunction",
    # Exceptions
    "TitanaxError",
    "MeshError",
    "PlanError",
    "CollectiveError",
    "EngineError",
    "CheckpointError",
    # Namespaces
    "optim",
    "loggers",
    "io",
    "quickstart",
    # Checkpointing shortcuts
    "OrbaxCheckpoint",
    "CheckpointMetadata",
]


# Convenience functions for common precision configurations
def bfloat16_precision():
    """BFloat16 mixed precision configuration."""
    return Precision(bfloat16=True)


def float16_precision():
    """Float16 mixed precision configuration."""
    return Precision(fp16=True)


def float32_precision():
    """Full precision (Float32) configuration."""
    return Precision()


# Aliases for backward compatibility
bf16_precision = bfloat16_precision
fp16_precision = float16_precision
fp32_precision = float32_precision
