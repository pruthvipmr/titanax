"""Quick-start utilities for common Titanax workflows.

This module provides high-level convenience functions for setting up
common training configurations quickly.
"""

from typing import Optional, List, Union
import jax

from .runtime import MeshSpec, auto_initialize
from .parallel import Plan, DP
from .exec import Engine, Precision as _Precision
from .optim import adamw
from .logging import Basic
from .io import OrbaxCheckpoint
from .types import Logger, CheckpointStrategy
from .exceptions import ValidationError, mesh_validation_error, plan_validation_error


def simple_data_parallel(
    batch_size: int,
    learning_rate: float = 3e-4,
    precision: str = "bf16",
    checkpoint_dir: Optional[str] = None,
    loggers: Optional[List[Logger]] = None,
    optimizer=None,
    devices: Union[str, List[jax.Device], None] = "all",
) -> Engine:
    """Create a simple data parallel training setup.

    This is a convenience function for the most common case: data parallel
    training with sensible defaults.

    Args:
        batch_size: Global batch size (must be divisible by number of devices)
        learning_rate: Learning rate for optimizer
        precision: Precision mode ("fp32", "bf16", or "fp16")
        checkpoint_dir: Directory for checkpoints (None to disable)
        loggers: List of loggers (None for basic stdout logger)
        optimizer: Custom optimizer (None for AdamW)
        devices: Device specification ("all", device list, or None)

    Returns:
        Configured Engine ready for training

    Example:
        ```python
        engine = tx.quickstart.simple_data_parallel(
            batch_size=128,
            learning_rate=1e-3,
            precision="bf16",
            checkpoint_dir="./checkpoints"
        )
        ```

    Raises:
        ValidationError: If configuration parameters are invalid
    """
    # Validate inputs before any expensive operations
    _validate_data_parallel_config(
        batch_size=batch_size,
        learning_rate=learning_rate,
        precision=precision,
        loggers=loggers,
    )

    # Initialize distributed if needed
    auto_initialize()

    # Create mesh with data parallelism
    mesh = MeshSpec(devices=devices, axes=("data",))

    try:
        mesh.validate_compatibility(batch_size)
    except Exception as e:
        # Get the actual device count for error message
        devices = mesh._resolve_devices()
        raise mesh_validation_error(
            f"Batch size {batch_size} incompatible with mesh layout",
            f"Choose a batch size divisible by the number of devices ({len(devices)}). "
            f"Current devices: {len(devices)}",
        ) from e

    # Create data parallel plan
    try:
        plan = Plan(data_parallel=DP(axis="data"))
        plan.validate(mesh)
    except Exception as e:
        raise plan_validation_error(
            "Failed to create data parallel plan",
            "Ensure mesh has 'data' axis and proper device layout",
        ) from e

    # Set up precision
    if precision == "bf16":
        precision_config = _Precision(bfloat16=True)
    elif precision == "fp16":
        precision_config = _Precision(fp16=True)
    elif precision == "fp32":
        precision_config = _Precision()
    else:
        raise ValidationError(
            f"Invalid precision: {precision}", "Use one of: 'fp32', 'bf16', or 'fp16'"
        )

    # Set up optimizer
    if optimizer is None:
        optimizer = adamw(learning_rate)

    # Set up checkpoint
    checkpoint: Optional[CheckpointStrategy] = None
    if checkpoint_dir is not None:
        checkpoint = OrbaxCheckpoint(checkpoint_dir)

    # Set up loggers
    if loggers is None:
        loggers = [Basic()]

    return Engine(
        mesh=mesh,
        plan=plan,
        optimizer=optimizer,
        precision=precision_config,
        checkpoint=checkpoint,
        loggers=loggers,
    )


def simple_tensor_parallel(
    batch_size: int,
    model_parallel_size: int,
    sharding_rules: dict,
    learning_rate: float = 3e-4,
    precision: str = "bf16",
    checkpoint_dir: Optional[str] = None,
    loggers: Optional[List[Logger]] = None,
    devices: Union[str, List[jax.Device], None] = "all",
) -> Engine:
    """Create a simple tensor parallel training setup.

    Note: This is a stub implementation. Full tensor parallel support
    will be available in phase P1.

    Args:
        batch_size: Global batch size
        model_parallel_size: Size of model parallel dimension
        sharding_rules: Parameter sharding rules dict
        learning_rate: Learning rate for optimizer
        precision: Precision mode ("fp32", "bf16", or "fp16")
        checkpoint_dir: Directory for checkpoints (None to disable)
        loggers: List of loggers (None for basic stdout logger)
        devices: Device specification

    Returns:
        Configured Engine ready for tensor parallel training

    Raises:
        NotImplementedError: Always raised as TP is not yet implemented
    """
    raise NotImplementedError(
        "Tensor parallel support is not yet implemented",
        "Use examples/tp_minimal_mlp.py as a reference for manual TP setup, "
        "or use simple_data_parallel() for now. Full TP quickstart will be "
        "available in phase P1.",
    )


def validate_setup(engine: Engine) -> dict:
    """Validate an engine setup and return diagnostic information.

    Args:
        engine: Engine to validate

    Returns:
        Dictionary with validation results and diagnostics
    """
    # Handle device information safely
    mesh_devices = engine._mesh.devices
    device_count = (
        len(mesh_devices) if hasattr(mesh_devices, "__len__") else mesh_devices.size
    )

    diagnostics = {
        "mesh_info": mesh_devices,
        "plan_info": engine.plan.describe(),
        "device_count": device_count,
        "precision_config": {
            "bfloat16": engine.precision.bfloat16,
            "fp16": engine.precision.fp16,
            "loss_scaling": engine.precision.loss_scaling,
            "enable_x32_params": engine.precision.enable_x32_params,
        },
        "optimizer_type": type(engine.optimizer).__name__,
        "has_checkpoint": engine.checkpoint is not None,
        "logger_count": len(engine.loggers),
        "validation_status": "ok",
    }

    return diagnostics


def _validate_data_parallel_config(
    batch_size: int,
    learning_rate: float,
    precision: str,
    loggers: Optional[List[Logger]] = None,
) -> None:
    """Validate data parallel configuration parameters.

    This function performs fail-fast validation of user inputs before
    any expensive operations like mesh creation or device allocation.

    Args:
        batch_size: Global batch size
        learning_rate: Learning rate for optimizer
        precision: Precision mode string
        loggers: Optional list of loggers

    Raises:
        ValidationError: If any parameter is invalid
    """
    # Validate batch size
    if not isinstance(batch_size, int):
        raise ValidationError(
            f"batch_size must be an integer, got {type(batch_size).__name__}",
            "Pass an integer value for batch_size",
        )

    if batch_size <= 0:
        raise ValidationError(
            f"batch_size must be positive, got {batch_size}",
            "Use a positive batch size (e.g., 32, 64, 128)",
        )

    # Validate learning rate
    if not isinstance(learning_rate, (int, float)):
        raise ValidationError(
            f"learning_rate must be numeric, got {type(learning_rate).__name__}",
            "Pass a numeric value for learning_rate",
        )

    if learning_rate <= 0:
        raise ValidationError(
            f"learning_rate must be positive, got {learning_rate}",
            "Use a positive learning rate (e.g., 1e-4, 3e-4, 1e-3)",
        )

    # Validate precision
    if not isinstance(precision, str):
        raise ValidationError(
            f"precision must be a string, got {type(precision).__name__}",
            "Use one of: 'fp32', 'bf16', or 'fp16'",
        )

    valid_precisions = {"fp32", "bf16", "fp16"}
    if precision not in valid_precisions:
        raise ValidationError(
            f"Invalid precision '{precision}'",
            f"Use one of: {', '.join(sorted(valid_precisions))}",
        )

    # Validate loggers
    if loggers is not None:
        if not isinstance(loggers, (list, tuple)):
            raise ValidationError(
                f"loggers must be a list, got {type(loggers).__name__}",
                "Pass a list of Logger instances or None",
            )

        for i, logger in enumerate(loggers):
            # Check if logger has the expected interface (duck typing)
            if not hasattr(logger, "log"):
                raise ValidationError(
                    f"loggers[{i}] does not have a 'log' method",
                    "Ensure all loggers implement the Logger protocol with a log() method",
                )
