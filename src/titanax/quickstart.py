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


def simple_data_parallel(
    batch_size: int,
    learning_rate: float = 3e-4,
    precision: str = "bf16", 
    checkpoint_dir: Optional[str] = None,
    loggers: Optional[List[Logger]] = None,
    optimizer = None,
    devices: Union[str, List[jax.Device], None] = "all"
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
    """
    # Initialize distributed if needed
    auto_initialize()
    
    # Create mesh with data parallelism
    mesh = MeshSpec(devices=devices, axes=("data",))
    mesh.validate_batch_compatibility(batch_size)
    
    # Create data parallel plan
    plan = Plan(data_parallel=DP(axis="data"))
    
    # Set up precision
    if precision == "bf16":
        precision_config = _Precision(bfloat16=True)
    elif precision == "fp16":
        precision_config = _Precision(fp16=True)
    elif precision == "fp32":
        precision_config = _Precision()
    else:
        raise ValueError(f"Invalid precision: {precision}. Use 'fp32', 'bf16', or 'fp16'")
    
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
    devices: Union[str, List[jax.Device], None] = "all"
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
    """
    raise NotImplementedError(
        "Tensor parallel support is not yet implemented. "
        "It will be available in phase P1. "
        "For now, use simple_data_parallel()."
    )


def validate_setup(engine: Engine) -> dict:
    """Validate an engine setup and return diagnostic information.
    
    Args:
        engine: Engine to validate
        
    Returns:
        Dictionary with validation results and diagnostics
    """
    diagnostics = {
        "mesh_info": engine.mesh.describe(),
        "plan_info": engine.plan.describe(),
        "device_count": len(engine.mesh.build().devices),
        "precision_config": {
            "bfloat16": engine.precision.bfloat16,
            "fp16": engine.precision.fp16,
            "loss_scaling": engine.precision.loss_scaling,
            "enable_x32_params": engine.precision.enable_x32_params,
        },
        "optimizer_type": type(engine.optimizer).__name__,
        "has_checkpoint": engine.checkpoint is not None,
        "logger_count": len(engine.loggers),
        "validation_status": "ok"
    }
    
    return diagnostics
