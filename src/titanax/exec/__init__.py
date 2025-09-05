"""Titanax execution engine and collectives.

This package provides the execution engine, step function decoration,
and collective operations for distributed training.
"""

from .collectives import collectives, set_current_mesh, get_current_mesh
from .engine import Engine, Precision, TrainState
from .step_fn import step_fn, update_rngs, split_rng, is_step_fn, get_original_step_fn

__all__ = [
    # Collectives
    "collectives",
    "set_current_mesh", 
    "get_current_mesh",
    # Engine components
    "Engine",
    "Precision", 
    "TrainState",
    # Step function utilities
    "step_fn",
    "update_rngs",
    "split_rng",
    "is_step_fn",
    "get_original_step_fn",
]
