"""Titanax execution engine and collectives.

This package provides the execution engine, step function decoration,
and collective operations for distributed training.
"""

from .collectives import collectives, set_current_mesh, get_current_mesh

__all__ = [
    "collectives",
    "set_current_mesh", 
    "get_current_mesh",
]
