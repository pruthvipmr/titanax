"""Titanax runtime and control plane components.

This package provides utilities for distributed initialization, mesh creation,
and process group management in JAX environments.
"""

from .init import (
    detect_distributed_env,
    is_distributed_env,
    initialize_distributed,
    enumerate_devices,
    get_device_info,
    validate_device_availability,
    auto_initialize,
)

from .mesh import MeshSpec

from .process_groups import ProcessGroups

__all__ = [
    "detect_distributed_env",
    "is_distributed_env",
    "initialize_distributed",
    "enumerate_devices",
    "get_device_info",
    "validate_device_availability",
    "auto_initialize",
    "MeshSpec",
    "ProcessGroups",
]
