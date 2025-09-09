"""Multi-device PRNG utilities for deterministic per-device random keys.

This module provides proper PRNG key management for multi-device scenarios,
ensuring each device gets unique but deterministic RNG streams that comply
with JAX multi-device best practices.
"""

from typing import Dict, Optional, Union
import jax
import jax.numpy as jnp
from ..types import Array


def create_per_device_rngs(
    base_key: Union[int, Array],
    named_keys: Optional[Dict[str, None]] = None,
    axis: str = "batch",
) -> Dict[str, Array]:
    """Create per-device PRNG keys using axis_index for deterministic streams.

    This function creates unique PRNG keys for each device in a multi-device setup.
    Each device gets deterministic but unique RNG streams by folding the device
    index into the base key.

    Args:
        base_key: Base PRNG key (int seed or JAX PRNGKey)
        named_keys: Dictionary of named keys to create (defaults to {'dropout': None})
        axis: Mesh axis name to use for device indexing (default: 'batch')

    Returns:
        Dictionary mapping key names to per-device PRNG keys

    Example:
        ```python
        # Inside a jitted function with mesh context
        rngs = create_per_device_rngs(42, {'dropout': None, 'params': None})
        # Each device will have unique but deterministic keys
        ```
    """
    if named_keys is None:
        named_keys = {"dropout": None}

    # Convert base_key to JAX PRNGKey if it's an integer
    if isinstance(base_key, int):
        base_key = jax.random.PRNGKey(base_key)

    # Create per-device keys
    per_device_rngs = {}

    # Split the base key for each named key
    keys = jax.random.split(base_key, len(named_keys))

    for i, key_name in enumerate(named_keys.keys()):
        # Create per-device key using axis_index and fold_in
        device_key = jax.random.fold_in(keys[i], jax.lax.axis_index(axis))
        per_device_rngs[key_name] = device_key

    return per_device_rngs


def update_per_device_rngs(
    current_rngs: Dict[str, Array],
    keys_to_update: Optional[list[str]] = None,
    axis: str = "batch",
) -> Dict[str, Array]:
    """Update per-device PRNG keys maintaining device uniqueness.

    This function updates PRNG keys while ensuring each device maintains
    unique RNG streams. It uses the current step and device index to
    create deterministic but unique updates.

    Args:
        current_rngs: Current PRNG keys dictionary
        keys_to_update: List of keys to update (None to update all)
        axis: Mesh axis name to use for device indexing

    Returns:
        Dictionary with updated per-device PRNG keys
    """
    if keys_to_update is None:
        keys_to_update = list(current_rngs.keys())

    updated_rngs = {}

    for key_name, current_key in current_rngs.items():
        if key_name in keys_to_update:
            # Split the current key and fold in device index for uniqueness
            new_key, _ = jax.random.split(current_key)
            device_unique_key = jax.random.fold_in(new_key, jax.lax.axis_index(axis))
            updated_rngs[key_name] = device_unique_key
        else:
            updated_rngs[key_name] = current_key

    return updated_rngs


def split_per_device_rng(
    rng: Array, num_splits: int = 2, axis: str = "batch"
) -> tuple[Array, ...]:
    """Split a per-device PRNG key while maintaining device uniqueness.

    Args:
        rng: Source per-device PRNG key
        num_splits: Number of keys to generate
        axis: Mesh axis name for device indexing

    Returns:
        Tuple of split per-device PRNG keys
    """
    # Split the key and ensure each split maintains device uniqueness
    split_keys = jax.random.split(rng, num_splits)

    # Fold in device index for each split to maintain uniqueness
    device_keys = []
    for i, key in enumerate(split_keys):
        # Fold in both the split index and device index for uniqueness
        unique_key = jax.random.fold_in(key, jax.lax.axis_index(axis) + i * 1000)
        device_keys.append(unique_key)

    return tuple(device_keys)


def validate_rng_keys(rngs: Dict[str, Array]) -> None:
    """Validate that RNG keys are proper JAX PRNGKeys.

    Args:
        rngs: Dictionary of PRNG keys to validate

    Raises:
        ValueError: If any key is not a valid JAX PRNGKey
    """
    for name, key in rngs.items():
        if not isinstance(key, jnp.ndarray):
            raise ValueError(f"RNG key '{name}' must be a JAX array, got {type(key)}")

        # Check shape - JAX PRNG keys should have specific shape
        if key.shape != (2,) and key.shape != (4,):
            raise ValueError(
                f"RNG key '{name}' has invalid shape {key.shape}. "
                f"Expected (2,) or (4,) for JAX PRNG keys"
            )

        # Check dtype - should be unsigned integer
        if not jnp.issubdtype(key.dtype, jnp.unsignedinteger):
            raise ValueError(
                f"RNG key '{name}' has invalid dtype {key.dtype}. "
                f"Expected unsigned integer dtype"
            )


def create_host_device_rngs(
    base_seed: int,
    named_keys: Optional[Dict[str, None]] = None,
    num_hosts: int = 1,
    num_devices_per_host: int = 1,
) -> Dict[str, Array]:
    """Create host-specific and device-specific PRNG keys for initialization.

    This is used during initialization before the mesh context is available.
    It creates keys that will be properly distributed across hosts and devices.

    Args:
        base_seed: Base random seed
        named_keys: Dictionary of named keys to create
        num_hosts: Number of hosts in the system
        num_devices_per_host: Number of devices per host

    Returns:
        Dictionary of PRNG keys suitable for multi-host/multi-device setup
    """
    if named_keys is None:
        named_keys = {"dropout": None}

    # Get host and device info
    host_id = jax.process_index()
    _ = jax.local_device_count()

    # Create host-specific base key
    base_key = jax.random.PRNGKey(base_seed)
    host_key = jax.random.fold_in(base_key, host_id)

    # Create named keys
    per_device_rngs = {}
    keys = jax.random.split(host_key, len(named_keys))

    for i, key_name in enumerate(named_keys.keys()):
        # Create device-specific key (will be further specialized in jitted functions)
        device_key = jax.random.fold_in(keys[i], jax.process_index())
        per_device_rngs[key_name] = device_key

    return per_device_rngs
