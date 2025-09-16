"""Multi-device PRNG utilities for deterministic per-device random keys."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Optional

import jax
import jax.numpy as jnp
from jax.sharding import Mesh

from ..types import Array


def _to_key(seed: int | Array) -> Array:
    """Normalize a seed into a JAX PRNG key."""

    if isinstance(seed, int):
        return jax.random.PRNGKey(seed)
    array = jnp.asarray(seed)
    if array.shape not in {(2,), (4,)}:
        raise ValueError(
            f"Seed must be a PRNG key shape (2,) or (4,), got {array.shape}. "
            "Fix: pass an integer seed or a valid key from jax.random.PRNGKey."
        )
    if not jnp.issubdtype(array.dtype, jnp.unsignedinteger):
        raise ValueError(
            f"Seed dtype must be unsigned integer, got {array.dtype}. "
            "Fix: use jax.random.PRNGKey to generate keys."
        )
    return array


def _mesh_shape(mesh: Mesh) -> tuple[int, ...]:
    """Return the mesh device shape as a tuple of ints."""

    return tuple(int(size) for size in mesh.devices.shape)


def _validate_per_device_array(name: str, value: Array) -> Array:
    array_value = jnp.asarray(value)
    if array_value.ndim < 1:
        raise ValueError(
            f"Per-device RNG '{name}' must have at least one dimension. "
            "Fix: ensure the value is shaped like mesh_shape + (2,)."
        )
    if array_value.shape[-1] not in (2, 4):
        raise ValueError(
            f"Per-device RNG '{name}' has invalid key width {array_value.shape[-1]}. "
            "Fix: PRNG keys should end with dimension 2 (uint32[2])."
        )
    if not jnp.issubdtype(array_value.dtype, jnp.unsignedinteger):
        raise ValueError(
            f"Per-device RNG '{name}' must use an unsigned integer dtype, got {array_value.dtype}."
        )
    return array_value


def _split_batched_keys(keys: Array, num: int) -> Array:
    flat_shape = (-1, keys.shape[-1])
    flat_keys = jnp.reshape(keys, flat_shape)

    splits = jax.vmap(lambda k: jax.random.split(k, num))(flat_keys)
    return jnp.reshape(splits, keys.shape[:-1] + (num, keys.shape[-1]))


def create_per_device_rngs(
    seed: int | Array,
    mesh: Mesh,
    *,
    names: Optional[Iterable[str]] = None,
) -> Dict[str, Array]:
    """Create deterministic per-device RNGs for each requested name.

    Args:
        seed: Integer seed or existing PRNG key.
        mesh: Mesh describing the device topology.
        names: Iterable of key names (defaults to ('dropout',)).

    Returns:
        Dictionary mapping each name to an array shaped like
        ``mesh.shape + (2,)`` holding device-specific RNG keys.
    """

    base_key = _to_key(seed)
    key_width = base_key.shape[-1]
    if names is None:
        names = ("dropout",)

    names = tuple(names)
    if not names:
        raise ValueError("names cannot be empty. Provide at least one RNG name.")

    mesh_shape = _mesh_shape(mesh)
    total_devices = mesh.devices.size

    keys = jax.random.split(base_key, total_devices * len(names))
    keys = keys.reshape((len(names),) + mesh_shape + (key_width,))

    rngs: Dict[str, Array] = {}
    for idx, name in enumerate(names):
        rngs[name] = _validate_per_device_array(name, keys[idx])

    return rngs


def update_per_device_rngs(rngs: Dict[str, Array]) -> Dict[str, Array]:
    """Split each per-device RNG once to advance the streams deterministically."""

    updated: Dict[str, Array] = {}
    for name, value in rngs.items():
        key_array = _validate_per_device_array(name, value)
        split_keys = _split_batched_keys(key_array, 2)
        # Take the second split to avoid reusing the original key.
        new_keys = split_keys[..., 1, :]
        updated[name] = _validate_per_device_array(name, new_keys)
    return updated


def split_per_device_rng(
    rngs: Dict[str, Array],
    num: int = 1,
) -> tuple[Dict[str, Array], ...]:
    """Split per-device RNGs ``num`` times.

    Returns a tuple containing ``num`` dictionaries of the same structure as
    ``rngs``. Each dictionary holds the per-device keys for the respective
    split index.
    """

    if num < 1:
        raise ValueError("num must be >= 1")

    splits = tuple({} for _ in range(num))  # type: ignore[var-annotated]

    for name, value in rngs.items():
        key_array = _validate_per_device_array(name, value)
        split_keys = _split_batched_keys(key_array, num)
        split_keys = jnp.moveaxis(split_keys, -2, 0)
        for idx in range(num):
            splits[idx][name] = _validate_per_device_array(name, split_keys[idx])

    return splits  # type: ignore[return-value]


def validate_rng_keys(rngs: Dict[str, Array]) -> None:
    """Validate that RNG keys are proper JAX PRNGKeys."""

    for name, key in rngs.items():
        if not isinstance(key, (jax.Array, jnp.ndarray)):
            raise ValueError(
                f"RNG key '{name}' must be a JAX array, got {type(key).__name__}."
            )
        key_array = jnp.asarray(key)
        if key_array.shape not in {(2,), (4,)}:
            raise ValueError(
                f"RNG key '{name}' has invalid shape {key_array.shape}. "
                "Expected (2,) or (4,) for JAX PRNG keys"
            )
        if not jnp.issubdtype(key_array.dtype, jnp.unsignedinteger):
            raise ValueError(
                f"RNG key '{name}' has invalid dtype {key_array.dtype}. "
                "Expected unsigned integer dtype"
            )


def create_host_device_rngs(
    base_seed: int,
    named_keys: Optional[Dict[str, None]] = None,
    num_hosts: int = 1,
    num_devices_per_host: int = 1,
) -> Dict[str, Array]:
    """Create host-specific and device-specific PRNG keys for initialization."""

    if named_keys is None:
        named_keys = {"dropout": None}

    host_id = jax.process_index()
    _ = jax.local_device_count()

    base_key = jax.random.PRNGKey(base_seed)
    host_key = jax.random.fold_in(base_key, host_id)

    per_device_rngs = {}
    keys = jax.random.split(host_key, len(named_keys))

    for i, key_name in enumerate(named_keys.keys()):
        device_key = jax.random.fold_in(keys[i], jax.process_index())
        per_device_rngs[key_name] = device_key

    return per_device_rngs
