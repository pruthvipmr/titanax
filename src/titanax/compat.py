"""Compatibility shims for different JAX versions.

This module provides stable imports for JAX APIs that have moved between versions,
allowing Titanax to work across multiple JAX releases.
"""

from typing import Any, Callable, Optional
import jax
import warnings


# =============================================================================
# pjit compatibility
# =============================================================================

pjit: Optional[Callable[..., Any]] = None

try:
    # JAX >= 0.4.14: pjit is in the main module
    from jax import pjit  # type: ignore
except ImportError:
    try:
        # JAX < 0.4.14: pjit is in jax.experimental
        from jax.experimental.pjit import pjit  # type: ignore
    except ImportError:
        # Very old JAX versions might not have pjit
        pjit = None


# =============================================================================
# shard_map compatibility
# =============================================================================

shard_map: Optional[Callable[..., Any]] = None

try:
    from jax.experimental.shard_map import shard_map  # type: ignore
except ImportError:
    try:
        # Alternative import location in some versions
        from jax.sharding.shard_map import shard_map  # type: ignore
    except ImportError:
        # Fallback if shard_map is not available
        shard_map = None


# =============================================================================
# Collective operations compatibility
# =============================================================================

# Core collective operations from lax
psum: Optional[Callable[..., Any]] = None
pmean: Optional[Callable[..., Any]] = None
pmax: Optional[Callable[..., Any]] = None
pmin: Optional[Callable[..., Any]] = None
all_gather: Optional[Callable[..., Any]] = None
reduce_scatter_p: Optional[Callable[..., Any]] = None
psum_scatter: Optional[Callable[..., Any]] = None

try:
    from jax.lax import psum, pmean, pmax, pmin, all_gather, psum_scatter  # type: ignore

    try:
        from jax.lax import reduce_scatter_p  # type: ignore
    except ImportError:
        reduce_scatter_p = None
except ImportError:
    # Fallback imports if some collectives are missing
    try:
        from jax.lax import psum, pmean  # type: ignore
    except ImportError:
        psum = None
        pmean = None
    try:
        from jax.lax import psum_scatter  # type: ignore
    except ImportError:
        psum_scatter = None

# Advanced collective operations
ppermute: Optional[Callable[..., Any]] = None
pshuffle: Optional[Callable[..., Any]] = None
axis_index: Optional[Callable[..., Any]] = None

try:
    from jax.lax import ppermute, pshuffle, axis_index  # type: ignore
except ImportError:
    pass


# =============================================================================
# Sharding compatibility
# =============================================================================

Mesh: Optional[Any] = None
PartitionSpec: Optional[Any] = None
NamedSharding: Optional[Any] = None
sharding_module: Optional[Any] = None

try:
    from jax.sharding import Mesh, PartitionSpec, NamedSharding  # type: ignore
    from jax import sharding as sharding_module  # type: ignore
except ImportError:
    try:
        # Older JAX versions
        from jax.experimental.sharding import Mesh, PartitionSpec, NamedSharding  # type: ignore
        from jax.experimental import sharding as sharding_module  # type: ignore
    except ImportError:
        # Very old versions - no sharding support
        pass


# =============================================================================
# Tree utilities compatibility
# =============================================================================

tree_map: Optional[Callable[..., Any]] = None
tree_flatten: Optional[Callable[..., Any]] = None
tree_unflatten: Optional[Callable[..., Any]] = None
tree_structure: Optional[Callable[..., Any]] = None
tree_leaves: Optional[Callable[..., Any]] = None

try:
    # New tree API (JAX >= 0.4.20)
    from jax import tree_util  # type: ignore

    tree_map = tree_util.tree_map
    tree_flatten = tree_util.tree_flatten
    tree_unflatten = tree_util.tree_unflatten
    tree_structure = tree_util.tree_structure
    tree_leaves = tree_util.tree_leaves
except ImportError:
    try:
        # Fallback to older API
        from jax.tree_util import tree_map, tree_flatten, tree_unflatten, tree_structure, tree_leaves  # type: ignore
    except ImportError:
        # Very old versions
        pass


# =============================================================================
# Version compatibility utilities
# =============================================================================


def get_jax_version() -> str:
    """Get the current JAX version string."""
    return jax.__version__


def check_jax_compatibility() -> bool:
    """Check if the current JAX version is compatible with Titanax.

    Returns:
        True if JAX version is supported, False otherwise
    """
    try:
        version = get_jax_version()
        # Basic compatibility check - we need pjit or shard_map
        if pjit is None and shard_map is None:
            warnings.warn(
                f"JAX {version} does not provide pjit or shard_map. "
                "Please upgrade to a newer JAX version (>= 0.4.0) for full Titanax functionality.",
                UserWarning,
            )
            return False

        if Mesh is None or PartitionSpec is None:
            warnings.warn(
                f"JAX {version} does not provide sharding APIs. "
                "Please upgrade to a newer JAX version (>= 0.4.0) for full Titanax functionality.",
                UserWarning,
            )
            return False

        return True
    except Exception:
        return False


# =============================================================================
# Preferred API selection
# =============================================================================


def get_preferred_pjit() -> Optional[Callable[..., Any]]:
    """Get the preferred pjit function for the current JAX version."""
    if pjit is not None:
        return pjit
    return None


def get_preferred_shard_map() -> Optional[Callable[..., Any]]:
    """Get the preferred shard_map function for the current JAX version."""
    if shard_map is not None:
        return shard_map
    return None


def has_collective_support() -> bool:
    """Check if the current JAX version has full collective operation support."""
    return all(
        [
            psum is not None,
            pmean is not None,
            all_gather is not None,
            ppermute is not None,
        ]
    )


# =============================================================================
# Initialization and compatibility check
# =============================================================================

# Run compatibility check on import
_compatibility_checked = False


def ensure_compatibility() -> None:
    """Ensure JAX compatibility, warn if issues found."""
    global _compatibility_checked
    if not _compatibility_checked:
        check_jax_compatibility()
        _compatibility_checked = True


# Check compatibility on first import
ensure_compatibility()


# =============================================================================
# Public API
# =============================================================================

__all__ = [
    # Core compilation functions
    "pjit",
    "shard_map",
    # Sharding utilities
    "Mesh",
    "PartitionSpec",
    "NamedSharding",
    "sharding_module",
    # Collective operations
    "psum",
    "pmean",
    "pmax",
    "pmin",
    "all_gather",
    "reduce_scatter_p",
    "psum_scatter",
    "ppermute",
    "pshuffle",
    "axis_index",
    # Tree utilities
    "tree_map",
    "tree_flatten",
    "tree_unflatten",
    "tree_structure",
    "tree_leaves",
    # Utility functions
    "get_jax_version",
    "check_jax_compatibility",
    "get_preferred_pjit",
    "get_preferred_shard_map",
    "has_collective_support",
    "ensure_compatibility",
]
