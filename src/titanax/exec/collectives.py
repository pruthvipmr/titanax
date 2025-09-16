"""Collective operations with explicit axis validation.

This module exposes a small namespace that mirrors common JAX collectives while
enforcing Titanax semantics:

- Every collective validates that the requested axis exists in the active mesh
  context before calling into JAX.
- All operations accept PyTrees and apply the primitive leaf-wise with
  consistent error conversion.
- Users must wrap collective invocations in ``with collectives.mesh_context(mesh):``
  (typically handled by the Engine) so that nested JITs still have access to the
  mesh topology for validation.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from typing import Callable, Iterator, Optional, Sequence, Tuple, TypeVar

import jax
import jax.numpy as jnp

from ..compat import (
    all_gather as lax_all_gather,
    all_to_all as lax_all_to_all,
    axis_index as lax_axis_index,
    pmean as lax_pmean,
    ppermute as lax_ppermute,
    psum as lax_psum,
    psum_scatter as lax_psum_scatter,
    tree_flatten as compat_tree_flatten,
    tree_structure as compat_tree_structure,
    tree_unflatten as compat_tree_unflatten,
)
from ..exceptions import CollectiveError, collective_error
from ..types import Array, AxisName, Mesh, PyTree


# Resolve tree utility shims once so we do not repeatedly branch later.
_tree_flatten = compat_tree_flatten or jax.tree_util.tree_flatten
_tree_structure = compat_tree_structure or jax.tree_util.tree_structure
_tree_unflatten = compat_tree_unflatten or jax.tree_util.tree_unflatten


# Thread-local storage so nested JITs/pmaps can still consult the mesh metadata
# without capturing it into the compiled program.
_thread_local = threading.local()


def set_current_mesh(mesh: Optional[Mesh]) -> None:
    """Set the active mesh for collective validation on the current thread."""

    _thread_local.current_mesh = mesh


def get_current_mesh() -> Optional[Mesh]:
    """Return the mesh registered for collective validation on this thread."""

    return getattr(_thread_local, "current_mesh", None)


@contextmanager
def mesh_context(mesh: Optional[Mesh]) -> Iterator[Optional[Mesh]]:
    """Temporarily install ``mesh`` as the current validation context.

    This helper is intended for the execution engine which binds the runtime
    mesh around user-specified step functions. Nested contexts are supported
    (the previous mesh is restored on exit).
    """

    previous = get_current_mesh()
    set_current_mesh(mesh)
    try:
        yield mesh
    finally:
        set_current_mesh(previous)


def _ensure_mesh(operation: str, axis: AxisName) -> Mesh:
    """Return the active mesh or raise a helpful error if none is configured."""

    mesh = get_current_mesh()
    if mesh is None:
        raise collective_error(
            operation,
            axis,
            "no active mesh context. Wrap the collective in `with collectives.mesh_context(mesh):`",
        )
    return mesh


def _validate_axis_name(operation: str, axis: AxisName) -> Mesh:
    """Validate that ``axis`` is a named mesh dimension available at runtime."""

    if not isinstance(axis, str):
        raise collective_error(
            operation,
            str(axis),
            f"axis must be provided as a string, received {type(axis)}",
        )

    if not axis:
        raise collective_error(operation, axis, "axis name cannot be empty")

    mesh = _ensure_mesh(operation, axis)

    if axis not in mesh.axis_names:
        available = ", ".join(f"'{name}'" for name in mesh.axis_names)
        raise collective_error(
            operation,
            axis,
            f"axis not found in mesh. Available axes: [{available}]",
        )

    return mesh


def _flatten_tree(
    operation: str, axis: AxisName, tree: PyTree
) -> Tuple[list[Array], jax.tree_util.PyTreeDef]:
    """Flatten ``tree`` to leaves and validate each leaf is a JAX array."""

    try:
        _tree_structure(tree)  # Validates PyTree compatibility
        leaves, treedef = _tree_flatten(tree)
    except Exception as exc:  # pragma: no cover - tree errors routed uniformly
        raise collective_error(
            operation,
            axis,
            f"invalid PyTree structure: {exc}",
        ) from exc

    for idx, leaf in enumerate(leaves):
        if not isinstance(leaf, Array):
            raise collective_error(
                operation,
                axis,
                f"leaf {idx} is not a JAX array (received {type(leaf)})",
            )

    return leaves, treedef


def _handle_jax_error(
    operation: str, axis: AxisName, error: Exception
) -> CollectiveError:
    """Convert arbitrary JAX exceptions into a :class:`CollectiveError`."""

    if isinstance(error, CollectiveError):
        return error

    message = str(error)
    lowered = message.lower()
    if "unbound axis name" in lowered or "axis name not found" in lowered:
        friendly = (
            f"Axis '{axis}' is not bound in the current JAX transformation. "
            "Ensure the collective is invoked inside a step function that binds the mesh."
        )
    else:
        friendly = f"JAX operation failed: {message}"

    return collective_error(operation, axis, friendly)


T = TypeVar("T")


def _invoke_collective(
    operation: str,
    axis: AxisName,
    func: Optional[Callable[..., T]],
    *args,
    **kwargs,
) -> T:
    """Call ``func`` while converting missing primitive or JAX failures."""

    if func is None:
        raise collective_error(
            operation,
            axis,
            "JAX installation does not expose this collective primitive",
        )

    try:
        return func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - exercised in tests but defensive
        raise _handle_jax_error(operation, axis, exc) from exc


def _map_collective(
    operation: str,
    axis: AxisName,
    tree: PyTree,
    leaf_op: Callable[[Array], Array],
) -> PyTree:
    """Apply ``leaf_op`` to every leaf in ``tree`` and rebuild the PyTree."""

    leaves, treedef = _flatten_tree(operation, axis, tree)
    results = []
    for leaf in leaves:
        try:
            results.append(leaf_op(leaf))
        except Exception as exc:  # pragma: no cover - handled via _handle_jax_error
            raise _handle_jax_error(operation, axis, exc) from exc
    return _tree_unflatten(treedef, results)


class collectives:
    """Namespace providing axis-validated collective operations."""

    mesh_context = staticmethod(mesh_context)

    @staticmethod
    def psum(tree: PyTree, axis: AxisName) -> PyTree:
        """Sum ``tree`` across ``axis`` of the active mesh.

        The result preserves the PyTree structure while summing each leaf across
        the specified mesh axis. Shapes are unchanged.
        """

        mesh = _validate_axis_name("psum", axis)
        try:
            return _map_collective(
                "psum",
                axis,
                tree,
                lambda leaf: _invoke_collective(
                    "psum", axis, lax_psum, leaf, axis_name=axis
                ),
            )
        except CollectiveError as exc:
            if int(mesh.shape[axis]) == 1 and "not bound" in str(exc).lower():
                return tree
            raise

    @staticmethod
    def pmean(tree: PyTree, axis: AxisName) -> PyTree:
        """Compute the mean of ``tree`` across ``axis`` of the active mesh."""

        mesh = _validate_axis_name("pmean", axis)
        try:
            return _map_collective(
                "pmean",
                axis,
                tree,
                lambda leaf: _invoke_collective(
                    "pmean", axis, lax_pmean, leaf, axis_name=axis
                ),
            )
        except CollectiveError as exc:
            if int(mesh.shape[axis]) == 1 and "not bound" in str(exc).lower():
                return tree
            raise

    @staticmethod
    def all_gather(tree: PyTree, axis: AxisName) -> PyTree:
        """Gather leaves from every device along ``axis``.

        Each leaf gains a new leading dimension equal to ``mesh.shape[axis]``
        containing the per-device values.
        """

        mesh = _validate_axis_name("all_gather", axis)
        try:
            return _map_collective(
                "all_gather",
                axis,
                tree,
                lambda leaf: _invoke_collective(
                    "all_gather",
                    axis,
                    lax_all_gather,
                    leaf,
                    axis_name=axis,
                    tiled=True,
                ),
            )
        except CollectiveError as exc:
            if int(mesh.shape[axis]) == 1 and "not bound" in str(exc).lower():
                return _map_collective(
                    "all_gather",
                    axis,
                    tree,
                    lambda leaf: jnp.expand_dims(leaf, 0),
                )
            raise

    @staticmethod
    def reduce_scatter(tree: PyTree, axis: AxisName) -> PyTree:
        """Sum ``tree`` across ``axis`` and scatter equal shards back.

        Equivalent to ``psum`` followed by a split along the leading dimension.
        The resulting leading dimension is divided by the mesh size for ``axis``.
        """

        mesh = _validate_axis_name("reduce_scatter", axis)
        try:
            return _map_collective(
                "reduce_scatter",
                axis,
                tree,
                lambda leaf: _invoke_collective(
                    "reduce_scatter",
                    axis,
                    lax_psum_scatter,
                    leaf,
                    axis_name=axis,
                    tiled=True,
                ),
            )
        except CollectiveError as exc:
            if int(mesh.shape[axis]) == 1 and "not bound" in str(exc).lower():
                return tree
            raise

    @staticmethod
    def broadcast(tree: PyTree, axis: AxisName, src_index: int = 0) -> PyTree:
        """Broadcast leaves from ``src_index`` along ``axis`` to all devices."""

        mesh = _validate_axis_name("broadcast", axis)
        if src_index < 0:
            raise collective_error(
                "broadcast",
                axis,
                f"src_index {src_index} must be non-negative",
            )

        axis_size = int(mesh.shape[axis])
        if src_index >= axis_size:
            raise collective_error(
                "broadcast",
                axis,
                f"src_index {src_index} must be < axis size {axis_size}",
            )

        permutation: Sequence[Tuple[int, int]] = tuple(
            (src_index, idx) for idx in range(axis_size)
        )

        def _broadcast_leaf(leaf: Array) -> Array:
            return _invoke_collective(
                "broadcast",
                axis,
                lax_ppermute,
                leaf,
                axis_name=axis,
                perm=permutation,
            )

        try:
            return _map_collective("broadcast", axis, tree, _broadcast_leaf)
        except CollectiveError as exc:
            if axis_size == 1 and "not bound" in str(exc).lower():
                return tree
            raise

    @staticmethod
    def all_to_all(
        tree: PyTree,
        axis: AxisName,
        *,
        split_axis: int = 0,
        concat_axis: int = 0,
    ) -> PyTree:
        """Exchange shards of each leaf across ``axis``.

        ``split_axis`` identifies the dimension on every leaf that should be
        partitioned across devices. ``concat_axis`` indicates the position where
        received shards should be concatenated.
        """

        mesh = _validate_axis_name("all_to_all", axis)

        def _all_to_all_leaf(leaf: Array) -> Array:
            return _invoke_collective(
                "all_to_all",
                axis,
                lax_all_to_all,
                leaf,
                axis_name=axis,
                split_axis=split_axis,
                concat_axis=concat_axis,
            )

        try:
            return _map_collective("all_to_all", axis, tree, _all_to_all_leaf)
        except CollectiveError as exc:
            if int(mesh.shape[axis]) == 1 and "not bound" in str(exc).lower():
                return tree
            raise

    @staticmethod
    def axis_index(axis: AxisName) -> int:
        """Return the caller's index along ``axis`` in the active mesh."""

        mesh = _validate_axis_name("axis_index", axis)
        try:
            return _invoke_collective("axis_index", axis, lax_axis_index, axis)
        except CollectiveError as exc:
            if int(mesh.shape[axis]) == 1 and "not bound" in str(exc).lower():
                return 0
            raise

    @staticmethod
    def ppermute(
        tree: PyTree, axis: AxisName, perm: Sequence[Tuple[int, int]]
    ) -> PyTree:
        """Leaf-wise wrapper around ``lax.ppermute`` with validation."""

        mesh = _validate_axis_name("ppermute", axis)

        if not isinstance(perm, (list, tuple)):
            raise collective_error(
                "ppermute",
                axis,
                f"perm must be a sequence of (src, dst) pairs, received {type(perm)}",
            )

        axis_size = int(mesh.shape[axis])
        for idx, pair in enumerate(perm):
            if not (isinstance(pair, (list, tuple)) and len(pair) == 2):
                raise collective_error(
                    "ppermute",
                    axis,
                    f"perm[{idx}] must be a (src, dst) pair",
                )
            src, dst = pair
            if not isinstance(src, int) or not isinstance(dst, int):
                raise collective_error(
                    "ppermute",
                    axis,
                    f"perm[{idx}] values must be ints (received {type(src)}, {type(dst)})",
                )
            if src < 0 or dst < 0 or src >= axis_size or dst >= axis_size:
                raise collective_error(
                    "ppermute",
                    axis,
                    f"perm[{idx}] indices must be within [0, {axis_size})",
                )

        return _map_collective(
            "ppermute",
            axis,
            tree,
            lambda leaf: _invoke_collective(
                "ppermute",
                axis,
                lax_ppermute,
                leaf,
                axis_name=axis,
                perm=tuple((int(src), int(dst)) for src, dst in perm),
            ),
        )


__all__ = [
    "collectives",
    "mesh_context",
    "set_current_mesh",
    "get_current_mesh",
]
