"""Collective operations with explicit axis validation.

This module provides typed wrappers around JAX collective operations with
comprehensive axis validation and tree compatibility checking.

All collective operations properly handle mesh context and work within
JAX transformations like pjit and shard_map.
"""

from typing import Optional
import threading

import jax
from ..compat import psum, pmean, all_gather, ppermute, psum_scatter

from ..types import PyTree, Array, AxisName
from ..exceptions import CollectiveError, collective_error


# Thread-local storage for mesh context to avoid tracer capture
_thread_local = threading.local()


def set_current_mesh(mesh: Optional["Mesh"]) -> None:
    """Set the current mesh for collective validation.

    This is called by the execution engine to provide mesh context
    for collective operations validation. Uses thread-local storage
    to avoid issues with JAX tracers.

    Args:
        mesh: Mesh to set as current, or None to clear
    """
    _thread_local.current_mesh = mesh


def get_current_mesh() -> Optional["Mesh"]:
    """Get the current mesh for collective validation.

    Returns:
        Current mesh if set, None otherwise
    """
    return getattr(_thread_local, "current_mesh", None)


def _validate_axis_name(
    axis: AxisName, operation: str, mesh: Optional["Mesh"] = None
) -> None:
    """Validate axis name and optionally check mesh compatibility.

    Args:
        axis: Axis name to validate
        operation: Name of collective operation for error context
        mesh: Optional mesh to validate axis against

    Raises:
        CollectiveError: If axis validation fails
    """
    if not isinstance(axis, str):
        raise collective_error(
            operation, str(axis), f"axis must be a string, got {type(axis)}"
        )

    if not axis:
        raise collective_error(operation, axis, "axis name cannot be empty")

    # Check axis exists in mesh if provided
    if mesh is not None:
        if axis not in mesh.axis_names:
            available_axes = ", ".join(f"'{ax}'" for ax in mesh.axis_names)
            raise collective_error(
                operation,
                axis,
                f"axis not found in mesh. Available axes: [{available_axes}]",
            )


def _validate_tree_structure(tree: PyTree, operation: str, axis: AxisName) -> None:
    """Validate PyTree structure for collective operations.

    Args:
        tree: PyTree to validate
        operation: Name of collective operation
        axis: Axis name for error context

    Raises:
        CollectiveError: If tree validation fails
    """
    try:
        # Check if it's a valid PyTree
        jax.tree_util.tree_structure(tree)

        # Check that all leaves are arrays
        leaves, _ = jax.tree_util.tree_flatten(tree)
        for i, leaf in enumerate(leaves):
            if not isinstance(leaf, jax.Array):
                raise collective_error(
                    operation, axis, f"leaf {i} is not a JAX array (got {type(leaf)})"
                )

    except Exception as e:
        if isinstance(e, CollectiveError):
            raise
        raise collective_error(operation, axis, f"invalid PyTree structure: {str(e)}")


class collectives:
    """Namespace class for collective operations with validation."""

    @staticmethod
    def psum(tree: PyTree, axis: AxisName) -> PyTree:
        """Sum PyTree across specified mesh axis.

        Performs an all-reduce sum operation across the specified mesh axis.
        All participating processes contribute their local values to compute
        the global sum.

        This operation works within pjit/shard_map contexts where mesh
        axes are properly bound.

        Args:
            tree: PyTree to sum across axis
            axis: Mesh axis name to sum over

        Returns:
            PyTree with same structure, values summed across axis

        Raises:
            CollectiveError: If axis name is invalid or tree is invalid

        Example:
            >>> # With mesh axis "data" of size 4:
            >>> local_grads = {"w": jnp.array([1.0, 2.0])}
            >>> global_grads = tx.collectives.psum(local_grads, axis="data")
            >>> # Result: {"w": jnp.array([4.0, 8.0])} (sum across 4 devices)
        """
        _validate_axis_name(axis, "psum", get_current_mesh())
        _validate_tree_structure(tree, "psum", axis)

        try:
            return psum(tree, axis_name=axis)
        except Exception as e:
            # Provide more helpful error message for common axis binding issues
            error_msg = str(e)
            if "unbound axis name" in error_msg.lower():
                raise collective_error(
                    "psum",
                    axis,
                    f"Axis '{axis}' is not bound in current JAX transformation context. "
                    "Ensure this collective is called within a step function that uses proper mesh context.",
                ) from e
            raise collective_error(
                "psum", axis, f"JAX operation failed: {error_msg}"
            ) from e

    @staticmethod
    def pmean(tree: PyTree, axis: AxisName) -> PyTree:
        """Mean PyTree across specified mesh axis.

        Performs an all-reduce mean operation across the specified mesh axis.
        Values are summed and then divided by the axis size.

        This operation works within pjit/shard_map contexts where mesh
        axes are properly bound.

        Args:
            tree: PyTree to average across axis
            axis: Mesh axis name to average over

        Returns:
            PyTree with same structure, values averaged across axis

        Raises:
            CollectiveError: If axis name is invalid or tree is invalid

        Example:
            >>> # With mesh axis "data" of size 4:
            >>> local_loss = jnp.array(2.0)
            >>> avg_loss = tx.collectives.pmean(local_loss, axis="data")
            >>> # If all devices have loss 2.0, result is 2.0
        """
        _validate_axis_name(axis, "pmean", get_current_mesh())
        _validate_tree_structure(tree, "pmean", axis)

        try:
            return pmean(tree, axis_name=axis)
        except Exception as e:
            # Provide more helpful error message for common axis binding issues
            error_msg = str(e)
            if "unbound axis name" in error_msg.lower():
                raise collective_error(
                    "pmean",
                    axis,
                    f"Axis '{axis}' is not bound in current JAX transformation context. "
                    "Ensure this collective is called within a step function that uses proper mesh context.",
                ) from e
            raise collective_error(
                "pmean", axis, f"JAX operation failed: {error_msg}"
            ) from e

    @staticmethod
    def all_gather(x: Array, axis: AxisName, axis_index: Optional[int] = None) -> Array:
        """Gather arrays from all processes along specified mesh axis.

        Performs an all-gather operation where each device contributes its local
        array and receives the concatenated result from all devices along the
        specified mesh axis. The result is tiled (concatenated) along axis 0.

        Args:
            x: Array to gather from all processes
            axis: Mesh axis name to gather along
            axis_index: Optional axis index for gathering (unused, for compatibility)

        Returns:
            Array with gathered values from all processes, tiled along axis 0

        Raises:
            CollectiveError: If axis name is invalid or JAX operation fails

        Example:
            >>> # With mesh axis "data" of size 4, each device has x=[1, 2]
            >>> result = tx.collectives.all_gather(x, axis="data")
            >>> # Result shape: [8] with values [1, 2, 1, 2, 1, 2, 1, 2]
        """
        _validate_axis_name(axis, "all_gather", get_current_mesh())

        if not isinstance(x, jax.Array):
            raise collective_error(
                "all_gather", axis, f"input must be a JAX array, got {type(x)}"
            )

        try:
            return lax.all_gather(x, axis_name=axis, tiled=True)
        except Exception as e:
            error_msg = str(e)
            if "unbound axis name" in error_msg.lower():
                raise collective_error(
                    "all_gather",
                    axis,
                    f"Axis '{axis}' is not bound in current JAX transformation context. "
                    "Ensure this collective is called within a step function that uses proper mesh context.",
                ) from e
            raise collective_error(
                "all_gather", axis, f"JAX operation failed: {error_msg}"
            ) from e

    @staticmethod
    def reduce_scatter(x: Array, axis: AxisName, op: str = "add") -> Array:
        """Reduce and scatter array along specified mesh axis.

        Performs a reduce-scatter operation where arrays are first summed across
        all devices along the specified mesh axis, then scattered so each device
        receives only its portion of the result. This is equivalent to a psum
        followed by a scatter operation.

        Args:
            x: Array to reduce and scatter
            axis: Mesh axis name to reduce-scatter along
            op: Reduction operation ("add" only supported for now)

        Returns:
            Array with reduced and scattered values

        Raises:
            CollectiveError: If axis name is invalid or operation is invalid

        Example:
            >>> # With mesh axis "data" of size 4, input x has shape [8]
            >>> # Each device contributes its portion
            >>> result = tx.collectives.reduce_scatter(x, axis="data")
            >>> # Result shape: [2] (8/4), values are sum of all inputs
        """
        _validate_axis_name(axis, "reduce_scatter", get_current_mesh())

        # For now we only support "add" operation, which maps to psum_scatter
        valid_ops = {"add"}
        if op not in valid_ops:
            raise collective_error(
                "reduce_scatter",
                axis,
                f"invalid operation '{op}'. Valid operations: {valid_ops}",
            )

        if not isinstance(x, jax.Array):
            raise collective_error(
                "reduce_scatter", axis, f"input must be a JAX array, got {type(x)}"
            )

        try:
            # Use psum_scatter with tiled=True for reduce-scatter behavior
            return psum_scatter(x, axis_name=axis, tiled=True)
        except Exception as e:
            error_msg = str(e)
            if "unbound axis name" in error_msg.lower():
                raise collective_error(
                    "reduce_scatter",
                    axis,
                    f"Axis '{axis}' is not bound in current JAX transformation context. "
                    "Ensure this collective is called within a step function that uses proper mesh context.",
                ) from e
            raise collective_error(
                "reduce_scatter", axis, f"JAX operation failed: {error_msg}"
            ) from e

    @staticmethod
    def broadcast(x: Array, axis: AxisName, src_index: int = 0) -> Array:
        """Broadcast array from source process to all processes along axis.

        Implements broadcast by using ppermute to send data from the source
        device (at src_index) to all other devices along the specified mesh axis.
        Only the source device's value is propagated to all devices.

        Args:
            x: Array to broadcast
            axis: Mesh axis name to broadcast along
            src_index: Source process index (rank) for broadcast

        Returns:
            Array broadcasted from source process

        Raises:
            CollectiveError: If axis name is invalid or src_index is negative

        Example:
            >>> # With mesh axis "data" of size 4, only device 0 has valid data
            >>> result = tx.collectives.broadcast(x, axis="data", src_index=0)
            >>> # All devices now have device 0's data
        """
        _validate_axis_name(axis, "broadcast", get_current_mesh())

        # Basic validation for src_index
        if src_index < 0:
            raise collective_error(
                "broadcast", axis, f"src_index {src_index} must be non-negative"
            )

        if not isinstance(x, jax.Array):
            raise collective_error(
                "broadcast", axis, f"input must be a JAX array, got {type(x)}"
            )

        try:
            # Get axis size and current index
            axis_size = lax.axis_size(axis)

            # Validate src_index is within bounds
            if src_index >= axis_size:
                raise collective_error(
                    "broadcast",
                    axis,
                    f"src_index {src_index} must be less than axis size {axis_size}",
                )

            # Create permutation: source sends to all others, others send nowhere
            perm = [(src_index, i) for i in range(axis_size)]

            # Use ppermute to broadcast from source to all
            return lax.ppermute(x, axis_name=axis, perm=perm)

        except Exception as e:
            error_msg = str(e)
            if "unbound axis name" in error_msg.lower():
                raise collective_error(
                    "broadcast",
                    axis,
                    f"Axis '{axis}' is not bound in current JAX transformation context. "
                    "Ensure this collective is called within a step function that uses proper mesh context.",
                ) from e
            raise collective_error(
                "broadcast", axis, f"JAX operation failed: {error_msg}"
            ) from e

    @staticmethod
    def ppermute(x: Array, axis: AxisName, perm) -> Array:
        """Permute array along specified mesh axis.

        Performs point-to-point communication between devices based on the
        specified permutation. Each device can send its data to any other
        device along the mesh axis. This is the most fundamental collective
        operation that enables arbitrary data movement patterns.

        Args:
            x: Array to permute
            axis: Mesh axis name to permute along
            perm: Permutation specification as list of (source, destination) tuples

        Returns:
            Array with permuted values

        Raises:
            CollectiveError: If axis name is invalid or permutation is invalid

        Example:
            >>> # Ring shift: each device sends to the next device
            >>> axis_size = 4
            >>> perm = [(i, (i + 1) % axis_size) for i in range(axis_size)]
            >>> result = tx.collectives.ppermute(x, axis="data", perm=perm)
            >>> # Device i receives data from device (i-1) % axis_size
        """
        _validate_axis_name(axis, "ppermute", get_current_mesh())

        if not isinstance(x, jax.Array):
            raise collective_error(
                "ppermute", axis, f"input must be a JAX array, got {type(x)}"
            )

        # Validate permutation structure
        if not isinstance(perm, (list, tuple)):
            raise collective_error(
                "ppermute",
                axis,
                f"perm must be a list or tuple of (source, dest) pairs, got {type(perm)}",
            )

        try:
            # Validate each permutation pair
            for i, pair in enumerate(perm):
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise collective_error(
                        "ppermute",
                        axis,
                        f"perm[{i}] must be a (source, dest) pair, got {pair}",
                    )

                src, dest = pair
                if not isinstance(src, int) or not isinstance(dest, int):
                    raise collective_error(
                        "ppermute",
                        axis,
                        f"perm[{i}] indices must be integers, got ({type(src)}, {type(dest)})",
                    )

                if src < 0 or dest < 0:
                    raise collective_error(
                        "ppermute",
                        axis,
                        f"perm[{i}] indices must be non-negative, got ({src}, {dest})",
                    )

            # Use JAX lax ppermute
            return lax.ppermute(x, axis_name=axis, perm=perm)

        except Exception as e:
            if isinstance(e, CollectiveError):
                raise

            error_msg = str(e)
            if "unbound axis name" in error_msg.lower():
                raise collective_error(
                    "ppermute",
                    axis,
                    f"Axis '{axis}' is not bound in current JAX transformation context. "
                    "Ensure this collective is called within a step function that uses proper mesh context.",
                ) from e
            raise collective_error(
                "ppermute", axis, f"JAX operation failed: {error_msg}"
            ) from e
