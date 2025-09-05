"""Collective operations with explicit axis validation.

This module provides typed wrappers around JAX collective operations with
comprehensive axis validation and tree compatibility checking.

Note: For P0, these are simplified wrappers that provide the API surface
but delegate to JAX's built-in validation within shard_map/pmap contexts.
Full mesh context validation will be added when the execution engine
is implemented.
"""

from typing import Any, Optional
import warnings

import jax
import jax.numpy as jnp
from jax import lax

from ..types import PyTree, Array, AxisName
from ..exceptions import CollectiveError, collective_error


# Global mesh state for validation - will be set by execution engine
_current_mesh: Optional[jax.sharding.Mesh] = None


def set_current_mesh(mesh: Optional[jax.sharding.Mesh]) -> None:
    """Set the current mesh for collective validation.
    
    This is called by the execution engine to provide mesh context
    for collective operations validation.
    
    Args:
        mesh: Mesh to set as current, or None to clear
    """
    global _current_mesh
    _current_mesh = mesh


def get_current_mesh() -> Optional[jax.sharding.Mesh]:
    """Get the current mesh for collective validation.
    
    Returns:
        Current mesh if set, None otherwise
    """
    return _current_mesh


def _validate_axis_name(axis: AxisName, operation: str, mesh: Optional[jax.sharding.Mesh] = None) -> None:
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
            operation,
            str(axis),
            f"axis must be a string, got {type(axis)}"
        )
    
    if not axis:
        raise collective_error(
            operation,
            axis,
            "axis name cannot be empty"
        )
    
    # Check axis exists in mesh if provided
    if mesh is not None:
        if axis not in mesh.axis_names:
            available_axes = ", ".join(f"'{ax}'" for ax in mesh.axis_names)
            raise collective_error(
                operation,
                axis,
                f"axis not found in mesh. Available axes: [{available_axes}]"
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
                    operation,
                    axis,
                    f"leaf {i} is not a JAX array (got {type(leaf)})"
                )
                
    except Exception as e:
        if isinstance(e, CollectiveError):
            raise
        raise collective_error(
            operation,
            axis,
            f"invalid PyTree structure: {str(e)}"
        )


class collectives:
    """Namespace class for collective operations with validation."""
    
    @staticmethod
    def psum(tree: PyTree, axis: AxisName) -> PyTree:
        """Sum PyTree across specified mesh axis.
        
        Performs an all-reduce sum operation across the specified mesh axis.
        All participating processes contribute their local values to compute
        the global sum.
        
        Note: This operation must be called within a JAX transformation context
        that supports named axes (like shard_map or pmap).
        
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
            return lax.psum(tree, axis_name=axis)
        except Exception as e:
            raise collective_error("psum", axis, f"JAX operation failed: {str(e)}") from e
    
    @staticmethod
    def pmean(tree: PyTree, axis: AxisName) -> PyTree:
        """Mean PyTree across specified mesh axis.
        
        Performs an all-reduce mean operation across the specified mesh axis.
        Values are summed and then divided by the axis size.
        
        Note: This operation must be called within a JAX transformation context
        that supports named axes (like shard_map or pmap).
        
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
            return lax.pmean(tree, axis_name=axis)
        except Exception as e:
            raise collective_error("pmean", axis, f"JAX operation failed: {str(e)}") from e
    
    @staticmethod
    def all_gather(x: Array, axis: AxisName, axis_index: Optional[int] = None) -> Array:
        """Gather arrays from all processes along specified mesh axis.
        
        Note: This is a stub implementation for P0. Full implementation
        will be added in P1 when tensor parallel support is needed.
        
        Args:
            x: Array to gather from all processes
            axis: Mesh axis name to gather along
            axis_index: Optional axis index for gathering (not used in stub)
            
        Returns:
            Array with gathered values from all processes
            
        Raises:
            CollectiveError: If axis name is invalid
            NotImplementedError: Stub implementation
        """
        _validate_axis_name(axis, "all_gather", get_current_mesh())
        
        # For P0, we just return a warning that this is a stub
        warnings.warn(
            "all_gather is a stub implementation. Full implementation coming in P1.",
            UserWarning,
            stacklevel=2
        )
        
        # Return input unchanged as stub behavior
        return x
    
    @staticmethod
    def reduce_scatter(x: Array, axis: AxisName, op: str = "add") -> Array:
        """Reduce and scatter array along specified mesh axis.
        
        Note: This is a stub implementation for P0. Full implementation
        will be added in P1 when tensor parallel support is needed.
        
        Args:
            x: Array to reduce and scatter
            axis: Mesh axis name to reduce-scatter along
            op: Reduction operation ("add", "mul", "min", "max")
            
        Returns:
            Array with reduced and scattered values
            
        Raises:
            CollectiveError: If axis name is invalid or operation is invalid
            NotImplementedError: Stub implementation
        """
        _validate_axis_name(axis, "reduce_scatter", get_current_mesh())
        
        valid_ops = {"add", "mul", "min", "max"}
        if op not in valid_ops:
            raise collective_error(
                "reduce_scatter", 
                axis, 
                f"invalid operation '{op}'. Valid operations: {valid_ops}"
            )
        
        warnings.warn(
            "reduce_scatter is a stub implementation. Full implementation coming in P1.",
            UserWarning,
            stacklevel=2
        )
        
        # Return input unchanged as stub behavior
        return x
    
    @staticmethod
    def broadcast(x: Array, axis: AxisName, src_index: int = 0) -> Array:
        """Broadcast array from source process to all processes along axis.
        
        Note: This is a stub implementation for P0. Full implementation
        will be added in P1 when needed for advanced collectives.
        
        Args:
            x: Array to broadcast
            axis: Mesh axis name to broadcast along
            src_index: Source process index (rank) for broadcast
            
        Returns:
            Array broadcasted from source process
            
        Raises:
            CollectiveError: If axis name is invalid or src_index is negative
            NotImplementedError: Stub implementation
        """
        _validate_axis_name(axis, "broadcast", get_current_mesh())
        
        # Basic validation for src_index
        if src_index < 0:
            raise collective_error(
                "broadcast",
                axis,
                f"src_index {src_index} must be non-negative"
            )
        
        warnings.warn(
            "broadcast is a stub implementation. Full implementation coming in P1.",
            UserWarning,
            stacklevel=2
        )
        
        # Return input unchanged as stub behavior
        return x
    
    @staticmethod
    def ppermute(x: Array, axis: AxisName, perm) -> Array:
        """Permute array along specified mesh axis.
        
        Note: This is a stub implementation for P0. Full implementation
        will be added in P2 when pipeline parallel support is needed.
        
        Args:
            x: Array to permute
            axis: Mesh axis name to permute along
            perm: Permutation specification
            
        Returns:
            Array with permuted values
            
        Raises:
            CollectiveError: If axis name is invalid
            NotImplementedError: Stub implementation
        """
        _validate_axis_name(axis, "ppermute", get_current_mesh())
        
        warnings.warn(
            "ppermute is a stub implementation. Full implementation coming in P2.",
            UserWarning,
            stacklevel=2
        )
        
        # Return input unchanged as stub behavior  
        return x
