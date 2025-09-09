"""Process group utilities for mesh axes.

This module provides the ProcessGroups class for querying mesh topology
and process relationships across different axes.
"""

from typing import Dict

import jax
import jax.numpy as jnp

from ..exceptions import mesh_validation_error
from ..types import Mesh, ProcessRank, WorldSize


class ProcessGroups:
    """Utility class for querying process group information within a mesh.
    
    This class provides convenient methods to query the size and rank
    of the current process along different mesh axes.
    """
    
    def __init__(self, mesh: Mesh):
        """Initialize process groups from a JAX mesh.
        
        Args:
            mesh: JAX mesh to extract process group information from
        """
        self._mesh = mesh
        self._axis_sizes: Dict[str, int] = {}
        self._axis_ranks: Dict[str, int] = {}
        
        # Precompute axis information
        self._compute_axis_info()
    
    def _compute_axis_info(self) -> None:
        """Precompute size and rank information for all axes."""
        for axis_name in self._mesh.axis_names:
            # mesh.shape is an OrderedDict, access by axis name directly
            self._axis_sizes[axis_name] = self._mesh.shape[axis_name]
            self._axis_ranks[axis_name] = self._get_axis_rank(axis_name)
    
    def _get_axis_rank(self, axis_name: str) -> int:
        """Get the rank of the current process along a specific axis.
        
        Args:
            axis_name: Name of the mesh axis
            
        Returns:
            Rank of current process along the specified axis
        """
        try:
            # Get current process index in global device array
            local_devices = jax.local_devices()
            if not local_devices:
                return 0
            
            # Find the first local device in the mesh
            local_device = local_devices[0]
            
            # Find position of this device in the mesh
            mesh_devices = self._mesh.devices.flatten()
            try:
                device_index = list(mesh_devices).index(local_device)
            except ValueError:
                # Local device not in mesh - this can happen in some configurations
                # Fall back to process index
                return jax.process_index() % self._axis_sizes[axis_name]
            
            # Convert flat index to mesh coordinates
            mesh_shape_tuple = tuple(self._mesh.shape.values())
            mesh_coords = jnp.unravel_index(device_index, mesh_shape_tuple)
            axis_index = self._mesh.axis_names.index(axis_name)
            
            return int(mesh_coords[axis_index])
            
        except Exception:
            # Fallback to process-based calculation
            axis_size = self._axis_sizes[axis_name]
            return jax.process_index() % axis_size
    
    def size(self, axis: str) -> WorldSize:
        """Get the world size along a specific mesh axis.
        
        Args:
            axis: Name of the mesh axis
            
        Returns:
            Number of processes along the specified axis
            
        Raises:
            MeshError: If axis doesn't exist in the mesh
        """
        if axis not in self._axis_sizes:
            available_axes = list(self._mesh.axis_names)
            raise mesh_validation_error(
                f"Axis '{axis}' not found in mesh",
                f"Available axes: {available_axes}"
            )
        
        return self._axis_sizes[axis]
    
    def rank(self, axis: str) -> ProcessRank:
        """Get the rank of the current process along a specific mesh axis.
        
        Args:
            axis: Name of the mesh axis
            
        Returns:
            Rank (0-indexed) of the current process along the specified axis
            
        Raises:
            MeshError: If axis doesn't exist in the mesh
        """
        if axis not in self._axis_ranks:
            available_axes = list(self._mesh.axis_names)
            raise mesh_validation_error(
                f"Axis '{axis}' not found in mesh",
                f"Available axes: {available_axes}"
            )
        
        return self._axis_ranks[axis]
    
    def is_first_in_axis(self, axis: str) -> bool:
        """Check if current process is rank 0 along the specified axis.
        
        Args:
            axis: Name of the mesh axis
            
        Returns:
            True if current process has rank 0 along the axis
        """
        return self.rank(axis) == 0
    
    def is_last_in_axis(self, axis: str) -> bool:
        """Check if current process is the last rank along the specified axis.
        
        Args:
            axis: Name of the mesh axis
            
        Returns:
            True if current process has the highest rank along the axis
        """
        return self.rank(axis) == self.size(axis) - 1
    
    def get_axis_info(self, axis: str) -> tuple[WorldSize, ProcessRank]:
        """Get both size and rank for an axis in one call.
        
        Args:
            axis: Name of the mesh axis
            
        Returns:
            Tuple of (world_size, process_rank) for the axis
        """
        return self.size(axis), self.rank(axis)
    
    def validate_axis(self, axis: str) -> None:
        """Validate that an axis exists in the mesh.
        
        Args:
            axis: Name of the mesh axis to validate
            
        Raises:
            MeshError: If axis doesn't exist
        """
        if axis not in self._mesh.axis_names:
            available_axes = list(self._mesh.axis_names)
            raise mesh_validation_error(
                f"Axis '{axis}' not found in mesh",
                f"Available axes: {available_axes}"
            )
    
    def describe(self) -> str:
        """Generate a description of process group information.
        
        Returns:
            Human-readable string describing process groups
        """
        lines = ["ProcessGroups:"]
        lines.append(f"  Mesh shape: {dict(self._mesh.shape)}")
        lines.append("  Current process ranks:")
        
        for axis_name in self._mesh.axis_names:
            size = self._axis_sizes[axis_name]
            rank = self._axis_ranks[axis_name]
            lines.append(f"    {axis_name}: {rank}/{size-1} (size={size})")
        
        lines.append(f"  Global process: {jax.process_index()}/{jax.process_count()-1}")
        
        return "\n".join(lines)
    
    @property
    def mesh(self) -> Mesh:
        """Get the underlying JAX mesh."""
        return self._mesh
    
    @property 
    def axis_names(self) -> tuple[str, ...]:
        """Get all axis names in the mesh."""
        return self._mesh.axis_names
    
    @property
    def global_process_count(self) -> int:
        """Get total number of processes across all axes."""
        return jax.process_count()
    
    @property
    def global_process_rank(self) -> int:
        """Get global rank of current process."""
        return jax.process_index()
