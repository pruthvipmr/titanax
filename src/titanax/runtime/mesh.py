"""Mesh specification and creation utilities.

This module provides the MeshSpec dataclass for declarative mesh creation
and utilities for building JAX meshes with validation.
"""

import dataclasses
import math
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
import jax.sharding as sharding

from ..exceptions import MeshError, mesh_validation_error
from ..types import Device, Mesh
from .init import enumerate_devices


@dataclasses.dataclass
class MeshSpec:
    """Specification for creating a JAX mesh.

    This class provides a declarative way to specify mesh topology and
    automatically handles device enumeration and axis size inference.

    Attributes:
        devices: Device specification. Can be:
            - "all": Use all available devices
            - List of devices: Use specific devices
            - None: Same as "all"
        axes: Tuple of axis names (e.g., ("data",) or ("data", "model"))
        shape: Optional tuple specifying size of each axis. None values are auto-inferred.
        topology: Optional dictionary with topology hints for optimization
    """

    devices: Union[str, List[Device], None] = "all"
    axes: Tuple[str, ...] = ("data",)
    shape: Optional[Tuple[Optional[int], ...]] = None
    topology: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate mesh specification after initialization."""
        self._validate_spec()

    def _validate_spec(self) -> None:
        """Validate the mesh specification."""
        if not self.axes:
            raise mesh_validation_error(
                "Empty axes tuple",
                "Provide at least one axis name, e.g., axes=('data',)",
            )

        if len(set(self.axes)) != len(self.axes):
            raise mesh_validation_error(
                f"Duplicate axis names in {self.axes}", "Each axis name must be unique"
            )

        for axis in self.axes:
            if not isinstance(axis, str) or not axis:
                raise mesh_validation_error(
                    f"Invalid axis name: {axis}", "Axis names must be non-empty strings"
                )

        if self.shape is not None:
            if len(self.shape) != len(self.axes):
                raise mesh_validation_error(
                    f"Shape length {len(self.shape)} doesn't match axes length {len(self.axes)}",
                    f"Provide a shape tuple with {len(self.axes)} elements",
                )

            for i, size in enumerate(self.shape):
                if size is not None and (not isinstance(size, int) or size <= 0):
                    raise mesh_validation_error(
                        f"Invalid shape[{i}] = {size}",
                        "Shape values must be positive integers or None",
                    )

    def _resolve_devices(self) -> List[Device]:
        """Resolve device specification to actual device list."""
        if self.devices == "all" or self.devices is None:
            devices = enumerate_devices(local_only=False)
        elif isinstance(self.devices, list):
            devices = self.devices
        else:
            raise mesh_validation_error(
                f"Invalid devices specification: {self.devices}",
                "Use 'all', None, or a list of JAX devices",
            )

        if not devices:
            raise mesh_validation_error(
                "No devices available", "Ensure JAX can access at least one device"
            )

        return devices

    def _infer_shape(self, devices: List[Device]) -> Tuple[int, ...]:
        """Infer axis sizes from device count and partial shape specification."""
        device_count = len(devices)

        if self.shape is None:
            # Auto-infer all dimensions
            if len(self.axes) == 1:
                return (device_count,)
            else:
                # For multi-axis, try to create a balanced factorization
                return self._factorize_device_count(device_count)

        # Partial shape inference
        shape_list = list(self.shape)
        known_size = 1
        none_indices = []

        for i, size in enumerate(shape_list):
            if size is None:
                none_indices.append(i)
            else:
                known_size *= size

        if not none_indices:
            # All dimensions specified
            if known_size != device_count:
                raise mesh_validation_error(
                    f"Shape product {known_size} doesn't match device count {device_count}",
                    f"Adjust shape {self.shape} to multiply to {device_count}",
                )
            return tuple(s for s in shape_list if s is not None)

        # Infer missing dimensions
        remaining_devices = device_count // known_size
        if device_count % known_size != 0:
            raise mesh_validation_error(
                f"Cannot divide {device_count} devices by known shape product {known_size}",
                "Ensure shape dimensions are compatible with device count",
            )

        if len(none_indices) == 1:
            # Single unknown dimension
            shape_list[none_indices[0]] = remaining_devices
        else:
            # Multiple unknown dimensions - factorize remaining
            factors = self._factorize_device_count(remaining_devices, len(none_indices))
            for i, factor in enumerate(factors):
                shape_list[none_indices[i]] = factor

        return tuple(s for s in shape_list if s is not None)

    def _factorize_device_count(
        self, device_count: int, num_factors: Optional[int] = None
    ) -> Tuple[int, ...]:
        """Factorize device count into balanced dimensions."""
        if num_factors is None:
            num_factors = len(self.axes)

        if num_factors == 1:
            return (device_count,)

        # Find factors of device_count
        factors = []
        temp = device_count

        # Start with small factors and work up
        for i in range(2, int(math.sqrt(device_count)) + 1):
            while temp % i == 0:
                factors.append(i)
                temp //= i
        if temp > 1:
            factors.append(temp)

        # If not enough factors, pad with 1s
        while len(factors) < num_factors:
            factors.append(1)

        # If too many factors, combine smallest ones
        while len(factors) > num_factors:
            factors.sort()
            factors[0] = factors[0] * factors[1]
            factors.pop(1)

        factors.sort(reverse=True)  # Largest first for better load balancing
        return tuple(factors)

    def build(self) -> Mesh:
        """Build a JAX mesh from this specification.

        Returns:
            JAX mesh with specified topology

        Raises:
            MeshError: If mesh creation fails
        """
        try:
            devices = self._resolve_devices()
            shape = self._infer_shape(devices)

            # Validate final shape
            if math.prod(shape) != len(devices):
                raise mesh_validation_error(
                    f"Shape product {math.prod(shape)} doesn't equal device count {len(devices)}",
                    "This should not happen - please report as a bug",
                )

            # Reshape devices into mesh topology
            device_array = np.array(devices).reshape(shape)

            mesh = sharding.Mesh(device_array, self.axes)
            return mesh

        except MeshError:
            raise
        except Exception as e:
            raise mesh_validation_error(
                f"Failed to create mesh: {e}",
                "Check device availability and mesh specification",
            ) from e

    def describe(self) -> str:
        """Generate a human-readable description of the mesh specification.

        Returns:
            Multi-line string describing the mesh configuration
        """
        try:
            devices = self._resolve_devices()
            shape = self._infer_shape(devices)

            lines = [
                f"MeshSpec: {len(self.axes)}-dimensional",
                f"  Axes: {self.axes}",
                f"  Shape: {shape}",
                f"  Devices: {len(devices)} ({devices[0].platform if devices else 'unknown'})",
            ]

            # Add device layout visualization for small meshes
            if math.prod(shape) <= 16 and len(self.axes) <= 2:
                lines.append("  Layout:")
                if len(shape) == 1:
                    device_ids = [f"D{d.id}" for d in devices]
                    lines.append(f"    {' '.join(device_ids)}")
                elif len(shape) == 2:
                    device_array = np.array([f"D{d.id}" for d in devices]).reshape(
                        shape
                    )
                    for row in device_array:
                        lines.append(f"    {' '.join(row)}")

            if self.topology:
                lines.append(f"  Topology hints: {self.topology}")

            return "\n".join(lines)

        except Exception as e:
            return f"MeshSpec: {self.axes} (error: {e})"

    def validate_compatibility(
        self, batch_size: int, allow_padding: bool = False
    ) -> None:
        """Validate that batch size is compatible with mesh.

        Args:
            batch_size: Global batch size
            allow_padding: Whether to allow automatic padding

        Raises:
            MeshError: If batch size is incompatible with data parallel axis
        """
        try:
            devices = self._resolve_devices()
            shape = self._infer_shape(devices)

            # Find data parallel axis (typically first or named 'data')
            data_axis_idx = 0
            if "data" in self.axes:
                data_axis_idx = self.axes.index("data")

            data_parallel_size = shape[data_axis_idx]

            if batch_size % data_parallel_size != 0:
                if not allow_padding:
                    raise mesh_validation_error(
                        f"Batch size {batch_size} not divisible by data parallel size {data_parallel_size}",
                        f"Use batch size divisible by {data_parallel_size}, or enable automatic padding",
                    )
                else:
                    # Calculate padding needed
                    padded_batch = (
                        (batch_size + data_parallel_size - 1) // data_parallel_size
                    ) * data_parallel_size
                    padding = padded_batch - batch_size
                    if padding > 0:
                        # This is just a warning, not an error
                        import logging

                        logging.warning(
                            f"Batch size {batch_size} will be padded to {padded_batch} "
                            f"(+{padding} samples) for data parallel size {data_parallel_size}"
                        )

        except MeshError:
            raise
        except Exception as e:
            raise mesh_validation_error(
                f"Failed to validate batch compatibility: {e}",
                "Check mesh specification and batch size",
            ) from e
