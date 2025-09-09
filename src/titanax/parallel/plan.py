"""Parallel plan definitions and composition.

This module provides dataclasses for defining data parallel, tensor parallel,
and pipeline parallel execution plans, along with validation and composition logic.
"""

import dataclasses
from typing import Optional, Dict, Tuple, Union

from ..exceptions import plan_validation_error
from ..types import AxisName
from ..runtime.mesh import MeshSpec


@dataclasses.dataclass(frozen=True)
class DP:
    """Data Parallel plan specification.

    Defines how to perform data parallel training across a mesh axis.
    Gradients are explicitly synchronized using collectives (psum/pmean).

    Attributes:
        axis: Name of the mesh axis for data parallelism (e.g., "data")
        accumulate_steps: Number of microbatches to accumulate before gradient sync (default: 1)
        sync_metrics: Whether to synchronize metrics across the DP axis (default: True)
    """

    axis: AxisName
    accumulate_steps: int = 1
    sync_metrics: bool = True

    def __post_init__(self):
        """Validate DP configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate the DP configuration."""
        if not self.axis:
            raise plan_validation_error(
                "DP axis cannot be empty", "Provide a valid axis name like 'data'"
            )

        if self.accumulate_steps < 1:
            raise plan_validation_error(
                f"accumulate_steps must be >= 1, got {self.accumulate_steps}",
                "Set accumulate_steps to a positive integer",
            )

    def validate_with_mesh(self, mesh_spec: MeshSpec) -> None:
        """Validate DP configuration against a mesh specification.

        Args:
            mesh_spec: The mesh specification to validate against

        Raises:
            PlanError: If the DP axis is not present in the mesh
        """
        if self.axis not in mesh_spec.axes:
            raise plan_validation_error(
                f"DP axis '{self.axis}' not found in mesh axes {mesh_spec.axes}",
                f"Add '{self.axis}' to mesh axes or change the DP axis name",
            )

    def describe(self) -> str:
        """Return a human-readable description of the DP plan."""
        description = f"Data Parallel on axis '{self.axis}'"

        if self.accumulate_steps > 1:
            description += f" with {self.accumulate_steps} microbatch accumulation"

        if not self.sync_metrics:
            description += " (metrics not synchronized)"

        return description


@dataclasses.dataclass(frozen=True)
class TP:
    """Tensor Parallel plan specification.

    Defines how to perform tensor (model) parallel training with explicit sharding rules.

    Attributes:
        axis: Name of the mesh axis for tensor parallelism (e.g., "model")
        rules: Dictionary mapping parameter path patterns to PartitionSpec dimensions
        prefer_reduce_scatter: Whether to prefer reduce_scatter over all_gather (default: True)
    """

    axis: AxisName
    rules: Dict[str, Tuple[Union[str, None], ...]]
    prefer_reduce_scatter: bool = True

    def __post_init__(self):
        """Validate TP configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate the TP configuration."""
        if not self.axis:
            raise plan_validation_error(
                "TP axis cannot be empty", "Provide a valid axis name like 'model'"
            )

        if not self.rules:
            raise plan_validation_error(
                "TP rules cannot be empty",
                "Provide at least one sharding rule mapping parameter paths to PartitionSpec",
            )

        # Validate rule format
        for path_pattern, partition_spec in self.rules.items():
            if not isinstance(path_pattern, str):
                raise plan_validation_error(
                    f"Rule key must be string, got {type(path_pattern).__name__}",
                    "Use string patterns for parameter paths",
                )

            if not isinstance(partition_spec, (tuple, list)):
                raise plan_validation_error(
                    f"Rule value must be tuple/list, got {type(partition_spec).__name__}",
                    "Use tuple of axis names or None for PartitionSpec",
                )

            # Check each element is str or None
            for i, element in enumerate(partition_spec):
                if element is not None and not isinstance(element, str):
                    raise plan_validation_error(
                        f"Rule '{path_pattern}' element {i} must be str or None, got {type(element).__name__}",
                        "PartitionSpec elements should be axis names (strings) or None",
                    )

    def validate_with_mesh(self, mesh_spec: MeshSpec) -> None:
        """Validate TP configuration against a mesh specification.

        Args:
            mesh_spec: The mesh specification to validate against

        Raises:
            PlanError: If the TP axis is not present in the mesh
        """
        if self.axis not in mesh_spec.axes:
            raise plan_validation_error(
                f"TP axis '{self.axis}' not found in mesh axes {mesh_spec.axes}",
                f"Add '{self.axis}' to mesh axes or change the TP axis name",
            )

        # Validate that rule axes are valid
        for path_pattern, partition_spec in self.rules.items():
            for axis_name in partition_spec:
                if axis_name is not None and axis_name not in mesh_spec.axes:
                    raise plan_validation_error(
                        f"Rule for '{path_pattern}' references unknown axis '{axis_name}'",
                        f"Use one of the mesh axes {mesh_spec.axes} or None",
                    )

    def describe(self) -> str:
        """Return a human-readable description of the TP plan."""
        description = (
            f"Tensor Parallel on axis '{self.axis}' with {len(self.rules)} rules"
        )

        if self.prefer_reduce_scatter:
            description += " (prefer reduce_scatter)"
        else:
            description += " (prefer all_gather)"

        return description


@dataclasses.dataclass(frozen=True)
class PP:
    """Pipeline Parallel plan specification.

    Defines how to perform pipeline parallel training with 1F1B scheduling.

    Attributes:
        axis: Name of the mesh axis for pipeline parallelism
        stages: List of pipeline stages
        microbatch_size: Size of each microbatch
        checkpoint_ratio: Ratio of layers to checkpoint for activation remat (default: 0.0)
    """

    axis: AxisName
    stages: list  # Will be defined later as List[Stage]
    microbatch_size: int
    checkpoint_ratio: float = 0.0

    def __post_init__(self):
        """Validate PP configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate the PP configuration."""
        if not self.axis:
            raise plan_validation_error(
                "PP axis cannot be empty", "Provide a valid axis name like 'pipe'"
            )

        if self.microbatch_size < 1:
            raise plan_validation_error(
                f"microbatch_size must be >= 1, got {self.microbatch_size}",
                "Set microbatch_size to a positive integer",
            )

        if not (0.0 <= self.checkpoint_ratio <= 1.0):
            raise plan_validation_error(
                f"checkpoint_ratio must be in [0.0, 1.0], got {self.checkpoint_ratio}",
                "Set checkpoint_ratio between 0.0 (no checkpointing) and 1.0 (full checkpointing)",
            )

    def validate_with_mesh(self, mesh_spec: MeshSpec) -> None:
        """Validate PP configuration against a mesh specification.

        Args:
            mesh_spec: The mesh specification to validate against

        Raises:
            PlanError: If the PP axis is not present in the mesh
        """
        if self.axis not in mesh_spec.axes:
            raise plan_validation_error(
                f"PP axis '{self.axis}' not found in mesh axes {mesh_spec.axes}",
                f"Add '{self.axis}' to mesh axes or change the PP axis name",
            )

    def describe(self) -> str:
        """Return a human-readable description of the PP plan."""
        description = (
            f"Pipeline Parallel on axis '{self.axis}' with {len(self.stages)} stages"
        )
        description += f", microbatch_size={self.microbatch_size}"

        if self.checkpoint_ratio > 0:
            description += f", {self.checkpoint_ratio:.1%} activation checkpointing"

        return description


@dataclasses.dataclass(frozen=True)
class Plan:
    """Composite parallel execution plan.

    Combines data parallel, tensor parallel, and pipeline parallel strategies.
    Multiple strategies can be composed (e.g., DP×TP, DP×PP, DP×TP×PP).

    Attributes:
        data_parallel: Optional data parallel configuration
        tensor_parallel: Optional tensor parallel configuration
        pipeline_parallel: Optional pipeline parallel configuration
    """

    data_parallel: Optional[DP] = None
    tensor_parallel: Optional[TP] = None
    pipeline_parallel: Optional[PP] = None

    def __post_init__(self):
        """Validate plan composition after initialization."""
        self._validate_composition()

    def _validate_composition(self) -> None:
        """Validate the parallel plan composition."""
        if not any([self.data_parallel, self.tensor_parallel, self.pipeline_parallel]):
            raise plan_validation_error(
                "Plan must specify at least one parallel strategy",
                "Add data_parallel=DP(...), tensor_parallel=TP(...), or pipeline_parallel=PP(...)",
            )

        # Check for axis conflicts
        used_axes = set()

        if self.data_parallel:
            used_axes.add(self.data_parallel.axis)

        if self.tensor_parallel:
            if self.tensor_parallel.axis in used_axes:
                raise plan_validation_error(
                    f"Axis '{self.tensor_parallel.axis}' is used by multiple parallel strategies",
                    "Use different axis names for each parallel strategy",
                )
            used_axes.add(self.tensor_parallel.axis)

        if self.pipeline_parallel:
            if self.pipeline_parallel.axis in used_axes:
                # Allow PP to reuse TP axis in some cases
                if (
                    self.tensor_parallel
                    and self.pipeline_parallel.axis == self.tensor_parallel.axis
                ):
                    pass  # This is allowed in the spec
                else:
                    raise plan_validation_error(
                        f"Axis '{self.pipeline_parallel.axis}' is used by multiple parallel strategies",
                        "Use different axis names or allow PP to reuse TP axis",
                    )
            used_axes.add(self.pipeline_parallel.axis)

    def validate(self, mesh_spec: MeshSpec) -> None:
        """Validate the plan against a mesh specification.

        Args:
            mesh_spec: The mesh specification to validate against

        Raises:
            PlanError: If any parallel strategy is incompatible with the mesh
        """
        if self.data_parallel:
            self.data_parallel.validate_with_mesh(mesh_spec)

        if self.tensor_parallel:
            self.tensor_parallel.validate_with_mesh(mesh_spec)

        if self.pipeline_parallel:
            self.pipeline_parallel.validate_with_mesh(mesh_spec)

        # Additional validation: check that all axes are present
        required_axes = set()
        if self.data_parallel:
            required_axes.add(self.data_parallel.axis)
        if self.tensor_parallel:
            required_axes.add(self.tensor_parallel.axis)
        if self.pipeline_parallel:
            required_axes.add(self.pipeline_parallel.axis)

        missing_axes = required_axes - set(mesh_spec.axes)
        if missing_axes:
            raise plan_validation_error(
                f"Mesh is missing required axes: {missing_axes}",
                f"Add missing axes to mesh_spec.axes: {tuple(mesh_spec.axes) + tuple(missing_axes)}",
            )

    def describe(self) -> str:
        """Return a human-readable description of the complete plan."""
        descriptions = []

        if self.data_parallel:
            descriptions.append(self.data_parallel.describe())

        if self.tensor_parallel:
            descriptions.append(self.tensor_parallel.describe())

        if self.pipeline_parallel:
            descriptions.append(self.pipeline_parallel.describe())

        if len(descriptions) == 0:
            return "Empty Plan"
        elif len(descriptions) == 1:
            return descriptions[0]
        else:
            return " × ".join(descriptions)

    def get_all_axes(self) -> Tuple[str, ...]:
        """Get all axes used by this plan."""
        axes = []

        if self.data_parallel:
            axes.append(self.data_parallel.axis)

        if self.tensor_parallel:
            axes.append(self.tensor_parallel.axis)

        if self.pipeline_parallel:
            if self.pipeline_parallel.axis not in axes:  # Avoid duplicates
                axes.append(self.pipeline_parallel.axis)

        return tuple(axes)

    def is_data_parallel_only(self) -> bool:
        """Check if this plan uses only data parallelism."""
        return (
            self.data_parallel is not None
            and self.tensor_parallel is None
            and self.pipeline_parallel is None
        )

    def has_microbatching(self) -> bool:
        """Check if this plan uses microbatching (DP accumulation or PP)."""
        dp_microbatching = (
            self.data_parallel and self.data_parallel.accumulate_steps > 1
        )
        pp_microbatching = self.pipeline_parallel is not None

        return dp_microbatching or pp_microbatching
