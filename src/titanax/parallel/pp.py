"""Pipeline Parallel stage definitions and scheduling.

This module provides the core abstractions for pipeline parallel training,
including stage definitions, forward/backward passes, and microbatch scheduling.
"""

import dataclasses
from typing import Callable, Optional, Protocol, Tuple, runtime_checkable

from ..types import PyTree
from ..exceptions import plan_validation_error


@runtime_checkable
class StageProtocol(Protocol):
    """Protocol defining the interface for a pipeline stage.

    Each pipeline stage must implement forward pass logic and provide
    metadata about rematerialization policies and stage boundaries.
    """

    def forward(self, inputs: PyTree, training: bool = True) -> Tuple[PyTree, PyTree]:
        """Forward pass through this pipeline stage.

        Args:
            inputs: Input activations from previous stage or data
            training: Whether in training mode (affects dropout, batch norm, etc.)

        Returns:
            Tuple of (outputs, activations) where:
            - outputs: Activations to send to next stage
            - activations: Saved activations for backward pass (can be None if no remat)
        """
        ...

    def backward(self, grad_outputs: PyTree, activations: PyTree) -> PyTree:
        """Backward pass through this pipeline stage.

        Args:
            grad_outputs: Gradient of loss w.r.t. stage outputs
            activations: Saved activations from forward pass

        Returns:
            Gradient of loss w.r.t. stage inputs
        """
        ...


@dataclasses.dataclass(frozen=True)
class Stage:
    """Pipeline stage implementation.

    Wraps a forward function with pipeline metadata and rematerialization policy.

    Attributes:
        forward_fn: Function implementing forward pass: (inputs, training) -> (outputs, activations)
        backward_fn: Optional custom backward function, defaults to JAX autodiff
        stage_id: Unique identifier for this stage within the pipeline
        remat_policy: Rematerialization policy for memory optimization
        stage_name: Optional human-readable name for debugging
    """

    forward_fn: Callable[[PyTree, bool], Tuple[PyTree, PyTree]]
    backward_fn: Optional[Callable[[PyTree, PyTree], PyTree]] = None
    stage_id: int = 0
    remat_policy: str = "none"  # "none", "full", "selective"
    stage_name: Optional[str] = None

    def __post_init__(self):
        """Validate stage configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate the stage configuration."""
        if not callable(self.forward_fn):
            raise plan_validation_error(
                "forward_fn must be callable",
                "Provide a function implementing forward pass logic",
            )

        if self.backward_fn is not None and not callable(self.backward_fn):
            raise plan_validation_error(
                "backward_fn must be callable or None",
                "Provide a function implementing custom backward pass or leave as None for autodiff",
            )

        if self.stage_id < 0:
            raise plan_validation_error(
                f"stage_id must be non-negative, got {self.stage_id}",
                "Use 0, 1, 2, ... for stage ordering",
            )

        valid_remat_policies = {"none", "full", "selective"}
        if self.remat_policy not in valid_remat_policies:
            raise plan_validation_error(
                f"Invalid remat_policy '{self.remat_policy}', must be one of {valid_remat_policies}",
                "Use 'none' (no remat), 'full' (checkpoint all), or 'selective' (heuristic-based)",
            )

    def forward(self, inputs: PyTree, training: bool = True) -> Tuple[PyTree, PyTree]:
        """Forward pass through this pipeline stage.

        Args:
            inputs: Input activations from previous stage or data
            training: Whether in training mode

        Returns:
            Tuple of (outputs, activations) for next stage and backward pass
        """
        return self.forward_fn(inputs, training)

    def backward(self, grad_outputs: PyTree, activations: PyTree) -> PyTree:
        """Backward pass through this pipeline stage.

        Args:
            grad_outputs: Gradient of loss w.r.t. stage outputs
            activations: Saved activations from forward pass

        Returns:
            Gradient of loss w.r.t. stage inputs
        """
        if self.backward_fn is not None:
            return self.backward_fn(grad_outputs, activations)
        else:
            # Default: use JAX autodiff (placeholder for now)
            raise NotImplementedError(
                "Default JAX autodiff backward pass not yet implemented. "
                "Provide custom backward_fn or use autodiff in higher-level training loop."
            )

    def describe(self) -> str:
        """Return a human-readable description of this stage."""
        name = self.stage_name if self.stage_name else f"Stage{self.stage_id}"
        desc = f"{name} (ID: {self.stage_id})"

        if self.remat_policy != "none":
            desc += f", remat={self.remat_policy}"

        return desc


@dataclasses.dataclass(frozen=True)
class PipelineSchedule:
    """Pipeline schedule configuration for microbatch execution.

    Defines how microbatches flow through the pipeline stages using various
    scheduling strategies (1F1B, interleaved, etc.).

    Attributes:
        strategy: Scheduling strategy name ("1F1B", "interleaved", "gpipe")
        num_microbatches: Number of microbatches per global batch
        warmup_steps: Number of warmup steps before steady state
        cooldown_steps: Number of cooldown steps after steady state
    """

    strategy: str = "1F1B"  # One Forward One Backward
    num_microbatches: int = 4
    warmup_steps: Optional[int] = None
    cooldown_steps: Optional[int] = None

    def __post_init__(self):
        """Validate schedule configuration after initialization."""
        self._validate()

    def _validate(self) -> None:
        """Validate the schedule configuration."""
        valid_strategies = {"1F1B", "interleaved", "gpipe"}
        if self.strategy not in valid_strategies:
            raise plan_validation_error(
                f"Invalid strategy '{self.strategy}', must be one of {valid_strategies}",
                "Use '1F1B' for basic pipeline, 'interleaved' for virtual stages, or 'gpipe' for simple forward-backward",
            )

        if self.num_microbatches < 1:
            raise plan_validation_error(
                f"num_microbatches must be >= 1, got {self.num_microbatches}",
                "Use at least 1 microbatch per global batch",
            )

        if self.warmup_steps is not None and self.warmup_steps < 0:
            raise plan_validation_error(
                f"warmup_steps must be non-negative, got {self.warmup_steps}",
                "Use 0 or positive integer for warmup phase",
            )

        if self.cooldown_steps is not None and self.cooldown_steps < 0:
            raise plan_validation_error(
                f"cooldown_steps must be non-negative, got {self.cooldown_steps}",
                "Use 0 or positive integer for cooldown phase",
            )

    def validate_with_pipeline(self, stages: list[Stage], microbatch_size: int) -> None:
        """Validate schedule compatibility with pipeline configuration.

        Args:
            stages: List of pipeline stages
            microbatch_size: Size of each microbatch

        Raises:
            PlanError: If schedule is incompatible with pipeline setup
        """
        num_stages = len(stages)

        if num_stages == 0:
            raise plan_validation_error(
                "Pipeline must have at least one stage",
                "Add stages to the pipeline before creating schedule",
            )

        # Validate microbatch configuration
        if self.num_microbatches < num_stages:
            raise plan_validation_error(
                f"num_microbatches ({self.num_microbatches}) should be >= num_stages ({num_stages})",
                f"Use at least {num_stages} microbatches for efficient pipeline utilization",
            )

        # Strategy-specific validation
        if self.strategy == "1F1B" and self.num_microbatches < 2 * num_stages:
            # For optimal 1F1B, need at least 2*num_stages microbatches
            pass  # Just a warning, not an error

        # Validate stage ordering
        stage_ids = [stage.stage_id for stage in stages]
        expected_ids = list(range(len(stages)))
        if sorted(stage_ids) != expected_ids:
            raise plan_validation_error(
                f"Stage IDs {stage_ids} should be consecutive starting from 0",
                f"Use stage IDs {expected_ids} for proper pipeline ordering",
            )

    def describe(self) -> str:
        """Return a human-readable description of this schedule."""
        desc = f"{self.strategy} schedule with {self.num_microbatches} microbatches"

        if self.warmup_steps:
            desc += f", warmup={self.warmup_steps}"
        if self.cooldown_steps:
            desc += f", cooldown={self.cooldown_steps}"

        return desc


def create_simple_stage(
    forward_fn: Callable[[PyTree, bool], Tuple[PyTree, PyTree]],
    stage_id: int,
    stage_name: Optional[str] = None,
    remat_policy: str = "none",
) -> Stage:
    """Create a simple pipeline stage with default configuration.

    Args:
        forward_fn: Function implementing forward pass
        stage_id: Unique stage identifier
        stage_name: Optional human-readable name
        remat_policy: Rematerialization policy

    Returns:
        Configured Stage instance

    Example:
        >>> def encoder_forward(inputs, training):
        ...     # Encoder logic here
        ...     outputs = encoder_layers(inputs)
        ...     return outputs, inputs  # Save inputs for backward
        >>>
        >>> encoder_stage = create_simple_stage(encoder_forward, stage_id=0, stage_name="encoder")
    """
    return Stage(
        forward_fn=forward_fn,
        stage_id=stage_id,
        stage_name=stage_name,
        remat_policy=remat_policy,
    )


def create_1f1b_schedule(
    num_stages: int, microbatch_size: int, global_batch_size: int
) -> PipelineSchedule:
    """Create a 1F1B (One Forward One Backward) pipeline schedule.

    Args:
        num_stages: Number of pipeline stages
        microbatch_size: Size of each microbatch
        global_batch_size: Total batch size across all microbatches

    Returns:
        Configured PipelineSchedule for 1F1B execution

    Raises:
        PlanError: If batch size configuration is invalid
    """
    if global_batch_size % microbatch_size != 0:
        raise plan_validation_error(
            f"global_batch_size ({global_batch_size}) must be divisible by microbatch_size ({microbatch_size})",
            "Adjust batch sizes so global_batch_size = num_microbatches * microbatch_size",
        )

    num_microbatches = global_batch_size // microbatch_size

    return PipelineSchedule(
        strategy="1F1B",
        num_microbatches=num_microbatches,
        warmup_steps=num_stages - 1,  # Standard 1F1B warmup
        cooldown_steps=num_stages - 1,  # Standard 1F1B cooldown
    )
