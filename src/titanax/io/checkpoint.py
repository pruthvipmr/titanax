"""Checkpoint system interface and utilities for Titanax.

This module defines the checkpoint interface and provides utility functions
for checkpoint management, path resolution, and metadata handling.
"""

import os
import re
from pathlib import Path
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

from ..types import PyTree, CheckpointStrategy
from ..exceptions import CheckpointError


@dataclass(frozen=True)
class CheckpointMetadata:
    """Metadata associated with a checkpoint."""
    
    step: int
    timestamp: float
    titanax_version: str
    jax_version: str
    mesh_spec: Optional[Dict[str, Any]] = None
    plan_spec: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None


class BaseCheckpointStrategy:
    """Base class providing common checkpoint functionality.
    
    This abstract base class provides utilities for path management,
    step-based naming, and metadata handling that can be shared
    across different checkpoint implementations.
    """
    
    def __init__(self, checkpoint_dir: str | Path):
        """Initialize base checkpoint strategy.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def get_checkpoint_path(self, step: int) -> Path:
        """Get the checkpoint path for a given step.
        
        Args:
            step: Training step number
            
        Returns:
            Path to checkpoint directory for this step
        """
        return self.checkpoint_dir / f"step_{step:08d}"
    
    def list_available_steps(self) -> List[int]:
        """List all available checkpoint steps in ascending order.
        
        Returns:
            Sorted list of available checkpoint step numbers
        """
        steps = []
        if not self.checkpoint_dir.exists():
            return steps
            
        step_pattern = re.compile(r"step_(\d+)")
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                match = step_pattern.match(path.name)
                if match:
                    steps.append(int(match.group(1)))
        
        return sorted(steps)
    
    def get_latest_step(self) -> Optional[int]:
        """Get the latest available checkpoint step.
        
        Returns:
            Latest checkpoint step, or None if no checkpoints exist
        """
        steps = self.list_available_steps()
        return steps[-1] if steps else None
    
    def cleanup_old_checkpoints(self, keep_last_n: int = 3) -> None:
        """Remove old checkpoints, keeping only the most recent N.
        
        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        if keep_last_n <= 0:
            raise CheckpointError(
                "keep_last_n must be positive",
                suggestion="Use a positive integer for keep_last_n parameter"
            )
        
        steps = self.list_available_steps()
        if len(steps) <= keep_last_n:
            return
            
        steps_to_remove = steps[:-keep_last_n]
        for step in steps_to_remove:
            checkpoint_path = self.get_checkpoint_path(step)
            if checkpoint_path.exists():
                # Remove directory and all contents
                import shutil
                shutil.rmtree(checkpoint_path)
    
    def checkpoint_exists(self, step: int) -> bool:
        """Check if a checkpoint exists for the given step.
        
        Args:
            step: Training step number
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        return self.get_checkpoint_path(step).exists()


def resolve_checkpoint_step(
    strategy: CheckpointStrategy, 
    step: Optional[int] = None
) -> int:
    """Resolve checkpoint step for loading.
    
    Args:
        strategy: Checkpoint strategy instance
        step: Specific step to load, or None for latest
        
    Returns:
        Resolved step number
        
    Raises:
        CheckpointError: If no checkpoint is available
    """
    if isinstance(strategy, BaseCheckpointStrategy):
        if step is not None:
            if not strategy.checkpoint_exists(step):
                available_steps = strategy.list_available_steps()
                raise CheckpointError(
                    f"Checkpoint for step {step} not found",
                    suggestion=f"Available steps: {available_steps}" if available_steps else "No checkpoints available"
                )
            return step
        else:
            latest = strategy.get_latest_step()
            if latest is None:
                raise CheckpointError(
                    "No checkpoints available for loading",
                    suggestion="Save a checkpoint first or specify an existing checkpoint step"
                )
            return latest
    else:
        # For custom checkpoint strategies that don't inherit from BaseCheckpointStrategy
        if step is None:
            raise CheckpointError(
                "Step must be specified for custom checkpoint strategies",
                suggestion="Provide a specific step number for loading"
            )
        return step


def validate_checkpoint_compatibility(
    checkpoint_metadata: CheckpointMetadata,
    current_mesh_spec: Optional[Dict[str, Any]] = None,
    current_plan_spec: Optional[Dict[str, Any]] = None,
    strict: bool = False
) -> None:
    """Validate checkpoint compatibility with current configuration.
    
    Args:
        checkpoint_metadata: Metadata from the checkpoint
        current_mesh_spec: Current mesh specification
        current_plan_spec: Current plan specification  
        strict: If True, require exact match; if False, allow compatible differences
        
    Raises:
        CheckpointError: If checkpoint is incompatible
    """
    if strict:
        if (current_mesh_spec is not None and 
            checkpoint_metadata.mesh_spec != current_mesh_spec):
            raise CheckpointError(
                "Mesh specification mismatch in strict mode",
                suggestion="Use strict=False to allow mesh resharding or update mesh configuration"
            )
            
        if (current_plan_spec is not None and
            checkpoint_metadata.plan_spec != current_plan_spec):
            raise CheckpointError(
                "Plan specification mismatch in strict mode", 
                suggestion="Use strict=False to allow plan changes or update plan configuration"
            )
    
    # TODO: Add more sophisticated compatibility checking in future versions
    # - Check if tensor shapes are compatible
    # - Validate that resharding is possible
    # - Check optimizer state compatibility
