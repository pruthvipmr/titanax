"""Orbax-based checkpoint implementation for Titanax.

This module provides an Orbax-based implementation of the CheckpointStrategy
protocol, supporting sharded parameter save/load, TrainState serialization,
and step-based checkpoint management.
"""

import time
import json
from pathlib import Path
from typing import Optional, Dict, Any

import jax
import orbax.checkpoint as ocp
from orbax.checkpoint import PyTreeCheckpointer

from ..types import PyTree
from ..exceptions import CheckpointError
from .checkpoint import BaseCheckpointStrategy, CheckpointMetadata, validate_checkpoint_compatibility


# Version info for metadata
try:
    import titanax
    TITANAX_VERSION = getattr(titanax, "__version__", "dev")
except ImportError:
    TITANAX_VERSION = "dev"


class OrbaxCheckpoint(BaseCheckpointStrategy):
    """Orbax-based checkpoint strategy with sharding support.
    
    This implementation uses Google's Orbax library for efficient checkpointing
    of large, sharded models. It supports:
    - Automatic sharded save/load
    - Step-based checkpoint organization
    - Metadata tracking and compatibility validation
    - Checkpoint cleanup and management
    """
    
    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_interval_steps: int = 1000,
        keep_checkpoints: int = 3,
        async_save: bool = True,
        validate_compatibility: bool = True,
        mesh_spec: Optional[Dict[str, Any]] = None,
        plan_spec: Optional[Dict[str, Any]] = None
    ):
        """Initialize Orbax checkpoint strategy.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
            save_interval_steps: Steps between automatic saves (0 to disable)
            keep_checkpoints: Number of recent checkpoints to keep (0 for unlimited)
            async_save: Whether to save asynchronously
            validate_compatibility: Whether to validate compatibility on load
            mesh_spec: Current mesh specification for compatibility checking
            plan_spec: Current plan specification for compatibility checking
        """
        super().__init__(checkpoint_dir)
        
        self.save_interval_steps = save_interval_steps
        self.keep_checkpoints = keep_checkpoints
        self.async_save = async_save
        self.validate_compatibility = validate_compatibility
        self.mesh_spec = mesh_spec
        self.plan_spec = plan_spec
        
        # Initialize Orbax checkpointer
        try:
            self.checkpointer = PyTreeCheckpointer()
        except Exception as e:
            raise CheckpointError(
                f"Failed to initialize Orbax checkpointer: {e}",
                suggestion="Ensure orbax-checkpoint is installed: pip install orbax-checkpoint"
            ) from e
        
        # Track last saved step for interval checking
        self._last_saved_step = -1
    
    def save(self, state: PyTree, step: int) -> None:
        """Save training state to checkpoint.
        
        Args:
            state: Training state (typically TrainState)
            step: Current training step
            
        Raises:
            CheckpointError: If save operation fails
        """
        # Check if we should save based on interval
        if (self.save_interval_steps > 0 and 
            step - self._last_saved_step < self.save_interval_steps):
            return
        
        checkpoint_path = self.get_checkpoint_path(step)
        
        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'state': state,
                'metadata': self._create_metadata(step)
            }
            
            # Save with Orbax (simplified approach - Orbax handles async internally)
            self.checkpointer.save(
                checkpoint_path / 'checkpoint',
                checkpoint_data
            )
            
            # Save human-readable metadata
            self._save_metadata_json(checkpoint_path, checkpoint_data['metadata'])
            
            self._last_saved_step = step
            
            # Cleanup old checkpoints if requested
            if self.keep_checkpoints > 0:
                self.cleanup_old_checkpoints(self.keep_checkpoints)
                
        except Exception as e:
            raise CheckpointError(
                f"Failed to save checkpoint at step {step}: {e}",
                suggestion="Check disk space and permissions for checkpoint directory"
            ) from e
    
    def load(self, step: Optional[int] = None) -> PyTree:
        """Load training state from checkpoint.
        
        Args:
            step: Specific step to load, or None for latest
            
        Returns:
            Loaded training state
            
        Raises:
            CheckpointError: If load operation fails
        """
        from .checkpoint import resolve_checkpoint_step
        
        try:
            resolved_step = resolve_checkpoint_step(self, step)
            checkpoint_path = self.get_checkpoint_path(resolved_step)
            
            # Load checkpoint data
            checkpoint_data = self.checkpointer.restore(
                checkpoint_path / 'checkpoint'
            )
            
            # Validate compatibility if enabled
            if self.validate_compatibility and 'metadata' in checkpoint_data:
                metadata = checkpoint_data['metadata']
                if isinstance(metadata, dict):
                    checkpoint_metadata = CheckpointMetadata(**metadata)
                    validate_checkpoint_compatibility(
                        checkpoint_metadata,
                        self.mesh_spec,
                        self.plan_spec,
                        strict=False  # Allow resharding by default
                    )
            
            return checkpoint_data['state']
            
        except CheckpointError:
            raise
        except Exception as e:
            available_steps = self.list_available_steps()
            raise CheckpointError(
                f"Failed to load checkpoint: {e}",
                suggestion=f"Available steps: {available_steps}" if available_steps else "No checkpoints available"
            ) from e
    
    def restore(self, state: PyTree, step: Optional[int] = None) -> PyTree:
        """Restore training state from checkpoint (alias for load).
        
        This method provides the same functionality as load() but matches
        the CheckpointStrategy protocol naming convention.
        
        Args:
            state: Unused (kept for protocol compatibility)
            step: Specific step to restore, or None for latest
            
        Returns:
            Restored training state
        """
        return self.load(step)
    
    def should_save(self, step: int) -> bool:
        """Check if checkpoint should be saved at this step.
        
        Args:
            step: Current training step
            
        Returns:
            True if checkpoint should be saved
        """
        if self.save_interval_steps <= 0:
            return False
        # Always save if no previous save (handle initial case)
        if self._last_saved_step == -1:
            return True
        return step - self._last_saved_step >= self.save_interval_steps
    
    def get_metadata(self, step: Optional[int] = None) -> CheckpointMetadata:
        """Get metadata for a checkpoint.
        
        Args:
            step: Specific step, or None for latest
            
        Returns:
            Checkpoint metadata
            
        Raises:
            CheckpointError: If metadata cannot be loaded
        """
        from .checkpoint import resolve_checkpoint_step
        
        try:
            resolved_step = resolve_checkpoint_step(self, step)
            checkpoint_path = self.get_checkpoint_path(resolved_step)
            
            # Try to load from JSON first (human-readable)
            metadata_json_path = checkpoint_path / 'metadata.json'
            if metadata_json_path.exists():
                with open(metadata_json_path, 'r') as f:
                    metadata_dict = json.load(f)
                    return CheckpointMetadata(**metadata_dict)
            
            # Fall back to loading from checkpoint
            checkpoint_data = self.checkpointer.restore(
                checkpoint_path / 'checkpoint'
            )
            if 'metadata' in checkpoint_data:
                return CheckpointMetadata(**checkpoint_data['metadata'])
            
            # No metadata available
            raise CheckpointError(
                f"No metadata found for checkpoint step {resolved_step}",
                suggestion="This may be an old checkpoint format"
            )
            
        except CheckpointError:
            raise
        except Exception as e:
            raise CheckpointError(
                f"Failed to load checkpoint metadata: {e}",
                suggestion="Check if checkpoint directory is accessible"
            ) from e
    
    def _create_metadata(self, step: int) -> Dict[str, Any]:
        """Create metadata for checkpoint.
        
        Args:
            step: Training step
            
        Returns:
            Metadata dictionary
        """
        return {
            'step': step,
            'timestamp': time.time(),
            'titanax_version': TITANAX_VERSION,
            'jax_version': jax.__version__,
            'mesh_spec': self.mesh_spec,
            'plan_spec': self.plan_spec,
            'extra': {
                'save_interval_steps': self.save_interval_steps,
                'async_save': self.async_save
            }
        }
    
    def _save_metadata_json(self, checkpoint_path: Path, metadata: Dict[str, Any]) -> None:
        """Save metadata as human-readable JSON.
        
        Args:
            checkpoint_path: Path to checkpoint directory
            metadata: Metadata dictionary
        """
        try:
            metadata_path = checkpoint_path / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        except Exception:
            # Non-critical operation, don't fail the entire save
            pass


def create_checkpoint_strategy(
    checkpoint_dir: str | Path,
    strategy: str = "orbax",
    **kwargs
) -> BaseCheckpointStrategy:
    """Factory function to create checkpoint strategies.
    
    Args:
        checkpoint_dir: Directory for checkpoints
        strategy: Strategy type ("orbax" supported)
        **kwargs: Additional arguments for strategy
        
    Returns:
        Checkpoint strategy instance
        
    Raises:
        CheckpointError: If strategy is not supported
    """
    if strategy.lower() == "orbax":
        return OrbaxCheckpoint(checkpoint_dir, **kwargs)
    else:
        raise CheckpointError(
            f"Unsupported checkpoint strategy: {strategy}",
            suggestion="Supported strategies: 'orbax'"
        )
