"""Unit tests for Titanax checkpoint system."""

import json
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

import jax
import jax.numpy as jnp

from src.titanax.io.checkpoint import (
    BaseCheckpointStrategy,
    CheckpointMetadata,
    resolve_checkpoint_step,
    validate_checkpoint_compatibility,
)
from src.titanax.io.orbax_io import OrbaxCheckpoint, create_checkpoint_strategy
from src.titanax.exceptions import CheckpointError


class TestCheckpointMetadata:
    """Test CheckpointMetadata dataclass."""
    
    def test_basic_metadata_creation(self):
        """Test creating basic metadata."""
        metadata = CheckpointMetadata(
            step=1000,
            timestamp=1234567890.0,
            titanax_version="0.1.0",
            jax_version="0.4.0"
        )
        
        assert metadata.step == 1000
        assert metadata.timestamp == 1234567890.0
        assert metadata.titanax_version == "0.1.0"
        assert metadata.jax_version == "0.4.0"
        assert metadata.mesh_spec is None
        assert metadata.plan_spec is None
        assert metadata.extra is None
    
    def test_metadata_with_specs(self):
        """Test metadata with mesh and plan specifications."""
        mesh_spec = {"axes": ["data", "model"], "shape": [4, 2]}
        plan_spec = {"data_parallel": {"axis": "data"}}
        extra = {"custom_field": "value"}
        
        metadata = CheckpointMetadata(
            step=2000,
            timestamp=1234567891.0,
            titanax_version="0.1.0",
            jax_version="0.4.0",
            mesh_spec=mesh_spec,
            plan_spec=plan_spec,
            extra=extra
        )
        
        assert metadata.mesh_spec == mesh_spec
        assert metadata.plan_spec == plan_spec
        assert metadata.extra == extra
    
    def test_metadata_immutability(self):
        """Test that metadata is immutable."""
        metadata = CheckpointMetadata(
            step=1000,
            timestamp=1234567890.0,
            titanax_version="0.1.0",
            jax_version="0.4.0"
        )
        
        # Should not be able to modify fields
        with pytest.raises(AttributeError):
            metadata.step = 2000


class TestBaseCheckpointStrategy:
    """Test BaseCheckpointStrategy functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.strategy = BaseCheckpointStrategy(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_initialization(self):
        """Test strategy initialization."""
        assert self.strategy.checkpoint_dir == self.temp_dir
        assert self.temp_dir.exists()
    
    def test_get_checkpoint_path(self):
        """Test checkpoint path generation."""
        path = self.strategy.get_checkpoint_path(1000)
        expected = self.temp_dir / "step_00001000"
        assert path == expected
    
    def test_get_checkpoint_path_formatting(self):
        """Test checkpoint path formatting with different step numbers."""
        # Test zero padding
        assert self.strategy.get_checkpoint_path(1).name == "step_00000001"
        assert self.strategy.get_checkpoint_path(1000).name == "step_00001000"
        assert self.strategy.get_checkpoint_path(99999999).name == "step_99999999"
    
    def test_list_available_steps_empty(self):
        """Test listing steps when no checkpoints exist."""
        steps = self.strategy.list_available_steps()
        assert steps == []
    
    def test_list_available_steps(self):
        """Test listing available checkpoint steps."""
        # Create some checkpoint directories
        (self.temp_dir / "step_00001000").mkdir()
        (self.temp_dir / "step_00002000").mkdir()
        (self.temp_dir / "step_00000500").mkdir()
        # Create non-checkpoint directory (should be ignored)
        (self.temp_dir / "not_a_step").mkdir()
        (self.temp_dir / "step_invalid").mkdir()
        
        steps = self.strategy.list_available_steps()
        assert steps == [500, 1000, 2000]  # Should be sorted
    
    def test_get_latest_step_empty(self):
        """Test getting latest step when no checkpoints exist."""
        latest = self.strategy.get_latest_step()
        assert latest is None
    
    def test_get_latest_step(self):
        """Test getting latest checkpoint step."""
        # Create checkpoint directories
        (self.temp_dir / "step_00001000").mkdir()
        (self.temp_dir / "step_00002000").mkdir()
        (self.temp_dir / "step_00000500").mkdir()
        
        latest = self.strategy.get_latest_step()
        assert latest == 2000
    
    def test_checkpoint_exists(self):
        """Test checking if checkpoint exists."""
        assert not self.strategy.checkpoint_exists(1000)
        
        # Create checkpoint directory
        (self.temp_dir / "step_00001000").mkdir()
        assert self.strategy.checkpoint_exists(1000)
    
    def test_cleanup_old_checkpoints(self):
        """Test cleanup of old checkpoints."""
        # Create 5 checkpoint directories
        steps = [100, 200, 300, 400, 500]
        for step in steps:
            (self.temp_dir / f"step_{step:08d}").mkdir()
        
        # Keep last 3
        self.strategy.cleanup_old_checkpoints(keep_last_n=3)
        
        remaining_steps = self.strategy.list_available_steps()
        assert remaining_steps == [300, 400, 500]
    
    def test_cleanup_with_zero_keep(self):
        """Test that cleanup with keep_last_n=0 raises error."""
        with pytest.raises(CheckpointError) as exc_info:
            self.strategy.cleanup_old_checkpoints(keep_last_n=0)
        
        assert "must be positive" in str(exc_info.value)
    
    def test_cleanup_fewer_than_keep(self):
        """Test cleanup when fewer checkpoints exist than keep_last_n."""
        # Create 2 checkpoints, try to keep 5
        (self.temp_dir / "step_00001000").mkdir()
        (self.temp_dir / "step_00002000").mkdir()
        
        # Should not remove anything
        self.strategy.cleanup_old_checkpoints(keep_last_n=5)
        
        remaining_steps = self.strategy.list_available_steps()
        assert remaining_steps == [1000, 2000]


class TestCheckpointUtilities:
    """Test checkpoint utility functions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.strategy = BaseCheckpointStrategy(self.temp_dir)
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_resolve_checkpoint_step_specific(self):
        """Test resolving specific checkpoint step."""
        # Create checkpoint
        (self.temp_dir / "step_00001000").mkdir()
        
        step = resolve_checkpoint_step(self.strategy, step=1000)
        assert step == 1000
    
    def test_resolve_checkpoint_step_specific_not_found(self):
        """Test resolving non-existent specific step."""
        with pytest.raises(CheckpointError) as exc_info:
            resolve_checkpoint_step(self.strategy, step=1000)
        
        assert "Checkpoint for step 1000 not found" in str(exc_info.value)
    
    def test_resolve_checkpoint_step_latest(self):
        """Test resolving latest checkpoint step."""
        # Create checkpoints
        (self.temp_dir / "step_00001000").mkdir()
        (self.temp_dir / "step_00002000").mkdir()
        
        step = resolve_checkpoint_step(self.strategy, step=None)
        assert step == 2000
    
    def test_resolve_checkpoint_step_latest_none_available(self):
        """Test resolving latest when no checkpoints exist."""
        with pytest.raises(CheckpointError) as exc_info:
            resolve_checkpoint_step(self.strategy, step=None)
        
        assert "No checkpoints available" in str(exc_info.value)
    
    def test_resolve_checkpoint_step_custom_strategy(self):
        """Test resolving with custom strategy (not BaseCheckpointStrategy)."""
        # Create mock custom strategy
        custom_strategy = Mock()
        
        # Should require explicit step for custom strategies
        with pytest.raises(CheckpointError) as exc_info:
            resolve_checkpoint_step(custom_strategy, step=None)
        
        assert "Step must be specified" in str(exc_info.value)
        
        # Should work with explicit step
        step = resolve_checkpoint_step(custom_strategy, step=1000)
        assert step == 1000
    
    def test_validate_checkpoint_compatibility_strict_match(self):
        """Test strict compatibility validation with matching specs."""
        metadata = CheckpointMetadata(
            step=1000,
            timestamp=1234567890.0,
            titanax_version="0.1.0",
            jax_version="0.4.0",
            mesh_spec={"axes": ["data"]},
            plan_spec={"data_parallel": {"axis": "data"}}
        )
        
        # Should not raise with matching specs
        validate_checkpoint_compatibility(
            metadata,
            current_mesh_spec={"axes": ["data"]},
            current_plan_spec={"data_parallel": {"axis": "data"}},
            strict=True
        )
    
    def test_validate_checkpoint_compatibility_strict_mismatch(self):
        """Test strict compatibility validation with mismatched specs."""
        metadata = CheckpointMetadata(
            step=1000,
            timestamp=1234567890.0,
            titanax_version="0.1.0",
            jax_version="0.4.0",
            mesh_spec={"axes": ["data"]},
            plan_spec={"data_parallel": {"axis": "data"}}
        )
        
        # Should raise with mismatched mesh
        with pytest.raises(CheckpointError) as exc_info:
            validate_checkpoint_compatibility(
                metadata,
                current_mesh_spec={"axes": ["data", "model"]},
                strict=True
            )
        assert "Mesh specification mismatch" in str(exc_info.value)
        
        # Should raise with mismatched plan
        with pytest.raises(CheckpointError) as exc_info:
            validate_checkpoint_compatibility(
                metadata,
                current_plan_spec={"tensor_parallel": {"axis": "model"}},
                strict=True
            )
        assert "Plan specification mismatch" in str(exc_info.value)
    
    def test_validate_checkpoint_compatibility_non_strict(self):
        """Test non-strict compatibility validation."""
        metadata = CheckpointMetadata(
            step=1000,
            timestamp=1234567890.0,
            titanax_version="0.1.0",
            jax_version="0.4.0",
            mesh_spec={"axes": ["data"]},
            plan_spec={"data_parallel": {"axis": "data"}}
        )
        
        # Should allow mismatches in non-strict mode
        validate_checkpoint_compatibility(
            metadata,
            current_mesh_spec={"axes": ["data", "model"]},
            current_plan_spec={"tensor_parallel": {"axis": "model"}},
            strict=False
        )


class TestOrbaxCheckpoint:
    """Test OrbaxCheckpoint implementation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_initialization(self, mock_checkpointer_class):
        """Test OrbaxCheckpoint initialization."""
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        strategy = OrbaxCheckpoint(
            self.temp_dir,
            save_interval_steps=500,
            keep_checkpoints=5,
            async_save=False
        )
        
        assert strategy.checkpoint_dir == self.temp_dir
        assert strategy.save_interval_steps == 500
        assert strategy.keep_checkpoints == 5
        assert strategy.async_save == False
        assert strategy.checkpointer == mock_checkpointer
        assert strategy._last_saved_step == -1
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_initialization_orbax_failure(self, mock_checkpointer_class):
        """Test initialization failure when Orbax is unavailable."""
        mock_checkpointer_class.side_effect = ImportError("orbax not found")
        
        with pytest.raises(CheckpointError) as exc_info:
            OrbaxCheckpoint(self.temp_dir)
        
        assert "Failed to initialize Orbax checkpointer" in str(exc_info.value)
        assert "orbax-checkpoint is installed" in str(exc_info.value)
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_should_save_interval(self, mock_checkpointer_class):
        """Test should_save logic with intervals."""
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        strategy = OrbaxCheckpoint(self.temp_dir, save_interval_steps=100)
        
        # Should save initially (no previous save)
        assert strategy.should_save(50)
        assert strategy.should_save(100)
        
        # Simulate saving at step 100
        strategy._last_saved_step = 100
        
        # Should not save until interval passed
        assert not strategy.should_save(150)
        assert not strategy.should_save(199)
        assert strategy.should_save(200)
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_should_save_disabled(self, mock_checkpointer_class):
        """Test should_save with disabled intervals."""
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        strategy = OrbaxCheckpoint(self.temp_dir, save_interval_steps=0)
        
        # Should never auto-save when disabled
        assert not strategy.should_save(100)
        assert not strategy.should_save(1000)
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    @patch('src.titanax.io.orbax_io.time.time')
    def test_save_basic(self, mock_time, mock_checkpointer_class):
        """Test basic save functionality."""
        mock_time.return_value = 1234567890.0
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        strategy = OrbaxCheckpoint(
            self.temp_dir,
            save_interval_steps=0,  # Disable interval checking
            keep_checkpoints=0,     # Disable cleanup
        )
        
        # Create test state
        test_state = {
            'params': {'layer1': jnp.array([1.0, 2.0])},
            'opt_state': {},
            'step': 1000
        }
        
        # Save
        strategy.save(test_state, step=1000)
        
        # Verify checkpointer.save was called
        mock_checkpointer.save.assert_called_once()
        
        # Check call arguments
        call_args = mock_checkpointer.save.call_args[0]
        save_path = call_args[0]
        checkpoint_data = call_args[1]
        
        assert str(save_path).endswith('step_00001000/checkpoint')
        assert 'state' in checkpoint_data
        assert 'metadata' in checkpoint_data
        assert checkpoint_data['state'] == test_state
        
        # Check metadata
        metadata = checkpoint_data['metadata']
        assert metadata['step'] == 1000
        assert metadata['timestamp'] == 1234567890.0
        
        # Check that last saved step was updated
        assert strategy._last_saved_step == 1000
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')  
    def test_save_with_interval_skip(self, mock_checkpointer_class):
        """Test that save skips when interval not reached."""
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        strategy = OrbaxCheckpoint(self.temp_dir, save_interval_steps=100)
        strategy._last_saved_step = 900
        
        test_state = {'step': 950}
        
        # Should skip saving (950 - 900 < 100)
        strategy.save(test_state, step=950)
        
        # Verify no save occurred
        mock_checkpointer.save.assert_not_called()
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_save_failure(self, mock_checkpointer_class):
        """Test save failure handling."""
        mock_checkpointer = Mock()
        mock_checkpointer.save.side_effect = Exception("Save failed")
        mock_checkpointer_class.return_value = mock_checkpointer
        
        strategy = OrbaxCheckpoint(self.temp_dir, save_interval_steps=0)
        
        test_state = {'step': 1000}
        
        with pytest.raises(CheckpointError) as exc_info:
            strategy.save(test_state, step=1000)
        
        assert "Failed to save checkpoint at step 1000" in str(exc_info.value)
        assert "disk space and permissions" in str(exc_info.value)
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_load_basic(self, mock_checkpointer_class):
        """Test basic load functionality."""
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        # Create checkpoint directory
        checkpoint_dir = self.temp_dir / "step_00001000"
        checkpoint_dir.mkdir()
        
        # Mock restore return value
        test_state = {'params': {'layer1': jnp.array([1.0, 2.0])}}
        checkpoint_data = {
            'state': test_state,
            'metadata': {
                'step': 1000,
                'timestamp': 1234567890.0,
                'titanax_version': '0.1.0',
                'jax_version': '0.4.0'
            }
        }
        mock_checkpointer.restore.return_value = checkpoint_data
        
        strategy = OrbaxCheckpoint(self.temp_dir, validate_compatibility=False)
        
        # Load specific step
        loaded_state = strategy.load(step=1000)
        
        assert loaded_state == test_state
        mock_checkpointer.restore.assert_called_once()
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_load_latest(self, mock_checkpointer_class):
        """Test loading latest checkpoint."""
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        # Create multiple checkpoint directories
        (self.temp_dir / "step_00001000").mkdir()
        (self.temp_dir / "step_00002000").mkdir()
        (self.temp_dir / "step_00001500").mkdir()
        
        test_state = {'params': {}}
        checkpoint_data = {
            'state': test_state,
            'metadata': {'step': 2000}
        }
        mock_checkpointer.restore.return_value = checkpoint_data
        
        strategy = OrbaxCheckpoint(self.temp_dir, validate_compatibility=False)
        
        # Load latest (should be step 2000)
        loaded_state = strategy.load(step=None)
        
        assert loaded_state == test_state
        
        # Verify it tried to load from step 2000
        call_args = mock_checkpointer.restore.call_args[0]
        restore_path = call_args[0]
        assert str(restore_path).endswith('step_00002000/checkpoint')
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_load_failure(self, mock_checkpointer_class):
        """Test load failure handling."""
        mock_checkpointer = Mock()
        mock_checkpointer.restore.side_effect = Exception("Load failed")
        mock_checkpointer_class.return_value = mock_checkpointer
        
        # Create checkpoint directory
        (self.temp_dir / "step_00001000").mkdir()
        
        strategy = OrbaxCheckpoint(self.temp_dir)
        
        with pytest.raises(CheckpointError) as exc_info:
            strategy.load(step=1000)
        
        assert "Failed to load checkpoint" in str(exc_info.value)
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_restore_alias(self, mock_checkpointer_class):
        """Test that restore() is an alias for load()."""
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        # Create checkpoint directory
        (self.temp_dir / "step_00001000").mkdir()
        
        test_state = {'params': {}}
        checkpoint_data = {'state': test_state, 'metadata': {}}
        mock_checkpointer.restore.return_value = checkpoint_data
        
        strategy = OrbaxCheckpoint(self.temp_dir, validate_compatibility=False)
        
        # Test that restore works the same as load
        dummy_state = {}
        restored_state = strategy.restore(dummy_state, step=1000)
        
        assert restored_state == test_state


class TestCheckpointFactory:
    """Test checkpoint strategy factory function."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
    
    def teardown_method(self):
        """Clean up test fixtures."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    @patch('src.titanax.io.orbax_io.PyTreeCheckpointer')
    def test_create_orbax_strategy(self, mock_checkpointer_class):
        """Test creating Orbax strategy through factory."""
        mock_checkpointer = Mock()
        mock_checkpointer_class.return_value = mock_checkpointer
        
        strategy = create_checkpoint_strategy(
            self.temp_dir,
            strategy="orbax",
            save_interval_steps=500
        )
        
        assert isinstance(strategy, OrbaxCheckpoint)
        assert strategy.save_interval_steps == 500
    
    def test_create_unsupported_strategy(self):
        """Test error for unsupported strategy."""
        with pytest.raises(CheckpointError) as exc_info:
            create_checkpoint_strategy(self.temp_dir, strategy="unsupported")
        
        assert "Unsupported checkpoint strategy" in str(exc_info.value)
        assert "orbax" in str(exc_info.value)
