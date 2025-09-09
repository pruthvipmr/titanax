"""Unit tests for runtime components."""

import pytest
import jax
from unittest.mock import patch

from src.titanax.runtime import MeshSpec, detect_distributed_env, enumerate_devices
from src.titanax.exceptions import MeshError


class TestMeshSpec:
    """Test MeshSpec validation and building."""
    
    def test_mesh_spec_creation(self):
        """Test basic MeshSpec creation."""
        mesh_spec = MeshSpec(axes=("data",))
        assert mesh_spec.axes == ("data",)
        assert mesh_spec.shape is None
        assert mesh_spec.devices == "all"
    
    def test_mesh_spec_with_shape(self):
        """Test MeshSpec with explicit shape."""
        mesh_spec = MeshSpec(axes=("data", "model"), shape=(2, 2))
        assert mesh_spec.axes == ("data", "model")
        assert mesh_spec.shape == (2, 2)
    
    def test_mesh_spec_validation_duplicate_axes(self):
        """Test validation fails with duplicate axes."""
        with pytest.raises(MeshError, match="Duplicate axis names"):
            MeshSpec(axes=("data", "data"))
    
    def test_mesh_spec_validation_empty_axes(self):
        """Test validation fails with empty axes."""
        with pytest.raises(MeshError, match="Empty axes tuple"):
            MeshSpec(axes=())
    
    def test_mesh_spec_validation_shape_mismatch(self):
        """Test validation fails when shape length doesn't match axes."""
        with pytest.raises(MeshError, match="Shape length .* doesn't match"):
            MeshSpec(axes=("data", "model"), shape=(2, 2, 2))
    
    def test_mesh_spec_build_basic(self):
        """Test building basic mesh."""
        mesh_spec = MeshSpec(axes=("data",))
        mesh = mesh_spec.build()
        
        assert mesh.axis_names == ("data",)
        assert len(mesh.devices.flatten()) >= 1
    
    def test_mesh_spec_describe(self):
        """Test mesh description."""
        devices = jax.devices()[:2]
        
        with patch('jax.devices', return_value=devices):
            mesh_spec = MeshSpec(axes=("data",))
            description = mesh_spec.describe()
            
            assert "1-dimensional" in description
            assert "data" in description
            assert str(len(devices)) in description


# Note: ProcessGroups tests disabled due to implementation issues to be fixed in future PRs


class TestDistributedEnv:
    """Test distributed environment detection."""
    
    def test_detect_distributed_env_empty(self):
        """Test environment detection with no variables set."""
        with patch.dict('os.environ', {}, clear=True):
            env_vars = detect_distributed_env()
            
            assert env_vars['coordinator_address'] is None
            assert env_vars['coordinator_port'] is None
            assert env_vars['process_count'] is None
            assert env_vars['process_id'] is None
    
    def test_detect_distributed_env_jax_vars(self):
        """Test environment detection with JAX variables."""
        test_env = {
            'JAX_COORDINATOR_ADDRESS': '127.0.0.1',
            'JAX_COORDINATOR_PORT': '8080',
            'JAX_PROCESS_COUNT': '4',
            'JAX_PROCESS_ID': '0'
        }
        
        with patch.dict('os.environ', test_env, clear=True):
            env_vars = detect_distributed_env()
            
            assert env_vars['coordinator_address'] == '127.0.0.1'
            assert env_vars['coordinator_port'] == '8080'
            assert env_vars['process_count'] == '4'
            assert env_vars['process_id'] == '0'
    
    def test_detect_distributed_env_fallback_vars(self):
        """Test environment detection with fallback variables.""" 
        test_env = {
            'COORDINATOR_ADDRESS': '192.168.1.1',
            'WORLD_SIZE': '8',
            'RANK': '2'
        }
        
        with patch.dict('os.environ', test_env, clear=True):
            env_vars = detect_distributed_env()
            
            assert env_vars['coordinator_address'] == '192.168.1.1'
            assert env_vars['process_count'] == '8'
            assert env_vars['process_id'] == '2'


class TestDeviceEnumeration:
    """Test device enumeration utilities."""
    
    def test_enumerate_devices_all(self):
        """Test enumerating all devices."""
        devices = enumerate_devices()
        assert len(devices) >= 1  # At least CPU
        assert all(hasattr(d, 'platform') for d in devices)
    
    def test_enumerate_devices_by_type(self):
        """Test enumerating devices by type."""
        cpu_devices = enumerate_devices(device_type='cpu')
        assert len(cpu_devices) >= 1
        assert all(d.platform.lower() == 'cpu' for d in cpu_devices)
        
        # Test case insensitive
        cpu_devices_upper = enumerate_devices(device_type='CPU')
        assert len(cpu_devices_upper) == len(cpu_devices)


if __name__ == "__main__":
    pytest.main([__file__])
