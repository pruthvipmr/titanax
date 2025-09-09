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

        with patch("jax.devices", return_value=devices):
            mesh_spec = MeshSpec(axes=("data",))
            description = mesh_spec.describe()

            assert "1-dimensional" in description
            assert "data" in description
            assert str(len(devices)) in description

    def test_deterministic_device_ordering(self):
        """Test that device ordering is deterministic across calls."""
        spec = MeshSpec(axes=("data",), devices="all")
        
        # Get devices multiple times and verify consistent ordering
        devices1 = spec._resolve_devices()
        devices2 = spec._resolve_devices() 
        devices3 = spec._resolve_devices()
        
        assert devices1 == devices2 == devices3, "Device ordering should be deterministic"
        
        # Verify devices are sorted by (platform, process_index, id)
        if len(devices1) > 1:
            for i in range(len(devices1) - 1):
                d1, d2 = devices1[i], devices1[i + 1]
                key1 = (d1.platform, d1.process_index, d1.id)
                key2 = (d2.platform, d2.process_index, d2.id)
                assert key1 <= key2, f"Devices not properly sorted: {key1} > {key2}"

    def test_deterministic_factorization(self):
        """Test that device count factorization is deterministic."""
        spec = MeshSpec(axes=("data", "model"))
        
        # Test multiple calls to factorization produce identical results
        factors1 = spec._factorize_device_count(8, 2)
        factors2 = spec._factorize_device_count(8, 2)
        factors3 = spec._factorize_device_count(8, 2)
        
        assert factors1 == factors2 == factors3, "Factorization should be deterministic"
        assert factors1 == (4, 2), f"Expected (4, 2), got {factors1}"
        
        # Test other device counts
        assert spec._factorize_device_count(16, 2) == (4, 4)
        assert spec._factorize_device_count(12, 3) == (3, 2, 2)
        assert spec._factorize_device_count(6, 2) == (3, 2)


class TestProcessGroups:
    """Test ProcessGroups functionality."""

    def test_process_groups_basic(self):
        """Test basic ProcessGroups functionality."""
        from src.titanax.runtime import ProcessGroups
        
        # Create a mesh for testing
        mesh_spec = MeshSpec(axes=("data",))
        mesh = mesh_spec.build()
        
        # Create ProcessGroups
        pg = ProcessGroups(mesh)
        
        # Test basic methods
        assert pg.size("data") >= 1
        assert pg.rank("data") >= 0
        assert pg.rank("data") < pg.size("data")
        
        # Test coordinates
        coords = pg.coords()
        assert "data" in coords
        assert coords["data"] == pg.rank("data")
        
        # Test description
        description = pg.describe()
        assert "ProcessGroups:" in description
        assert "data:" in description

    def test_process_groups_multi_axis(self):
        """Test ProcessGroups with multiple axes."""
        from src.titanax.runtime import ProcessGroups
        
        # Create a multi-axis mesh
        mesh_spec = MeshSpec(axes=("data", "model"), shape=(1, 1))  # Force 1x1 for single device
        mesh = mesh_spec.build()
        
        pg = ProcessGroups(mesh)
        
        # Test all axes exist
        assert pg.size("data") == 1
        assert pg.size("model") == 1
        assert pg.rank("data") == 0
        assert pg.rank("model") == 0
        
        # Test coordinates
        coords = pg.coords()
        assert len(coords) == 2
        assert coords["data"] == 0
        assert coords["model"] == 0

    def test_process_groups_validation(self):
        """Test ProcessGroups axis validation."""
        from src.titanax.runtime import ProcessGroups
        
        mesh_spec = MeshSpec(axes=("data",))
        mesh = mesh_spec.build()
        pg = ProcessGroups(mesh)
        
        # Test invalid axis
        with pytest.raises(MeshError, match="Axis 'invalid' not found"):
            pg.size("invalid")
        
        with pytest.raises(MeshError, match="Axis 'invalid' not found"):
            pg.rank("invalid")


class TestDistributedEnv:
    """Test distributed environment detection."""

    def test_detect_distributed_env_empty(self):
        """Test environment detection with no variables set."""
        with patch.dict("os.environ", {}, clear=True):
            env_vars = detect_distributed_env()

            assert env_vars["coordinator_address"] is None
            assert env_vars["coordinator_port"] is None
            assert env_vars["process_count"] is None
            assert env_vars["process_id"] is None

    def test_detect_distributed_env_jax_vars(self):
        """Test environment detection with JAX variables."""
        test_env = {
            "JAX_COORDINATOR_ADDRESS": "127.0.0.1",
            "JAX_COORDINATOR_PORT": "8080",
            "JAX_PROCESS_COUNT": "4",
            "JAX_PROCESS_ID": "0",
        }

        with patch.dict("os.environ", test_env, clear=True):
            env_vars = detect_distributed_env()

            assert env_vars["coordinator_address"] == "127.0.0.1"
            assert env_vars["coordinator_port"] == "8080"
            assert env_vars["process_count"] == "4"
            assert env_vars["process_id"] == "0"

    def test_detect_distributed_env_fallback_vars(self):
        """Test environment detection with fallback variables."""
        test_env = {
            "COORDINATOR_ADDRESS": "192.168.1.1",
            "WORLD_SIZE": "8",
            "RANK": "2",
        }

        with patch.dict("os.environ", test_env, clear=True):
            env_vars = detect_distributed_env()

            assert env_vars["coordinator_address"] == "192.168.1.1"
            assert env_vars["process_count"] == "8"
            assert env_vars["process_id"] == "2"


class TestDeviceEnumeration:
    """Test device enumeration utilities."""

    def test_enumerate_devices_all(self):
        """Test enumerating all devices."""
        devices = enumerate_devices()
        assert len(devices) >= 1  # At least CPU
        assert all(hasattr(d, "platform") for d in devices)

    def test_enumerate_devices_by_type(self):
        """Test enumerating devices by type."""
        cpu_devices = enumerate_devices(device_type="cpu")
        assert len(cpu_devices) >= 1
        assert all(d.platform.lower() == "cpu" for d in cpu_devices)

        # Test case insensitive
        cpu_devices_upper = enumerate_devices(device_type="CPU")
        assert len(cpu_devices_upper) == len(cpu_devices)


if __name__ == "__main__":
    pytest.main([__file__])
