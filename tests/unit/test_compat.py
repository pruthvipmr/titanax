"""Unit tests for JAX compatibility layer."""

import pytest
import jax

from src.titanax.compat import (
    pjit,
    shard_map,
    PartitionSpec,
    Mesh,
    NamedSharding,
    psum,
    pmean,
    all_gather,
    ppermute,
    get_jax_version,
    check_jax_compatibility,
    get_preferred_pjit,
    get_preferred_shard_map,
    has_collective_support,
    ensure_compatibility,
)


class TestCompatImports:
    """Test that compatibility imports work."""

    def test_core_jax_apis_imported(self):
        """Test that core JAX APIs are available."""
        # These should always be available in any supported JAX version
        assert pjit is not None, "pjit should be available"
        assert shard_map is not None, "shard_map should be available"
        assert PartitionSpec is not None, "PartitionSpec should be available"
        assert Mesh is not None, "Mesh should be available"

    def test_collective_operations_imported(self):
        """Test that collective operations are available."""
        assert psum is not None, "psum should be available"
        assert pmean is not None, "pmean should be available"
        # all_gather and ppermute might be None in some versions

    def test_sharding_types_imported(self):
        """Test that sharding types are available."""
        assert PartitionSpec is not None, "PartitionSpec should be available"
        assert Mesh is not None, "Mesh should be available"
        # NamedSharding might be None in older versions

    def test_compatibility_functions_work(self):
        """Test that compatibility utility functions work."""
        version = get_jax_version()
        assert isinstance(version, str), "JAX version should be a string"
        assert len(version) > 0, "JAX version should not be empty"

        # Check compatibility
        is_compatible = check_jax_compatibility()
        assert isinstance(
            is_compatible, bool
        ), "Compatibility check should return boolean"

        # Get preferred APIs
        preferred_pjit = get_preferred_pjit()
        assert preferred_pjit is not None, "Should have a preferred pjit implementation"

        preferred_shard_map = get_preferred_shard_map()
        assert (
            preferred_shard_map is not None
        ), "Should have a preferred shard_map implementation"

        # Check collective support
        has_collectives = has_collective_support()
        assert isinstance(
            has_collectives, bool
        ), "Collective support check should return boolean"

        # Test ensure_compatibility (should not raise)
        ensure_compatibility()

    def test_basic_functionality_with_compat_imports(self):
        """Test basic functionality using compatibility imports."""
        # Create a simple PartitionSpec
        spec = PartitionSpec("data")
        assert spec is not None

        # Create a basic mesh (should work on any device)
        devices = jax.devices()[:1]  # Use just one device for simplicity
        mesh = Mesh(devices, ("data",))
        assert mesh is not None
        assert mesh.axis_names == ("data",)

    def test_pjit_availability(self):
        """Test that pjit is available and functional."""
        preferred_pjit = get_preferred_pjit()
        assert preferred_pjit is not None
        assert callable(preferred_pjit)

    def test_shard_map_availability(self):
        """Test that shard_map is available and functional."""
        preferred_shard_map = get_preferred_shard_map()
        assert preferred_shard_map is not None
        assert callable(preferred_shard_map)


if __name__ == "__main__":
    pytest.main([__file__])
