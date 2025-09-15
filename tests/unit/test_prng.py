"""Tests for multi-device PRNG utilities."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch

from src.titanax.exec.prng import (
    create_per_device_rngs,
    update_per_device_rngs,
    split_per_device_rng,
    validate_rng_keys,
    create_host_device_rngs,
)


class TestPRNGUtilities:
    """Test proper multi-device PRNG key handling."""

    def test_create_host_device_rngs(self):
        """Test creation of host-specific device PRNG keys."""
        # Mock JAX device info
        with (
            patch("jax.process_index", return_value=0),
            patch("jax.local_device_count", return_value=2),
        ):

            rngs = create_host_device_rngs(
                base_seed=42, named_keys={"dropout": None, "params": None}
            )

            assert isinstance(rngs, dict)
            assert "dropout" in rngs
            assert "params" in rngs

            # Verify keys are valid JAX PRNGKeys
            for key_name, key in rngs.items():
                assert isinstance(key, jnp.ndarray)
                assert key.shape in [(2,), (4,)]  # Valid JAX PRNG key shapes
                assert jnp.issubdtype(key.dtype, jnp.unsignedinteger)

    def test_validate_rng_keys_valid(self):
        """Test validation of valid PRNG keys."""
        valid_rngs = {
            "dropout": jax.random.PRNGKey(42),
            "params": jax.random.PRNGKey(123),
        }

        # Should not raise any exception
        validate_rng_keys(valid_rngs)

    def test_validate_rng_keys_invalid_type(self):
        """Test validation with invalid key types."""
        invalid_rngs = {
            "dropout": 42,  # Not a JAX array
            "params": jax.random.PRNGKey(123),
        }

        with pytest.raises(ValueError, match="must be a JAX array"):
            validate_rng_keys(invalid_rngs)

    def test_validate_rng_keys_invalid_shape(self):
        """Test validation with invalid key shapes."""
        invalid_rngs = {
            "dropout": jnp.array([1, 2, 3], dtype=jnp.uint32),  # Wrong shape
            "params": jax.random.PRNGKey(123),
        }

        with pytest.raises(ValueError, match="invalid shape"):
            validate_rng_keys(invalid_rngs)

    def _create_mesh(self, axis_size: int = 1):
        devices = jax.devices()
        if len(devices) < axis_size:
            pytest.skip(f"requires {axis_size} devices, found {len(devices)}")
        device_array = np.array(devices[:axis_size]).reshape((axis_size,))
        return jax.sharding.Mesh(device_array, ("data",))

    def test_create_per_device_rngs_single_device(self):
        """Single-device meshes should receive deterministic keys."""
        mesh = self._create_mesh(axis_size=1)
        rngs = create_per_device_rngs(42, mesh, names=("dropout", "noise"))

        assert set(rngs.keys()) == {"dropout", "noise"}
        for value in rngs.values():
            assert value.shape == mesh.devices.shape + (2,)
            assert jnp.issubdtype(value.dtype, jnp.unsignedinteger)

        # Deterministic for the same seed and mesh
        rngs_2 = create_per_device_rngs(42, mesh, names=("dropout", "noise"))
        for key in rngs:
            assert jnp.array_equal(rngs[key], rngs_2[key])

    def test_create_per_device_rngs_multi_device_unique(self):
        """Different devices should receive different keys."""
        if len(jax.devices()) < 2:
            pytest.skip("requires at least 2 devices to validate uniqueness")

        mesh = self._create_mesh(axis_size=2)
        rngs = create_per_device_rngs(123, mesh, names=("dropout",))

        values = rngs["dropout"].reshape(-1, rngs["dropout"].shape[-1])
        unique_rows = {tuple(row.tolist()) for row in values}
        assert len(unique_rows) == values.shape[0]

    def test_update_per_device_rngs(self):
        """Updating RNGs should advance the stream while preserving shape."""
        mesh = self._create_mesh(axis_size=1)
        rngs = create_per_device_rngs(7, mesh, names=("dropout",))
        updated = update_per_device_rngs(rngs)
        updated_again = update_per_device_rngs(rngs)

        assert updated["dropout"].shape == rngs["dropout"].shape
        assert not jnp.array_equal(updated["dropout"], rngs["dropout"])
        assert jnp.array_equal(updated["dropout"], updated_again["dropout"])

    def test_split_per_device_rng(self):
        """Splitting should return multiple per-device dictionaries."""
        mesh = self._create_mesh(axis_size=1)
        rngs = create_per_device_rngs(0, mesh, names=("dropout",))
        splits = split_per_device_rng(rngs, num=3)

        assert len(splits) == 3
        for split in splits:
            assert set(split.keys()) == {"dropout"}
            assert split["dropout"].shape == rngs["dropout"].shape

    def test_fallback_without_mesh_context(self):
        """Test that functions work without mesh context (single device)."""
        # Test update_rngs fallback
        from src.titanax.exec.step_fn import update_rngs, split_rng

        initial_rngs = {"dropout": jax.random.PRNGKey(42)}

        # Should work without axis_index (no mesh context)
        updated_rngs = update_rngs(initial_rngs)
        assert "dropout" in updated_rngs
        assert not jnp.array_equal(updated_rngs["dropout"], initial_rngs["dropout"])

        # Test split_rng fallback
        base_rng = jax.random.PRNGKey(42)
        split_keys = split_rng(base_rng, num=2)
        assert len(split_keys) == 2
        assert not jnp.array_equal(split_keys[0], split_keys[1])

    def test_deterministic_reproduction(self):
        """Test that same seed produces same results."""
        seed = 12345
        named_keys = {"dropout": None, "params": None}

        # Mock same device conditions
        with (
            patch("jax.process_index", return_value=0),
            patch("jax.local_device_count", return_value=1),
        ):

            rngs1 = create_host_device_rngs(seed, named_keys)
            rngs2 = create_host_device_rngs(seed, named_keys)

            # Should produce identical keys for same seed
            for key_name in named_keys:
                assert jnp.array_equal(rngs1[key_name], rngs2[key_name])

    def test_different_hosts_different_keys(self):
        """Test that different hosts get different keys."""
        seed = 42
        named_keys = {"dropout": None}

        # Mock host 0
        with (
            patch("jax.process_index", return_value=0),
            patch("jax.local_device_count", return_value=1),
        ):
            rngs_host_0 = create_host_device_rngs(seed, named_keys)

        # Mock host 1
        with (
            patch("jax.process_index", return_value=1),
            patch("jax.local_device_count", return_value=1),
        ):
            rngs_host_1 = create_host_device_rngs(seed, named_keys)

        # Different hosts should get different keys
        assert not jnp.array_equal(rngs_host_0["dropout"], rngs_host_1["dropout"])


if __name__ == "__main__":
    pytest.main([__file__])
