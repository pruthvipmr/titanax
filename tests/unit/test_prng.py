"""Tests for multi-device PRNG utilities."""

import pytest
import jax
import jax.numpy as jnp
import numpy as np
from unittest.mock import patch, MagicMock

from src.titanax.exec.prng import (
    create_per_device_rngs,
    update_per_device_rngs,
    split_per_device_rng,
    validate_rng_keys,
    create_host_device_rngs
)


class TestPRNGUtilities:
    """Test proper multi-device PRNG key handling."""
    
    def test_create_host_device_rngs(self):
        """Test creation of host-specific device PRNG keys."""
        # Mock JAX device info
        with patch('jax.process_index', return_value=0), \
             patch('jax.local_device_count', return_value=2):
            
            rngs = create_host_device_rngs(
                base_seed=42,
                named_keys={'dropout': None, 'params': None}
            )
            
            assert isinstance(rngs, dict)
            assert 'dropout' in rngs
            assert 'params' in rngs
            
            # Verify keys are valid JAX PRNGKeys
            for key_name, key in rngs.items():
                assert isinstance(key, jnp.ndarray)
                assert key.shape in [(2,), (4,)]  # Valid JAX PRNG key shapes
                assert jnp.issubdtype(key.dtype, jnp.unsignedinteger)
    
    def test_validate_rng_keys_valid(self):
        """Test validation of valid PRNG keys."""
        valid_rngs = {
            'dropout': jax.random.PRNGKey(42),
            'params': jax.random.PRNGKey(123)
        }
        
        # Should not raise any exception
        validate_rng_keys(valid_rngs)
    
    def test_validate_rng_keys_invalid_type(self):
        """Test validation with invalid key types."""
        invalid_rngs = {
            'dropout': 42,  # Not a JAX array
            'params': jax.random.PRNGKey(123)
        }
        
        with pytest.raises(ValueError, match="must be a JAX array"):
            validate_rng_keys(invalid_rngs)
    
    def test_validate_rng_keys_invalid_shape(self):
        """Test validation with invalid key shapes."""
        invalid_rngs = {
            'dropout': jnp.array([1, 2, 3], dtype=jnp.uint32),  # Wrong shape
            'params': jax.random.PRNGKey(123)
        }
        
        with pytest.raises(ValueError, match="invalid shape"):
            validate_rng_keys(invalid_rngs)
    
    def test_create_per_device_rngs_mock_axis(self):
        """Test per-device RNG creation with mocked axis_index."""
        base_key = jax.random.PRNGKey(42)
        named_keys = {'dropout': None, 'noise': None}
        
        # Mock axis_index to simulate device 0
        with patch('jax.lax.axis_index', return_value=0):
            rngs = create_per_device_rngs(base_key, named_keys, axis='batch')
            
            assert len(rngs) == 2
            assert 'dropout' in rngs
            assert 'noise' in rngs
            
            # Verify keys are different from each other
            assert not jnp.array_equal(rngs['dropout'], rngs['noise'])
    
    def test_create_per_device_rngs_different_devices(self):
        """Test that different devices get different RNG keys."""
        base_key = jax.random.PRNGKey(42)
        named_keys = {'dropout': None}
        
        # Mock two different devices
        with patch('jax.lax.axis_index', return_value=0):
            rngs_device_0 = create_per_device_rngs(base_key, named_keys)
        
        with patch('jax.lax.axis_index', return_value=1):
            rngs_device_1 = create_per_device_rngs(base_key, named_keys)
        
        # Keys should be different between devices
        assert not jnp.array_equal(rngs_device_0['dropout'], rngs_device_1['dropout'])
    
    def test_update_per_device_rngs_mock_axis(self):
        """Test updating per-device RNG keys."""
        initial_rngs = {
            'dropout': jax.random.PRNGKey(42),
            'noise': jax.random.PRNGKey(123)
        }
        
        with patch('jax.lax.axis_index', return_value=0):
            updated_rngs = update_per_device_rngs(initial_rngs, axis='batch')
            
            # Keys should be updated (different from initial)
            assert not jnp.array_equal(updated_rngs['dropout'], initial_rngs['dropout'])
            assert not jnp.array_equal(updated_rngs['noise'], initial_rngs['noise'])
            
            # But still valid PRNG keys
            validate_rng_keys(updated_rngs)
    
    def test_update_per_device_rngs_partial_update(self):
        """Test updating only specific RNG keys."""
        initial_rngs = {
            'dropout': jax.random.PRNGKey(42),
            'noise': jax.random.PRNGKey(123)
        }
        
        with patch('jax.lax.axis_index', return_value=0):
            updated_rngs = update_per_device_rngs(
                initial_rngs, 
                keys_to_update=['dropout'],
                axis='batch'
            )
            
            # Only dropout should be updated
            assert not jnp.array_equal(updated_rngs['dropout'], initial_rngs['dropout'])
            assert jnp.array_equal(updated_rngs['noise'], initial_rngs['noise'])
    
    def test_split_per_device_rng_mock_axis(self):
        """Test splitting per-device RNG keys."""
        base_rng = jax.random.PRNGKey(42)
        
        with patch('jax.lax.axis_index', return_value=0):
            split_keys = split_per_device_rng(base_rng, num_splits=3, axis='batch')
            
            assert len(split_keys) == 3
            
            # All splits should be different
            for i in range(len(split_keys)):
                for j in range(i + 1, len(split_keys)):
                    assert not jnp.array_equal(split_keys[i], split_keys[j])
                    
            # All should be valid PRNG keys
            for key in split_keys:
                assert isinstance(key, jnp.ndarray)
                assert key.shape in [(2,), (4,)]
    
    def test_fallback_without_mesh_context(self):
        """Test that functions work without mesh context (single device)."""
        # Test update_rngs fallback
        from src.titanax.exec.step_fn import update_rngs, split_rng
        
        initial_rngs = {'dropout': jax.random.PRNGKey(42)}
        
        # Should work without axis_index (no mesh context)
        updated_rngs = update_rngs(initial_rngs)
        assert 'dropout' in updated_rngs
        assert not jnp.array_equal(updated_rngs['dropout'], initial_rngs['dropout'])
        
        # Test split_rng fallback
        base_rng = jax.random.PRNGKey(42)
        split_keys = split_rng(base_rng, num=2)
        assert len(split_keys) == 2
        assert not jnp.array_equal(split_keys[0], split_keys[1])
    
    def test_deterministic_reproduction(self):
        """Test that same seed produces same results."""
        seed = 12345
        named_keys = {'dropout': None, 'params': None}
        
        # Mock same device conditions
        with patch('jax.process_index', return_value=0), \
             patch('jax.local_device_count', return_value=1):
            
            rngs1 = create_host_device_rngs(seed, named_keys)
            rngs2 = create_host_device_rngs(seed, named_keys)
            
            # Should produce identical keys for same seed
            for key_name in named_keys:
                assert jnp.array_equal(rngs1[key_name], rngs2[key_name])
    
    def test_different_hosts_different_keys(self):
        """Test that different hosts get different keys."""
        seed = 42
        named_keys = {'dropout': None}
        
        # Mock host 0
        with patch('jax.process_index', return_value=0), \
             patch('jax.local_device_count', return_value=1):
            rngs_host_0 = create_host_device_rngs(seed, named_keys)
        
        # Mock host 1  
        with patch('jax.process_index', return_value=1), \
             patch('jax.local_device_count', return_value=1):
            rngs_host_1 = create_host_device_rngs(seed, named_keys)
        
        # Different hosts should get different keys
        assert not jnp.array_equal(rngs_host_0['dropout'], rngs_host_1['dropout'])


if __name__ == '__main__':
    pytest.main([__file__])
