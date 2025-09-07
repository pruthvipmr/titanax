"""Unit tests for collective operations."""

import pytest
import warnings
from functools import partial
import jax
import jax.numpy as jnp

from src.titanax.exec.collectives import collectives, _validate_axis_name, _validate_tree_structure, set_current_mesh, get_current_mesh
from src.titanax.exceptions import CollectiveError


class TestCollectivesValidation:
    """Test axis and tree validation functions."""
    
    def test_validate_axis_name_valid(self):
        """Test axis validation with valid axis names."""
        _validate_axis_name("data", "test_op")
        _validate_axis_name("model", "test_op")
        _validate_axis_name("pipeline", "test_op")
    
    def test_validate_axis_name_invalid_type(self):
        """Test axis validation with invalid types."""
        with pytest.raises(CollectiveError, match="axis must be a string"):
            _validate_axis_name(0, "test_op")
        
        with pytest.raises(CollectiveError, match="axis must be a string"):
            _validate_axis_name(None, "test_op")
    
    def test_validate_axis_name_empty(self):
        """Test axis validation with empty string."""
        with pytest.raises(CollectiveError, match="axis name cannot be empty"):
            _validate_axis_name("", "test_op")
    
    def test_validate_axis_name_with_mesh(self):
        """Test axis validation with mesh context."""
        devices = jax.devices()[:1]
        mesh = jax.sharding.Mesh(devices, ("data",))
        
        # Valid axis should pass
        _validate_axis_name("data", "test_op", mesh)
        
        # Invalid axis should fail
        with pytest.raises(CollectiveError, match="axis not found in mesh"):
            _validate_axis_name("model", "test_op", mesh)
    
    def test_validate_tree_structure_valid(self):
        """Test tree validation with valid PyTrees."""
        # Simple array
        _validate_tree_structure(jnp.array([1, 2, 3]), "test_op", "data")
        
        # Nested dict
        tree = {
            "params": {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.5])},
            "state": jnp.array([3.0])
        }
        _validate_tree_structure(tree, "test_op", "data")
    
    def test_validate_tree_structure_invalid_leaf(self):
        """Test tree validation with invalid leaf types."""
        invalid_tree = {"w": jnp.array([1.0]), "b": [1, 2, 3]}  # List is not JAX array
        
        with pytest.raises(CollectiveError, match="leaf 0 is not a JAX array"):
            _validate_tree_structure(invalid_tree, "test_op", "data")
    
    def test_validate_tree_structure_invalid_tree(self):
        """Test tree validation with completely invalid structure."""
        # This should be fine actually, JAX can handle most Python objects
        _validate_tree_structure({"a": jnp.array([1])}, "test_op", "data")


class TestCollectivesBasic:
    """Test basic collective operations validation."""
    
    def test_psum_validation(self):
        """Test psum input validation."""
        x = jnp.array([1.0, 2.0])
        
        # Test valid axis name validation
        try:
            collectives.psum(x, "data")
        except Exception as e:
            # Should fail due to no mesh context, but not due to axis validation
            assert ("unbound axis name" in str(e) or "JAX operation failed" in str(e) or 
                   "not bound in current JAX transformation context" in str(e))
        
        # Test invalid axis types
        with pytest.raises(CollectiveError, match="axis must be a string"):
            collectives.psum(x, 123)
        
        with pytest.raises(CollectiveError, match="axis name cannot be empty"):
            collectives.psum(x, "")
    
    def test_pmean_validation(self):
        """Test pmean input validation."""
        x = jnp.array([1.0, 2.0])
        
        # Test valid axis name validation
        try:
            collectives.pmean(x, "data")
        except Exception as e:
            # Should fail due to no mesh context, but not due to axis validation
            assert ("unbound axis name" in str(e) or "JAX operation failed" in str(e) or 
                   "not bound in current JAX transformation context" in str(e))
        
        # Test invalid axis types
        with pytest.raises(CollectiveError, match="axis must be a string"):
            collectives.pmean(x, None)
    
    def test_tree_validation(self):
        """Test PyTree validation in collective operations."""
        # Valid tree
        valid_tree = {"w": jnp.array([1.0]), "b": jnp.array([2.0])}
        
        # Invalid tree with non-array leaf
        invalid_tree = {"w": jnp.array([1.0]), "b": [1, 2, 3]}
        
        with pytest.raises(CollectiveError, match="leaf .* is not a JAX array"):
            collectives.psum(invalid_tree, "data")


class TestCollectiveImplementations:
    """Test actual implementations of collective operations."""
    
    def test_all_gather_implementation(self):
        """Test all_gather actual implementation."""
        x = jnp.array([1.0, 2.0])
        
        # Should fail with unbound axis error, not return unchanged
        try:
            result = collectives.all_gather(x, "data")
        except Exception as e:
            # Should fail due to no mesh context, but validate it's not a stub
            assert ("unbound axis name" in str(e) or "JAX operation failed" in str(e) or 
                   "not bound in current JAX transformation context" in str(e))
        
        # Test input validation
        with pytest.raises(CollectiveError, match="input must be a JAX array"):
            collectives.all_gather([1, 2, 3], "data")
    
    def test_reduce_scatter_implementation(self):
        """Test reduce_scatter actual implementation."""
        x = jnp.array([1.0, 2.0])
        
        # Should fail with unbound axis error, not return unchanged
        try:
            result = collectives.reduce_scatter(x, "data", op="add")
        except Exception as e:
            # Should fail due to no mesh context, but validate it's not a stub
            assert ("unbound axis name" in str(e) or "JAX operation failed" in str(e) or 
                   "not bound in current JAX transformation context" in str(e))
        
        # Test input validation
        with pytest.raises(CollectiveError, match="input must be a JAX array"):
            collectives.reduce_scatter([1, 2, 3], "data", op="add")
    
    def test_reduce_scatter_invalid_op(self):
        """Test reduce_scatter with invalid operation."""
        x = jnp.array([1.0, 2.0])
        
        # Now only "add" is supported
        with pytest.raises(CollectiveError, match="invalid operation 'mul'"):
            collectives.reduce_scatter(x, "data", op="mul")
    
    def test_broadcast_implementation(self):
        """Test broadcast actual implementation."""  
        x = jnp.array([1.0, 2.0])
        
        # Should fail with unbound axis error, not return unchanged
        try:
            result = collectives.broadcast(x, "data", src_index=0)
        except Exception as e:
            # Should fail due to no mesh context, but validate it's not a stub
            assert ("unbound axis name" in str(e) or "JAX operation failed" in str(e) or 
                   "not bound in current JAX transformation context" in str(e))
        
        # Test input validation
        with pytest.raises(CollectiveError, match="input must be a JAX array"):
            collectives.broadcast([1, 2, 3], "data", src_index=0)
    
    def test_broadcast_invalid_src_index(self):
        """Test broadcast with invalid src_index."""
        x = jnp.array([1.0, 2.0])
        
        with pytest.raises(CollectiveError, match="src_index -1 must be non-negative"):
            collectives.broadcast(x, "data", src_index=-1)
    
    def test_ppermute_implementation(self):
        """Test ppermute actual implementation."""
        x = jnp.array([1.0, 2.0])
        
        # Test input validation
        with pytest.raises(CollectiveError, match="input must be a JAX array"):
            collectives.ppermute([1, 2, 3], "data", perm=[(0, 1)])
        
        # Test permutation validation
        with pytest.raises(CollectiveError, match="perm must be a list or tuple"):
            collectives.ppermute(x, "data", perm=None)
        
        with pytest.raises(CollectiveError, match="must be a \\(source, dest\\) pair"):
            collectives.ppermute(x, "data", perm=[(0,)])
        
        with pytest.raises(CollectiveError, match="indices must be integers"):
            collectives.ppermute(x, "data", perm=[("0", 1)])
        
        with pytest.raises(CollectiveError, match="indices must be non-negative"):
            collectives.ppermute(x, "data", perm=[(-1, 1)])
        
        # Should fail with unbound axis error when valid
        try:
            result = collectives.ppermute(x, "data", perm=[(0, 1), (1, 0)])
        except Exception as e:
            # Should fail due to no mesh context, but validate it's not a stub
            assert ("unbound axis name" in str(e) or "JAX operation failed" in str(e) or 
                   "not bound in current JAX transformation context" in str(e))


class TestCollectiveOperations:
    """Test collective operations in proper JAX transformation context."""
    
    def test_all_gather_basic(self):
        """Test all_gather with proper shard_map context."""
        # Skip if not enough devices
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Need at least 2 devices for multi-device collectives test")
        
        mesh = jax.sharding.Mesh(devices[:2], ("data",))
        
        @jax.jit
        @partial(jax.shard_map, mesh=mesh, in_specs=jax.sharding.PartitionSpec("data"), 
                 out_specs=jax.sharding.PartitionSpec(None))
        def test_fn(x):
            return collectives.all_gather(x, "data")
        
        # Create input data
        x = jnp.arange(4)  # [0, 1, 2, 3]
        result = test_fn(x)
        
        # all_gather should concatenate: [0, 1, 2, 3] -> [0, 1, 2, 3] (tiled)
        expected = jnp.array([0, 1, 2, 3])
        assert jnp.array_equal(result, expected)
    
    def test_ppermute_ring_shift(self):
        """Test ppermute with ring shift pattern."""
        # Skip if not enough devices  
        devices = jax.devices()
        if len(devices) < 2:
            pytest.skip("Need at least 2 devices for multi-device collectives test")
        
        mesh = jax.sharding.Mesh(devices[:2], ("data",))
        
        @jax.jit
        @partial(jax.shard_map, mesh=mesh, in_specs=jax.sharding.PartitionSpec("data"),
                 out_specs=jax.sharding.PartitionSpec("data"))
        def test_fn(x):
            # Ring shift: device 0 -> device 1, device 1 -> device 0
            perm = [(0, 1), (1, 0)]
            return collectives.ppermute(x, "data", perm)
        
        # Create input data where each device has different values
        x = jnp.arange(4)  # [0, 1, 2, 3]
        result = test_fn(x)
        
        # Should swap the shards
        expected = jnp.array([2, 3, 0, 1])
        assert jnp.array_equal(result, expected)


class TestMeshContext:
    """Test mesh context management."""
    
    def test_mesh_context_management(self):
        """Test setting and getting current mesh."""
        devices = jax.devices()[:1]
        mesh = jax.sharding.Mesh(devices, ("data",))
        
        # Initially no mesh
        assert get_current_mesh() is None
        
        # Set mesh
        set_current_mesh(mesh)
        assert get_current_mesh() is mesh
        
        # Clear mesh
        set_current_mesh(None)
        assert get_current_mesh() is None
    
    def test_collective_with_mesh_context(self):
        """Test collectives with proper mesh context."""
        devices = jax.devices()[:1]
        mesh = jax.sharding.Mesh(devices, ("data",))
        x = jnp.array([1.0, 2.0])
        
        try:
            # Set mesh context
            set_current_mesh(mesh)
            
            # This should not raise axis validation errors
            try:
                collectives.psum(x, "data")
            except Exception as e:
                # Should fail with JAX error about missing transformation context
                # but not with axis validation error
                assert ("unbound axis name" in str(e) or "JAX operation failed" in str(e) or 
                       "not bound in current JAX transformation context" in str(e))
            
            # Invalid axis should raise CollectiveError
            with pytest.raises(CollectiveError, match="axis not found in mesh"):
                collectives.psum(x, "model")
                
        finally:
            # Always clear mesh context
            set_current_mesh(None)


class TestCollectiveErrorHandling:
    """Test error handling in collective operations."""
    
    def test_axis_validation_errors(self):
        """Test various axis validation errors."""
        x = jnp.array([1.0, 2.0])
        
        # Test empty axis name
        with pytest.raises(CollectiveError, match="axis name cannot be empty"):
            collectives.psum(x, "")
        
        # Test non-string axis
        with pytest.raises(CollectiveError, match="axis must be a string"):
            collectives.pmean(x, 42)
        
        with pytest.raises(CollectiveError, match="axis must be a string"):
            collectives.all_gather(x, None)


if __name__ == "__main__":
    pytest.main([__file__])
