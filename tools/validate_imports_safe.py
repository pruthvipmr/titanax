#!/usr/bin/env python3
"""Safe import validation that works without heavyweight dependencies.

This script validates that all Titanax imports work correctly by:
1. Providing lightweight stubs for JAX/Optax/Orbax
2. Byte-compiling all modules to check syntax
3. Testing actual imports to verify public API
"""

import sys
import types
import pathlib
import compileall
import traceback
from typing import List, Tuple

# Add src directory to path for direct imports
src_dir = pathlib.Path(__file__).parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))


def create_stubs():
    """Create minimal stubs for heavyweight dependencies."""
    
    # JAX stub
    if "jax" not in sys.modules:
        jax_stub = types.ModuleType("jax")
        sys.modules["jax"] = jax_stub
        
        # Add minimal attributes Titanax uses at import time
        jax_stub.Device = object
        jax_stub.Array = object  
        jax_stub.value_and_grad = lambda f: f
        jax_stub.grad = lambda f: f
        jax_stub.jit = lambda f: f
        jax_stub.tree_util = types.ModuleType("tree_util")
        jax_stub.tree_util.tree_map = lambda f, x: x
        jax_stub.tree_util.tree_flatten = lambda x: ([], None)
        jax_stub.tree_util.tree_unflatten = lambda treedef, leaves: {}
        jax_stub.tree_util.register_pytree_node = lambda cls, flatten, unflatten: None
        jax_stub.lax = types.ModuleType("lax")
        jax_stub.lax.psum = lambda x, axis: x
        jax_stub.lax.pmean = lambda x, axis: x
        jax_stub.sharding = types.ModuleType("sharding") 
        jax_stub.sharding.Mesh = object
        jax_stub.sharding.PartitionSpec = object
        jax_stub.sharding.NamedSharding = object
    
    # JAX.numpy stub
    if "jax.numpy" not in sys.modules:
        jnp_stub = types.ModuleType("jax.numpy")
        sys.modules["jax.numpy"] = jnp_stub
        jax_stub.numpy = jnp_stub
    
    # Optax stub
    if "optax" not in sys.modules:
        optax_stub = types.ModuleType("optax")
        sys.modules["optax"] = optax_stub
        optax_stub.adamw = lambda lr: object()
        optax_stub.sgd = lambda lr: object()
        optax_stub.adam = lambda lr: object()
        optax_stub.cosine_decay_schedule = lambda init: object()
        optax_stub.exponential_decay = lambda init, decay: object()
        optax_stub.warmup_cosine_decay_schedule = lambda init, peak, warmup, decay: object()
        optax_stub.GradientTransformation = object
        
        # Add typing support
        optax_stub.typing = types.ModuleType("typing")
        sys.modules["optax.typing"] = optax_stub.typing
        optax_stub.typing.GradientTransformation = object
    
    # Orbax checkpoint stub  
    if "orbax_checkpoint" not in sys.modules:
        orbax_stub = types.ModuleType("orbax_checkpoint")
        sys.modules["orbax_checkpoint"] = orbax_stub
        orbax_stub.PyTreeCheckpointer = object
    
    print("✅ Created stubs for heavyweight dependencies")


def byte_compile_check(src_dir: pathlib.Path) -> bool:
    """Byte-compile all Python files to check syntax and imports."""
    try:
        success = compileall.compile_dir(
            src_dir,
            quiet=1,  # Don't print filenames
            force=True,  # Recompile even if .pyc exists
        )
        if success:
            print("✅ All modules byte-compile successfully")
            return True
        else:
            print("❌ Byte compilation failed")
            return False
    except Exception as e:
        print(f"❌ Byte compilation error: {e}")
        return False


def test_import(module_path: str, components: List[str] = None) -> Tuple[bool, str]:
    """Test importing a module or specific components.
    
    Args:
        module_path: Module path like 'titanax' or 'titanax.runtime'
        components: List of specific components to test, if any
        
    Returns:
        Tuple of (success, error_message)
    """
    try:
        if components:
            # Test importing specific components
            module = __import__(module_path, fromlist=components)
            for component in components:
                if not hasattr(module, component):
                    return False, f"Component '{component}' not found in {module_path}"
        else:
            # Test importing the entire module
            __import__(module_path)
        return True, ""
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)}"


def main():
    """Main validation function."""
    print("🧪 Safe Titanax Package Import Validation")
    print("=" * 50)
    
    # Step 1: Create stubs
    create_stubs()
    
    # Step 2: Byte-compile check
    titanax_dir = pathlib.Path(__file__).parent.parent / "src" / "titanax"
    if not byte_compile_check(titanax_dir):
        return 1
    
    # Step 3: Test imports
    test_cases = [
        # Main package
        ("titanax", ["__version__", "MeshSpec", "Plan", "Engine"]),
        
        # Runtime components
        ("titanax.runtime", ["MeshSpec", "ProcessGroups"]),
        
        # Parallel plans
        ("titanax.parallel", ["DP", "TP", "PP", "Plan"]),
        
        # Execution engine  
        ("titanax.exec", ["Engine", "TrainState", "collectives"]),
        
        # Optimizer integration
        ("titanax.optim", ["OptaxAdapter", "adamw", "sgd"]),
        
        # Logging
        ("titanax.logging", ["Basic", "CompactBasic"]),
        
        # I/O and checkpointing
        ("titanax.io", ["OrbaxCheckpoint", "CheckpointMetadata"]),
        
        # Types and exceptions
        ("titanax.types", ["PyTree", "Array", "Logger"]),
        ("titanax.exceptions", ["TitanaxError", "MeshError"]),
        
        # Convenience modules
        ("titanax.quickstart", ["simple_data_parallel", "validate_setup"]),
    ]
    
    # Special namespace tests
    namespace_tests = [
        ("import titanax as tx; tx.optim.adamw", "Optim namespace"),
        ("import titanax as tx; tx.loggers.Basic", "Loggers namespace"),
        ("import titanax as tx; tx.io.OrbaxCheckpoint", "IO namespace"),
        ("import titanax as tx; tx.quickstart.simple_data_parallel", "Quickstart namespace"),
        ("import titanax as tx; tx.Precision.bf16()", "Precision convenience"),
    ]
    
    # Run tests
    passed = 0
    total = len(test_cases) + len(namespace_tests)
    
    print("\n📦 Testing module imports...")
    for module_path, components in test_cases:
        success, error = test_import(module_path, components)
        if success:
            print(f"✅ {module_path}: {components or 'all'}")
            passed += 1
        else:
            print(f"❌ {module_path}: {error}")
    
    print("\n🔗 Testing namespaces...")
    for test_code, description in namespace_tests:
        try:
            exec(test_code)
            print(f"✅ {description}")
            passed += 1
        except Exception as e:
            print(f"❌ {description}: {type(e).__name__}: {str(e)}")
    
    print("=" * 50)
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All imports working correctly!")
        return 0
    else:
        print("⚠️  Some imports failed. Check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
