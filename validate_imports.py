#!/usr/bin/env python3
"""Validation script to test all Titanax imports work correctly."""

import sys
import traceback
from typing import List, Tuple


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
    print("🧪 Validating Titanax package imports...")
    print("=" * 50)
    
    # Test cases: (module_path, components_to_check)
    test_cases = [
        # Main package
        ("titanax", ["__version__", "MeshSpec", "Plan", "Engine"]),
        
        # Runtime components
        ("titanax.runtime", ["MeshSpec", "ProcessGroups"]),
        
        # Parallel plans
        ("titanax.parallel", ["DP", "TP", "PP", "Plan"]),
        
        # Execution engine
        ("titanax.exec", ["Engine", "TrainState", "Precision", "collectives"]),
        
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
    
    for module_path, components in test_cases:
        success, error = test_import(module_path, components)
        if success:
            print(f"✅ {module_path}: {components or 'all'}")
            passed += 1
        else:
            print(f"❌ {module_path}: {error}")
    
    # Test namespaces
    for test_code, description in namespace_tests:
        try:
            exec(test_code)
            print(f"✅ {description}: {test_code}")
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
