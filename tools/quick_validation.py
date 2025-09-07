#!/usr/bin/env python3
"""Quick validation of package structure without heavyweight dependencies."""

import ast
import pathlib
import sys
from typing import Dict, List, Set


def parse_imports_from_file(file_path: pathlib.Path) -> Dict[str, List[str]]:
    """Parse imports from a Python file using AST."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        imports = {"from": [], "import": []}
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for name in node.names:
                    imports["import"].append(name.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for name in node.names:
                        imports["from"].append(f"{node.module}.{name.name}")
        
        return imports
    except Exception as e:
        print(f"❌ Error parsing {file_path}: {e}")
        return {"from": [], "import": []}


def check_package_structure(src_dir: pathlib.Path) -> bool:
    """Check that the package structure is complete."""
    titanax_dir = src_dir / "titanax"
    
    required_files = [
        "__init__.py",
        "_version.py",
        "types.py",
        "exceptions.py",
        "quickstart.py",
    ]
    
    required_packages = [
        "runtime",
        "parallel", 
        "exec",
        "optim",
        "logging",
        "io",
    ]
    
    print("📦 Checking package structure...")
    
    # Check required files
    for file in required_files:
        file_path = titanax_dir / file
        if file_path.exists():
            print(f"✅ {file}")
        else:
            print(f"❌ Missing: {file}")
            return False
    
    # Check required packages
    for package in required_packages:
        package_dir = titanax_dir / package
        init_file = package_dir / "__init__.py"
        if package_dir.exists() and init_file.exists():
            print(f"✅ {package}/")
        else:
            print(f"❌ Missing package: {package}")
            return False
    
    return True


def check_syntax_all_files(src_dir: pathlib.Path) -> bool:
    """Check syntax of all Python files."""
    titanax_dir = src_dir / "titanax"
    
    print("\n🔍 Checking Python syntax...")
    
    all_good = True
    for py_file in titanax_dir.rglob("*.py"):
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                source = f.read()
            ast.parse(source)
            rel_path = py_file.relative_to(src_dir)
            print(f"✅ {rel_path}")
        except SyntaxError as e:
            rel_path = py_file.relative_to(src_dir)
            print(f"❌ {rel_path}: Syntax error at line {e.lineno}")
            all_good = False
        except Exception as e:
            rel_path = py_file.relative_to(src_dir)
            print(f"❌ {rel_path}: {e}")
            all_good = False
    
    return all_good


def check_init_exports(src_dir: pathlib.Path) -> bool:
    """Check that main __init__.py exports make sense."""
    main_init = src_dir / "titanax" / "__init__.py"
    
    print("\n📋 Checking main __init__.py exports...")
    
    try:
        with open(main_init, 'r', encoding='utf-8') as f:
            source = f.read()
        
        tree = ast.parse(source)
        
        # Find __all__ definition
        all_exports = None
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and target.id == "__all__":
                        if isinstance(node.value, ast.List):
                            all_exports = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
        
        if all_exports:
            print(f"✅ Found __all__ with {len(all_exports)} exports")
            
            # Check some key exports
            key_exports = [
                "__version__", "MeshSpec", "Plan", "DP", "Engine", 
                "TrainState", "step_fn", "collectives", "optim", "loggers"
            ]
            
            missing = []
            for export in key_exports:
                if export in all_exports:
                    print(f"  ✅ {export}")
                else:
                    print(f"  ❌ {export}")
                    missing.append(export)
            
            if missing:
                print(f"❌ Missing key exports: {missing}")
                return False
            else:
                print("✅ All key exports found")
                return True
        else:
            print("❌ No __all__ found in main __init__.py")
            return False
            
    except Exception as e:
        print(f"❌ Error checking exports: {e}")
        return False


def main():
    """Main validation function."""
    print("🧪 Quick Titanax Package Structure Validation")
    print("=" * 50)
    
    src_dir = pathlib.Path(__file__).parent.parent / "src"
    
    if not src_dir.exists():
        print(f"❌ Source directory not found: {src_dir}")
        return 1
    
    # Step 1: Check package structure
    if not check_package_structure(src_dir):
        print("❌ Package structure check failed")
        return 1
    
    # Step 2: Check syntax
    if not check_syntax_all_files(src_dir):
        print("❌ Syntax check failed")
        return 1
    
    # Step 3: Check exports
    if not check_init_exports(src_dir):
        print("❌ Export check failed")
        return 1
    
    print("\n" + "=" * 50)
    print("🎉 Package structure and syntax validation passed!")
    print("📝 Note: This validates structure and syntax only.")
    print("   Full import testing requires Python 3.11+ environment.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
