"""Smoke tests for Titanax imports.

Tests that all documented public symbols can be imported without error.
This validates the package structure and import surface.
"""

import pytest


def test_titanax_main_import():
    """Test main titanax import."""
    import titanax as tx

    # Basic import should succeed
    assert tx is not None
    assert hasattr(tx, "__version__")


def test_core_runtime_imports():
    """Test core runtime components import."""
    import titanax as tx

    # Mesh and process management
    assert hasattr(tx, "MeshSpec")
    assert hasattr(tx, "ProcessGroups")

    # Distributed utilities
    assert hasattr(tx, "detect_distributed_env")
    assert hasattr(tx, "is_distributed_env")
    assert hasattr(tx, "initialize_distributed")
    assert hasattr(tx, "auto_initialize")


def test_parallel_plan_imports():
    """Test parallel plan components import."""
    import titanax as tx

    # Plan and parallelism types
    assert hasattr(tx, "Plan")
    assert hasattr(tx, "DP")
    assert hasattr(tx, "TP")  # stub implementation
    assert hasattr(tx, "PP")  # stub implementation


def test_execution_engine_imports():
    """Test execution engine components import."""
    import titanax as tx

    # Engine and state
    assert hasattr(tx, "Engine")
    assert hasattr(tx, "TrainState")
    assert hasattr(tx, "Precision")

    # Step function decorator
    assert hasattr(tx, "step_fn")

    # Collectives namespace
    assert hasattr(tx, "collectives")

    # PRNG utilities
    assert hasattr(tx, "update_rngs")
    assert hasattr(tx, "split_rng")
    assert hasattr(tx, "create_per_device_rngs")
    assert hasattr(tx, "update_per_device_rngs")
    assert hasattr(tx, "split_per_device_rng")
    assert hasattr(tx, "validate_rng_keys")
    assert hasattr(tx, "create_host_device_rngs")


def test_type_system_imports():
    """Test type system imports."""
    import titanax as tx

    # JAX type aliases
    assert hasattr(tx, "Array")
    assert hasattr(tx, "PyTree")
    assert hasattr(tx, "Mesh")
    assert hasattr(tx, "PartitionSpec")
    assert hasattr(tx, "NamedSharding")

    # Protocol types
    assert hasattr(tx, "Logger")
    assert hasattr(tx, "CheckpointStrategy")
    assert hasattr(tx, "StepFunction")


def test_exception_imports():
    """Test exception hierarchy imports."""
    import titanax as tx

    # Exception classes
    assert hasattr(tx, "TitanaxError")
    assert hasattr(tx, "MeshError")
    assert hasattr(tx, "PlanError")
    assert hasattr(tx, "CollectiveError")
    assert hasattr(tx, "EngineError")
    assert hasattr(tx, "CheckpointError")


def test_namespace_imports():
    """Test namespace module imports."""
    import titanax as tx

    # Namespaces
    assert hasattr(tx, "optim")
    assert hasattr(tx, "loggers")
    assert hasattr(tx, "io")
    assert hasattr(tx, "quickstart")


def test_checkpoint_imports():
    """Test checkpoint system imports."""
    import titanax as tx

    # Checkpoint shortcuts
    assert hasattr(tx, "OrbaxCheckpoint")
    assert hasattr(tx, "CheckpointMetadata")


def test_optimizer_namespace():
    """Test optimizer namespace imports."""
    import titanax as tx

    # Optimizer namespace should be available
    optim = tx.optim
    assert optim is not None

    # Key optimizer functions
    assert hasattr(optim, "adamw")
    assert hasattr(optim, "sgd")
    assert hasattr(optim, "adam")
    assert hasattr(optim, "OptaxAdapter")


def test_logger_namespace():
    """Test logger namespace imports."""
    import titanax as tx

    # Logger namespace
    loggers = tx.loggers
    assert loggers is not None

    # Basic logger implementations
    assert hasattr(loggers, "Basic")
    assert hasattr(loggers, "CompactBasic")
    assert hasattr(loggers, "MultiLogger")
    assert hasattr(loggers, "NullLogger")


def test_collectives_namespace():
    """Test collectives namespace imports."""
    import titanax as tx

    # Collectives should be available
    collectives = tx.collectives
    assert collectives is not None

    # Core collective operations
    assert hasattr(collectives, "psum")
    assert hasattr(collectives, "pmean")

    # Additional collectives (may be stubs)
    assert hasattr(collectives, "all_gather")
    assert hasattr(collectives, "reduce_scatter")
    assert hasattr(collectives, "broadcast")


def test_precision_shortcuts():
    """Test precision configuration shortcuts."""
    import titanax as tx

    # Precision helpers
    assert hasattr(tx, "bfloat16_precision")
    assert hasattr(tx, "float16_precision")
    assert hasattr(tx, "float32_precision")

    # Aliases
    assert hasattr(tx, "bf16_precision")
    assert hasattr(tx, "fp16_precision")
    assert hasattr(tx, "fp32_precision")


def test_version_and_metadata():
    """Test version and metadata imports."""
    import titanax as tx

    # Version info
    assert hasattr(tx, "__version__")
    assert hasattr(tx, "__author__")
    assert hasattr(tx, "__email__")
    assert hasattr(tx, "__description__")
    assert hasattr(tx, "__url__")

    # All should be strings
    assert isinstance(tx.__version__, str)
    assert len(tx.__version__) > 0


def test_functional_imports():
    """Test that key classes can be instantiated without error."""
    import titanax as tx
    import jax

    # MeshSpec with single CPU device
    mesh_spec = tx.MeshSpec(devices=[jax.devices("cpu")[0]], axes=("data",))
    assert mesh_spec is not None

    # Plan with DP
    plan = tx.Plan(data_parallel=tx.DP(axis="data"))
    assert plan is not None

    # Precision configs
    precision = tx.Precision()
    assert precision is not None

    bf16_precision = tx.bfloat16_precision()
    assert bf16_precision is not None

    # Optimizer
    optimizer = tx.optim.sgd(0.01)
    assert optimizer is not None

    # Logger
    logger = tx.loggers.Basic()
    assert logger is not None


if __name__ == "__main__":
    pytest.main([__file__])
