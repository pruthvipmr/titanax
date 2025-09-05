# Titanax Implementation Plan

## Phase 0: Project Setup and Foundation

### 0.1 Project Structure Setup ✅ COMPLETED
- [x] Set up python environment with uv init
- [x] Set up core package structure with empty `__init__.py` files
- [x] Create basic test structure (`tests/unit/`, `tests/integration/`, `tests/benchmarks/`)
- [x] Create `examples/` directory structure
- [x] Set up GitHub Actions/CI for testing
- [x] Add core dependencies: `jax`, `optax`, `orbax-checkpoint`, `numpy`

**Notes:**
- Project initialized with uv using `--lib` flag for library structure
- Package structure follows spec layout with all required directories:
  - `src/titanax/` with subpackages: runtime, parallel, exec, optim, io, data, logging, launch
  - Complete test structure with unit, integration, and benchmark directories
  - Examples structure prepared for MNIST-DP, GPT-small-TP, and GPT-small-PP
- GitHub Actions CI configured for Python 3.9-3.11 with comprehensive testing
- Core dependencies added: jax, optax, orbax-checkpoint, numpy, typer, pyyaml
- Dev dependencies added: pytest, mypy, black, ruff for code quality
- All directories include appropriate `__init__.py` files with descriptive comments

### 0.2 Core Type Definitions ✅ COMPLETED
- [x] Create `titanax/types.py` with common type aliases (`PyTree`, etc.)
- [x] Define base exception classes in `titanax/exceptions.py`
- [x] Set up logging utilities in `titanax/logging/base.py`

**Notes:**
- Created comprehensive type system with JAX-specific aliases (Array, PyTree, Mesh, etc.)
- Defined protocol-based interfaces for StepFunction, Logger, and CheckpointStrategy
- Implemented hierarchical exception system with TitanaxError as base class
- Added specialized exceptions for each component: MeshError, PlanError, CollectiveError, etc.
- Exception classes include suggestion system for better error messages
- Created robust logging base with BaseLogger ABC, MultiLogger for multiplexing, and NullLogger
- Added utility functions for metric formatting and aggregation
- All components follow typing best practices with runtime_checkable protocols

## Phase P0: Data Parallel Core (Milestone 1)

### P0.1 Runtime & Control Plane
- [ ] **File: `titanax/runtime/init.py`**
  - [ ] Implement JAX distributed initialization helper
  - [ ] Handle multi-host environment variable detection
  - [ ] Add device enumeration utilities

- [ ] **File: `titanax/runtime/mesh.py`**
  - [ ] Implement `MeshSpec` dataclass with validation
  - [ ] Add `build()` method to create `jax.sharding.Mesh`
  - [ ] Add `describe()` method for debugging
  - [ ] Implement shape inference when `shape=None`
  - [ ] Add device topology hints support

- [ ] **File: `titanax/runtime/process_groups.py`**
  - [ ] Implement `ProcessGroups` class
  - [ ] Add `size()` and `rank()` methods for mesh axes
  - [ ] Add validation for axis existence

### P0.2 Data Parallel Plan
- [ ] **File: `titanax/parallel/plan.py`**
  - [ ] Implement `DP` dataclass with axis, accumulate_steps, sync_metrics
  - [ ] Implement `Plan` dataclass with data_parallel field
  - [ ] Add `validate()` method to check axis compatibility with mesh
  - [ ] Add `describe()` method for debugging

### P0.3 Collectives Layer
- [ ] **File: `titanax/exec/collectives.py`**
  - [ ] Implement `psum()` with axis validation
  - [ ] Implement `pmean()` with axis validation
  - [ ] Add runtime checks for axis existence in current mesh
  - [ ] Add tree shape compatibility validation
  - [ ] Implement `all_gather()`, `reduce_scatter()`, `broadcast()` stubs

### P0.4 Execution Engine Core
- [ ] **File: `titanax/exec/engine.py`**
  - [ ] Implement `Precision` dataclass
  - [ ] Implement `TrainState` dataclass with params, opt_state, step, rngs
  - [ ] Implement basic `Engine` class with `__init__`
  - [ ] Add mesh and plan validation in Engine constructor

- [ ] **File: `titanax/exec/step_fn.py`**
  - [ ] Implement `@step_fn` decorator
  - [ ] Add JIT compilation of decorated functions
  - [ ] Add PRNG management and state threading
  - [ ] Add gradient accumulation support for microbatching

- [ ] **File: `titanax/exec/engine.py` (continued)**
  - [ ] Implement `Engine.fit()` method
  - [ ] Add training loop with step execution
  - [ ] Add metrics collection and logging hooks
  - [ ] Add checkpoint save/restore integration

### P0.5 Optimizer Integration
- [ ] **File: `titanax/optim/optax_adapter.py`**
  - [ ] Create `OptaxAdapter` wrapper class
  - [ ] Implement `adamw()`, `sgd()` factory functions
  - [ ] Add learning rate scheduling support
  - [ ] Ensure compatibility with sharded parameters

### P0.6 Basic Logging
- [ ] **File: `titanax/logging/basic.py`**
  - [ ] Implement `Basic` logger with stdout output
  - [ ] Add scalar and dict logging methods
  - [ ] Implement step-based formatting

### P0.7 Checkpoint System
- [ ] **File: `titanax/io/checkpoint.py`**
  - [ ] Define `CheckpointStrategy` protocol
  - [ ] Add save/load/restore method signatures

- [ ] **File: `titanax/io/orbax_io.py`**
  - [ ] Implement `OrbaxCheckpoint` strategy
  - [ ] Add sharded parameter save/load
  - [ ] Add TrainState serialization/deserialization
  - [ ] Add step-based checkpoint naming

### P0.8 Package Integration
- [ ] **File: `titanax/__init__.py`**
  - [ ] Import and expose all P0 public APIs
  - [ ] Create convenience imports (`tx.DP`, `tx.Engine`, etc.)
  - [ ] Add version information

### P0.9 MNIST Example
- [ ] **File: `examples/mnist_dp/model.py`**
  - [ ] Implement simple CNN model with pure JAX functions
  - [ ] Add parameter initialization
  - [ ] Add forward pass implementation

- [ ] **File: `examples/mnist_dp/data.py`**
  - [ ] Implement MNIST data loading
  - [ ] Add batch sharding for DP
  - [ ] Add data preprocessing

- [ ] **File: `examples/mnist_dp/train.py`**
  - [ ] Implement complete DP training script
  - [ ] Use `@tx.step_fn` with explicit `psum`
  - [ ] Add loss and accuracy metrics
  - [ ] Test scaling from 1→8 GPUs

### P0.10 Unit Tests
- [ ] **File: `tests/unit/test_mesh.py`**
  - [ ] Test MeshSpec validation and building
  - [ ] Test device enumeration edge cases
  - [ ] Test shape inference

- [ ] **File: `tests/unit/test_plan.py`**
  - [ ] Test Plan validation
  - [ ] Test DP parameter validation
  - [ ] Test axis compatibility checks

- [ ] **File: `tests/unit/test_collectives.py`**
  - [ ] Test psum/pmean with various tree structures
  - [ ] Test axis validation errors
  - [ ] Test shape compatibility

- [ ] **File: `tests/unit/test_engine.py`**
  - [ ] Test Engine initialization
  - [ ] Test step function decoration
  - [ ] Test TrainState management

### P0.11 Integration Tests
- [ ] **File: `tests/integration/test_mnist_dp.py`**
  - [ ] Test MNIST training convergence
  - [ ] Test 1-device vs multi-device loss parity (within 1e-4)
  - [ ] Test checkpoint save/resume
  - [ ] Test microbatching equivalence

## Phase P1: Tensor Parallel Core (Milestone 2)

### P1.1 Tensor Parallel Plan
- [ ] **File: `titanax/parallel/rules.py`**
  - [ ] Implement parameter path pattern matching
  - [ ] Add `PartitionSpec` creation from rules
  - [ ] Add transformer-specific rule templates

- [ ] **File: `titanax/parallel/plan.py` (extend)**
  - [ ] Implement `TP` dataclass with axis, rules, prefer_reduce_scatter
  - [ ] Add `spec_for()` method for parameter-specific partitioning
  - [ ] Extend `Plan` to include tensor_parallel field
  - [ ] Add DP×TP composition validation

### P1.2 Extended Collectives
- [ ] **File: `titanax/exec/collectives.py` (extend)**
  - [ ] Implement `all_gather()` with axis_index support
  - [ ] Implement `reduce_scatter()` with operation types
  - [ ] Add `ppermute()` for arbitrary permutations
  - [ ] Add collective validation for TP operations

### P1.3 Sharding Utilities
- [ ] **File: `titanax/parallel/sharding.py`**
  - [ ] Implement `NamedSharding` creation helpers
  - [ ] Add parameter tree sharding utilities
  - [ ] Add activation sharding helpers for TP layers

### P1.4 TP Layer Wrappers
- [ ] **File: `titanax/layers/__init__.py`**
  - [ ] Create layer wrapper infrastructure

- [ ] **File: `titanax/layers/linear.py`**
  - [ ] Implement column-parallel linear layer
  - [ ] Implement row-parallel linear layer
  - [ ] Add automatic collective insertion

- [ ] **File: `titanax/layers/attention.py`**
  - [ ] Implement sharded multi-head attention
  - [ ] Add QKV projection sharding
  - [ ] Add output projection sharding

### P1.5 Engine TP Support
- [ ] **File: `titanax/exec/engine.py` (extend)**
  - [ ] Add TP parameter sharding in Engine
  - [ ] Integrate TP rules into TrainState creation
  - [ ] Add DP×TP gradient synchronization
  - [ ] Add TP-aware checkpoint integration

### P1.6 Extended Checkpointing
- [ ] **File: `titanax/io/orbax_io.py` (extend)**
  - [ ] Add sharded parameter save with TP layout
  - [ ] Add parameter resharding on load
  - [ ] Add TP→DP and DP→TP checkpoint conversion
  - [ ] Add validation for sharding compatibility

### P1.7 GPT-Small Example
- [ ] **File: `examples/gpt_small_tp/model.py`**
  - [ ] Implement transformer model with TP support
  - [ ] Use column/row parallel linear layers
  - [ ] Add proper attention sharding

- [ ] **File: `examples/gpt_small_tp/data.py`**
  - [ ] Implement language modeling data loading
  - [ ] Add tokenization and sequence processing

- [ ] **File: `examples/gpt_small_tp/train.py`**
  - [ ] Implement DP×TP training script
  - [ ] Add perplexity evaluation
  - [ ] Test TP=2,4,8 configurations

### P1.8 Advanced Logging
- [ ] **File: `titanax/logging/csv.py`**
  - [ ] Implement CSV file logger
  - [ ] Add append-mode support
  - [ ] Add metric aggregation across devices

- [ ] **File: `titanax/logging/tensorboard.py`**
  - [ ] Implement TensorBoard integration (optional dependency)
  - [ ] Add scalar/histogram logging
  - [ ] Add distributed logging coordination

### P1.9 Unit Tests for TP
- [ ] **File: `tests/unit/test_tp.py`**
  - [ ] Test TP rule parsing and validation
  - [ ] Test PartitionSpec generation
  - [ ] Test parameter path matching

- [ ] **File: `tests/unit/test_layers.py`**
  - [ ] Test linear layer sharding equivalence
  - [ ] Test attention layer TP correctness
  - [ ] Test collective insertion

### P1.10 Integration Tests for TP
- [ ] **File: `tests/integration/test_gpt_tp.py`**
  - [ ] Test GPT-small TP vs single-device equivalence
  - [ ] Test DP×TP scaling and convergence
  - [ ] Test checkpoint resharding (TP=2 → TP=4)
  - [ ] Test perplexity targets

## Phase P2: Pipeline Parallel Core (Milestone 3)

### P2.1 Pipeline Parallel Plan
- [ ] **File: `titanax/parallel/pipeline.py`**
  - [ ] Implement `Stage` dataclass with param_filter and fwd function
  - [ ] Implement `PP` dataclass with axis, stages, microbatch_size
  - [ ] Add stage graph construction
  - [ ] Add parameter partitioning by stage

### P2.2 1F1B Scheduler
- [ ] **File: `titanax/parallel/scheduler.py`**
  - [ ] Implement 1F1B scheduling algorithm
  - [ ] Add warm-up, steady-state, cool-down phases
  - [ ] Add microbatch slicing and distribution
  - [ ] Add inter-stage communication orchestration

### P2.3 Pipeline Engine Integration
- [ ] **File: `titanax/exec/pipeline.py`**
  - [ ] Implement pipeline-specific step function decoration
  - [ ] Add stage-local compilation and execution
  - [ ] Add activation transfer between stages
  - [ ] Add gradient accumulation across microbatches

### P2.4 Activation Checkpointing
- [ ] **File: `titanax/exec/remat.py`**
  - [ ] Implement selective activation checkpointing
  - [ ] Add checkpoint_ratio-based selection
  - [ ] Integrate with 1F1B scheduler

### P2.5 Extended Plan Composition
- [ ] **File: `titanax/parallel/plan.py` (extend)**
  - [ ] Add pipeline_parallel field to Plan
  - [ ] Add DP×TP×PP validation
  - [ ] Add PP-specific axis management

### P2.6 Pipeline Example
- [ ] **File: `examples/gpt_small_pp/model.py`**
  - [ ] Implement 2-stage GPT model split
  - [ ] Add stage-specific parameter filters
  - [ ] Add stage forward functions

- [ ] **File: `examples/gpt_small_pp/train.py`**
  - [ ] Implement pipeline training script
  - [ ] Add throughput measurements
  - [ ] Compare vs single-stage baseline

### P2.7 PP Tests
- [ ] **File: `tests/unit/test_pp.py`**
  - [ ] Test 1F1B scheduler correctness
  - [ ] Test stage graph construction
  - [ ] Test microbatch scheduling

- [ ] **File: `tests/integration/test_pipeline.py`**
  - [ ] Test 2-stage pipeline training
  - [ ] Test gradient flow across stages
  - [ ] Test throughput vs memory trade-offs

## Phase P3: Developer Experience & Documentation (Milestone 4)

### P3.1 CLI Launcher
- [ ] **File: `titanax/launch/cli.py`**
  - [ ] Implement `titanax.run` command using typer
  - [ ] Add multi-host coordination setup
  - [ ] Add config file loading
  - [ ] Add process spawning and management

### P3.2 YAML Configuration
- [ ] **File: `titanax/config/__init__.py`**
  - [ ] Implement config schema validation
  - [ ] Add YAML→dataclass conversion
  - [ ] Add environment variable substitution

- [ ] **File: `titanax/config/schema.py`**
  - [ ] Define complete config schema
  - [ ] Add validation rules
  - [ ] Add default value handling

### P3.3 Profiling Integration
- [ ] **File: `titanax/profiling/__init__.py`**
  - [ ] Add XLA profiler integration hooks
  - [ ] Add compile vs runtime measurements
  - [ ] Add memory watermark tracking
  - [ ] Add tokens/sec throughput calculation

### P3.4 Enhanced Error Messages
- [ ] **File: `titanax/diagnostics.py`**
  - [ ] Add mesh validation with helpful error messages
  - [ ] Add plan compatibility diagnostics
  - [ ] Add collective operation debugging
  - [ ] Add suggested fixes for common errors

### P3.5 Documentation Site
- [ ] **File: `docs/quickstart.md`**
  - [ ] Write getting started tutorial
  - [ ] Add single-GPU to multi-GPU progression
  - [ ] Add example walkthroughs

- [ ] **File: `docs/concepts.md`**
  - [ ] Document mesh, plan, collective concepts
  - [ ] Add parallelism strategy guidance
  - [ ] Add performance tuning tips

- [ ] **File: `docs/api/`**
  - [ ] Generate API documentation
  - [ ] Add code examples for each component
  - [ ] Add troubleshooting guides

### P3.6 Example Configs
- [ ] **File: `examples/configs/mnist_dp.yaml`**
  - [ ] MNIST DP configuration example

- [ ] **File: `examples/configs/gpt_small_tp.yaml`**
  - [ ] GPT-small TP configuration example

- [ ] **File: `examples/configs/gpt_small_pp.yaml`**
  - [ ] GPT-small PP configuration example

### P3.7 Advanced Examples
- [ ] **File: `examples/mixed_precision/`**
  - [ ] Implement bf16 training example
  - [ ] Add loss scaling demonstration
  - [ ] Show precision policy configuration

- [ ] **File: `examples/checkpoint_reshard/`**
  - [ ] Demonstrate checkpoint resharding
  - [ ] Show TP=2 → TP=4 conversion
  - [ ] Add validation scripts

## Phase P4: Testing & Validation

### P4.1 Comprehensive Unit Tests
- [ ] Achieve >90% test coverage for all core components
- [ ] Add property-based tests for sharding logic
- [ ] Add performance regression tests
- [ ] Add error path testing

### P4.2 Integration Test Suite
- [ ] **Acceptance Test: MNIST-DP**
  - [ ] 1→8 GPU scaling validation
  - [ ] Loss parity within 1e-4 tolerance
  - [ ] Checkpoint resume functionality

- [ ] **Acceptance Test: GPT-small TP**
  - [ ] Reference perplexity achievement
  - [ ] Sharded checkpoint round-trip
  - [ ] DP×TP composition validation

- [ ] **Acceptance Test: GPT-small PP**
  - [ ] Pipeline training convergence
  - [ ] Throughput improvement validation
  - [ ] Memory efficiency demonstration

### P4.3 Benchmark Suite
- [ ] **File: `tests/benchmarks/scaling.py`**
  - [ ] DP scaling benchmarks (1,2,4,8 GPUs)
  - [ ] TP scaling benchmarks (TP=1,2,4)
  - [ ] PP throughput benchmarks

- [ ] **File: `tests/benchmarks/compilation.py`**
  - [ ] Compile time measurements
  - [ ] Recompilation detection tests
  - [ ] Cache hit rate monitoring

### P4.4 Multi-Host Testing
- [ ] Set up multi-host test environment
- [ ] Add distributed initialization tests
- [ ] Add network failure simulation
- [ ] Add coordinator service tests

## Phase P5: Release Preparation

### P5.1 Package Finalization
- [ ] Update `pyproject.toml` with complete metadata
- [ ] Add optional dependencies configuration
- [ ] Set up wheel building and PyPI publishing
- [ ] Add license and README files

### P5.2 Documentation Review
- [ ] Review all documentation for accuracy
- [ ] Add missing API documentation
- [ ] Create video tutorials/walkthroughs
- [ ] Set up documentation hosting

### P5.3 Performance Validation
- [ ] Run full benchmark suite
- [ ] Validate scaling claims
- [ ] Profile memory usage patterns
- [ ] Optimize critical paths

### P5.4 Security & Quality
- [ ] Security audit of checkpoint loading
- [ ] Code quality review
- [ ] Dependency vulnerability scan
- [ ] License compliance check

## Implementation Guidelines

### Code Quality Standards
- Use type hints for all public APIs
- Follow Google Python style guide
- Add docstrings to all public functions
- Use dataclasses for configuration objects
- Prefer explicit over implicit behavior

### Testing Standards
- Write tests before implementation (TDD)
- Aim for >90% test coverage
- Include both positive and negative test cases
- Use small, fast test configurations
- Mock external dependencies where appropriate

### Performance Considerations
- Profile critical paths regularly
- Use static shapes to avoid recompilation
- Minimize Python overhead in hot paths
- Cache expensive computations
- Monitor memory usage patterns

### Documentation Requirements
- Document all public APIs with examples
- Include troubleshooting guides
- Add performance tuning recommendations
- Provide migration guides between versions
- Keep examples up-to-date with API changes
