# Titanax Updated Plan (P1+)

This plan reflects the current codebase after P0 completion and adjusts Phases P1–P5 to match what’s implemented vs. what remains. It focuses on concrete, digestible tasks that align with existing modules and tests.

---

## Phase P1: Tensor Parallel Core (Milestone 2)

### P1.0 Alignment & Scaffolding
- Ensure all references cite `parallel/tp_helpers.py` as the canonical rule helper module.
- Introduce `parallel/sharding.py` for rule application utilities (pattern matching, PartitionSpec tree building, NamedSharding application).
- Add unit tests for sharding utilities.
- ✅ Completed: canonicalized documentation to `tp_helpers`, added `parallel/sharding.py` scaffolding with typed stubs, exported the module, and checked in a skipped `tests/unit/test_sharding_utils.py` harness to anchor future work.

### P1.1 Rule Matching → PartitionSpec Utilities
- Implement `build_param_specs(params_tree, rules, default=PartitionSpec())` in `parallel/sharding.py`.
- Support glob-like pattern matching on “/”-joined param paths (use `fnmatch.fnmatch`), with longest/most-specific match precedence.
- Add `spec_for(path, rules)` helper returning a single `PartitionSpec`.
- Add `tree_paths(tree)` utility to enumerate param paths.
- Tests: `tests/unit/test_sharding_utils.py` covering exact, glob, and conflict-precedence cases.
- ✅ Completed: implemented path enumeration, rule matching, spec tree construction, and activated unit coverage in `tests/unit/test_sharding_utils.py`.

### P1.2 NamedSharding Application Helpers
- ✅ Implemented `apply_named_sharding(tree, mesh, spec_tree)` with structure validation, `NamedSharding(mesh, spec)`, and Titanax `ShardingError` wrapping.
- ✅ Implemented `shard_batch_specs(batch_example, dp_axis)` to shard the leading batch dimension by the DP axis and replicate metadata/scalars.
- ✅ Added unit tests covering placement correctness, structure-mismatch errors, non-array safeguards, and batch spec defaults; ensured public `titanax.*` imports remain usable.

### P1.3 Compile/Engine Integration for TP
- Extend `exec/compile.compile_step_with_plan` to accept optional `param_spec_tree` and `batch_spec_tree` to build pjit `in_shardings`/`out_shardings` automatically when Plan has TP.
- Add lazy compilation path in `Engine`: if TP is present and no `in_shardings`/`out_shardings` were provided via decorator, defer compilation until the first `step()` when state/batch structures are available.
- Add `Engine.shard_params(params)` that applies `apply_named_sharding` from P1.2 using `Plan.tensor_parallel.rules`.
- Tests: `tests/unit/test_compile.py` additions for TP-aware shardings; `tests/unit/test_engine.py` for lazy compile path.

### P1.4 TP-Aware Checkpoint Metadata & Reshard
- Update `io/orbax_io.py` to embed `mesh_spec.describe()` and a compact `plan_spec` (DP axis, TP axis, number of rules) in metadata.
- Add `io/checkpoint.py` helpers: `validate_checkpoint_compatibility(meta, mesh_spec, plan)` and `maybe_reshard_on_load(state, mesh, plan)` that re-applies NamedSharding if current plan differs.
- Tests: round-trip save/restore with sharded params; metadata validation; reshard when TP axis changes size.

### P1.5 GPT-Small TP Example
- `examples/gpt_small_tp/model.py`: minimal Transformer block using `tp_helpers` rules; no external deps.
- `examples/gpt_small_tp/data.py`: toy LM dataset/tokenization stub (synthetic tokens).
- `examples/gpt_small_tp/train.py`: DP×TP training script using Engine + pjit shardings from P1.3.
- Integration test: `tests/integration/test_gpt_tp.py` for TP=2 on CPU (via XLA host device count) with convergence smoke test.

### P1.6 Quickstart TP
- Implement `quickstart.simple_tensor_parallel(...)` using P1.1–P1.3 utilities.
- README quickstart section for TP with 2-device CPU example and troubleshooting note on `XLA_FLAGS`.

### P1.7 Observability (adjusted)
- CSV and TensorBoard loggers already implemented; add metric sync guidance in docs for DP aggregation patterns using `collectives`.
- Optional: add `logging/meter.py` hooks to expose rolling throughput and step time histograms to CSV/TB.

---

## Phase P2: Pipeline Parallel Core (Milestone 3)

### P2.0 Alignment
- Keep `parallel/pp.py` as the canonical module (Stage, PipelineSchedule). Remove/avoid `parallel/pipeline.py` naming.
- Confirm unit tests in `tests/unit/test_pp.py` remain green.

### P2.1 Pipeline Runner (host baseline → JIT-capable)
- Create `exec/pipeline.py` with a `run_1f1b(stages, schedule, microbatches, optimizer, *, loss_fn, lr, precision)` reference runner (host loops) mirroring `examples/pp_minimal_two_stage.py` logic.
- Add structure for future `shard_map`/`pjit` execution (function signatures and TODOs), but keep initial implementation host-side to unblock tests.
- Tests: `tests/unit/test_pipeline_runner.py` to validate tick progression, gradient aggregation, and loss decrease on a toy 2-stage model.

### P2.2 Activation Checkpointing
- `exec/remat.py`: thin wrappers for `jax.checkpoint`/`jax.remat` with policies (“none”, “full”, “selective”).
- Integrate optional remat into `Stage` via a helper that wraps `forward_fn` based on `Stage.remat_policy`.
- Tests: verify recompute vs saved activation behavior on small graphs.

### P2.3 Engine/Pipeline Integration
- Provide `create_pipeline_step_fn(stages, schedule, loss_fn, optimizer)` that returns a Titanax `@step_fn` compatible function for Engine.
- Add `Plan.pipeline_parallel` handling in `compile_step_with_plan` for batch partitioning across PP axis when it overlaps TP axis.
- Integration test: `tests/integration/test_pipeline.py` single-host 2-stage run, smoke-level parity vs non-pipeline baseline.

### P2.4 GPT-Small PP Example
- `examples/gpt_small_pp/model.py`: 2-stage GPT split (encoder/decoder-like) with clear stage boundaries.
- `examples/gpt_small_pp/train.py`: pipeline training script using P2.3 utilities; reports throughput/memory.

---

## Phase P3: Developer Experience & Documentation (Milestone 4)

### P3.1 CLI Launcher
- `launch/cli.py` using Typer: `titanax.run` command with subcommands `dp`, `tp`, `pp`.
- Features: config file path, multi-host env var setup guidance, run directory creation, logging setup.
- Tests: CLI argument parsing and help text; dry-run that builds Engine and prints `.describe()`.

### P3.2 YAML Configuration
- `config/schema.py`: dataclasses for mesh, plan (DP/TP/PP), optimizer, precision, logging, checkpoint.
- `config/__init__.py`: `load_config(path)`, env var substitution `${ENV:DEFAULT}`, and validation.
- Examples: `examples/configs/mnist_dp.yaml`, `gpt_small_tp.yaml`, `gpt_small_pp.yaml` aligned with quickstarts.

### P3.3 Profiling & Diagnostics
- `profiling/__init__.py`: context managers for compile/runtime timing, memory watermark, and basic XLA profiler hooks.
- `diagnostics.py`: mesh/plan validation report with suggested fixes (leverage existing exception messages).
- Engine: optional profiling hooks around `register_step_fn` (compile) and `step()` (runtime) exposed to loggers.

### P3.4 Documentation Site Scaffolding
- `docs/quickstart.md`, `docs/concepts.md`, `docs/api/` stubs; wire `pdoc` or `sphinx` generation locally.
- Content: mesh/plan/collectives overview, DP→TP→PP progression, troubleshooting.

---

## Phase P4: Testing & Validation

### P4.1 Unit Test Expansion
- Property-based tests for rule matching and spec building (Hypothesis optional, or table-driven cases).
- Error-path tests for checkpoint compatibility and resharding failures.
- PRNG determinism tests across device counts for TP/DP paths.

### P4.2 Integration Tests
- MNIST-DP scaling (already present): extend to 1→N CPU with forced device count; add tolerance thresholds.
- GPT-small TP smoke test (from P1.5) across TP=2,4 (CPU where feasible).
- Pipeline 2-stage smoke test: convergence trend and scheduling correctness.

### P4.3 Benchmarks (lightweight, offline)
- `tests/benchmarks/scaling.py`: DP throughput vs device count; TP throughput vs shard count; simple timing harness.
- `tests/benchmarks/compilation.py`: compile time vs step time measurements; cache effects.

---

## Phase P5: Release Preparation

### P5.1 Packaging & Metadata
- Complete `pyproject.toml` metadata (classifiers, URLs, optional extras for tensorboard/orbax).
- Add `python_requires` and pin core dependency ranges compatible with implemented shims.

### P5.2 Documentation & Examples Polish
- Verify all quickstarts and examples runnable via CLI and YAML configs.
- Add FAQ and troubleshooting to README and docs.

### P5.3 Quality & Security
- Lint/type checks in CI across 3.11/3.12/3.13; ensure examples/tests green.
- Basic security audit of checkpoint loading paths and JSON metadata handling.

---

## Notes on What’s Already Implemented (and not repeated here)
- Mesh-aware compilation path via `exec/compile.py` with `pjit`/`shard_map` selection is in place.
- Extended collectives (`psum`, `pmean`, `all_gather`, `reduce_scatter`, `ppermute`, `all_to_all`, `axis_index`) exist with axis validation.
- TP helper rule generators (`parallel/tp_helpers.py`) and PP scaffolding (`parallel/pp.py`) with tests are present.
- CSV and TensorBoard loggers are implemented.

This plan focuses remaining work on sharding utilities, TP/PP integration with the Engine and compile path, runnable TP/PP examples, DX (CLI + YAML), and rounding out tests, docs, and packaging.
