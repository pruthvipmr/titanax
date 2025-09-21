# Titanax — Phase P0 Fixlist & Additions
_Last updated: 2025-09-09_


This document enumerates **all fixes, changes, and additions** to address before moving to the next phase of Titanax. Each item includes a suggested **priority**, **effort**, and a concrete **definition of done (DoD)**. Paths are indicative; adapt to your current layout.

> Scope: ergonomics like Accelerate/TorchTitan, but **explicit DP/TP/PP** in JAX (no auto-sharding). The goal of this pass is to make DP rock-solid end‑to‑end, stub minimal TP/PP, and ship observability+checkpointing with CI.

---

## 0) Repo hygiene & quick wins

- [x] **Add a 10‑line runnable example (CPU-only)**
  - **Where:** `examples/minimal_dp.py`
  - **Why:** Proves import surface, engine wiring, and logging.
  - **DoD:** `uv run python examples/minimal_dp.py` runs in <30s on CPU, prints one metric line and exits 0.

- [x] **Smoke test for imports**
  - **Where:** `tests/test_imports.py`
  - **DoD:** `import titanax as tx` plus the documented public symbols (mesh/plan/engine/optim/loggers/io) all import without error.

- [x] **Project README refresh**
  - **Where:** `README.md`
  - **DoD:** Include: (a) 10‑line example, (b) quickstart snippet for DP, (c) a roadmap table (DP ✅, TP 🚧, PP 🚧), (d) note on explicit meshes & collectives.

---

## 1) Runtime, Mesh, and Process Groups

- [x] **Mesh validation API: single canonical name**
  - **Where:** `titanax/runtime/mesh.py`
  - **Change:** Ensure one public method for batch/mesh compatibility (e.g., `validate_compatibility(batch_size: int, allow_padding: bool=False)`) and update all call-sites (quickstarts, tests).
  - **DoD:** Grep for previous aliases; no mismatched calls remain; unit test asserts error message on incompatible batch size.
  - **✅ COMPLETED:** Single canonical `validate_compatibility()` method already implemented and used consistently throughout codebase.

- [x] **Deterministic device factorization**
  - **Where:** `titanax/runtime/mesh.py`
  - **Change:** Make device-count factorization stable across hosts (e.g., lexicographic ordering of devices before partitioning).
  - **DoD:** Multi-host simulated test verifies identical mesh shapes on all ranks.
  - **✅ COMPLETED:** Added deterministic device sorting by (platform, process_index, id) and deterministic factorization algorithm with comprehensive unit tests.

- [x] **Process group description utilities**
  - **Where:** `titanax/runtime/process_groups.py`
  - **Add:** `rank(axis)`, `size(axis)`, `coords()` helpers plus a `describe()` string for logging.
  - **DoD:** Unit test asserts ranks/sizes for 1D and 2D meshes.
  - **✅ COMPLETED:** Added `coords()` method to existing `rank(axis)`, `size(axis)`, and `describe()` methods. Full unit test coverage with 3 new test cases.

- [x] **Compatibility imports centralization**
  - **Where:** `titanax/compat.py`
  - **Add:** Try/except shims for `pjit`, `shard_map`, and collective APIs that shift between JAX versions.
  - **DoD:** CI matrix passes across two recent JAX versions using only `from titanax.compat import pjit, shard_map, ...` in the codebase.
  - **✅ COMPLETED:** Created comprehensive `titanax.compat` module with version-adaptive imports for all JAX APIs. Updated all imports throughout codebase to use compatibility layer. 7 new unit tests verify compatibility across JAX versions.

---

## 2) Plans: DP / TP / PP ✅ COMPLETED

- [x] **Plan validation hardening**
  - **Where:** `titanax/parallel/plan.py`
  - **Change:** Validate that all referenced axis names exist in the mesh; forbid duplicates unless explicitly whitelisted (e.g., PP and TP sharing a “model” axis if supported).
  - **DoD:** Unit tests for good/bad combos (DP-only, TP-only, DP+TP, DP+PP, TP+PP).
  - **✅ COMPLETED:** Added comprehensive plan combination validation tests covering all valid/invalid parallelism combinations (DP-only, TP-only, PP-only, DP+TP, DP+PP, TP+PP, DP+TP+PP) with proper axis conflict detection and helpful error messages.

- [x] **TP rule helpers (minimal)**
  - **Where:** `titanax/parallel/tp_helpers.py`
  - **Add:** Convenience functions to generate common `PartitionSpec`s by param name (e.g., MLP `{in,out}` splits, attention projections).
  - **DoD:** Toy MLP’s params get partitioned by a helper; golden specs asserted in tests.

- [x] **PP stage type (skeleton)**
  - **Where:** `titanax/parallel/pp.py`
  - **Add:** `Stage` dataclass/protocol: `forward(inputs) -> (outputs, activations)`, remat policy field, and stage boundary metadata.
  - **DoD:** `Plan` can be constructed with two stages and validates microbatch size vs. pipeline schedule (see §5).

**✅ Section 2 Summary:** Successfully implemented all Plan validation hardening with comprehensive test coverage for all parallelism combinations. Created complete TP helper functions with golden spec validation for MLP, attention, embedding, and layer norm patterns. Implemented full PP infrastructure with Stage protocol, PipelineSchedule, and 1F1B scheduling support. All components are fully tested (68 total tests) and integrated into the main Titanax API.

---

## 3) Execution Core (Engine, Step Functions, PRNG)

- [x] **Step function decorator guarantees**
  - **Where:** `titanax/exec/step_fn.py`
  - **Change:** Enforce: (a) first arg = `TrainState`, (b) metrics are scalar or reducible, (c) shape/dtype checks under jit.
  - **DoD:** Negative tests raise clear `ValueError` with actionable “fix” text.
  - **✅ COMPLETED:** Added validation helpers to the decorator so mis-specified signatures and non-scalar metrics raise descriptive `ValueError`s both eagerly and inside compiled step functions.

- [x] **Plan‑driven compilation (pjit path)**
  - **Where:** `titanax/exec/compile.py`
  - **Add:** Function: `compile_step_with_plan(step_fn, plan, mesh, in_shardings, out_shardings)` that prefers **pjit** when shardings are provided; fallback to `shard_map` for map-style collectives.
  - **DoD:** Unit test compiles a toy step with explicit `PartitionSpec`s and executes on CPU.
  - **✅ COMPLETED:** Introduced `compile_step_with_plan` (pjit first, shard_map fallback) plus coverage in `tests/unit/test_compile.py` ensuring explicit shardings execute correctly on the CPU mesh.

- [x] **Per-device RNG utilities**
  - **Where:** `titanax/exec/prng.py`
  - **Change:** Ensure APIs: `create_per_device_rngs(seed, mesh)`, `split_per_device_rng(rngs, num=1)`, `update_per_device_rngs(rngs)` with shape/type checks.
  - **DoD:** Tests verify unique streams across devices and determinism for fixed seeds.
  - **✅ COMPLETED:** Reworked the RNG helpers around mesh-aware seeds, added batched splitting utilities, and refreshed `tests/unit/test_prng.py` to check determinism and per-device uniqueness.

- [x] **Engine.fit error contracts**
  - **Where:** `titanax/exec/engine.py`
  - **Change:** `continue_on_error: bool`; on step exception, log and continue or raise; always attempt a final checkpoint on `KeyboardInterrupt`.
  - **DoD:** Tests inject a failing step every N steps and assert both behaviors; final checkpoint exists on interrupt.
  - **✅ COMPLETED:** Existing Engine logic already satisfied the contract; verified via the error-handling unit tests (including KeyboardInterrupt checkpointing) with the new compilation path in place.

- [x] **Microbatch gradient accumulation (DP)**
  - **Where:** `titanax/exec/engine.py` or `titanax/exec/grad_accum.py`
  - **Add:** Accumulate with `lax.scan` over microbatches; support loss-scaling compatible with Optax.
  - **DoD:** Test shows identical results between large-batch single step and microbatched accumulation within tolerance.
  - **✅ COMPLETED:** Extended the accumulation helpers to unscale loss-scaled gradients and surfaced `loss_scale` metrics; added targeted loss-scale equivalence tests in `tests/unit/test_microbatch_accumulation.py`.

---

## 4) Collectives Namespace

- [x] **Axis‑validated collectives coverage**
  - **Where:** `titanax/exec/collectives.py`
  - **Add:** `psum`, `pmean`, `all_gather`, `reduce_scatter`, `broadcast`, `all_to_all` with: (a) mesh/axis validation, (b) PyTree support, (c) docstrings with shape semantics.
  - **DoD:** Unit tests for each op on 1D “data” axis and 2D (“data”, “model”) meshes.

- [x] **Thread‑local mesh context**
  - **Where:** `titanax/exec/collectives.py`
  - **Add:** Context manager `with mesh_context(mesh):` so collectives verify axis names at runtime even under nested jits.
  - **DoD:** Mis-specified axis raises a custom `CollectiveError` with suggestions.

**✅ COMPLETED:** Rebuilt the collectives namespace with PyTree-aware wrappers around all required primitives, thread-local mesh validation, and richer error handling. Added a public `mesh_context` helper and exported it through `titanax.exec`. Comprehensive unit tests exercise every collective across both 1D and 2D mesh layouts, verifying behaviour against the corresponding `jax.lax` primitives.

---

## 5) Minimal TP & PP (practical stubs)

- [x] **TP demo (1D model parallel MLP)**
  - **Where:** `examples/tp_minimal_mlp.py`
  - **Add:** Shard the MLP’s hidden dimension along `axis="model"`; run fwd+bwd on tiny data to verify numerics.
  - **DoD:** Script runs end‑to‑end on single host multi-device; asserts activation/grad shapes and a decreasing loss for 5 steps.

- [x] **PP demo (2‑stage encoder/decoder split)**
  - **Where:** `examples/pp_minimal_two_stage.py`
  - **Add:** Two `Stage`s with a simple 1F1B scheduler; microbatching across the pipeline; CPU-friendly sizes.
  - **DoD:** Script executes for 4 global steps; activations moved only at boundaries; loss decreases.

---

## 6) Checkpointing (Orbax strategy)

- [x] **Checkpoint strategy implementation**
  - **Where:** `titanax/io/orbax_io.py`
  - **Add:** `OrbaxCheckpoint(path, keep_n=3)` implementing `save(state)`, `restore() -> TrainState`, `latest_step() -> int` with metadata sidecar and TrainState rehydration.
  - **DoD:** Unit test: round-trip a tiny `TrainState` on CPU, verify params and opt state equality; retention policy enforced.

- [x] **Engine integration**
  - **Where:** `titanax/exec/engine.py`
  - **Add:** Hooks: save on start, periodic via `checkpoint_interval`, graceful shutdown, and optimizer/RNG restoration through `_rehydrate_restored_state`.
  - **DoD:** E2E tests: MNIST checkpoint/resume flows and fit `resume_from=path` continue the same global step count.

---

## 7) Logging & Observability

- [x] **Stdout loggers + TB/CSV**
  - **Where:** `titanax/logging/`
  - **Add:** `Basic` and `CompactBasic` stdout loggers (if not present), plus `TensorBoardLogger` and `CSVLogger`.
  - **DoD:** Example scripts can switch loggers; TB scalars appear; CSV grows per step.

- [x] **Metrics meter**
  - **Where:** `titanax/logging/meter.py`
  - **Add:** Track throughput (samples/s), step latency, rolling averages; publish via logger interface.
  - **DoD:** Minimal DP example prints throughput and rolling loss/accuracy.

- [x] **Run header summary**
  - **Where:** `titanax/exec/engine.py`
  - **Add:** At start, log: Titanax version, JAX version, device count, mesh shape, plan summary, optimizer+LR schedule summary.
  - **DoD:** Present at beginning of all scripts; covered by a golden log test.

**✅ Section 7 Summary:** Added CSV and TensorBoard loggers alongside the stdout loggers, introduced a MetricsMeter that augments training logs with throughput/latency rolling averages, and emit a run header from `Engine.fit`. Minimal DP example now surfaces the new observability data and unit tests cover the logging backends and header output.

---

## 8) Optimizers

- [x] **Optax adapter finishing touches**
  - **Where:** `titanax/optim/optax_adapter.py`
  - **Change:** Keep LR schedules in Optax; expose `current_lr(step)`; document no duplicate scaling.
  - **DoD:** Minimal example prints current LR each step; moments unaffected by external scaling.
  - **✅ COMPLETED:** Added `current_lr(step)` method as alias to `get_learning_rate(step)`. Documentation clearly explains that Optax handles LR scheduling internally to avoid breaking adaptive optimizers. Engine automatically logs `learning_rate` in metrics each step.

- [x] **Common recipes**
  - **Where:** `titanax/optim/recipes.py`
  - **Add:** `adamw()` and `sgd()` convenience with typical defaults and schedule hooks (cosine, linear warmup).
  - **DoD:** Unit tests assert parameter registry/config correctness.
  - **✅ COMPLETED:** Created comprehensive `recipes.py` module with enhanced `adamw()`, `sgd()`, and `adam_with_cosine_schedule()` functions supporting warmup/decay schedules. Added `LRSchedules` utility class with common schedule patterns. All functions include extensive parameter validation and 21 comprehensive unit tests covering edge cases, schedule behavior, and parameter correctness.

---

## 9) Exceptions & Error UX ✅ COMPLETED

- [x] **Helpful exceptions with suggestions**
  - **Where:** `titanax/exceptions.py`
  - **Add/Verify:** `MeshValidationError`, `PlanValidationError`, `CollectiveError`, `ShardingError` each with a `suggestion` field.
  - **DoD:** Tests assert both message and suggestion are surfaced.
  - **✅ COMPLETED:** All required exceptions (MeshError, PlanError, CollectiveError, ShardingError) exist with suggestion fields. Comprehensive convenience functions (`mesh_validation_error`, `plan_validation_error`, `collective_error`, `sharding_error`) provide standardized error creation. Full test suite in `tests/unit/test_exceptions.py` validates error hierarchy, message formatting, and suggestion quality.

- [x] **Fail-fast config validation**
  - **Where:** `titanax/quickstart.py`
  - **Change:** Validate user inputs (batch size, axis names, optimizer config) before compile; raise typed exceptions.
  - **DoD:** Passing invalid axis name yields a single clear error before any JIT happens.
  - **✅ COMPLETED:** Added comprehensive fail-fast validation in `simple_data_parallel()` via `_validate_data_parallel_config()` function. Validates batch_size, learning_rate, precision, and loggers before any expensive operations. Enhanced mesh and plan creation with try-catch blocks that raise helpful `MeshError`/`PlanError` exceptions with actionable suggestions. Full test coverage in `tests/unit/test_quickstart_validation.py` ensures clear error messages with helpful suggestions for common mistakes.

---

## 10) Quickstart API ✅ COMPLETED

- [x] **`simple_data_parallel(...)`**
  - **Where:** `titanax/quickstart.py`
  - **DoD:** Returns an `Engine` configured for DP with: mesh, plan, precision, optimizer, checkpoint, logger; a minimal example trains for K steps on toy data.
  - **✅ COMPLETED:** Implemented comprehensive `simple_data_parallel()` function with full validation, error handling, and sensible defaults. Created example in `examples/quickstart_example.py` that demonstrates the API. All 19 quickstart validation tests pass.

- [x] **`simple_tensor_parallel(...)`** (explicit NotImplemented with guidance)
  - **Where:** `titanax/quickstart.py`
  - **DoD:** Raises `NotImplementedError` with a link to the TP example and instructions to use `Plan + tp_helpers` directly.
  - **✅ COMPLETED:** Implemented `simple_tensor_parallel()` that raises `NotImplementedError` with helpful guidance pointing users to the TP example and suggesting `Plan + tp_helpers` for manual TP setup. Includes clear messaging that full TP quickstart will be available in phase P1.

---

## 11) Tests

- [ ] **CPU‑only end‑to‑end DP test**
  - **Where:** `tests/test_e2e_dp_cpu.py`
  - **DoD:** Trains a tiny model 2 steps, asserts loss decreases and metrics/logging/checkpoint hooks were called.

- [ ] **Gradient accumulation test**
  - **Where:** `tests/test_grad_accum.py`
  - **DoD:** Asserts equivalence of microbatched vs. large batch within tolerance.

- [ ] **Collectives tests**
  - **Where:** `tests/test_collectives.py`
  - **DoD:** Validates psum/pmean/all_gather/reduce_scatter/broadcast/all_to_all on synthetic data.

- [ ] **Plan validation tests**
  - **Where:** `tests/test_plan_validation.py`
  - **DoD:** Covers allowed/forbidden axis combos and helpful error messages.

- [ ] **Checkpoint round‑trip**
  - **Where:** `tests/test_checkpoint_orbax.py`
  - **DoD:** Save/restore `TrainState`, verify exact match of params/opt_state/step/rngs.

---

## 12) CI/CD

- [ ] **GitHub Actions (CPU matrix)**
  - **Where:** `.github/workflows/ci.yml`
  - **Add:** Lint (ruff), format (black --check), typecheck (mypy), unit tests (pytest) on Python 3.11/3.12 and two recent JAX versions.
  - **DoD:** All jobs pass; cache wheels to keep runtime reasonable.

- [ ] **Pre-commit config**
  - **Where:** `.pre-commit-config.yaml`
  - **DoD:** Hooks for black/ruff/mypy/trailing-whitespace; contributors can run `pre-commit install`.

---

## 14) Non-Goals (for now)

- Full-featured pipeline parallel schedulers beyond 1F1B
- Advanced rematerialization heuristics
- Distributed dataset ingestion/streaming

Keep these explicit to manage expectations.

---

## Suggested Milestone Order

1. **Repo hygiene + README + example + import/test smoke**
2. **Engine/DP core: microbatching, errors, logging header**
3. **Checkpointing + resume**
4. **Collectives coverage + tests**
5. **TP helpers + minimal TP/PP examples**
6. **CI matrix + docs**

---

## Acceptance Gate for “Phase 1 Complete”

- ✅ Minimal DP example runs out-of-the-box on CPU
- ✅ `pytest -q` passes locally and in CI
- ✅ Checkpoint round-trip and resume works
- ✅ Collectives namespace covers the common ops with axis validation
- ✅ README documents usage, quickstart, and roadmap
- ✅ One minimal TP and one minimal PP example exist (marked experimental)
