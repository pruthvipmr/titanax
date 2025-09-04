# Titanax: Explicit-Parallel JAX Training Framework — Technical Specification

**Version:** v0.1 (Design Freeze)

**Audience:** ML/Systems engineers and an AI coding agent implementing the framework.

**One‑sentence summary:** Titanax is a lightweight training framework that brings the user ergonomics of Hugging Face Accelerate / TorchTitan to **JAX**, while requiring **explicit** data/tensor/pipeline parallelization (no XLA auto‑sharding). Users declare meshes, sharding rules, and collectives; Titanax wires the training loop, checkpointing, and observability around those choices.

---

## 0) Goals & Non‑Goals

### Goals
1. **Ergonomics:** Run the same training script on 1 GPU, multi‑GPU, or multi‑host with minimal user code changes.
2. **Explicit parallelism:** Users must **declare** how arrays and computations are partitioned (DP/TP/PP). The framework performs **no automatic partitioning**.
3. **Production scaffolding:** Provide primitives for initialization, logging, checkpointing, mixed precision, microbatching, and reproducible randomness.
4. **Composable plans:** Parallel “plans” (DP/TP/PP) can be composed (e.g., DP×TP) and inspected/validated before compile.

### Non‑Goals (v0)
- No XLA auto‑sharding/auto‑partitioning.
- No model zoo; users bring their own model functions (optionally via Flax/Haiku adapters in v0.2+).
- No elastic world size / job preemption recovery beyond basic checkpoint resume.
- No TPU‑specific pipeline scheduling (generic SPMD only; TPU supported as standard JAX devices).

---

## 1) Glossary
- **JAX:** Numerical computing with composable transformations (`jit`, `grad`, `pmap`, `pjit`).
- **XLA:** Compiler used by JAX; Titanax uses XLA compilation but does **not** use auto‑sharding.
- **SPMD:** Single Program, Multiple Data; you write one program executed in parallel across shards.
- **Mesh:** Logical grid of devices with named axes (e.g., `("data", "model")`).
- **PartitionSpec:** Declarative mapping from array dimensions to mesh axes.
- **NamedSharding:** A JAX sharding object tying `PartitionSpec` to a mesh.
- **Collectives:** Cross‑replica ops like `psum`, `pmean`, `all_gather`, `reduce_scatter`, `ppermute`.
- **DP/TP/PP:** Data Parallel / Tensor (model) Parallel / Pipeline Parallel.

---

## 2) High‑Level User Experience (UX)

### Example: Minimal DP training script
```python
import jax, jax.numpy as jnp
import titanax as tx

mesh = tx.MeshSpec(devices="all", axes=("data",))
plan = tx.Plan(data_parallel=tx.DP(axis="data"))

engine = tx.Engine(mesh=mesh, plan=plan,
                   optimizer=tx.optax.adamw(3e-4),
                   precision=tx.Precision(bfloat16=True),
                   checkpoint=tx.OrbaxCheckpoint("ckpts/run1"),
                   loggers=[tx.loggers.Basic()])

@tx.step_fn
def train_step(state, batch):
    def loss_fn(p):
        logits = model_apply(p, batch["x"])  # pure function
        return cross_entropy(logits, batch["y"])  
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = tx.collectives.psum(grads, axis="data")  # explicit collective
    state = state.apply_gradients(grads=grads)
    return state, {"loss": loss}

engine.fit(train_step, data=train_data, steps=10_000)
```

### Example: DP × TP with explicit rules
```python
mesh = tx.MeshSpec(devices="all", axes=("data", "model"), shape=(None, 2))
plan = tx.Plan(
    data_parallel=tx.DP(axis="data"),
    tensor_parallel=tx.TP(axis="model", rules={
        # Parameter path patterns → PartitionSpec layouts
        "transformer/attn/qkv/kernel": ("model", None),
        "transformer/mlp/wi/kernel": ("model", None),
        "transformer/mlp/wo/kernel": (None, "model"),
    })
)
```

---

## 3) System Overview & Architecture

**Layers:**
1. **Runtime & Control Plane**
   - Multi‑host initialization, mesh discovery, process group scaffolding.
2. **Parallel Plans**
   - `DP`, `TP`, `PP` plan objects with validators; composition engine.
3. **Collectives Layer**
   - Typed wrappers for `psum`, `all_gather`, `reduce_scatter`, `ppermute`, `broadcast`.
4. **Execution Engine**
   - Step decoration, compilation, gradient accumulation, precision, metrics, hooks.
5. **State & Checkpointing**
   - Sharded TrainState, optimizer adapters, partition‑aware checkpoint I/O.
6. **Data Pipeline**
   - Sharded dataloaders, host‑local prefetch, global PRNG coordination.
7. **Observability**
   - Logging, metrics, profiling hooks.
8. **CLI/Launcher & Config**
   - Simple multi‑host launch and YAML/CLI config mapping.

---

## 4) Detailed Component Specs

### 4.1 Runtime & Control Plane
**Responsibilities**
- Initialize JAX distributed (`jax.distributed.initialize`) when multi‑host env vars are present.
- Enumerate devices and build logical meshes; support 1D/2D meshes.
- Validate mesh axis sizes vs. world/device counts.

**Public API**
```python
@dataclasses.dataclass
class MeshSpec:
    devices: str | list | None  # "all" | list of jax.Device | None (default all)
    axes: tuple[str, ...]       # e.g., ("data",) or ("data","model")
    shape: tuple[int|None, ...] | None = None  # optional axis lengths; None→infer
    topology: dict[str, any] | None = None     # optional hints (NVLink groups, etc.)

    def build(self) -> jax.sharding.Mesh: ...
    def describe(self) -> str: ...
```

```python
class ProcessGroups:
    def __init__(self, mesh: jax.sharding.Mesh): ...
    def size(self, axis: str) -> int: ...
    def rank(self, axis: str) -> int: ...
```

**Behavioral Notes**
- If `shape` has `None`, auto‑compute length from device count/divisibility.
- Fail fast if `batch_size` not divisible by DP axis size (unless microbatching is enabled).

### 4.2 Parallel Plans

#### 4.2.1 Plan Composition
```python
@dataclasses.dataclass
class Plan:
    data_parallel: DP | None = None
    tensor_parallel: TP | None = None
    pipeline_parallel: PP | None = None

    def validate(self, mesh: MeshSpec) -> None: ...
    def describe(self) -> str: ...
```

#### 4.2.2 Data Parallel (DP)
- **Axis:** required name (e.g., `"data"`).
- **Semantics:**
  - Split per‑step batch across DP shards.
  - Explicit gradient sync via `psum` or `pmean` on that axis.
  - Optional gradient accumulation across microbatches prior to all‑reduce.

```python
@dataclasses.dataclass
class DP:
    axis: str
    accumulate_steps: int = 1  # microbatches per step
    sync_metrics: bool = True
```

#### 4.2.3 Tensor Parallel (TP)
- **Axis:** name (e.g., `"model"`).
- **Rules:** mapping from parameter path patterns to `PartitionSpec` tuples.
- **Collectives:** explicit `all_gather`, `reduce_scatter`, or local matmul when shards align.

```python
@dataclasses.dataclass
class TP:
    axis: str
    rules: dict[str, tuple[str|None, ...]]  # param path → PartitionSpec dims
    prefer_reduce_scatter: bool = True

    def spec_for(self, param_path: str, param_shape: tuple[int,...]) -> jax.sharding.PartitionSpec: ...
```

**Reference rules for Transformers (baseline):**
- `attn/qkv/kernel: ("model", None)`  — shard columns per‑head dimension.
- `attn/out/kernel: (None, "model")`  — shard rows.
- `mlp/wi/kernel: ("model", None)`    — shard intermediate dim.
- `mlp/wo/kernel: (None, "model")`    — shard output dim.

#### 4.2.4 Pipeline Parallel (PP)
- **Stages:** user‑declared boundaries; each stage owns a param subtree and forward fn.
- **Schedule:** 1F1B with microbatches; explicit inter‑stage sends/receives.

```python
@dataclasses.dataclass
class Stage:
    name: str
    param_filter: list[str]  # param path globs belonging to this stage
    fwd: typing.Callable     # (params_subset, x) -> y

@dataclasses.dataclass
class PP:
    axis: str                # can reuse TP axis or a dedicated pipeline axis
    stages: list[Stage]
    microbatch_size: int
    checkpoint_ratio: float = 0.0  # activation remat
```

**Runtime responsibilities:** construct stage graph, slice microbatches, orchestrate 1F1B schedule, insert collectives for boundary transfers (e.g., `all_gather` when shape partitions differ).

### 4.3 Collectives Layer
Typed wrappers that assert axis presence and tree‑shape compatibility.

```python
class collectives:
    @staticmethod
    def psum(tree, axis: str): ...
    @staticmethod
    def pmean(tree, axis: str): ...
    @staticmethod
    def all_gather(x, axis: str, axis_index: int | None = None): ...
    @staticmethod
    def reduce_scatter(x, axis: str, op: str = "add"): ...
    @staticmethod
    def broadcast(x, axis: str, src_index: int = 0): ...
    @staticmethod
    def ppermute(x, axis: str, perm): ...
```

### 4.4 Execution Engine

**Responsibilities**
- Decorate step functions, perform JIT compilation, manage PRNG, apply precision policy, handle gradient accumulation, metrics/logging hooks, and distributed barriers when necessary.

```python
@dataclasses.dataclass
class Precision:
    bfloat16: bool = False
    fp16: bool = False
    loss_scaling: bool = False
    enable_x32_params: bool = False  # params stay in f32 while activations in bf16

@dataclasses.dataclass
class TrainState:
    params: PyTree
    opt_state: PyTree
    step: int
    rngs: dict[str, jax.Array]  # named rng streams per host/device

class Engine:
    def __init__(self, mesh: MeshSpec, plan: Plan, optimizer, precision: Precision,
                 checkpoint: "CheckpointStrategy" | None = None,
                 loggers: list["Logger"] | None = None,
                 hooks: list["Hook"] | None = None): ...

    def fit(self, step_fn, data, steps: int, eval_every: int | None = None): ...
    def eval(self, step_fn, data, steps: int): ...
    def compile(self, step_fn, example_batch): ...
    def state_dict(self) -> dict: ...
```

**Step decoration**
```python

def step_fn(f):
    """Decorator that:
    - Injects precision casts (policy‑driven)
    - Pre/post hooks (metrics, logging)
    - Validates collective usage against plan axes
    """
    ...
```

**Gradient Accumulation**
- Split local per‑device batch into `accumulate_steps` microbatches.
- Compute grads; either sum locally then `psum` once, or `psum` each microbatch (configurable).

**PRNG Management**
- Global seed at job start; derive per‑host, per‑device streams via `fold_in(host_id)` and `fold_in(device_id)`.
- Named streams (`"data"`, `"dropout"`, `"init"`).

**Compilation Policy**
- Cache compiled executables by `(mesh_signature, shapes, dtypes, precision policy)`.
- Guard against unintended recompiles (shape polymorphism) with optional static shape assertions.

### 4.5 Optimizers & Losses
- **Optax adapter** (stateless) to connect any `GradientTransformation`.
- Built‑in helpers for AdamW, Lion (thin wrappers over Optax factories).

```python
class OptimizerAdapter:
    def __init__(self, tx): ...  # tx: optax.GradientTransformation
    def init(self, params): ...
    def update(self, grads, opt_state, params=None): ...
```

### 4.6 Checkpointing (Partition‑Aware)

**Strategy Interface**
```python
class CheckpointStrategy(typing.Protocol):
    def save(self, state: TrainState, step: int): ...
    def load(self, step: int | str | None = None) -> TrainState: ...
    def latest_step(self) -> int | None: ...
```

**Orbax Implementation** (default)
- Save param/optimizer pytrees with sharding metadata.
- Metadata file includes: mesh axes, PartitionSpecs, dtypes, optimizer config hash, Titanax version.
- Load path supports **resharding** when current mesh differs:
  1) Load on single host
  2) Repartition according to current `Plan` specs
  3) Broadcast to replicas (explicit `broadcast`/`all_gather`).

### 4.7 Data Pipeline

**Interfaces**
```python
class Dataset(typing.Protocol):
    def __iter__(self): ...  # yields host-local batches (already sharded or shardable)

class ShardedDataLoader:
    def __init__(self, dataset: Dataset, dp_axis_size: int, prefetch: int = 2,
                 global_shuffle: bool = True, seed: int | None = None): ...
    def __iter__(self): ...
```

**Features**
- Host‑local prefetch queues.
- Optional global shuffle via collective PRNG coordination (same seed across hosts, different per‑step offsets).
- Batch divisibility checks and automatic padding (configurable) for last batch.

### 4.8 Observability

**Logger protocol**
```python
class Logger(typing.Protocol):
    def log_scalar(self, name: str, value: float, step: int): ...
    def log_dict(self, metrics: dict[str, float], step: int): ...
```

**Included loggers**
- `Basic` (stdout)
- `CSV` (append to file)
- `TensorBoard` (optional)

**Profiling Hooks**
- Start/stop XLA profiler around training windows.
- Emit compile time vs. run time, tokens/sec, memory watermark if available.

### 4.9 CLI / Launcher & Config

**CLI Entrypoint**
```
$ titanax.run \
  --module user_script:main \
  --config configs/gpt_small.yaml \
  --coordinator 10.0.0.1:1234 \
  --num_processes 8 --process_id $RANK
```

**Behavior**
- If `--coordinator`, call `jax.distributed.initialize` with provided world info; otherwise single‑process mode.
- Pass config as env vars or argv into `main(config)`.

**YAML Config Schema (excerpt)**
```yaml
mesh:
  axes: [data, model]
  shape: [8, 2]
plan:
  dp:
    axis: data
    accumulate_steps: 4
  tp:
    axis: model
    rules:
      "transformer/attn/qkv/kernel": [model, null]
      "transformer/mlp/wo/kernel": [null, model]
precision:
  bfloat16: true
optimizer:
  name: adamw
  lr: 3.0e-4
checkpoint:
  path: ckpts/run1
logging:
  - type: basic
  - type: tensorboard
```

---

## 5) Parallelism Algorithms (Operational Details)

### 5.1 Data Parallel (DP)
**Per step:**
1. Split batch into `dp_world_size` shards (host‑local split + device local split if needed).
2. Compute grads locally on each shard.
3. `psum` grads across DP axis.
4. Average grads by DP world size; optimizer update.
5. Optionally `pmean` metrics.

**Microbatching:**
- Repeat (2) for `accumulate_steps` microbatches; sum grads locally; perform (3) once at the end.

### 5.2 Tensor Parallel (TP)
**Weight layouts:** dictated by `rules` → PartitionSpecs. For linear layers with weight `W[in,out]`:
- Column parallel: shard `out` across `model` axis → local matmul `X @ W_local`, `all_gather` outputs if consumer needs full `out`.
- Row parallel: shard `in` across `model` axis → `all_gather` inputs or `reduce_scatter` outputs depending on consumer.

**Attention patterns:**
- `qkv` projection often column‑sharded → local compute, concatenate via `all_gather` across model axis。
- Output projection often row‑sharded → `reduce_scatter` or local accumulation.

**Autograd:** gradients follow reverse collectives (e.g., `reduce_scatter` in fwd → `all_gather` in bwd), made explicit in layer wrappers or in `step_fn` utilities.

### 5.3 Pipeline Parallel (PP)
**1F1B schedule:**
- Divide global batch into `M` microbatches.
- Warm‑up: stage 0 runs μ0 fwd, sends activations to stage 1; stage 1 starts when data arrives; etc.
- Steady state: each stage alternates forward on μi and backward on μi−k where k = #stages−1.
- Cool‑down: finalize remaining backwards.

**Implementation mechanics:**
- Stage objects own params and local `jit`ted fwd/bwd fns.
- Inter‑stage transfers: `all_gather` or `ppermute` depending on partitioning mismatch.
- Activation remat according to `checkpoint_ratio` (e.g., remat every N layers per stage).

---

## 6) Error Handling & Validation
- **Mesh/Plan validation:** axis existence, size > 0, divisibility checks.
- **Sharding rule validation:** parameter paths must match at least one rule (or explicit `unsharded` intent).
- **Collective guards:** `collectives.*` assert `axis` exists in current mesh; shape compatibility checks for tree leaves.
- **Determinism:** optional flags to assert equal results between single‑device and DP runs for small examples.
- **Failure modes:** clean shutdown on one‑host failure when possible; understandable error messages with suggested fixes.

---

## 7) Performance Guidance (Docs to Ship)
- When to use `pmap` vs `pjit` vs `shard_map`.
- Avoiding recompiles: static shapes, avoid Python control flow in `step_fn`; use `jax.lax.cond`/`switch`.
- Choosing DP×TP: start with DP, add TP when model no longer fits or when comms/batch scaling saturate.
- Microbatching vs activation checkpointing trade‑offs.

---

## 8) Testing Strategy

### 8.1 Unit Tests
- MeshSpec build/describe; Plan validators.
- Collectives type/shape checks; axis presence.
- OptimizerAdapter: parity with pure Optax call.

### 8.2 Integration Tests
- DP MNIST: loss parity between 1‑device and N‑device within tolerance.
- TP toy MLP: check forward equivalence vs single‑device gather.
- PP 2‑stage toy CNN: verify schedule and gradients flow.
- Checkpoint save/load/reshard round‑trip.

### 8.3 Benchmark Suites
- Throughput tokens/sec vs. device count for GPT‑small.
- Compile time vs run time; effect of accumulate_steps.
- DP vs TP scaling curves.

---

## 9) Example Recipes (to ship in `examples/`)
1. **MNIST‑DP** (single/multi‑host) with explicit psum and microbatching.
2. **GPT‑Small TP (1D)** with canonical transformer sharding rules; DP×TP composition.
3. **GPT‑Small PP (2 stages)** with 1F1B schedule.
4. **Mixed Precision (bf16)** DP×TP with loss scaling.
5. **Checkpoint/Reshard**: save on TP=2, restore on TP=4.

---

## 10) Repo Layout & Deliverables
```
 titanax/
   __init__.py
   runtime/
     mesh.py           # MeshSpec, ProcessGroups
     init.py           # distributed init helpers
   parallel/
     plan.py           # Plan, DP, TP, PP
     rules.py          # sharding rules utilities
     pipeline.py       # Stage, 1F1B scheduler
   exec/
     engine.py         # Engine, step_fn, TrainState, Precision
     collectives.py    # typed wrappers
     prng.py           # RNG management
   optim/
     optax_adapter.py
   io/
     checkpoint.py     # Strategy interface
     orbax_io.py       # Orbax implementation
   data/
     dataloader.py
   logging/
     base.py
     basic.py
     csv.py
     tensorboard.py
   launch/
     cli.py            # titanax.run
   docs/
     *.md
   examples/
     mnist_dp/
     gpt_small_tp/
     gpt_small_pp/
   tests/
     unit/
     integration/
     benchmarks/
```

---

## 11) Milestones & Acceptance Criteria

### P0 — DP Core (2–3 weeks)
- MeshSpec, DP plan with microbatching, collectives wrappers.
- Engine.fit with step decoration; Optax adapter.
- Basic logger; Orbax checkpoint save/load.
- **Acceptance:** MNIST‑DP scales 1→8 GPUs; parity of loss within 1e‑4; resume from checkpoint.

### P1 — TP Core (2–3 weeks)
- TP rules and PartitionSpec emission; layer wrappers for common linear/attn kernels.
- DP×TP composition; example GPT‑small TP.
- **Acceptance:** GPT‑small reaches reference perplexity within tolerance; sharded checkpoint round‑trip.

### P2 — PP Core (2–3 weeks)
- Stage API, 1F1B scheduler, activation remat.
- Example 2‑stage pipeline.
- **Acceptance:** Pipeline example trains; throughput gain vs single‑stage at equivalent memory.

### P3 — DX & Docs (1–2 weeks)
- CLI launcher, YAML config, profiling hooks, docs site.
- **Acceptance:** Quickstart completes on single and multi‑host; docs cover parallel plans and troubleshooting.

---

## 12) Risks & Mitigations
- **Mesh/rule complexity:** Provide `plan.describe()` with explicit printed layouts; dry‑run mode that compiles small shapes.
- **Collective misuse:** Strong runtime assertions; test coverage of typical error paths.
- **Recompile storms:** Shape guards; cache keys; documentation on static shapes.
- **Multi‑host fragility:** Clear env var contract; backoff/retry on rendezvous; timeouts with helpful errors.

---

## 13) Security, Compliance, and Reproducibility
- No PII in logs by default; redact batch contents in logging.
- Deterministic runs when seeds and data order fixed; document sources of nondeterminism (e.g., atomics).
- Version stamps: include Titanax version, JAX version, and plan/mesh signature in checkpoints.

---

## 14) Future Work (v0.2+)
- Flax/Haiku adapters; parameter naming conventions helper.
- FP8 flows; ZeRO‑style optimizer state partitioning.
- 2D/3D tensor parallel patterns; sequence parallel.
- Advanced schedulers (interleaved PP), overlap compute/comm.
- Elastic world size via coordination service.

---

## 15) Implementation Notes (for coding agent)
- Prefer `pjit` with `NamedSharding` for TP; prefer `pmap` for pure DP where appropriate; `shard_map` for custom SPMD loops.
- Use small toy configs in tests to minimize compile times.
- Heavily type‑annotate public APIs; use `dataclasses` over `pydantic` to avoid extra deps.
- Keep `examples/` runnable in <10 minutes on commodity GPUs.

---

**End of Technical Specification**

