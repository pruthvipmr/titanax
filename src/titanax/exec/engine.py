"""Titanax execution engine for distributed training.

This module provides the core Engine class, TrainState, and Precision configuration
for managing training loops with explicit parallelization.
"""

import dataclasses
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from ..optim.optax_adapter import OptaxAdapter

import jax
import jax.numpy as jnp
from jax import tree_util

from ..types import (
    PyTree,
    Array,
    BatchData,
    LogDict,
    StepFunction,
    Logger,
    CheckpointStrategy,
    Params,
    OptState,
    Mesh,
    PartitionSpec,
)
from ..logging.meter import MetricsMeter
from ..runtime.mesh import MeshSpec
from ..parallel.plan import Plan
from ..exceptions import CheckpointError, EngineError
from ..parallel.sharding import (
    apply_named_sharding,
    build_param_specs,
    shard_batch_specs,
)
from .collectives import set_current_mesh
from .step_fn import update_rngs
from .compile import compile_step_with_plan
from .prng import create_host_device_rngs, validate_rng_keys
from .._version import __version__ as _titanax_version


@dataclasses.dataclass(frozen=True)
class Precision:
    """Precision policy configuration for training.

    Controls numeric precision, mixed precision training, and loss scaling.

    Args:
        bfloat16: Use bfloat16 for activations and gradients
        fp16: Use float16 for activations and gradients (alternative to bfloat16)
        loss_scaling: Enable automatic loss scaling for fp16 training
        enable_x32_params: Keep parameters in float32 while using lower precision for activations
    """

    bfloat16: bool = False
    fp16: bool = False
    loss_scaling: bool = False
    enable_x32_params: bool = False

    def __post_init__(self) -> None:
        """Validate precision configuration."""
        if self.bfloat16 and self.fp16:
            raise EngineError("Cannot enable both bfloat16 and fp16 precision")

        if self.loss_scaling and not self.fp16:
            raise EngineError("Loss scaling requires fp16=True")

    @property
    def dtype(self) -> jnp.dtype:
        """Get the target dtype for activations."""
        if self.bfloat16:
            return jnp.bfloat16
        elif self.fp16:
            return jnp.float16
        else:
            return jnp.float32

    @property
    def param_dtype(self) -> jnp.dtype:
        """Get the target dtype for parameters."""
        if self.enable_x32_params:
            return jnp.float32
        return self.dtype

    def describe(self) -> str:
        """Return a human-readable description of the precision policy."""
        parts = []
        if self.bfloat16:
            parts.append("bfloat16")
        elif self.fp16:
            parts.append("float16")
        else:
            parts.append("float32")

        if self.enable_x32_params:
            parts.append("x32_params")
        if self.loss_scaling:
            parts.append("loss_scaling")

        return " + ".join(parts)


@dataclasses.dataclass
class TrainState:
    """Training state containing parameters, optimizer state, and metadata.

    This class holds all the mutable state needed for training, including
    model parameters, optimizer state, current step count, and PRNG keys.

    Args:
        params: Model parameters as a PyTree
        opt_state: Optimizer state as a PyTree
        step: Current training step number
        rngs: Dictionary of named PRNG keys for different uses
        _optimizer: Optional reference to the optimizer (set by Engine)
    """

    params: Params
    opt_state: OptState
    step: int
    rngs: Dict[str, Array]
    _optimizer: Optional["OptaxAdapter"] = None

    def apply_gradients(
        self, *, grads: PyTree, optimizer: Optional["OptaxAdapter"] = None, **kwargs
    ) -> "TrainState":
        """Apply gradients and return new training state.

        Args:
            grads: Gradients to apply
            optimizer: Optional optimizer adapter (required if not set during creation)
            **kwargs: Additional arguments for optimizer

        Returns:
            New TrainState with updated parameters and optimizer state
        """
        if optimizer is None:
            optimizer = self._optimizer

        if optimizer is None:
            raise EngineError(
                "No optimizer provided for apply_gradients",
                suggestion="Pass optimizer argument or ensure Engine has optimizer configured",
            )

        try:
            # Apply gradients using the optimizer
            new_params, new_opt_state = optimizer.apply_gradients(
                grads=grads,
                opt_state=self.opt_state,
                params=self.params,
                step=self.step,
                **kwargs,
            )

            return dataclasses.replace(
                self, params=new_params, opt_state=new_opt_state, step=self.step + 1
            )

        except Exception as e:
            raise EngineError(
                f"Failed to apply gradients: {e}",
                suggestion="Check that gradients and parameters have compatible shapes",
            ) from e

    def replace(self, **kwargs) -> "TrainState":
        """Return a copy of the state with specified fields replaced."""
        return dataclasses.replace(self, **kwargs)

    def tree_flatten(self):
        """Flatten TrainState for JAX PyTree operations."""
        children = (self.params, self.opt_state, self.rngs)
        aux_data = (self.step, self._optimizer)
        return children, aux_data

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten TrainState from JAX PyTree operations."""
        step, optimizer = aux_data
        params, opt_state, rngs = children
        return cls(
            params=params,
            opt_state=opt_state,
            step=step,
            rngs=rngs,
            _optimizer=optimizer,
        )


# Register TrainState as a JAX PyTree
tree_util.register_pytree_node(
    TrainState, TrainState.tree_flatten, TrainState.tree_unflatten
)


class Engine:
    """Core training engine for distributed JAX training.

    The Engine orchestrates training loops, manages state, handles checkpointing,
    and coordinates logging across distributed devices.

    Args:
        mesh: Mesh specification for device layout
        plan: Parallel plan (DP/TP/PP configuration)
        optimizer: Optimizer instance (placeholder for now)
        precision: Precision policy configuration
        checkpoint: Checkpoint strategy for save/load operations
        loggers: List of logger instances for metrics
        checkpoint_interval: Save checkpoint every N steps (None disables periodic saves)
    """

    def __init__(
        self,
        mesh: MeshSpec,
        plan: Plan,
        optimizer: "OptaxAdapter",
        precision: Precision = Precision(),
        checkpoint: Optional[CheckpointStrategy] = None,
        loggers: Optional[List[Logger]] = None,
        checkpoint_interval: int | None = 1000,
        metrics_meter: Optional[MetricsMeter] = None,
    ):
        self.mesh_spec = mesh
        self.plan = plan
        self.optimizer = optimizer
        self.precision = precision
        self.checkpoint = checkpoint
        self.loggers = loggers or []
        self._meter = metrics_meter or MetricsMeter()
        if checkpoint_interval is not None and checkpoint_interval <= 0:
            raise EngineError(
                "checkpoint_interval must be positive when provided",
                suggestion="Provide checkpoint_interval=None to disable periodic saves or a positive integer",
            )
        self.checkpoint_interval = checkpoint_interval

        # Build and validate the mesh
        self._mesh = self._validate_and_build_mesh()

        # Set the mesh in the collectives context
        set_current_mesh(self._mesh)

        # Validate plan compatibility with mesh
        self._validate_plan()

        # Initialize compiled step function (will be set when step_fn is registered)
        self._compiled_step_fn: Optional[StepFunction] = None
        self._registered_step_fn: Optional[StepFunction] = None
        self._pending_step_fn: Optional[StepFunction] = None
        self._pending_compile_params: Dict[str, Any] = {}
        self._param_spec_tree: Optional[PyTree] = None
        self._state_spec_tree: Optional[PyTree] = None
        self._batch_spec_tree: Optional[PyTree] = None

    def _validate_and_build_mesh(self) -> Mesh:
        """Validate and build the JAX mesh from specification."""
        try:
            mesh = self.mesh_spec.build()
            return mesh
        except Exception as e:
            raise EngineError(
                f"Failed to build mesh from specification: {e}",
                suggestion="Check mesh axes and device availability",
            ) from e

    def _validate_plan(self) -> None:
        """Validate the parallel plan against the mesh specification."""
        try:
            # Check if plan has validate method (for composite plans) or validate_with_mesh
            if hasattr(self.plan, "validate"):
                self.plan.validate(self.mesh_spec)
            elif hasattr(self.plan, "validate_with_mesh"):
                self.plan.validate_with_mesh(self.mesh_spec)
            else:
                raise AttributeError(f"Plan {type(self.plan)} has no validation method")
        except Exception as e:
            raise EngineError(
                f"Plan validation failed: {e}",
                suggestion="Check that plan axes match mesh axes",
            ) from e

    def register_step_fn(self, step_fn: StepFunction) -> None:
        """Register and compile a step function for training.

        Args:
            step_fn: The step function to compile and use for training
        """
        self._registered_step_fn = step_fn

        # Get compilation parameters from the decorated function
        compile_params = getattr(step_fn, "_compile_params", {})
        donate_argnums = compile_params.get("donate_argnums", (0,))
        static_argnums = compile_params.get("static_argnums", ())
        in_shardings = compile_params.get("in_shardings")
        out_shardings = compile_params.get("out_shardings")

        # Reset cached sharding state whenever the step function changes
        self._param_spec_tree = None
        self._state_spec_tree = None
        self._batch_spec_tree = None

        pending_tp = (
            self.plan.tensor_parallel is not None
            and in_shardings is None
            and out_shardings is None
        )

        if pending_tp:
            # Defer compilation until runtime structures are available
            self._pending_step_fn = step_fn
            self._pending_compile_params = {
                "donate_argnums": donate_argnums,
                "static_argnums": static_argnums,
            }
            self._compiled_step_fn = None
        else:
            self._pending_step_fn = None
            self._pending_compile_params = {}

            # Compile with proper mesh context for collectives
            self._compiled_step_fn = compile_step_with_plan(
                step_fn,
                self.plan,
                self._mesh,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                donate_argnums=donate_argnums,
                static_argnums=static_argnums,
            )

    def create_state(
        self,
        params: Params,
        rngs: Optional[Dict[str, Array]] = None,
        rng_seed: int = 42,
    ) -> TrainState:
        """Create initial training state with proper per-device PRNG keys.

        Args:
            params: Initial model parameters
            rngs: Named PRNG keys, will create default 'dropout' key if None
            rng_seed: Seed for default PRNG key generation

        Returns:
            Initialized TrainState with proper multi-device PRNG keys
        """
        if rngs is None:
            # Create default per-device PRNG keys using host-device utilities
            rngs = create_host_device_rngs(
                base_seed=rng_seed, named_keys={"dropout": None}
            )

        # Validate PRNG keys
        validate_rng_keys(rngs)

        # Cast parameters to the target precision dtype
        cast_params = self._cast_params_to_precision(params)

        # Initialize optimizer state
        try:
            opt_state = self.optimizer.init(cast_params)
        except Exception as e:
            raise EngineError(
                f"Failed to initialize optimizer state: {e}",
                suggestion="Check that parameters are valid JAX PyTrees",
            ) from e

        new_state = TrainState(
            params=cast_params,
            opt_state=opt_state,
            step=0,
            rngs=rngs,
            _optimizer=self.optimizer,
        )

        self._ensure_state_spec_tree(new_state)

        return new_state

    def shard_params(self, params: Params) -> Params:
        """Apply tensor-parallel ``NamedSharding`` placements to ``params``."""

        if self.plan.tensor_parallel is None:
            return params

        spec_tree = self._ensure_param_spec_tree(params)
        if spec_tree is None:
            return params

        try:
            return apply_named_sharding(params, self._mesh, spec_tree)
        except Exception as exc:  # pragma: no cover - sharding errors handled upstream
            raise EngineError(
                f"Failed to apply tensor parallel sharding: {exc}",
                suggestion="Verify tensor parallel rules match parameter structure and mesh axes",
            ) from exc

    def step(self, state: TrainState, batch: BatchData) -> tuple[TrainState, LogDict]:
        """Execute a single training step.

        Args:
            state: Current training state
            batch: Input batch data

        Returns:
            Tuple of (updated_state, metrics)
        """
        if self._compiled_step_fn is None and self._pending_step_fn is not None:
            self._compile_pending_step_fn(state, batch)

        if self._compiled_step_fn is None:
            raise EngineError(
                "No step function registered",
                suggestion="Call register_step_fn() or engine.fit() first",
            )

        # Ensure mesh context is available for thread-local access
        set_current_mesh(self._mesh)

        # Execute the step
        return self._execute_step(state, batch)

    def save_checkpoint(self, state: TrainState) -> None:
        """Persist the provided training state using the configured checkpoint."""
        if self.checkpoint is None:
            raise EngineError(
                "No checkpoint strategy configured",
                suggestion="Configure checkpoint strategy in Engine constructor",
            )

        self._save_checkpoint(state, tag="manual", force=True)

    def _save_checkpoint(self, state: TrainState, *, tag: str, force: bool) -> None:
        """Internal helper for checkpoint persistence with logging."""
        if self.checkpoint is None:
            return

        if not isinstance(getattr(state, "step", None), int):
            raise EngineError(
                "State.step must be an integer before checkpointing",
                suggestion="Ensure the TrainState.step field is an int",
            )

        if not force:
            if self.checkpoint_interval is None:
                return
            if state.step % self.checkpoint_interval != 0:
                return

        state_to_save = state
        if hasattr(state, "replace") and getattr(state, "_optimizer", None) is None:
            state_to_save = state.replace(_optimizer=self.optimizer)

        try:
            self.checkpoint.save(state_to_save)
        except CheckpointError as err:
            raise EngineError(
                f"Failed to save checkpoint at step {state.step}: {err}",
                suggestion="Verify checkpoint backend configuration and disk permissions",
            ) from err

        self._log_scalar(f"checkpoint/{tag}", float(state.step), state.step)

    def _rehydrate_restored_state(
        self, state: TrainState | PyTree
    ) -> TrainState | PyTree:
        """Rehydrate optimizer state after loading raw checkpoint data."""
        if not isinstance(state, TrainState):
            return state

        if self.optimizer is None:
            return state

        try:
            reference_opt_state = self.optimizer.init(state.params)
        except Exception:
            return state

        try:
            ref_leaves, ref_treedef = tree_util.tree_flatten(reference_opt_state)
            saved_leaves, _ = tree_util.tree_flatten(state.opt_state)
            if len(ref_leaves) != len(saved_leaves):
                return state
            restored_opt_state = tree_util.tree_unflatten(ref_treedef, saved_leaves)
            return state.replace(opt_state=restored_opt_state)
        except Exception:
            return state

    def load_checkpoint(self, step: Optional[int] = None) -> TrainState:
        """Load training state from checkpoint.

        Args:
            step: Optional step number to load (loads latest if None)

        Returns:
            Loaded training state
        """
        if self.checkpoint is None:
            raise EngineError(
                "No checkpoint strategy configured",
                suggestion="Configure checkpoint strategy in Engine constructor",
            )

        state = self._rehydrate_restored_state(self.checkpoint.restore(step))

        # Ensure state is a TrainState with proper attributes
        if not hasattr(state, "step"):
            raise EngineError(
                f"Loaded checkpoint is not a valid TrainState: {type(state)}",
                suggestion="Check checkpoint format and ensure it was saved with current Titanax version",
            )

        if hasattr(state, "replace"):
            state = state.replace(_optimizer=self.optimizer)

        self._log_scalar("checkpoint/loaded_step", float(state.step), state.step)
        return state

    def fit(
        self,
        step_fn: StepFunction,
        data: Iterable[BatchData],
        steps: Optional[int] = None,
        state: Optional[TrainState] = None,
        continue_on_error: bool = False,
        resume_from: int | str | Path | None = None,
    ) -> TrainState:
        """Run the training loop.

        Args:
            step_fn: The step function to execute each training step
            data: Iterable of training data batches
            steps: Maximum number of steps to train (None for unlimited)
            state: Initial training state (None to restore from checkpoint)
            continue_on_error: If False (default), re-raise exceptions after logging.
                             If True, continue training after logging errors.
            resume_from: Step number or checkpoint directory path to resume from

        Returns:
            Final training state after training
        """
        if self._compiled_step_fn is None:
            self.register_step_fn(step_fn)

        run_state = state
        if run_state is None:
            if self.checkpoint is None:
                raise EngineError(
                    "No initial state provided and no checkpoint strategy configured",
                    suggestion="Either provide state parameter or configure checkpoint strategy",
                )

            resume_step: Optional[int] = None
            if resume_from is not None:
                if isinstance(resume_from, int):
                    resume_step = resume_from
                elif isinstance(resume_from, (str, Path)):
                    resume_path = Path(resume_from)
                    checkpoint_dir = getattr(self.checkpoint, "checkpoint_dir", None)
                    if (
                        checkpoint_dir is not None
                        and Path(checkpoint_dir) != resume_path
                    ):
                        raise EngineError(
                            "resume_from path does not match configured checkpoint directory",
                            suggestion="Instantiate the Engine checkpoint with the same directory",
                        )
                else:
                    raise EngineError(
                        f"Unsupported resume_from value: {resume_from}",
                        suggestion="Provide an integer step or a checkpoint directory path",
                    )

            try:
                restored_state = self._rehydrate_restored_state(
                    self.checkpoint.restore(resume_step)
                )
            except CheckpointError as err:
                raise EngineError(
                    f"Failed to restore checkpoint: {err}",
                    suggestion="Ensure checkpoints exist or provide an explicit initial state",
                ) from err

            if hasattr(restored_state, "replace"):
                run_state = restored_state.replace(_optimizer=self.optimizer)
            else:
                run_state = restored_state

            self._log_scalar(
                "checkpoint/loaded_step",
                float(getattr(run_state, "step", 0)),
                getattr(run_state, "step", 0),
            )
        else:
            if hasattr(run_state, "replace"):
                run_state = run_state.replace(_optimizer=self.optimizer)

        # Ensure mesh context is available for thread-local access
        set_current_mesh(self._mesh)

        current_step = 0
        self._meter.reset()
        self._log_run_header(run_state)
        self._save_checkpoint(run_state, tag="start", force=True)
        try:
            for batch in data:
                if (
                    self._compiled_step_fn is None
                    and self._pending_step_fn is not None
                    and run_state is not None
                ):
                    self._compile_pending_step_fn(run_state, batch)

                if self._compiled_step_fn is None:
                    raise EngineError(
                        "No step function registered",
                        suggestion="Call register_step_fn() or engine.step() before fit",
                    )

                if steps is not None and current_step >= steps:
                    break

                try:
                    step_start = time.perf_counter()
                    # Execute one training step
                    run_state, metrics = self._execute_step(run_state, batch)
                    step_time = time.perf_counter() - step_start
                    batch_size = self._infer_batch_size(batch)
                    meter_metrics = self._meter.update(
                        step=run_state.step,
                        metrics=metrics,
                        batch_size=batch_size,
                        step_time_s=step_time,
                    )
                    combined_metrics = dict(metrics)
                    combined_metrics.update(meter_metrics)
                    current_step += 1

                    # Log metrics
                    self._log_metrics(combined_metrics, run_state.step)

                    # Save checkpoint periodically
                    self._save_checkpoint(run_state, tag="periodic", force=False)

                except Exception as step_error:
                    # Log the error first
                    error_msg = (
                        f"Step execution failed at step {run_state.step}: {step_error}"
                    )
                    self._log_scalar(
                        "training/step_error", float(run_state.step), run_state.step
                    )
                    print(f"ERROR: {error_msg}")

                    # Re-raise unless continue_on_error is True
                    if not continue_on_error:
                        raise EngineError(error_msg) from step_error
                    else:
                        # Continue training, but skip this step
                        print(
                            f"WARNING: Continuing training despite error at step {run_state.step}"
                        )
                        current_step += 1
                        continue

        except KeyboardInterrupt:
            self._log_scalar(
                "training/interrupted", float(run_state.step), run_state.step
            )
            self._save_checkpoint(run_state, tag="shutdown", force=True)
            raise

        # Final checkpoint save
        self._save_checkpoint(run_state, tag="final", force=True)

        return run_state

    def _ensure_param_spec_tree(self, params: Params) -> Optional[PyTree]:
        """Return cached parameter ``PartitionSpec`` tree, recomputing if needed."""

        if self.plan.tensor_parallel is None:
            return None

        tp_config = self.plan.tensor_parallel
        default_spec = PartitionSpec()
        self._param_spec_tree = build_param_specs(
            params,
            tp_config.rules,
            default=default_spec,
        )
        return self._param_spec_tree

    def _ensure_state_spec_tree(self, state: TrainState) -> Optional[PyTree]:
        """Return cached TrainState sharding specs derived from parameter specs."""

        if self.plan.tensor_parallel is None:
            return None

        param_spec_tree = self._ensure_param_spec_tree(state.params)
        if param_spec_tree is None:
            return None

        opt_state_spec = tree_util.tree_map(lambda _: PartitionSpec(), state.opt_state)
        rng_spec = tree_util.tree_map(lambda _: PartitionSpec(), state.rngs)
        aux_data = (state.step, getattr(state, "_optimizer", None))
        self._state_spec_tree = TrainState.tree_unflatten(
            aux_data,
            (param_spec_tree, opt_state_spec, rng_spec),
        )
        return self._state_spec_tree

    def _ensure_batch_spec_tree(self, batch: BatchData) -> Optional[PyTree]:
        """Return cached batch sharding specs derived from plan configuration."""

        if self.plan.tensor_parallel is None:
            return None

        if self.plan.data_parallel is not None:
            self._batch_spec_tree = shard_batch_specs(
                batch, self.plan.data_parallel.axis
            )
        else:
            self._batch_spec_tree = tree_util.tree_map(lambda _: PartitionSpec(), batch)

        return self._batch_spec_tree

    def _compile_pending_step_fn(self, state: TrainState, batch: BatchData) -> None:
        """Compile deferred TP step functions once runtime structures are known."""

        if self._pending_step_fn is None:
            return

        state_spec_tree = self._ensure_state_spec_tree(state)
        if state_spec_tree is None:
            raise EngineError(
                "Tensor parallel compilation requires parameter sharding specs",
                suggestion="Ensure Engine.create_state was called before the first step",
            )

        batch_spec_tree = self._ensure_batch_spec_tree(batch)

        try:
            self._compiled_step_fn = compile_step_with_plan(
                self._pending_step_fn,
                self.plan,
                self._mesh,
                donate_argnums=self._pending_compile_params.get("donate_argnums", (0,)),
                static_argnums=self._pending_compile_params.get("static_argnums", ()),
                state_spec_tree=state_spec_tree,
                batch_spec_tree=batch_spec_tree,
            )
        except Exception as exc:
            raise EngineError(
                f"Failed to compile tensor parallel step function: {exc}",
                suggestion="Verify tensor parallel rules align with mesh axes and step signature",
            ) from exc

        self._pending_step_fn = None
        self._pending_compile_params = {}

    def _get_batch_axis(self) -> str:
        """Get the batch axis name for PRNG key updates.

        Returns the appropriate axis name from the mesh for per-device
        PRNG key generation.

        Returns:
            String name of the batch axis (defaults to 'batch' or 'data')
        """
        if self._mesh is not None:
            mesh_axes = self._mesh.axis_names
            # Look for common batch axis names
            for axis_name in ["batch", "data", "dp"]:
                if axis_name in mesh_axes:
                    return axis_name
            # Fallback to first axis if no standard batch axis found
            if mesh_axes:
                return mesh_axes[0]

        # Default fallback
        return "batch"

    def _execute_step(
        self, state: TrainState, batch: BatchData
    ) -> tuple[TrainState, LogDict]:
        """Execute a single training step with microbatch accumulation support.

        Args:
            state: Current training state
            batch: Input batch data (may contain 'microbatches' for gradient accumulation)

        Returns:
            Tuple of (updated_state, metrics)
        """
        if self._compiled_step_fn is None:
            raise EngineError("No step function registered")

        # Apply precision policy to batch if needed
        processed_batch = self._apply_precision_policy(batch)

        # Check if this plan uses microbatching and if batch contains microbatches
        if (
            self.plan.has_microbatching()
            and hasattr(self.plan, "data_parallel")
            and self.plan.data_parallel
            and self.plan.data_parallel.accumulate_steps > 1
        ):

            # For DP microbatching, ensure batch has microbatches
            if "microbatches" not in processed_batch:
                raise EngineError(
                    f"DP plan requires microbatching (accumulate_steps={self.plan.data_parallel.accumulate_steps}) "
                    "but batch does not contain 'microbatches' key",
                    suggestion="Modify your dataloader to provide microbatches or set accumulate_steps=1",
                )

        # Update PRNG keys for this step with proper per-device handling
        # The axis parameter will be used inside the jitted function context
        updated_rngs = update_rngs(state.rngs, axis=self._get_batch_axis())
        state_with_updated_rngs = state.replace(rngs=updated_rngs)

        # Execute the step function with optimizer integration
        try:
            new_state, metrics = self._compiled_step_fn(
                state_with_updated_rngs, processed_batch
            )

            # Add optimizer learning rate to metrics
            current_lr = self.optimizer.get_learning_rate(new_state.step)
            metrics = dict(metrics)  # Make a copy
            metrics["learning_rate"] = current_lr

            return new_state, metrics

        except Exception as e:
            raise EngineError(
                f"Step execution failed at step {state.step}: {e}",
                suggestion="Check step function implementation and data shapes",
            ) from e

    def _log_metrics(
        self, metrics: LogDict, step: int, continue_on_error: bool = True
    ) -> None:
        """Log metrics to all registered loggers.

        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
            continue_on_error: If True, log warnings for errors but don't raise.
                             If False, re-raise logging exceptions after warning.
        """
        for logger in self.loggers:
            try:
                logger.log_dict(metrics, step)
            except Exception as e:
                # Always log the warning
                print(f"Warning: Logger failed: {e}")

                # Re-raise if continue_on_error is False
                if not continue_on_error:
                    raise EngineError(f"Logging failed: {e}") from e

    def _log_scalar(
        self, name: str, value: float, step: int, continue_on_error: bool = True
    ) -> None:
        """Log a scalar value to all registered loggers.

        Args:
            name: Metric name to log
            value: Metric value to log
            step: Current training step
            continue_on_error: If True, log warnings for errors but don't raise.
                             If False, re-raise logging exceptions after warning.
        """
        for logger in self.loggers:
            try:
                logger.log_scalar(name, value, step)
            except Exception as e:
                # Always log the warning
                print(f"Warning: Logger failed: {e}")

                # Re-raise if continue_on_error is False
                if not continue_on_error:
                    raise EngineError(f"Logging failed: {e}") from e

    def _apply_precision_policy(self, batch: BatchData) -> BatchData:
        """Apply precision policy to batch data.

        Args:
            batch: Input batch data

        Returns:
            Batch data with precision policy applied
        """
        if self.precision.dtype == jnp.float32:
            return batch  # No conversion needed

        def convert_arrays(x):
            if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(self.precision.dtype)
            return x

        return jax.tree_util.tree_map(convert_arrays, batch)

    def _cast_params_to_precision(self, params: Params) -> Params:
        """Cast parameters to the target precision dtype.

        Args:
            params: Input parameters

        Returns:
            Parameters cast to precision.param_dtype
        """
        if self.precision.param_dtype == jnp.float32:
            # No conversion needed if target is float32
            return params

        def cast_param(x):
            if isinstance(x, jax.Array) and jnp.issubdtype(x.dtype, jnp.floating):
                return x.astype(self.precision.param_dtype)
            return x

        return jax.tree_util.tree_map(cast_param, params)

    def _infer_batch_size(self, batch: BatchData) -> Optional[int]:
        """Attempt to infer the batch size from a batch PyTree."""

        leaves, _ = tree_util.tree_flatten(batch)
        for leaf in leaves:
            # Prefer JAX arrays but fall back to generic objects with shape attribute
            if isinstance(leaf, jax.Array) and leaf.shape:
                return int(leaf.shape[0])
            if hasattr(leaf, "shape") and getattr(leaf, "shape"):
                try:
                    return int(leaf.shape[0])
                except (TypeError, ValueError):
                    continue
        return None

    def _log_run_header(self, state: Optional[TrainState]) -> None:
        """Log a run header summary with environment and configuration details."""

        start_step = getattr(state, "step", 0) if state is not None else 0
        header: LogDict = {
            "run/titanax_version": _titanax_version,
            "run/jax_version": jax.__version__,
            "run/device_count": jax.device_count(),
            "run/mesh": self.mesh_spec.describe(),
            "run/plan": self.plan.describe(),
            "run/optimizer": self.optimizer.describe(),
            "run/learning_rate_start": self.optimizer.get_learning_rate(start_step),
            "run/start_step": start_step,
        }

        self._log_metrics(header, step=start_step, continue_on_error=True)

    def apply_loss_scaling(self, loss: Array) -> Array:
        """Apply loss scaling for fp16 training.

        Args:
            loss: Unscaled loss value

        Returns:
            Scaled loss for gradient computation
        """
        if self.precision.loss_scaling and self.precision.fp16:
            # Simple fixed scaling factor (more sophisticated scaling can be added later)
            scale_factor = 2**14  # Common starting point for loss scaling
            return loss * scale_factor
        return loss

    def scale_gradients(self, grads: PyTree) -> PyTree:
        """Scale gradients for fp16 training with loss scaling.

        Args:
            grads: Gradients to scale

        Returns:
            Scaled gradients
        """
        if self.precision.loss_scaling and self.precision.fp16:
            scale_factor = 2**14  # Must match loss scaling factor
            return jax.tree_util.tree_map(lambda g: g / scale_factor, grads)
        return grads

    def describe(self) -> str:
        """Return a human-readable description of the engine configuration."""
        lines = [
            "Titanax Engine Configuration:",
            f"  Mesh: {self.mesh_spec.describe()}",
            f"  Plan: {self.plan.describe()}",
            f"  Optimizer: {self.optimizer.describe()}",
            f"  Precision: {self.precision.describe()}",
            f"  Checkpoint: {'enabled' if self.checkpoint else 'disabled'}",
            f"  Loggers: {len(self.loggers)} configured",
        ]
        if self.checkpoint is not None:
            interval = (
                f"every {self.checkpoint_interval} steps"
                if self.checkpoint_interval is not None
                else "disabled"
            )
            lines.append(f"  Checkpoint interval: {interval}")
        return "\n".join(lines)
