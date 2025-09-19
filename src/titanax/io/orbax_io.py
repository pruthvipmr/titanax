"""Orbax-based checkpoint strategy for Titanax."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import jax
from orbax.checkpoint import PyTreeCheckpointer  # type: ignore[import]

from ..exceptions import CheckpointError
from ..types import PyTree
from .checkpoint import (
    BaseCheckpointStrategy,
    CheckpointMetadata,
    resolve_checkpoint_step,
)

try:
    import titanax

    TITANAX_VERSION = getattr(titanax, "__version__", "dev")
except ImportError:  # pragma: no cover - defensive guard for early bootstrapping
    TITANAX_VERSION = "dev"


class OrbaxCheckpoint(BaseCheckpointStrategy):
    """Persist Titanax `TrainState` objects using Orbax."""

    def __init__(self, checkpoint_dir: str | Path, keep_n: int = 3) -> None:
        super().__init__(checkpoint_dir)
        if keep_n <= 0:
            raise CheckpointError(
                "keep_n must be a positive integer",
                suggestion="Pass keep_n >= 1 to retain at least one checkpoint",
            )
        self.keep_n = keep_n
        try:
            self._checkpointer = PyTreeCheckpointer()
        except Exception as exc:  # pragma: no cover - falls back to suggestion
            raise CheckpointError(
                f"Failed to initialize Orbax checkpointer: {exc}",
                suggestion="Install orbax-checkpoint and ensure it matches the JAX version",
            ) from exc

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------
    def save(self, state: PyTree) -> None:
        """Save a `TrainState`-compatible PyTree.

        The current step is derived from the state's `step` attribute and used
        to generate a monotonic directory name (e.g., `step_00000042`).
        """
        step = getattr(state, "step", None)
        if step is None:
            raise CheckpointError(
                "State object is missing a 'step' attribute",
                suggestion="Ensure the Engine passes a TrainState with a numeric step",
            )
        if not isinstance(step, int):
            raise CheckpointError(
                "State.step must be an integer",
                suggestion="Cast the step to int before saving",
            )

        checkpoint_path = self.get_checkpoint_path(step)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        state_dir = checkpoint_path / "state"
        try:
            metadata = self._build_metadata(step)
            self._checkpointer.save(state_dir.as_posix(), state, force=True)
            self._write_metadata(checkpoint_path, metadata)
            self.cleanup_old_checkpoints(self.keep_n)
        except Exception as exc:
            raise CheckpointError(
                f"Failed to save checkpoint for step {step}: {exc}",
                suggestion="Verify filesystem permissions and available disk space",
            ) from exc

    def restore(self, step: int | None = None) -> PyTree:
        """Restore a checkpoint, defaulting to the latest available step."""
        resolved_step = resolve_checkpoint_step(self, step)
        state_dir = self.get_checkpoint_path(resolved_step) / "state"
        try:
            state = self._checkpointer.restore(state_dir.as_posix())
        except Exception as exc:
            available = self.list_available_steps()
            raise CheckpointError(
                f"Failed to restore checkpoint for step {resolved_step}: {exc}",
                suggestion=(
                    f"Available steps: {available}"
                    if available
                    else "No checkpoints saved yet"
                ),
            ) from exc

        metadata = self._read_metadata(self.get_checkpoint_path(resolved_step))

        if not hasattr(state, "step"):
            # The checkpointer may return the flattened children when the original
            # dataclass type is not reconstructed automatically. Rehydrate using
            # the registered PyTree helpers.
            try:
                from ..exec.engine import TrainState  # Local import to avoid cycles

                step_value = metadata.step if metadata is not None else resolved_step
                state = TrainState.tree_unflatten((step_value, None), tuple(state))
            except Exception as exc:
                raise CheckpointError(
                    f"Failed to reconstruct TrainState: {exc}",
                    suggestion="Ensure the checkpoint was created with a compatible Titanax version",
                ) from exc

        if getattr(state, "step", resolved_step) != resolved_step:
            # Keep metadata consistent for downstream logging.
            replace = getattr(state, "replace", None)
            if callable(replace):
                state = replace(step=resolved_step)
        return state

    def latest_step(self) -> int:
        """Return the most recent checkpoint step."""
        return super().latest_step()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _build_metadata(self, step: int) -> CheckpointMetadata:
        return CheckpointMetadata(
            step=step,
            timestamp=time.time(),
            titanax_version=TITANAX_VERSION,
            jax_version=jax.__version__,
            mesh_spec=None,
            plan_spec=None,
            extra={"keep_n": self.keep_n},
        )

    def _write_metadata(self, path: Path, metadata: CheckpointMetadata) -> None:
        metadata_path = path / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as fh:
            json.dump(self._metadata_to_dict(metadata), fh, indent=2)

    def _read_metadata(self, path: Path) -> CheckpointMetadata | None:
        metadata_path = path / "metadata.json"
        if not metadata_path.exists():
            return None
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                data: Dict[str, Any] = json.load(fh)
            return CheckpointMetadata(**data)
        except Exception:
            return None

    @staticmethod
    def _metadata_to_dict(metadata: CheckpointMetadata) -> Dict[str, Any]:
        metadata_dict = asdict(metadata)
        # Drop None fields for brevity.
        return {k: v for k, v in metadata_dict.items() if v is not None}


def create_checkpoint_strategy(
    checkpoint_dir: str | Path, strategy: str = "orbax", **kwargs: Any
) -> BaseCheckpointStrategy:
    """Factory for checkpoint strategies."""
    if strategy.lower() == "orbax":
        return OrbaxCheckpoint(checkpoint_dir, **kwargs)
    raise CheckpointError(
        f"Unsupported checkpoint strategy: {strategy}",
        suggestion="Use 'orbax' for Orbax-based checkpoints",
    )
