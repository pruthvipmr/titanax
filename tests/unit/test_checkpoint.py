"""Unit tests for Titanax checkpoint utilities and Orbax integration."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.titanax.exec.engine import TrainState
from src.titanax.exceptions import CheckpointError
from src.titanax.io.checkpoint import (
    BaseCheckpointStrategy,
    CheckpointMetadata,
    resolve_checkpoint_step,
    validate_checkpoint_compatibility,
)
from src.titanax.io.orbax_io import OrbaxCheckpoint, create_checkpoint_strategy


def make_train_state(step: int) -> TrainState:
    """Create a tiny TrainState for round-trip checkpoint tests."""
    params = {
        "linear": {
            "kernel": jnp.arange(4, dtype=jnp.float32).reshape(2, 2) + float(step)
        }
    }
    opt_state = {"momentum": jnp.ones((2, 2), dtype=jnp.float32) * step}
    rngs = {"dropout": jax.random.PRNGKey(step)}
    return TrainState(params=params, opt_state=opt_state, step=step, rngs=rngs)


def assert_state_equal(a: TrainState, b: TrainState) -> None:
    """Assert that two TrainState objects are equal."""
    assert a.step == b.step

    def _assert_close(x: Any, y: Any) -> None:
        np.testing.assert_allclose(np.asarray(x), np.asarray(y))

    jax.tree_util.tree_map(_assert_close, a.params, b.params)
    jax.tree_util.tree_map(_assert_close, a.opt_state, b.opt_state)
    jax.tree_util.tree_map(_assert_close, a.rngs, b.rngs)


class TestCheckpointMetadata:
    def test_metadata_is_immutable(self) -> None:
        metadata = CheckpointMetadata(
            step=1,
            timestamp=123.0,
            titanax_version="0.0.1",
            jax_version="0.4.30",
        )

        assert metadata.step == 1
        with pytest.raises(AttributeError):
            metadata.step = 2  # type: ignore[misc]


class TestBaseCheckpointStrategy:
    def setup_method(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.strategy = BaseCheckpointStrategy(self.temp_dir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_checkpoint_paths_and_listing(self) -> None:
        assert self.strategy.get_checkpoint_path(7).name == "step_00000007"
        for step in (3, 7, 11):
            self.strategy.get_checkpoint_path(step).mkdir(parents=True)
        assert self.strategy.list_available_steps() == [3, 7, 11]

    def test_latest_step_and_cleanup(self) -> None:
        for step in (1, 2, 3, 4):
            self.strategy.get_checkpoint_path(step).mkdir(parents=True)
        assert self.strategy.latest_step() == 4
        self.strategy.cleanup_old_checkpoints(keep_last_n=2)
        assert self.strategy.list_available_steps() == [3, 4]

    def test_latest_step_without_checkpoints_raises(self) -> None:
        with pytest.raises(CheckpointError):
            self.strategy.latest_step()


class TestCheckpointUtilities:
    def setup_method(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())
        self.strategy = BaseCheckpointStrategy(self.temp_dir)

    def teardown_method(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_resolve_specific_step(self) -> None:
        self.strategy.get_checkpoint_path(5).mkdir(parents=True)
        assert resolve_checkpoint_step(self.strategy, 5) == 5

    def test_resolve_latest(self) -> None:
        for step in (2, 4, 6):
            self.strategy.get_checkpoint_path(step).mkdir(parents=True)
        assert resolve_checkpoint_step(self.strategy, None) == 6

    def test_resolve_latest_without_checkpoints(self) -> None:
        with pytest.raises(CheckpointError):
            resolve_checkpoint_step(self.strategy, None)

    def test_resolve_requires_step_for_custom_strategy(self) -> None:
        class DummyStrategy:
            pass

        with pytest.raises(CheckpointError):
            resolve_checkpoint_step(DummyStrategy(), None)

        assert resolve_checkpoint_step(DummyStrategy(), 12) == 12

    def test_validate_checkpoint_strict(self) -> None:
        metadata = CheckpointMetadata(
            step=10,
            timestamp=1.0,
            titanax_version="0.1.0",
            jax_version="0.4.30",
            mesh_spec={"axes": ["data"]},
            plan_spec={"data_parallel": {"axis": "data"}},
        )
        validate_checkpoint_compatibility(
            metadata,
            current_mesh_spec={"axes": ["data"]},
            current_plan_spec={"data_parallel": {"axis": "data"}},
            strict=True,
        )

    def test_validate_checkpoint_strict_mismatch_raises(self) -> None:
        metadata = CheckpointMetadata(
            step=10,
            timestamp=1.0,
            titanax_version="0.1.0",
            jax_version="0.4.30",
            mesh_spec={"axes": ["data"]},
            plan_spec={"data_parallel": {"axis": "data"}},
        )
        with pytest.raises(CheckpointError):
            validate_checkpoint_compatibility(
                metadata, current_mesh_spec={"axes": ["model"]}, strict=True
            )
        with pytest.raises(CheckpointError):
            validate_checkpoint_compatibility(
                metadata,
                current_plan_spec={"tensor_parallel": {"axis": "model"}},
                strict=True,
            )


class TestOrbaxCheckpoint:
    def setup_method(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_invalid_keep_n_raises(self) -> None:
        with pytest.raises(CheckpointError):
            OrbaxCheckpoint(self.temp_dir, keep_n=0)

    def test_save_requires_step_attribute(self) -> None:
        strategy = OrbaxCheckpoint(self.temp_dir)
        with pytest.raises(CheckpointError):
            strategy.save({"params": {}})

    def test_round_trip_preserves_state(self) -> None:
        strategy = OrbaxCheckpoint(self.temp_dir)
        state = make_train_state(step=3)
        strategy.save(state)
        restored = strategy.restore()
        assert isinstance(restored, TrainState)
        assert_state_equal(state, restored)

    def test_latest_step_returns_int(self) -> None:
        strategy = OrbaxCheckpoint(self.temp_dir)
        for step in (1, 2, 3):
            strategy.save(make_train_state(step))
        assert strategy.latest_step() == 3

    def test_retention_policy_keeps_recent_checkpoints(self) -> None:
        strategy = OrbaxCheckpoint(self.temp_dir, keep_n=2)
        for step in (1, 2, 3):
            strategy.save(make_train_state(step))
        assert strategy.list_available_steps() == [2, 3]
        # Directory for step 1 should be gone
        assert not (self.temp_dir / "step_00000001").exists()


class TestCheckpointFactory:
    def setup_method(self) -> None:
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self) -> None:
        shutil.rmtree(self.temp_dir)

    def test_create_orbax_strategy(self) -> None:
        strategy = create_checkpoint_strategy(self.temp_dir, strategy="orbax")
        assert isinstance(strategy, OrbaxCheckpoint)

    def test_create_unsupported_strategy(self) -> None:
        with pytest.raises(CheckpointError):
            create_checkpoint_strategy(self.temp_dir, strategy="unsupported")
