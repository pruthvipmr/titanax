"""Tests for compile_step_with_plan helper."""

import pytest
import jax
import jax.numpy as jnp

from src.titanax.exec import TrainState, step_fn
from src.titanax.exec.compile import compile_step_with_plan
from src.titanax.parallel import Plan, DP, TP
from src.titanax.runtime import MeshSpec
from src.titanax.types import PyTree
from src.titanax.compat import PartitionSpec
from src.titanax.parallel.sharding import build_param_specs


def _build_state() -> TrainState:
    params = {"w": jnp.array([1.0, 2.0])}
    opt_state: PyTree = {"m": jnp.zeros_like(params["w"])}
    return TrainState(
        params=params,
        opt_state=opt_state,
        step=0,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )


def _build_mesh():
    spec = MeshSpec(devices="all", axes=("data",))
    return spec.build()


def test_compile_step_with_plan_pjit():
    """Compilation uses pjit's sharding when provided."""
    mesh = _build_mesh()
    plan = Plan(data_parallel=DP(axis="data"))

    @step_fn()
    def simple_step(state, batch):
        return state.replace(step=state.step + 1), {"loss": jnp.array(1.0)}

    compiled = compile_step_with_plan(
        simple_step,
        plan,
        mesh,
        in_shardings=(PartitionSpec(), PartitionSpec("data")),
        out_shardings=(PartitionSpec(), None),
    )

    state = _build_state()
    batch = {"x": jnp.ones((1,))}
    new_state, metrics = compiled(state, batch)

    assert new_state.step == state.step + 1
    assert "loss" in metrics


def test_compile_step_with_plan_shard_map_fallback():
    """Compilation falls back to shard_map when no shardings provided."""
    mesh = _build_mesh()
    plan = Plan(data_parallel=DP(axis="data"))

    @step_fn()
    def simple_step(state, batch):
        return state.replace(step=state.step + 2), {"loss": jnp.array(2.0)}

    compiled = compile_step_with_plan(simple_step, plan, mesh)

    state = _build_state()
    batch = {"x": jnp.ones((1,))}
    new_state, metrics = compiled(state, batch)

    assert new_state.step == state.step + 2
    assert "loss" in metrics


class _DummyDevices:
    size = 2


class _DummyMesh:
    devices = _DummyDevices()
    axis_names = ("data", "model")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_compile_step_with_plan_auto_tp_shardings(monkeypatch: pytest.MonkeyPatch):
    """Tensor-parallel spec trees auto-populate pjit's shardings."""

    mesh = _DummyMesh()
    plan = Plan(
        tensor_parallel=TP(axis="model", rules={"linear/kernel": ("model", None)})
    )

    @step_fn()
    def simple_step(state, batch):
        return state.replace(step=state.step + 1), {"loss": jnp.array(3.0)}

    state = TrainState(
        params={"linear": {"kernel": jnp.ones((2, 2))}},
        opt_state={"linear": {"momentum": jnp.zeros((2, 2))}},
        step=0,
        rngs={"dropout": jax.random.PRNGKey(0)},
    )

    param_spec_tree = build_param_specs(
        state.params,
        plan.tensor_parallel.rules,
        default=PartitionSpec(),
    )
    opt_spec = jax.tree_util.tree_map(lambda _: PartitionSpec(), state.opt_state)
    rng_spec = jax.tree_util.tree_map(lambda _: PartitionSpec(), state.rngs)
    state_spec_tree = TrainState.tree_unflatten(
        (state.step, getattr(state, "_optimizer", None)),
        (param_spec_tree, opt_spec, rng_spec),
    )
    batch_spec_tree = {"x": PartitionSpec(), "y": PartitionSpec()}

    call_record: dict[str, PyTree] = {}

    def fake_pjit(fn, *, in_shardings, out_shardings, donate_argnums, static_argnums):
        call_record["in"] = in_shardings
        call_record["out"] = out_shardings
        call_record["donate"] = donate_argnums
        call_record["static"] = static_argnums

        def wrapped(*args, **kwargs):
            return fn(*args, **kwargs)

        return wrapped

    monkeypatch.setattr("src.titanax.exec.compile.pjit", fake_pjit)

    compiled = compile_step_with_plan(
        simple_step,
        plan,
        mesh,
        state_spec_tree=state_spec_tree,
        batch_spec_tree=batch_spec_tree,
    )

    assert call_record["in"][0] is state_spec_tree
    assert call_record["in"][1] is batch_spec_tree
    assert call_record["out"] == (state_spec_tree, None)

    batch = {"x": jnp.ones((2, 2)), "y": jnp.ones((2, 2))}
    new_state, metrics = compiled(state, batch)

    assert new_state.step == state.step + 1
    assert float(metrics["loss"]) == 3.0
