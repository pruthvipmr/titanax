"""Tests for compile_step_with_plan helper."""

import jax
import jax.numpy as jnp

from src.titanax.exec import TrainState, step_fn
from src.titanax.exec.compile import compile_step_with_plan
from src.titanax.parallel import Plan, DP
from src.titanax.runtime import MeshSpec
from src.titanax.types import PyTree
from src.titanax.compat import PartitionSpec


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
