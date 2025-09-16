"""Compilation helpers for Titanax step functions.

This module centralizes compilation logic so we can preferentially use ``pjit``
when explicit shardings are provided while still supporting ``shard_map`` for
pure data-parallel map style execution.
"""

from __future__ import annotations

import functools
from typing import Any, Optional, Callable, cast

import jax

from ..compat import PartitionSpec, shard_map, pjit
from ..exceptions import CompilationError
from ..parallel.plan import Plan
from ..types import StepFunction


def _get_partition_spec_ctor() -> Callable[..., Any]:
    """Return the PartitionSpec constructor, raising if unavailable."""

    if PartitionSpec is None:
        raise CompilationError(
            "PartitionSpec is unavailable in this JAX version",
            "Upgrade to a JAX release with sharding support or avoid compiling with sharding.",
        )
    return cast(Callable[..., Any], PartitionSpec)


def _get_validated_function(step_fn: StepFunction) -> StepFunction:
    """Return the validation-aware body stored on the decorated step function."""

    if hasattr(step_fn, "_validated_fn"):
        return getattr(step_fn, "_validated_fn")  # type: ignore[return-value]
    if hasattr(step_fn, "_original_fn"):
        return getattr(step_fn, "_original_fn")  # type: ignore[return-value]
    return step_fn


def _infer_default_specs(plan: Plan, num_args: int) -> tuple[Any, ...]:
    """Infer default input shard specs based on the plan configuration."""

    spec_ctor = _get_partition_spec_ctor()
    state_spec = spec_ctor()
    batch_spec = spec_ctor()
    if plan.data_parallel is not None:
        batch_spec = spec_ctor(plan.data_parallel.axis)

    specs = [state_spec, batch_spec]
    # Any additional args (e.g. static configs) default to replicated specs.
    specs.extend(spec_ctor() for _ in range(max(0, num_args - 2)))
    return tuple(specs)


def compile_step_with_plan(
    step_fn: StepFunction,
    plan: Plan,
    mesh: Any,
    *,
    in_shardings: Optional[Any] = None,
    out_shardings: Optional[Any] = None,
    donate_argnums: tuple[int, ...] = (0,),
    static_argnums: tuple[int, ...] = (),
) -> StepFunction:
    """Compile ``step_fn`` using ``plan`` information and the provided mesh.

    When sharding information is provided we prefer ``pjit`` so explicit
    ``PartitionSpec`` configurations are honoured. Otherwise we fall back to a
    ``shard_map``-wrapped ``jax.jit`` which keeps the ergonomics of map-style
    collectives while still running under the correct mesh context.
    """

    body = _get_validated_function(step_fn)
    parameter_names = getattr(step_fn, "_parameter_names", None)
    num_args = len(parameter_names) if parameter_names is not None else 2

    mesh_devices = getattr(mesh, "devices", None)
    mesh_size = getattr(mesh_devices, "size", None)

    if mesh_size == 1:
        @functools.wraps(body)
        def single_device_fn(*args: Any, **kwargs: Any):
            with mesh:
                return body(*args, **kwargs)

        return jax.jit(
            single_device_fn,
            donate_argnums=donate_argnums,
            static_argnums=static_argnums,
        )

    if in_shardings is not None or out_shardings is not None:
        if in_shardings is None or out_shardings is None:
            raise CompilationError(
                "compile_step_with_plan requires both in_shardings and out_shardings when using pjit",
                "Pass both sharding arguments or omit them to use shard_map fallback.",
            )

        pjit_fn = cast(Optional[Callable[..., Any]], pjit)
        if pjit_fn is None:
            raise CompilationError(
                "pjit is unavailable in this JAX version",
                "Upgrade to a JAX release that provides jax.pjit to use sharded execution.",
            )

        try:
            compiled = pjit_fn(
                body,
                in_shardings=in_shardings,
                out_shardings=out_shardings,
                donate_argnums=donate_argnums,
                static_argnums=static_argnums,
            )
        except Exception as exc:
            raise CompilationError(
                f"pjit compilation failed: {exc}",
                "Verify that your sharding specs match the Plan and mesh axes.",
            ) from exc

        @functools.wraps(body)
        def run_with_mesh(*args: Any, **kwargs: Any):
            with mesh:
                return compiled(*args, **kwargs)

        return run_with_mesh

    in_specs = _infer_default_specs(plan, num_args)

    spec_ctor = _get_partition_spec_ctor()
    state_out_spec = spec_ctor()

    shard_map_fn = cast(Optional[Callable[..., Any]], shard_map)
    if shard_map_fn is None:
        raise CompilationError(
            "shard_map is unavailable in this JAX version",
            "Upgrade to a JAX release with shard_map support or provide explicit sharding specs for pjit.",
        )

    def shard_mapped(*args: Any, **kwargs: Any):
        with mesh:
            mapped = shard_map_fn(
                body,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=(state_out_spec, spec_ctor()),
            )
            return mapped(*args, **kwargs)

    try:
        return jax.jit(
            shard_mapped,
            donate_argnums=donate_argnums,
            static_argnums=static_argnums,
        )
    except Exception as exc:  # pragma: no cover - defensive
        raise CompilationError(
            f"jax.jit compilation failed: {exc}",
            "Ensure the step function arguments align with the mesh configuration.",
        ) from exc
