#!/usr/bin/env python3
"""Minimal Titanax tensor-parallel MLP example.

This script demonstrates a 1D model-parallel MLP that shards the hidden
representation across a ``model`` mesh axis. It runs a tiny regression task
and verifies that activations and gradients carry the expected sharding.
"""

from __future__ import annotations

import functools
from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp

import titanax as tx
from titanax.compat import NamedSharding, PartitionSpec, pjit


def _tree_map_with_path(tree: Any, fn, prefix: Tuple[str, ...] = ()):
    if isinstance(tree, dict):
        return {
            key: _tree_map_with_path(value, fn, prefix + (key,))
            for key, value in tree.items()
        }
    return fn("/".join(prefix), tree)


def _build_partition_specs(params: Dict[str, Any], rules: Dict[str, Tuple[Any, ...]]):
    if PartitionSpec is None:
        raise SystemExit("PartitionSpec is not available in this JAX version.")

    def to_spec(path: str, _: Any):
        rule = rules.get(path)
        if rule is None:
            return PartitionSpec()
        return PartitionSpec(*rule)

    return _tree_map_with_path(params, to_spec)


def _shard_tree(tree: Dict[str, Any], specs: Dict[str, Any], mesh) -> Dict[str, Any]:
    if NamedSharding is None:
        raise SystemExit("NamedSharding is not available in this JAX version.")

    return jax.tree_util.tree_map(
        lambda value, spec: jax.device_put(value, NamedSharding(mesh, spec)),
        tree,
        specs,
    )


def main() -> None:
    key = jax.random.PRNGKey(0)
    input_dim = 4
    hidden_dim = 16
    output_dim = 1
    batch_size = 8
    learning_rate = 0.5
    steps = 5

    mesh_spec = tx.MeshSpec(devices="all", axes=("model",))
    mesh = mesh_spec.build()
    model_axis_size = mesh.shape["model"]
    if model_axis_size < 2:
        raise SystemExit(
            "Tensor parallel example requires at least 2 devices. "
            "Set XLA_FLAGS=--xla_force_host_platform_device_count=2 when running on CPU."
        )

    if hidden_dim % model_axis_size != 0:
        hidden_dim = model_axis_size * (
            (hidden_dim + model_axis_size - 1) // model_axis_size
        )

    x_key, y_key, w1_key, w2_key = jax.random.split(key, 4)
    features = jax.random.normal(x_key, (batch_size, input_dim))
    targets = jnp.sin(jax.random.normal(y_key, (batch_size, output_dim)))

    params = {
        "mlp": {
            "in_proj": {
                "kernel": jax.random.normal(w1_key, (input_dim, hidden_dim)) * 0.1,
                "bias": jnp.zeros((hidden_dim,)),
            },
            "out_proj": {
                "kernel": jax.random.normal(w2_key, (hidden_dim, output_dim)) * 0.1,
                "bias": jnp.zeros((output_dim,)),
            },
        }
    }

    tp_rules = tx.tp_helpers.mlp_rules("mlp", "model")
    plan = tx.Plan(tensor_parallel=tx.TP(axis="model", rules=tp_rules))
    plan.validate(mesh_spec)
    print(f"Plan: {plan.describe()}")

    param_specs = _build_partition_specs(params, tp_rules)
    sharded_params = _shard_tree(params, param_specs, mesh)

    batch_specs = {
        "x": PartitionSpec(),
        "y": PartitionSpec(),
    }
    sharded_batch = _shard_tree({"x": features, "y": targets}, batch_specs, mesh)

    @functools.partial(
        pjit,
        in_shardings=(param_specs, batch_specs),
        out_shardings=(
            param_specs,
            None,
            param_specs,
            {"logits": PartitionSpec(), "hidden": PartitionSpec(None, "model")},
        ),
    )
    def train_step(params_tree, batch):
        with tx.collectives.mesh_context(mesh):

            def model_apply(p, inputs):
                w1 = p["mlp"]["in_proj"]["kernel"]
                b1 = p["mlp"]["in_proj"]["bias"]
                w2 = p["mlp"]["out_proj"]["kernel"]
                b2 = p["mlp"]["out_proj"]["bias"]

                hidden = jnp.dot(inputs, w1) + b1
                hidden = jax.nn.relu(hidden)
                partial_out = jnp.dot(hidden, w2)
                logits = tx.collectives.psum(partial_out, axis="model") + b2
                return logits, hidden

            def loss_and_aux(p):
                logits, hidden = model_apply(p, batch["x"])
                loss = jnp.mean((logits - batch["y"]) ** 2)
                return loss, {"logits": logits, "hidden": hidden}

            (loss, aux), grads = jax.value_and_grad(loss_and_aux, has_aux=True)(
                params_tree
            )
            grads = jax.tree_util.tree_map(
                lambda g: tx.collectives.psum(g, axis="model"), grads
            )
            new_params = jax.tree_util.tree_map(
                lambda p, g: p - learning_rate * g, params_tree, grads
            )
            return new_params, loss, grads, aux

    losses = []
    first_aux = None
    first_grads = None

    with mesh:
        for step in range(steps):
            sharded_params, loss, grads, aux = train_step(sharded_params, sharded_batch)
            loss_value = float(jax.device_get(loss))
            losses.append(loss_value)
            print(f"Step {step + 1}: loss={loss_value:.6f}")

            if step == 0:
                first_aux = aux
                first_grads = grads

    assert first_aux is not None and first_grads is not None

    hidden = first_aux["hidden"]
    assert hidden.shape == (batch_size, hidden_dim)
    hidden_spec = getattr(hidden, "sharding", None)
    if hidden_spec is not None:
        assert hidden_spec.spec == PartitionSpec(None, "model")

    kernel_grad = first_grads["mlp"]["in_proj"]["kernel"]
    assert kernel_grad.shape == (input_dim, hidden_dim)
    kernel_spec = getattr(kernel_grad, "sharding", None)
    if kernel_spec is not None:
        assert kernel_spec.spec == PartitionSpec(None, "model")

    out_kernel_grad = first_grads["mlp"]["out_proj"]["kernel"]
    assert out_kernel_grad.shape == (hidden_dim, output_dim)
    out_kernel_spec = getattr(out_kernel_grad, "sharding", None)
    if out_kernel_spec is not None:
        assert out_kernel_spec.spec == PartitionSpec("model", None)

    assert losses[-1] < losses[0], "Loss did not decrease across training steps."
    print("Tensor parallel example completed successfully!")


if __name__ == "__main__":
    main()
