#!/usr/bin/env python3
"""Minimal Titanax pipeline-parallel example with two stages.

This script sets up a tiny encoder/decoder pipeline using Titanax ``Stage``
constructs and executes a simple 1F1B (one-forward/one-backward) schedule across
microbatches. It verifies that loss decreases over a few training steps while
passing activations only at pipeline boundaries.
"""

from __future__ import annotations

from typing import Dict, Tuple

import jax
import jax.numpy as jnp

import titanax as tx

ArrayTree = Dict[str, jnp.ndarray]


def mse_loss(logits: jnp.ndarray, targets: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((logits - targets) ** 2)


def tree_zeros_like(tree: ArrayTree) -> ArrayTree:
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_add(a: ArrayTree, b: ArrayTree) -> ArrayTree:
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_average(tree: ArrayTree, divisor: float) -> ArrayTree:
    return jax.tree_util.tree_map(lambda x: x / divisor, tree)


def tree_update(params: ArrayTree, grads: ArrayTree, lr: float) -> ArrayTree:
    return jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)


def main() -> None:
    key = jax.random.PRNGKey(0)
    input_dim = 4
    hidden_dim = 8
    output_dim = 1
    batch_size = 8
    microbatch_size = 2
    learning_rate = 0.3
    steps = 4

    if batch_size % microbatch_size != 0:
        raise SystemExit("batch_size must be divisible by microbatch_size")

    num_micro = batch_size // microbatch_size

    data_key, target_key, s0_key, s1_key = jax.random.split(key, 4)
    features = jax.random.normal(data_key, (batch_size, input_dim))
    target_w = jax.random.normal(target_key, (input_dim, output_dim))
    targets = jnp.tanh(features @ target_w)

    stage0_params: ArrayTree = {
        "kernel": jax.random.normal(s0_key, (input_dim, hidden_dim)) * 0.1,
        "bias": jnp.zeros((hidden_dim,)),
    }
    stage1_params: ArrayTree = {
        "kernel": jax.random.normal(s1_key, (hidden_dim, output_dim)) * 0.1,
        "bias": jnp.zeros((output_dim,)),
    }

    def stage0_forward(
        inputs: jnp.ndarray, training: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        z = jnp.dot(inputs, stage0_params["kernel"]) + stage0_params["bias"]
        return jax.nn.relu(z), {"inputs": inputs, "pre": z}

    def stage0_backward(grad_outputs: jnp.ndarray, activations: Dict[str, jnp.ndarray]):
        inputs = activations["inputs"]

        def apply(p, x):
            return jax.nn.relu(jnp.dot(x, p["kernel"]) + p["bias"])

        _, pullback = jax.vjp(apply, stage0_params, inputs)
        grad_params, grad_inputs = pullback(grad_outputs)
        return grad_inputs, grad_params

    def stage1_forward(
        inputs: jnp.ndarray, training: bool = True
    ) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
        logits = jnp.dot(inputs, stage1_params["kernel"]) + stage1_params["bias"]
        return logits, {"inputs": inputs}

    def stage1_backward(grad_outputs: jnp.ndarray, activations: Dict[str, jnp.ndarray]):
        inputs = activations["inputs"]

        def apply(p, x):
            return jnp.dot(x, p["kernel"]) + p["bias"]

        _, pullback = jax.vjp(apply, stage1_params, inputs)
        grad_params, grad_inputs = pullback(grad_outputs)
        return grad_inputs, grad_params

    stage0 = tx.Stage(
        stage0_forward, backward_fn=stage0_backward, stage_id=0, stage_name="encoder"
    )
    stage1 = tx.Stage(
        stage1_forward, backward_fn=stage1_backward, stage_id=1, stage_name="decoder"
    )
    stages = [stage0, stage1]

    schedule = tx.create_1f1b_schedule(
        num_stages=len(stages),
        microbatch_size=microbatch_size,
        global_batch_size=batch_size,
    )
    schedule.validate_with_pipeline(stages, microbatch_size)

    print(f"Schedule: {schedule.describe()}")
    print("Stages:")
    for stage in stages:
        print(f"  - {stage.describe()}")

    micro_features = jnp.split(features, num_micro, axis=0)
    micro_targets = jnp.split(targets, num_micro, axis=0)

    losses = []

    def pipeline_step() -> float:
        nonlocal stage0_params, stage1_params

        stage0_out: Dict[int, jnp.ndarray] = {}
        stage0_act: Dict[int, Dict[str, jnp.ndarray]] = {}
        stage1_act: Dict[int, Dict[str, jnp.ndarray]] = {}
        stage1_out: Dict[int, jnp.ndarray] = {}
        stage1_input_grads: Dict[int, jnp.ndarray] = {}

        s0_grads = tree_zeros_like(stage0_params)
        s1_grads = tree_zeros_like(stage1_params)
        total_loss = 0.0

        fwd0 = fwd1 = bwd1 = bwd0 = 0
        ticks = num_micro + len(stages) - 1 + num_micro

        for _ in range(ticks):
            if fwd0 < num_micro:
                out0, act0 = stage0.forward(micro_features[fwd0], training=True)
                stage0_out[fwd0] = out0
                stage0_act[fwd0] = act0
                fwd0 += 1

            if fwd1 < num_micro and fwd1 in stage0_out:
                out1, act1 = stage1.forward(stage0_out.pop(fwd1), training=True)
                stage1_out[fwd1] = out1
                stage1_act[fwd1] = act1
                fwd1 += 1

            if bwd1 < num_micro and bwd1 in stage1_out:
                logits = stage1_out.pop(bwd1)
                target = micro_targets[bwd1]
                loss_value = mse_loss(logits, target)
                total_loss += float(loss_value)
                loss_grad = (2.0 / microbatch_size) * (logits - target)
                grad_to_stage0, grad_params1 = stage1.backward(
                    loss_grad, stage1_act.pop(bwd1)
                )
                s1_grads = tree_add(s1_grads, grad_params1)
                stage1_input_grads[bwd1] = grad_to_stage0
                bwd1 += 1

            if bwd0 < num_micro and bwd0 in stage1_input_grads:
                grad_from_stage1 = stage1_input_grads.pop(bwd0)
                _, grad_params0 = stage0.backward(
                    grad_from_stage1, stage0_act.pop(bwd0)
                )
                s0_grads = tree_add(s0_grads, grad_params0)
                bwd0 += 1

            if bwd0 == num_micro:
                break

        avg_loss = total_loss / num_micro
        s0_grads_avg = tree_average(s0_grads, num_micro)
        s1_grads_avg = tree_average(s1_grads, num_micro)
        stage0_params = tree_update(stage0_params, s0_grads_avg, learning_rate)
        stage1_params = tree_update(stage1_params, s1_grads_avg, learning_rate)
        return avg_loss

    for step in range(steps):
        loss_value = pipeline_step()
        losses.append(loss_value)
        print(f"Global step {step + 1}: loss={loss_value:.6f}")

    assert losses[-1] < losses[0], "Pipeline loss did not decrease."
    print("Pipeline parallel example completed successfully!")


if __name__ == "__main__":
    main()
