#!/usr/bin/env python3
"""Minimal Titanax Data Parallel example - runs on CPU in <30 seconds."""

from itertools import cycle

import jax
import jax.numpy as jnp

import titanax as tx


def model_apply(params, x):
    logits = x @ params["w"] + params["b"]
    return logits.squeeze(-1)


def loss_and_metrics(params, batch):
    logits = model_apply(params, batch["x"])
    labels = batch["y"]
    loss = jnp.mean(
        jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))
    )
    preds = jnp.where(logits > 0, 1.0, 0.0)
    accuracy = jnp.mean(preds == labels)
    return loss, {"loss": loss, "accuracy": accuracy}


def create_batch():
    features = jnp.array(
        [
            [-2.0, -1.0],
            [-1.0, -1.5],
            [1.0, 1.5],
            [2.0, 1.0],
        ]
    )
    labels = jnp.array([0.0, 0.0, 1.0, 1.0])
    return {"x": features, "y": labels}


def data_stream():
    batch = create_batch()
    for item in cycle([batch]):
        yield item


# Initialize model parameters
key = jax.random.PRNGKey(42)
params = {
    "w": jax.random.normal(key, (2, 1)) * 0.1,
    "b": jnp.zeros((1,)),
}

# Set up Titanax engine
mesh = tx.MeshSpec(devices=[jax.devices("cpu")[0]], axes=("data",))
plan = tx.Plan(data_parallel=tx.DP(axis="data"))
logger = tx.loggers.Basic()
engine = tx.Engine(
    mesh=mesh,
    plan=plan,
    optimizer=tx.optim.sgd(0.1),
    loggers=[logger],
)


@tx.step_fn
def train_step(state, batch):
    def loss_fn(p):
        loss_value, _ = loss_and_metrics(p, batch)
        return loss_value

    loss_value, grads = jax.value_and_grad(loss_fn)(state.params)
    state = state.apply_gradients(grads=grads)
    _, metric_values = loss_and_metrics(state.params, batch)
    metric_values["loss"] = loss_value
    return state, metric_values


if __name__ == "__main__":
    print("=== Minimal Titanax DP Example ===")
    initial_state = engine.create_state(params)
    final_state = engine.fit(train_step, data_stream(), steps=5, state=initial_state)
    print(f"Completed at step {final_state.step}")
