"""Simple CNN/MLP model for MNIST classification."""

import jax
import jax.numpy as jnp
from jax import lax
from typing import Tuple, Any

from src.titanax.types import PyTree, Array


def create_cnn_model() -> Tuple[Any, PyTree]:
    """Create a simple CNN model for MNIST.

    Returns:
        (apply_fn, params): Model function and initialized parameters
    """

    # Dimension numbers for NHWC input and HWIO kernel format
    CONV_DIMS = ("NHWC", "HWIO", "NHWC")

    def cnn_model(params: PyTree, x: Array) -> Array:
        """CNN forward pass: Conv2D -> ReLU -> MaxPool -> Conv2D -> ReLU -> MaxPool -> Dense -> ReLU -> Dense

        Args:
            params: Model parameters
            x: Input batch [batch_size, 28, 28, 1]

        Returns:
            logits: [batch_size, 10]
        """
        # First conv layer: 28x28x1 -> 24x24x32 (valid conv)
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=params["conv1"]["kernel"],
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=CONV_DIMS,
        ) + params["conv1"]["bias"].reshape(1, 1, 1, -1)
        x = jax.nn.relu(x)
        # Simple 2x2 max pooling: 24x24x32 -> 12x12x32
        x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

        # Second conv layer: 12x12x32 -> 8x8x64 (valid conv)
        x = lax.conv_general_dilated(
            lhs=x,
            rhs=params["conv2"]["kernel"],
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=CONV_DIMS,
        ) + params["conv2"]["bias"].reshape(1, 1, 1, -1)
        x = jax.nn.relu(x)
        # Max pooling: 8x8x64 -> 4x4x64
        x = lax.reduce_window(x, -jnp.inf, lax.max, (1, 2, 2, 1), (1, 2, 2, 1), "VALID")

        # Flatten: 4x4x64 -> 1024
        x = x.reshape(x.shape[0], -1)

        # Dense layer: 1024 -> 128
        x = jnp.dot(x, params["dense1"]["kernel"]) + params["dense1"]["bias"]
        x = jax.nn.relu(x)

        # Output layer: 128 -> 10
        x = jnp.dot(x, params["dense2"]["kernel"]) + params["dense2"]["bias"]

        return x

    return cnn_model, None  # Parameters will be initialized separately


def create_mlp_model() -> Tuple[Any, PyTree]:
    """Create a simple MLP model for MNIST.

    Returns:
        (apply_fn, params): Model function and initialized parameters
    """

    def mlp_model(params: PyTree, x: Array) -> Array:
        """MLP forward pass: Flatten -> Dense -> ReLU -> Dense -> ReLU -> Dense

        Args:
            params: Model parameters
            x: Input batch [batch_size, 28, 28, 1]

        Returns:
            logits: [batch_size, 10]
        """
        # Flatten: [batch_size, 28, 28, 1] -> [batch_size, 784]
        x = x.reshape(x.shape[0], -1)

        # First hidden layer: 784 -> 256
        x = jnp.dot(x, params["dense1"]["kernel"]) + params["dense1"]["bias"]
        x = jax.nn.relu(x)

        # Second hidden layer: 256 -> 128
        x = jnp.dot(x, params["dense2"]["kernel"]) + params["dense2"]["bias"]
        x = jax.nn.relu(x)

        # Output layer: 128 -> 10
        x = jnp.dot(x, params["dense3"]["kernel"]) + params["dense3"]["bias"]

        return x

    return mlp_model, None  # Parameters will be initialized separately


def init_cnn_params(rng: Array) -> PyTree:
    """Initialize CNN parameters.

    Args:
        rng: JAX random key

    Returns:
        params: Initialized parameter tree
    """
    rngs = jax.random.split(rng, 6)

    # Xavier/Glorot initialization
    def xavier_init(key, shape):
        scale = jnp.sqrt(2.0 / (shape[0] + shape[-1]))
        return jax.random.normal(key, shape) * scale

    params = {
        "conv1": {
            "kernel": xavier_init(rngs[0], (5, 5, 1, 32)),  # 5x5 conv, 1->32 channels
            "bias": jnp.zeros((32,)),
        },
        "conv2": {
            "kernel": xavier_init(rngs[1], (5, 5, 32, 64)),  # 5x5 conv, 32->64 channels
            "bias": jnp.zeros((64,)),
        },
        "dense1": {
            "kernel": xavier_init(rngs[2], (4 * 4 * 64, 128)),  # 1024 -> 128
            "bias": jnp.zeros((128,)),
        },
        "dense2": {
            "kernel": xavier_init(rngs[3], (128, 10)),  # 128 -> 10
            "bias": jnp.zeros((10,)),
        },
    }

    return params


def init_mlp_params(rng: Array) -> PyTree:
    """Initialize MLP parameters.

    Args:
        rng: JAX random key

    Returns:
        params: Initialized parameter tree
    """
    rngs = jax.random.split(rng, 6)

    # Xavier/Glorot initialization
    def xavier_init(key, shape):
        scale = jnp.sqrt(2.0 / (shape[0] + shape[-1]))
        return jax.random.normal(key, shape) * scale

    params = {
        "dense1": {
            "kernel": xavier_init(rngs[0], (784, 256)),  # 28*28 -> 256
            "bias": jnp.zeros((256,)),
        },
        "dense2": {
            "kernel": xavier_init(rngs[1], (256, 128)),  # 256 -> 128
            "bias": jnp.zeros((128,)),
        },
        "dense3": {
            "kernel": xavier_init(rngs[2], (128, 10)),  # 128 -> 10
            "bias": jnp.zeros((10,)),
        },
    }

    return params


def cross_entropy_loss(logits: Array, labels: Array) -> Array:
    """Compute cross-entropy loss.

    Args:
        logits: [batch_size, num_classes]
        labels: [batch_size] integer class labels

    Returns:
        loss: scalar loss value
    """
    log_probs = jax.nn.log_softmax(logits)
    one_hot = jax.nn.one_hot(labels, num_classes=10)
    return -jnp.mean(jnp.sum(one_hot * log_probs, axis=1))


def accuracy(logits: Array, labels: Array) -> Array:
    """Compute classification accuracy.

    Args:
        logits: [batch_size, num_classes]
        labels: [batch_size] integer class labels

    Returns:
        accuracy: scalar accuracy value
    """
    predictions = jnp.argmax(logits, axis=1)
    return jnp.mean(predictions == labels)


def create_model(model_type: str = "mlp") -> Tuple[Any, Any]:
    """Create MNIST model based on type.

    Args:
        model_type: "mlp" or "cnn"

    Returns:
        (model_fn, init_fn): Model function and parameter initialization function
    """
    if model_type == "mlp":
        model_fn, _ = create_mlp_model()
        return model_fn, init_mlp_params
    elif model_type == "cnn":
        model_fn, _ = create_cnn_model()
        return model_fn, init_cnn_params
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'mlp' or 'cnn'.")
