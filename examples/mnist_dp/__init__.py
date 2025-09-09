"""MNIST Data Parallel Example.

This package contains a complete example of training a simple MNIST classifier
using Titanax's data parallel capabilities.

Modules:
    model: CNN/MLP model definitions and utilities
    data: MNIST data loading and preprocessing
    train: Main training script
"""

from .model import create_model, cross_entropy_loss, accuracy
from .data import create_data_loaders, get_sample_batch

__all__ = [
    "create_model",
    "cross_entropy_loss",
    "accuracy",
    "create_data_loaders",
    "get_sample_batch",
]
