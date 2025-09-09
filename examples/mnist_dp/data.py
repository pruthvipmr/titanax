"""MNIST data loading and preprocessing."""

import numpy as np
import jax.numpy as jnp
from typing import Tuple, Iterator, Dict
import gzip
import os
from urllib.request import urlretrieve

from src.titanax.types import Array


def download_mnist(data_dir: str = "data/mnist") -> None:
    """Download MNIST dataset if not already present.

    Args:
        data_dir: Directory to store MNIST data
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]

    for filename in files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urlretrieve(base_url + filename, filepath)


def _read_mnist_images(filepath: str) -> np.ndarray:
    """Read MNIST images from gzipped file.

    Args:
        filepath: Path to MNIST images file

    Returns:
        images: [num_images, 28, 28, 1] array
    """
    with gzip.open(filepath, "rb") as f:
        # Skip magic number and dimensions
        f.read(16)
        # Read all image data
        buf = f.read()
        data = np.frombuffer(buf, dtype=np.uint8)
        # Reshape to [num_images, 28, 28, 1]
        return data.reshape(-1, 28, 28, 1).astype(np.float32)


def _read_mnist_labels(filepath: str) -> np.ndarray:
    """Read MNIST labels from gzipped file.

    Args:
        filepath: Path to MNIST labels file

    Returns:
        labels: [num_labels] array
    """
    with gzip.open(filepath, "rb") as f:
        # Skip magic number and count
        f.read(8)
        # Read all label data
        buf = f.read()
        return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(
    data_dir: str = "data/mnist", normalize: bool = True
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]:
    """Load MNIST dataset.

    Args:
        data_dir: Directory containing MNIST data
        normalize: Whether to normalize pixel values to [0, 1]

    Returns:
        ((train_images, train_labels), (test_images, test_labels))
    """
    download_mnist(data_dir)

    # Load training data
    train_images = _read_mnist_images(
        os.path.join(data_dir, "train-images-idx3-ubyte.gz")
    )
    train_labels = _read_mnist_labels(
        os.path.join(data_dir, "train-labels-idx1-ubyte.gz")
    )

    # Load test data
    test_images = _read_mnist_images(
        os.path.join(data_dir, "t10k-images-idx3-ubyte.gz")
    )
    test_labels = _read_mnist_labels(
        os.path.join(data_dir, "t10k-labels-idx1-ubyte.gz")
    )

    if normalize:
        train_images = train_images / 255.0
        test_images = test_images / 255.0

    return (train_images, train_labels), (test_images, test_labels)


class MNISTDataLoader:
    """MNIST data loader with support for data parallel sharding."""

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray,
        batch_size: int,
        shuffle: bool = True,
        drop_last: bool = True,
        data_parallel_rank: int = 0,
        data_parallel_size: int = 1,
    ):
        """Initialize MNIST data loader.

        Args:
            images: Images array [num_samples, 28, 28, 1]
            labels: Labels array [num_samples]
            batch_size: Batch size per device
            shuffle: Whether to shuffle data each epoch
            drop_last: Whether to drop incomplete final batch
            data_parallel_rank: Rank in data parallel group (0-indexed)
            data_parallel_size: Total number of data parallel processes
        """
        self.images = images
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.dp_rank = data_parallel_rank
        self.dp_size = data_parallel_size

        # Shard data for data parallelism
        self._shard_data()

    def _shard_data(self) -> None:
        """Shard data across data parallel processes."""
        num_samples = len(self.images)

        if num_samples % self.dp_size != 0:
            if self.drop_last:
                # Drop samples to make evenly divisible
                keep_samples = (num_samples // self.dp_size) * self.dp_size
                self.images = self.images[:keep_samples]
                self.labels = self.labels[:keep_samples]
                num_samples = keep_samples
            else:
                print(
                    f"Warning: {num_samples} samples not evenly divisible by "
                    f"dp_size={self.dp_size}. Some processes may have different batch counts."
                )

        # Calculate shard boundaries
        samples_per_shard = num_samples // self.dp_size
        start_idx = self.dp_rank * samples_per_shard
        end_idx = start_idx + samples_per_shard

        # Extract shard
        self.images = self.images[start_idx:end_idx]
        self.labels = self.labels[start_idx:end_idx]

    def __len__(self) -> int:
        """Number of batches per epoch for this shard."""
        num_samples = len(self.images)
        if self.drop_last:
            return num_samples // self.batch_size
        else:
            return (num_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, Array]]:
        """Iterate over batches."""
        num_samples = len(self.images)
        indices = np.arange(num_samples)

        if self.shuffle:
            np.random.shuffle(indices)

        num_batches = len(self)

        for i in range(num_batches):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_samples)

            batch_indices = indices[start_idx:end_idx]
            batch_images = self.images[batch_indices]
            batch_labels = self.labels[batch_indices]

            # Convert to JAX arrays
            yield {"x": jnp.array(batch_images), "y": jnp.array(batch_labels)}


def create_data_loaders(
    batch_size: int,
    data_dir: str = "data/mnist",
    data_parallel_rank: int = 0,
    data_parallel_size: int = 1,
    shuffle_train: bool = True,
    normalize: bool = True,
) -> Tuple[MNISTDataLoader, MNISTDataLoader]:
    """Create MNIST train and test data loaders.

    Args:
        batch_size: Batch size per device
        data_dir: Directory containing MNIST data
        data_parallel_rank: Rank in data parallel group
        data_parallel_size: Total data parallel processes
        shuffle_train: Whether to shuffle training data
        normalize: Whether to normalize pixel values

    Returns:
        (train_loader, test_loader)
    """
    (train_images, train_labels), (test_images, test_labels) = load_mnist(
        data_dir=data_dir, normalize=normalize
    )

    train_loader = MNISTDataLoader(
        images=train_images,
        labels=train_labels,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=True,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
    )

    test_loader = MNISTDataLoader(
        images=test_images,
        labels=test_labels,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        data_parallel_rank=data_parallel_rank,
        data_parallel_size=data_parallel_size,
    )

    return train_loader, test_loader


def get_sample_batch(batch_size: int = 32) -> Dict[str, Array]:
    """Get a sample batch for testing purposes.

    Args:
        batch_size: Batch size

    Returns:
        Sample batch dict with 'x' and 'y' keys
    """
    # Generate random sample data
    x = jnp.ones((batch_size, 28, 28, 1)) * 0.5  # Dummy images
    y = jnp.zeros(batch_size, dtype=jnp.int32)  # All zeros

    return {"x": x, "y": y}
