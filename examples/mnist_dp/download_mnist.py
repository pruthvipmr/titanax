#!/usr/bin/env python3
"""Download MNIST dataset for local use."""

import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from data import download_mnist, load_mnist


def main():
    data_dir = "data/mnist"
    print(f"Downloading MNIST dataset to {data_dir}...")

    # Download the data
    download_mnist(data_dir)

    print("Download complete! Loading data to verify...")

    # Test loading the data
    try:
        (train_images, train_labels), (test_images, test_labels) = load_mnist(data_dir)
        print(
            f"Training data: {train_images.shape} images, {train_labels.shape} labels"
        )
        print(f"Test data: {test_images.shape} images, {test_labels.shape} labels")
        print(f"Label range: {train_labels.min()} to {train_labels.max()}")
        print(
            f"Image value range: {train_images.min():.3f} to {train_images.max():.3f}"
        )
        print("MNIST dataset ready for use!")

    except Exception as e:
        print(f"Error loading data: {e}")
        return False

    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
