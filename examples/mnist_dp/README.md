# MNIST Data Parallel Example

This example demonstrates how to train a simple MNIST classifier using Titanax's data parallel capabilities.

## Files

- `model.py`: Contains CNN and MLP model definitions, parameter initialization, and utility functions
- `data.py`: MNIST data loading, preprocessing, and data parallel sharding
- `train.py`: Main training script with evaluation loop
- `README.md`: This file

## Usage

### Single Device Training
```bash
# From the project root
cd /Users/pruthvirajaghatta/Documents/Projects/titanax
uv run python examples/mnist_dp/train.py
```

### Multi-Device Training (if available)
```bash
# Use all available devices
uv run python examples/mnist_dp/train.py --devices=all

# Use specific number of devices
uv run python examples/mnist_dp/train.py --devices=2
```

### Training Options
```bash
# Use CNN instead of MLP
uv run python examples/mnist_dp/train.py --model=cnn

# Change batch size and learning rate
uv run python examples/mnist_dp/train.py --batch-size=256 --learning-rate=1e-4

# Use mixed precision training
uv run python examples/mnist_dp/train.py --precision=bfloat16

# Enable checkpointing
uv run python examples/mnist_dp/train.py --checkpoint-dir=checkpoints/mnist_run

# Longer training
uv run python examples/mnist_dp/train.py --steps=5000 --eval-every=500
```

## Expected Results

The model should achieve:
- **MLP**: ~98% accuracy on MNIST test set
- **CNN**: ~99% accuracy on MNIST test set

Training typically takes 1-5 minutes depending on the model and hardware.

## Features Demonstrated

- **Data Parallel Training**: Gradients are aggregated across devices using `tx.collectives.psum`
- **Sharded Data Loading**: Training data is automatically sharded across data parallel processes
- **Engine Integration**: Uses Titanax's `Engine.create_state()` and step function decoration
- **Mixed Precision**: Support for bfloat16 and fp16 training
- **Checkpointing**: Optional checkpoint saving and loading
- **Evaluation**: Periodic evaluation with metrics aggregation
- **Logging**: Training and evaluation metrics logging

## Architecture

### MLP Model
- Input: 784 (28×28 flattened)
- Hidden 1: 256 units (ReLU)
- Hidden 2: 128 units (ReLU)
- Output: 10 classes

### CNN Model
- Conv1: 5×5, 1→32 channels, ReLU, 2×2 MaxPool
- Conv2: 5×5, 32→64 channels, ReLU, 2×2 MaxPool
- Dense1: 3136→128, ReLU
- Dense2: 128→10

## Implementation Notes

- Uses explicit gradient aggregation with `tx.collectives.psum`
- Metrics are averaged across devices with `tx.collectives.pmean`
- Data is sharded at the dataset level for perfect load balancing
- MNIST data is automatically downloaded on first run
- Supports both single and multi-device training with the same code
