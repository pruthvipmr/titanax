# Titanax Examples

This directory contains example training scripts demonstrating different parallelism strategies:

- `mnist_dp/` - MNIST training with Data Parallel
- `gpt_small_tp/` - GPT-small with Tensor Parallel
- `gpt_small_pp/` - GPT-small with Pipeline Parallel
- `configs/` - YAML configuration examples

Each example includes:
- Model definition
- Data loading
- Training script
- Configuration files

Run examples with:
```bash
uv run python examples/{example_name}/train.py
```
