# Titanax Examples

This directory contains example training scripts demonstrating different parallelism strategies:

- `mnist_dp/` - MNIST training with Data Parallel
- `gpt_small_tp/` - GPT-small with Tensor Parallel
- `gpt_small_pp/` - GPT-small with Pipeline Parallel
- `configs/` - YAML configuration examples
- `minimal_dp.py` - Minimal CPU-only data parallel training loop
- `tp_minimal_mlp.py` - Minimal tensor-parallel MLP with sharded hidden states
- `pp_minimal_two_stage.py` - Minimal two-stage pipeline with 1F1B scheduling

Each example includes:
- Model definition
- Data loading
- Training script
- Configuration files

Run examples with:
```bash
uv run python examples/{example_name}/train.py
```
