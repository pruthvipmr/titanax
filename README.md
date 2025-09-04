# Titanax

**Explicit-Parallel JAX Training Framework**

Titanax is a lightweight training framework that brings the user ergonomics of Hugging Face Accelerate / TorchTitan to **JAX**, while requiring **explicit** data/tensor/pipeline parallelization (no XLA auto-sharding). Users declare meshes, sharding rules, and collectives; Titanax wires the training loop, checkpointing, and observability around those choices.

## Features

- **Explicit Parallelism**: Users must declare how arrays and computations are partitioned (DP/TP/PP)
- **Ergonomic API**: Run the same training script on 1 GPU, multi-GPU, or multi-host with minimal code changes
- **Production Ready**: Built-in logging, checkpointing, mixed precision, and reproducible randomness
- **Composable**: Parallel plans (DP/TP/PP) can be composed and validated before compilation
