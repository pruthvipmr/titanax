"""Tensor Parallel rule helpers for common model architectures.

This module provides convenience functions to generate common PartitionSpec
patterns for tensor parallel sharding of typical neural network layers.
"""

from typing import Dict, Optional, Tuple, Union, List

from ..types import AxisName


def mlp_rules(
    layer_prefix: str,
    tp_axis: AxisName,
    in_features_dim: int = -1,
    out_features_dim: int = 0,
) -> Dict[str, Tuple[Union[str, None], ...]]:
    """Generate TP rules for MLP layers.

    Args:
        layer_prefix: Prefix pattern for MLP layers (e.g., "mlp", "feedforward")
        tp_axis: Name of the tensor parallel axis
        in_features_dim: Which dimension to shard for input projections (-1 for last dimension)
        out_features_dim: Which dimension to shard for output projections (0 for first dimension)

    Returns:
        Dictionary of parameter path patterns to PartitionSpec tuples

    Example:
        >>> rules = mlp_rules("mlp", "model")
        >>> # Results in:
        >>> # {
        >>> #     "mlp/in_proj/kernel": (None, "model"),  # Column parallel
        >>> #     "mlp/in_proj/bias": ("model",),         # Shard bias
        >>> #     "mlp/out_proj/kernel": ("model", None), # Row parallel
        >>> #     "mlp/out_proj/bias": (None,)            # Replicated bias
        >>> # }
    """
    rules: Dict[str, Tuple[Union[str, None], ...]] = {}

    # Input/hidden projection (column parallel)
    rules[f"{layer_prefix}/in_proj/kernel"] = (
        (None, tp_axis) if in_features_dim == -1 else (tp_axis, None)
    )
    rules[f"{layer_prefix}/in_proj/bias"] = (tp_axis,)

    # Alternative naming patterns
    rules[f"{layer_prefix}/dense_in/kernel"] = (
        (None, tp_axis) if in_features_dim == -1 else (tp_axis, None)
    )
    rules[f"{layer_prefix}/dense_in/bias"] = (tp_axis,)

    # Output projection (row parallel)
    rules[f"{layer_prefix}/out_proj/kernel"] = (
        (tp_axis, None) if out_features_dim == 0 else (None, tp_axis)
    )
    rules[f"{layer_prefix}/out_proj/bias"] = (None,)  # Replicated

    # Alternative naming patterns
    rules[f"{layer_prefix}/dense_out/kernel"] = (
        (tp_axis, None) if out_features_dim == 0 else (None, tp_axis)
    )
    rules[f"{layer_prefix}/dense_out/bias"] = (None,)

    return rules


def attention_rules(
    layer_prefix: str, tp_axis: AxisName, num_heads_dim: int = 1, head_dim_idx: int = 2
) -> Dict[str, Tuple[Union[str, None], ...]]:
    """Generate TP rules for multi-head attention layers.

    Args:
        layer_prefix: Prefix pattern for attention layers (e.g., "attention", "attn")
        tp_axis: Name of the tensor parallel axis
        num_heads_dim: Which dimension contains the number of heads (1 for typical)
        head_dim_idx: Which dimension contains the head dimension (2 for typical)

    Returns:
        Dictionary of parameter path patterns to PartitionSpec tuples

    Example:
        >>> rules = attention_rules("attention", "model")
        >>> # Results in:
        >>> # {
        >>> #     "attention/qkv_proj/kernel": (None, "model", None),  # Shard num_heads
        >>> #     "attention/qkv_proj/bias": ("model", None),          # Shard heads
        >>> #     "attention/out_proj/kernel": ("model", None, None),  # Row parallel
        >>> #     "attention/out_proj/bias": (None,)                   # Replicated
        >>> # }
    """
    rules: Dict[str, Tuple[Union[str, None], ...]] = {}

    # QKV projections - shard along num_heads dimension
    qkv_spec: List[Union[str, None]] = [None, None, None]
    qkv_spec[num_heads_dim] = tp_axis
    rules[f"{layer_prefix}/qkv_proj/kernel"] = tuple(qkv_spec)

    # QKV bias - shard along heads dimension
    qkv_bias_spec: List[Union[str, None]] = [None, None]
    qkv_bias_spec[0] = tp_axis  # Assuming bias shape is [num_heads, head_dim]
    rules[f"{layer_prefix}/qkv_proj/bias"] = tuple(qkv_bias_spec)

    # Alternative individual Q, K, V projections
    q_spec: List[Union[str, None]] = [None, None]
    k_spec: List[Union[str, None]] = [None, None]
    v_spec: List[Union[str, None]] = [None, None]
    q_spec[1] = k_spec[1] = v_spec[1] = tp_axis  # Column parallel
    rules[f"{layer_prefix}/q_proj/kernel"] = tuple(q_spec)
    rules[f"{layer_prefix}/k_proj/kernel"] = tuple(k_spec)
    rules[f"{layer_prefix}/v_proj/kernel"] = tuple(v_spec)
    rules[f"{layer_prefix}/q_proj/bias"] = (tp_axis,)
    rules[f"{layer_prefix}/k_proj/bias"] = (tp_axis,)
    rules[f"{layer_prefix}/v_proj/bias"] = (tp_axis,)

    # Output projection (row parallel)
    out_spec: List[Union[str, None]] = [None, None, None]
    out_spec[0] = tp_axis  # Reduce along concatenated heads dimension
    rules[f"{layer_prefix}/out_proj/kernel"] = tuple(out_spec)
    rules[f"{layer_prefix}/out_proj/bias"] = (None,)  # Replicated

    return rules


def embedding_rules(
    layer_prefix: str, tp_axis: AxisName, vocab_parallel: bool = True
) -> Dict[str, Tuple[Union[str, None], ...]]:
    """Generate TP rules for embedding layers.

    Args:
        layer_prefix: Prefix pattern for embedding layers (e.g., "embedding", "embed")
        tp_axis: Name of the tensor parallel axis
        vocab_parallel: If True, shard vocabulary dimension; if False, shard hidden dimension

    Returns:
        Dictionary of parameter path patterns to PartitionSpec tuples

    Example:
        >>> rules = embedding_rules("embedding", "model", vocab_parallel=True)
        >>> # Results in:
        >>> # {
        >>> #     "embedding/kernel": ("model", None),  # Shard vocab dimension
        >>> # }
    """
    rules: Dict[str, Tuple[Union[str, None], ...]] = {}

    if vocab_parallel:
        # Shard along vocabulary dimension (first dimension typically)
        rules[f"{layer_prefix}/kernel"] = (tp_axis, None)
        rules[f"{layer_prefix}/weight"] = (tp_axis, None)  # Alternative naming
    else:
        # Shard along hidden dimension (second dimension typically)
        rules[f"{layer_prefix}/kernel"] = (None, tp_axis)
        rules[f"{layer_prefix}/weight"] = (None, tp_axis)

    return rules


def layer_norm_rules(
    layer_prefix: str, tp_axis: Optional[AxisName] = None
) -> Dict[str, Tuple[Union[str, None], ...]]:
    """Generate TP rules for layer normalization.

    Args:
        layer_prefix: Prefix pattern for layer norm (e.g., "layer_norm", "ln")
        tp_axis: Name of tensor parallel axis, or None for replicated parameters

    Returns:
        Dictionary of parameter path patterns to PartitionSpec tuples

    Note:
        Layer norm parameters are typically replicated across tensor parallel ranks
        unless the normalized dimension is sharded.
    """
    rules: Dict[str, Tuple[Union[str, None], ...]] = {}

    if tp_axis is None:
        # Replicated layer norm parameters (most common)
        rules[f"{layer_prefix}/scale"] = (None,)
        rules[f"{layer_prefix}/bias"] = (None,)
        rules[f"{layer_prefix}/weight"] = (None,)  # Alternative naming
        rules[f"{layer_prefix}/gamma"] = (None,)  # Alternative naming
        rules[f"{layer_prefix}/beta"] = (None,)  # Alternative naming
    else:
        # Sharded layer norm (rare, when normalizing sharded dimension)
        rules[f"{layer_prefix}/scale"] = (tp_axis,)
        rules[f"{layer_prefix}/bias"] = (tp_axis,)
        rules[f"{layer_prefix}/weight"] = (tp_axis,)
        rules[f"{layer_prefix}/gamma"] = (tp_axis,)
        rules[f"{layer_prefix}/beta"] = (tp_axis,)

    return rules


def transformer_block_rules(
    block_prefix: str, tp_axis: AxisName, layer_idx: Optional[int] = None
) -> Dict[str, Tuple[Union[str, None], ...]]:
    """Generate comprehensive TP rules for a transformer block.

    Args:
        block_prefix: Prefix for the transformer block (e.g., "transformer/layer")
        tp_axis: Name of the tensor parallel axis
        layer_idx: Optional layer index to include in patterns

    Returns:
        Dictionary combining attention, MLP, and layer norm rules

    Example:
        >>> rules = transformer_block_rules("transformer/layer", "model", layer_idx=0)
        >>> # Combines attention_rules, mlp_rules, and layer_norm_rules with
        >>> # prefix "transformer/layer_0/"
    """
    if layer_idx is not None:
        full_prefix = f"{block_prefix}_{layer_idx}"
    else:
        full_prefix = block_prefix

    rules: Dict[str, Tuple[Union[str, None], ...]] = {}

    # Multi-head attention rules
    attn_rules = attention_rules(f"{full_prefix}/attention", tp_axis)
    rules.update(attn_rules)

    # MLP/feedforward rules
    mlp_rules_dict = mlp_rules(f"{full_prefix}/mlp", tp_axis)
    rules.update(mlp_rules_dict)

    # Layer norm rules (typically replicated)
    ln1_rules = layer_norm_rules(f"{full_prefix}/ln_1")
    ln2_rules = layer_norm_rules(f"{full_prefix}/ln_2")
    rules.update(ln1_rules)
    rules.update(ln2_rules)

    # Alternative layer norm naming patterns
    pre_ln_rules = layer_norm_rules(f"{full_prefix}/pre_attention_layer_norm")
    post_ln_rules = layer_norm_rules(f"{full_prefix}/pre_mlp_layer_norm")
    rules.update(pre_ln_rules)
    rules.update(post_ln_rules)

    return rules
