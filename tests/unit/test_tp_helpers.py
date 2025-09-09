"""Unit tests for tensor parallel rule helpers."""

from src.titanax.parallel.tp_helpers import (
    mlp_rules,
    attention_rules,
    embedding_rules,
    layer_norm_rules,
    transformer_block_rules,
)


class TestMLPRules:
    """Test MLP tensor parallel rules."""

    def test_mlp_rules_basic(self):
        """Test basic MLP rule generation."""
        rules = mlp_rules("mlp", "model")

        # Check that all expected patterns are present
        expected_patterns = [
            "mlp/in_proj/kernel",
            "mlp/in_proj/bias",
            "mlp/dense_in/kernel",
            "mlp/dense_in/bias",
            "mlp/out_proj/kernel",
            "mlp/out_proj/bias",
            "mlp/dense_out/kernel",
            "mlp/dense_out/bias",
        ]

        for pattern in expected_patterns:
            assert pattern in rules

        # Verify column parallel input projection (last dimension sharded)
        assert rules["mlp/in_proj/kernel"] == (None, "model")
        assert rules["mlp/in_proj/bias"] == ("model",)

        # Verify row parallel output projection (first dimension sharded)
        assert rules["mlp/out_proj/kernel"] == ("model", None)
        assert rules["mlp/out_proj/bias"] == (None,)  # Replicated

    def test_mlp_rules_custom_dimensions(self):
        """Test MLP rules with custom sharding dimensions."""
        # Test first dimension sharding for input
        rules = mlp_rules("feedforward", "tp", in_features_dim=0, out_features_dim=1)

        assert rules["feedforward/in_proj/kernel"] == ("tp", None)
        assert rules["feedforward/out_proj/kernel"] == (None, "tp")


class TestAttentionRules:
    """Test attention tensor parallel rules."""

    def test_attention_rules_basic(self):
        """Test basic attention rule generation."""
        rules = attention_rules("attention", "model")

        # Check QKV projection patterns
        expected_patterns = [
            "attention/qkv_proj/kernel",
            "attention/qkv_proj/bias",
            "attention/q_proj/kernel",
            "attention/k_proj/kernel",
            "attention/v_proj/kernel",
            "attention/q_proj/bias",
            "attention/k_proj/bias",
            "attention/v_proj/bias",
            "attention/out_proj/kernel",
            "attention/out_proj/bias",
        ]

        for pattern in expected_patterns:
            assert pattern in rules

        # Verify QKV sharding (num_heads dimension = 1)
        assert rules["attention/qkv_proj/kernel"] == (None, "model", None)
        assert rules["attention/qkv_proj/bias"] == ("model", None)

        # Verify individual Q/K/V projections (column parallel)
        assert rules["attention/q_proj/kernel"] == (None, "model")
        assert rules["attention/k_proj/kernel"] == (None, "model")
        assert rules["attention/v_proj/kernel"] == (None, "model")

        # Verify output projection (row parallel)
        assert rules["attention/out_proj/kernel"] == ("model", None, None)
        assert rules["attention/out_proj/bias"] == (None,)  # Replicated

    def test_attention_rules_custom_dimensions(self):
        """Test attention rules with custom head dimensions."""
        rules = attention_rules("attn", "tp", num_heads_dim=0, head_dim_idx=1)

        # Verify custom num_heads dimension sharding
        assert rules["attn/qkv_proj/kernel"] == ("tp", None, None)


class TestEmbeddingRules:
    """Test embedding tensor parallel rules."""

    def test_embedding_rules_vocab_parallel(self):
        """Test embedding rules with vocabulary parallelism."""
        rules = embedding_rules("embedding", "model", vocab_parallel=True)

        expected_patterns = ["embedding/kernel", "embedding/weight"]
        for pattern in expected_patterns:
            assert pattern in rules

        # Vocabulary dimension sharded (first dimension)
        assert rules["embedding/kernel"] == ("model", None)
        assert rules["embedding/weight"] == ("model", None)

    def test_embedding_rules_hidden_parallel(self):
        """Test embedding rules with hidden dimension parallelism."""
        rules = embedding_rules("embed", "tp", vocab_parallel=False)

        # Hidden dimension sharded (second dimension)
        assert rules["embed/kernel"] == (None, "tp")
        assert rules["embed/weight"] == (None, "tp")


class TestLayerNormRules:
    """Test layer normalization tensor parallel rules."""

    def test_layer_norm_rules_replicated(self):
        """Test replicated layer norm rules."""
        rules = layer_norm_rules("layer_norm")

        expected_patterns = [
            "layer_norm/scale",
            "layer_norm/bias",
            "layer_norm/weight",
            "layer_norm/gamma",
            "layer_norm/beta",
        ]

        for pattern in expected_patterns:
            assert pattern in rules
            assert rules[pattern] == (None,)  # All replicated

    def test_layer_norm_rules_sharded(self):
        """Test sharded layer norm rules."""
        rules = layer_norm_rules("ln", tp_axis="model")

        # All parameters sharded along model axis
        assert rules["ln/scale"] == ("model",)
        assert rules["ln/bias"] == ("model",)
        assert rules["ln/weight"] == ("model",)


class TestTransformerBlockRules:
    """Test transformer block tensor parallel rules."""

    def test_transformer_block_rules_basic(self):
        """Test basic transformer block rule generation."""
        rules = transformer_block_rules("transformer/layer", "model")

        # Should contain attention patterns
        assert "transformer/layer/attention/qkv_proj/kernel" in rules
        assert "transformer/layer/attention/out_proj/kernel" in rules

        # Should contain MLP patterns
        assert "transformer/layer/mlp/in_proj/kernel" in rules
        assert "transformer/layer/mlp/out_proj/kernel" in rules

        # Should contain layer norm patterns
        assert "transformer/layer/ln_1/scale" in rules
        assert "transformer/layer/ln_2/bias" in rules

    def test_transformer_block_rules_with_index(self):
        """Test transformer block rules with layer index."""
        rules = transformer_block_rules("transformer/layer", "model", layer_idx=3)

        # Check that layer index is included in patterns
        assert "transformer/layer_3/attention/qkv_proj/kernel" in rules
        assert "transformer/layer_3/mlp/in_proj/kernel" in rules
        assert "transformer/layer_3/ln_1/scale" in rules

        # Verify some key sharding patterns
        assert rules["transformer/layer_3/attention/qkv_proj/kernel"] == (
            None,
            "model",
            None,
        )
        assert rules["transformer/layer_3/mlp/in_proj/kernel"] == (None, "model")
        assert rules["transformer/layer_3/mlp/out_proj/kernel"] == ("model", None)
        assert rules["transformer/layer_3/ln_1/scale"] == (None,)  # Replicated


class TestGoldenSpecsValidation:
    """Test golden PartitionSpec generation for toy MLP."""

    def test_toy_mlp_golden_specs(self):
        """Test that toy MLP gets partitioned correctly by helpers."""
        # Toy MLP parameter structure
        toy_mlp_params = {
            "mlp/in_proj/kernel": {"shape": (128, 512)},  # Input projection
            "mlp/in_proj/bias": {"shape": (512,)},  # Input bias
            "mlp/out_proj/kernel": {"shape": (512, 128)},  # Output projection
            "mlp/out_proj/bias": {"shape": (128,)},  # Output bias
        }

        # Generate rules using helper
        rules = mlp_rules("mlp", "model")

        # Validate golden specs for each parameter
        golden_specs = {
            "mlp/in_proj/kernel": (
                None,
                "model",
            ),  # Column parallel: shard output features
            "mlp/in_proj/bias": ("model",),  # Shard bias along same dimension
            "mlp/out_proj/kernel": (
                "model",
                None,
            ),  # Row parallel: shard input features
            "mlp/out_proj/bias": (None,),  # Replicated bias (reduced in collective)
        }

        # Assert each parameter gets the correct golden spec
        for param_name, expected_spec in golden_specs.items():
            assert param_name in rules
            assert (
                rules[param_name] == expected_spec
            ), f"Parameter {param_name}: expected {expected_spec}, got {rules[param_name]}"

        # Additional validation: ensure we have rules for all toy parameters
        for param_name in toy_mlp_params.keys():
            assert param_name in rules, f"Missing rule for {param_name}"

    def test_attention_golden_specs(self):
        """Test golden specs for attention mechanism."""
        rules = attention_rules("attention", "model")

        # Key golden specs for attention
        golden_specs = {
            "attention/qkv_proj/kernel": (None, "model", None),  # Shard num_heads
            "attention/q_proj/kernel": (None, "model"),  # Column parallel
            "attention/k_proj/kernel": (None, "model"),  # Column parallel
            "attention/v_proj/kernel": (None, "model"),  # Column parallel
            "attention/out_proj/kernel": ("model", None, None),  # Row parallel
            "attention/out_proj/bias": (None,),  # Replicated
        }

        for param_name, expected_spec in golden_specs.items():
            assert (
                rules[param_name] == expected_spec
            ), f"Attention parameter {param_name}: expected {expected_spec}, got {rules[param_name]}"
