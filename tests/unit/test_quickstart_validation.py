"""Tests for quickstart fail-fast validation."""

import pytest
from unittest.mock import MagicMock

from src.titanax.quickstart import (
    simple_data_parallel,
    simple_tensor_parallel,
    _validate_data_parallel_config,
)
from src.titanax.exceptions import ValidationError


class TestDataParallelConfigValidation:
    """Test validation of data parallel configuration parameters."""

    def test_batch_size_must_be_integer(self):
        """Test that batch_size must be an integer."""
        with pytest.raises(ValidationError, match="batch_size must be an integer"):
            _validate_data_parallel_config(
                batch_size="128",  # string instead of int
                learning_rate=3e-4,
                precision="bf16",
            )

        with pytest.raises(ValidationError, match="batch_size must be an integer"):
            _validate_data_parallel_config(
                batch_size=128.0,  # float instead of int
                learning_rate=3e-4,
                precision="bf16",
            )

    def test_batch_size_must_be_positive(self):
        """Test that batch_size must be positive."""
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            _validate_data_parallel_config(
                batch_size=0,
                learning_rate=3e-4,
                precision="bf16",
            )

        with pytest.raises(ValidationError, match="batch_size must be positive"):
            _validate_data_parallel_config(
                batch_size=-128,
                learning_rate=3e-4,
                precision="bf16",
            )

    def test_learning_rate_must_be_numeric(self):
        """Test that learning_rate must be numeric."""
        with pytest.raises(ValidationError, match="learning_rate must be numeric"):
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate="3e-4",  # string instead of float
                precision="bf16",
            )

    def test_learning_rate_must_be_positive(self):
        """Test that learning_rate must be positive."""
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate=0.0,
                precision="bf16",
            )

        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate=-0.001,
                precision="bf16",
            )

    def test_precision_must_be_string(self):
        """Test that precision must be a string."""
        with pytest.raises(ValidationError, match="precision must be a string"):
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate=3e-4,
                precision=16,  # int instead of string
            )

    def test_precision_must_be_valid(self):
        """Test that precision must be one of the valid values."""
        with pytest.raises(ValidationError, match="Invalid precision 'invalid'"):
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate=3e-4,
                precision="invalid",
            )

        # Should contain suggestions
        with pytest.raises(ValidationError) as exc_info:
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate=3e-4,
                precision="float16",  # wrong name
            )

        error = exc_info.value
        assert "bf16" in error.suggestion
        assert "fp16" in error.suggestion
        assert "fp32" in error.suggestion

    def test_loggers_must_be_list(self):
        """Test that loggers must be a list if provided."""
        mock_logger = MagicMock()
        mock_logger.log = MagicMock()

        with pytest.raises(ValidationError, match="loggers must be a list"):
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate=3e-4,
                precision="bf16",
                loggers=mock_logger,  # single logger instead of list
            )

    def test_loggers_must_have_log_method(self):
        """Test that loggers must implement the Logger protocol."""
        bad_logger = MagicMock()
        del bad_logger.log  # Remove log method

        with pytest.raises(ValidationError, match="does not have a 'log' method"):
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate=3e-4,
                precision="bf16",
                loggers=[bad_logger],
            )

    def test_valid_config_passes(self):
        """Test that valid configuration passes validation."""
        mock_logger = MagicMock()
        mock_logger.log = MagicMock()

        # Should not raise any exceptions
        _validate_data_parallel_config(
            batch_size=128,
            learning_rate=3e-4,
            precision="bf16",
            loggers=[mock_logger],
        )

        # Should also work with no loggers
        _validate_data_parallel_config(
            batch_size=64,
            learning_rate=1e-3,
            precision="fp32",
            loggers=None,
        )


class TestSimpleDataParallelValidation:
    """Test fail-fast validation in simple_data_parallel function."""

    def test_invalid_batch_size_fails_fast(self):
        """Test that invalid batch_size fails before mesh creation."""
        with pytest.raises(ValidationError, match="batch_size must be positive"):
            simple_data_parallel(batch_size=0)

    def test_invalid_precision_fails_fast(self):
        """Test that invalid precision fails before mesh creation."""
        with pytest.raises(ValidationError, match="Invalid precision"):
            simple_data_parallel(batch_size=128, precision="invalid_precision")

    def test_invalid_learning_rate_fails_fast(self):
        """Test that invalid learning_rate fails before mesh creation."""
        with pytest.raises(ValidationError, match="learning_rate must be positive"):
            simple_data_parallel(batch_size=128, learning_rate=-0.001)

    def test_error_messages_include_suggestions(self):
        """Test that error messages include helpful suggestions."""
        with pytest.raises(ValidationError) as exc_info:
            simple_data_parallel(batch_size=128, precision="float16")

        error = exc_info.value
        assert error.suggestion is not None
        assert "bf16" in error.suggestion or "fp16" in error.suggestion

        with pytest.raises(ValidationError) as exc_info:
            simple_data_parallel(batch_size=0)

        error = exc_info.value
        assert error.suggestion is not None
        assert "positive batch size" in error.suggestion


class TestSimpleTensorParallelStub:
    """Test the tensor parallel stub function."""

    def test_raises_not_implemented(self):
        """Test that simple_tensor_parallel raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="not yet implemented"):
            simple_tensor_parallel(
                batch_size=128, model_parallel_size=2, sharding_rules={}
            )

    def test_error_includes_helpful_guidance(self):
        """Test that the NotImplementedError includes helpful guidance."""
        with pytest.raises(NotImplementedError) as exc_info:
            simple_tensor_parallel(
                batch_size=128, model_parallel_size=2, sharding_rules={}
            )

        error_str = str(exc_info.value)
        # Should mention the TP example
        assert "examples/tp_minimal_mlp.py" in error_str
        # Should mention data parallel alternative
        assert "simple_data_parallel" in error_str
        # Should mention phase P1
        assert "phase P1" in error_str


class TestErrorSuggestionQuality:
    """Test that error suggestions are helpful and actionable."""

    def test_batch_size_suggestions_are_specific(self):
        """Test that batch size error suggestions are specific."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_data_parallel_config(
                batch_size=0, learning_rate=3e-4, precision="bf16"
            )

        suggestion = exc_info.value.suggestion
        assert "32" in suggestion or "64" in suggestion or "128" in suggestion

    def test_learning_rate_suggestions_are_specific(self):
        """Test that learning rate error suggestions are specific."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_data_parallel_config(
                batch_size=128, learning_rate=0, precision="bf16"
            )

        suggestion = exc_info.value.suggestion
        assert "1e-4" in suggestion or "3e-4" in suggestion or "1e-3" in suggestion

    def test_precision_suggestions_list_all_options(self):
        """Test that precision error suggestions list all valid options."""
        with pytest.raises(ValidationError) as exc_info:
            _validate_data_parallel_config(
                batch_size=128, learning_rate=3e-4, precision="invalid"
            )

        suggestion = exc_info.value.suggestion
        assert "fp32" in suggestion
        assert "bf16" in suggestion
        assert "fp16" in suggestion

    def test_logger_suggestions_mention_protocol(self):
        """Test that logger error suggestions mention the Logger protocol."""
        bad_logger = MagicMock()
        del bad_logger.log

        with pytest.raises(ValidationError) as exc_info:
            _validate_data_parallel_config(
                batch_size=128,
                learning_rate=3e-4,
                precision="bf16",
                loggers=[bad_logger],
            )

        suggestion = exc_info.value.suggestion
        assert "Logger protocol" in suggestion
        assert "log() method" in suggestion
