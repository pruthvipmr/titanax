"""Tests for the exception hierarchy and error handling."""

from src.titanax.exceptions import (
    TitanaxError,
    ValidationError,
    MeshError,
    PlanError,
    ShardingError,
    CollectiveError,
    CheckpointError,
    DataError,
    EngineError,
    CompilationError,
    DistributedError,
    OptimizerError,
    mesh_validation_error,
    plan_validation_error,
    collective_error,
    sharding_error,
)


class TestTitanaxError:
    """Test the base TitanaxError class."""

    def test_basic_error(self):
        """Test basic error without suggestion."""
        error = TitanaxError("Something went wrong")
        assert str(error) == "Something went wrong"
        assert error.message == "Something went wrong"
        assert error.suggestion is None

    def test_error_with_suggestion(self):
        """Test error with suggestion."""
        error = TitanaxError("Something went wrong", "Try this fix")
        expected = "Something went wrong\n\nSuggestion: Try this fix"
        assert str(error) == expected
        assert error.message == "Something went wrong"
        assert error.suggestion == "Try this fix"

    def test_error_inheritance(self):
        """Test that TitanaxError is a proper Exception."""
        error = TitanaxError("test")
        assert isinstance(error, Exception)


class TestExceptionHierarchy:
    """Test the exception hierarchy structure."""

    def test_validation_error_inheritance(self):
        """Test ValidationError inherits from TitanaxError."""
        error = ValidationError("validation failed", "check inputs")
        assert isinstance(error, TitanaxError)
        assert isinstance(error, Exception)

    def test_mesh_error_inheritance(self):
        """Test MeshError inherits from ValidationError."""
        error = MeshError("mesh invalid", "fix mesh")
        assert isinstance(error, ValidationError)
        assert isinstance(error, TitanaxError)

    def test_plan_error_inheritance(self):
        """Test PlanError inherits from ValidationError."""
        error = PlanError("plan invalid", "fix plan")
        assert isinstance(error, ValidationError)
        assert isinstance(error, TitanaxError)

    def test_all_errors_inherit_from_titanax_error(self):
        """Test all custom errors inherit from TitanaxError."""
        errors = [
            ShardingError("test"),
            CollectiveError("test"),
            CheckpointError("test"),
            DataError("test"),
            EngineError("test"),
            CompilationError("test"),
            DistributedError("test"),
            OptimizerError("test"),
        ]

        for error in errors:
            assert isinstance(error, TitanaxError)
            assert isinstance(error, Exception)


class TestConvenienceFunctions:
    """Test the convenience functions for creating exceptions."""

    def test_mesh_validation_error(self):
        """Test mesh_validation_error function."""
        error = mesh_validation_error("axis conflict", "use different axes")

        assert isinstance(error, MeshError)
        assert "Mesh validation failed: axis conflict" in str(error)
        assert "Suggestion: use different axes" in str(error)
        assert error.suggestion == "use different axes"

    def test_mesh_validation_error_no_suggestion(self):
        """Test mesh_validation_error without suggestion."""
        error = mesh_validation_error("axis conflict")

        assert isinstance(error, MeshError)
        assert str(error) == "Mesh validation failed: axis conflict"
        assert error.suggestion is None

    def test_plan_validation_error(self):
        """Test plan_validation_error function."""
        error = plan_validation_error("invalid DP config", "check axis name")

        assert isinstance(error, PlanError)
        assert "Plan validation failed: invalid DP config" in str(error)
        assert "Suggestion: check axis name" in str(error)
        assert error.suggestion == "check axis name"

    def test_plan_validation_error_no_suggestion(self):
        """Test plan_validation_error without suggestion."""
        error = plan_validation_error("invalid DP config")

        assert isinstance(error, PlanError)
        assert str(error) == "Plan validation failed: invalid DP config"
        assert error.suggestion is None

    def test_collective_error(self):
        """Test collective_error function."""
        error = collective_error("psum", "data", "timeout occurred")

        assert isinstance(error, CollectiveError)
        assert (
            "Collective operation 'psum' on axis 'data' failed: timeout occurred"
            in str(error)
        )
        assert "Check that axis 'data' exists in the current mesh" in str(error)
        assert "expected size" in error.suggestion

    def test_sharding_error(self):
        """Test sharding_error function."""
        error = sharding_error("params.dense.weight", "incompatible shape")

        assert isinstance(error, ShardingError)
        assert (
            "Parameter 'params.dense.weight' sharding failed: incompatible shape"
            in str(error)
        )
        assert "Check the sharding rules for 'params.dense.weight'" in str(error)
        assert "ensure they match the parameter shape" in error.suggestion

    def test_sharding_error_custom_suggestion(self):
        """Test sharding_error with custom suggestion."""
        error = sharding_error(
            "params.dense.weight",
            "incompatible shape",
            "Use a different partitioning strategy",
        )

        assert isinstance(error, ShardingError)
        assert (
            "Parameter 'params.dense.weight' sharding failed: incompatible shape"
            in str(error)
        )
        assert "Suggestion: Use a different partitioning strategy" in str(error)
        assert error.suggestion == "Use a different partitioning strategy"


class TestErrorMessages:
    """Test that error messages are helpful and informative."""

    def test_error_message_formatting(self):
        """Test that error messages are properly formatted."""
        error = TitanaxError("Base error message", "This is the suggestion")

        lines = str(error).split("\n")
        assert lines[0] == "Base error message"
        assert lines[1] == ""
        assert lines[2] == "Suggestion: This is the suggestion"

    def test_collective_error_context(self):
        """Test that collective errors provide good context."""
        error = collective_error("all_gather", "model", "device mismatch")

        error_str = str(error)
        # Should mention the operation name
        assert "all_gather" in error_str
        # Should mention the axis
        assert "model" in error_str
        # Should mention the specific failure
        assert "device mismatch" in error_str
        # Should provide actionable suggestion
        assert "Check that axis 'model' exists" in error_str

    def test_sharding_error_context(self):
        """Test that sharding errors provide parameter context."""
        error = sharding_error("model.layers.0.attention.query", "shape mismatch")

        error_str = str(error)
        # Should mention the specific parameter
        assert "model.layers.0.attention.query" in error_str
        # Should mention the failure reason
        assert "shape mismatch" in error_str
        # Should provide actionable suggestion
        assert "sharding rules" in error_str
        assert "parameter shape" in error_str


class TestSuggestionFormatting:
    """Test that suggestions are properly formatted and helpful."""

    def test_suggestion_not_duplicated(self):
        """Test that suggestions aren't duplicated in nested exceptions."""
        base_error = TitanaxError("base message", "base suggestion")

        # The suggestion should only appear once in the string representation
        error_str = str(base_error)
        suggestion_count = error_str.count("Suggestion:")
        assert suggestion_count == 1

    def test_empty_suggestion_handling(self):
        """Test that empty suggestions are handled correctly."""
        error = TitanaxError("message", "")

        # Empty suggestion should be preserved as empty string
        assert error.suggestion == ""
        # But empty suggestion is falsy, so no "Suggestion:" should appear
        assert "Suggestion:" not in str(error)
        assert str(error) == "message"

    def test_none_suggestion_handling(self):
        """Test that None suggestions don't appear in output."""
        error = TitanaxError("message", None)

        assert error.suggestion is None
        assert "Suggestion:" not in str(error)
        assert str(error) == "message"
