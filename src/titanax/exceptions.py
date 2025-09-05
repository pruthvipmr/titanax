"""Base exception classes for Titanax.

This module defines the exception hierarchy used throughout the Titanax framework.
All Titanax-specific exceptions inherit from TitanaxError.
"""


class TitanaxError(Exception):
    """Base exception class for all Titanax errors."""
    
    def __init__(self, message: str, suggestion: str | None = None):
        """Initialize with error message and optional suggestion.
        
        Args:
            message: The error message
            suggestion: Optional suggestion for fixing the error
        """
        self.message = message
        self.suggestion = suggestion
        
        full_message = message
        if suggestion:
            full_message += f"\n\nSuggestion: {suggestion}"
            
        super().__init__(full_message)


class ValidationError(TitanaxError):
    """Raised when validation of configuration or parameters fails."""
    pass


class MeshError(ValidationError):
    """Raised when mesh configuration is invalid."""
    pass


class PlanError(ValidationError):
    """Raised when parallel plan configuration is invalid."""
    pass


class ShardingError(TitanaxError):
    """Raised when sharding operations fail."""
    pass


class CollectiveError(TitanaxError):
    """Raised when collective operations fail."""
    pass


class CheckpointError(TitanaxError):
    """Raised when checkpoint operations fail."""
    pass


class DataError(TitanaxError):
    """Raised when data loading or processing fails."""
    pass


class EngineError(TitanaxError):
    """Raised when engine operations fail."""
    pass


class CompilationError(TitanaxError):
    """Raised when JAX compilation fails."""
    pass


class DistributedError(TitanaxError):
    """Raised when distributed initialization or coordination fails."""
    pass


# Convenience functions for common error patterns

def mesh_validation_error(message: str, suggestion: str | None = None) -> MeshError:
    """Create a MeshError with standardized messaging."""
    return MeshError(f"Mesh validation failed: {message}", suggestion)


def plan_validation_error(message: str, suggestion: str | None = None) -> PlanError:
    """Create a PlanError with standardized messaging."""
    return PlanError(f"Plan validation failed: {message}", suggestion)


def collective_error(operation: str, axis: str, message: str) -> CollectiveError:
    """Create a CollectiveError with operation and axis context."""
    full_message = f"Collective operation '{operation}' on axis '{axis}' failed: {message}"
    suggestion = f"Check that axis '{axis}' exists in the current mesh and has the expected size"
    return CollectiveError(full_message, suggestion)


def sharding_error(param_path: str, message: str, suggestion: str | None = None) -> ShardingError:
    """Create a ShardingError with parameter context."""
    full_message = f"Parameter '{param_path}' sharding failed: {message}"
    if suggestion is None:
        suggestion = f"Check the sharding rules for '{param_path}' and ensure they match the parameter shape"
    return ShardingError(full_message, suggestion)
