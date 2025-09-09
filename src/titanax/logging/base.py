"""Base logging utilities for Titanax.

This module provides the foundation for logging throughout the Titanax framework,
including abstract base classes and common utilities.
"""

import time
from abc import ABC, abstractmethod
from typing import List
from ..types import Logger, LogValue, LogDict


class BaseLogger(ABC):
    """Abstract base class for all Titanax loggers."""

    def __init__(self, name: str = "titanax"):
        """Initialize the logger.

        Args:
            name: Name/prefix for this logger
        """
        self.name = name
        self._start_time = time.time()

    @abstractmethod
    def log_scalar(self, name: str, value: LogValue, step: int) -> None:
        """Log a single scalar value.

        Args:
            name: Name of the metric
            value: Scalar value to log
            step: Training step number
        """
        pass

    @abstractmethod
    def log_dict(self, metrics: LogDict, step: int) -> None:
        """Log a dictionary of metrics.

        Args:
            metrics: Dictionary mapping metric names to values
            step: Training step number
        """
        pass

    def flush(self) -> None:
        """Flush any buffered log data. Override if needed."""
        pass

    def close(self) -> None:
        """Close the logger and cleanup resources. Override if needed."""
        pass

    def _format_value(self, value: LogValue) -> str:
        """Format a log value for display."""
        if isinstance(value, float):
            if abs(value) < 1e-3 or abs(value) > 1e3:
                return f"{value:.3e}"
            else:
                return f"{value:.6f}"
        else:
            return str(value)

    def _get_timestamp(self) -> str:
        """Get formatted timestamp."""
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    def _get_elapsed_time(self) -> float:
        """Get elapsed time since logger initialization."""
        return time.time() - self._start_time


class MultiLogger:
    """Combines multiple loggers into a single interface."""

    def __init__(self, loggers: List[Logger]):
        """Initialize with a list of loggers.

        Args:
            loggers: List of logger instances to multiplex
        """
        self.loggers = loggers

    def log_scalar(self, name: str, value: LogValue, step: int) -> None:
        """Log to all configured loggers."""
        for logger in self.loggers:
            logger.log_scalar(name, value, step)

    def log_dict(self, metrics: LogDict, step: int) -> None:
        """Log to all configured loggers."""
        for logger in self.loggers:
            logger.log_dict(metrics, step)

    def flush(self) -> None:
        """Flush all loggers."""
        for logger in self.loggers:
            if hasattr(logger, "flush"):
                logger.flush()

    def close(self) -> None:
        """Close all loggers."""
        for logger in self.loggers:
            if hasattr(logger, "close"):
                logger.close()


class NullLogger:
    """No-op logger for when logging is disabled."""

    def log_scalar(self, name: str, value: LogValue, step: int) -> None:
        """No-op."""
        pass

    def log_dict(self, metrics: LogDict, step: int) -> None:
        """No-op."""
        pass

    def flush(self) -> None:
        """No-op."""
        pass

    def close(self) -> None:
        """No-op."""
        pass


# Utility functions


def format_metrics_summary(metrics: LogDict, step: int, elapsed_time: float) -> str:
    """Format metrics into a readable summary string.

    Args:
        metrics: Dictionary of metrics to format
        step: Current training step
        elapsed_time: Elapsed time in seconds

    Returns:
        Formatted string suitable for console display
    """
    formatted_metrics = []
    for key, value in metrics.items():
        if isinstance(value, float):
            if abs(value) < 1e-3 or abs(value) > 1e3:
                formatted_metrics.append(f"{key}={value:.3e}")
            else:
                formatted_metrics.append(f"{key}={value:.6f}")
        else:
            formatted_metrics.append(f"{key}={value}")

    metrics_str = " ".join(formatted_metrics)
    return f"Step {step:6d} | {elapsed_time:8.2f}s | {metrics_str}"


def aggregate_metrics(metrics_list: List[LogDict]) -> LogDict:
    """Aggregate a list of metric dictionaries.

    Computes the mean of all numeric values across the list.
    Non-numeric values use the last seen value.

    Args:
        metrics_list: List of metric dictionaries to aggregate

    Returns:
        Aggregated metrics dictionary
    """
    if not metrics_list:
        return {}

    aggregated: LogDict = {}

    # Get all unique keys
    all_keys: set[str] = set()
    for metrics in metrics_list:
        all_keys.update(metrics.keys())

    # Aggregate each metric
    for key in all_keys:
        values = [m[key] for m in metrics_list if key in m]

        if not values:
            continue

        # For numeric values, compute mean
        if isinstance(values[0], (int, float)):
            numeric_values = [v for v in values if isinstance(v, (int, float))]
            aggregated[key] = sum(numeric_values) / len(numeric_values)
        else:
            # For non-numeric, use the last value
            aggregated[key] = values[-1]

    return aggregated
