"""Basic stdout logger for Titanax.

This module provides a simple logger that outputs training metrics to stdout
with step-based formatting.
"""

import sys
from typing import Optional, TextIO

from .base import BaseLogger
from ..types import LogValue, LogDict


class Basic(BaseLogger):
    """Basic logger that outputs to stdout with step-based formatting.

    This logger provides simple, readable output suitable for console monitoring
    during training. It formats metrics with appropriate precision and includes
    timing information.

    Example:
        >>> logger = Basic(name="training")
        >>> logger.log_scalar("loss", 0.123456, step=100)
        [2024-01-15 10:30:45] Step    100 |     5.23s | loss=0.123456

        >>> logger.log_dict({"loss": 0.1, "acc": 0.95}, step=101)
        [2024-01-15 10:30:46] Step    101 |     6.24s | loss=0.100000 acc=0.950000
    """

    def __init__(
        self,
        name: str = "titanax",
        output: Optional[TextIO] = None,
        show_timestamp: bool = True,
        show_elapsed: bool = True,
    ):
        """Initialize the Basic logger.

        Args:
            name: Name/prefix for this logger
            output: Output stream (defaults to sys.stdout)
            show_timestamp: Whether to show timestamps in output
            show_elapsed: Whether to show elapsed time since initialization
        """
        super().__init__(name)
        self.output = output or sys.stdout
        self.show_timestamp = show_timestamp
        self.show_elapsed = show_elapsed

    def log_scalar(self, name: str, value: LogValue, step: int) -> None:
        """Log a single scalar value to stdout.

        Args:
            name: Name of the metric
            value: Scalar value to log
            step: Training step number
        """
        metrics = {name: value}
        self.log_dict(metrics, step)

    def log_dict(self, metrics: LogDict, step: int) -> None:
        """Log a dictionary of metrics to stdout.

        Args:
            metrics: Dictionary mapping metric names to values
            step: Training step number
        """
        if not metrics:
            return

        # Build the log line components
        components = []

        # Add timestamp if enabled
        if self.show_timestamp:
            timestamp = self._get_timestamp()
            components.append(f"[{timestamp}]")

        # Add step information
        components.append(f"Step {step:6d}")

        # Add elapsed time if enabled
        if self.show_elapsed:
            elapsed = self._get_elapsed_time()
            components.append(f"{elapsed:8.2f}s")

        # Format metrics
        formatted_metrics = []
        for key, value in metrics.items():
            formatted_value = self._format_value(value)
            formatted_metrics.append(f"{key}={formatted_value}")

        if formatted_metrics:
            metrics_str = " ".join(formatted_metrics)
            components.append(metrics_str)

        # Join components and output
        log_line = " | ".join(components)
        print(log_line, file=self.output)

    def flush(self) -> None:
        """Flush the output stream."""
        if hasattr(self.output, "flush"):
            self.output.flush()

    def close(self) -> None:
        """Close the logger.

        Note: Does not close stdout/stderr if they were used as output.
        """
        # Don't close stdout/stderr
        if self.output not in (sys.stdout, sys.stderr):
            if hasattr(self.output, "close"):
                self.output.close()


class CompactBasic(BaseLogger):
    """Compact variant of Basic logger with shorter output format.

    This logger provides more condensed output suitable for scenarios where
    screen real estate is limited or log files need to be more compact.

    Example:
        >>> logger = CompactBasic()
        >>> logger.log_dict({"loss": 0.123, "acc": 0.95}, step=100)
        100: loss=0.123000 acc=0.950000
    """

    def __init__(self, name: str = "titanax", output: Optional[TextIO] = None):
        """Initialize the CompactBasic logger.

        Args:
            name: Name/prefix for this logger
            output: Output stream (defaults to sys.stdout)
        """
        super().__init__(name)
        self.output = output or sys.stdout

    def log_scalar(self, name: str, value: LogValue, step: int) -> None:
        """Log a single scalar value in compact format.

        Args:
            name: Name of the metric
            value: Scalar value to log
            step: Training step number
        """
        metrics = {name: value}
        self.log_dict(metrics, step)

    def log_dict(self, metrics: LogDict, step: int) -> None:
        """Log a dictionary of metrics in compact format.

        Args:
            metrics: Dictionary mapping metric names to values
            step: Training step number
        """
        if not metrics:
            return

        # Format metrics
        formatted_metrics = []
        for key, value in metrics.items():
            formatted_value = self._format_value(value)
            formatted_metrics.append(f"{key}={formatted_value}")

        if formatted_metrics:
            metrics_str = " ".join(formatted_metrics)
            log_line = f"{step}: {metrics_str}"
            print(log_line, file=self.output)

    def flush(self) -> None:
        """Flush the output stream."""
        if hasattr(self.output, "flush"):
            self.output.flush()

    def close(self) -> None:
        """Close the logger.

        Note: Does not close stdout/stderr if they were used as output.
        """
        # Don't close stdout/stderr
        if self.output not in (sys.stdout, sys.stderr):
            if hasattr(self.output, "close"):
                self.output.close()
