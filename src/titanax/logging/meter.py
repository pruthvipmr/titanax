"""Metrics metering utilities for Titanax logging.

This module provides a `MetricsMeter` helper that tracks rolling statistics
such as throughput, step latency, and moving averages of scalar metrics.
It is designed to integrate with Titanax loggers so that training scripts can
report rich observability signals without bespoke plumbing.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Iterable, Mapping, MutableMapping, Optional

from ..types import LogDict, LogValue


Number = float | int


def _is_number(value: LogValue) -> bool:
    """Return True if *value* is an int or float (bools excluded)."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


class MetricsMeter:
    """Track rolling metrics for training observability.

    The meter keeps short-term windows of scalar metrics to compute moving
    averages and derives performance properties such as throughput and step
    latency. It can be queried each step to supply additional metrics to a
    logger in a consistent namespace (`meter/`).
    """

    def __init__(
        self,
        window: int = 20,
        *,
        track_keys: Optional[Iterable[str]] = None,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")

        self.window = window
        self._start_time = time.perf_counter()
        self._history: Dict[str, Deque[float]] = {}
        self._step_times: Deque[float] = deque(maxlen=window)
        self._window_samples: Deque[int] = deque(maxlen=window)
        self._total_samples: int = 0
        self._track_keys = set(track_keys or [])

    def reset(self) -> None:
        """Reset all accumulated statistics."""
        self._start_time = time.perf_counter()
        self._history.clear()
        self._step_times.clear()
        self._window_samples.clear()
        self._total_samples = 0

    def update(
        self,
        *,
        step: int,
        metrics: Mapping[str, LogValue],
        batch_size: Optional[int],
        step_time_s: float,
    ) -> LogDict:
        """Update the meter with metrics from a training step.

        Args:
            step: Global step number (used for reference only)
            metrics: Metrics dictionary produced by the training step
            batch_size: Number of samples processed this step (if known)
            step_time_s: Wall-clock latency for the step in seconds

        Returns:
            Additional metrics to log alongside the provided metrics.
        """
        if step_time_s < 0:
            raise ValueError("step_time_s must be non-negative")

        now = time.perf_counter()
        self._step_times.append(step_time_s)

        if batch_size is not None and batch_size >= 0:
            self._total_samples += batch_size
            self._window_samples.append(batch_size)
        else:
            batch_size = None

        # Track moving averages for scalar metrics
        for key, value in metrics.items():
            if not _is_number(value):
                continue

            if self._track_keys and key not in self._track_keys:
                continue

            history = self._history.setdefault(key, deque(maxlen=self.window))
            history.append(float(value))

        extras: MutableMapping[str, LogValue] = {}

        extras["meter/step_time_s"] = step_time_s
        if self._step_times:
            extras["meter/step_time_s_ma"] = sum(self._step_times) / len(
                self._step_times
            )

        if batch_size is not None and step_time_s > 0:
            extras["meter/throughput_samples_per_s"] = batch_size / step_time_s

        if self._window_samples and sum(self._step_times) > 0:
            window_samples = sum(self._window_samples)
            window_time = sum(self._step_times)
            extras["meter/throughput_samples_per_s_ma"] = window_samples / window_time

        elapsed = now - self._start_time
        if elapsed > 0 and self._total_samples:
            extras["meter/throughput_samples_per_s_avg"] = self._total_samples / elapsed

        for key, history in self._history.items():
            if not history:
                continue
            extras[f"meter/{key}_ma"] = sum(history) / len(history)

        extras["meter/step"] = step
        return dict(extras)


__all__ = ["MetricsMeter"]
