"""TensorBoard logger implementation for Titanax."""

from __future__ import annotations

import time
from pathlib import Path

from tensorboard.compat.proto import event_pb2, summary_pb2  # type: ignore
from tensorboard.summary.writer.event_file_writer import EventFileWriter  # type: ignore

from .base import BaseLogger
from ..types import LogDict, LogValue


class TensorBoardLogger(BaseLogger):
    """Log scalar metrics to TensorBoard event files."""

    def __init__(
        self,
        logdir: str | Path,
        *,
        name: str = "titanax",
        flush_secs: float = 2.0,
        filename_suffix: str = "",
        max_queue: int = 10,
    ) -> None:
        super().__init__(name=name)
        self.logdir = Path(logdir)
        self.logdir.mkdir(parents=True, exist_ok=True)
        self._writer = EventFileWriter(
            str(self.logdir),
            max_queue_size=max_queue,
            flush_secs=flush_secs,
            filename_suffix=filename_suffix,
        )

    def log_scalar(self, name: str, value: LogValue, step: int) -> None:
        if not isinstance(value, (int, float)):
            raise TypeError(
                f"TensorBoardLogger only supports numeric scalars, got {type(value).__name__}"
            )
        self._add_event(name, float(value), step)

    def log_dict(self, metrics: LogDict, step: int) -> None:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                self._add_event(key, float(value), step)

    def _add_event(self, tag: str, value: float, step: int) -> None:
        summary = summary_pb2.Summary(
            value=[summary_pb2.Summary.Value(tag=tag, simple_value=value)]
        )
        event = event_pb2.Event(wall_time=time.time(), step=step, summary=summary)
        self._writer.add_event(event)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        try:
            self._writer.flush()
        finally:
            self._writer.close()


__all__ = ["TensorBoardLogger"]
