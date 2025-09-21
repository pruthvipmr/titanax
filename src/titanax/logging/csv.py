"""CSV logger implementation for Titanax."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Optional

from .base import BaseLogger
from ..types import LogDict, LogValue


class CSVLogger(BaseLogger):
    """Append training metrics to a CSV file."""

    def __init__(
        self,
        path: str | Path,
        *,
        name: str = "titanax",
        append: bool = True,
        delimiter: str = ",",
    ) -> None:
        super().__init__(name=name)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.delimiter = delimiter
        mode = "a+" if append else "w+"
        self._file = self.path.open(mode, newline="", encoding="utf-8")
        # Ensure subsequent writes append at end
        self._file.seek(0, 2)
        self._writer: Optional[csv.DictWriter[str]] = None
        self._fieldnames: list[str] = []

    def log_scalar(self, name: str, value: LogValue, step: int) -> None:
        self.log_dict({name: value}, step)

    def log_dict(self, metrics: LogDict, step: int) -> None:
        if not metrics:
            return

        row = {"step": step, **metrics}

        if self._writer is None:
            self._fieldnames = list(row.keys())
            self._writer = csv.DictWriter(
                self._file, fieldnames=self._fieldnames, delimiter=self.delimiter
            )
            self._writer.writeheader()
        else:
            # If new keys appear, extend the header and rewrite file if necessary
            new_keys = [k for k in row.keys() if k not in self._fieldnames]
            if new_keys:
                self._fieldnames.extend(new_keys)
                self._rewrite_with_new_header()

        assert self._writer is not None  # for type checkers
        self._writer.writerow({k: row.get(k, "") for k in self._fieldnames})
        self.flush()

    def _rewrite_with_new_header(self) -> None:
        """Rewrite the CSV file with an updated header to accommodate new keys."""
        self._file.flush()
        # Read existing rows (excluding header)
        self._file.seek(0)
        existing_rows = list(csv.DictReader(self._file, delimiter=self.delimiter))

        # Rewrite the file with the expanded header
        self._file.close()
        with self.path.open("w", newline="", encoding="utf-8") as rewritten:
            writer = csv.DictWriter(
                rewritten, fieldnames=self._fieldnames, delimiter=self.delimiter
            )
            writer.writeheader()
            for row in existing_rows:
                writer.writerow({k: row.get(k, "") for k in self._fieldnames})

        # Re-open in append mode for subsequent writes
        self._file = self.path.open("a+", newline="", encoding="utf-8")
        self._file.seek(0, 2)
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._fieldnames, delimiter=self.delimiter
        )

    def flush(self) -> None:
        self._file.flush()

    def close(self) -> None:
        try:
            self._file.flush()
        finally:
            self._file.close()


__all__ = ["CSVLogger"]
