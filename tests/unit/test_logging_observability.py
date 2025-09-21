"""Tests for logging and observability components."""

from __future__ import annotations

import csv
from pathlib import Path

import pytest

from tensorboard.backend.event_processing import event_file_loader

from src.titanax.logging.csv import CSVLogger
from src.titanax.logging.tensorboard import TensorBoardLogger
from src.titanax.logging.meter import MetricsMeter


def test_csv_logger_handles_dynamic_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "metrics.csv"
    logger = CSVLogger(csv_path, append=False)

    logger.log_dict({"loss": 0.5}, step=1)
    logger.log_dict({"loss": 0.25, "accuracy": 0.9}, step=2)
    logger.close()

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    assert reader.fieldnames == ["step", "loss", "accuracy"]
    assert rows[0]["accuracy"] == ""
    assert float(rows[1]["accuracy"]) == pytest.approx(0.9)


def test_tensorboard_logger_writes_events(tmp_path: Path) -> None:
    logdir = tmp_path / "tb"
    logger = TensorBoardLogger(logdir)

    logger.log_scalar("loss", 0.5, step=1)
    logger.log_dict({"accuracy": 0.75, "note": "ignored"}, step=2)
    logger.flush()
    logger.close()

    event_files = list(logdir.glob("events.*"))
    assert event_files, "Expected TensorBoard event file to be created"

    loader = event_file_loader.EventFileLoader(str(event_files[0]))
    events = list(loader.Load())
    scalars: dict[str, float] = {}
    for event in events:
        if not event.summary.value:
            continue
        for value in event.summary.value:
            if value.HasField("simple_value"):
                scalars[value.tag] = value.simple_value
            elif value.HasField("tensor") and value.tensor.float_val:
                scalars[value.tag] = value.tensor.float_val[0]

    assert scalars["loss"] == pytest.approx(0.5)
    assert scalars["accuracy"] == pytest.approx(0.75)
    assert "note" not in scalars


def test_metrics_meter_tracks_throughput_and_averages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    times = iter([0.0, 1.0, 2.0])
    monkeypatch.setattr(
        "src.titanax.logging.meter.time.perf_counter", lambda: next(times)
    )

    meter = MetricsMeter(window=2)

    update1 = meter.update(
        step=1,
        metrics={"loss": 1.0},
        batch_size=4,
        step_time_s=0.5,
    )

    update2 = meter.update(
        step=2,
        metrics={"loss": 0.5, "accuracy": 0.25},
        batch_size=4,
        step_time_s=0.25,
    )

    assert update1["meter/throughput_samples_per_s"] == pytest.approx(8.0)
    assert update1["meter/loss_ma"] == pytest.approx(1.0)

    assert update2["meter/throughput_samples_per_s"] == pytest.approx(16.0)
    assert update2["meter/throughput_samples_per_s_ma"] == pytest.approx(
        10.666666, rel=1e-6
    )
    assert update2["meter/step_time_s_ma"] == pytest.approx(0.375)
    assert update2["meter/throughput_samples_per_s_avg"] == pytest.approx(4.0)
    assert update2["meter/loss_ma"] == pytest.approx(0.75)
    assert update2["meter/accuracy_ma"] == pytest.approx(0.25)
