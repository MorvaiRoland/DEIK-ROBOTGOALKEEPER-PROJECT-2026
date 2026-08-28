"""Egységtesztek az időbélyeg-alapú 2D Kalman követőhöz."""

import sys
from pathlib import Path

import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from detection.kalman_tracker import KalmanTracker2D


def test_project_uses_elapsed_seconds_not_inference_frame_count() -> None:
    """A GUI előrevetítés valódi időt használjon, ne egy fix frame-lépést."""
    tracker = KalmanTracker2D(process_noise=0.01, measurement_noise=0.01)
    for index in range(6):
        tracker.update(100.0 + 100.0 * index, 50.0, timestamp=1.0 + 0.1 * index)

    x, y = tracker.project(1.7, max_horizon_s=0.2)

    assert x == pytest.approx(800.0, abs=8.0)
    assert y == pytest.approx(50.0, abs=2.0)


def test_project_horizon_is_limited() -> None:
    """Régi eredmény ne küldhesse a zöld jelölést kontrollálatlanul messzire."""
    tracker = KalmanTracker2D(process_noise=0.01, measurement_noise=0.01)
    for index in range(6):
        tracker.update(100.0 * index, 0.0, timestamp=1.0 + 0.1 * index)

    x, _ = tracker.project(5.0, max_horizon_s=0.05)

    assert x == pytest.approx(550.0, abs=8.0)
