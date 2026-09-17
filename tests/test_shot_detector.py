"""A kézi élesítéses, indítózóna-alapú lövésdetektor tesztjei."""

import sys
from pathlib import Path

from prediction.trajectory_predictor import ImpactPrediction

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from gui.main_window import ShotDetector


CONFIG = {
    "shot_detection": {
        "manual_arm_required": True,
        "launch_z_min_mm": 5000.0,
        "launch_z_max_mm": 12000.0,
        "min_travel_mm": 2000.0,
        "min_samples": 6,
        "min_speed_mm_s": 4000.0,
        "min_r2": 0.85,
        "finish_z_mm": 700.0,
        "max_missing_frames": 2,
        "cooldown_s": 2.5,
    }
}


def _impact() -> ImpactPrediction:
    return ImpactPrediction(x_mm=0.0, y_mm=800.0, time_to_impact_s=0.2, valid=True)


def _feed_valid_shot(detector: ShotDetector):
    status = None
    for index, z_mm in enumerate((9000.0, 8500.0, 8000.0, 7500.0, 7000.0, 6500.0)):
        status = detector.update(z_mm, index * 0.1, _impact(), True)
    return status


def test_unarmed_detector_rejects_anotherwise_valid_shot() -> None:
    detector = ShotDetector(CONFIG)
    status = _feed_valid_shot(detector)
    assert not status.confirmed
    assert not status.active
    assert not status.armed


def test_armed_detector_survives_frames_without_a_ball() -> None:
    detector = ShotDetector(CONFIG)
    detector.arm_next_shot()

    for timestamp_s in (0.0, 0.1, 0.2):
        status = detector.update(None, timestamp_s, None, False)
        assert status.armed

    confirmed = _feed_valid_shot(detector)
    assert confirmed.confirmed
    assert confirmed.active


def test_shot_requires_starting_in_launch_zone() -> None:
    detector = ShotDetector(CONFIG)
    detector.arm_next_shot()
    status = None
    for index, z_mm in enumerate((4500.0, 4000.0, 3500.0, 3000.0, 2500.0, 2000.0)):
        status = detector.update(z_mm, index * 0.1, _impact(), True)
    assert not status.confirmed
    assert status.armed


def test_shot_requires_minimum_travel_distance() -> None:
    detector = ShotDetector(CONFIG)
    detector.arm_next_shot()
    status = None
    for index, z_mm in enumerate((9000.0, 8800.0, 8600.0, 8400.0, 8200.0, 8000.0)):
        status = detector.update(z_mm, index * 0.1, _impact(), True)
    assert not status.confirmed
    assert status.armed


def test_confirmed_shot_is_finished_once_at_goal_plane() -> None:
    detector = ShotDetector(CONFIG)
    detector.arm_next_shot()

    confirmed = _feed_valid_shot(detector)
    assert confirmed.confirmed
    assert confirmed.active
    assert not confirmed.armed

    finished = detector.update(600.0, 0.6, _impact(), True)
    assert finished.finished
    assert not finished.active
    assert not finished.armed

    cooldown = detector.update(500.0, 0.7, _impact(), True)
    assert not cooldown.confirmed
    assert not cooldown.finished
