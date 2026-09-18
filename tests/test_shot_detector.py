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
    return ImpactPrediction(x_mm=0.0, y_mm=800.0, time_to_impact_s=0.2, valid=True, in_goal=True)


def _feed_valid_shot(detector: ShotDetector):
    # Z egészen a (default) confirm_z_mm=1500 mm-es küszöb alá esik, különben a
    # megerősítés sosem történik meg – lásd ShotDetector.update() confirm_z_mm gate.
    status = None
    for index, z_mm in enumerate((9000.0, 7360.0, 5720.0, 4080.0, 2440.0, 800.0)):
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


def test_kinematic_shot_confirms_without_in_goal_prediction() -> None:
    # Gyors lövésnél a ballisztikus in_goal-predikció megbízhatatlan (sztereó Y-hiba),
    # ezért alapból (require_impact_in_goal=False) a mozgás alapján erősítünk meg.
    detector = ShotDetector(CONFIG)
    detector.arm_next_shot()
    off_target = ImpactPrediction(x_mm=0.0, y_mm=-3000.0, time_to_impact_s=0.2, valid=True, in_goal=False)
    status = None
    for index, z_mm in enumerate((9000.0, 7360.0, 5720.0, 4080.0, 2440.0, 800.0)):
        status = detector.update(z_mm, index * 0.1, off_target, True)
    assert status.confirmed


def test_require_in_goal_flag_restores_strict_gate() -> None:
    strict_cfg = {"shot_detection": {**CONFIG["shot_detection"], "require_impact_in_goal": True}}
    detector = ShotDetector(strict_cfg)
    detector.arm_next_shot()
    off_target = ImpactPrediction(x_mm=0.0, y_mm=-3000.0, time_to_impact_s=0.2, valid=True, in_goal=False)
    status = None
    for index, z_mm in enumerate((9000.0, 7360.0, 5720.0, 4080.0, 2440.0, 800.0)):
        status = detector.update(z_mm, index * 0.1, off_target, True)
    assert not status.confirmed
    assert status.armed

