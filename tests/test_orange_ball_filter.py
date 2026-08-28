"""Szintetikus tesztek a RT-DETR találatok narancssárga-labda validációjához."""

import sys
from pathlib import Path

import cv2
import numpy as np

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from detection.ball_detector import BallDetector


def _filter() -> BallDetector:
    """Modellbetöltés nélküli példány, kizárólag a post-filter tesztelésére."""
    detector = object.__new__(BallDetector)
    detector._hsv_enabled = True
    detector._hsv_h_min, detector._hsv_h_max = 5, 25
    detector._hsv_s_min, detector._hsv_s_max = 110, 255
    detector._hsv_v_min, detector._hsv_v_max = 100, 255
    detector._hsv_min_ratio = 0.20
    detector._min_circularity = 0.55
    detector._min_aspect_ratio, detector._max_aspect_ratio = 0.60, 1.55
    detector._max_colored_blob_ratio = 0.90
    return detector


def test_accepts_orange_circle() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 35, (0, 140, 255), -1)  # BGR orange
    assert _filter()._validate_orange_color(image, 0, 0, 100, 100)


def test_rejects_solid_orange_rectangle() -> None:
    image = np.full((100, 100, 3), (0, 140, 255), dtype=np.uint8)
    assert not _filter()._validate_orange_color(image, 0, 0, 100, 100)


def test_rejects_non_orange_object() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.circle(image, (50, 50), 35, (0, 255, 0), -1)  # BGR green
    assert not _filter()._validate_orange_color(image, 0, 0, 100, 100)
