"""
DEIK Robot Foci Kapus – Tesztek: Kamera Pozíció Beállítási Segéd (test_alignment_helper.py)
"""

import math
import numpy as np
import pytest

from calibration.alignment_helper import (
    AlignmentResult,
    calculate_camera_alignment,
    draw_alignment_hud,
)


def _generate_mock_corners(
    center_x: float,
    center_y: float,
    angle_deg: float = 0.0,
    cols: int = 11,
    rows: int = 8,
    spacing: float = 20.0
) -> np.ndarray:
    """Segédfüggvény teszt sarokpontok generálásához."""
    obj = np.zeros((cols * rows, 2), dtype=np.float32)
    idx = 0
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)

    # Grids centered at (0, 0)
    grid_w = (cols - 1) * spacing
    grid_h = (rows - 1) * spacing

    for r in range(rows):
        for c in range(cols):
            x0 = c * spacing - grid_w / 2.0
            y0 = r * spacing - grid_h / 2.0
            # Rotation
            xr = x0 * cos_a - y0 * sin_a + center_x
            yr = x0 * sin_a + y0 * cos_a + center_y
            obj[idx] = [xr, yr]
            idx += 1

    return obj.reshape(-1, 1, 2)


def test_alignment_no_corners():
    image_size = (1920, 1200)
    res = calculate_camera_alignment(None, None, image_size)
    assert res.score == 0.0
    assert not res.both_found
    assert "Sakktábla nem látható" in res.general_summary


def test_alignment_perfect():
    image_size = (1920, 1200)
    # Tökéletesen szimmetrikus pozíciók
    # Bal kamera jobbra látja a pályaközepet (+200px offset a középponttól)
    # Jobb kamera balra látja a pályaközepet (-200px offset a középponttól)
    corners_l = _generate_mock_corners(960 + 200, 600, angle_deg=0.0)
    corners_r = _generate_mock_corners(960 - 200, 600, angle_deg=0.0)

    res = calculate_camera_alignment(corners_l, corners_r, image_size)

    assert res.both_found
    assert res.score >= 90.0
    assert abs(res.delta_y_px) < 1.0
    assert abs(res.asym_x_px) < 1.0
    assert "Tökéletes" in res.general_summary


def test_alignment_pitch_disparity():
    image_size = (1920, 1200)
    # Bal kamera 30 pixel-lel alacsonyabbra vetíti a táblát mint a jobb
    corners_l = _generate_mock_corners(1160, 630)
    corners_r = _generate_mock_corners(760, 600)

    res = calculate_camera_alignment(corners_l, corners_r, image_size)

    assert res.both_found
    assert res.score < 90.0
    assert abs(res.delta_y_px - 30.0) < 1.0

    # Utasítás ellenőrzés
    left_pitch_actions = [i.action for i in res.left_instructions if i.category == "pitch"]
    right_pitch_actions = [i.action for i in res.right_instructions if i.category == "pitch"]

    assert left_pitch_actions == ["DOWN"]
    assert right_pitch_actions == ["UP"]


def test_alignment_roll_angle():
    image_size = (1920, 1200)
    # Bal kamera +5 fokkal el van forgatva
    corners_l = _generate_mock_corners(1160, 600, angle_deg=5.0)
    corners_r = _generate_mock_corners(760, 600, angle_deg=0.0)

    res = calculate_camera_alignment(corners_l, corners_r, image_size)

    assert abs(res.roll_l_deg - 5.0) < 0.5
    left_roll_actions = [i.action for i in res.left_instructions if i.category == "roll"]
    assert left_roll_actions == ["CW"]


def test_alignment_roll_angle_portrait():
    # Álló kép (1216x1936): magasság > szélesség
    image_size = (1216, 1936)
    # Álló képen a függőleges sakktábla felirat angle_deg=90° alapértelmezetten 0° roll-nak felel meg
    corners_l = _generate_mock_corners(600, 960, angle_deg=92.0)
    corners_r = _generate_mock_corners(600, 960, angle_deg=90.0)

    res = calculate_camera_alignment(corners_l, corners_r, image_size)

    # Roll eltérés a 90°-os függőlegestől: 92° - 90° = +2.0°
    assert abs(res.roll_l_deg - 2.0) < 0.5
    assert abs(res.roll_r_deg) < 0.5


def test_alignment_one_camera():
    image_size = (1920, 1200)
    corners_l = _generate_mock_corners(1160, 600)

    res = calculate_camera_alignment(corners_l, None, image_size)

    assert res.left_found
    assert not res.right_found
    assert not res.both_found
    assert res.score == 30.0


def test_draw_alignment_hud():
    image_size = (1920, 1200)
    img_l = np.zeros((1200, 1920, 3), dtype=np.uint8)
    corners_l = _generate_mock_corners(1160, 600)
    corners_r = _generate_mock_corners(760, 600)

    res = calculate_camera_alignment(corners_l, corners_r, image_size)
    hud_img = draw_alignment_hud(img_l, corners_l, is_left=True, result=res)

    assert hud_img.shape == (1200, 1920, 3)
    assert not np.array_equal(hud_img, img_l)  # Rajzolt rá elemeket
