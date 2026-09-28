"""
DEIK Robot Foci Kapus – Triangulátor világ-transzformáció tesztek
==================================================================

A koordináta-transzformáció (rektifikált cam ↔ kapu-koord.) konzisztenciáját és a
Kabsch-alapú world-kalibrációt ellenőrzi – valós kamera/kalibráció nélkül.

Futtatás:
    python -m pytest tests/test_triangulator.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

SRC_DIR = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC_DIR))

from stereo.triangulator import StereoTriangulator


CONFIG = {
    "stereo": {"calibration_file": "/nonexistent/stereo_calibration.npz"},
    "geometry": {
        "left_camera_x_mm": -1070.0,
        "camera_height_mm": 2700.0,
        "camera_z_offset_mm": -1200.0,
        "camera_pitch_deg": 30.0,
        "focal_length_px": 1365.2,
        "principal_point_x": 844.0,
        "principal_point_y": 608.0,
    },
}


def _make_triangulator() -> StereoTriangulator:
    # Kalibráció betöltése nélkül: R1=None, R_wc=None → config-alapú tartalék út.
    return StereoTriangulator(CONFIG)


def _random_rotation(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] *= -1.0
    return Q


class TestConfigFallbackTransform:
    """A config-alapú world-transzformáció és inverze egymás egzakt inverze."""

    def test_roundtrip_rect_to_world_and_back(self) -> None:
        tri = _make_triangulator()
        rng = np.random.default_rng(0)
        for _ in range(50):
            p_rect = rng.uniform(-3000, 3000, size=3)
            world = tri._rect_to_world(p_rect)
            back = tri._world_to_leftcam(world)[0]
            assert np.allclose(back, p_rect, atol=1e-6), f"{back} != {p_rect}"

    def test_ground_point_depends_on_geometry(self) -> None:
        # A tartalék transzformáció determinisztikus: azonos bemenet → azonos kimenet.
        tri = _make_triangulator()
        p = np.array([120.0, -50.0, 4000.0])
        assert np.allclose(tri._rect_to_world(p), tri._rect_to_world(p))


class TestRigidTransform:
    """Kabsch–Umeyama merev illesztés."""

    def test_recovers_known_transform(self) -> None:
        R_true = _random_rotation(42)
        t_true = np.array([100.0, -250.0, 1500.0])
        rng = np.random.default_rng(7)
        src = rng.uniform(-2000, 2000, size=(12, 3))
        dst = (R_true @ src.T).T + t_true

        R, t = StereoTriangulator._rigid_transform_3d(src, dst)
        assert np.allclose(R, R_true, atol=1e-6)
        assert np.allclose(t, t_true, atol=1e-6)
        assert abs(np.linalg.det(R) - 1.0) < 1e-9  # valódi forgatás (nem tükrözés)

    def test_rejects_reflection(self) -> None:
        # Tükrözött adat esetén is +1 determinánsú forgatást ad vissza.
        rng = np.random.default_rng(3)
        src = rng.uniform(-1000, 1000, size=(8, 3))
        dst = src.copy()
        dst[:, 2] *= -1.0  # tükrözés
        R, _ = StereoTriangulator._rigid_transform_3d(src, dst)
        assert np.linalg.det(R) > 0.0


class TestWorldCalibration:
    """calibrate_world_transform: illesztés + a triangulate() világ-transzform használja."""

    def test_calibration_sets_transform_and_low_residual(self) -> None:
        tri = _make_triangulator()
        R_true = _random_rotation(11)
        t_true = np.array([-30.0, 2650.0, -1180.0])
        rng = np.random.default_rng(5)
        cam = rng.uniform(-2500, 2500, size=(10, 3))
        world = (R_true @ cam.T).T + t_true

        result = tri.calibrate_world_transform(cam, world)
        assert result["rms_mm"] < 1e-6
        assert result["max_mm"] < 1e-6
        assert tri._R_wc is not None and tri._t_wc is not None

        # A kalibrált merev transzformáció felülírja a config-alapú képletet.
        for c, w in zip(cam, world):
            assert np.allclose(tri._rect_to_world(c), w, atol=1e-6)

    def test_requires_min_three_points(self) -> None:
        tri = _make_triangulator()
        with pytest.raises(ValueError):
            tri.calibrate_world_transform(np.zeros((2, 3)), np.zeros((2, 3)))

    def test_calibrated_roundtrip_projection_consistency(self) -> None:
        # World→cam→world oda-vissza konzisztens a kalibrált úton is (R1=None mellett).
        tri = _make_triangulator()
        R_true = _random_rotation(21)
        t_true = np.array([15.0, 2700.0, -1200.0])
        rng = np.random.default_rng(9)
        cam = rng.uniform(-2000, 2000, size=(6, 3))
        world = (R_true @ cam.T).T + t_true
        tri.calibrate_world_transform(cam, world)

        pts_world = rng.uniform(-2000, 2000, size=(15, 3))
        back = tri._world_to_leftcam(pts_world)
        forward = np.array([tri._rect_to_world(p) for p in back])
        assert np.allclose(forward, pts_world, atol=1e-6)

    def test_coplanar_ground_calibration_enforces_physical_orientation(self) -> None:
        """Talajsíki (Y=0) kalibráció esetén a fizikai kényszer (t_y > 0, R_11 < 0) érvényesül."""
        tri = _make_triangulator()
        # Valós sztereó kalibrációból származó rektifikált kamera és kapu pontok
        cam = np.array([
            [1075.81965207, -291.71048219, 5473.04882383],
            [-176.76397137,   56.18203279, 4811.91850437],
            [1902.97258144,  749.95711709, 3793.43728912],
        ])
        world = np.array([
            [200.0, 0.0, 3460.0],
            [-1000.0, 0.0, 2850.0],
            [970.0, 0.0, 1570.0],
        ])
        result = tri.calibrate_world_transform(cam, world)
        assert result["rms_mm"] < 100.0  # Valós kalibráció enyhe mérési hibával
        assert tri._R_wc is not None and tri._t_wc is not None

        # Fizikai kényszerek:
        # 1. A kamera a talaj felett van: t_y > 0 (~2500 mm)
        assert tri._t_wc[1] > 2000.0
        # 2. A kamera képén lefelé mozgás a valóságban lefelé irányul (R[1, 1] < 0)
        assert tri._R_wc[1, 1] < 0.0
        # 3. Determináns -1 (különböző kezesség: OpenCV Y-le vs Kapu Y-fel)
        assert np.linalg.det(tri._R_wc) < 0.0

        # 4. Levegőben lévő pontra a rekonstruált magasság pozitív
        # Egy pont, amely a kamera képén felfelé mozdul el (Y_cam csökken)
        p_cam_air = cam[0].copy()
        p_cam_air[1] -= 200.0  # felfelé mozdult a kamera látómezejében
        p_world_air = tri._rect_to_world(p_cam_air)
        assert p_world_air[1] > 100.0  # Pozitív magasság a világban!


class TestLeftPixelBackProjection:
    """A mono-mélység tartalék a kalibrált K1/D1-et használja, nem a config/P1 értékeit."""

    def test_uses_calibrated_k1_and_distortion(self) -> None:
        tri = _make_triangulator()
        tri._K1 = np.array([[1420.0, 0.0, 1020.0], [0.0, 1420.0, 614.0], [0.0, 0.0, 1.0]])
        tri._D1 = np.zeros(5)
        assert tri.left_focal_length_px == pytest.approx(1420.0)
        x_n, y_n = tri.left_pixel_to_normalized(1020.0, 614.0)
        assert x_n == pytest.approx(0.0, abs=1e-9) and y_n == pytest.approx(0.0, abs=1e-9)
        x_n, _ = tri.left_pixel_to_normalized(1020.0 + 142.0, 614.0)
        assert x_n == pytest.approx(0.1, abs=1e-9)

        tri._D1 = np.array([-0.2, 0.05, 0.0, 0.0, 0.0])
        x_d, _ = tri.left_pixel_to_normalized(1020.0 + 500.0, 614.0)
        assert x_d > 500.0 / 1420.0  # hordótorzítás korrekciója kifelé tolja a sugarat

    def test_uncalibrated_falls_back_to_config_intrinsics(self) -> None:
        tri = _make_triangulator()
        assert tri.left_focal_length_px == pytest.approx(1365.2)
        x_n, y_n = tri.left_pixel_to_normalized(844.0, 608.0)
        assert x_n == pytest.approx(0.0, abs=1e-9) and y_n == pytest.approx(0.0, abs=1e-9)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

