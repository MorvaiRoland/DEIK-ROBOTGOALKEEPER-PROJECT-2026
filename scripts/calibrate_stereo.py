"""
DEIK Robot Foci Kapus – Sztereó Kalibrálás Szkript
====================================================

Ez a szkript interaktívan elvégzi a két Ximea kamera sztereó kalibrálását.

Mit csinál:
    1. Megnyitja mindkét kamerát
    2. Live képet mutat (bal/jobb egymás mellett)
    3. Sakktábla detektálás: SPACE → mentés, 'q' → kilépés
    4. Legalább min. 20 képpárnál elvégzi a kalibrálást
    5. Elmenti: data/calibration/stereo_calibration.npz

Szükséges:
    - Nyomtatott sakktábla: 9×6 belső sarok, 30 mm négyzetméret
    - Jó megvilágítás (ne legyen tükröző fény a sakktáblán)
    - Különböző szögek és távolságok (0.5m – 3m)

Futtatás:
    python scripts/calibrate_stereo.py
    python scripts/calibrate_stereo.py --config config/config.yaml
    python scripts/calibrate_stereo.py --mock    # Webcam kamerakkal

Hivatkozás:
    OpenCV kalibrálási doku: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Projekt gyökér elérési út beállítása
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from camera.camera_manager import CameraManager

logger = logging.getLogger(__name__)

# Megjelenítési beállítások
DISPLAY_WIDTH = 960    # Megjelenítési ablak szélessége (px, mindkét kamera együtt)
DISPLAY_HEIGHT = 360   # Megjelenítési ablak magassága (px)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sztereó kalibrálás Ximea kamerákkal")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--mock", action="store_true", help="Mock kamerák használata")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data/calibration/stereo_calibration.npz"))
    parser.add_argument("--min-frames", type=int, default=None,
                        help="Minimum képpárok száma (konfig-ból veszi ha nincs megadva)")
    return parser.parse_args()


def find_chessboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Megkeresi a sakktábla sarkait egy képen.

    Args:
        image:        BGR kép
        pattern_size: (belső_sarkok_x, belső_sarkok_y)

    Returns:
        (N, 1, 2) float32 tömb a sarokpontokkal, vagy None ha nem talált
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Gyors keresés először
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_FAST_CHECK
    )

    if not found:
        return None

    # Sub-pixel pontosítás
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return corners_refined


def run_calibration(
    all_obj_points: List[np.ndarray],
    all_img_pts_left: List[np.ndarray],
    all_img_pts_right: List[np.ndarray],
    image_size: Tuple[int, int],
    stereo_cfg: dict,
    geo_cfg: dict,
) -> dict:
    """
    Elvégzi a sztereó kalibrálást és visszaadja az eredményt.

    Args:
        all_obj_points:    3D sakktábla pontok (mindegyik képpárhoz azonos)
        all_img_pts_left:  Bal képek sarokpontjai
        all_img_pts_right: Jobb képek sarokpontjai
        image_size:        (szélesség, magasság) pixelben
        stereo_cfg:        A konfig "stereo" szekciója
        geo_cfg:           A konfig "geometry" szekciója

    Returns:
        Dict a kalibrálási eredményekkel (K1, K2, D1, D2, R, T, P1, P2, Q, rmse)
    """
    logger.info("Sztereó kalibrálás futtatása %d képpárral...", len(all_obj_points))

    # Becsült belső mátrix (kalibrálás előtt)
    f_px = geo_cfg.get("focal_length_px", 1365.2)
    cx = geo_cfg.get("principal_point_x", 968.0)
    cy = geo_cfg.get("principal_point_y", 608.0)

    K_init = np.array([
        [f_px,  0.0,  cx],
        [ 0.0, f_px,  cy],
        [ 0.0,  0.0, 1.0],
    ], dtype=np.float64)

    # 1. Egyedi kamera kalibrálás (bal és jobb külön-külön)
    logger.info("  Bal kamera kalibrálás...")
    rmse_l, K1, D1, _, _ = cv2.calibrateCamera(
        all_obj_points, all_img_pts_left, image_size, K_init.copy(), None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    logger.info("  Bal kamera RMSE: %.4f px", rmse_l)

    logger.info("  Jobb kamera kalibrálás...")
    rmse_r, K2, D2, _, _ = cv2.calibrateCamera(
        all_obj_points, all_img_pts_right, image_size, K_init.copy(), None,
        flags=cv2.CALIB_USE_INTRINSIC_GUESS
    )
    logger.info("  Jobb kamera RMSE: %.4f px", rmse_r)

    # 2. Sztereó kalibrálás (relatív R, T meghatározása)
    logger.info("  Sztereó kalibrálás...")
    stereo_flags = (
        cv2.CALIB_FIX_INTRINSIC   # Az egyedi kalibrálásból rögzítjük K1, K2-t
    )
    rmse_stereo, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
        all_obj_points,
        all_img_pts_left, all_img_pts_right,
        K1, D1, K2, D2,
        image_size,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
        flags=stereo_flags,
    )
    logger.info("  Sztereó kalibrálás RMSE: %.4f px", rmse_stereo)

    # 3. Sztereó rektifikáció
    logger.info("  Rektifikációs mátrixok számítása...")
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K1, D1, K2, D2, image_size,
        R, T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    # Baseline (T vektor norma)
    baseline_mm = float(np.linalg.norm(T))
    logger.info("  Mért baseline: %.1f mm (referencia: %.1f mm)",
                baseline_mm, geo_cfg.get("baseline_mm", 4900.0))

    return {
        "K1": K1, "D1": D1, "K2": K2, "D2": D2,
        "R": R, "T": T, "E": E, "F": F,
        "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
        "rmse": rmse_stereo,
        "image_width": image_size[0],
        "image_height": image_size[1],
        "baseline_mm": baseline_mm,
    }


def main() -> None:
    """Főprogram: interaktív kalibrálás."""
    # Logging beállítása
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S"
    )

    args = parse_args()

    # Konfiguráció betöltése
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.mock:
        config["camera"]["type"] = "mock"

    stereo_cfg = config.get("stereo", {})
    geo_cfg = config.get("geometry", {})

    # Sakktábla paraméterek
    cb_cfg = stereo_cfg.get("chessboard", {})
    pattern_size = (
        int(cb_cfg.get("inner_corners_x", 9)),
        int(cb_cfg.get("inner_corners_y", 6))
    )
    square_size_mm = float(cb_cfg.get("square_size_mm", 30.0))
    min_frames = args.min_frames or int(stereo_cfg.get("min_calibration_frames", 20))
    max_rmse = float(stereo_cfg.get("max_acceptable_rmse_px", 1.0))

    logger.info("=" * 55)
    logger.info("DEIK Sztereó Kalibrálás")
    logger.info("  Sakktábla: %dx%d belső sarok, %g mm négyzetméret",
                *pattern_size, square_size_mm)
    logger.info("  Min. képpárok: %d", min_frames)
    logger.info("=" * 55)

    # 3D sakktábla pontok (Z=0, a sakktábla síkján)
    obj_pattern = np.zeros((pattern_size[0] * pattern_size[1], 3), dtype=np.float32)
    obj_pattern[:, :2] = np.mgrid[
        0:pattern_size[0], 0:pattern_size[1]
    ].T.reshape(-1, 2) * square_size_mm

    # Gyűjtött adatpontok
    all_obj_pts: List[np.ndarray] = []
    all_pts_left: List[np.ndarray] = []
    all_pts_right: List[np.ndarray] = []
    image_size: Optional[Tuple[int, int]] = None

    # Kamerák megnyitása
    cam_manager = CameraManager(config)
    if not cam_manager.open():
        logger.error("Kamerák megnyitása sikertelen!")
        sys.exit(1)

    logger.info("Kamerák megnyitva. Vezérlők:")
    logger.info("  SPACE    → Képpár mentése (ha sakktáblát talált)")
    logger.info("  'c'      → Kalibrálás futtatása (ha van elég képpár)")
    logger.info("  'r'      → Képpárok törlése, újrakezd")
    logger.info("  'q'      → Kilépés")
    print()

    cv2.namedWindow("Sztereó Kalibrálás", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Sztereó Kalibrálás", DISPLAY_WIDTH, DISPLAY_HEIGHT)

    try:
        while True:
            pair = cam_manager.read_stereo_pair()
            if not pair.success:
                continue

            fl = pair.left.image.copy()
            fr = pair.right.image.copy()

            if image_size is None:
                h, w = fl.shape[:2]
                image_size = (w, h)

            # Sakktábla keresés mindkét képen
            corners_l = find_chessboard_corners(fl, pattern_size)
            corners_r = find_chessboard_corners(fr, pattern_size)
            both_found = corners_l is not None and corners_r is not None

            # Rajzolás (sarokpontok megjelenítése)
            display_l = fl.copy()
            display_r = fr.copy()

            if corners_l is not None:
                cv2.drawChessboardCorners(display_l, pattern_size, corners_l, True)
            if corners_r is not None:
                cv2.drawChessboardCorners(display_r, pattern_size, corners_r, True)

            # Státusz szöveg
            status_color = (0, 255, 0) if both_found else (0, 100, 255)
            status_text = f"Kepparok: {len(all_obj_pts)}/{min_frames}"
            find_text = "MINDKET KAMERA: TALALT" if both_found else "Sakktabla keresese..."

            cv2.putText(display_l, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            cv2.putText(display_l, find_text, (10, 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.putText(display_r, "SPACE=mentes  c=kalibralas  q=kilepes", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Összefűzés és méretezés
            combined_h = min(display_l.shape[0], display_r.shape[0])
            combined = cv2.hconcat([
                cv2.resize(display_l, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT)),
                cv2.resize(display_r, (DISPLAY_WIDTH // 2, DISPLAY_HEIGHT)),
            ])
            cv2.imshow("Sztereó Kalibrálás", combined)

            # Billentyű feldolgozás
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' '):
                # SPACE: képpár mentése (ha mindkét kamera talált)
                if both_found:
                    all_obj_pts.append(obj_pattern.astype(np.float32))
                    all_pts_left.append(corners_l)
                    all_pts_right.append(corners_r)
                    logger.info("✓ Képpár mentve: %d/%d", len(all_obj_pts), min_frames)
                    # Rövid villanás visszajelzésként
                    time.sleep(0.3)
                else:
                    logger.warning("⚠ Sakktábla nem látható mindkét kamerában!")

            elif key == ord('c'):
                # c: kalibrálás futtatása
                if len(all_obj_pts) < min_frames:
                    logger.warning("Nincs elég képpár: %d/%d", len(all_obj_pts), min_frames)
                else:
                    logger.info("Kalibrálás megkezdése...")
                    break

            elif key == ord('r'):
                # r: törlés
                all_obj_pts.clear()
                all_pts_left.clear()
                all_pts_right.clear()
                logger.info("Képpárok törölve. Újrakezd.")

            elif key == ord('q'):
                logger.info("Kilépés kalibrálás nélkül.")
                cam_manager.close()
                cv2.destroyAllWindows()
                sys.exit(0)

    except KeyboardInterrupt:
        logger.info("Megszakítva.")
        cam_manager.close()
        cv2.destroyAllWindows()
        sys.exit(0)

    cam_manager.close()
    cv2.destroyAllWindows()

    # --- Kalibrálás futtatása ---
    if len(all_obj_pts) < min_frames:
        logger.error("Nincs elég képpár a kalibráláshoz (%d/%d)!", len(all_obj_pts), min_frames)
        sys.exit(1)

    result = run_calibration(
        all_obj_pts, all_pts_left, all_pts_right,
        image_size, stereo_cfg, geo_cfg
    )

    # --- Eredmény mentése ---
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        str(output_path),
        K1=result["K1"], D1=result["D1"],
        K2=result["K2"], D2=result["D2"],
        R=result["R"], T=result["T"],
        E=result["E"], F=result["F"],
        R1=result["R1"], R2=result["R2"],
        P1=result["P1"], P2=result["P2"],
        Q=result["Q"],
        rmse=result["rmse"],
        image_width=result["image_width"],
        image_height=result["image_height"],
        baseline_mm=result["baseline_mm"],
    )

    # --- Eredmény kiírása ---
    rmse = result["rmse"]
    quality = "KIVÁLÓ" if rmse < 0.5 else ("JÓ" if rmse < 1.0 else "GYENGE – adj hozzá több képet!")

    print()
    print("=" * 55)
    print("KALIBRÁLÁS KÉSZ!")
    print(f"  RMSE újraveítítési hiba: {rmse:.4f} px  [{quality}]")
    print(f"  Mért baseline: {result['baseline_mm']:.1f} mm")
    print(f"  Bal K[0,0] (f_px): {result['K1'][0,0]:.2f}")
    print(f"  Jobb K[0,0] (f_px): {result['K2'][0,0]:.2f}")
    print(f"  Mentve: {output_path}")
    print("=" * 55)

    if rmse > max_rmse:
        logger.warning(
            "RMSE=%.4f px > küszöb=%.1f px. "
            "Adj hozzá több képpárt különböző szögekből!",
            rmse, max_rmse
        )


if __name__ == "__main__":
    main()
