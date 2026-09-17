"""
DEIK Robot Foci Kapus – Világ-kalibrációs pontok interaktív kijelölése
=======================================================================

Segédeszköz a calibrate_world.py-hoz szükséges pont-korrespondenciák
(uL,vL,uR,vR pixel + X,Y,Z mm) összegyűjtéséhez.

Munkamenet:
    1. Tedd le a referenciapontot (pl. labdát) a pályára, MÉRD LE mérőszalaggal
       a valós X,Y,Z koordinátáját (mm, kapu-koordinátarendszer).
    2. Futtasd ezt a szkriptet, fagyaszd le a képet (SPACE), majd kattints
       a referenciapontra előbb a BAL, aztán a JOBB kamera ablakában.
       A kurzor körül egy nagyító segít pixel-pontosan eltalálni a labda közepét.
    3. A képen megjelenő beviteli mezőbe (nem a terminálba!) írd be a lemért
       X, majd ENTER, Y, majd ENTER, Z, majd ENTER értéket – a pont automatikusan
       bekerül a listába ÉS azonnal mentésre kerül a kimeneti JSON-be.
    4. Ismételd meg (ajánlott 6–10 pont), majd 'q'-val kilépés (már minden mentve van).

Vezérlők:
    SPACE      → élő kép lefagyasztása / feloldása
    kattintás  → pont kijelölése (bal ablak, majd jobb ablak)
    b          → megjelenítési fényerő-erősítés ciklikus váltása (sötét teremhez)
    0-9 . -    → szám beírása az aktív X/Y/Z mezőbe
    BACKSPACE  → utolsó karakter törlése a mezőből
    ENTER      → mező megerősítése, ugrás a következőre (Z után: pont mentése)
    ESC        → folyamatban lévő pont/beviteli mező megszakítása
    u          → utolsó MENTETT pont törlése
    q          → kilépés (a JSON már útközben mentve van)

Futtatás:
    python scripts/pick_world_points.py
    python scripts/pick_world_points.py --output data/calibration/points.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from camera.camera_manager import CameraManager  # noqa: E402

logger = logging.getLogger(__name__)

DISPLAY_MAX_WIDTH = 1000  # a kijelzett ablak célszélessége; a magasságot az eredeti
                          # kamera-arány (pl. 1688x1216) alapján számoljuk, hogy ne torzuljon
MAGNIFIER_SIZE = 220     # a nagyító ablak mérete (px, a kijelzőn)
MAGNIFIER_HALF_SRC = 22  # ennyi px sugarú területet nagyít fel az EREDETI képből
FIELDS = ("X", "Y", "Z")
FIELD_HINTS = {"X": "mm, jobbra+", "Y": "mm, felfelé+", "Z": "mm, pályára+"}

# Csak MEGJELENÍTÉSHEZ használt fényerő-erősítési szintek (a mentett pixel-
# koordinátákra és a kamera valós expozíciójára nincs hatással).
BRIGHTNESS_LEVELS = (1.0, 1.8, 2.6, 3.4)


def enhance_for_display(img: np.ndarray, level: float) -> np.ndarray:
    """Sötét terem esetén csak a kijelzett képet élénkíti (CLAHE + lineáris erősítés)."""
    if level <= 1.0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_ch = clahe.apply(l_ch)
    bright = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
    return cv2.convertScaleAbs(bright, alpha=level, beta=10)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Világ-kalibrációs pontok interaktív kijelölése")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "data" / "calibration" / "points.json"),
                        help="Kimeneti JSON útja (minden pont után azonnal frissül)")
    return parser.parse_args()


class MouseState:
    """Egy ablakon belüli utolsó kattintás és élő kurzorpozíció (KIJELZETT képkoordináta)."""

    def __init__(self) -> None:
        self.hover: Optional[Tuple[int, int]] = None
        self.clicked: Optional[Tuple[int, int]] = None

    def callback(self, event: int, x: int, y: int, flags: int, param) -> None:
        self.hover = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked = (x, y)


def draw_magnifier(display_img: np.ndarray, src_img: np.ndarray,
                    hover_xy: Optional[Tuple[int, int]],
                    scale_x: float, scale_y: float) -> None:
    """Nagyított kivágatot rajzol a kurzor köré (EREDETI felbontásból), pontos kattintáshoz."""
    if hover_xy is None:
        return
    src_x = int(hover_xy[0] * scale_x)
    src_y = int(hover_xy[1] * scale_y)
    h, w = src_img.shape[:2]
    x0 = max(0, min(w - 1, src_x - MAGNIFIER_HALF_SRC))
    y0 = max(0, min(h - 1, src_y - MAGNIFIER_HALF_SRC))
    x1 = max(0, min(w, src_x + MAGNIFIER_HALF_SRC))
    y1 = max(0, min(h, src_y + MAGNIFIER_HALF_SRC))
    if x1 - x0 < 4 or y1 - y0 < 4:
        return

    crop = src_img[y0:y1, x0:x1]
    zoom = cv2.resize(crop, (MAGNIFIER_SIZE, MAGNIFIER_SIZE), interpolation=cv2.INTER_NEAREST)
    cv2.drawMarker(zoom, (MAGNIFIER_SIZE // 2, MAGNIFIER_SIZE // 2),
                    (0, 0, 255), cv2.MARKER_CROSS, 18, 1)
    cv2.rectangle(zoom, (0, 0), (MAGNIFIER_SIZE - 1, MAGNIFIER_SIZE - 1), (0, 255, 255), 2)

    dh, dw = display_img.shape[:2]
    mx0, my0 = dw - MAGNIFIER_SIZE - 10, 10
    display_img[my0:my0 + MAGNIFIER_SIZE, mx0:mx0 + MAGNIFIER_SIZE] = zoom


def draw_input_box(display_img: np.ndarray, field: str, buffer: str) -> None:
    """A képen megjelenő X/Y/Z beviteli mező kirajzolása (nem terminál!)."""
    h, w = display_img.shape[:2]
    box_h = 70
    overlay = display_img.copy()
    cv2.rectangle(overlay, (0, h - box_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, display_img, 0.25, 0, dst=display_img)

    hint = FIELD_HINTS.get(field, "")
    cv2.putText(display_img, f"{field} ({hint}):", (14, h - box_h + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    cursor = buffer + "_"
    cv2.putText(display_img, cursor, (14, h - box_h + 55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def append_point_and_save(points: List[Dict[str, float]], output_path: Path,
                            uL: float, vL: float, uR: float, vR: float,
                            x_mm: float, y_mm: float, z_mm: float) -> None:
    points.append({
        "uL": uL, "vL": vL, "uR": uR, "vR": vR,
        "X": x_mm, "Y": y_mm, "Z": z_mm,
    })
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"points": points}, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("✓ Pont hozzáadva és mentve (%d db összesen): %s", len(points), output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    output_path = Path(args.output)
    points: List[Dict[str, float]] = []
    if output_path.exists():
        try:
            existing = json.loads(output_path.read_text(encoding="utf-8"))
            points = existing.get("points", []) if isinstance(existing, dict) else existing
            logger.info("Meglévő fájl betöltve, folytatás %d ponttal: %s", len(points), output_path)
        except (json.JSONDecodeError, OSError):
            logger.warning("A meglévő %s nem olvasható, üresen kezdünk.", output_path)

    cam_manager = CameraManager(config)
    if not cam_manager.open():
        logger.error("Kamerák megnyitása sikertelen!")
        sys.exit(1)

    mouse_left = MouseState()
    mouse_right = MouseState()

    win_left, win_right = "Bal kamera", "Jobb kamera"
    cv2.namedWindow(win_left, cv2.WINDOW_NORMAL)
    cv2.namedWindow(win_right, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win_left, mouse_left.callback)
    cv2.setMouseCallback(win_right, mouse_right.callback)

    logger.info("Vezérlők: SPACE=fagyaszt | kattintás=pont | b=fényerő | szám+ENTER=X/Y/Z beírás | "
                "ESC=megszakít | u=utolsó mentett pont törlése | q=kilépés")

    frozen = False
    frame_l: Optional[np.ndarray] = None
    frame_r: Optional[np.ndarray] = None
    pending_left_src: Optional[Tuple[float, float]] = None
    pending_right_src: Optional[Tuple[float, float]] = None
    # Sötét terem esetére: 1-es index = közepes fényerő-erősítés induláskor.
    brightness_idx = 1

    # Beviteli mód állapota: None amíg mindkét kattintás meg nem történt.
    input_field_idx: Optional[int] = None
    input_values: Dict[str, float] = {}
    input_buffer = ""

    try:
        while True:
            if not frozen:
                pair = cam_manager.read_stereo_pair()
                if pair.success:
                    frame_l = pair.left.image
                    frame_r = pair.right.image

            if frame_l is None or frame_r is None:
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    break
                continue

            src_h, src_w = frame_l.shape[:2]
            # A kijelzett méret az EREDETI kamera-arányt követi (pl. 1688x1216), hogy ne torzuljon.
            disp_w = DISPLAY_MAX_WIDTH
            disp_h = int(round(disp_w * src_h / src_w))
            scale_x = src_w / disp_w
            scale_y = src_h / disp_h

            # Kattintás feldolgozása csak akkor, ha még nem vagyunk beviteli módban.
            if input_field_idx is None:
                if mouse_left.clicked is not None:
                    cx, cy = mouse_left.clicked
                    pending_left_src = (cx * scale_x, cy * scale_y)
                    mouse_left.clicked = None
                if mouse_right.clicked is not None:
                    cx, cy = mouse_right.clicked
                    pending_right_src = (cx * scale_x, cy * scale_y)
                    mouse_right.clicked = None

                if pending_left_src is not None and pending_right_src is not None:
                    input_field_idx = 0
                    input_values = {}
                    input_buffer = ""
            else:
                # Beviteli módban a kattintások nem számítanak, csak elnyeljük őket.
                mouse_left.clicked = None
                mouse_right.clicked = None

            brightness = BRIGHTNESS_LEVELS[brightness_idx]
            display_l = enhance_for_display(cv2.resize(frame_l, (disp_w, disp_h)), brightness)
            display_r = enhance_for_display(cv2.resize(frame_r, (disp_w, disp_h)), brightness)

            if pending_left_src is not None:
                px, py = int(pending_left_src[0] / scale_x), int(pending_left_src[1] / scale_y)
                cv2.drawMarker(display_l, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)
            if pending_right_src is not None:
                px, py = int(pending_right_src[0] / scale_x), int(pending_right_src[1] / scale_y)
                cv2.drawMarker(display_r, (px, py), (0, 0, 255), cv2.MARKER_CROSS, 22, 2)

            status = (f"Pontok: {len(points)}  |  {'LEFAGYASZTVA' if frozen else 'ELO KEP'}  |  "
                      f"Fenyero: x{brightness:.1f} (b=valt)")
            cv2.putText(display_l, status, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
            cv2.putText(display_r, "SPACE=fagyaszt  b=fenyero  u=torol  q=kilepes",
                        (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            # Nagyító: fagyasztott állapotban segít pontosan a labda közepére kattintani.
            if frozen:
                draw_magnifier(display_l, frame_l, mouse_left.hover, scale_x, scale_y)
                draw_magnifier(display_r, frame_r, mouse_right.hover, scale_x, scale_y)

            if input_field_idx is not None:
                draw_input_box(display_l, FIELDS[input_field_idx], input_buffer)
                draw_input_box(display_r, FIELDS[input_field_idx], input_buffer)

            cv2.imshow(win_left, display_l)
            cv2.imshow(win_right, display_r)

            key = cv2.waitKey(20) & 0xFF

            if input_field_idx is not None:
                # --- Beviteli mód: szám/ENTER/BACKSPACE/ESC ---
                if key in (13, 10):  # ENTER
                    try:
                        value = float(input_buffer)
                    except ValueError:
                        logger.warning("Érvénytelen szám: '%s' – próbáld újra.", input_buffer)
                        input_buffer = ""
                        continue
                    input_values[FIELDS[input_field_idx]] = value
                    input_buffer = ""
                    input_field_idx += 1
                    if input_field_idx >= len(FIELDS):
                        append_point_and_save(
                            points, output_path,
                            uL=pending_left_src[0], vL=pending_left_src[1],
                            uR=pending_right_src[0], vR=pending_right_src[1],
                            x_mm=input_values["X"], y_mm=input_values["Y"], z_mm=input_values["Z"],
                        )
                        pending_left_src, pending_right_src = None, None
                        input_field_idx = None
                        input_values = {}
                elif key == 27:  # ESC – pont megszakítása
                    logger.info("Pont megszakítva.")
                    pending_left_src, pending_right_src = None, None
                    input_field_idx = None
                    input_values = {}
                    input_buffer = ""
                elif key in (8, 127):  # BACKSPACE
                    input_buffer = input_buffer[:-1]
                elif key != 255 and chr(key) in "0123456789.-":
                    input_buffer += chr(key)
                continue

            # --- Normál mód: SPACE/b/u/q ---
            if key == ord(' '):
                frozen = not frozen
            elif key == ord('b'):
                brightness_idx = (brightness_idx + 1) % len(BRIGHTNESS_LEVELS)
            elif key == ord('u'):
                if points:
                    removed = points.pop()
                    output_path.write_text(
                        json.dumps({"points": points}, indent=2, ensure_ascii=False), encoding="utf-8"
                    )
                    logger.info("Utolsó mentett pont törölve és fájl frissítve: %s", removed)
            elif key == ord('q'):
                break
    finally:
        cam_manager.close()
        cv2.destroyAllWindows()

    if len(points) < 3:
        logger.warning("Csak %d pont lett felvéve (min. 3 kell a calibrate_world.py-hoz).", len(points))
    logger.info("Kész. Végleges fájl: %s (%d pont)", output_path, len(points))


if __name__ == "__main__":
    main()
