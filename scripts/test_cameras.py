"""
DEIK Robot Foci Kapus – Kamera Kapcsolat és Teljesítmény Teszt
===============================================================

Ez a szkript ellenőrzi:
    1. Mindkét Ximea kamera elérhetőségét
    2. A valódi FPS-t (frame rate) terhelés alatt
    3. USB3 sávszélesség elégségességét
    4. Kamera hőmérsékletet
    5. Szenzor adatait

Futtatás:
    python scripts/test_cameras.py
    python scripts/test_cameras.py --mock    # Webcam kamerával
    python scripts/test_cameras.py --duration 10  # 10 másodperc teszt
"""

import argparse
import logging
import sys
import time
from pathlib import Path

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Kamera kapcsolat és teljesítmény teszt")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0,
                        help="Teszt időtartama másodpercben (alapértelmezett: 5)")
    parser.add_argument("--show-frames", action="store_true",
                        help="Kamera képek megjelenítése (OpenCV ablak)")
    return parser.parse_args()


def print_header() -> None:
    """Fejléc kiírása."""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   DEIK Robot Foci Kapus – Kamera Teszt               ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def test_cameras(config: dict, duration_s: float, show_frames: bool) -> bool:
    """
    Elvégzi a kamera tesztet és kiírja az eredményeket.

    Args:
        config:       Konfiguráció dict
        duration_s:   Teszt időtartama másodpercben
        show_frames:  Ha True, megjeleníti a kamera képeket

    Returns:
        True ha minden teszt sikeresen lefutott
    """
    # ── Kamera megnyitás teszt ────────────────────────────────────────
    print("[1/4] Kamerák megnyitása...")
    cam_manager = CameraManager(config)

    open_start = time.perf_counter()
    success = cam_manager.open()
    open_time = time.perf_counter() - open_start

    if not success:
        print("  ✗ HIBA: Kamerák megnyitása SIKERTELEN!")
        print("  Ellenőrizd az USB3 kapcsolatokat (EP-USB3HybridcableU-20).")
        print("  lsusb | grep -i ximea  # USB eszköz azonosítás")
        return False

    print(f"  ✓ Kamerák megnyitva ({open_time:.1f} s)")

    # ── Első frame teszt ─────────────────────────────────────────────
    print("[2/4] Első frame olvasása...")
    pair = None
    for _ in range(20):  # Max 2 mp várakozás az első frame megérkezésére
        pair = cam_manager.read_stereo_pair()
        if pair.success:
            break
        time.sleep(0.1)

    if pair is None or not pair.success:
        print("  ✗ HIBA: Nem sikerült frame-et olvasni!")
        cam_manager.close()
        return False

    left = pair.left.image
    right = pair.right.image
    h_l, w_l = left.shape[:2]
    h_r, w_r = right.shape[:2]

    print(f"  ✓ Bal kamera frame:  {w_l}×{h_l} px")
    print(f"  ✓ Jobb kamera frame: {w_r}×{h_r} px")

    # ── FPS teszt ────────────────────────────────────────────────────
    print(f"[3/4] FPS mérés ({duration_s:.0f} másodpercig)...")

    if show_frames:
        cv2.namedWindow("Kamera Teszt", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Kamera Teszt", 1280, 400)

    frame_count = 0
    t_start = time.perf_counter()
    fps_measurements = []

    fps_window_start = t_start
    fps_window_count = 0

    while (time.perf_counter() - t_start) < duration_s:
        pair = cam_manager.read_stereo_pair()
        if pair.success:
            frame_count += 1
            fps_window_count += 1

        # FPS mérés 0.5 s ablakokban
        now = time.perf_counter()
        if now - fps_window_start >= 0.5:
            window_fps = fps_window_count / (now - fps_window_start)
            fps_measurements.append(window_fps)
            fps_window_start = now
            fps_window_count = 0

            # Haladás kiírása
            elapsed = now - t_start
            remaining = duration_s - elapsed
            print(f"\r  Mért FPS: {window_fps:.1f}  |  "
                  f"Frame-ek: {frame_count}  |  "
                  f"Hátralévő: {remaining:.1f} s", end="", flush=True)

        # Megjelenítés
        if show_frames and pair.success:
            h_disp = 380
            display = np.hstack([
                cv2.resize(pair.left.image, (int(h_disp * w_l / h_l), h_disp)),
                cv2.resize(pair.right.image, (int(h_disp * w_r / h_r), h_disp)),
            ])
            cv2.putText(display, f"FPS: {window_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("Kamera Teszt", display)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    print()

    t_total = time.perf_counter() - t_start
    avg_fps = frame_count / t_total if t_total > 0 else 0
    min_fps = min(fps_measurements) if fps_measurements else 0
    max_fps = max(fps_measurements) if fps_measurements else 0

    print(f"  ✓ FPS eredmények:")
    print(f"    Átlagos FPS: {avg_fps:.1f}")
    print(f"    Minimum FPS: {min_fps:.1f}")
    print(f"    Maximum FPS: {max_fps:.1f}")
    print(f"    Összes frame: {frame_count}")

    # FPS értékelés
    target_fps = float(config["camera"].get("fps", 100))
    fps_ok = avg_fps >= target_fps * 0.85  # 15%-os tolerancia
    if fps_ok:
        print(f"  ✓ FPS rendben (cél: {target_fps:.0f} FPS, elért: {avg_fps:.1f} FPS)")
    else:
        print(f"  ⚠ FPS alacsony! (cél: {target_fps:.0f} FPS, elért: {avg_fps:.1f} FPS)")
        print("    → Csökkentsd a bandwidth_limit_mbs értéket a config-ban")
        print("    → Ellenőrizd, hogy mindkét kamera külön USB vezérlőn van-e")

    # ── Kamera hőmérséklet ───────────────────────────────────────────
    print("[4/4] Kamera hőmérsékletek...")
    status = cam_manager.get_camera_status()
    temp_l = status.get("temp_left", 0.0)
    temp_r = status.get("temp_right", 0.0)

    if temp_l > 0:
        print(f"  Bal kamera hőmérséklet: {temp_l:.1f}°C", end="")
        print(" ⚠ MELEG!" if temp_l > 60 else " ✓")
    else:
        print("  Bal kamera hőmérséklet: N/A (mock vagy nem elérhető)")

    if temp_r > 0:
        print(f"  Jobb kamera hőmérséklet: {temp_r:.1f}°C", end="")
        print(" ⚠ MELEG!" if temp_r > 60 else " ✓")
    else:
        print("  Jobb kamera hőmérséklet: N/A (mock vagy nem elérhető)")

    if show_frames:
        cv2.destroyAllWindows()

    cam_manager.close()

    # ── Összegzés ────────────────────────────────────────────────────
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print(f"║  TESZT EREDMÉNY:  {'SIKERES ✓' if fps_ok else 'FIGYELMEZTETÉS ⚠'}                             ║")
    fps_str = f"║  Átlagos FPS: {avg_fps:.1f}"
    print(f"{fps_str:<55}║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    return fps_ok


def check_cuda() -> None:
    """CUDA/GPU elérhetőség ellenőrzése."""
    print("[GPU] CUDA elérhetőség ellenőrzése...")
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"  ✓ GPU: {name} ({vram:.1f} GB VRAM)")
            # Gyors CUDA teszt
            dummy = torch.zeros(1, device="cuda")
            del dummy
            print("  ✓ CUDA tensor allokáció: OK")
        else:
            print("  ⚠ CUDA nem elérhető – a detektálás CPU-n fut (lassabb)")
    except ImportError:
        print("  ⚠ PyTorch nincs telepítve!")
    print()


def check_ximea() -> None:
    """Ximea SDK elérhetőség ellenőrzése."""
    print("[SDK] Ximea xiAPI ellenőrzése...")
    try:
        from ximea import xiapi
        print("  ✓ Ximea xiAPI elérhető")

        # Csatlakozott kamerák száma
        try:
            cam = xiapi.Camera()
            num_cams = cam.get_number_devices()
            print(f"  ✓ Csatlakozott Ximea kamerák száma: {num_cams}")
            if num_cams < 2:
                print(f"  ⚠ FIGYELEM: Csak {num_cams} kamera látható (2 szükséges)")
        except Exception as exc:
            print(f"  ⚠ Kamera felsorolás hiba: {exc}")

    except ImportError:
        print("  ⚠ Ximea xiAPI nem található!")
        print("    Telepítsd a Ximea Linux SDK-t, majd:")
        print("    cd /opt/XIMEA/api/Python/v3 && sudo python3 setup.py install")
    print()


def main() -> None:
    """Főprogram."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.mock:
        config["camera"]["type"] = "mock"

    print_header()

    # Rendszer ellenőrzések
    check_cuda()
    check_ximea()

    # Kamera teszt
    success = test_cameras(config, args.duration, args.show_frames)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
