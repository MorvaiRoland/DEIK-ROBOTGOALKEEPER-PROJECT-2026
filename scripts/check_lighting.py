"""
DEIK Robot Foci Kapus – Megvilágítás-ellenőrző
===============================================

Három dolgot mér a főprogrammal AZONOS kamerabeállításokkal
(config.yaml + config/gui_settings.json):

  1. Villódzás: a képkockák átlagos fényességének ingadozása (50 Hz-es hálózatnál
     a lámpák 100 Hz-cel pulzálnak; 10 ms-nál rövidebb expozíciónál ez látszik).
  2. HSV "narancs" háttér: a jelenlegi ball.hsv_filter küszöbökön átjutó pixelek
     aránya + kép mentése (data/lighting_check_left.png / _right.png, lila = átjut).
  3. Fehéregyensúly: a kép KÖZEPÉRE tartott fehér/szürke lapból (1.0-s erősítésekkel)
     kiszámolja a kr/kb értékeket a gui_settings.json-be. Az AWB-t szándékosan nem
     használja: az a teljes kép átlagából dolgozik, így a két kamera a saját látványától
     függően eltérő színeket ad ugyanarra a tárgyra.

Futtatás (a főprogram legyen BEZÁRVA, mert a kamerákat kizárólagosan nyitja):
    python scripts/check_lighting.py
    python scripts/check_lighting.py --exposure 10000        # villódzás-teszt 10 ms-mal
    python scripts/check_lighting.py --exposure 3000 --gain 6
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from camera.camera_manager import CameraManager  # noqa: E402

FLICKER_STEP_LIMIT_PCT = 1.0  # képkockánkénti átlagos fényesség-ugrás e felett = villódzás


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Megvilágítás-ellenőrző (villódzás, HSV, WB)")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--gui-settings", default=str(PROJECT_ROOT / "config" / "gui_settings.json"))
    parser.add_argument("--frames", type=int, default=180, help="Mérési képpárok száma")
    parser.add_argument("--exposure", type=int, default=None, help="Expozíció felülírása (µs)")
    parser.add_argument("--gain", type=float, default=None, help="Erősítés felülírása (dB)")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict:
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    gui_path = Path(args.gui_settings)
    if gui_path.exists():
        gui_cfg = json.loads(gui_path.read_text(encoding="utf-8"))
        for side in ("left", "right"):
            config["camera"].setdefault(side, {}).update(gui_cfg.get(side, {}))
    for side in ("left", "right"):
        if args.exposure is not None:
            config["camera"][side]["exposure_time_us"] = args.exposure
        if args.gain is not None:
            config["camera"][side]["gain_db"] = args.gain
    return config


def grab_pairs(cam: CameraManager, n: int, timeout_s: float = 20.0) -> list:
    pairs, t0 = [], time.monotonic()
    while len(pairs) < n and time.monotonic() - t0 < timeout_s:
        pair = cam.read_stereo_pair()
        if pair.success:
            pairs.append((pair.left.image.copy(), pair.right.image.copy()))
        else:
            time.sleep(0.002)
    return pairs


def hsv_overlay(img: np.ndarray, hsv_cfg: dict) -> tuple:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([hsv_cfg.get("h_min", 5), hsv_cfg.get("s_min", 75), hsv_cfg.get("v_min", 60)], np.uint8)
    upper = np.array([hsv_cfg.get("h_max", 25), hsv_cfg.get("s_max", 255), hsv_cfg.get("v_max", 255)], np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    out = (img * 0.45).astype(np.uint8)
    out[mask > 0] = (255, 0, 255)
    return out, 100.0 * np.count_nonzero(mask) / mask.size


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    args = parse_args()
    config = load_config(args)
    hsv_cfg = config.get("ball", {}).get("hsv_filter", {})

    cam = CameraManager(config)
    if not cam.open():
        print("HIBA: a kamerák nem nyithatók meg (fut még a főprogram?)")
        sys.exit(1)

    try:
        for side in ("left", "right"):
            c = config["camera"][side]
            print(f"{side.upper():5s}: expozíció={c.get('exposure_time_us')} µs, "
                  f"erősítés={c.get('gain_db')} dB, AWB={c.get('auto_white_balance')}")

        grab_pairs(cam, 30)  # bemelegedés (AWB / trigger stabilizálódás)
        pairs = grab_pairs(cam, args.frames)
        if len(pairs) < 20:
            print(f"HIBA: csak {len(pairs)} képpár érkezett.")
            sys.exit(1)

        print(f"\n1) VILLÓDZÁS ({len(pairs)} képpár)")
        for idx, side in enumerate(("left", "right")):
            means = np.array([cv2.cvtColor(p[idx], cv2.COLOR_BGR2GRAY).mean() for p in pairs])
            step_pct = 100.0 * np.mean(np.abs(np.diff(means))) / means.mean()
            ptp_pct = 100.0 * np.ptp(means) / means.mean()
            verdict = "VILLÓDZÁS VALÓSZÍNŰ" if step_pct > FLICKER_STEP_LIMIT_PCT else "rendben"
            print(f"   {side.upper():5s}: átlag fényesség={means.mean():6.1f}, "
                  f"képkockánkénti ugrás={step_pct:4.1f}%, min-max={ptp_pct:4.1f}%  → {verdict}")

        print("\n2) HSV 'NARANCS' HÁTTÉR (a labdát vedd ki a képből!)")
        for idx, side in enumerate(("left", "right")):
            overlay, pct = hsv_overlay(pairs[-1][idx], hsv_cfg)
            out_path = PROJECT_ROOT / "data" / f"lighting_check_{side}.png"
            cv2.imwrite(str(out_path), cv2.resize(overlay, None, fx=0.5, fy=0.5))
            print(f"   {side.upper():5s}: {pct:5.2f}% pixel jut át a HSV szűrőn → {out_path}")

        print("\n3) FEHÉREGYENSÚLY fehér lapról (a lap legyen MINDKÉT kamera képének KÖZEPÉN)")
        for is_left in (True, False):
            cam.set_camera_awb(is_left, False)
            cam.set_camera_wb(is_left, 1.0, 1.0, 1.0)
        wb_pairs = grab_pairs(cam, 40, timeout_s=8.0)[10:]
        for idx, side in enumerate(("left", "right")):
            img = np.mean([p[idx].astype(np.float32) for p in wb_pairs], axis=0)
            h, w = img.shape[:2]
            y0, y1, x0, x1 = int(h * 0.4), int(h * 0.6), int(w * 0.4), int(w * 0.6)
            patch = img[y0:y1, x0:x1].reshape(-1, 3)
            unsaturated = patch[patch.max(axis=1) < 250]
            preview = img.astype(np.uint8)
            cv2.rectangle(preview, (x0, y0), (x1, y1), (0, 0, 255), 4)
            prev_path = PROJECT_ROOT / "data" / f"lighting_check_wb_{side}.png"
            cv2.imwrite(str(prev_path), cv2.resize(preview, None, fx=0.5, fy=0.5))
            if len(unsaturated) < 0.5 * len(patch):
                print(f"   {side.upper():5s}: a lap túlexponált (kiégett) → csökkentsd az expozíciót! ({prev_path})")
                continue
            b, g, r = unsaturated.mean(axis=0)
            if g < 40:
                print(f"   {side.upper():5s}: a lap túl sötét (G={g:.0f}) → nincs a kép közepén? ({prev_path})")
                continue
            print(f"   {side.upper():5s}: lap fényesség (G)={g:5.1f}  →  \"wb_kr\": {g / r:.3f}, "
                  f"\"wb_kg\": 1.0, \"wb_kb\": {g / b:.3f}   (ellenőrző kép: {prev_path})")
        print('   → Írd be a gui_settings.json "left"/"right" blokkjába, és állítsd: '
              '"auto_white_balance": false')
        print("   → Ha a két kamera lap-fényessége >15%-kal eltér, ellenőrizd az objektívek"
              " írisz- (rekesz-) gyűrűjét!")
    finally:
        cam.close()


if __name__ == "__main__":
    main()
