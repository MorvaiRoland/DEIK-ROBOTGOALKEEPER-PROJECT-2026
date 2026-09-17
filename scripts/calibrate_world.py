"""
DEIK Robot Foci Kapus – Világ-koordináta (Talajsík) Kalibráló
=============================================================

Miért kell ez?
    A sztereó kalibráció (calibrate_stereo.py / GUI varázsló) csak a kamerák
    egymáshoz képesti geometriáját (K, D, R, T) adja meg. A háromszögelés így a
    REKTIFIKÁLT BAL KAMERA keretében ad 3D pontot – ez NEM a kapu-koordináta.

    Ahhoz, hogy a talajon lévő labda tényleg Y≈0-t adjon, és a mélység változása
    ne "billentse" a magasságot, egy MÉRT, merev világ-transzformáció kell
    (R_world_cam, t_world_cam). Ezt ez a szkript állítja elő ismert talajpontokból,
    Kabsch-illesztéssel, majd beírja a stereo_calibration.npz fájlba.

Munkafolyamat:
    1. Készíts sablont:
         python scripts/calibrate_world.py --template points_template.json
    2. Helyezz a pályára több jól detektálható referenciapontot (pl. labda),
       MÉRD meg a valós kapu-koordinátáikat (X jobbra+, Y felfelé+, Z pályára+),
       és olvasd le a két kamera képén a pixelkoordinátáikat (uL,vL,uR,vR).
       Ajánlott: 6–10 pont, változó X és Z (mélység) értékekkel, mind a talajon
       (Y=0) + néhány ismert magasságú pont.
    3. Töltsd ki a JSON-t, majd:
         python scripts/calibrate_world.py --points points.json
    4. A szkript kiírja az illesztési hibát (RMS/max mm) és elmenti a transzformációt.

A JSON formátuma:
    {
      "points": [
        {"uL": 812.0, "vL": 640.0, "uR": 470.0, "vR": 642.0, "X": 0.0,    "Y": 0.0, "Z": 0.0},
        {"uL": ...,   "vL": ...,   "uR": ...,   "vR": ...,   "X": 1000.0, "Y": 0.0, "Z": 2000.0}
      ]
    }
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import yaml

# Projekt src elérési út
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from stereo.triangulator import StereoTriangulator  # noqa: E402

logger = logging.getLogger(__name__)

_REQUIRED_KEYS = ("uL", "vL", "uR", "vR", "X", "Y", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Világ-koordináta (talajsík) kalibráló")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "config" / "config.yaml"))
    parser.add_argument("--points", default=None, help="Pont-korrespondenciák JSON fájlja")
    parser.add_argument("--calibration", default=None,
                        help="stereo_calibration.npz útja (alapból a config-ból)")
    parser.add_argument("--output", default=None,
                        help="Kimeneti .npz (alapból felülírja a bemenetit)")
    parser.add_argument("--template", default=None,
                        help="Sablon JSON kiírása a megadott útra, majd kilépés")
    return parser.parse_args()


def write_template(path: Path) -> None:
    template = {
        "_comment": (
            "Töltsd ki a 'points' listát. Legalább 3 (ajánlott 6-10) pont kell. "
            "uL,vL = bal kép pixel; uR,vR = jobb kép pixel; X,Y,Z = mért kapu-koord. mm "
            "(X jobbra+, Y felfelé+, Z pályára+). Talajon lévő pontnál Y=0."
        ),
        "points": [
            {"uL": 0.0, "vL": 0.0, "uR": 0.0, "vR": 0.0, "X": 0.0, "Y": 0.0, "Z": 0.0},
            {"uL": 0.0, "vL": 0.0, "uR": 0.0, "vR": 0.0, "X": 1000.0, "Y": 0.0, "Z": 2000.0},
            {"uL": 0.0, "vL": 0.0, "uR": 0.0, "vR": 0.0, "X": -1000.0, "Y": 0.0, "Z": 4000.0},
        ],
    }
    path.write_text(json.dumps(template, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Sablon kiírva: %s", path)


def load_points(path: Path) -> List[dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    pts = raw["points"] if isinstance(raw, dict) else raw
    if not isinstance(pts, list) or len(pts) < 3:
        raise ValueError("Legalább 3 pont-korrespondencia szükséges a JSON-ban.")
    for i, p in enumerate(pts):
        missing = [k for k in _REQUIRED_KEYS if k not in p]
        if missing:
            raise ValueError(f"#{i + 1} pontból hiányzik: {missing}")
    return pts


def triangulate_points(
    triangulator: StereoTriangulator, pts: List[dict]
) -> Tuple[np.ndarray, np.ndarray]:
    cam_rect: List[np.ndarray] = []
    world: List[np.ndarray] = []
    for i, p in enumerate(pts):
        p_rect = triangulator.triangulate_rect((float(p["uL"]), float(p["vL"])),
                                               (float(p["uR"]), float(p["vR"])))
        if p_rect is None:
            logger.warning("#%d pont háromszögelése sikertelen (kihagyva).", i + 1)
            continue
        cam_rect.append(p_rect)
        world.append(np.array([float(p["X"]), float(p["Y"]), float(p["Z"])], dtype=np.float64))
    if len(cam_rect) < 3:
        raise ValueError(f"Túl kevés érvényes pont a háromszögelés után: {len(cam_rect)} (min. 3).")
    return np.array(cam_rect), np.array(world)


def save_transform(npz_path: Path, output_path: Path, R: np.ndarray, t: np.ndarray) -> None:
    data = dict(np.load(str(npz_path)))
    data["R_world_cam"] = R.astype(np.float64)
    data["t_world_cam"] = t.astype(np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(str(output_path), **data)
    logger.info("Világ-transzformáció mentve: %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = parse_args()

    if args.template:
        write_template(Path(args.template))
        return

    if not args.points:
        logger.error("Adj meg --points JSON-t, vagy --template sablonkészítéshez.")
        sys.exit(1)

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cal_file = args.calibration or config.get("stereo", {}).get(
        "calibration_file", "data/calibration/stereo_calibration.npz"
    )
    cal_path = Path(cal_file)
    if not cal_path.is_absolute():
        cal_path = PROJECT_ROOT / cal_path
    if not cal_path.exists():
        logger.error("Kalibrációs fájl nem található: %s\nFuttasd előbb a sztereó kalibrálást!", cal_path)
        sys.exit(1)

    triangulator = StereoTriangulator(config)
    if not triangulator.load_calibration(str(cal_path)):
        logger.error("Sztereó kalibráció betöltése sikertelen – nem lehet háromszögelni.")
        sys.exit(1)

    pts = load_points(Path(args.points))
    cam_rect, world = triangulate_points(triangulator, pts)

    result = triangulator.calibrate_world_transform(cam_rect, world)

    logger.info("=" * 55)
    logger.info("VILÁG-KALIBRÁCIÓ KÉSZ (%d pont)", len(cam_rect))
    logger.info("  Illesztési hiba: RMS=%.1f mm, max=%.1f mm", result["rms_mm"], result["max_mm"])
    if result["rms_mm"] > 50.0:
        logger.warning("  RMS > 50 mm – ellenőrizd a mért koordinátákat és a pixelleolvasást!")
    logger.info("=" * 55)

    output_path = Path(args.output) if args.output else cal_path
    save_transform(cal_path, output_path, result["R_world_cam"], result["t_world_cam"])
    logger.info("Kész. Indítsd újra a fő alkalmazást a pontos X/Y/Z értékekhez.")


if __name__ == "__main__":
    main()
