"""
DEIK Robot Foci Kapus – YOLOv10n Modell Letöltő
=================================================

Letölti a YOLOv10n pre-trained modellt az Ultralytics szerverről
és elhelyezi a models/ mappába.

Futtatás:
    python scripts/download_model.py
"""

import logging
import sys
from pathlib import Path

# Projekt gyökér
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def download_yolov10n(output_dir: Path) -> bool:
    """
    Letölti a YOLOv10n modellt.

    Args:
        output_dir: Célmappa a modell fájlnak

    Returns:
        True ha a letöltés sikeres
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    target_path = output_dir / "yolov10n.pt"

    if target_path.exists():
        size_mb = target_path.stat().st_size / 1e6
        logger.info("✓ YOLOv10n már megvan: %s (%.1f MB)", target_path, size_mb)
        return True

    logger.info("YOLOv10n letöltése az Ultralytics szerverről...")
    logger.info("(Ez néhány MB letöltés, kérlek várj...)")

    try:
        from ultralytics import YOLO

        # Ultralytics automatikusan letölti, ha nincs meg
        # A letöltött fájl a ~/.ultralytics/assets/ mappába kerül
        model = YOLO("yolov10n.pt")

        # Megkeressük és átmásoljuk a models/ mappába
        import shutil

        # Lehetséges helyek ahol az Ultralytics letölti
        possible_paths = [
            Path("yolov10n.pt"),
            Path.home() / ".ultralytics" / "assets" / "yolov10n.pt",
        ]

        for src in possible_paths:
            if src.exists():
                shutil.copy2(str(src), str(target_path))
                size_mb = target_path.stat().st_size / 1e6
                logger.info("✓ Modell mentve: %s (%.1f MB)", target_path, size_mb)
                return True

        # Ha nem találjuk, mentsük el közvetlenül
        model.save(str(target_path))
        logger.info("✓ Modell mentve: %s", target_path)
        return True

    except ImportError:
        logger.error("Az 'ultralytics' csomag nincs telepítve!")
        logger.error("Telepítés: pip install ultralytics")
        return False
    except Exception as exc:
        logger.error("Letöltési hiba: %s", exc)
        return False


def main() -> None:
    """Főprogram."""
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   DEIK Robot Foci Kapus – Modell Letöltő             ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    models_dir = PROJECT_ROOT / "models"
    success = download_yolov10n(models_dir)

    if success:
        print()
        print("Letöltött modellek a models/ mappában:")
        for f in sorted(models_dir.iterdir()):
            if f.suffix in (".pt", ".onnx", ".engine"):
                size_mb = f.stat().st_size / 1e6
                print(f"  {f.name:<30} {size_mb:.1f} MB")
        print()
        print("✓ Modell letöltés kész! Indítsd el a fő programot:")
        print("  python src/main.py --mock")
    else:
        print()
        print("✗ Letöltés sikertelen. Ellenőrizd az internetkapcsolatot!")
        sys.exit(1)


if __name__ == "__main__":
    main()
