"""
DEIK Robot Foci Kapus – Belépési Pont (Entry Point)
====================================================

Futtatás:
    python src/main.py                    # Ximea kamerákkal (valódi hardware)
    python src/main.py --mock             # Mock kamerával (fejlesztés/teszt)
    python src/main.py --mock --no-gui    # Fejléc nélküli (headless) mód

Ez a fájl:
    1. Beolvassa a parancssori argumentumokat
    2. Betölti a konfigurációt (config/config.yaml)
    3. Beállítja a naplózást
    4. Elindítja a PyQt6 GUI-t (vagy headless módot)
"""

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

# Disable Qt QPA portal DBus registration warnings and third-party deprecation warnings
os.environ["QT_LOGGING_RULES"] = "qt.qpa.services.warning=false;qt.qpa.*=false"
os.environ["QT_NO_DESKTOP_PORTAL"] = "1"

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*pynvml.*")

# ── Projekt gyökér hozzáadása a Python elérési úthoz ────────────────────────
# Ez szükséges, hogy az abszolút importok működjenek
# (pl. "from camera.camera_manager import CameraManager")
PROJECT_ROOT = Path(__file__).parent.parent
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml


def parse_arguments() -> argparse.Namespace:
    """
    Parancssori argumentumok elemzése.

    Returns:
        Névtér az elemzett argumentumokkal
    """
    parser = argparse.ArgumentParser(
        description="DEIK Robot Foci Kapus – Valós Idejű Labdadetektor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Példák:
  python src/main.py                    # Normál indítás (Ximea kamerák)
  python src/main.py --mock             # Mock/webcam kamera módban
  python src/main.py --config config/my_config.yaml  # Egyedi konfig
        """,
    )

    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config" / "config.yaml"),
        help="Konfiguráció fájl elérési útja (alapértelmezett: config/config.yaml)"
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Mock kamera mód (webcam/szintetikus) Ximea hardver nélkül"
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Fejléc nélküli (headless) mód – csak konzol kimenet"
    )

    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default=None,
        help="Napló szintje (felülírja a konfig beállítást)"
    )

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """
    Betölti a YAML konfigurációt.

    Args:
        config_path: A YAML fájl elérési útja

    Returns:
        A konfiguráció dict-ként

    Raises:
        SystemExit: Ha a fájl nem található vagy érvénytelen
    """
    config_file = Path(config_path)
    if not config_file.exists():
        print(f"HIBA: Konfiguráció fájl nem található: {config_file}")
        sys.exit(1)

    try:
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        return config
    except yaml.YAMLError as exc:
        print(f"HIBA: Érvénytelen YAML konfiguráció: {exc}")
        sys.exit(1)


def setup_logging(config: dict, log_level_override: str = None) -> None:
    """
    Beállítja a Python logging rendszert.

    Args:
        config:             A konfiguráció dict (log szintet és fájlt tartalmaz)
        log_level_override: Ha megadott, felülírja a konfig szintjét
    """
    log_cfg = config.get("logging", {})
    level_str = log_level_override or log_cfg.get("level", "INFO")
    level = getattr(logging, level_str.upper(), logging.INFO)

    # Formátum beállítása
    formatter = logging.Formatter(
        fmt="%(asctime)s  %(levelname)-8s  %(name)s – %(message)s",
        datefmt="%H:%M:%S"
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Konzol handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Fájl handler (ha konfiguráltunk)
    log_file = log_cfg.get("log_file")
    if log_file:
        log_path = PROJECT_ROOT / log_file
        log_path.parent.mkdir(parents=True, exist_ok=True)

        from logging.handlers import RotatingFileHandler
        max_bytes = int(log_cfg.get("max_log_size_mb", 50)) * 1024 * 1024
        backup_count = int(log_cfg.get("backup_count", 3))

        file_handler = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding="utf-8"
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Ultralytics logolását csökkentjük (túl zajos)
    logging.getLogger("ultralytics").setLevel(logging.WARNING)

    logging.info("Naplózás beállítva: szint=%s", level_str)


def run_health_check(config: dict) -> list:
    """
    Rendszer önellenőrzés indításkor.
    Visszaad egy listát dict-ekkél: {'icon': str, 'text': str, 'level': str}
    level: 'ok' | 'warning' | 'error'
    """
    import os
    results = []

    # 1. GPU és CUDA
    try:
        # pyrefly: ignore [missing-import]
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
            results.append({
                "icon": "✓", "level": "ok",
                "text": f"GPU: {gpu_name} ({gpu_mem:.1f} GB VRAM) – CUDA {torch.version.cuda}"
            })
        else:
            results.append({
                "icon": "⚠", "level": "warning",
                "text": "CUDA nem elérhető – CPU módban fut (lassabb!)"
            })
    except ImportError:
        results.append({
            "icon": "✗", "level": "error",
            "text": "PyTorch nincs telepítve – GPU nem használható"
        })

    # 2. TensorRT / detekálási modell fájl
    model_path = config.get("detection", {}).get("model_path", "models/rtdetr-l.engine")
    full_model = str(PROJECT_ROOT / model_path)
    if os.path.exists(full_model):
        size_mb = os.path.getsize(full_model) / (1024 * 1024)
        results.append({
            "icon": "✓", "level": "ok",
            "text": f"AI modell: {model_path} ({size_mb:.0f} MB)"
        })
    else:
        results.append({
            "icon": "✗", "level": "error",
            "text": f"AI modell nem található: {model_path}"
        })

    # 3. Kalibraciós fájl
    cal_path = config.get("stereo", {}).get(
        "calibration_file", "data/calibration/stereo_calibration.npz"
    )
    full_cal = str(PROJECT_ROOT / cal_path)
    if os.path.exists(full_cal):
        results.append({
            "icon": "✓", "level": "ok",
            "text": f"Kalibració: {cal_path} megtalálva"
        })
    else:
        results.append({
            "icon": "⚠", "level": "warning",
            "text": f"Nincs kalibrációs fájl – szükséges a kalibració előtt!"
        })

    # 4. Szabad RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        avail_gb = mem.available / (1024 ** 3)
        if avail_gb >= 4.0:
            results.append({
                "icon": "✓", "level": "ok",
                "text": f"RAM: {avail_gb:.1f} GB szabad / {mem.total / (1024**3):.1f} GB összes"
            })
        elif avail_gb >= 2.0:
            results.append({
                "icon": "⚠", "level": "warning",
                "text": f"RAM: csak {avail_gb:.1f} GB szabad – ajánlott 4 GB+"
            })
        else:
            results.append({
                "icon": "✗", "level": "error",
                "text": f"RAM: kritíkusan alacsony ({avail_gb:.1f} GB szabad)!"
            })
    except Exception:
        pass

    # 5. Kamera típus
    cam_type = config.get("camera", {}).get("type", "ximea")
    if cam_type == "ximea":
        results.append({
            "icon": "✓", "level": "ok",
            "text": "Kamera: Ximea CMOS sztereó kámera mód aktiv"
        })
    else:
        results.append({
            "icon": "⚠", "level": "warning",
            "text": f"Kamera: {cam_type.upper()} mód (fejlesztési / teszt mód!)"
        })

    return results


def start_gui(config: dict) -> None:
    """
    Elindítja a PyQt6 grafikus felületet.

    Args:
        config: A konfiguráció dict
    """
    try:
        # pyrefly: ignore [missing-import]
        from PyQt6.QtWidgets import QApplication
        from gui.main_window import MainWindow
        from gui.splash_screen import SplashScreen
        from gui.theme import get_app_icon
    except ImportError as exc:
        logging.error("PyQt6 importálási hiba: %s", exc)
        logging.error("Telepítés: pip install PyQt6")
        sys.exit(1)

    app = QApplication(sys.argv)
    app.setApplicationName("DEIK Robot Foci Kapus")
    app.setApplicationVersion("1.0.0")
    app.setOrganizationName("DEIK")
    app.setDesktopFileName("deik-robotgoalkeeper")

    app_icon = get_app_icon()
    if not app_icon.isNull():
        app.setWindowIcon(app_icon)

    # Betöltő ablak (SplashScreen)
    splash = SplashScreen()
    splash.show()
    splash.run_loading_sequence()

    # Health check – önellenőrzés az indítás után
    checks = run_health_check(config)
    splash.show_health_check(checks)

    # Főablak megnyitása teljes képernyős módban
    window = MainWindow(config)
    window.showFullScreen()
    splash.close()

    logging.info("GUI elindult – kattints a INDÍTÁS gombra a kamerák aktiválásához")

    # Qt eseményciklus futtatása (blokkol amíg az ablak nyitva van)
    exit_code = app.exec()
    sys.exit(exit_code)


def start_headless(config: dict) -> None:
    """
    Fejléc nélküli (headless) mód – GUI nélkül, csak konzol kimenet.

    Hasznos teszteléshez, szerver-futtatáshoz.

    Args:
        config: A konfiguráció dict
    """
    import time
    from camera.camera_manager import CameraManager
    from detection.ball_detector import BallDetector
    from stereo.triangulator import StereoTriangulator
    from prediction.trajectory_predictor import TrajectoryPredictor

    logger = logging.getLogger(__name__)
    logger.info("Headless mód indítása...")

    # Komponensek
    cam_manager = CameraManager(config)
    detector = BallDetector(config["detection"], full_config=config)
    triangulator = StereoTriangulator(config)
    predictor = TrajectoryPredictor(config)

    cal_file = config.get("stereo", {}).get("calibration_file", "data/calibration/stereo_calibration.npz")
    triangulator.load_calibration(cal_file)

    if not cam_manager.open():
        logger.error("Kamerák megnyitása sikertelen!")
        sys.exit(1)

    logger.info("Headless feldolgozás megkezdése (Ctrl+C a leállításhoz)...")

    try:
        while True:
            pair = cam_manager.read_stereo_pair()
            if not pair.success:
                continue

            detection = detector.detect(pair.left.image, pair.right.image)

            pos_3d = None
            if detection.both_found:
                pos_3d = triangulator.triangulate(
                    (detection.left.x, detection.left.y),
                    (detection.right.x, detection.right.y),
                )

            if pos_3d is not None:
                predictor.add_measurement(*pos_3d)
                impact = predictor.get_impact_prediction()
                if impact.valid:
                    logger.info(
                        "IMPACT  X=%+.0f mm  Y=%.0f mm  T=%.3f s  Conf=%.0f%%  %s",
                        impact.x_mm, impact.y_mm, impact.time_to_impact_s,
                        impact.confidence * 100,
                        "GOOL IRÁNY!" if impact.in_goal else "mellé"
                    )

    except KeyboardInterrupt:
        logger.info("Leállítás (Ctrl+C)...")
    finally:
        cam_manager.close()


# --------------------------------------------------------------------------- #
# MAIN ENTRY POINT
# --------------------------------------------------------------------------- #

def main() -> None:
    """A program belépési pontja."""
    args = parse_arguments()

    # Konfiguráció betöltése
    config = load_config(args.config)

    # Mock mód: kamera típus felülírása
    if args.mock:
        config["camera"]["type"] = "mock"
        print("INFO: Mock kamera mód aktiválva")

    # Naplózás beállítása
    setup_logging(config, args.log_level)

    logger = logging.getLogger(__name__)
    logger.info("=" * 60)
    logger.info("DEIK Robot Foci Kapus v1.0")
    logger.info("Kamera típus: %s", config["camera"]["type"])
    logger.info("Konfiguráció: %s", args.config)
    logger.info("=" * 60)

    # Ellenőrzés: CUDA elérhető-e?
    try:
        # pyrefly: ignore [missing-import]
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            logger.info("✓ GPU: %s (%.1f GB VRAM)", gpu_name, gpu_mem)
        else:
            logger.warning("⚠ CUDA nem elérhető – CPU módban fut (lassabb!)")
            config["detection"]["device"] = "cpu"
    except ImportError:
        logger.warning("PyTorch nincs telepítve – CPU módban fut")
        config["detection"]["device"] = "cpu"

    # Indítás
    if args.no_gui:
        start_headless(config)
    else:
        start_gui(config)


if __name__ == "__main__":
    main()
