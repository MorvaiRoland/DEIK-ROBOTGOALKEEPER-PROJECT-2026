"""
DEIK Robot Foci Kapus – Kamera Manager (Szinkronizált Dual-Camera Kezelő)
=========================================================================

Ez a modul koordinálja a bal és jobb oldali kamerák párhuzamos működését.

Feladatai:
    - Mindkét kamera párhuzamos megnyitása (ThreadPoolExecutor)
    - Szinkronizált frame-párok megszerzése (szoftver szinkron)
    - Kamera típus alapján példányosítás (Ximea / Mock)
    - Rendszer szintű FPS és állapot monitoring

Megjegyzés a szinkronizálásról:
    A szoftver szinkronizáció (mindkét kamerától egymás után olvasunk)
    ~0.5-2 ms eltérést okozhat a két frame között. 100 FPS-en egy labda
    8000 mm/s sebességgel kb. 4-16 mm-t tesz meg ennyi idő alatt – ez
    elfogadható a rendszer pontosságához.
    Ha jobb szinkronizálás kell: hardver trigger (Ximea GPIO).
"""

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from camera.base_camera import BaseCamera, CameraFrame
from camera.ximea_camera import XimeaCamera
from camera.mock_camera import MockCamera

logger = logging.getLogger(__name__)


@dataclass
class StereoPair:
    """
    Egy szinkronizált sztereó képpárt tárol.

    Attributes:
        left:      Bal kamera frame-je
        right:     Jobb kamera frame-je
        timestamp: A pár megszerzési ideje (UNIX timestamp)
        pair_id:   Monoton növekvő azonosító
        success:   True ha mindkét frame sikeresen megszerzett
    """
    left: CameraFrame
    right: CameraFrame
    timestamp: float
    pair_id: int
    success: bool


class CameraManager:
    """
    Koordinálja a bal és jobb oldali kamerák párhuzamos működését.

    A CameraManager felelős:
        - A megfelelő kamera osztály példányosításáért (Ximea / Mock)
        - A két kamera párhuzamos megnyitásáért
        - Szinkronizált frame-párok lekéréséért
        - Erőforrások tiszta felszabadításáért

    Example:
        manager = CameraManager(config)
        if manager.open():
            pair = manager.read_stereo_pair()
            if pair.success:
                show(pair.left.image, pair.right.image)
        manager.close()
    """

    def __init__(self, config: dict):
        """
        Args:
            config: A system_config.yaml teljes tartalma (dict)
        """
        self._config = config
        self._cam_config = config["camera"]

        self._cam_left: Optional[BaseCamera] = None
        self._cam_right: Optional[BaseCamera] = None

        self._pair_count: int = 0
        self._is_open: bool = False

        # Kombinált FPS mérés
        self._last_pair_time: float = 0.0
        self._measured_fps: float = 0.0
        self._fps_alpha: float = 0.1  # EMA simítás

        logger.info("CameraManager inicializálva (típus: %s)", self._cam_config["type"])

    # ------------------------------------------------------------------
    # Kamera életciklus
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """
        Megnyitja mindkét kamerát párhuzamosan.

        A párhuzamos megnyitás azért fontos, mert a Ximea SDK USB
        inicializációja ~2-3 másodpercig tarthat kameránként. Párhuzamosan
        ez csak egyszer telik el.

        Returns:
            True ha MINDKÉT kamera sikeresen megnyitva.
        """
        logger.info("Mindkét kamera párhuzamos megnyitása...")

        # Kamera objektumok létrehozása a típus alapján
        self._cam_left = self._create_camera(is_left=True)
        self._cam_right = self._create_camera(is_left=False)

        # Ximea SDK esetén a kamerák megnyitását egymás után (szekvenciálisan) kell végezni,
        # különben a C++ xiAPI driver mutex ütközést (Error 57) ad.
        ok_left = self._cam_left.open()
        ok_right = self._cam_right.open()

        if not ok_left:
            logger.error("Bal kamera megnyitása SIKERTELEN")
        if not ok_right:
            logger.error("Jobb kamera megnyitása SIKERTELEN")

        if ok_left and ok_right:
            self._is_open = True
            self._last_pair_time = time.perf_counter()
            logger.info("✓ Mindkét kamera sikeresen megnyitva")
            return True

        # Ha valamelyik sikertelen, a másikat is bezárjuk
        self.close()
        return False

    def close(self) -> None:
        """Bezárja mindkét kamerát és felszabadítja az erőforrásokat."""
        logger.info("Kamerák bezárása...")

        if self._cam_left:
            self._cam_left.close()
            self._cam_left = None

        if self._cam_right:
            self._cam_right.close()
            self._cam_right = None

        self._is_open = False
        logger.info("✓ Mindkét kamera bezárva")

    # ------------------------------------------------------------------
    # Frame olvasás
    # ------------------------------------------------------------------

    def read_stereo_pair(self) -> StereoPair:
        """
        Olvas egy szinkronizált frame-párt mindkét kamerából.

        A két kamera frame-je párhuzamosan kerül megszerzésre a minimális
        időeltérés érdekében. Az aktuális frame-pár azonnali elérhető
        (nem blokkolódik).

        Returns:
            StereoPair: A bal és jobb kamera legfrissebb frame-je.
        """
        if not self._is_open or not self._cam_left or not self._cam_right:
            logger.warning("read_stereo_pair() hívás zárt kamerán!")
            empty = CameraFrame(success=False)
            return StereoPair(
                left=empty, right=empty,
                timestamp=time.perf_counter(),
                pair_id=self._pair_count,
                success=False
            )

        # Párhuzamos frame-olvasás (minimális időeltérés)
        frames = [None, None]

        def _read_left():
            frames[0] = self._cam_left.read()

        def _read_right():
            frames[1] = self._cam_right.read()

        t_left = threading.Thread(target=_read_left, daemon=True)
        t_right = threading.Thread(target=_read_right, daemon=True)
        t_left.start()
        t_right.start()
        t_left.join()
        t_right.join()

        frame_left: CameraFrame = frames[0]
        frame_right: CameraFrame = frames[1]

        # FPS mérés
        now = time.perf_counter()
        if self._last_pair_time > 0:
            dt = now - self._last_pair_time
            instant_fps = 1.0 / max(dt, 1e-6)
            self._measured_fps = (
                (1.0 - self._fps_alpha) * self._measured_fps +
                self._fps_alpha * instant_fps
            )
        self._last_pair_time = now
        self._pair_count += 1

        success = frame_left.success and frame_right.success

        return StereoPair(
            left=frame_left,
            right=frame_right,
            timestamp=now,
            pair_id=self._pair_count,
            success=success,
        )

    # ------------------------------------------------------------------
    # Kamera factory metódus
    # ------------------------------------------------------------------

    def _create_camera(self, is_left: bool) -> BaseCamera:
        """
        Létrehozza a megfelelő kamera objektumot a konfig alapján.

        Args:
            is_left: True = bal kamera, False = jobb kamera

        Returns:
            BaseCamera: A példányosított kamera objektum
        """
        cam_type = self._cam_config["type"].lower()
        side = "bal" if is_left else "jobb"

        if cam_type == "ximea":
            # Ximea index: 0 = bal, 1 = jobb
            index = 0 if is_left else 1
            cam_specific = self._cam_config.get("left", {}) if is_left else self._cam_config.get("right", {})
            merged_config = self._cam_config.copy()
            merged_config.update(cam_specific)
            
            sn = merged_config.get("serial_number")
            if not sn:
                sn = self._cam_config.get("serial_number_left") if is_left else self._cam_config.get("serial_number_right")
                
            logger.info("Ximea kamera létrehozása (%s, index=%d, sn=%s)", side, index, sn)
            return XimeaCamera(
                camera_index=index,
                is_left=is_left,
                config=merged_config,
                serial_number=sn,
            )

        elif cam_type in ("mock", "webcam"):
            # Mock forrás konfig
            mock_cfg = self._cam_config.get("mock", {})
            if is_left:
                source = mock_cfg.get("left_source", 0)
            else:
                source = mock_cfg.get("right_source", 1)

            logger.info("Mock kamera létrehozása (%s, forrás='%s')", side, source)
            return MockCamera(
                source=source,
                is_left=is_left,
                config=self._cam_config,
            )

        else:
            # Ismeretlen típus → mock módba esünk vissza
            logger.warning(
                "Ismeretlen kamera típus: '%s'. Mock módra váltok.", cam_type
            )
            source = 0 if is_left else 1
            return MockCamera(
                source=source,
                is_left=is_left,
                config=self._cam_config,
            )

    # ------------------------------------------------------------------
    # Property-k és monitoring
    # ------------------------------------------------------------------

    @property
    def is_open(self) -> bool:
        """True ha mindkét kamera aktív."""
        return self._is_open

    @property
    def measured_fps(self) -> float:
        """A mért sztereó frame rate (frame-pár per másodperc)."""
        return self._measured_fps

    @property
    def pair_count(self) -> int:
        """Az eddig megszerzett frame-párok száma."""
        return self._pair_count

    def get_camera_status(self) -> dict:
        """
        Visszaadja mindkét kamera állapotát monitoring célokra.

        Returns:
            Dict tartalmazza: fps_left, fps_right, temp_left, temp_right, pair_count
        """
        return {
            "fps_left": self._cam_left.get_fps() if self._cam_left else 0.0,
            "fps_right": self._cam_right.get_fps() if self._cam_right else 0.0,
            "temp_left": self._cam_left.get_temperature() if self._cam_left else 0.0,
            "temp_right": self._cam_right.get_temperature() if self._cam_right else 0.0,
            "pair_fps": self._measured_fps,
            "pair_count": self._pair_count,
            "is_open": self._is_open,
        }

    # ------------------------------------------------------------------
    # Kamera valós idejű vezérlése
    # ------------------------------------------------------------------

    def set_camera_exposure(self, is_left: bool, exposure_us: int) -> None:
        cam = self._cam_left if is_left else self._cam_right
        if cam and hasattr(cam, "set_exposure"):
            cam.set_exposure(exposure_us)

    def set_camera_gain(self, is_left: bool, gain_db: float) -> None:
        cam = self._cam_left if is_left else self._cam_right
        if cam and hasattr(cam, "set_gain"):
            cam.set_gain(gain_db)

    def set_camera_awb(self, is_left: bool, enabled: bool) -> None:
        cam = self._cam_left if is_left else self._cam_right
        if cam and hasattr(cam, "set_awb"):
            cam.set_awb(enabled)

    def set_camera_wb(self, is_left: bool, kr: float, kg: float, kb: float) -> None:
        cam = self._cam_left if is_left else self._cam_right
        if cam and hasattr(cam, "set_wb"):
            cam.set_wb(kr, kg, kb)

    # ------------------------------------------------------------------
    # Context manager támogatás
    # ------------------------------------------------------------------

    def __enter__(self) -> "CameraManager":
        if not self.open():
            raise RuntimeError("Kamerák megnyitása sikertelen!")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
