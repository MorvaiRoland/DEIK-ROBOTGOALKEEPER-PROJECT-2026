"""
DEIK Robot Foci Kapus – Mock / Teszt Kamera Implementáció
==========================================================

Ez a modul egy szimulált kamerát valósít meg fejlesztéshez és teszteléshez,
amikor a valódi Ximea kamerák nem állnak rendelkezésre.

Két üzemmód:
    1. Webcam mód:    OpenCV VideoCapture(index) – valódi webkamera
    2. Videó mód:     OpenCV VideoCapture(fájlút) – előre felvett videó
    3. Szintetikus:   Generált tesztkép (ha sem webcam, sem videó)

Jellemzők:
    - Ugyanolyan interfész mint az XimeaCamera (polimorfizmus)
    - Szimulált FPS korlát (valós kamera sebességének imitálása)
    - Videó ciklikus lejátszása (loop)
"""

import logging
import time
from typing import Optional, Union

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

from camera.base_camera import BaseCamera, CameraFrame, CameraInfo

logger = logging.getLogger(__name__)


class MockCamera(BaseCamera):
    """
    Fejlesztési/teszt kamera implementáció.

    Webcamot, videófájlt, vagy szintetikusan generált képet használ
    a valódi Ximea kamera kiváltására tesztelés közben.

    Example:
        # Webcam használata (bal kamera szimulációja)
        cam = MockCamera(source=0, is_left=True, config=cfg)
        cam.open()
        frame = cam.read()
    """

    def __init__(
        self,
        source: Union[int, str],
        is_left: bool,
        config: dict,
    ):
        """
        Args:
            source:   Kamera index (0, 1, ...) webcamhoz, vagy videófájl elérési út
            is_left:  True = bal oldali kamera szimulációja
            config:   A system_config.yaml "camera" szekciója
        """
        target_w = config["resolution"]["width"]
        target_h = config["resolution"]["height"]
        target_fps = float(config.get("fps", 30))

        side = "BAL" if is_left else "JOBB"
        cam_info = CameraInfo(
            name=f"MockCamera [{side}] ← {source}",
            width=target_w,
            height=target_h,
            fps=target_fps,
            is_left=is_left,
        )
        super().__init__(cam_info)

        self._source = source
        self._target_fps = target_fps
        self._target_w = target_w
        self._target_h = target_h
        self._loop_video = config.get("mock", {}).get("loop_video", True)

        self._cap: Optional[cv2.VideoCapture] = None
        self._last_frame_time: float = 0.0
        self._frame_interval: float = 1.0 / target_fps

        # FPS mérés
        self._fps_counter: int = 0
        self._fps_start_time: float = 0.0
        self._measured_fps: float = 0.0

        # Kezdő offset és transzformációs értékek betöltése a konfigból
        self._offset_x = int(config.get("offset_x", 0))
        self._offset_y = int(config.get("offset_y", 0))
        self._flip_h = bool(config.get("flip_h", False))
        self._flip_v = bool(config.get("flip_v", False))
        self._rotation = int(config.get("rotation", 0))

        logger.info("MockCamera létrehozva: forrás='%s', %dx%d @ %.0f FPS",
                    source, target_w, target_h, target_fps)


    def open(self) -> bool:
        """
        Megnyitja az OpenCV VideoCapture forrást.

        Returns:
            True ha a megnyitás sikeres.
        """
        logger.info("MockCamera megnyitása: %s", self._info.name)
        try:
            self._cap = cv2.VideoCapture(self._source)

            if not self._cap.isOpened():
                logger.warning(
                    "Forrás nem megnyitható: '%s'. Szintetikus módra váltok.",
                    self._source
                )
                self._cap = None
                self._is_open = True   # Szintetikus módban is "nyitva" vagyunk
                return True

            # Felbontás kérése (ha a kamera/videó támogatja)
            if isinstance(self._source, int):
                self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._target_w)
                self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._target_h)
                self._cap.set(cv2.CAP_PROP_FPS, self._target_fps)

            actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            logger.info("MockCamera kész: tényleges felbontás %dx%d", actual_w, actual_h)

            self._fps_start_time = time.perf_counter()
            self._is_open = True
            return True

        except Exception as exc:
            logger.error("MockCamera megnyitási hiba: %s", exc)
            return False

    def close(self) -> None:
        """Felszabadítja az OpenCV VideoCapture erőforrásait."""
        if self._cap:
            self._cap.release()
            self._cap = None
        self._is_open = False
        logger.info("MockCamera bezárva: %s", self._info.name)

    def read(self) -> CameraFrame:
        """
        Olvas egy frame-et a forrásból, FPS korláttal.

        Ha nincs valódi forrás (szintetikus mód), akkor egy tesztkép
        kerül visszaadásra a labda detektálás teszteléséhez.

        Returns:
            CameraFrame: A következő frame, vagy tesztkép.
        """
        if not self._is_open:
            return CameraFrame(success=False)

        # FPS korlát szimulálása (nem akarjuk a tényleges CPU-t maximálisan terhelni)
        now = time.perf_counter()
        wait = self._frame_interval - (now - self._last_frame_time)
        if wait > 0:
            time.sleep(wait)
        self._last_frame_time = time.perf_counter()

        self._frame_count += 1

        # --- 1. eset: Van OpenCV forrás ---
        if self._cap and self._cap.isOpened():
            ret, frame = self._cap.read()

            if not ret:
                # Videó végére érve újraindítjuk (loop)
                if self._loop_video and isinstance(self._source, str):
                    self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = self._cap.read()

            if ret and frame is not None:
                # Átméretezés a célzott felbontásra (ha szükséges)
                h, w = frame.shape[:2]
                if w != self._target_w or h != self._target_h:
                    frame = cv2.resize(
                        frame,
                        (self._target_w, self._target_h),
                        interpolation=cv2.INTER_LINEAR
                    )
                return self._make_frame(frame)

        # --- 2. eset: Szintetikus tesztkép ---
        synthetic = self._generate_synthetic_frame()
        return self._make_frame(synthetic)

    def _make_frame(self, image: np.ndarray) -> CameraFrame:
        """Létrehoz egy CameraFrame objektumot a mért FPS-sel."""
        # Alkalmazzuk az X/Y elmozdulást, tükrözést és elforgatást
        image = self.apply_image_transformations(image)

        # FPS mérés (minden 30 frame-enként)
        self._fps_counter += 1

        if self._fps_counter >= 30:
            elapsed = time.perf_counter() - self._fps_start_time
            self._measured_fps = self._fps_counter / elapsed
            self._fps_counter = 0
            self._fps_start_time = time.perf_counter()

        return CameraFrame(
            image=image,
            timestamp=time.perf_counter(),
            frame_id=self._frame_count,
            success=True,
        )

    def _generate_synthetic_frame(self) -> np.ndarray:
        """
        Szintetikusan generál egy tesztképet fehér focilabdával.

        A labda szinuszos pályán mozog, így a detektálás tesztelhető.

        Returns:
            BGR formátumú NumPy tömb (szintetikus kép)
        """
        # Zöld pálya háttér
        frame = np.full(
            (self._target_h, self._target_w, 3),
            fill_value=(34, 139, 34),  # Forest green (BGR)
            dtype=np.uint8
        )

        # Sávvonalak rajzolása
        cv2.line(frame,
                 (self._target_w // 2, 0),
                 (self._target_w // 2, self._target_h),
                 (255, 255, 255), 3)

        # Mozgó labda (szinuszos pálya)
        t = self._frame_count / max(self._target_fps, 1)
        ball_x = int(self._target_w // 2 + np.sin(t * 0.8) * self._target_w * 0.3)
        ball_y = int(self._target_h // 2 + np.cos(t * 0.3) * self._target_h * 0.2)

        # Fehér focilabda rajzolása
        ball_radius = 30
        cv2.circle(frame, (ball_x, ball_y), ball_radius, (255, 255, 255), -1)
        # Fekete mintázat a labdán (focilabda kinézet)
        for dx, dy in [(0, 0), (15, 10), (-15, 10), (0, -18)]:
            cv2.circle(frame,
                       (ball_x + dx, ball_y + dy),
                       8, (30, 30, 30), -1)

        # Felirat: TESZT MÓD
        side = "BAL [TESZT]" if self._info.is_left else "JOBB [TESZT]"
        cv2.putText(frame, side, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
        cv2.putText(frame, f"Frame: {self._frame_count}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 1)

        return frame

    def set_exposure(self, exposure_us: int) -> None:
        """Szimulált – nincs valódi hatása a webcamera expozíciójára."""
        logger.debug("MockCamera: set_exposure(%d µs) – szimulált", exposure_us)

    def set_gain(self, gain_db: float) -> None:
        """Szimulált – nincs valódi hatása."""
        logger.debug("MockCamera: set_gain(%.1f dB) – szimulált", gain_db)

    def get_fps(self) -> float:
        """Visszaadja a mért (szimulált) frame rate-t."""
        return self._measured_fps or self._target_fps
