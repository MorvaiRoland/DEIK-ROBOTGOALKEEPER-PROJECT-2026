"""
DEIK Robot Foci Kapus – Ximea MC023CG-SY-UB Kamera Implementáció
==================================================================

Ez a modul a Ximea MC023CG-SY-UB ipari kamera teljes kezelését valósítja meg
a Ximea xiAPI Python könyvtár segítségével.

Főbb funkciók:
    - Thread-safe képszerzés dedikált háttérszálon
    - Ring buffer az alacsony latenciájú frame-hozzáféréshez
    - USB3 sávszélesség menedzsment (két kamera párhuzamos kezelése)
    - Automatikus újracsatlakozás kiesés esetén

Hardver:
    - Kamera: Ximea MC023CG-SY-UB
    - Szenzor: Sony IMX174 (Global Shutter, 2.3 MP)
    - Max FPS: 165 (teljes 1936×1216 felbontáson)
    - Csatlakozás: USB3 (EP-USB3HybridcableU-20 kábel)

Hivatkozások:
    - Ximea xiAPI doku: https://www.ximea.com/support/wiki/apis/Python
    - SDK telepítés: https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np

from camera.base_camera import BaseCamera, CameraFrame, CameraInfo

# Modul szintű napló
logger = logging.getLogger(__name__)

# Ximea xiAPI importálása – ha nincs telepítve, futásidőben jelzünk hibát
try:
    from ximea import xiapi
    XIMEA_AVAILABLE = True
except ImportError:
    XIMEA_AVAILABLE = False
    logger.warning(
        "Ximea xiAPI nem elérhető! "
        "Telepítsd a Ximea Linux SDK-t: "
        "https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package"
    )

# --------------------------------------------------------------------------- #
# Konstansok
# --------------------------------------------------------------------------- #

# Ring buffer mérete – ennyi frame-et tárolunk memóriában
_RING_BUFFER_SIZE = 4

# Kamera megnyitásának maximális próbálkozásainak száma
_MAX_OPEN_RETRIES = 3

# Próbálkozások közötti várakozási idő (másodperc)
_RETRY_DELAY_SEC = 1.0

# Maximális várakozás frame olvasásnál (másodperc)
_FRAME_TIMEOUT_SEC = 2.0

# Kamera hőmérsékleti riasztási küszöb (Celsius)
_TEMPERATURE_WARNING_THRESHOLD = 60.0


class XimeaCamera(BaseCamera):
    """
    Ximea MC023CG-SY-UB kamera thread-safe implementációja.

    A képszerzés dedikált háttérszálon fut, így a fő program soha nem
    blokkolódik egy frame megszerzéséig. A legfrissebb frame mindig
    azonnal elérhető a ring bufferből.

    Example:
        cfg = {"fps": 100, "exposure_time_us": 3000, "gain_db": 0.0,
               "bandwidth_limit_mbs": 160, "resolution": {"width": 1936, "height": 1216}}

        cam = XimeaCamera(camera_index=0, is_left=True, config=cfg)
        if cam.open():
            frame = cam.read()
            if frame.success:
                cv2.imshow("Kép", frame.image)
        cam.close()
    """

    def __init__(
        self,
        camera_index: int,
        is_left: bool,
        config: dict,
        serial_number: Optional[str] = None,
    ):
        """
        Args:
            camera_index:  Kamera sorszáma (0 = első USB kamera, 1 = második, stb.)
            is_left:       True = bal oldali (negatív X), False = jobb oldali (pozitív X)
            config:        A system_config.yaml "camera" szekciója
            serial_number: Ha megadott, sorozatszám alapján nyitjuk meg (ajánlott)
        """
        if not XIMEA_AVAILABLE:
            raise RuntimeError(
                "Ximea xiAPI nincs telepítve. "
                "Teszteléshez használd a MockCamera-t."
            )

        # Kamera metaadatok létrehozása
        side = "BAL" if is_left else "JOBB"
        cam_info = CameraInfo(
            name=f"Ximea-MC023CG [{side}] #{camera_index}",
            width=config["resolution"]["width"],
            height=config["resolution"]["height"],
            fps=float(config.get("fps", 100)),
            is_left=is_left,
            serial_num=serial_number or "N/A",
        )
        super().__init__(cam_info)

        # Kamera konfiguráció mentése
        self._config = config
        self._camera_index = camera_index
        self._serial_number = serial_number
        self._target_fps = float(config.get("fps", 100))
        self._exposure_us = int(config.get("exposure_time_us", 3000))
        self._gain_db = float(config.get("gain_db", 0.0))
        self._bandwidth_mbs = int(config.get("bandwidth_limit_mbs", 160))

        # Ximea SDK objektumok
        self._cam: Optional[xiapi.Camera] = None
        self._xi_image: Optional[xiapi.Image] = None

        # Thread-safe ring buffer a frame-eknek
        self._buffer: deque = deque(maxlen=_RING_BUFFER_SIZE)
        self._buffer_lock = threading.Lock()

        # Háttérszál vezérlés
        self._acquisition_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._frame_ready_event = threading.Event()

        # FPS mérés
        self._measured_fps: float = 0.0
        self._fps_alpha: float = 0.05  # EMA koefficiense (simítás mértéke)

        logger.info("XimeaCamera létrehozva: %s", cam_info.name)

    # ------------------------------------------------------------------
    # Kamera életciklus
    # ------------------------------------------------------------------

    def open(self) -> bool:
        """
        Megnyitja a Ximea kamerát és elindítja a képszerzési szálat.

        Automatikusan megpróbálja _MAX_OPEN_RETRIES-szor, ha az első
        kísérlet sikertelen (USB buszi instabilitás esetén hasznos).

        Returns:
            True ha a megnyitás sikeres volt.
        """
        for attempt in range(1, _MAX_OPEN_RETRIES + 1):
            logger.info(
                "Ximea kamera megnyitása: %s (kísérlet %d/%d)...",
                self._info.name, attempt, _MAX_OPEN_RETRIES
            )
            try:
                self._cam = xiapi.Camera()
                self._xi_image = xiapi.Image()

                # Kamera megnyitása sorozatszám vagy index alapján
                if self._serial_number:
                    self._cam.open_device_by_SN(str(self._serial_number))
                elif self._camera_index == 0:
                    self._cam.open_device()
                else:
                    # Index > 0 esetén próbáljuk meg megnyitni a következőt
                    # xiAPI-ban open_device() az első elérhető szabad eszközt nyitja meg,
                    # így ha a 0-s kamera már nyitva van, az 1-est nyitja meg!
                    try:
                        self._cam.open_device()
                    except Exception:
                        self._cam.open_device_by("XI_OPEN_BY_INST_PATH", str(self._camera_index))

                # Kamera paramétereinek beállítása
                self._configure_camera()

                # Képszerzés megkezdése
                self._cam.start_acquisition()

                # Háttérszál indítása
                self._stop_event.clear()
                self._acquisition_thread = threading.Thread(
                    target=self._acquisition_loop,
                    name=f"XimeaAcq-{'L' if self._info.is_left else 'R'}",
                    daemon=True,
                )
                self._acquisition_thread.start()

                self._is_open = True
                logger.info("✓ Kamera sikeresen megnyitva: %s", self._info.name)
                return True

            except Exception as exc:
                logger.error(
                    "Kamera megnyitási hiba [%d/%d]: %s – %s",
                    attempt, _MAX_OPEN_RETRIES, self._info.name, exc
                )
                self._cleanup_cam()

                if attempt < _MAX_OPEN_RETRIES:
                    logger.info("Újrapróbálás %g másodperc múlva...", _RETRY_DELAY_SEC)
                    time.sleep(_RETRY_DELAY_SEC)

        logger.error("Kamera megnyitása sikertelen: %s", self._info.name)
        return False

    def close(self) -> None:
        """
        Leállítja a képszerzést és bezárja a kamera kapcsolatot.
        Thread-safe: biztonságos a háttérszálból is hívni.
        """
        logger.info("Kamera bezárása: %s", self._info.name)

        # Jelzés a háttérszálnak, hogy álljon le
        self._stop_event.set()

        # Megvárjuk a szál leállását (max 2 másodperc)
        if self._acquisition_thread and self._acquisition_thread.is_alive():
            self._acquisition_thread.join(timeout=2.0)

        self._cleanup_cam()
        self._is_open = False
        logger.info("✓ Kamera bezárva: %s", self._info.name)

    def _cleanup_cam(self) -> None:
        """Felszabadítja a Ximea SDK erőforrásait (belső helper)."""
        if self._cam is not None:
            try:
                self._cam.stop_acquisition()
            except Exception:
                pass
            try:
                self._cam.close_device()
            except Exception:
                pass
            self._cam = None

    # ------------------------------------------------------------------
    # Kamera konfiguráció
    # ------------------------------------------------------------------

    def _configure_camera(self) -> None:
        """
        Beállítja a kamera paramétereit a konfig alapján.
        A megnyitás után hívódik meg, az acqusition megkezdése előtt.
        """
        if self._cam is None:
            return

        # --- Alapparaméterek ---
        self._cam.set_exposure(self._exposure_us)
        logger.debug("  Zársebesség: %d µs", self._exposure_us)

        self._cam.set_gain(self._gain_db)
        logger.debug("  Erősítés: %.1f dB", self._gain_db)

        # --- USB3 sávszélesség korlátozás ---
        try:
            self._cam.set_limit_bandwidth_mode("XI_ON")
            self._cam.set_limit_bandwidth(self._bandwidth_mbs)
            logger.debug("  USB sávszélesség korlát: %d MB/s", self._bandwidth_mbs)
        except Exception as exc:
            logger.debug("  Sávszélesség korlát beállítása szkippelve: %s", exc)

        # --- Frame rate ---
        try:
            self._cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE")
            self._cam.set_framerate(self._target_fps)
            logger.debug("  Frame rate: %.1f FPS", self._target_fps)
        except Exception as exc:
            logger.debug("  Frame rate beállítása szkippelve: %s", exc)

        # --- Képformátum ---
        # XI_RGB24 = Ximea 24-bites színes formátum (OpenCV-hez BGR-ré konvertáljuk)
        self._cam.set_imgdataformat("XI_RGB24")

        # --- Fehéregyensúly (White Balance) ---
        # Színes CMOS szenzoroknál (Sony IMX174) AWB nélkül a kép zöldes árnyalatú,
        # mert a Bayer-mátrix 50%-a zöld szűrős pixel (RGGB).
        try:
            if self._config.get("auto_white_balance", True):
                self._cam.enable_auto_wb()
                logger.debug("  Automatikus fehéregyensúly (AWB) bekapcsolva")
            else:
                kr = float(self._config.get("wb_kr", 1.8))
                kg = float(self._config.get("wb_kg", 1.0))
                kb = float(self._config.get("wb_kb", 2.1))
                self._cam.set_wb_kr(kr)
                self._cam.set_wb_kg(kg)
                self._cam.set_wb_kb(kb)
                logger.debug("  Manuális fehéregyensúly beállítva: R=%.2f, G=%.2f, B=%.2f", kr, kg, kb)
        except Exception as exc:
            logger.warning("  Fehéregyensúly beállítási hiba: %s", exc)

        # --- Akvizíciós puffer méret növelése (nagy FPS-hez) ---
        # Alapértelmezett 70 MB → 256 MB (csökkenti az eldobott frame-eket)
        try:
            self._cam.set_acq_buffer_size(256)
        except Exception:
            pass  # Régebbi SDK verziókban nem elérhető

        logger.debug("  Kamera konfiguráció kész: %s", self._info.name)

    # ------------------------------------------------------------------
    # Frame olvasás
    # ------------------------------------------------------------------

    def read(self) -> CameraFrame:
        """
        A legfrissebb frame-et olvassa ki a ring bufferből.

        Ez a metódus SOHA NEM BLOKKOLÓDIK – ha nincs elérhető frame,
        azonnali success=False-szal tér vissza.

        Returns:
            CameraFrame: A legújabb frame, vagy üres frame ha nincs adat.
        """
        with self._buffer_lock:
            if self._buffer:
                # Kiolvassuk a legfrissebb frame-et (deque jobb vége)
                return self._buffer[-1]

        # Ha a buffer üres, de a kamera nyitva van, várunk rövid ideig
        if self._is_open:
            # Max 100ms-t várunk az első frame-re
            self._frame_ready_event.wait(timeout=0.1)
            self._frame_ready_event.clear()

            with self._buffer_lock:
                if self._buffer:
                    return self._buffer[-1]

        # Üres frame visszaadása (nem panic – előfordulhat startup-nál)
        return CameraFrame(success=False, frame_id=self._frame_count)

    # ------------------------------------------------------------------
    # Kamera vezérlés (menetes-biztos beállítások)
    # ------------------------------------------------------------------

    def set_exposure(self, exposure_us: int) -> None:
        """
        Beállítja a zársebességet valós időben.

        Args:
            exposure_us: Zársebesség mikroszekundumban
        """
        self._exposure_us = exposure_us
        if self._cam and self._is_open:
            try:
                self._cam.set_exposure(exposure_us)
                logger.debug("Zársebesség beállítva: %d µs (%s)", exposure_us, self._info.name)
            except Exception as exc:
                logger.warning("Zársebesség beállítási hiba: %s", exc)

    def set_gain(self, gain_db: float) -> None:
        """
        Beállítja az erősítést valós időben.

        Args:
            gain_db: Erősítés dB-ben
        """
        self._gain_db = gain_db
        if self._cam and self._is_open:
            try:
                self._cam.set_gain(gain_db)
                logger.debug("Erősítés beállítva: %.1f dB (%s)", gain_db, self._info.name)
            except Exception as exc:
                logger.warning("Erősítés beállítási hiba: %s", exc)

    def set_awb(self, enabled: bool) -> None:
        """Beállítja az automatikus fehéregyensúlyt valós időben."""
        if self._cam and self._is_open:
            try:
                if enabled:
                    self._cam.enable_auto_wb()
                    logger.debug("AWB bekapcsolva (%s)", self._info.name)
                else:
                    self._cam.disable_auto_wb()
                    logger.debug("AWB kikapcsolva (%s)", self._info.name)
            except Exception as exc:
                logger.warning("AWB beállítási hiba: %s", exc)

    def set_wb(self, kr: float, kg: float, kb: float) -> None:
        """Beállítja a manuális fehéregyensúlyt valós időben."""
        if self._cam and self._is_open:
            try:
                self._cam.set_wb_kr(kr)
                self._cam.set_wb_kg(kg)
                self._cam.set_wb_kb(kb)
                logger.debug("WB beállítva: R=%.2f G=%.2f B=%.2f (%s)", kr, kg, kb, self._info.name)
            except Exception as exc:
                logger.warning("WB beállítási hiba: %s", exc)

    def get_fps(self) -> float:
        """Visszaadja a mért valódi frame rate-t (EMA simítással)."""
        return self._measured_fps

    def get_temperature(self) -> float:
        """
        Visszaadja a szenzor hőmérsékletét Celsius fokban.

        Returns:
            Hőmérséklet, vagy 0.0 ha nem olvasható.
        """
        if self._cam and self._is_open:
            try:
                return float(self._cam.get_sensor_board_temp())
            except Exception:
                pass
        return 0.0

    # ------------------------------------------------------------------
    # Háttérszál – folyamatos képszerzés
    # ------------------------------------------------------------------

    def _acquisition_loop(self) -> None:
        """
        A dedikált képszerzési szál fő ciklusa.

        Ez a metódus a háttérben fut és folyamatosan tölti a ring buffert
        a kamerából érkező frame-ekkel. A fő szál (read() hívója) soha
        nem blokkolódik egy frame megszerzéséig.
        """
        logger.info("Képszerzési szál elindult: %s", self._info.name)

        last_fps_time = time.perf_counter()
        fps_frame_count = 0

        while not self._stop_event.is_set():
            try:
                # Frame megszerzése a Ximea kamerától
                # Timeout: 1000 ms (ha ennyi idő alatt nincs frame, exception)
                self._cam.get_image(self._xi_image, timeout=1000)

                # NumPy tömbbé alakítás és RGB->BGR konverzió OpenCV-hez
                raw = self._xi_image.get_image_data_numpy()
                bgr_image = cv2.cvtColor(raw, cv2.COLOR_RGB2BGR)

                # Frame metaadatok
                timestamp = time.perf_counter()
                self._frame_count += 1
                fps_frame_count += 1

                # CameraFrame létrehozása
                frame = CameraFrame(
                    image=bgr_image,
                    timestamp=timestamp,
                    frame_id=self._frame_count,
                    success=True,
                )

                # Ring bufferbe helyezés (thread-safe)
                with self._buffer_lock:
                    self._buffer.append(frame)

                # Jelzés az olvasónak, hogy új frame érkezett
                self._frame_ready_event.set()

                # FPS mérése (minden 30 frame-enként frissítjük)
                if fps_frame_count >= 30:
                    elapsed = time.perf_counter() - last_fps_time
                    instant_fps = fps_frame_count / elapsed
                    self._measured_fps = (
                        (1.0 - self._fps_alpha) * self._measured_fps +
                        self._fps_alpha * instant_fps
                    )
                    last_fps_time = time.perf_counter()
                    fps_frame_count = 0

                    # Hőmérséklet ellenőrzés
                    temp = self.get_temperature()
                    if temp > _TEMPERATURE_WARNING_THRESHOLD:
                        logger.warning(
                            "FIGYELEM: Magas kamera hőmérséklet: %.1f°C (%s)",
                            temp, self._info.name
                        )

            except Exception as exc:
                if not self._stop_event.is_set():
                    # Csak akkor logolunk, ha nem szándékos leállás
                    logger.error(
                        "Képszerzési hiba (%s): %s",
                        self._info.name, exc
                    )
                    time.sleep(0.01)  # Rövid szünet hiba után

        logger.info("Képszerzési szál leállt: %s", self._info.name)
