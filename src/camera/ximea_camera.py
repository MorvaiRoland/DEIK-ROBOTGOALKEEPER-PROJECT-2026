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
    - Hardveres MASTER/SLAVE GPIO szinkronizáció (CBL-702-8P-SYNC-5M0 kábel)

Hardver:
    - Kamera: Ximea MC023CG-SY-UB
    - Szenzor: Sony IMX174 (Global Shutter, 2.3 MP)
    - Max FPS: 165 (teljes 1936×1216 felbontáson)
    - Csatlakozás: USB3 (EP-USB3HybridcableU-20 kábel)
    - Szinkron kábel: CBL-702-8P-SYNC-5M0 (M9, 8 pólusú)

Hardveres szinkron bekötés (CBL-702-8P-SYNC-5M0):
    MASTER kábel Pin 3 (Zöld/OUT1)     → SLAVE kábel Pin 5 (Szürke/IN1)
    MASTER kábel Pin 4 (Sárga/OUT-GND) → SLAVE kábel Pin 6 (Rózsaszín/IN-GND)
    MASTER kábel Pin 7 (Kék/GND)       → SLAVE kábel Pin 7 (Kék/GND)

Hivatkozások:
    - Ximea xiAPI doku: https://www.ximea.com/support/wiki/apis/Python
    - SDK telepítés: https://www.ximea.com/support/wiki/apis/XIMEA_Linux_Software_Package
    - GPIO szinkron: https://www.ximea.com/support/wiki/apis/Trigger_and_synchronization
"""

import logging
import threading
import time
from collections import deque
from typing import Optional

# pyrefly: ignore [missing-import]
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

# Maximális várakozás frame olvasásnál (másodperc) – MASTER módban
_FRAME_TIMEOUT_SEC = 2.0

# Maximális várakozás frame olvasásnál SLAVE módban (ms) – várja a trigger jelet
# Ha 5000 ms alatt nem kap triggert, az SDK hibát dob (kábel nincs bekötve?)
_SLAVE_FRAME_TIMEOUT_MS = 5000

# Kamera hőmérsékleti riasztási küszöb (Celsius)
_TEMPERATURE_WARNING_THRESHOLD = 60.0

# Érvényes sync szerepek
_SYNC_ROLES = {"master", "slave", None}


class XimeaCamera(BaseCamera):
    """
    Ximea MC023CG-SY-UB kamera thread-safe implementációja.

    A képszerzés dedikált háttérszálon fut, így a fő program soha nem
    blokkolódik egy frame megszerzéséig. A legfrissebb frame mindig
    azonnal elérhető a ring bufferből.

    Example:
        cfg = {"fps": 100, "exposure_time_us": 3000, "gain_db": 0.0,
               "bandwidth_mode": "unlimited",
               "resolution": {"width": 1936, "height": 1216}}

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
        sync_role: Optional[str] = None,
        sync_config: Optional[dict] = None,
    ):
        """
        Args:
            camera_index:  Kamera sorszáma (0 = első USB kamera, 1 = második, stb.)
            is_left:       True = bal oldali (negatív X), False = jobb oldali (pozitív X)
            config:        A system_config.yaml "camera" szekciója
            serial_number: Ha megadott, sorozatszám alapján nyitjuk meg (ajánlott)
            sync_role:     Szinkronizáció szerepe: "master" | "slave" | None (szoftver szinkron)
                           MASTER: GPIO OUT1-en adja ki az expozíciós trigger pulzust
                           SLAVE:  GPIO IN1-en várja a trigger jelet a MASTER-től
            sync_config:   A config.yaml "camera.sync" szekciója (GPIO pin beállítások)
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
        self._bandwidth_mode = str(config.get("bandwidth_mode", "unlimited")).lower()
        raw_bandwidth_limit = config.get("bandwidth_limit_mbps")
        self._bandwidth_limit_mbps: Optional[int] = (
            int(raw_bandwidth_limit) if raw_bandwidth_limit is not None else None
        )
        if self._bandwidth_mode not in {"unlimited", "limited"}:
            raise ValueError(
                "Ismeretlen bandwidth_mode: "
                f"{self._bandwidth_mode!r}; 'unlimited' vagy 'limited' lehet."
            )
        if self._bandwidth_mode == "limited" and self._bandwidth_limit_mbps is None:
            raise ValueError(
                "bandwidth_mode='limited' esetén a bandwidth_limit_mbps kötelező "
                "(xiAPI egység: Mbit/s)."
            )
        self._offset_x = int(config.get("offset_x", 0))
        self._offset_y = int(config.get("offset_y", 0))
        self._flip_h = bool(config.get("flip_h", False))
        self._flip_v = bool(config.get("flip_v", False))
        self._rotation = int(config.get("rotation", 0))  # Landscape mód alapból

        # --- Hardveres GPIO szinkronizáció ---
        if sync_role not in _SYNC_ROLES:
            raise ValueError(
                f"Ismeretlen sync_role: {sync_role!r}; "
                "'master', 'slave' vagy None lehet."
            )
        self._sync_role: Optional[str] = sync_role  # "master" | "slave" | None
        self._sync_config: dict = sync_config or {}
        # Ha SLAVE, hosszabb timeout-ot használunk (trigger jelet vár a MASTER-től)
        self._frame_timeout_ms: int = (
            _SLAVE_FRAME_TIMEOUT_MS if sync_role == "slave" else 1000
        )
        # Fallback flag: ha a SLAVE nem kap triggert, visszaesünk szoftver szinkronra
        self._sync_fallback_active: bool = False

        if sync_role:
            logger.info(
                "XimeaCamera sync szerepe: %s (%s)",
                sync_role.upper(), cam_info.name
            )


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

        Szinkronizáció:
            MASTER: GPO OUT1 → XI_GPO_EXPOSURE_ACTIVE (pulzus az expozíció alatt)
            SLAVE:  GPI IN1  → XI_GPI_TRIGGER + XI_TRG_EDGE_RISING (emelkedő élre triggerel)
        """
        if self._cam is None:
            return

        # --- Alapparaméterek ---
        self._cam.set_exposure(self._exposure_us)
        logger.debug("  Zársebesség: %d µs", self._exposure_us)

        self._cam.set_gain(self._gain_db)
        logger.debug("  Erősítés: %.1f dB", self._gain_db)

        # --- Képformátum ---
        # A formatum megváltoztathatja a lehetséges frame rate-et, ezért ezt
        # a sávszélesség- és FPS-paraméterek ELŐTT kell beállítani.
        # XI_RGB24 = Ximea 24-bites színes formátum (OpenCV-hez BGR sorrendben)
        self._cam.set_imgdataformat("XI_RGB24")

        # --- USB3 sávszélesség ---
        # XI_PRM_LIMIT_BANDWIDTH egysége Mbit/s, nem MB/s. Külön 5 Gbit/s
        # USB root hubon a XI_OFF engedi a kamera által elérhető maximumot.
        try:
            if self._bandwidth_mode == "unlimited":
                self._cam.set_limit_bandwidth_mode("XI_OFF")
                logger.info("  USB sávszélesség: korlátlan (XI_OFF)")
            else:
                self._cam.set_limit_bandwidth(self._bandwidth_limit_mbps)
                self._cam.set_limit_bandwidth_mode("XI_ON")
                logger.info(
                    "  USB sávszélesség limit: %d Mbit/s",
                    self._bandwidth_limit_mbps,
                )
        except Exception as exc:
            logger.warning("  USB sávszélesség beállítási hiba: %s", exc)

        # --- Frame rate (csak MASTER és szoftver-szinkron módban) ---
        # SLAVE módban a frame rate-et a MASTER trigger jele határozza meg,
        # ezért frame rate limitet nem állítunk be.
        if self._sync_role != "slave":
            try:
                self._cam.set_acq_timing_mode("XI_ACQ_TIMING_MODE_FRAME_RATE_LIMIT")
                self._cam.set_framerate(self._target_fps)
                logger.info("  Cél frame rate: %.1f FPS", self._target_fps)
            except Exception as exc:
                logger.warning("  Frame rate beállítási hiba: %s", exc)
        else:
            logger.info(
                "  Frame rate: SLAVE módban a MASTER trigger határozza meg "
                "(%.1f FPS várható)", self._target_fps
            )

        # A tényleges SDK-értékek naplózása teszteléshez. Ezek mutatják meg,
        # hogy a kábel/port valóban SuperSpeed-en és elegendő sávszélességgel fut-e.
        try:
            logger.info(
                "  XiAPI kapcsolat: elérhető=%d Mbit/s, limit=%d Mbit/s, "
                "beállított=%.1f FPS, max=%.1f FPS, payload=%d byte",
                self._cam.get_available_bandwidth(),
                self._cam.get_limit_bandwidth(),
                self._cam.get_framerate(),
                self._cam.get_framerate_maximum(),
                self._cam.get_image_payload_size(),
            )
        except Exception as exc:
            logger.warning("  XiAPI diagnosztikai értékek nem olvashatók: %s", exc)

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

        # --- Akvizíciós puffer optimalizálás (XIMEA dokumentáció alapján) ---
        # Forrás: https://www.ximea.com/support/wiki/usb3/How_to_optimize_software_performance_on_high_frame_rates
        #
        # Nagy FPS-hez (100 FPS, 1936×1216, RGB24 ≈ 6.8 MB/frame) két dolog kell:
        # 1. ACQ_TRANSPORT_BUFFER_SIZE → payload méretéhez igazítva (ne legyen feleslegesen nagy)
        # 2. BUFFERS_QUEUE_SIZE → maximumra állítva (minél több puffer, annál kevesebb eldobott frame)
        try:
            # 1. Payload méret lekérdezése
            payload = self._cam.get_image_payload_size()

            # 2. Transport puffer méret lekérdezése és optimalizálása
            transport_default = self._cam.get_acq_transport_buffer_size()
            try:
                transport_increment = self._cam.get_acq_transport_buffer_size_increment()
            except Exception:
                transport_increment = 0
            try:
                transport_minimum = self._cam.get_acq_transport_buffer_size_minimum()
            except Exception:
                transport_minimum = 0

            # Ha a payload kisebb mint az alapértelmezett transport puffer,
            # érdemes a transport puffert a payload-hoz igazítani
            if transport_default > 0 and payload < transport_default + max(transport_increment, 1):
                transport_size = payload
                if transport_increment > 0:
                    remainder = transport_size % transport_increment
                    if remainder:
                        transport_size += transport_increment - remainder
                if transport_minimum > 0 and transport_size < transport_minimum:
                    transport_size = transport_minimum
                try:
                    self._cam.set_acq_transport_buffer_size(transport_size)
                    logger.info(
                        "  Transport puffer: %d byte (payload=%d, alap=%d)",
                        transport_size, payload, transport_default
                    )
                except Exception as exc:
                    logger.debug("  Transport puffer beállítás nem sikerült: %s", exc)
            else:
                logger.debug("  Transport puffer: alapértelmezett (%d byte)", transport_default)

            # 3. Queue puffer szám maximalizálása
            try:
                max_queue = self._cam.get_buffers_queue_size_maximum()
                if max_queue and max_queue > 0:
                    self._cam.set_buffers_queue_size(max_queue)
                    logger.info(
                        "  Queue puffer szám: %d (maximum) → kevesebb eldobott frame 100 FPS-nél",
                        max_queue
                    )
            except Exception as exc:
                # Fallback: ha nem sikerül lekérdezni a maximumot, 32-t állítunk be
                try:
                    self._cam.set_buffers_queue_size(32)
                    logger.info("  Queue puffer szám: 32 (fallback)")
                except Exception:
                    pass
                logger.debug("  Queue puffer maximum lekérdezés nem sikerült: %s", exc)

        except Exception as exc:
            logger.warning("  Puffer optimalizálás részben sikertelen: %s", exc)


        # ================================================================
        # Hardveres GPIO szinkronizáció konfigurálása
        # ================================================================
        # A MASTER expozíció ELEJÉN az OUT1 (Pin 3, Zöld) kimenet HIGH lesz.
        # Ez az optó-izolált jel a kábelen keresztül a SLAVE IN1 (Pin 5, Szürke)
        # bemenetére kerül, ahol az emelkedő él (RISING EDGE) triggeri a SLAVE
        # kamera expozícióját. Eredmény: <10 µs szinkron jitter.
        if self._sync_role == "master":
            self._configure_gpio_master()
        elif self._sync_role == "slave":
            self._configure_gpio_slave()

        logger.debug("  Kamera konfiguráció kész: %s", self._info.name)

    def _configure_gpio_master(self) -> None:
        """
        MASTER kamera GPIO kimenet konfigurálása.

        - Nem izolált mód (XI_GPO_PORT2): Pin 8 (Piros / INOUT1) 3.3V LVTTL kimenet
        - Opto-izolált mód (XI_GPO_PORT1): Pin 3 (Zöld / OUT1) Open Collector kimenet
        """
        if self._cam is None:
            return

        gpo_selector = self._sync_config.get("gpo_selector", "XI_GPO_PORT2")
        gpo_mode = self._sync_config.get("gpo_mode", "XI_GPO_EXPOSURE_ACTIVE")

        try:
            self._cam.set_gpo_selector(gpo_selector)
            self._cam.set_gpo_mode(gpo_mode)
            pin_desc = (
                "Pin 8 (Piros/INOUT1 nem izolált)" if gpo_selector == "XI_GPO_PORT2"
                else "Pin 3 (Zöld/OUT1 opto-izolált)"
            )
            logger.info(
                "  [MASTER] GPIO kimenet beállítva: %s → %s | "
                "Kábel %s adja a trigger pulzust a SLAVE-nek.",
                gpo_selector, gpo_mode, pin_desc
            )
        except Exception as exc:
            logger.error(
                "  [MASTER] GPIO kimenet beállítási HIBA (%s → %s): %s "
                "| Ellenőrizd a kábel csatlakozását!",
                gpo_selector, gpo_mode, exc
            )

    def _configure_gpio_slave(self) -> None:
        """
        SLAVE kamera GPIO bemenet és trigger konfigurálása.

        - Nem izolált mód (XI_GPI_PORT2): Pin 8 (Piros / INOUT1) 3.3V LVTTL bemenet
        - Opto-izolált mód (XI_GPI_PORT1): Pin 5 (Szürke / IN1) opto-izolált bemenet
        """
        if self._cam is None:
            return

        gpi_selector = self._sync_config.get("gpi_selector", "XI_GPI_PORT2")
        gpi_mode = self._sync_config.get("gpi_mode", "XI_GPI_TRIGGER")
        trigger_source = self._sync_config.get("trigger_source", "XI_TRG_EDGE_RISING")

        try:
            # 1. GPI port kiválasztása és trigger módba állítása
            self._cam.set_gpi_selector(gpi_selector)
            self._cam.set_gpi_mode(gpi_mode)
            pin_desc = (
                "Pin 8 (Piros/INOUT1 nem izolált)" if gpi_selector == "XI_GPI_PORT2"
                else "Pin 5 (Szürke/IN1 opto-izolált)"
            )
            logger.info(
                "  [SLAVE] GPIO bemenet beállítva: %s → %s | "
                "Kábel %s fogadja a MASTER trigger pulzusát.",
                gpi_selector, gpi_mode, pin_desc
            )

            # 2. Bidirekcionális pin (Pin 8 / PORT2) esetén a SLAVE kimeneti meghajtóját
            # High Impedance (XI_GPO_OFF) állapotba állítjuk, hogy bemenetként működjön!
            if gpi_selector == "XI_GPI_PORT2":
                try:
                    self._cam.set_gpo_selector("XI_GPO_PORT2")
                    self._cam.set_gpo_mode("XI_GPO_OFF")
                    logger.info("  [SLAVE] PORT2 (Pin 8 INOUT1) kimenet kikapcsolva: XI_GPO_OFF (High Impedance / bemenet)")
                except Exception as exc:
                    logger.debug("  [SLAVE] GPO OFF beállítás nem sikerült (elmaradhat): %s", exc)

            # 3. Trigger forrás: külső jel emelkedő élére
            self._cam.set_trigger_source(trigger_source)
            logger.info(
                "  [SLAVE] Trigger forrás: %s (MASTER expozíció kezdetén triggerel)",
                trigger_source
            )

        except Exception as exc:
            logger.error(
                "  [SLAVE] GPIO trigger beállítási HIBA (%s/%s/%s): %s "
                "| Ellenőrizd a kábel csatlakozását!",
                gpi_selector, gpi_mode, trigger_source, exc
            )

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

        SLAVE módban:
            A get_image() hívás _SLAVE_FRAME_TIMEOUT_MS ms-ig vár a trigger
            jelre. Ha ez letelik (kábel nincs bekötve?), hibát logolunk.
        """
        role_tag = f"[{self._sync_role.upper()}] " if self._sync_role else ""
        logger.info("Képszerzési szál elindult: %s%s", role_tag, self._info.name)

        last_fps_time = time.perf_counter()
        fps_frame_count = 0
        slave_timeout_warned = False  # Csak egyszer figyelmeztetünk timeout esetén

        while not self._stop_event.is_set():
            try:
                # Frame megszerzése a Ximea kamerától
                # MASTER/szoftver-szinkron: 1000 ms timeout
                # SLAVE: _SLAVE_FRAME_TIMEOUT_MS (5000 ms) – várja a MASTER trigger jelét
                self._cam.get_image(self._xi_image, timeout=self._frame_timeout_ms)

                # Ha idáig eljutunk, a SLAVE megkapta a trigger jelet
                if self._sync_role == "slave" and slave_timeout_warned:
                    slave_timeout_warned = False
                    logger.info(
                        "  [SLAVE] Trigger jel újra megérkezett – HW szinkron helyreállt."
                    )

                # NumPy tömbbé alakítás
                # MEGJEGYZÉS: A Ximea XI_RGB24 formátum BGR byte-sorrendben adja a pixeleket,
                # ami az OpenCV natív formátuma. NEM kell COLOR_RGB2BGR konverzió!
                # (Korábbi COLOR_RGB2BGR hívás HIBÁSAN megcserélte a piros és kék csatornákat,
                #  ezért jelent meg a narancssárga labda kékként.)
                bgr_image = self._xi_image.get_image_data_numpy()

                # Alkalmazzuk az X/Y elmozdulást, tükrözést és elforgatást
                bgr_image = self.apply_image_transformations(bgr_image)

                # Frame metaadatok: hardveres kamera időbélyeg (ha elérhető), egyébként rendszer óra
                try:
                    timestamp = float(self._xi_image.tsSec) + (float(self._xi_image.tsUSec) * 1e-6)
                except (AttributeError, TypeError):
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
                    # SLAVE timeout esetén külön figyelmeztetés (kábel nincs bekötve?)
                    if self._sync_role == "slave" and not slave_timeout_warned:
                        logger.error(
                            "  [SLAVE] Trigger timeout! A MASTER kamera nem küld jelet. "
                            "Ellenőrizd a kábel bekötését: "
                            "Nem izolált mód: MASTER Pin8(Piros) → SLAVE Pin8(Piros) és MASTER Pin7(Kék/GND) → SLAVE Pin7(Kék/GND). "
                            "Hiba: %s",
                            exc
                        )
                        slave_timeout_warned = True
                    elif self._sync_role != "slave":
                        logger.error(
                            "Képszerzési hiba (%s%s): %s",
                            role_tag, self._info.name, exc
                        )
                    time.sleep(0.01)  # Rövid szünet hiba után

        logger.info("Képszerzési szál leállt: %s%s", role_tag, self._info.name)
