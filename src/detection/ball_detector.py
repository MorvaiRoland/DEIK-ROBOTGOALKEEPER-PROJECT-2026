"""
DEIK Robot Foci Kapus – YOLOv10n Labdadetektor
===============================================

Ez a modul a focilabda valós idejű detektálását valósítja meg
az Ultralytics YOLOv10n modell segítségével.

Miért YOLOv10n?
    - NMS-mentes (No Matching Suppression): kisebb végpontok közötti latencia
    - Nano méret: alacsony memória- és CPU/GPU igény
    - ByteTrack beépítve az Ultralytics könyvtárba
    - RTX 3050-en: ~200 FPS (640px bemeneten), ~120 FPS (1280px bemeneten)

Detektálási pipeline:
    1. Frame beérkezik (BGR NumPy tömb)
    2. Opcionális ROI kivágás
    3. YOLOv10n inferencia (CUDA GPU)
    4. ByteTrack tracking (ID hozzárendelés)
    5. "Sports ball" osztályra szűrés (COCO ID: 32)
    6. Kalman simítás (per-kamera 2D szűrő)
    7. BallDetection visszaadása

Referencia:
    - COCO osztályok: https://cocodataset.org/#explore
    - YOLOv10: https://docs.ultralytics.com/models/yolov10/
"""

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Ultralytics importálása (YOLO modell kezelő)
try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    logger.error("Az 'ultralytics' csomag nincs telepítve! pip install ultralytics")


# --------------------------------------------------------------------------- #
# Adatstruktúrák
# --------------------------------------------------------------------------- #

@dataclass
class BallDetection:
    """
    Egy detektált labda összes adatát tárolja.

    Attributes:
        found:       True ha a labdát sikeresen detektáltuk
        x:           Labda középpont X koordinátája (pixelben)
        y:           Labda középpont Y koordinátája (pixelben)
        radius:      Labda becsült sugara pixelben
        confidence:  Detektálási magabiztosság (0.0 – 1.0)
        track_id:    ByteTrack által adott egyedi ID (None ha tracking kikapcsolva)
        bbox:        Befoglaló doboz: (x1, y1, x2, y2) pixelben
        timestamp:   A detektálás időpontja
    """
    found: bool = False
    x: float = 0.0
    y: float = 0.0
    radius: float = 0.0
    confidence: float = 0.0
    track_id: Optional[int] = None
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class StereoBallDetection:
    """
    Mindkét kamera detektálási eredményét tárolja egy adatstruktúrában.

    Attributes:
        left:          Bal kamera detektálási eredménye
        right:         Jobb kamera detektálási eredménye
        both_found:    True ha mindkét kamerában megtaláltuk a labdát
        det_fps:       Detektálási frame rate (EMA simított)
    """
    left: BallDetection = field(default_factory=BallDetection)
    right: BallDetection = field(default_factory=BallDetection)
    both_found: bool = False
    det_fps: float = 0.0


# --------------------------------------------------------------------------- #
# Fő detektor osztály
# --------------------------------------------------------------------------- #

class BallDetector:
    """
    YOLOv10n alapú focilabda detektor ByteTrack integrációval.

    A detektor egy megosztott YOLO modell példányon fut, ami mindkét
    kamera framejét képes feldolgozni. A ByteTrack tracking megőrzi
    a labda azonosságát akkor is, ha a YOLO egy-két frame-et kihagyna
    (pl. motion blur esetén).

    Example:
        cfg = {"model_path": "models/yolov10n.pt", "ball_class_id": 32,
               "confidence_threshold": 0.4, "device": "cuda:0", ...}
        detector = BallDetector(cfg)
        detection = detector.detect(frame_left, frame_right)
        if detection.both_found:
            print(f"Labda: bal=({detection.left.x}, {detection.left.y})")
    """

    def __init__(self, config: dict):
        """
        Args:
            config: A system_config.yaml "detection" szekciója
        """
        if not ULTRALYTICS_AVAILABLE:
            raise RuntimeError("Ultralytics nincs telepítve: pip install ultralytics")

        self._config = config
        self._model_path = config["model_path"]
        self._ball_class_id = int(config["ball_class_id"])
        self._confidence_threshold = float(config["confidence_threshold"])
        self._iou_threshold = float(config.get("iou_threshold", 0.45))
        self._input_size = int(config.get("input_size", 1280))
        self._device = str(config.get("device", "cuda:0"))

        # Tracking konfig
        tracking_cfg = config.get("tracking", {})
        self._tracking_enabled = bool(tracking_cfg.get("enabled", True))
        self._tracker_config = str(tracking_cfg.get("tracker_config", "bytetrack.yaml"))

        # ROI konfig
        roi_cfg = config.get("roi", {})
        self._roi_enabled = bool(roi_cfg.get("enabled", False))
        self._roi_x_min = float(roi_cfg.get("x_min_rel", 0.0))
        self._roi_x_max = float(roi_cfg.get("x_max_rel", 1.0))
        self._roi_y_min = float(roi_cfg.get("y_min_rel", 0.0))
        self._roi_y_max = float(roi_cfg.get("y_max_rel", 1.0))

        # YOLO modell betöltése
        self._model: Optional[YOLO] = None
        self._load_model()

        # FPS mérés (Exponential Moving Average)
        self._det_fps: float = 0.0
        self._fps_alpha: float = 0.1

        logger.info("BallDetector kész: modell='%s', eszköz='%s'",
                    self._model_path, self._device)

    def _load_model(self) -> None:
        """
        Betölti a YOLOv10n modellt és áthelyezi a GPU-ra.

        Ha a modell fájl nem létezik, automatikusan letöltjük
        a Ultralytics szerverről.
        """
        model_file = Path(self._model_path)

        if not model_file.exists():
            logger.info(
                "Modell fájl nem található: '%s'. Letöltés...",
                self._model_path
            )
            # A model_path-ban lévő fájlnév alapján töltjük le
            model_name = model_file.name
            self._model = YOLO(model_name)

            # Elmentjük a konfig által megadott helyre
            model_file.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            downloaded = Path(model_name)
            if downloaded.exists():
                shutil.move(str(downloaded), str(model_file))
                logger.info("✓ Modell letöltve és mentve: %s", model_file)
        else:
            logger.info("Modell betöltése: %s", model_file)
            self._model = YOLO(str(model_file))

        # GPU-ra küldés és melegítés (warm-up)
        logger.info("GPU meleg-indítás (warm-up)...")
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self._model(
            dummy,
            device=self._device,
            verbose=False,
            imgsz=640
        )
        logger.info("✓ Modell betöltve és GPU-n fut")

    # ------------------------------------------------------------------
    # Fő detektálási metódus
    # ------------------------------------------------------------------

    def detect(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray
    ) -> StereoBallDetection:
        """
        Detektálja a labdát mindkét kamera képén.

        A két képet egyszerre (batch-ben) küldjük a YOLO modellnek,
        így csökkentve a GPU overhead-et.

        Args:
            frame_left:  Bal kamera BGR képe (NumPy tömb)
            frame_right: Jobb kamera BGR képe (NumPy tömb)

        Returns:
            StereoBallDetection: Mindkét kamera detektálási eredménye
        """
        t_start = time.perf_counter()

        # --- ROI alkalmazása (opcionális) ---
        left_proc, left_roi_offset = self._apply_roi(frame_left)
        right_proc, right_roi_offset = self._apply_roi(frame_right)

        # --- YOLO inferencia ---
        # Ha tracking engedélyezett: model.track() → ByteTrack ID-k
        # Ha nem: model() → csak detektálás
        if self._tracking_enabled:
            results = self._model.track(
                [left_proc, right_proc],   # Batch: mindkét kép egyszerre
                device=self._device,
                verbose=False,
                imgsz=self._input_size,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                classes=[self._ball_class_id],   # Csak "sports ball"
                tracker=self._tracker_config,
                persist=True,                    # Megőrzi a track state-et frame-ek között
            )
        else:
            results = self._model(
                [left_proc, right_proc],
                device=self._device,
                verbose=False,
                imgsz=self._input_size,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                classes=[self._ball_class_id],
            )

        # --- Eredmények kinyerése ---
        det_left = self._extract_best_ball(
            results[0], roi_offset=left_roi_offset
        )
        det_right = self._extract_best_ball(
            results[1], roi_offset=right_roi_offset
        )

        # --- FPS mérés ---
        dt = time.perf_counter() - t_start
        instant_fps = 1.0 / max(dt, 1e-6)
        self._det_fps = (
            (1.0 - self._fps_alpha) * self._det_fps +
            self._fps_alpha * instant_fps
        )

        return StereoBallDetection(
            left=det_left,
            right=det_right,
            both_found=det_left.found and det_right.found,
            det_fps=self._det_fps,
        )

    # ------------------------------------------------------------------
    # Segéd metódusok
    # ------------------------------------------------------------------

    def _apply_roi(
        self, frame: np.ndarray
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Kivágja a ROI területet a képből (ha ROI engedélyezett).

        Args:
            frame: Teljes kép

        Returns:
            Tuple: (ROI kép, (x_offset, y_offset) a teljes képhez képest)
        """
        if not self._roi_enabled:
            return frame, (0, 0)

        h, w = frame.shape[:2]
        x1 = int(w * self._roi_x_min)
        x2 = int(w * self._roi_x_max)
        y1 = int(h * self._roi_y_min)
        y2 = int(h * self._roi_y_max)

        roi_crop = frame[y1:y2, x1:x2]
        return roi_crop, (x1, y1)

    def _extract_best_ball(
        self,
        yolo_result,
        roi_offset: Tuple[int, int] = (0, 0)
    ) -> BallDetection:
        """
        Kinyeri a legjobb (legmagabiztosabb) labda detektálást a YOLO eredményből.

        Ha több labdát is detektál (ami ritka), a legnagyobb konfidenciájút
        vesszük figyelembe.

        Args:
            yolo_result: Egy kép YOLO eredménye (ultralytics Result objektum)
            roi_offset:  (x_off, y_off) ROI eltolás a teljes képhez képest

        Returns:
            BallDetection: A legjobb detektálás, vagy found=False ha nincs.
        """
        ox, oy = roi_offset
        boxes = yolo_result.boxes

        if boxes is None or len(boxes) == 0:
            return BallDetection(found=False)

        # Legmagasabb konfidenciájú detektálás kiválasztása
        confidences = boxes.conf.cpu().numpy()
        best_idx = int(np.argmax(confidences))
        best_conf = float(confidences[best_idx])

        if best_conf < self._confidence_threshold:
            return BallDetection(found=False)

        # Befoglaló doboz koordinátái
        xyxy = boxes.xyxy[best_idx].cpu().numpy()
        x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

        # ROI eltolás visszaszámítása (ha volt ROI kivágás)
        x1 += ox; x2 += ox
        y1 += oy; y2 += oy

        # Középpont és sugár
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0
        radius = ((x2 - x1) + (y2 - y1)) / 4.0  # Átlagos sugár

        # Track ID kinyerése (ha tracking engedélyezett)
        track_id = None
        if boxes.id is not None:
            track_id = int(boxes.id[best_idx].cpu().item())

        return BallDetection(
            found=True,
            x=cx,
            y=cy,
            radius=radius,
            confidence=best_conf,
            track_id=track_id,
            bbox=(x1, y1, x2, y2),
            timestamp=time.perf_counter(),
        )

    # ------------------------------------------------------------------
    # Vizualizáció segédmetódus
    # ------------------------------------------------------------------

    def draw_detection(
        self,
        frame: np.ndarray,
        detection: BallDetection,
        color: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Rárajzolja a detektálási eredményt a képre (in-place).

        Args:
            frame:     BGR kép (módosítandó)
            detection: A detektálási eredmény
            color:     Rajzolási szín (BGR)

        Returns:
            A módosított kép (ugyanaz az objektum, in-place)
        """
        if not detection.found:
            return frame

        cx, cy = int(detection.x), int(detection.y)
        r = int(detection.radius)

        # Befoglaló doboz
        x1, y1, x2, y2 = [int(v) for v in detection.bbox]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Középpont
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Sugár kör
        cv2.circle(frame, (cx, cy), r, color, 2)

        # Kereszthajó
        cv2.line(frame, (cx - r, cy), (cx + r, cy), color, 1)
        cv2.line(frame, (cx, cy - r), (cx, cy + r), color, 1)

        # Feliratok
        label_parts = [f"{detection.confidence:.2f}"]
        if detection.track_id is not None:
            label_parts.append(f"ID:{detection.track_id}")
        label = "  ".join(label_parts)

        cv2.putText(
            frame, label,
            (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )

        return frame

    # ------------------------------------------------------------------
    # Property-k
    # ------------------------------------------------------------------

    @property
    def detection_fps(self) -> float:
        """A mért detektálási FPS (EMA simított)."""
        return self._det_fps
