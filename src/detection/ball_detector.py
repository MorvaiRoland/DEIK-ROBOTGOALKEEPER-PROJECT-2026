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

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np

logger = logging.getLogger(__name__)

# Ultralytics importálása (YOLO modell kezelő)
try:
    # pyrefly: ignore [missing-import]
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

    def __init__(self, config: dict, full_config: Optional[dict] = None):
        """
        Args:
            config: A system_config.yaml "detection" szekciója
            full_config: A teljes system_config.yaml (a ball.hsv_filter szekció eléréséhez)
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

        # ROI konfig (külön bal és jobb kamerára)
        roi_cfg = config.get("roi", {})
        default_roi = {
            "enabled": bool(roi_cfg.get("enabled", False)),
            "x_min_rel": float(roi_cfg.get("x_min_rel", 0.0)),
            "x_max_rel": float(roi_cfg.get("x_max_rel", 1.0)),
            "y_min_rel": float(roi_cfg.get("y_min_rel", 0.1)),
            "y_max_rel": float(roi_cfg.get("y_max_rel", 0.9)),
        }
        self._left_roi = default_roi.copy()
        self._right_roi = default_roi.copy()

        # ----- HSV szín-ellenőrző konfiguráció (narancssárga labda) -----
        ball_cfg = (full_config or {}).get("ball", {})
        hsv_cfg = ball_cfg.get("hsv_filter", {})
        self._hsv_enabled = bool(hsv_cfg.get("enabled", False))
        self._hsv_h_min = int(hsv_cfg.get("h_min", 5))
        self._hsv_h_max = int(hsv_cfg.get("h_max", 25))
        self._hsv_s_min = int(hsv_cfg.get("s_min", 100))
        self._hsv_s_max = int(hsv_cfg.get("s_max", 255))
        self._hsv_v_min = int(hsv_cfg.get("v_min", 100))
        self._hsv_v_max = int(hsv_cfg.get("v_max", 255))
        self._hsv_min_ratio = float(hsv_cfg.get("min_color_ratio", hsv_cfg.get("min_orange_ratio", 0.08)))
        self._min_circularity = float(hsv_cfg.get("min_circularity", 0.15))
        self._min_aspect_ratio = float(hsv_cfg.get("min_aspect_ratio", 0.35))
        self._max_aspect_ratio = float(hsv_cfg.get("max_aspect_ratio", 2.50))
        self._max_colored_blob_ratio = float(hsv_cfg.get("max_colored_blob_ratio", 0.95))

        if self._hsv_enabled:
            logger.info(
                "HSV szín-ellenőrző AKTÍV: H=[%d-%d], S=[%d-%d], V=[%d-%d], min_ratio=%.0f%%",
                self._hsv_h_min, self._hsv_h_max,
                self._hsv_s_min, self._hsv_s_max,
                self._hsv_v_min, self._hsv_v_max,
                self._hsv_min_ratio * 100,
            )

        # YOLO modell betöltése
        self._model: Optional[YOLO] = None
        self._load_model()

        # FPS mérés (Exponential Moving Average)
        self._det_fps: float = 0.0
        self._fps_alpha: float = 0.1

        logger.info("BallDetector kész: modell='%s', eszköz='%s'",
                    self._model_path, self._device)

    def set_roi(
        self,
        is_left: bool,
        enabled: bool,
        x_min_rel: float,
        x_max_rel: float,
        y_min_rel: float,
        y_max_rel: float
    ) -> None:
        """Beállítja egy kamera ROI paramétereit dinamikusan."""
        roi_dict = {
            "enabled": bool(enabled),
            "x_min_rel": max(0.0, min(1.0, float(x_min_rel))),
            "x_max_rel": max(0.0, min(1.0, float(x_max_rel))),
            "y_min_rel": max(0.0, min(1.0, float(y_min_rel))),
            "y_max_rel": max(0.0, min(1.0, float(y_max_rel))),
        }
        if is_left:
            self._left_roi = roi_dict
        else:
            self._right_roi = roi_dict
        logger.debug("ROI frissítve (%s): %s", "bal" if is_left else "jobb", roi_dict)

    def draw_roi(self, frame: np.ndarray, is_left: bool) -> np.ndarray:
        """Kirajzolja az aktív ROI keretet a kameraképre."""
        roi = self._left_roi if is_left else self._right_roi
        if not roi.get("enabled", False):
            return frame

        h, w = frame.shape[:2]
        x1 = int(w * roi["x_min_rel"])
        x2 = int(w * roi["x_max_rel"])
        y1 = int(h * roi["y_min_rel"])
        y2 = int(h * roi["y_max_rel"])

        # Sárgás-cián téglalap és felirat a ROI kijelzéséhez
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame, "ROI ACTIVE", (x1 + 6, max(y1 + 20, 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1
        )
        return frame

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

        # Automatikus class ID felismerése a modell neveiből (user error megelőzése)
        if hasattr(self._model, "names") and self._model.names:
            names = self._model.names
            if len(names) == 1:
                self._ball_class_id = 0
                logger.info("✓ Automatikus osztály felismerve: 1-osztályos labdamodell (class_id=0)")
            else:
                # Keresés a 'ball' vagy 'sports ball' névre
                found_ball_id = None
                for cid, cname in names.items():
                    if "ball" in str(cname).lower():
                        found_ball_id = cid
                        break
                if found_ball_id is not None:
                    self._ball_class_id = found_ball_id
                    logger.info(
                        "✓ Automatikus osztály felismerve: '%s' (class_id=%d)",
                        names[found_ball_id], found_ball_id
                    )

        # GPU-ra küldés és melegítés (warm-up) 2 képből álló batch-csel (sztereó)
        logger.info("GPU meleg-indítás (warm-up)...")
        dummy = np.zeros((self._input_size, self._input_size, 3), dtype=np.uint8)
        self._model(
            [dummy, dummy],
            device=self._device,
            verbose=False,
            imgsz=self._input_size
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
        left_proc, left_roi_offset = self._apply_roi(frame_left, is_left=True)
        right_proc, right_roi_offset = self._apply_roi(frame_right, is_left=False)

        # --- YOLO inferencia ---
        # model() → csak detektálás, ByteTrack nélkül (gyorsabb)
        # stream=True: TensorRT GPU pipeline mód – csökkenti a GPU→CPU lateónciát
        if self._tracking_enabled:
            results = list(self._model.track(
                [left_proc, right_proc],
                device=self._device,
                verbose=False,
                imgsz=self._input_size,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                classes=[self._ball_class_id],
                tracker=self._tracker_config,
                persist=True,
                stream=True,
            ))
        else:
            results = list(self._model(
                [left_proc, right_proc],
                device=self._device,
                verbose=False,
                imgsz=self._input_size,
                conf=self._confidence_threshold,
                iou=self._iou_threshold,
                classes=[self._ball_class_id],
                stream=True,
            ))

        # --- Eredmények kinyerése (YOLO detektálás) ---
        det_left = self._extract_best_ball(
            results[0], frame=left_proc, roi_offset=left_roi_offset
        )
        det_right = self._extract_best_ball(
            results[1], frame=right_proc, roi_offset=right_roi_offset
        )

        # --- TARTALÉK (Hybrid Fallback): Szín- és körkörösség alapú detektálás ---
        # Ha a YOLO modell nem detektálja a labdát a levegőben (pl. hiányzó talaj-kontextus miatt),
        # a tartalék HSV szín- és körkörösség detektor azonnal megtalálja a rikító narancssárga labdát.
        if not det_left.found:
            det_left = self._detect_color_blob(left_proc, roi_offset=left_roi_offset)
        if not det_right.found:
            det_right = self._detect_color_blob(right_proc, roi_offset=right_roi_offset)

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

    def _detect_color_blob(
        self,
        frame: np.ndarray,
        roi_offset: Tuple[int, int] = (0, 0)
    ) -> BallDetection:
        """
        Tartalék (Fallback) detektor: narancssárga kör alakú objektumot keres a képen
        HSV küszöböléssel és kontúr-körkörösség szűréssel.

        Ezt használjuk, ha a YOLO modell nem detektálja a labdát a levegőben
        (pl. hiányzó talaj-kontextus miatt).
        
        Teljesítmény-optimalizált: 50%-os leméretezést használ a gyorsabb CPU feldolgozáshoz.

        Args:
            frame:      BGR kép
            roi_offset: (x_off, y_off) ROI eltolás a teljes képhez képest

        Returns:
            BallDetection: A talált narancssárga labda, vagy found=False ha nincs.
        """
        ox, oy = roi_offset
        h, w = frame.shape[:2]

        # --- OPTIMALIZÁCIÓ: 50%-os leméretezés a gyorsabb HSV és morfológiai műveletekhez ---
        scale = 0.5
        inv_scale = 1.0 / scale
        small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)

        # BGR → HSV konverzió
        hsv = cv2.cvtColor(small_frame, cv2.COLOR_BGR2HSV)

        # Narancssárga maszk létrehozása (a konfigurációban megadott HSV határok alapján)
        lower = np.array([self._hsv_h_min, self._hsv_s_min, self._hsv_v_min], dtype=np.uint8)
        upper = np.array([self._hsv_h_max, self._hsv_s_max, self._hsv_v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Morfológiai szűrés (zajcsökkentés és lyukkitöltés)
        # Kisebb kernel a leméretezett képhez (3x3 a korábbi 5x5 helyett)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel)

        # Kontúrok keresése
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_candidate = None
        best_score = -1.0
        
        small_w = w * scale
        small_h = h * scale

        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Terület küszöbök leméretezve (eredeti min. 120 -> 30, max. 18% terület)
            if area < (120 * scale**2) or area > (small_w * small_h * 0.18):
                continue

            perimeter = cv2.arcLength(cnt, True)
            if perimeter <= 0:
                continue

            # Körkörösség számítása (circularity = 4 * pi * Area / Perimeter^2)
            circularity = (4.0 * np.pi * area) / (perimeter * perimeter)
            if circularity < 0.40:  # Szigorúbb körkörösségi küszöb: kizárja a cipőket, ruhadarabokat
                continue

            # Bounding box méretarány (Aspect Ratio = szélesség / magasság)
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bh <= 0:
                continue
            aspect_ratio = bw / float(bh)
            if not (0.35 <= aspect_ratio <= 2.50):  # Mozgáselmosódott (megnyúlt) labda engedése
                continue

            # Konvex kiterjedés (Solidity = terület / konvex burok területe)
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = area / float(hull_area)
            if solidity < 0.60:  # Szigorúbb tömörség: nem gömbölyű alakok kiszűrése
                continue

            (x, y), radius = cv2.minEnclosingCircle(cnt)
            # Sugár szűrés leméretezve: minimum 12px az eredeti felbontáson (6px 50%-on)
            # Kis „pontok" (zajdetektálás) kiszűrése a megnövelt minimummal
            if radius < (6.0 * scale) or radius > (160.0 * scale):
                continue

            # Szaturáció átlagának kiszámítása a kontúron belül
            c_mask = np.zeros(mask.shape, dtype=np.uint8)
            cv2.drawContours(c_mask, [cnt], -1, 255, -1)
            mean_s = cv2.mean(hsv[:, :, 1], mask=c_mask)[0]
            
            # Magasabb szaturáció-küszöb: halvány/szürke, nem narancsos objektumok kiszűrése
            if mean_s < 70:
                continue

            # Pontszámítás: (Körkörösség négyzete) * Tömörség * (Szaturáció aránya)
            score = (circularity ** 2) * solidity * (mean_s / 255.0)
            if score > best_score:
                best_score = score
                best_candidate = (x, y, radius, area, circularity)

        # Minimális pontszám-küszöb: kizárja a gyenge, bizonytalan jelölteket
        # (pl. halvány narancsos foltok a falon, cipő, ruha sarokpontjai)
        MIN_SCORE_THRESHOLD = 0.25
        if best_candidate is not None and best_score >= MIN_SCORE_THRESHOLD:
            x_small, y_small, radius_small, area_small, circ = best_candidate
            
            # Visszaszorzás az eredeti felbontásra
            x = x_small * inv_scale
            y = y_small * inv_scale
            radius = radius_small * inv_scale
            
            x1 = x - radius + ox
            y1 = y - radius + oy
            x2 = x + radius + ox
            y2 = y + radius + oy
            cx = x + ox
            cy = y + oy

            logger.debug(
                "✓ Color-blob tartalék detektálás SIKERES: (%.1f, %.1f) r=%.1f circ=%.2f",
                cx, cy, radius, circ
            )
            return BallDetection(
                found=True,
                x=cx,
                y=cy,
                radius=radius,
                confidence=0.60,  # Tartalék detektálás mérsékelt konfidenciával
                track_id=None,
                bbox=(x1, y1, x2, y2),
                timestamp=time.perf_counter(),
            )

        return BallDetection(found=False)

    def _apply_roi(
        self, frame: np.ndarray, is_left: bool
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """
        Kivágja a ROI területet a képből (ha ROI engedélyezett az adott kamerán).

        Args:
            frame:   Teljes kép
            is_left: True = bal kamera, False = jobb kamera

        Returns:
            Tuple: (ROI kép, (x_offset, y_offset) a teljes képhez képest)
        """
        roi = self._left_roi if is_left else self._right_roi
        if not roi.get("enabled", False):
            return frame, (0, 0)

        h, w = frame.shape[:2]
        x1 = int(w * roi["x_min_rel"])
        x2 = int(w * roi["x_max_rel"])
        y1 = int(h * roi["y_min_rel"])
        y2 = int(h * roi["y_max_rel"])

        x1 = max(0, min(w - 1, x1))
        x2 = max(x1 + 1, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(y1 + 1, min(h, y2))

        roi_crop = frame[y1:y2, x1:x2]
        return roi_crop, (x1, y1)


    def _validate_orange_color(
        self,
        frame: np.ndarray,
        x1: int, y1: int, x2: int, y2: int
    ) -> bool:
        """
        Ellenőrzi, hogy a bounding box területén dominánsan narancssárga szín van-e.

        Ez a HSV szín-validáció kiszűri a hamis pozitívokat (pl. cipők, kezek),
        amelyeket a YOLO tévesen "sports ball"-nak detektált.

        Args:
            frame: A feldolgozott kép (ROI-vágott, BGR)
            x1, y1, x2, y2: Bounding box koordináták (a frame-en belül)

        Returns:
            True ha a terület elegendő narancssárga pixelt tartalmaz
        """
        if not self._hsv_enabled:
            return True

        h, w = frame.shape[:2]
        # Bounding box koordináták clampelése a kép tartományára
        bx1 = max(0, min(w - 1, int(x1)))
        by1 = max(0, min(h - 1, int(y1)))
        bx2 = max(bx1 + 1, min(w, int(x2)))
        by2 = max(by1 + 1, min(h, int(y2)))

        roi = frame[by1:by2, bx1:bx2]
        if roi.size == 0:
            return False

        # BGR → HSV konverzió
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Narancssárga maszk
        lower = np.array([self._hsv_h_min, self._hsv_s_min, self._hsv_v_min], dtype=np.uint8)
        upper = np.array([self._hsv_h_max, self._hsv_s_max, self._hsv_v_max], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)

        # Narancssárga pixelek aránya
        total_pixels = mask.shape[0] * mask.shape[1]
        orange_pixels = int(np.count_nonzero(mask))
        ratio = orange_pixels / max(total_pixels, 1)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        colored_blob_ratio = area / max(float(total_pixels), 1.0)
        perimeter = cv2.arcLength(largest, True)
        circularity = (
            (4.0 * np.pi * area) / (perimeter * perimeter)
            if perimeter > 0.0 else 0.0
        )

        logger.debug(
            "HSV ellenőrzés: orange_ratio=%.1f%% (küszöb=%.0f%%), box=[%d,%d,%d,%d]",
            ratio * 100, self._hsv_min_ratio * 100, bx1, by1, bx2, by2
        )

        return (
            ratio >= self._hsv_min_ratio
            and circularity >= self._min_circularity
            and colored_blob_ratio <= self._max_colored_blob_ratio
        )

    def _extract_best_ball(
        self,
        yolo_result,
        frame: Optional[np.ndarray] = None,
        roi_offset: Tuple[int, int] = (0, 0)
    ) -> BallDetection:
        """
        Kinyeri a legjobb (legmagabiztosabb) labda detektálást a YOLO eredményből.

        Ha több labdát is detektál (ami ritka), a legnagyobb konfidenciájút
        vesszük figyelembe. A detektálást a HSV szín-ellenőrző post-filterrel
        validáljuk (ha engedélyezve van).

        Args:
            yolo_result: Egy kép YOLO eredménye (ultralytics Result objektum)
            frame:       A feldolgozott kép (ROI-vágott, BGR) – HSV ellenőrzéshez
            roi_offset:  (x_off, y_off) ROI eltolás a teljes képhez képest

        Returns:
            BallDetection: A legjobb detektálás, vagy found=False ha nincs.
        """
        ox, oy = roi_offset
        boxes = yolo_result.boxes

        if boxes is None or len(boxes) == 0:
            return BallDetection(found=False)

        # Konfidencia szerinti csökkenő sorrend
        confidences = boxes.conf.cpu().numpy()
        sorted_indices = np.argsort(-confidences)  # Csökkenő sorrend

        # Végigmegyünk az összes detektáláson (csökkenő konfidencia sorrendben)
        # Az elsőt fogadjuk el, amelyik átmegy a HSV szín-ellenőrzésen
        for idx in sorted_indices:
            conf = float(confidences[idx])
            if conf < self._confidence_threshold:
                break  # Alacsonyabb konfidenciájúakat nem nézzük

            # Befoglaló doboz koordinátái
            xyxy = boxes.xyxy[idx].cpu().numpy()
            x1, y1, x2, y2 = float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3])

            # HSV szín-ellenőrzés (a ROI-vágott képen, ROI offsetelés előtt)
            # Bypass: csak magas konfidenciájú (>= 0.40) YOLO detektálás kerüli el a szín-ellenőrzést.
            # Alacsony konf. esetén (cipő, kéz, ruha) kötelező a narancssárga szín ellenőrzése.
            if frame is not None and self._hsv_enabled and conf < 0.40:
                if not self._validate_orange_color(frame, x1, y1, x2, y2):
                    logger.debug(
                        "HSV ellenőrzés BUKOTT: box=[%.0f,%.0f,%.0f,%.0f] conf=%.2f → kihagyva",
                        x1, y1, x2, y2, conf
                    )
                    continue  # Nem narancssárga → következő detektálás

            # ROI eltolás visszaszámítása (ha volt ROI kivágás)
            x1 += ox; x2 += ox
            y1 += oy; y2 += oy

            # Középpont és sugár
            cx = (x1 + x2) / 2.0
            cy = (y1 + y2) / 2.0
            radius = ((x2 - x1) + (y2 - y1)) / 4.0  # Átlagos sugár

            # Minimális sugár ellenőrzése (zaj kiszűrése)
            if radius < 5.0:
                continue

            # Track ID kinyerése (ha tracking engedélyezett)
            track_id = None
            if boxes.id is not None:
                track_id = int(boxes.id[idx].cpu().item())

            return BallDetection(
                found=True,
                x=cx,
                y=cy,
                radius=radius,
                confidence=conf,
                track_id=track_id,
                bbox=(x1, y1, x2, y2),
                timestamp=time.perf_counter(),
            )

        # Egyetlen detektálás sem ment át a szín-ellenőrzésen
        return BallDetection(found=False)

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
