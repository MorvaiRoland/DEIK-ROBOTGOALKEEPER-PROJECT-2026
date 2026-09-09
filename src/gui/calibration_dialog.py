"""
DEIK Robot Foci Kapus – Sztereó Kalibrációs Dialog (PyQt6)
==========================================================

Teljes grafikus kalibrációs műhely a két Ximea kamera sztereó kalibrálásához.

Fül 1 – Beállítások:   Tábla paraméterek (Chessboard / ChArUco), minimum képpárok, kimeneti fájl
Fül 2 – Pozíció:       Élő kamera kép + tábla overlay + igazítási segéd
Fül 3 – Képrögzítés:   Élő kamera kép + tábla overlay, képpár mentés, progress
Fül 4 – Kalibrálás:    Kalibrálás indítása, log, eredmény összesítő

TELJESÍTMÉNY OPTIMALIZÁCIÓ:
  - Detektálás csak minden 6. frame-nél (throttling) → ~22 FPS detektálás 130 FPS-es kamera esetén
  - 50%-os kicsinyítés detektálás előtt → 4x gyorsabb sarokpont keresés
  - GUI frissítés max. 30 FPS-re korlátozva (33ms interval)

ChArUco mód: cv2.aruco.CharucoBoard / cv2.aruco.CharucoDetector (OpenCV 5.x újabb API)
"""

import logging
import time
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import cv2.aruco as aruco
# pyrefly: ignore [missing-import]
import numpy as np

# pyrefly: ignore [missing-import]
from PyQt6.QtCore import QThread, Qt, pyqtSignal, pyqtSlot
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import QImage, QPixmap, QTextCursor
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import (
    QDialog, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSpinBox, QDoubleSpinBox, QLineEdit,
    QGroupBox, QFormLayout, QProgressBar, QPlainTextEdit,
    QFileDialog, QMessageBox, QFrame, QSizePolicy, QScrollArea,
    QComboBox,
)

from calibration.alignment_helper import (
    AlignmentInstruction,
    AlignmentResult,
    calculate_camera_alignment,
    draw_alignment_hud,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Tábla típus enum
# --------------------------------------------------------------------------- #

class BoardType(Enum):
    CHESSBOARD = "chessboard"
    CHARUCO    = "charuco"


# --------------------------------------------------------------------------- #
# ArUco szótár neve -> OpenCV konstans leképezés
# --------------------------------------------------------------------------- #

_ARUCO_DICT_MAP = {
    "DICT_4X4_50":    aruco.DICT_4X4_50,
    "DICT_4X4_100":   aruco.DICT_4X4_100,
    "DICT_4X4_250":   aruco.DICT_4X4_250,
    "DICT_4X4_1000":  aruco.DICT_4X4_1000,
    "DICT_5X5_50":    aruco.DICT_5X5_50,
    "DICT_5X5_100":   aruco.DICT_5X5_100,
    "DICT_5X5_250":   aruco.DICT_5X5_250,
    "DICT_5X5_1000":  aruco.DICT_5X5_1000,
    "DICT_6X6_50":    aruco.DICT_6X6_50,
    "DICT_6X6_100":   aruco.DICT_6X6_100,
    "DICT_6X6_250":   aruco.DICT_6X6_250,
    "DICT_6X6_1000":  aruco.DICT_6X6_1000,
    "DICT_7X7_50":    aruco.DICT_7X7_50,
    "DICT_7X7_100":   aruco.DICT_7X7_100,
    "DICT_7X7_250":   aruco.DICT_7X7_250,
    "DICT_7X7_1000":  aruco.DICT_7X7_1000,
}


# --------------------------------------------------------------------------- #
# Sarokpont kereső függvények
# --------------------------------------------------------------------------- #

def find_chessboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """Hagyományos sakktábla belső sarokpontjai sub-pixel pontossággal."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


def find_charuco_corners(
    image: np.ndarray,
    board: "aruco.CharucoBoard",
    detector: "aruco.CharucoDetector",
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    ChArUco sarokpontok detektálása OpenCV 5.x API-val.

    OpenCV 5.0.0 detectBoard() visszaadási formátum:
      - charuco_corners: (N, 2) float32  ← NEM (N, 1, 2)!
      - charuco_ids:     (N,)   int32    ← NEM (N, 1)!

    A drawDetectedCornersCharuco és calibrateCameraCharuco CV_32FC2-t vár:
      - corners (N, 1, 2) float32 → OpenCV: CV_32FC2 [N,1], total()=N ✓
      - ids     (N, 1)    int32   → OpenCV: CV_32SC1 [N,1], total()=N ✓

    Ha corners (N, 2) float32 → CV_32FC1 [N,2], total()=2N ✗ → assertion FAIL!
    Ezért itt normalizáljuk a visszaadás előtt.

    Returns: (charuco_corners (N,1,2) float32, charuco_ids (N,1) int32) vagy (None, None)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
        return None, None
    # Normalizálás: OpenCV 5.0.0 (N,2)/(N,) → (N,1,2)/(N,1)
    corners_norm = np.ascontiguousarray(charuco_corners.reshape(-1, 1, 2), dtype=np.float32)
    ids_norm     = np.ascontiguousarray(charuco_ids.reshape(-1, 1),        dtype=np.int32)
    return corners_norm, ids_norm


# --------------------------------------------------------------------------- #
# Stílusok
# --------------------------------------------------------------------------- #

_BTN_PRIMARY = (
    "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; "
    "border-radius: 6px; font-size: 12px; border: none; padding: 7px 16px; }"
    "QPushButton:hover { background-color: #146C43; }"
    "QPushButton:disabled { background-color: #CBD5E1; color: #94A3B8; }"
)
_BTN_WARN = (
    "QPushButton { background-color: #DC2626; color: #FFFFFF; font-weight: 800; "
    "border-radius: 6px; font-size: 12px; border: none; padding: 7px 16px; }"
    "QPushButton:hover { background-color: #B91C1C; }"
    "QPushButton:disabled { background-color: #CBD5E1; color: #94A3B8; }"
)
_BTN_SEC = (
    "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; "
    "border-radius: 6px; font-size: 12px; border: 1px solid #CBD5E1; padding: 7px 16px; }"
    "QPushButton:hover { background-color: #E2E8F0; color: #0F5132; }"
    "QPushButton:disabled { background-color: #F8FAFC; color: #CBD5E1; }"
)
_BTN_SEC_DARK = (
    "QPushButton { background-color: #151D2A; color: #94A3B8; font-weight: 700; "
    "border-radius: 6px; font-size: 12px; border: 1px solid #26334D; padding: 7px 16px; }"
    "QPushButton:hover { background-color: #1E293B; color: #10B981; border-color: #10B981; }"
    "QPushButton:disabled { background-color: #0B0F17; color: #475569; border-color: #26334D; }"
)
_BTN_CAPTURE = (
    "QPushButton { background-color: #D97706; color: #FFFFFF; font-weight: 900; "
    "border-radius: 6px; font-size: 13px; border: none; padding: 8px 20px; }"
    "QPushButton:hover { background-color: #B45309; }"
    "QPushButton:disabled { background-color: #CBD5E1; color: #94A3B8; }"
)


# --------------------------------------------------------------------------- #
# Kamera Capture Worker (QThread)
# --------------------------------------------------------------------------- #

class CalibrationCaptureWorker(QThread):
    """
    Élő kamera loop ChArUco + chessboard detektálással.

    TELJESÍTMÉNY THROTTLE:
      DETECT_EVERY_N_FRAMES = 6   → detektálás csak minden 6. frame-nél
      PREVIEW_DOWNSCALE = 0.5     → 50% kicsinyítés detektálás előtt
      GUI_MIN_INTERVAL_SEC = 0.033 → max. 30 FPS GUI frissítés
    """

    frame_ready    = pyqtSignal(np.ndarray, np.ndarray, object, object, object)
    capture_result = pyqtSignal(bool, str)
    error_occurred = pyqtSignal(str)
    stopped        = pyqtSignal()

    DETECT_EVERY_N_FRAMES: int   = 6
    PREVIEW_DOWNSCALE:     float = 0.5
    GUI_MIN_INTERVAL_SEC:  float = 0.033
    MIN_POSE_DIFF_PX:      float = 20.0

    def __init__(
        self,
        config:          dict,
        board_type:      BoardType,
        cols:            int,
        rows:            int,
        square_mm:       float,
        marker_mm:       float = 50.0,
        aruco_dict_name: str   = "DICT_6X6_250",
        parent=None,
    ):
        super().__init__(parent)
        self._config          = config
        self._board_type      = board_type
        self._cols            = cols
        self._rows            = rows
        self._square_mm       = square_mm
        self._marker_mm       = marker_mm
        self._aruco_dict_name = aruco_dict_name
        self._running         = False
        self._capture_req     = False

        # Belso sarokpontok szama az alignment_helper-nek
        self._pattern_size = (cols - 1, rows - 1)

        # ChArUco objektumok (run()-ban inicializalva)
        self._charuco_board:    Optional[aruco.CharucoBoard]    = None
        self._charuco_detector: Optional[aruco.CharucoDetector] = None

        # Gyujtott adatok - Chessboard mod
        self.collected_obj_pts:   List[np.ndarray] = []
        self.collected_pts_left:  List[np.ndarray] = []
        self.collected_pts_right: List[np.ndarray] = []

        # Gyujtott adatok - ChArUco mod
        self.collected_charuco_corners_left:  List[np.ndarray] = []
        self.collected_charuco_corners_right: List[np.ndarray] = []
        self.collected_charuco_ids_left:      List[np.ndarray] = []
        self.collected_charuco_ids_right:     List[np.ndarray] = []

        self.image_size: Optional[Tuple[int, int]] = None

    def _init_charuco(self) -> None:
        dict_id = _ARUCO_DICT_MAP.get(self._aruco_dict_name, aruco.DICT_6X6_250)
        dictionary = aruco.getPredefinedDictionary(dict_id)
        self._charuco_board = aruco.CharucoBoard(
            (self._cols, self._rows),
            self._square_mm,
            self._marker_mm,
            dictionary,
        )
        det_params = aruco.DetectorParameters()
        det_params.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        charuco_params = aruco.CharucoParameters()
        charuco_params.tryRefineMarkers = True
        self._charuco_detector = aruco.CharucoDetector(
            self._charuco_board, charuco_params, det_params
        )
        logger.info(
            "ChArUco board: %dx%d, negyzet=%.1fmm, marker=%.1fmm, szotar=%s",
            self._cols, self._rows, self._square_mm, self._marker_mm, self._aruco_dict_name
        )

    def _make_obj_pattern_chessboard(self) -> np.ndarray:
        cx, cy = self._pattern_size
        obj = np.zeros((cx * cy, 3), dtype=np.float32)
        obj[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2) * self._square_mm
        return obj

    def _is_pose_too_similar(self, new_corners: np.ndarray, existing: List[np.ndarray]) -> Tuple[bool, float]:
        """Chessboard módhoz: mindig ugyanannyi sarokpont, közvetlen összehasonlítás."""
        if not existing:
            return False, 9999.0
        pts_new = new_corners.reshape(-1, 2)
        min_diff = 9999.0
        for prev in existing:
            pts_prev = prev.reshape(-1, 2)
            if pts_prev.shape[0] != pts_new.shape[0]:
                # Ha mégis eltérne a méret (pl. hiányos detekció), ugorj át
                continue
            diff = float(np.mean(np.linalg.norm(pts_new - pts_prev, axis=1)))
            if diff < min_diff:
                min_diff = diff
        return min_diff < self.MIN_POSE_DIFF_PX, min_diff

    def _is_charuco_pose_too_similar(
        self,
        new_corners: np.ndarray,         # (N1, 1, 2) float32
        new_ids:     np.ndarray,         # (N1, 1) int32
        existing_corners: List[np.ndarray],
        existing_ids:     List[np.ndarray],
    ) -> Tuple[bool, float]:
        """
        ChArUco módhoz: ID-alapú összehasonlítás.

        ChArUco-nál a detektált sarokpontok száma képenként eltér
        (pl. 86 vs 80 – a tábla részlegesen látszik), ezért közvetlen
        numpy broadcasttal nem lehet összehasonlítani.

        Megoldás: csak a KÖZÖS ID-kkel rendelkező sarokpontokat hasonlítjuk össze.
        Ha kevesebb mint 4 közös sarokpont van → annyira eltérő a póz, hogy elfogadható.
        """
        if not existing_corners:
            return False, 9999.0

        pts_new  = new_corners.reshape(-1, 2)   # (N1, 2)
        ids_new  = new_ids.flatten()            # (N1,)
        min_diff = 9999.0

        for prev_corners, prev_ids in zip(existing_corners, existing_ids):
            pts_prev = prev_corners.reshape(-1, 2)  # (N2, 2)
            ids_prev = prev_ids.flatten()           # (N2,)

            common_ids = np.intersect1d(ids_new, ids_prev)
            if len(common_ids) < 4:
                # Kevés közös sarokpont → biztosan más póz, ne blokkoljuk
                continue

            idx_n = np.array([np.where(ids_new  == cid)[0][0] for cid in common_ids])
            idx_p = np.array([np.where(ids_prev == cid)[0][0] for cid in common_ids])

            diff = float(np.mean(np.linalg.norm(pts_new[idx_n] - pts_prev[idx_p], axis=1)))
            if diff < min_diff:
                min_diff = diff

        return min_diff < self.MIN_POSE_DIFF_PX, min_diff

    @pyqtSlot()
    def request_capture(self):
        self._capture_req = True

    @pyqtSlot()
    def clear_collected(self):
        self.collected_obj_pts.clear()
        self.collected_pts_left.clear()
        self.collected_pts_right.clear()
        self.collected_charuco_corners_left.clear()
        self.collected_charuco_corners_right.clear()
        self.collected_charuco_ids_left.clear()
        self.collected_charuco_ids_right.clear()

    def collected_count(self) -> int:
        if self._board_type == BoardType.CHARUCO:
            return len(self.collected_charuco_corners_left)
        return len(self.collected_obj_pts)

    def stop(self):
        self._running = False

    def run(self) -> None:
        self._running = True

        if self._board_type == BoardType.CHARUCO:
            try:
                self._init_charuco()
            except Exception as exc:
                self.error_occurred.emit(f"ChArUco inicializalas hiba: {exc}")
                self.stopped.emit()
                return

        try:
            from camera.camera_manager import CameraManager
            cam = CameraManager(self._config)
        except Exception as exc:
            self.error_occurred.emit(f"CameraManager import hiba: {exc}")
            self.stopped.emit()
            return

        if not cam.open():
            self.error_occurred.emit("Kamerak megnyitasa sikertelen!")
            self.stopped.emit()
            return

        obj_pattern   = self._make_obj_pattern_chessboard()
        frame_counter = 0
        last_gui_time = 0.0

        last_corners_l:   Optional[np.ndarray] = None
        last_corners_r:   Optional[np.ndarray] = None
        last_charuco_cl:  Optional[np.ndarray] = None
        last_charuco_cr:  Optional[np.ndarray] = None
        last_charuco_il:  Optional[np.ndarray] = None
        last_charuco_ir:  Optional[np.ndarray] = None
        last_align_res:   Optional[AlignmentResult] = None

        try:
            while self._running:
                pair = cam.read_stereo_pair()
                if not pair.success:
                    time.sleep(0.005)
                    continue

                fl_raw = pair.left.image.copy()
                fr_raw = pair.right.image.copy()

                if self.image_size is None:
                    h, w = fl_raw.shape[:2]
                    self.image_size = (w, h)

                frame_counter += 1
                do_detect = (frame_counter % self.DETECT_EVERY_N_FRAMES == 0)

                if do_detect:
                    s    = self.PREVIEW_DOWNSCALE
                    h_s  = int(fl_raw.shape[0] * s)
                    w_s  = int(fl_raw.shape[1] * s)
                    fl_s = cv2.resize(fl_raw, (w_s, h_s), interpolation=cv2.INTER_AREA)
                    fr_s = cv2.resize(fr_raw, (w_s, h_s), interpolation=cv2.INTER_AREA)

                    if self._board_type == BoardType.CHARUCO:
                        cc_l, ci_l = find_charuco_corners(fl_s, self._charuco_board, self._charuco_detector)
                        cc_r, ci_r = find_charuco_corners(fr_s, self._charuco_board, self._charuco_detector)
                        # FONTOS: explicit float32 cast!
                        # cc_l / s Python float osztas float64-et ad vissza,
                        # de drawDetectedCornersCharuco CV_32FC2-t var (total=N).
                        # float64-kent az OpenCV 3D Mat-kent latja (total=2N) → assertion fail!
                        if cc_l is not None:
                            last_charuco_cl = np.ascontiguousarray(cc_l / s, dtype=np.float32)
                            last_charuco_il = np.ascontiguousarray(ci_l, dtype=np.int32).reshape(-1, 1)
                        else:
                            last_charuco_cl = None
                            last_charuco_il = None
                        if cc_r is not None:
                            last_charuco_cr = np.ascontiguousarray(cc_r / s, dtype=np.float32)
                            last_charuco_ir = np.ascontiguousarray(ci_r, dtype=np.int32).reshape(-1, 1)
                        else:
                            last_charuco_cr = None
                            last_charuco_ir = None
                        last_corners_l  = last_charuco_cl
                        last_corners_r  = last_charuco_cr
                    else:
                        cx = find_chessboard_corners(fl_s, self._pattern_size)
                        cr = find_chessboard_corners(fr_s, self._pattern_size)
                        last_corners_l = cx / s if cx is not None else None
                        last_corners_r = cr / s if cr is not None else None

                    last_align_res = calculate_camera_alignment(
                        last_corners_l, last_corners_r,
                        self.image_size, self._pattern_size
                    )

                    if self._capture_req:
                        self._capture_req = False
                        self._handle_capture_request(
                            fl_raw, fr_raw,
                            last_corners_l, last_corners_r,
                            last_charuco_cl, last_charuco_cr,
                            last_charuco_il, last_charuco_ir,
                            obj_pattern,
                        )

                now = time.monotonic()
                if now - last_gui_time >= self.GUI_MIN_INTERVAL_SEC:
                    last_gui_time = now
                    fl_disp = fl_raw.copy()
                    fr_disp = fr_raw.copy()

                    if self._board_type == BoardType.CHARUCO:
                        if last_charuco_cl is not None and last_charuco_il is not None:
                            aruco.drawDetectedCornersCharuco(fl_disp, last_charuco_cl, last_charuco_il, (0, 220, 60))
                        if last_charuco_cr is not None and last_charuco_ir is not None:
                            aruco.drawDetectedCornersCharuco(fr_disp, last_charuco_cr, last_charuco_ir, (0, 220, 60))
                    else:
                        if last_corners_l is not None:
                            cv2.drawChessboardCorners(fl_disp, self._pattern_size, last_corners_l, True)
                        if last_corners_r is not None:
                            cv2.drawChessboardCorners(fr_disp, self._pattern_size, last_corners_r, True)

                    self.frame_ready.emit(
                        fl_disp, fr_disp,
                        last_corners_l, last_corners_r,
                        last_align_res,
                    )

        except Exception as exc:
            logger.exception("Capture worker hiba: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            cam.close()
            self.stopped.emit()

    def _handle_capture_request(
        self,
        fl_raw:     np.ndarray,
        fr_raw:     np.ndarray,
        corners_l:  Optional[np.ndarray],
        corners_r:  Optional[np.ndarray],
        cc_l:       Optional[np.ndarray],
        cc_r:       Optional[np.ndarray],
        ci_l:       Optional[np.ndarray],
        ci_r:       Optional[np.ndarray],
        obj_pattern: np.ndarray,
    ) -> None:
        if self._board_type == BoardType.CHARUCO:
            both = cc_l is not None and cc_r is not None
            if not both:
                self.capture_result.emit(False, "Kepp ar nem mentheto: ChArUco tabla nem lathato mindket kameraban!")
                return
            too_l, diff_l = self._is_charuco_pose_too_similar(
                cc_l, ci_l,
                self.collected_charuco_corners_left,
                self.collected_charuco_ids_left,
            )
            too_r, diff_r = self._is_charuco_pose_too_similar(
                cc_r, ci_r,
                self.collected_charuco_corners_right,
                self.collected_charuco_ids_right,
            )
            min_diff = min(diff_l, diff_r)
            if too_l or too_r:
                self.capture_result.emit(False, f"Tulontosan hasonlo poz! ({min_diff:.0f} px < {self.MIN_POSE_DIFF_PX:.0f} px) – mozd el a tablat!")
                return
            self.collected_charuco_corners_left.append(cc_l)
            self.collected_charuco_corners_right.append(cc_r)
            self.collected_charuco_ids_left.append(ci_l)
            self.collected_charuco_ids_right.append(ci_r)
            n = len(self.collected_charuco_corners_left)
            self.capture_result.emit(True, f"ChArUco keppar mentve: {n} db | bal sarkok: {len(cc_l)}, jobb: {len(cc_r)} | elters: {min_diff:.0f} px")
        else:
            both = corners_l is not None and corners_r is not None
            if not both:
                self.capture_result.emit(False, "Keppar nem mentheto: sakktabla nem lathato mindket kameraban!")
                return
            too_l, diff_l = self._is_pose_too_similar(corners_l, self.collected_pts_left)
            too_r, diff_r = self._is_pose_too_similar(corners_r, self.collected_pts_right)
            min_diff = min(diff_l, diff_r)
            if too_l or too_r:
                self.capture_result.emit(False, f"Tulontosan hasonlo poz! ({min_diff:.0f} px < {self.MIN_POSE_DIFF_PX:.0f} px) – mozd el a tablat!")
                return
            self.collected_obj_pts.append(obj_pattern.copy())
            self.collected_pts_left.append(corners_l)
            self.collected_pts_right.append(corners_r)
            n = len(self.collected_obj_pts)
            self.capture_result.emit(True, f"Keppar mentve: {n} db (elters: {min_diff:.0f} px)")


# --------------------------------------------------------------------------- #
# Kalibrálás Futtatási Worker (QThread)
# --------------------------------------------------------------------------- #

class CalibrationRunWorker(QThread):
    """OpenCV sztereó kalibrálást futtat háttérszálban (Chessboard + ChArUco)."""

    log_line       = pyqtSignal(str)
    finished       = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        board_type:    BoardType,
        image_size:    Tuple[int, int],
        geo_cfg:       dict,
        stereo_cfg:    dict,
        obj_pts:       Optional[List[np.ndarray]] = None,
        pts_l:         Optional[List[np.ndarray]] = None,
        pts_r:         Optional[List[np.ndarray]] = None,
        charuco_board: Optional["aruco.CharucoBoard"] = None,
        charuco_cl:    Optional[List[np.ndarray]] = None,
        charuco_cr:    Optional[List[np.ndarray]] = None,
        charuco_il:    Optional[List[np.ndarray]] = None,
        charuco_ir:    Optional[List[np.ndarray]] = None,
        parent=None,
    ):
        super().__init__(parent)
        self._board_type    = board_type
        self._image_size    = image_size
        self._geo_cfg       = geo_cfg
        self._stereo_cfg    = stereo_cfg
        self._obj_pts       = obj_pts or []
        self._pts_l         = pts_l   or []
        self._pts_r         = pts_r   or []
        self._charuco_board = charuco_board
        self._charuco_cl    = charuco_cl or []
        self._charuco_cr    = charuco_cr or []
        self._charuco_il    = charuco_il or []
        self._charuco_ir    = charuco_ir or []

    def _log(self, msg: str):
        logger.info(msg)
        self.log_line.emit(msg)

    def _per_image_reproj_errors(self, obj_pts, img_pts, K, D, rvecs, tvecs):
        errors = []
        for op, ip, rv, tv in zip(obj_pts, img_pts, rvecs, tvecs):
            proj, _ = cv2.projectPoints(op, rv, tv, K, D)
            err = float(np.mean(np.linalg.norm(ip.reshape(-1, 2) - proj.reshape(-1, 2), axis=1)))
            errors.append(err)
        return errors

    def _filter_outliers(self, obj_pts, pts_l, pts_r, K1, D1, K2, D2,
                         rvecs_l, tvecs_l, rvecs_r, tvecs_r, threshold_factor=1.5):
        errs_l   = self._per_image_reproj_errors(obj_pts, pts_l, K1, D1, rvecs_l, tvecs_l)
        errs_r   = self._per_image_reproj_errors(obj_pts, pts_r, K2, D2, rvecs_r, tvecs_r)
        combined = [max(el, er) for el, er in zip(errs_l, errs_r)]
        median   = float(np.median(combined))
        thresh   = median * threshold_factor
        kept_obj, kept_l, kept_r, removed = [], [], [], 0
        for i, err in enumerate(combined):
            if err <= thresh:
                kept_obj.append(obj_pts[i]); kept_l.append(pts_l[i]); kept_r.append(pts_r[i])
            else:
                removed += 1
                self._log(f"  [outlier] keppar #{i+1} kizarva – hiba: {err:.3f} px > kuszob: {thresh:.3f} px")
        return kept_obj, kept_l, kept_r, removed

    def _run_chessboard_calibration(self) -> None:
        n = len(self._obj_pts)
        self._log(f"[Chessboard] Sztereó kalibrálás: {n} képpár, képméret: {self._image_size}")

        f_px = float(self._geo_cfg.get("focal_length_px", 1365.2))
        cx   = float(self._geo_cfg.get("principal_point_x", 968.0))
        cy_  = float(self._geo_cfg.get("principal_point_y", 608.0))
        ref_baseline = float(self._geo_cfg.get("baseline_mm", 2200.0))
        max_rmse     = float(self._stereo_cfg.get("max_acceptable_rmse_px", 1.0))
        square_mm    = float(self._stereo_cfg.get("chessboard", {}).get("square_size_mm", 65.0))

        self._log(f"  Négyzetméret: {square_mm:.1f} mm | Kezdő f_px: {f_px:.1f}")
        K_init = np.array([[f_px, 0.0, cx], [0.0, f_px, cy_], [0.0, 0.0, 1.0]], dtype=np.float64)

        self._log("[1/3] Bal kamera kalibrálás...")
        rmse_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
            self._obj_pts, self._pts_l, self._image_size,
            K_init.copy(), None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
        )
        self._log(f"  Bal RMSE: {rmse_l:.4f} px")

        self._log("[2/3] Jobb kamera kalibrálás...")
        rmse_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
            self._obj_pts, self._pts_r, self._image_size,
            K_init.copy(), None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
        )
        self._log(f"  Jobb RMSE: {rmse_r:.4f} px")

        self._log("[+] Outlier szűrés (medián × 1.5)...")
        obj_c, pts_l_c, pts_r_c, n_rm = self._filter_outliers(
            self._obj_pts, self._pts_l, self._pts_r,
            K1, D1, K2, D2,
            list(rvecs_l), list(tvecs_l), list(rvecs_r), list(tvecs_r),
        )
        if n_rm > 0:
            self._log(f"  {n_rm} keppar kizarva, maradt: {len(obj_c)} db – ujrakalibralás...")
            rmse_l, K1, D1, _, _ = cv2.calibrateCamera(obj_c, pts_l_c, self._image_size, K1.copy(), D1.copy(),
                                                         flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO)
            rmse_r, K2, D2, _, _ = cv2.calibrateCamera(obj_c, pts_r_c, self._image_size, K2.copy(), D2.copy(),
                                                         flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO)
            self._log(f"  Ujrakalibr. RMSE – Bal: {rmse_l:.4f} px | Jobb: {rmse_r:.4f} px")
        else:
            self._log("  Nem talaltunk outlier kepparokat.")

        self._log("[3/3] Sztereó kalibrálás (R, T)...")
        rmse_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_c, pts_l_c, pts_r_c,
            K1, D1, K2, D2, self._image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        self._emit_result(rmse_s, float(rmse_l), float(rmse_r), K1, D1, K2, D2, R, T, E, F,
                          ref_baseline, max_rmse)

    def _run_charuco_calibration(self) -> None:
        n = len(self._charuco_cl)
        self._log(f"[ChArUco] Sztereó kalibrálás: {n} képpár, képméret: {self._image_size}")
        self._log("  API: OpenCV 5.0.0 – board.matchImagePoints() + cv2.calibrateCamera()")

        f_px = float(self._geo_cfg.get("focal_length_px", 1365.2))
        cx   = float(self._geo_cfg.get("principal_point_x", 968.0))
        cy_  = float(self._geo_cfg.get("principal_point_y", 608.0))
        ref_baseline = float(self._geo_cfg.get("baseline_mm", 2200.0))
        max_rmse     = float(self._stereo_cfg.get("max_acceptable_rmse_px", 1.0))

        K_init = np.array([[f_px, 0.0, cx], [0.0, f_px, cy_], [0.0, 0.0, 1.0]], dtype=np.float64)

        # ─── matchImagePoints: ChArUco corners → (obj_pts, img_pts) párok ─────
        # OpenCV 5.0.0: calibrateCameraCharuco el lett távolítva.
        # board.matchImagePoints(charuco_corners, charuco_ids) adja vissza a
        # 3D objektumpontokat és 2D képpontokat, amelyek cv2.calibrateCamera()-ba mennek.
        self._log("[0/3] ChArUco sarokpontok konvertálása obj/img pontpárokra (matchImagePoints)...")

        obj_pts_l, img_pts_l = [], []
        obj_pts_r, img_pts_r = [], []
        obj_pts_stereo       = []
        pts_l_stereo         = []
        pts_r_stereo         = []

        for i, (cl, cr, il, ir) in enumerate(zip(
            self._charuco_cl, self._charuco_cr,
            self._charuco_il, self._charuco_ir
        )):
            # Bal kamera
            try:
                obj_l, img_l = self._charuco_board.matchImagePoints(cl, il)
            except Exception as e:
                self._log(f"  [skip-L] képpár #{i+1}: matchImagePoints hiba: {e}")
                continue

            # Jobb kamera
            try:
                obj_r, img_r = self._charuco_board.matchImagePoints(cr, ir)
            except Exception as e:
                self._log(f"  [skip-R] képpár #{i+1}: matchImagePoints hiba: {e}")
                continue

            if obj_l is None or obj_r is None or len(obj_l) < 4 or len(obj_r) < 4:
                self._log(f"  [skip] képpár #{i+1}: túl kevés sarokpont (bal:{len(obj_l) if obj_l is not None else 0}, jobb:{len(obj_r) if obj_r is not None else 0})")
                continue

            # (N, 3) float32 → (N, 1, 3) az OpenCV calibrateCamera-hoz
            obj_l_3d = np.ascontiguousarray(obj_l.reshape(-1, 1, 3), dtype=np.float32)
            img_l_2d = np.ascontiguousarray(img_l.reshape(-1, 1, 2), dtype=np.float32)
            obj_r_3d = np.ascontiguousarray(obj_r.reshape(-1, 1, 3), dtype=np.float32)
            img_r_2d = np.ascontiguousarray(img_r.reshape(-1, 1, 2), dtype=np.float32)

            obj_pts_l.append(obj_l_3d)
            img_pts_l.append(img_l_2d)
            obj_pts_r.append(obj_r_3d)
            img_pts_r.append(img_r_2d)

            # Sztereó kalibráláshoz: közös obj_pts (bal == jobb ha mindkettő lefedett)
            # Csak a közös (metszet) sarokpontokat vesszük
            ids_l = il.flatten()
            ids_r = ir.flatten()
            common_ids = np.intersect1d(ids_l, ids_r)
            if len(common_ids) >= 4:
                idx_l = np.array([np.where(ids_l == cid)[0][0] for cid in common_ids])
                idx_r = np.array([np.where(ids_r == cid)[0][0] for cid in common_ids])

                # Közös obj_pts a board-ból (ID alapján)
                board_obj_pts = self._charuco_board.getChessboardCorners()
                common_obj = board_obj_pts[common_ids].reshape(-1, 1, 3).astype(np.float32)
                common_l   = cl[idx_l].reshape(-1, 1, 2).astype(np.float32)
                common_r   = cr[idx_r].reshape(-1, 1, 2).astype(np.float32)
                obj_pts_stereo.append(common_obj)
                pts_l_stereo.append(common_l)
                pts_r_stereo.append(common_r)

        n_l = len(obj_pts_l)
        n_r = len(obj_pts_r)
        n_s = len(obj_pts_stereo)
        self._log(f"  Érvényes: bal={n_l}, jobb={n_r}, sztereó={n_s} képpár")

        if n_l < 5 or n_r < 5:
            raise RuntimeError(
                f"Túl kevés érvényes képpár (bal:{n_l}, jobb:{n_r}). Min. 5 szükséges!"
            )
        if n_s < 5:
            raise RuntimeError(
                f"Csak {n_s} képpár alkalmas a sztereó kalibráláshoz (min. 5 kell)!"
            )

        # ─── [1/3] Bal kamera kalibrálás ─────────────────────────────────────
        self._log("[1/3] Bal kamera kalibrálás (cv2.calibrateCamera)...")
        rmse_l, K1, D1, rvecs_l, tvecs_l = cv2.calibrateCamera(
            obj_pts_l, img_pts_l, self._image_size,
            K_init.copy(), None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO,
        )
        self._log(f"  Bal RMSE: {rmse_l:.4f} px")

        # ─── [2/3] Jobb kamera kalibrálás ────────────────────────────────────
        self._log("[2/3] Jobb kamera kalibrálás (cv2.calibrateCamera)...")
        rmse_r, K2, D2, rvecs_r, tvecs_r = cv2.calibrateCamera(
            obj_pts_r, img_pts_r, self._image_size,
            K_init.copy(), None,
            flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO,
        )
        self._log(f"  Jobb RMSE: {rmse_r:.4f} px")

        # ─── [3/3] Sztereó kalibrálás ─────────────────────────────────────────
        self._log(f"[3/3] Sztereó kalibrálás (R, T) – {n_s} képpár...")
        rmse_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            obj_pts_stereo, pts_l_stereo, pts_r_stereo,
            K1, D1, K2, D2, self._image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        self._emit_result(rmse_s, float(rmse_l), float(rmse_r), K1, D1, K2, D2, R, T, E, F,
                          ref_baseline, max_rmse)


    def _emit_result(self, rmse_s, rmse_l, rmse_r, K1, D1, K2, D2, R, T, E, F,
                     ref_baseline, max_rmse) -> None:
        if rmse_s > 1.0:
            self._log(f"  ⚠ Magas RMSE ({rmse_s:.2f} px)! Probálj élesebb, változatosabb pózú képpárokat.")
        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, self._image_size, R, T,
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
        )
        baseline_mm   = float(np.linalg.norm(T))
        baseline_diff = abs(baseline_mm - ref_baseline)
        self._log(f"  Mért baseline: {baseline_mm:.1f} mm (ref: {ref_baseline:.0f} mm, eltérés: {baseline_diff:.1f} mm)")
        quality = "KIVÁLÓ ✓" if rmse_s < 0.5 else ("JÓ ✓" if rmse_s < max_rmse else "GYENGE ⚠")
        self._log(f"Minőség: {quality} | RMSE: {rmse_s:.4f} px | Baseline: {baseline_mm:.1f} mm")
        self.finished.emit({
            "K1": K1, "D1": D1, "K2": K2, "D2": D2, "R": R, "T": T, "E": E, "F": F,
            "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
            "rmse": rmse_s, "rmse_left": rmse_l, "rmse_right": rmse_r,
            "baseline_mm": baseline_mm, "quality": quality,
            "image_width": self._image_size[0], "image_height": self._image_size[1],
        })

    def run(self) -> None:
        try:
            if self._board_type == BoardType.CHARUCO:
                self._run_charuco_calibration()
            else:
                self._run_chessboard_calibration()
        except Exception as exc:
            logger.exception("Kalibrálás hiba: %s", exc)
            self.error_occurred.emit(str(exc))


# --------------------------------------------------------------------------- #
# Fő Kalibrációs Dialog
# --------------------------------------------------------------------------- #

class CalibrationDialog(QDialog):
    """
    Négyfüles kalibrációs munkafolyamat:
      ① Beállítások – Chessboard / ChArUco paraméterek, geometria, kimeneti fájl
      ② Pozíció beállítás – Élő kamera + HUD vizuális célkeresztekkel
      ③ Képrögzítés – Élő kamera + tábla overlay + képpár mentés
      ④ Kalibrálás  – futtatás, log, eredmény, mentés .npz-be
    """

    def __init__(self, config: dict, is_dark: bool = False, parent=None):
        super().__init__(parent)
        self._config         = config
        self._is_dark        = is_dark
        self._capture_worker: Optional[CalibrationCaptureWorker] = None
        self._run_worker:     Optional[CalibrationRunWorker]     = None
        self._last_result:    Optional[dict]                     = None

        self.setWindowTitle("DEIK – Sztereó Kalibrációs Munkafolyamat")
        self.setMinimumSize(900, 620)

        if self._is_dark:
            self.setStyleSheet(
                "QDialog, QWidget { background-color: #0B0F17; color: #F8FAFC; "
                "font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }"
                "QGroupBox { background-color: #151D2A; border: 1px solid #26334D; "
                "border-radius: 8px; margin-top: 18px; padding: 16px 12px 12px 12px; font-weight: 700; color: #F8FAFC; }"
                "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; "
                "left: 12px; top: 0px; padding: 4px 12px; background-color: #0F5132; color: #FFFFFF; "
                "border-radius: 4px; font-size: 12px; font-weight: 800; }"
                "QTabWidget::pane { border: 1px solid #26334D; border-radius: 6px; background: #0B0F17; }"
                "QTabBar::tab { background: #151D2A; color: #94A3B8; border: 1px solid #26334D; "
                "padding: 10px 22px; border-top-left-radius: 6px; border-top-right-radius: 6px; "
                "font-weight: 700; font-size: 13px; margin-right: 4px; }"
                "QTabBar::tab:selected { background: #0F5132; color: #FFFFFF; border-bottom: none; }"
                "QTabBar::tab:hover:!selected { background: #1E293B; color: #10B981; }"
                "QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox { border: 1px solid #26334D; "
                "border-radius: 6px; padding: 5px 10px; background: #1E293B; min-height: 28px; color: #F8FAFC; font-weight: 600; }"
                "QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus { border: 2px solid #10B981; }"
                "QProgressBar { border: 1px solid #26334D; border-radius: 6px; background: #151D2A; "
                "height: 24px; text-align: center; font-weight: 700; color: #F8FAFC; }"
                "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                "stop:0 #0F5132, stop:1 #10B981); border-radius: 5px; }"
                "QPlainTextEdit { background: #020617; color: #10B981; "
                "font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; "
                "border-radius: 6px; padding: 8px; border: 1px solid #26334D; }"
                "QLabel { color: #F8FAFC; }"
            )
        else:
            self.setStyleSheet(
                "QDialog, QWidget { background-color: #F8FAFC; color: #0F172A; "
                "font-family: 'Segoe UI', Arial, sans-serif; font-size: 13px; }"
                "QGroupBox { background-color: #FFFFFF; border: 1px solid #CBD5E1; "
                "border-radius: 8px; margin-top: 18px; padding: 16px 12px 12px 12px; font-weight: 700; }"
                "QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left; "
                "left: 12px; top: 0px; padding: 4px 12px; background-color: #0F5132; color: #FFFFFF; "
                "border-radius: 4px; font-size: 12px; font-weight: 800; }"
                "QTabWidget::pane { border: 1px solid #CBD5E1; border-radius: 6px; background: #F8FAFC; }"
                "QTabBar::tab { background: #E2E8F0; color: #334155; border: 1px solid #CBD5E1; "
                "padding: 10px 22px; border-top-left-radius: 6px; border-top-right-radius: 6px; "
                "font-weight: 700; font-size: 13px; margin-right: 4px; }"
                "QTabBar::tab:selected { background: #0F5132; color: #FFFFFF; border-bottom: none; }"
                "QTabBar::tab:hover:!selected { background: #CBD5E1; color: #0F5132; }"
                "QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox { border: 1px solid #CBD5E1; "
                "border-radius: 6px; padding: 5px 10px; background: #FFFFFF; min-height: 28px; color: #0F172A; font-weight: 600; }"
                "QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus, QComboBox:focus { border: 2px solid #0F5132; }"
                "QProgressBar { border: 1px solid #CBD5E1; border-radius: 6px; background: #F1F5F9; "
                "height: 24px; text-align: center; font-weight: 700; color: #0F172A; }"
                "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
                "stop:0 #0F5132, stop:1 #16A34A); border-radius: 5px; }"
                "QPlainTextEdit { background: #0F172A; color: #A3E635; "
                "font-family: 'Consolas', 'Courier New', monospace; font-size: 12px; "
                "border-radius: 6px; padding: 8px; }"
                "QLabel { color: #0F172A; }"
            )
        self._build_ui()

    def _wrap_in_scroll(self, widget: QWidget) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(widget)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")
        return scroll

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        hdr = QLabel("⚙  DEIK Sztereó Kalibrációs Munkafolyamat")
        hdr.setStyleSheet("font-size: 18px; font-weight: 900; color: #4ADE80; padding: 2px 0;" if self._is_dark else "font-size: 18px; font-weight: 900; color: #0F5132; padding: 2px 0;")
        root.addWidget(hdr)

        sub = QLabel("Lépések: ① Beállítások  →  ② Pozíció beállítás  →  ③ Képrögzítés  →  ④ Kalibrálás.")
        sub.setStyleSheet("color: #94A3B8; font-size: 12px; font-weight: 600;" if self._is_dark else "color: #475569; font-size: 12px; font-weight: 600;")
        root.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #26334D;" if self._is_dark else "color: #CBD5E1;")
        root.addWidget(sep)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab_settings(),  "① Beállítások")
        self._tabs.addTab(self._build_tab_alignment(), "② Pozíció beállítás")
        self._tabs.addTab(self._build_tab_capture(),   "③ Képrögzítés")
        self._tabs.addTab(self._build_tab_run(),       "④ Kalibrálás")
        root.addWidget(self._tabs, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QPushButton("✕  Bezárás")
        btn_close.setStyleSheet(_BTN_SEC_DARK if self._is_dark else _BTN_SEC)
        btn_close.clicked.connect(self._on_close_clicked)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    def _flbl(self, txt: str) -> QLabel:
        l = QLabel(txt)
        l.setStyleSheet("font-weight: 700; color: #F8FAFC; font-size: 13px;" if self._is_dark else "font-weight: 700; color: #0F172A; font-size: 13px;")
        return l

    # --- ① Beállítások ---

    def _build_tab_settings(self) -> QWidget:
        w  = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(16, 16, 16, 16)
        ly.setSpacing(16)

        stereo_cfg = self._config.get("stereo", {})
        cb_cfg     = stereo_cfg.get("chessboard", {})
        geo_cfg    = self._config.get("geometry", {})

        # --- Tábla típus ---
        type_grp  = QGroupBox("Kalibrációs Tábla Típusa")
        type_form = QFormLayout(type_grp)
        type_form.setSpacing(12)
        self._combo_board_type = QComboBox()
        self._combo_board_type.addItem("ChArUco  (ArUco marker + sakktábla hibrid) – AJÁNLOTT", "charuco")
        self._combo_board_type.addItem("Chessboard  (hagyományos sakktábla)", "chessboard")
        default_type = cb_cfg.get("board_type", "charuco")
        self._combo_board_type.setCurrentIndex(0 if default_type == "charuco" else 1)
        self._combo_board_type.currentIndexChanged.connect(self._on_board_type_changed)
        type_form.addRow(self._flbl("Tábla típus:"), self._combo_board_type)
        ly.addWidget(type_grp)

        # --- Tábla paraméterek ---
        cb_grp  = QGroupBox("Tábla Paraméterek")
        cb_form = QFormLayout(cb_grp)
        cb_form.setSpacing(12)

        self._info_label = QLabel("")
        self._info_label.setWordWrap(True)
        self._info_label.setStyleSheet(
            "background: #064E3B; border: 1px solid #10B981; border-radius: 6px; padding: 10px; color: #D1FAE5; font-size: 13px;"
            if self._is_dark else
            "background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 6px; padding: 10px; color: #065F46; font-size: 13px;"
        )
        cb_form.addRow("", self._info_label)

        self._spin_cols = QSpinBox()
        self._spin_cols.setRange(3, 30)
        self._spin_cols.setValue(int(cb_cfg.get("cols", 12)))
        self._spin_cols.setToolTip("Vízszintes négyzetek száma (pl. 12)")
        self._spin_cols.setMinimumWidth(100)

        self._spin_rows = QSpinBox()
        self._spin_rows.setRange(3, 30)
        self._spin_rows.setValue(int(cb_cfg.get("rows", 9)))
        self._spin_rows.setToolTip("Függőleges négyzetek száma (pl. 9)")
        self._spin_rows.setMinimumWidth(100)

        lbl_cols = QLabel("Vízszintes (cols):")
        lbl_cols.setStyleSheet("font-weight: 600; color: #CBD5E1;" if self._is_dark else "font-weight: 600; color: #334155;")
        lbl_rows = QLabel("Függőleges (rows):")
        lbl_rows.setStyleSheet("font-weight: 600; color: #CBD5E1;" if self._is_dark else "font-weight: 600; color: #334155;")

        size_row = QHBoxLayout()
        size_row.addWidget(lbl_cols)
        size_row.addWidget(self._spin_cols)
        size_row.addSpacing(20)
        size_row.addWidget(lbl_rows)
        size_row.addWidget(self._spin_rows)
        size_row.addStretch(1)
        cb_form.addRow(self._flbl("Tábla méret:"), size_row)

        self._spin_sq = QDoubleSpinBox()
        self._spin_sq.setRange(1.0, 300.0)
        self._spin_sq.setDecimals(1)
        self._spin_sq.setSuffix(" mm")
        self._spin_sq.setValue(float(cb_cfg.get("square_size_mm", 65.0)))
        self._spin_sq.setMinimumWidth(140)
        cb_form.addRow(self._flbl("Négyzetméret:"), self._spin_sq)

        self._lbl_marker = self._flbl("ArUco marker méret:")
        self._spin_marker = QDoubleSpinBox()
        self._spin_marker.setRange(1.0, 290.0)
        self._spin_marker.setDecimals(1)
        self._spin_marker.setSuffix(" mm")
        self._spin_marker.setValue(float(cb_cfg.get("marker_size_mm", 50.0)))
        self._spin_marker.setMinimumWidth(140)
        cb_form.addRow(self._lbl_marker, self._spin_marker)

        self._lbl_aruco_dict = self._flbl("ArUco szótár:")
        self._combo_aruco_dict = QComboBox()
        for dict_name in _ARUCO_DICT_MAP.keys():
            self._combo_aruco_dict.addItem(dict_name, dict_name)
        default_dict = cb_cfg.get("aruco_dict", "DICT_6X6_250")
        dict_idx = self._combo_aruco_dict.findData(default_dict)
        if dict_idx >= 0:
            self._combo_aruco_dict.setCurrentIndex(dict_idx)
        cb_form.addRow(self._lbl_aruco_dict, self._combo_aruco_dict)

        self._spin_min = QSpinBox()
        self._spin_min.setRange(5, 100)
        self._spin_min.setValue(int(stereo_cfg.get("min_calibration_frames", 20)))
        self._spin_min.setToolTip("Min. 15-20 képpár ajánlott")
        self._spin_min.setMinimumWidth(140)
        cb_form.addRow(self._flbl("Min. képpárok:"), self._spin_min)
        ly.addWidget(cb_grp)

        # --- Geometria (csak tájékoztató) ---
        geo_grp  = QGroupBox("Kamera Geometria (Jelenlegi Konfiguráció)")
        geo_form = QFormLayout(geo_grp)
        geo_form.setSpacing(10)

        def glbl(v, u="mm"):
            l = QLabel(f"{v} {u}")
            l.setStyleSheet(
                "background: #1E293B; color: #10B981; font-weight: 800; font-size: 13px; font-family: Consolas, monospace; border: 1px solid #26334D; border-radius: 4px; padding: 3px 10px;"
                if self._is_dark else
                "background: #F1F5F9; color: #0F5132; font-weight: 800; font-size: 13px; font-family: Consolas, monospace; border: 1px solid #CBD5E1; border-radius: 4px; padding: 3px 10px;"
            )
            return l

        geo_form.addRow(self._flbl("Bal kamera X:"),    glbl(geo_cfg.get("left_camera_x_mm",  -1100.0)))
        geo_form.addRow(self._flbl("Jobb kamera X:"),   glbl(geo_cfg.get("right_camera_x_mm",  1100.0)))
        geo_form.addRow(self._flbl("Baseline:"),        glbl(geo_cfg.get("baseline_mm",        2200.0)))
        geo_form.addRow(self._flbl("Kamera magasság:"), glbl(geo_cfg.get("camera_height_mm",   2900.0)))
        geo_form.addRow(self._flbl("Kapu szélesség:"),  glbl(geo_cfg.get("goal_width_mm",      4000.0)))
        geo_form.addRow(self._flbl("Kapu magasság:"),   glbl(geo_cfg.get("goal_height_mm",     2000.0)))
        ly.addWidget(geo_grp)

        # --- Kimeneti fájl ---
        out_grp  = QGroupBox("Kalibrációs Fájl Mentési Helye")
        out_form = QFormLayout(out_grp)
        out_form.setSpacing(10)
        default_out = str(
            (Path(__file__).parent.parent.parent /
             stereo_cfg.get("calibration_file", "data/calibration/stereo_calibration.npz")
            ).resolve()
        )
        self._edit_out = QLineEdit(default_out)
        btn_browse = QPushButton("Tallózás…")
        btn_browse.setStyleSheet(_BTN_SEC_DARK if self._is_dark else _BTN_SEC)
        btn_browse.clicked.connect(self._on_browse)
        row_out = QHBoxLayout()
        row_out.addWidget(self._edit_out, stretch=3)
        row_out.addWidget(btn_browse)
        out_form.addRow(self._flbl("Fájl:"), row_out)
        ly.addWidget(out_grp)

        btn_apply = QPushButton("✔  Beállítások Alkalmazása")
        btn_apply.setStyleSheet(_BTN_PRIMARY)
        btn_apply.setMinimumHeight(38)
        btn_apply.clicked.connect(self._apply_settings)
        ly.addWidget(btn_apply)
        ly.addStretch(1)

        self._update_board_type_ui()
        return self._wrap_in_scroll(w)

    def _update_board_type_ui(self) -> None:
        is_charuco = (self._combo_board_type.currentData() == "charuco")
        self._lbl_marker.setVisible(is_charuco)
        self._spin_marker.setVisible(is_charuco)
        self._lbl_aruco_dict.setVisible(is_charuco)
        self._combo_aruco_dict.setVisible(is_charuco)

        cols = self._spin_cols.value()
        rows = self._spin_rows.value()
        sq   = self._spin_sq.value()

        if is_charuco:
            mk        = self._spin_marker.value()
            dict_name = self._combo_aruco_dict.currentData() or "DICT_6X6_250"
            info_html = (
                f"ℹ  <b>ChArUco mód</b> – Tábla: <b>{cols}×{rows}</b> négyzet, "
                f"négyzetméret: <b>{sq:.1f} mm</b>, marker: <b>{mk:.1f} mm</b><br>"
                f"Szótár: <b>{dict_name}</b><br>"
                f"A ChArUco tábla <b>részlegesen is detektálható</b> (min. 4 sarok elég). "
                f"Összesen <b>{(cols-1)*(rows-1)}</b> belső sarokpont."
            )
        else:
            info_html = (
                f"ℹ  <b>Chessboard mód</b> – Tábla: <b>{cols}×{rows}</b> négyzet, "
                f"négyzetméret: <b>{sq:.1f} mm</b><br>"
                f"OpenCV belső sarokpontok: <b>{cols-1}×{rows-1}</b>  (négyzetek – 1 per tengely)."
            )
        self._info_label.setText(info_html)

    @pyqtSlot(int)
    def _on_board_type_changed(self, _: int) -> None:
        self._update_board_type_ui()

    # --- ② Pozíció beállítás ---

    def _build_tab_alignment(self) -> QWidget:
        w  = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(12, 12, 12, 12)
        ly.setSpacing(10)

        hdr_grp = QGroupBox("Kamera Pozícionálási Állapot és Pontszám")
        hdr_ly  = QVBoxLayout(hdr_grp)
        hdr_ly.setContentsMargins(10, 14, 10, 10)
        hdr_ly.setSpacing(8)

        top_row = QHBoxLayout()
        self._lbl_align_status = QLabel("⏸ Kamera inaktív – Indítsd el a kamerát a pozíció beállításához")
        self._lbl_align_status.setStyleSheet(
            "background: #1E293B; color: #94A3B8; font-weight: 700; border-radius: 6px; padding: 8px 14px; font-size: 13px;"
            if self._is_dark else
            "background: #F1F5F9; color: #475569; font-weight: 700; border-radius: 6px; padding: 8px 14px; font-size: 13px;"
        )
        self._lbl_align_score = QLabel(" Pontszám: — % ")
        self._lbl_align_score.setStyleSheet(
            "font-weight: 900; font-size: 15px; color: #4ADE80; padding: 4px 14px; background: #064E3B; border: 1px solid #10B981; border-radius: 6px;"
            if self._is_dark else
            "font-weight: 900; font-size: 15px; color: #0F5132; padding: 4px 14px; background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 6px;"
        )
        top_row.addWidget(self._lbl_align_status, stretch=1)
        top_row.addWidget(self._lbl_align_score)
        hdr_ly.addLayout(top_row)

        self._align_progress = QProgressBar()
        self._align_progress.setRange(0, 100)
        self._align_progress.setValue(0)
        self._align_progress.setFormat("Illeszkedési minőség: %v%")
        hdr_ly.addWidget(self._align_progress)
        ly.addWidget(hdr_grp)

        cam_grp = QGroupBox("Élő Dual Kamera Nézet  —  Vizuális Célkeresztekkel és Utasításokkal")
        cam_box = QVBoxLayout(cam_grp)
        cam_box.setContentsMargins(6, 18, 6, 6)
        self._align_cam_lbl = QLabel(
            "A kamera élő képe és a HUD segédvonalak itt jelennek meg.\n\n"
            "• Helyezd a kalibrációs táblát pontosan a pálya közepére.\n"
            "• Döntsd és forgass a kamerákon az alábbi utasítások alapján."
        )
        self._align_cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._align_cam_lbl.setStyleSheet(
            "background: #0F172A; color: #94A3B8; border-radius: 8px; font-size: 13px; font-weight: 600; padding: 20px;"
        )
        self._align_cam_lbl.setFixedHeight(340)
        self._align_cam_lbl.setScaledContents(False)
        self._align_cam_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cam_box.addWidget(self._align_cam_lbl)
        ly.addWidget(cam_grp)

        instr_row = QHBoxLayout()
        instr_row.setSpacing(12)

        left_grp = QGroupBox("📷 BAL KAMERA IGAZÍTÁSI UTASÍTÁSOK")
        self._left_instr_box = QVBoxLayout(left_grp)
        self._lbl_left_instr = QLabel("Indítsd el a kamerát a méréshez…")
        self._lbl_left_instr.setStyleSheet("color: #94A3B8; font-weight: 600; padding: 6px;" if self._is_dark else "color: #475569; font-weight: 600; padding: 6px;")
        self._lbl_left_instr.setWordWrap(True)
        self._left_instr_box.addWidget(self._lbl_left_instr)
        instr_row.addWidget(left_grp, stretch=1)

        right_grp = QGroupBox("📷 JOBB KAMERA IGAZÍTÁSI UTASÍTÁSOK")
        self._right_instr_box = QVBoxLayout(right_grp)
        self._lbl_right_instr = QLabel("Indítsd el a kamerát a méréshez…")
        self._lbl_right_instr.setStyleSheet("color: #94A3B8; font-weight: 600; padding: 6px;" if self._is_dark else "color: #475569; font-weight: 600; padding: 6px;")
        self._lbl_right_instr.setWordWrap(True)
        self._right_instr_box.addWidget(self._lbl_right_instr)
        instr_row.addWidget(right_grp, stretch=1)

        ly.addLayout(instr_row)

        ctrl = QHBoxLayout()
        self._btn_start_align = QPushButton("▶  Kamerák Indítása")
        self._btn_start_align.setStyleSheet(_BTN_PRIMARY)
        self._btn_start_align.setMinimumHeight(38)
        self._btn_start_align.clicked.connect(self._on_start_camera)

        self._btn_stop_align = QPushButton("⏹  Leállítás")
        self._btn_stop_align.setStyleSheet(_BTN_SEC_DARK if self._is_dark else _BTN_SEC)
        self._btn_stop_align.setMinimumHeight(38)
        self._btn_stop_align.setEnabled(False)
        self._btn_stop_align.clicked.connect(self._on_stop_camera)

        self._btn_to_capture = QPushButton("➡  Tovább a Képrögzítéshez")
        self._btn_to_capture.setStyleSheet(_BTN_PRIMARY)
        self._btn_to_capture.setMinimumHeight(38)
        self._btn_to_capture.clicked.connect(lambda: self._tabs.setCurrentIndex(2))

        ctrl.addWidget(self._btn_start_align)
        ctrl.addWidget(self._btn_stop_align)
        ctrl.addStretch(1)
        ctrl.addWidget(self._btn_to_capture)
        ly.addLayout(ctrl)

        return self._wrap_in_scroll(w)

    def _update_alignment_ui(self, res: AlignmentResult) -> None:
        self._lbl_align_score.setText(f" Pontszám: {res.score:.0f}% ")
        self._align_progress.setValue(int(res.score))
        self._lbl_align_status.setText(res.general_summary)
        self._lbl_align_status.setStyleSheet(
            f"background: {res.status_color}22; color: {res.status_color}; font-weight: 800; "
            f"border: 1px solid {res.status_color}; border-radius: 6px; padding: 8px 14px; font-size: 13px;"
        )
        text_color = "#F8FAFC" if self._is_dark else "#0F172A"

        def _html(instrs):
            html = ""
            for ins in instrs:
                color = "#10B981" if ins.severity == "ok" else ("#D97706" if ins.severity == "warning" else "#DC2626")
                html += (
                    f"<div style='margin-bottom: 6px; font-size: 13px; color: {text_color};'>"
                    f"<span style='font-size: 16px;'>{ins.icon}</span> &nbsp;"
                    f"<b style='color: {color};'>[{ins.value_str}]</b> {ins.text}"
                    f"</div>"
                )
            return html or "—"

        self._lbl_left_instr.setText(_html(res.left_instructions))
        self._lbl_right_instr.setText(_html(res.right_instructions))

    # --- ③ Képrögzítés ---

    def _build_tab_capture(self) -> QWidget:
        w  = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(12, 12, 12, 12)
        ly.setSpacing(10)

        status_row = QHBoxLayout()
        self._lbl_status = QLabel("⏸  Kamera inaktív – kattints a START gombra")
        self._lbl_status.setStyleSheet(
            "background: #451A03; color: #FCD34D; font-weight: 700; border-radius: 6px; padding: 8px 14px; font-size: 13px;"
            if self._is_dark else
            "background: #FEF3C7; color: #92400E; font-weight: 700; border-radius: 6px; padding: 8px 14px; font-size: 13px;"
        )
        self._lbl_count = QLabel("0 / 20  képpár")
        self._lbl_count.setStyleSheet(
            "font-weight: 900; font-size: 14px; color: #4ADE80; padding: 4px 12px; background: #064E3B; border: 1px solid #10B981; border-radius: 6px;"
            if self._is_dark else
            "font-weight: 900; font-size: 14px; color: #0F5132; padding: 4px 12px; background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 6px;"
        )
        status_row.addWidget(self._lbl_status, stretch=1)
        status_row.addWidget(self._lbl_count)
        ly.addLayout(status_row)

        self._progress = QProgressBar()
        self._progress.setRange(0, 20)
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m képpár rögzítve")
        ly.addWidget(self._progress)

        l_cfg   = self._config.get("camera", {}).get("left", {})
        r_cfg   = self._config.get("camera", {}).get("right", {})
        def_cfg = self._config.get("camera", {})
        l_exp  = l_cfg.get("exposure_time_us", def_cfg.get("exposure_time_us", 3000))
        l_gain = l_cfg.get("gain_db", def_cfg.get("gain_db", 0.0))
        r_exp  = r_cfg.get("exposure_time_us", def_cfg.get("exposure_time_us", 3000))
        r_gain = r_cfg.get("gain_db", def_cfg.get("gain_db", 0.0))
        lbl_cam_info = QLabel(
            f"📷 <b>A főoldalon beállított kamera paraméterek:</b> &nbsp; "
            f"<b>BAL:</b> {l_exp} µs / {l_gain:.1f} dB &nbsp;|&nbsp; "
            f"<b>JOBB:</b> {r_exp} µs / {r_gain:.1f} dB"
        )
        lbl_cam_info.setStyleSheet(
            "background: #1E293B; color: #4ADE80; border: 1px solid #26334D; border-radius: 6px; padding: 6px 12px; font-size: 12px;"
            if self._is_dark else
            "background: #F1F5F9; color: #0F5132; border: 1px solid #CBD5E1; border-radius: 6px; padding: 6px 12px; font-size: 12px;"
        )
        ly.addWidget(lbl_cam_info)

        cam_grp = QGroupBox("Élő Kamera Kép  —  Bal  |  Jobb")
        cam_box = QVBoxLayout(cam_grp)
        cam_box.setContentsMargins(6, 18, 6, 6)
        self._cam_lbl = QLabel(
            "A kamera élő képe itt jelenik meg.\n\n"
            "Zöld sarokpontok = tábla mindkét kamerában látható → SPACE a mentéshez.\n"
            "Kék = csak az egyik kamerában látszik."
        )
        self._cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_lbl.setStyleSheet(
            "background: #0F172A; color: #94A3B8; border-radius: 8px; font-size: 13px; font-weight: 600; padding: 20px;"
        )
        self._cam_lbl.setFixedHeight(360)
        self._cam_lbl.setScaledContents(False)
        self._cam_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        cam_box.addWidget(self._cam_lbl)
        ly.addWidget(cam_grp, stretch=1)

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        self._btn_start = QPushButton("▶  Kamerák Indítása")
        self._btn_start.setStyleSheet(_BTN_PRIMARY)
        self._btn_start.setMinimumHeight(38)
        self._btn_start.clicked.connect(self._on_start_camera)

        self._btn_stop = QPushButton("⏹  Leállítás")
        self._btn_stop.setStyleSheet(_BTN_SEC_DARK if self._is_dark else _BTN_SEC)
        self._btn_stop.setMinimumHeight(38)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_camera)

        self._btn_cap = QPushButton("📸  KÉPPÁR MENTÉSE  [SPACE]")
        self._btn_cap.setStyleSheet(_BTN_CAPTURE)
        self._btn_cap.setMinimumHeight(38)
        self._btn_cap.setEnabled(False)
        self._btn_cap.clicked.connect(self._on_capture_click)

        self._btn_clear = QPushButton("🗑  Képpárok Törlése")
        self._btn_clear.setStyleSheet(_BTN_WARN)
        self._btn_clear.setMinimumHeight(38)
        self._btn_clear.setEnabled(False)
        self._btn_clear.clicked.connect(self._on_clear)

        ctrl.addWidget(self._btn_start)
        ctrl.addWidget(self._btn_stop)
        ctrl.addStretch(1)
        ctrl.addWidget(self._btn_cap)
        ctrl.addWidget(self._btn_clear)
        ly.addLayout(ctrl)

        tip = QLabel(
            "💡  Tartsd a kalibrációs táblát különböző szögekben és távolságokban (0.5 – 3 m). "
            "Min. 15–20 képpár szükséges. "
            "ChArUco módban a részlegesen látható tábla is elfogadott!"
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #94A3B8; font-size: 12px; padding: 4px; font-weight: 500;" if self._is_dark else "color: #475569; font-size: 12px; padding: 4px; font-weight: 500;")
        ly.addWidget(tip)

        return w

    # --- ④ Kalibrálás ---

    def _build_tab_run(self) -> QWidget:
        w  = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(14, 14, 14, 14)
        ly.setSpacing(12)

        btn_row = QHBoxLayout()
        self._btn_run = QPushButton("🔬  KALIBRÁLÁS INDÍTÁSA")
        self._btn_run.setStyleSheet(_BTN_PRIMARY)
        self._btn_run.setMinimumHeight(44)
        self._btn_run.setEnabled(False)
        self._btn_run.clicked.connect(self._on_run)

        self._btn_save = QPushButton("💾  Kalibrációs Fájl Mentése (.npz)")
        self._btn_save.setStyleSheet(_BTN_PRIMARY)
        self._btn_save.setMinimumHeight(44)
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)

        btn_row.addWidget(self._btn_run, stretch=1)
        btn_row.addWidget(self._btn_save, stretch=1)
        ly.addLayout(btn_row)

        res_grp  = QGroupBox("Kalibrálási Eredmény")
        res_form = QFormLayout(res_grp)
        res_form.setSpacing(10)

        def rl():
            l = QLabel("—")
            l.setStyleSheet("font-weight: 800; font-size: 14px; font-family: Consolas, monospace; color: #F8FAFC;" if self._is_dark else "font-weight: 800; font-size: 14px; font-family: Consolas, monospace; color: #0F172A;")
            return l

        def rlbl(txt):
            l = QLabel(txt)
            l.setStyleSheet("font-weight: 700; color: #CBD5E1; font-size: 13px;" if self._is_dark else "font-weight: 700; color: #0F172A; font-size: 13px;")
            return l

        self._r_rmse   = rl()
        self._r_rmse_l = rl()
        self._r_rmse_r = rl()
        self._r_base   = rl()
        self._r_k1     = rl()
        self._r_k2     = rl()
        self._r_qual   = rl()

        res_form.addRow(rlbl("Sztereó RMSE:"),      self._r_rmse)
        res_form.addRow(rlbl("Bal kamera RMSE:"),   self._r_rmse_l)
        res_form.addRow(rlbl("Jobb kamera RMSE:"),  self._r_rmse_r)
        res_form.addRow(rlbl("Mért baseline:"),     self._r_base)
        res_form.addRow(rlbl("Bal f_px (K[0,0]):"), self._r_k1)
        res_form.addRow(rlbl("Jobb f_px (K[0,0]):"), self._r_k2)
        res_form.addRow(rlbl("Minőség:"),           self._r_qual)
        ly.addWidget(res_grp)

        log_grp = QGroupBox("Kalibrálási Log")
        log_box = QVBoxLayout(log_grp)
        log_box.setContentsMargins(6, 18, 6, 6)
        self._log_txt = QPlainTextEdit()
        self._log_txt.setReadOnly(True)
        self._log_txt.setMaximumBlockCount(500)
        self._log_txt.setMinimumHeight(180)
        log_box.addWidget(self._log_txt)
        btn_clr_log = QPushButton("Log Törlése")
        btn_clr_log.setStyleSheet(_BTN_SEC_DARK if self._is_dark else _BTN_SEC)
        btn_clr_log.clicked.connect(self._log_txt.clear)
        log_box.addWidget(btn_clr_log)
        ly.addWidget(log_grp, stretch=1)

        return self._wrap_in_scroll(w)

    # ------------------------------------------------------------------ #
    # Slotok
    # ------------------------------------------------------------------ #

    def _get_current_board_type(self) -> BoardType:
        val = self._combo_board_type.currentData()
        return BoardType.CHARUCO if val == "charuco" else BoardType.CHESSBOARD

    @pyqtSlot()
    def _apply_settings(self) -> None:
        s = self._config.setdefault("stereo", {})
        c = s.setdefault("chessboard", {})
        c["board_type"]             = self._combo_board_type.currentData()
        c["cols"]                   = self._spin_cols.value()
        c["rows"]                   = self._spin_rows.value()
        c["inner_corners_x"]        = self._spin_cols.value() - 1
        c["inner_corners_y"]        = self._spin_rows.value() - 1
        c["square_size_mm"]         = self._spin_sq.value()
        c["marker_size_mm"]         = self._spin_marker.value()
        c["aruco_dict"]             = self._combo_aruco_dict.currentData()
        s["min_calibration_frames"] = self._spin_min.value()
        s["calibration_file"]       = self._edit_out.text()
        self._progress.setMaximum(self._spin_min.value())
        self._progress.setFormat(f"%v / {self._spin_min.value()} képpár rögzítve")
        self._update_board_type_ui()
        QMessageBox.information(self, "Beállítások alkalmazva",
                                "A kalibrációs tábla paraméterek frissítve a munkamenetben.")

    @pyqtSlot()
    def _on_browse(self) -> None:
        p, _ = QFileDialog.getSaveFileName(
            self, "Kalibrációs fájl mentési helye",
            self._edit_out.text(), "NumPy Archive (*.npz)"
        )
        if p:
            self._edit_out.setText(p)

    @pyqtSlot()
    def _on_start_camera(self) -> None:
        if self._capture_worker and self._capture_worker.isRunning():
            return

        self._capture_worker = CalibrationCaptureWorker(
            config          = self._config,
            board_type      = self._get_current_board_type(),
            cols            = self._spin_cols.value(),
            rows            = self._spin_rows.value(),
            square_mm       = self._spin_sq.value(),
            marker_mm       = self._spin_marker.value(),
            aruco_dict_name = self._combo_aruco_dict.currentData() or "DICT_6X6_250",
            parent          = self,
        )
        self._capture_worker.frame_ready.connect(self._on_frame)
        self._capture_worker.capture_result.connect(self._on_capture_result)
        self._capture_worker.error_occurred.connect(self._on_cam_error)
        self._capture_worker.stopped.connect(self._on_cam_stopped)
        self._capture_worker.start()

        self._btn_start.setEnabled(False)
        if hasattr(self, "_btn_start_align"):
            self._btn_start_align.setEnabled(False)
        self._btn_stop.setEnabled(True)
        if hasattr(self, "_btn_stop_align"):
            self._btn_stop_align.setEnabled(True)
        self._btn_cap.setEnabled(True)
        self._btn_clear.setEnabled(True)
        self._set_status("⏺  Kamerák aktívak – mutasd a kalibrációs táblát!", "#DCFCE7", "#14532D")

    @pyqtSlot()
    def _on_stop_camera(self) -> None:
        if self._capture_worker:
            self._capture_worker.stop()
            self._capture_worker.wait(4000)

    @pyqtSlot(np.ndarray, np.ndarray, object, object, object)
    def _on_frame(self, fl: np.ndarray, fr: np.ndarray, cl, cr, align_res: AlignmentResult) -> None:
        if hasattr(self, "_align_cam_lbl") and align_res is not None:
            fl_hud = draw_alignment_hud(fl, cl, is_left=True, result=align_res)
            fr_hud = draw_alignment_hud(fr, cr, is_left=False, result=align_res)
            if fl_hud.shape[0] > fl_hud.shape[1]:
                target_h = 400
                target_w = int(round(target_h * (fl_hud.shape[1] / fl_hud.shape[0])))
            else:
                target_w = 640
                target_h = int(round(target_w * (fl_hud.shape[0] / fl_hud.shape[1])))
            rl_hud   = cv2.resize(fl_hud, (target_w, target_h))
            rr_hud   = cv2.resize(fr_hud, (target_w, target_h))
            comb_hud = np.ascontiguousarray(np.hstack([rl_hud, rr_hud]))
            cv2.line(comb_hud, (target_w, 0), (target_w, target_h), (60, 60, 60), 2)
            hh, hw, hch = comb_hud.shape
            q_align  = QImage(bytes(comb_hud.data), hw, hh, hw * hch, QImage.Format.Format_BGR888)
            px_align = QPixmap.fromImage(q_align).scaled(760, 356,
                           Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            self._align_cam_lbl.setPixmap(px_align)
            self._update_alignment_ui(align_res)

        both  = cl is not None and cr is not None
        n     = self._capture_worker.collected_count() if self._capture_worker else 0
        min_f = self._spin_min.value()

        status_color = (0, 200, 60) if both else (30, 120, 255)
        board_label  = "ChArUco" if self._get_current_board_type() == BoardType.CHARUCO else "Sakktabla"
        txt = f"Kepparok: {n}/{min_f}  |  " + (
            f"{board_label} MINDKET KAMERABAN! -> SPACE" if both else f"{board_label} keresese..."
        )
        cv2.putText(fl, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)
        cv2.putText(fr, "SPACE=mentes", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        if fl.shape[0] > fl.shape[1]:
            target_h = 400
            target_w = int(round(target_h * (fl.shape[1] / fl.shape[0])))
        else:
            target_w = 640
            target_h = int(round(target_w * (fl.shape[0] / fl.shape[1])))
        rl       = cv2.resize(fl, (target_w, target_h))
        rr       = cv2.resize(fr, (target_w, target_h))
        combined = np.ascontiguousarray(np.hstack([rl, rr]))
        cv2.line(combined, (target_w, 0), (target_w, target_h), (60, 60, 60), 2)

        h, w, ch = combined.shape
        q_img    = QImage(bytes(combined.data), w, h, w * ch, QImage.Format.Format_BGR888)
        pixmap   = QPixmap.fromImage(q_img).scaled(760, 356,
                       Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self._cam_lbl.setPixmap(pixmap)

        self._lbl_count.setText(f"{n} / {min_f}  képpár")
        self._progress.setMaximum(min_f)
        self._progress.setValue(min(n, min_f))

        if both:
            self._set_status("✅  Tábla MINDKÉT kamerában látható!  →  SPACE = képpár mentése", "#DCFCE7", "#14532D")
        else:
            nf = int(cl is not None) + int(cr is not None)
            self._set_status(f"🔍  Tábla keresése... ({nf}/2 kamera talált)", "#FEF3C7", "#92400E")

        self._update_run_btn(n)

    def _set_status(self, txt: str, bg: str, fg: str) -> None:
        self._lbl_status.setText(txt)
        self._lbl_status.setStyleSheet(
            f"background: {bg}; color: {fg}; font-weight: 700; "
            f"border-radius: 6px; padding: 6px 12px; font-size: 13px;"
        )

    @pyqtSlot(bool, str)
    def _on_capture_result(self, success: bool, msg: str) -> None:
        if success:
            n     = self._capture_worker.collected_count() if self._capture_worker else 0
            min_f = self._spin_min.value()
            self._lbl_count.setText(f"{n} / {min_f}  képpár")
            self._progress.setValue(min(n, min_f))
            self._set_status(f"✅  {msg}", "#DCFCE7", "#14532D")
            self._update_run_btn(n)
        else:
            self._set_status(f"⚠  {msg}", "#FEF3C7", "#92400E")

    @pyqtSlot()
    def _on_capture_click(self) -> None:
        if self._capture_worker and self._capture_worker.isRunning():
            self._capture_worker.request_capture()

    @pyqtSlot()
    def _on_clear(self) -> None:
        reply = QMessageBox.question(
            self, "Képpárok törlése",
            "Biztosan törlöd az összes rögzített képpárt?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes and self._capture_worker:
            self._capture_worker.clear_collected()
            self._lbl_count.setText(f"0 / {self._spin_min.value()}  képpár")
            self._progress.setValue(0)
            self._update_run_btn(0)

    @pyqtSlot(str)
    def _on_cam_error(self, msg: str) -> None:
        self._set_status(f"❌  Hiba: {msg}", "#FEE2E2", "#991B1B")
        QMessageBox.critical(self, "Kamera hiba", msg)

    @pyqtSlot()
    def _on_cam_stopped(self) -> None:
        self._btn_start.setEnabled(True)
        if hasattr(self, "_btn_start_align"):
            self._btn_start_align.setEnabled(True)
        self._btn_stop.setEnabled(False)
        if hasattr(self, "_btn_stop_align"):
            self._btn_stop_align.setEnabled(False)
        self._btn_cap.setEnabled(False)
        self._set_status("⏸  Kamera leállítva.", "#F1F5F9", "#475569")

    def _update_run_btn(self, n: int) -> None:
        min_f = self._spin_min.value()
        ok    = n >= min_f
        self._btn_run.setEnabled(ok)
        if ok:
            self._btn_run.setText(f"🔬  KALIBRÁLÁS INDÍTÁSA  ({n} képpár)")
        else:
            self._btn_run.setText(f"🔬  Kalibrálás  (még {max(0, min_f - n)} képpár kell)")

    @pyqtSlot()
    def _on_run(self) -> None:
        if not self._capture_worker or self._capture_worker.collected_count() == 0:
            QMessageBox.warning(self, "Hiba", "Nincs rögzített kamera adat!")
            return

        n     = self._capture_worker.collected_count()
        min_f = self._spin_min.value()
        if n < min_f:
            QMessageBox.warning(self, "Nincs elég képpár", f"Minimum {min_f} szükséges, jelenleg {n} van!")
            return
        if self._capture_worker.image_size is None:
            QMessageBox.warning(self, "Hiba", "Kép méret ismeretlen!")
            return

        if self._capture_worker.isRunning():
            self._capture_worker.stop()
            self._capture_worker.wait(5000)

        self._log_txt.clear()
        self._log_append(f"Kalibrálás indítása: {n} képpár, méret: {self._capture_worker.image_size}")

        board_type = self._capture_worker._board_type

        self._run_worker = CalibrationRunWorker(
            board_type    = board_type,
            image_size    = self._capture_worker.image_size,
            geo_cfg       = self._config.get("geometry", {}),
            stereo_cfg    = self._config.get("stereo", {}),
            obj_pts       = self._capture_worker.collected_obj_pts,
            pts_l         = self._capture_worker.collected_pts_left,
            pts_r         = self._capture_worker.collected_pts_right,
            charuco_board = self._capture_worker._charuco_board,
            charuco_cl    = self._capture_worker.collected_charuco_corners_left,
            charuco_cr    = self._capture_worker.collected_charuco_corners_right,
            charuco_il    = self._capture_worker.collected_charuco_ids_left,
            charuco_ir    = self._capture_worker.collected_charuco_ids_right,
            parent        = self,
        )
        self._run_worker.log_line.connect(self._log_append)
        self._run_worker.finished.connect(self._on_cal_done)
        self._run_worker.error_occurred.connect(self._on_cal_error)
        self._run_worker.start()

        self._btn_run.setEnabled(False)
        self._btn_run.setText("⏳  Kalibrálás folyamatban...")
        self._tabs.setCurrentIndex(3)

    @pyqtSlot(str)
    def _log_append(self, msg: str) -> None:
        self._log_txt.appendPlainText(f"[{time.strftime('%H:%M:%S')}]  {msg}")
        c = self._log_txt.textCursor()
        c.movePosition(QTextCursor.MoveOperation.End)
        self._log_txt.setTextCursor(c)

    @pyqtSlot(dict)
    def _on_cal_done(self, res: dict) -> None:
        self._last_result = res
        rmse = res["rmse"]

        rc = "#16A34A" if rmse < 0.5 else ("#D97706" if rmse < 1.0 else "#DC2626")
        self._r_rmse.setText(f"{rmse:.4f} px")
        self._r_rmse.setStyleSheet(f"font-weight: 700; font-size: 13px; font-family: Consolas; color: {rc};")
        self._r_rmse_l.setText(f"{res.get('rmse_left', 0):.4f} px")
        self._r_rmse_r.setText(f"{res.get('rmse_right', 0):.4f} px")
        self._r_base.setText(f"{res['baseline_mm']:.1f} mm")
        self._r_k1.setText(f"{res['K1'][0, 0]:.2f}")
        self._r_k2.setText(f"{res['K2'][0, 0]:.2f}")
        q  = res.get("quality", "—")
        qc = "#16A34A" if "KIVÁLÓ" in q else ("#D97706" if "JÓ" in q else "#DC2626")
        self._r_qual.setText(q)
        self._r_qual.setStyleSheet(f"font-weight: 900; font-size: 14px; font-family: Consolas; color: {qc};")

        self._btn_run.setEnabled(True)
        self._btn_run.setText("🔬  Újra Kalibrál")
        self._btn_save.setEnabled(True)

        reply = QMessageBox.question(
            self, "Kalibrálás kész!",
            f"Kalibrálás sikeresen befejeződött!\n\n"
            f"  Sztereó RMSE: {rmse:.4f} px  [{q}]\n"
            f"  Mért baseline: {res['baseline_mm']:.1f} mm\n\n"
            f"Mentsd el a kalibrációs fájlt?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._on_save()

    @pyqtSlot(str)
    def _on_cal_error(self, msg: str) -> None:
        self._btn_run.setEnabled(True)
        self._btn_run.setText("🔬  KALIBRÁLÁS INDÍTÁSA")
        self._log_append(f"❌  HIBA: {msg}")
        QMessageBox.critical(self, "Kalibrálási hiba", f"Kalibrálás sikertelen:\n{msg}")

    @pyqtSlot()
    def _on_save(self) -> None:
        if not self._last_result:
            QMessageBox.warning(self, "Nincs eredmény", "Előbb futtasd a kalibrálást!")
            return
        out = Path(self._edit_out.text())
        out.parent.mkdir(parents=True, exist_ok=True)
        r = self._last_result
        try:
            np.savez(
                str(out),
                K1=r["K1"], D1=r["D1"], K2=r["K2"], D2=r["D2"],
                R=r["R"],   T=r["T"],   E=r["E"],   F=r["F"],
                R1=r["R1"], R2=r["R2"], P1=r["P1"], P2=r["P2"], Q=r["Q"],
                rmse=r["rmse"],
                image_width=r["image_width"], image_height=r["image_height"],
                baseline_mm=r["baseline_mm"],
            )
            self._log_append(f"✓ Fájl mentve: {out}")
            QMessageBox.information(
                self, "Mentés sikeres",
                f"Kalibrációs fájl mentve:\n{out}\n\n"
                f"A főprogram automatikusan betölti következő indításkor."
            )
        except Exception as exc:
            QMessageBox.critical(self, "Mentési hiba", str(exc))

    @pyqtSlot()
    def _on_close_clicked(self) -> None:
        if self._capture_worker and self._capture_worker.isRunning():
            self._capture_worker.stop()
            self._capture_worker.wait(3000)
        if self._run_worker and self._run_worker.isRunning():
            self._run_worker.wait(3000)
        self.accept()

    def closeEvent(self, event) -> None:
        self._on_close_clicked()
        event.accept()

    def keyPressEvent(self, event) -> None:
        """SPACE = képpár mentése, ha a Képrögzítés fül (index 2) aktív."""
        if event.key() == Qt.Key.Key_Space and self._tabs.currentIndex() == 2:
            self._on_capture_click()
        else:
            super().keyPressEvent(event)
