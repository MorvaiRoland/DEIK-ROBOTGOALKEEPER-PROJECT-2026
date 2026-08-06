"""
DEIK Robot Foci Kapus – Sztereó Kalibrációs Dialog (PyQt6)
==========================================================

Teljes grafikus kalibrációs műhely a két Ximea kamera sztereó kalibrálásához.

Felépítés:
    CalibrationCaptureWorker  – QThread: élő kamera loop + sarokpont detektálás
    CalibrationRunWorker      – QThread: OpenCV sztereó kalibrálás (nem fagyasztja a GUI-t)
    CalibrationDialog         – QDialog: 3 füles felület

Fül 1 – Beállítások:   Sakktábla paraméterek, minimum képpárok, kimeneti fájl
Fül 2 – Képrögzítés:   Élő kamera kép + sakktábla overlay, képpár mentés, progress
Fül 3 – Kalibrálás:    Kalibrálás indítása, log, eredmény összesítő
"""

import logging
import time
from pathlib import Path
from typing import List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
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
    QFileDialog, QMessageBox, QFrame, QSizePolicy,
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Sakktábla sarokpont keresés
# --------------------------------------------------------------------------- #

def find_chessboard_corners(
    image: np.ndarray,
    pattern_size: Tuple[int, int],
) -> Optional[np.ndarray]:
    """
    Megkeresi a sakktábla belső sarokpontjait sub-pixel pontossággal.

    Args:
        image:        BGR kép
        pattern_size: (belső_sarkok_x, belső_sarkok_y)

    Returns:
        (N, 1, 2) float32 tömb, vagy None ha nem talált
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray, pattern_size,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    if not found:
        return None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)


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
    Élő kamera loop. Frameket + detektált sarokpontokat emittál a GUI-nak.
    Belső gyűjteménybe menti a képpárokat request_capture() hívásra.

    Pózdiverzitás ellenőrzés: elutasítja a túzontosan hasonló nézőpontokat,
    mert az egyforma pózok degenárált kalibrálást okoznak!
    """

    frame_ready    = pyqtSignal(np.ndarray, np.ndarray, object, object)
    capture_result = pyqtSignal(bool, str)
    error_occurred = pyqtSignal(str)
    stopped        = pyqtSignal()

    # Minimum "különbözőségi" küszöb: a sarokpontok átlagos eltolódása (pixel)
    MIN_POSE_DIFF_PX = 25.0

    def __init__(self, config: dict, pattern_size: Tuple[int, int], parent=None):
        super().__init__(parent)
        self._config       = config
        self._pattern_size = pattern_size
        self._running      = False
        self._capture_req  = False

        # Gyűjtött adatok (a kalibráláshoz)
        self.collected_obj_pts:  List[np.ndarray] = []
        self.collected_pts_left: List[np.ndarray] = []
        self.collected_pts_right: List[np.ndarray] = []
        self.image_size: Optional[Tuple[int, int]] = None

        self._square_mm = float(
            config.get("stereo", {}).get("chessboard", {}).get("square_size_mm", 30.0)
        )

    def _make_obj_pattern(self) -> np.ndarray:
        cx, cy = self._pattern_size  # cx=8 (vízszintes), cy=6 (függőleges)
        # Standard OpenCV konvenció: np.mgrid[0:cols, 0:rows].T.reshape(-1,2)
        # ahol cols=cx (vízszintes sarkok), rows=cy (függőleges sarkok)
        # Pontsor: (0,0),(1,0),...,(cx-1,0),(0,1),...,(cx-1,cy-1)
        obj = np.zeros((cx * cy, 3), dtype=np.float32)
        obj[:, :2] = np.mgrid[0:cx, 0:cy].T.reshape(-1, 2) * self._square_mm
        return obj

    def _is_pose_too_similar(self, new_corners: np.ndarray, existing: List[np.ndarray]) -> Tuple[bool, float]:
        """Megvizsgálja, hogy az új sarokpont-készlet eléggé különbözik-e a már gyűjtött pózoktól."""
        if not existing:
            return False, 9999.0
        pts_new = new_corners.reshape(-1, 2)
        min_diff = 9999.0
        for prev in existing:
            pts_prev = prev.reshape(-1, 2)
            diff = float(np.mean(np.linalg.norm(pts_new - pts_prev, axis=1)))
            if diff < min_diff:
                min_diff = diff
        return min_diff < self.MIN_POSE_DIFF_PX, min_diff

    @pyqtSlot()
    def request_capture(self):
        """Jelez: a következő érvényes képpárt mentse."""
        self._capture_req = True

    @pyqtSlot()
    def clear_collected(self):
        self.collected_obj_pts.clear()
        self.collected_pts_left.clear()
        self.collected_pts_right.clear()

    def stop(self):
        self._running = False

    def run(self) -> None:
        self._running = True
        try:
            from camera.camera_manager import CameraManager
            cam = CameraManager(self._config)
        except Exception as exc:
            self.error_occurred.emit(f"CameraManager import hiba: {exc}")
            self.stopped.emit()
            return

        if not cam.open():
            self.error_occurred.emit("Kamerák megnyitása sikertelen! Ellenőrizd a csatlakozásokat.")
            self.stopped.emit()
            return

        obj_pattern = self._make_obj_pattern()

        try:
            while self._running:
                pair = cam.read_stereo_pair()
                if not pair.success:
                    time.sleep(0.005)
                    continue

                fl = pair.left.image.copy()
                fr = pair.right.image.copy()

                if self.image_size is None:
                    h, w = fl.shape[:2]
                    self.image_size = (w, h)

                corners_l = find_chessboard_corners(fl, self._pattern_size)
                corners_r = find_chessboard_corners(fr, self._pattern_size)
                both      = corners_l is not None and corners_r is not None

                if corners_l is not None:
                    cv2.drawChessboardCorners(fl, self._pattern_size, corners_l, True)
                if corners_r is not None:
                    cv2.drawChessboardCorners(fr, self._pattern_size, corners_r, True)

                # Képpár rögzítés pózdiverzitás-ellenőrzéssel
                if self._capture_req:
                    self._capture_req = False
                    if both:
                        too_similar_l, diff_l = self._is_pose_too_similar(corners_l, self.collected_pts_left)
                        too_similar_r, diff_r = self._is_pose_too_similar(corners_r, self.collected_pts_right)
                        too_similar = too_similar_l or too_similar_r
                        min_diff = min(diff_l, diff_r)

                        if too_similar:
                            msg = f"Túzontosan hasonló póz! (eltérés: {min_diff:.0f} px < {self.MIN_POSE_DIFF_PX:.0f} px) – mozd el a sakktáblát!"
                            logger.warning(msg)
                            self.capture_result.emit(False, msg)
                        else:
                            self.collected_obj_pts.append(obj_pattern.copy())
                            self.collected_pts_left.append(corners_l)
                            self.collected_pts_right.append(corners_r)
                            n = len(self.collected_obj_pts)
                            msg = f"Képpár mentve: {n} db (eltérés: {min_diff:.0f} px)"
                            logger.info(msg)
                            self.capture_result.emit(True, msg)
                    else:
                        msg = "Képpár nem menthető: sakktábla nem látható mindkét kamerában!"
                        logger.warning(msg)
                        self.capture_result.emit(False, msg)

                self.frame_ready.emit(fl, fr, corners_l, corners_r)

        except Exception as exc:
            logger.exception("Capture worker hiba: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            cam.close()
            self.stopped.emit()


# --------------------------------------------------------------------------- #
# Kalibrálás Futtatási Worker (QThread)
# --------------------------------------------------------------------------- #

class CalibrationRunWorker(QThread):
    """OpenCV sztereó kalibrálást futtat háttérszálban."""

    log_line       = pyqtSignal(str)
    finished       = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(
        self,
        obj_pts:    List[np.ndarray],
        pts_l:      List[np.ndarray],
        pts_r:      List[np.ndarray],
        image_size: Tuple[int, int],
        geo_cfg:    dict,
        stereo_cfg: dict,
        parent=None
    ):
        super().__init__(parent)
        self._obj_pts    = obj_pts
        self._pts_l      = pts_l
        self._pts_r      = pts_r
        self._image_size = image_size
        self._geo_cfg    = geo_cfg
        self._stereo_cfg = stereo_cfg

    def _log(self, msg: str):
        logger.info(msg)
        self.log_line.emit(msg)

    def run(self) -> None:
        try:
            self._run_calibration()
        except Exception as exc:
            logger.exception("Kalibrálás hiba: %s", exc)
            self.error_occurred.emit(str(exc))

    def _run_calibration(self) -> None:
        n = len(self._obj_pts)
        self._log(f"Sztereó kalibrálás: {n} képpár, képméret: {self._image_size}")

        f_px = float(self._geo_cfg.get("focal_length_px", 1365.2))
        cx   = float(self._geo_cfg.get("principal_point_x", 968.0))
        cy   = float(self._geo_cfg.get("principal_point_y", 608.0))
        ref_baseline = float(self._geo_cfg.get("baseline_mm", 2200.0))
        max_rmse     = float(self._stereo_cfg.get("max_acceptable_rmse_px", 1.0))
        square_mm    = float(self._stereo_cfg.get("chessboard", {}).get("square_size_mm", 30.0))

        self._log(f"  Négyzetméret: {square_mm:.1f} mm | Kezdő f_px: {f_px:.1f}")
        K_init = np.array([[f_px, 0.0, cx], [0.0, f_px, cy], [0.0, 0.0, 1.0]], dtype=np.float64)

        self._log("[1/3] Bal kamera kalibrálás...")
        rmse_l, K1, D1, _, _ = cv2.calibrateCamera(
            self._obj_pts, self._pts_l, self._image_size,
            K_init.copy(), None, flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
        )
        ratio_l = K1[0, 0] / f_px if f_px > 0 else 1.0
        if ratio_l < 0.5 or ratio_l > 2.0:
            self._log(f"  ⚠ FIGYELEM! Helytelen négyzetméret gyanú! Becsült valódi: ~{square_mm * ratio_l:.1f} mm")

        self._log("[2/3] Jobb kamera kalibrálás...")
        rmse_r, K2, D2, _, _ = cv2.calibrateCamera(
            self._obj_pts, self._pts_r, self._image_size,
            K_init.copy(), None, flags=cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
        )

        self._log("[3/3] Sztereó kalibrálás (R, T meghatározása)...")
        # CALIB_FIX_INTRINSIC: az egyenkénti kalibrálás K1,D1,K2,D2 eredményeit rögzítjük,
        # a sztereó lépés csak R és T-t becsüli → sokkal stabilabb, kisebb RMSE
        rmse_s, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
            self._obj_pts, self._pts_l, self._pts_r,
            K1, D1, K2, D2, self._image_size,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
            flags=cv2.CALIB_FIX_INTRINSIC,
        )
        
        if rmse_s > 1.0:
            self._log(f"  ⚠ Magas RMSE ({rmse_s:.1f} px)! Ellenőrizd: négyzetméret, pózdiverzitás, sarokpontok.")

        R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
            K1, D1, K2, D2, self._image_size, R, T, flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
        )

        baseline_mm = float(np.linalg.norm(T))
        quality = "KIVÁLÓ ✓" if rmse_s < 0.5 else ("JÓ ✓" if rmse_s < max_rmse else "GYENGE ⚠")
        self._log(f"Minőség: {quality} | RMSE: {rmse_s:.4f} px | Baseline: {baseline_mm:.1f} mm")

        self.finished.emit({
            "K1": K1, "D1": D1, "K2": K2, "D2": D2, "R": R, "T": T, "E": E, "F": F,
            "R1": R1, "R2": R2, "P1": P1, "P2": P2, "Q": Q,
            "rmse": rmse_s, "baseline_mm": baseline_mm, "quality": quality,
        })


# --------------------------------------------------------------------------- #
# Fő Kalibrációs Dialog
# --------------------------------------------------------------------------- #

class CalibrationDialog(QDialog):
    """
    Háromfüles kalibrációs munkafolyamat:
      ① Beállítások – sakktábla paraméterek, geometria összefoglaló, kimeneti fájl
      ② Képrögzítés – élő kamera + sarokpont detektálás + képpár mentés
      ③ Kalibrálás  – futtatás, log, eredmény, mentés .npz-be
    """

    def __init__(self, config: dict, parent=None):
        super().__init__(parent)
        self._config         = config
        self._capture_worker: Optional[CalibrationCaptureWorker] = None
        self._run_worker:     Optional[CalibrationRunWorker]     = None
        self._last_result:    Optional[dict]                     = None

        self.setWindowTitle("DEIK – Sztereó Kalibrációs Munkafolyamat")
        self.setMinimumSize(1050, 740)
        self.setStyleSheet(
            "QDialog, QWidget { background-color: #F8FAFC; color: #0F172A; "
            "font-family: 'Segoe UI', Arial, sans-serif; }"
            "QGroupBox { border: 1px solid #CBD5E1; border-radius: 6px; margin-top: 12px; "
            "font-weight: 700; font-size: 12px; padding: 6px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 10px; color: #0F5132; }"
            "QTabWidget::pane { border: 1px solid #CBD5E1; border-radius: 6px; "
            "background: #F8FAFC; }"
            "QTabBar::tab { background: #E2E8F0; border: 1px solid #CBD5E1; "
            "padding: 9px 22px; border-top-left-radius: 6px; border-top-right-radius: 6px; "
            "font-weight: 700; font-size: 12px; }"
            "QTabBar::tab:selected { background: #0F5132; color: #FFFFFF; }"
            "QSpinBox, QDoubleSpinBox, QLineEdit { border: 1px solid #CBD5E1; "
            "border-radius: 4px; padding: 4px 8px; background: #FFFFFF; min-height: 28px; }"
            "QProgressBar { border: 1px solid #CBD5E1; border-radius: 6px; "
            "background: #F1F5F9; height: 22px; text-align: center; font-weight: 700; }"
            "QProgressBar::chunk { background: qlineargradient(x1:0, y1:0, x2:1, y2:0, "
            "stop:0 #0F5132, stop:1 #16A34A); border-radius: 5px; }"
            "QPlainTextEdit { background: #0F172A; color: #A3E635; "
            "font-family: 'Consolas', 'Courier New', monospace; font-size: 11px; "
            "border-radius: 6px; }"
        )
        self._build_ui()

    # ------------------------------------------------------------------ #
    # UI felépítés
    # ------------------------------------------------------------------ #

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)
        root.setSpacing(10)

        # Fejléc
        hdr = QLabel("⚙  DEIK Sztereó Kalibrációs Munkafolyamat")
        hdr.setStyleSheet("font-size: 17px; font-weight: 900; color: #0F5132; padding: 2px 0;")
        root.addWidget(hdr)

        sub = QLabel(
            "Lépések: ① Ellenőrizd a beállításokat  →  "
            "② Rögzíts képpárokat a sakktáblával  →  ③ Futtasd a kalibrálást és mentsd el."
        )
        sub.setStyleSheet("color: #475569; font-size: 12px;")
        root.addWidget(sub)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("color: #CBD5E1;")
        root.addWidget(sep)

        self._tabs = QTabWidget()
        self._tabs.addTab(self._build_tab_settings(), "① Beállítások")
        self._tabs.addTab(self._build_tab_capture(),  "② Képrögzítés")
        self._tabs.addTab(self._build_tab_run(),      "③ Kalibrálás")
        root.addWidget(self._tabs, stretch=1)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_close = QPushButton("✕  Bezárás")
        btn_close.setStyleSheet(_BTN_SEC)
        btn_close.clicked.connect(self._on_close_clicked)
        btn_row.addWidget(btn_close)
        root.addLayout(btn_row)

    # --- ① Beállítások ---

    def _build_tab_settings(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(16, 16, 16, 16)
        ly.setSpacing(14)

        stereo_cfg = self._config.get("stereo", {})
        cb_cfg     = stereo_cfg.get("chessboard", {})
        geo_cfg    = self._config.get("geometry", {})

        # Sakktábla paraméterek
        cb_grp  = QGroupBox("Sakktábla Paraméterek")
        cb_form = QFormLayout(cb_grp)
        cb_form.setSpacing(10)

        info = QLabel(
            "ℹ  Jelenlegi A4 sakktáblánk: <b>9 × 7 négyzet</b>, négyzetméret: <b>30 mm</b><br>"
            "Az OpenCV belső sarokpontok: <b>8 × 6</b>  (négyzetek – 1 per tengely).<br>"
            "Ha eltérő táblát használsz, módosítsd az alábbi értékeket."
        )
        info.setStyleSheet(
            "background: #ECFDF5; border: 1px solid #6EE7B7; border-radius: 6px; "
            "padding: 10px; color: #065F46; font-size: 12px;"
        )
        info.setWordWrap(True)
        cb_form.addRow("", info)

        self._spin_cx = QSpinBox()
        self._spin_cx.setRange(3, 20)
        self._spin_cx.setValue(int(cb_cfg.get("inner_corners_x", 8)))
        self._spin_cx.setToolTip("Vízszintes belső sarokpontok (9 négyzet → 8)")

        self._spin_cy = QSpinBox()
        self._spin_cy.setRange(3, 20)
        self._spin_cy.setValue(int(cb_cfg.get("inner_corners_y", 6)))
        self._spin_cy.setToolTip("Függőleges belső sarokpontok (7 négyzet → 6)")

        corners_row = QHBoxLayout()
        corners_row.addWidget(QLabel("Vízszintes (X):"))
        corners_row.addWidget(self._spin_cx)
        corners_row.addWidget(QLabel("   Függőleges (Y):"))
        corners_row.addWidget(self._spin_cy)
        corners_row.addStretch(1)
        cb_form.addRow("Belső sarkok:", corners_row)

        self._spin_sq = QDoubleSpinBox()
        self._spin_sq.setRange(1.0, 300.0)
        self._spin_sq.setDecimals(1)
        self._spin_sq.setSuffix(" mm")
        self._spin_sq.setValue(float(cb_cfg.get("square_size_mm", 30.0)))
        cb_form.addRow("Négyzetméret:", self._spin_sq)

        self._spin_min = QSpinBox()
        self._spin_min.setRange(5, 100)
        self._spin_min.setValue(int(stereo_cfg.get("min_calibration_frames", 20)))
        self._spin_min.setToolTip("Min. 15-20 képpár ajánlott a pontos kalibráláshoz")
        cb_form.addRow("Min. képpárok:", self._spin_min)

        ly.addWidget(cb_grp)

        # Kamera geometria (csak tájékoztató)
        geo_grp  = QGroupBox("Kamera Geometria (Jelenlegi Konfiguráció)")
        geo_form = QFormLayout(geo_grp)
        geo_form.setSpacing(8)

        def glbl(v, u="mm"):
            l = QLabel(f"<b>{v}</b> {u}")
            l.setStyleSheet("color: #0F5132; font-size: 12px;")
            return l

        geo_form.addRow("Bal kamera X:",    glbl(geo_cfg.get("left_camera_x_mm",  -1100.0)))
        geo_form.addRow("Jobb kamera X:",   glbl(geo_cfg.get("right_camera_x_mm",  1100.0)))
        geo_form.addRow("Baseline:",        glbl(geo_cfg.get("baseline_mm",        2200.0)))
        geo_form.addRow("Kamera magasság:", glbl(geo_cfg.get("camera_height_mm",   2900.0)))
        geo_form.addRow("Kapu szélesség:",  glbl(geo_cfg.get("goal_width_mm",      4000.0)))
        geo_form.addRow("Kapu magasság:",   glbl(geo_cfg.get("goal_height_mm",     2000.0)))
        ly.addWidget(geo_grp)

        # Kimeneti fájl
        out_grp  = QGroupBox("Kalibrációs Fájl Mentési Helye")
        out_form = QFormLayout(out_grp)
        out_form.setSpacing(8)

        default_out = str(
            (Path(__file__).parent.parent.parent /
             stereo_cfg.get("calibration_file", "data/calibration/stereo_calibration.npz")
            ).resolve()
        )
        self._edit_out = QLineEdit(default_out)

        btn_browse = QPushButton("Tallózás…")
        btn_browse.setStyleSheet(_BTN_SEC)
        btn_browse.clicked.connect(self._on_browse)

        row = QHBoxLayout()
        row.addWidget(self._edit_out, stretch=3)
        row.addWidget(btn_browse)
        out_form.addRow("Fájl:", row)
        ly.addWidget(out_grp)

        btn_apply = QPushButton("✔  Beállítások Alkalmazása")
        btn_apply.setStyleSheet(_BTN_PRIMARY)
        btn_apply.clicked.connect(self._apply_settings)
        ly.addWidget(btn_apply)

        ly.addStretch(1)
        return w

    # --- ② Képrögzítés ---

    def _build_tab_capture(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(12, 12, 12, 12)
        ly.setSpacing(8)

        # Státusz sáv
        status_row = QHBoxLayout()
        self._lbl_status = QLabel("⏸  Kamera inaktív – kattints a START gombra")
        self._lbl_status.setStyleSheet(
            "background: #FEF3C7; color: #92400E; font-weight: 700; "
            "border-radius: 6px; padding: 6px 12px; font-size: 13px;"
        )
        self._lbl_count = QLabel("0 / 20  képpár")
        self._lbl_count.setStyleSheet(
            "font-weight: 900; font-size: 14px; color: #0F5132; padding: 0 10px;"
        )
        status_row.addWidget(self._lbl_status, stretch=1)
        status_row.addWidget(self._lbl_count)
        ly.addLayout(status_row)

        # Progressbar
        self._progress = QProgressBar()
        self._progress.setRange(0, 20)
        self._progress.setValue(0)
        self._progress.setFormat("%v / %m képpár rögzítve")
        ly.addWidget(self._progress)

        # Kamera megjelenítő
        cam_grp = QGroupBox("Élő Kamera Kép  —  Bal  |  Jobb")
        cam_box = QVBoxLayout(cam_grp)
        cam_box.setContentsMargins(6, 16, 6, 6)
        self._cam_lbl = QLabel(
            "A kamera élő képe itt jelenik meg.\n\n"
            "Zöld sarokpontok = sakktábla mindkét kamerában látható → SPACE a mentéshez.\n"
            "Kék = csak az egyik kamerában látszik."
        )
        self._cam_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_lbl.setStyleSheet(
            "background: #0F172A; color: #64748B; border-radius: 8px; "
            "font-size: 13px; font-weight: 600;"
        )
        self._cam_lbl.setMinimumHeight(360)
        self._cam_lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        cam_box.addWidget(self._cam_lbl)
        ly.addWidget(cam_grp, stretch=1)

        # Vezérlő gombsor
        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)

        self._btn_start = QPushButton("▶  Kamerák Indítása")
        self._btn_start.setStyleSheet(_BTN_PRIMARY)
        self._btn_start.clicked.connect(self._on_start_camera)

        self._btn_stop = QPushButton("⏹  Leállítás")
        self._btn_stop.setStyleSheet(_BTN_SEC)
        self._btn_stop.setEnabled(False)
        self._btn_stop.clicked.connect(self._on_stop_camera)

        self._btn_cap = QPushButton("📸  KÉPPÁR MENTÉSE  [SPACE]")
        self._btn_cap.setStyleSheet(_BTN_CAPTURE)
        self._btn_cap.setEnabled(False)
        self._btn_cap.clicked.connect(self._on_capture_click)

        self._btn_clear = QPushButton("🗑  Képpárok Törlése")
        self._btn_clear.setStyleSheet(_BTN_WARN)
        self._btn_clear.setEnabled(False)
        self._btn_clear.clicked.connect(self._on_clear)

        ctrl.addWidget(self._btn_start)
        ctrl.addWidget(self._btn_stop)
        ctrl.addStretch(1)
        ctrl.addWidget(self._btn_cap)
        ctrl.addWidget(self._btn_clear)
        ly.addLayout(ctrl)

        tip = QLabel(
            "💡  Tartsd a sakktáblát különböző szögekben és távolságokban (0.5 – 3 m). "
            "Szükséges: legalább 15–20 képpár a pontos kalibráláshoz."
        )
        tip.setWordWrap(True)
        tip.setStyleSheet("color: #475569; font-size: 11px; padding: 2px;")
        ly.addWidget(tip)

        return w

    # --- ③ Kalibrálás ---

    def _build_tab_run(self) -> QWidget:
        w = QWidget()
        ly = QVBoxLayout(w)
        ly.setContentsMargins(14, 14, 14, 14)
        ly.setSpacing(10)

        # Gombok
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

        # Eredmény kártyák
        res_grp  = QGroupBox("Kalibrálási Eredmény")
        res_form = QFormLayout(res_grp)
        res_form.setSpacing(8)

        def rl():
            l = QLabel("—")
            l.setStyleSheet("font-weight: 700; font-size: 13px; font-family: Consolas;")
            return l

        self._r_rmse   = rl()
        self._r_rmse_l = rl()
        self._r_rmse_r = rl()
        self._r_base   = rl()
        self._r_k1     = rl()
        self._r_k2     = rl()
        self._r_qual   = rl()

        res_form.addRow("Sztereó RMSE:",      self._r_rmse)
        res_form.addRow("Bal kamera RMSE:",   self._r_rmse_l)
        res_form.addRow("Jobb kamera RMSE:",  self._r_rmse_r)
        res_form.addRow("Mért baseline:",     self._r_base)
        res_form.addRow("Bal f_px (K[0,0]):", self._r_k1)
        res_form.addRow("Jobb f_px (K[0,0]):", self._r_k2)
        res_form.addRow("Minőség:",           self._r_qual)
        ly.addWidget(res_grp)

        # Log konzol
        log_grp = QGroupBox("Kalibrálási Log")
        log_box = QVBoxLayout(log_grp)
        log_box.setContentsMargins(6, 16, 6, 6)
        self._log_txt = QPlainTextEdit()
        self._log_txt.setReadOnly(True)
        self._log_txt.setMaximumBlockCount(500)
        self._log_txt.setMinimumHeight(180)
        log_box.addWidget(self._log_txt)
        btn_clr_log = QPushButton("Log Törlése")
        btn_clr_log.setStyleSheet(_BTN_SEC)
        btn_clr_log.clicked.connect(self._log_txt.clear)
        log_box.addWidget(btn_clr_log)
        ly.addWidget(log_grp, stretch=1)

        return w

    # ------------------------------------------------------------------ #
    # Slotok
    # ------------------------------------------------------------------ #

    @pyqtSlot()
    def _apply_settings(self) -> None:
        s = self._config.setdefault("stereo", {})
        c = s.setdefault("chessboard", {})
        c["inner_corners_x"]          = self._spin_cx.value()
        c["inner_corners_y"]          = self._spin_cy.value()
        c["square_size_mm"]           = self._spin_sq.value()
        s["min_calibration_frames"]   = self._spin_min.value()
        s["calibration_file"]         = self._edit_out.text()
        self._progress.setMaximum(self._spin_min.value())
        self._progress.setFormat(f"%v / {self._spin_min.value()} képpár rögzítve")
        QMessageBox.information(self, "Beállítások alkalmazva",
                                "A sakktábla paraméterek frissítve a munkamenetben.")

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
        pattern = (self._spin_cx.value(), self._spin_cy.value())
        self._capture_worker = CalibrationCaptureWorker(self._config, pattern, parent=self)
        self._capture_worker.frame_ready.connect(self._on_frame)
        self._capture_worker.capture_result.connect(self._on_capture_result)
        self._capture_worker.error_occurred.connect(self._on_cam_error)
        self._capture_worker.stopped.connect(self._on_cam_stopped)
        self._capture_worker.start()
        self._btn_start.setEnabled(False)
        self._btn_stop.setEnabled(True)
        self._btn_cap.setEnabled(True)
        self._btn_clear.setEnabled(True)
        self._set_status("⏺  Kamerák aktívak – mutasd a sakktáblát!", "#DCFCE7", "#14532D")

    @pyqtSlot()
    def _on_stop_camera(self) -> None:
        if self._capture_worker:
            self._capture_worker.stop()
            self._capture_worker.wait(4000)

    @pyqtSlot(np.ndarray, np.ndarray, object, object)
    def _on_frame(self, fl: np.ndarray, fr: np.ndarray, cl, cr) -> None:
        both = cl is not None and cr is not None
        n    = len(self._capture_worker.collected_obj_pts) if self._capture_worker else 0
        min_f = self._spin_min.value()

        # Overlay szöveg
        status_color = (0, 200, 60) if both else (30, 120, 255)
        txt = f"Kepparok: {n}/{min_f}  |  " + (
            "MINDKET KAMERA LATJA! -> SPACE" if both else "Sakktabla keresese..."
        )
        cv2.putText(fl, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)
        cv2.putText(fr, "SPACE=mentes", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (160, 160, 160), 1)

        # Megjelenítés
        lw = max(self._cam_lbl.width(), 640)
        lh = max(self._cam_lbl.height(), 300)
        hw = lw // 2
        rl = cv2.resize(fl, (hw, lh))
        rr = cv2.resize(fr, (hw, lh))
        combined = np.hstack([rl, rr])
        cv2.line(combined, (hw, 0), (hw, lh), (60, 60, 60), 2)

        h, w, ch = combined.shape
        q_img = QImage(combined.data, w, h, w * ch, QImage.Format.Format_BGR888)
        self._cam_lbl.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                lw, lh,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
        )

        # Számláló és progress
        self._lbl_count.setText(f"{n} / {min_f}  képpár")
        self._progress.setMaximum(min_f)
        self._progress.setValue(min(n, min_f))

        # Státusz szín
        if both:
            self._set_status(
                "✅  Sakktábla MINDKÉT kamerában látható!  →  SPACE = képpár mentése",
                "#DCFCE7", "#14532D"
            )
        else:
            nf = int(cl is not None) + int(cr is not None)
            self._set_status(
                f"🔍  Sakktábla keresése... ({nf}/2 kamera talált)",
                "#FEF3C7", "#92400E"
            )

        self._update_run_btn(n)

    def _set_status(self, txt: str, bg: str, fg: str) -> None:
        self._lbl_status.setText(txt)
        self._lbl_status.setStyleSheet(
            f"background: {bg}; color: {fg}; font-weight: 700; "
            f"border-radius: 6px; padding: 6px 12px; font-size: 13px;"
        )

    @pyqtSlot(bool, str)
    def _on_capture_result(self, success: bool, msg: str) -> None:
        """Képpár mentés eredménye – visszajelzés a felhasználónak."""
        if success:
            n = len(self._capture_worker.collected_obj_pts) if self._capture_worker else 0
            min_f = self._spin_min.value()
            self._lbl_count.setText(f"{n} / {min_f}  képpár")
            self._progress.setValue(min(n, min_f))
            self._set_status(f"✅  {msg}", "#DCFCE7", "#14532D")
            self._update_run_btn(n)
        else:
            # Elégé hasonló póz vagy hiba
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
        self._btn_stop.setEnabled(False)
        self._btn_cap.setEnabled(False)
        self._set_status("⏸  Kamera leállítva.", "#F1F5F9", "#475569")

    def _update_run_btn(self, n: int) -> None:
        min_f = self._spin_min.value()
        ok = n >= min_f
        self._btn_run.setEnabled(ok)
        if ok:
            self._btn_run.setText(f"🔬  KALIBRÁLÁS INDÍTÁSA  ({n} képpár)")
        else:
            self._btn_run.setText(f"🔬  Kalibrálás  (még {max(0, min_f - n)} képpár kell)")

    @pyqtSlot()
    def _on_run(self) -> None:
        if not self._capture_worker:
            QMessageBox.warning(self, "Hiba", "Nincs rögzített kamera adat!")
            return
        n     = len(self._capture_worker.collected_obj_pts)
        min_f = self._spin_min.value()
        if n < min_f:
            QMessageBox.warning(self, "Nincs elég képpár",
                                f"Minimum {min_f} szükséges, jelenleg {n} van!")
            return
        if self._capture_worker.image_size is None:
            QMessageBox.warning(self, "Hiba", "Kép méret ismeretlen!")
            return

        # Kamera leállítás
        if self._capture_worker.isRunning():
            self._capture_worker.stop()
            self._capture_worker.wait(5000)

        self._log_txt.clear()
        self._log_append(f"Kalibrálás indítása: {n} képpár, méret: {self._capture_worker.image_size}")

        self._run_worker = CalibrationRunWorker(
            obj_pts    = self._capture_worker.collected_obj_pts,
            pts_l      = self._capture_worker.collected_pts_left,
            pts_r      = self._capture_worker.collected_pts_right,
            image_size = self._capture_worker.image_size,
            geo_cfg    = self._config.get("geometry", {}),
            stereo_cfg = self._config.get("stereo", {}),
            parent     = self,
        )
        self._run_worker.log_line.connect(self._log_append)
        self._run_worker.finished.connect(self._on_cal_done)
        self._run_worker.error_occurred.connect(self._on_cal_error)
        self._run_worker.start()

        self._btn_run.setEnabled(False)
        self._btn_run.setText("⏳  Kalibrálás folyamatban...")
        self._tabs.setCurrentIndex(2)

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
        q = res.get("quality", "—")
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
        """SPACE = képpár mentése, ha a Képrögzítés fül aktív."""
        if event.key() == Qt.Key.Key_Space and self._tabs.currentIndex() == 1:
            self._on_capture_click()
        else:
            super().keyPressEvent(event)
