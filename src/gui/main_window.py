"""
DEIK Robot Foci Kapus – Fehér DEIK GUI Ablak (PyQt6)
====================================================

Ez a modul a Debreceni Egyetem Informatikai Karának hivatalos arculatával,
a DEIK logóval és formális, letisztult FEHÉR témával rendelkező szoftveres felületet valósítja meg.

Fő elemek:
    - Fejléc: Hivatalos DEIK Címer / Logó + Megnevezés + Vezérlő gombok
    - Téma: Letisztult fehér / világos szürke háttér DEIK zöld és arany akcentusokkal
    - Kamerák: Bal és Jobb kamera élő felvétele és elmozdulás (Offset) visszajelzése
    - 2D Kapu Nézet: Letisztult becsapódási rajz és zóna előrejelzés
    - Vezérlőpanel: Közvetlenül elérhető X/Y elmozdulás, ROI és kamera finomhangolás
    - Telemetria & Log: 3D pozíció, sebesség és lebegő rendszernapló
"""

import json
import logging
import os
import sys
import time
from typing import Any, Dict, Optional

# pyrefly: ignore [missing-import]
# type: ignore
import cv2
import numpy as np
import psutil
# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtCore import (
    QObject, QPoint, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
)
# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtGui import (
    QColor, QFont, QIcon, QImage, QKeySequence, QPixmap, QShortcut, QTextCursor, QWheelEvent
)
# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QDialog, QDockWidget, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QDoubleSpinBox, QStackedWidget, QStatusBar, QTabWidget,
    QToolBar, QVBoxLayout, QWidget
)

from camera.camera_manager import CameraManager, StereoPair
from detection.ball_detector import BallDetector, StereoBallDetection
from detection.kalman_tracker import KalmanTracker2D
from stereo.triangulator import StereoTriangulator
from prediction.trajectory_predictor import TrajectoryPredictor, ImpactPrediction
from gui.goal_view import GoalViewWidget
from gui.calibration_dialog import CalibrationDialog
from gui.theme import (
    LIGHT_DEIK_QSS, get_status_pill_style, get_app_icon, COLOR_DEIK_GREEN, COLOR_DEIK_GOLD
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Qt Log handler
# --------------------------------------------------------------------------- #

class _QtLogSignals(QObject):
    log_msg = pyqtSignal(str)


class _QtLogHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.signals = _QtLogSignals()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.signals.log_msg.emit(msg)


# --------------------------------------------------------------------------- #
# Interaktív kamera nézetlabel (zoom + pan)
# --------------------------------------------------------------------------- #

class ZoomableLabel(QLabel):
    """
    QLabel utód, amely egérgörgős zoomot és drag/pan mozgatást valósít meg
    a kamera képeken. A zoom és pan state a widget-ben tárolódik, és a
    set_frame() metóduson keresztül frissül a megjelenített kép.

    Kezelők:
        - wheelEvent:           zoom be/ki (0.25x – 10x)
        - mousePressEvent:      pan kezdete (bal gomb)
        - mouseMoveEvent:       pan mozgatás
        - mouseReleaseEvent:    pan vége
        - mouseDoubleClickEvent: zoom/pan reset
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._zoom: float = 1.0
        self._pan_x: float = 0.0   # relatív, 0.0–1.0 tartomány (kép-koordinátában)
        self._pan_y: float = 0.0
        self._dragging: bool = False
        self._drag_start: QPoint = QPoint()
        self._drag_pan_start_x: float = 0.0
        self._drag_pan_start_y: float = 0.0
        self._current_frame: Optional[np.ndarray] = None

        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)

    # ------------------------------------------------------------------
    # Nyilvános API
    # ------------------------------------------------------------------

    def set_frame(self, frame: np.ndarray) -> None:
        """Beállítja az aktuális képkockát és frissíti a megjelenítést."""
        self._current_frame = frame
        self._render()

    def reset_zoom(self) -> None:
        """Visszaállítja a zoom és pan értékeket az alapértelmezésre."""
        self._zoom = 1.0
        self._pan_x = 0.0
        self._pan_y = 0.0
        self._render()

    @property
    def zoom_level(self) -> float:
        return self._zoom

    # ------------------------------------------------------------------
    # Eseménykezelők
    # ------------------------------------------------------------------

    def wheelEvent(self, event: QWheelEvent) -> None:
        """Egérgörgős zoom – az egér pozíciójára fókuszálva."""
        if self._current_frame is None:
            return

        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else (1.0 / 1.15)
        new_zoom = max(0.25, min(10.0, self._zoom * factor))

        # Az egér kép-koordinátáihoz igazítjuk a pan-t, hogy a kurzor "helyen maradjon"
        w_lbl = max(self.width(), 1)
        h_lbl = max(self.height(), 1)
        mpos = event.position()
        mx_rel = mpos.x() / w_lbl   # 0..1
        my_rel = mpos.y() / h_lbl

        # Zoom előtti viewport szélesség/magasság (kép-arányban)
        vw_old = 1.0 / self._zoom
        vh_old = 1.0 / self._zoom
        vw_new = 1.0 / new_zoom
        vh_new = 1.0 / new_zoom

        # Pan korrekció, hogy az egér alatt lévő pont helyen maradjon
        self._pan_x += (mx_rel - 0.5) * (vw_old - vw_new)
        self._pan_y += (my_rel - 0.5) * (vh_old - vh_new)

        self._zoom = new_zoom
        self._clamp_pan()
        self._render()
        event.accept()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton and self._zoom > 1.001:
            self._dragging = True
            self._drag_start = event.pos()
            self._drag_pan_start_x = self._pan_x
            self._drag_pan_start_y = self._pan_y
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event) -> None:
        if self._dragging and self._current_frame is not None:
            dx = event.pos().x() - self._drag_start.x()
            dy = event.pos().y() - self._drag_start.y()
            w_lbl = max(self.width(), 1)
            h_lbl = max(self.height(), 1)
            # Konvertálás kép-arányos koordinátákba
            self._pan_x = self._drag_pan_start_x - dx / w_lbl / self._zoom
            self._pan_y = self._drag_pan_start_y - dy / h_lbl / self._zoom
            self._clamp_pan()
            self._render()
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging = False
            cursor = Qt.CursorShape.CrossCursor if self._zoom <= 1.001 else Qt.CursorShape.OpenHandCursor
            self.setCursor(cursor)
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:
        """Dupla kattintás: zoom és pan visszaállítása."""
        self.reset_zoom()
        self.setCursor(Qt.CursorShape.CrossCursor)
        event.accept()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._render()

    # ------------------------------------------------------------------
    # Belső segédfüggvények
    # ------------------------------------------------------------------

    def _clamp_pan(self) -> None:
        """Korlátozza a pan értékeket, hogy a kép ne csússzon ki a nézetből."""
        half = 0.5 / self._zoom
        self._pan_x = max(-half + 0.5 / self._zoom, min(1.0 - 0.5 / self._zoom - (1.0 / self._zoom - 2 * half), self._pan_x))
        self._pan_y = max(-half + 0.5 / self._zoom, min(1.0 - 0.5 / self._zoom - (1.0 / self._zoom - 2 * half), self._pan_y))
        # Egyszerűbb határok: pan ne vihet ki a képből
        margin = (1.0 - 1.0 / self._zoom) / 2.0
        self._pan_x = max(-margin, min(margin, self._pan_x))
        self._pan_y = max(-margin, min(margin, self._pan_y))

    def _render(self) -> None:
        """Rendereli a képet a jelenlegi zoom/pan beállítások alapján."""
        if self._current_frame is None:
            return

        w_lbl = self.width()
        h_lbl = self.height()
        if w_lbl <= 0 or h_lbl <= 0:
            return

        frame = self._current_frame
        fh, fw = frame.shape[:2]

        if self._zoom <= 1.001 and self._pan_x == 0.0 and self._pan_y == 0.0:
            # Nincs zoom – egyszerű skálázás
            cropped = frame
        else:
            # Viewport kiszámítása kép-koordinátákban
            vw = fw / self._zoom          # viewport szélesség pixelben
            vh = fh / self._zoom          # viewport magasság pixelben
            cx = (0.5 + self._pan_x) * fw # középpont x
            cy = (0.5 + self._pan_y) * fh # középpont y

            x1 = int(max(0, cx - vw / 2))
            y1 = int(max(0, cy - vh / 2))
            x2 = int(min(fw, cx + vw / 2))
            y2 = int(min(fh, cy + vh / 2))

            if x2 <= x1 or y2 <= y1:
                return
            cropped = frame[y1:y2, x1:x2]

        # Skálázás a label méretére
        ch = cropped.shape[2] if len(cropped.shape) == 3 else 1
        q_img = QImage(
            cropped.data, cropped.shape[1], cropped.shape[0],
            cropped.shape[1] * ch,
            QImage.Format.Format_BGR888
        )
        pixmap = QPixmap.fromImage(q_img).scaled(
            w_lbl, h_lbl,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.setPixmap(pixmap)

        # Zoom szint felirat
        if self._zoom > 1.05:
            overlay_text = f"🔍 {self._zoom:.1f}×"
            # A pixmap-ra ráírjuk (egyszerű megközelítés: label toolTip)
            self.setToolTip(f"Zoom: {self._zoom:.1f}×  (dupla kattintás = reset)")
        else:
            self.setToolTip("Egérgörgővel zoom | Húzással mozgatás | Dupla katt = reset")


# --------------------------------------------------------------------------- #
# Háttérszál (TrackerWorker)
# --------------------------------------------------------------------------- #

class TrackerWorker(QThread):
    frames_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    error_occurred = pyqtSignal(str)
    tracker_stopped = pyqtSignal()

    def __init__(self, config: dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._config = config
        self._running = False

        self._cam_manager: Optional[CameraManager] = None
        self._detector: Optional[BallDetector] = None
        self._triangulator: Optional[StereoTriangulator] = None
        self._predictor: Optional[TrajectoryPredictor] = None

        det_cfg = config.get("detection", {})
        kalman_cfg = det_cfg.get("kalman", {})
        self._kalman_left = KalmanTracker2D(
            process_noise=kalman_cfg.get("process_noise", 0.03),
            measurement_noise=kalman_cfg.get("measurement_noise", 0.1),
            max_coast_frames=kalman_cfg.get("max_coast_frames", 10),
        )
        self._kalman_right = KalmanTracker2D(
            process_noise=kalman_cfg.get("process_noise", 0.03),
            measurement_noise=kalman_cfg.get("measurement_noise", 0.1),
            max_coast_frames=kalman_cfg.get("max_coast_frames", 10),
        )

        self._gui_fps_limit = float(config.get("gui", {}).get("gui_fps_limit", 60))
        self._gui_interval = 1.0 / self._gui_fps_limit
        self._last_gui_emit = 0.0

    def stop(self) -> None:
        self._running = False

    @pyqtSlot(bool, int, int)
    def set_camera_offset(self, is_left: bool, offset_x: int, offset_y: int) -> None:
        if self._cam_manager:
            self._cam_manager.set_camera_offset(is_left, offset_x, offset_y)

    @pyqtSlot(bool, int)
    def set_camera_exposure(self, is_left: bool, exposure_us: int) -> None:
        if self._cam_manager:
            self._cam_manager.set_camera_exposure(is_left, exposure_us)

    @pyqtSlot(bool, float)
    def set_camera_gain(self, is_left: bool, gain_db: float) -> None:
        if self._cam_manager:
            self._cam_manager.set_camera_gain(is_left, gain_db)

    @pyqtSlot(bool, bool)
    def set_camera_awb(self, is_left: bool, enabled: bool) -> None:
        if self._cam_manager:
            self._cam_manager.set_camera_awb(is_left, enabled)

    @pyqtSlot(bool, float, float, float)
    def set_camera_wb(self, is_left: bool, kr: float, kg: float, kb: float) -> None:
        if self._cam_manager:
            self._cam_manager.set_camera_wb(is_left, kr, kg, kb)

    @pyqtSlot(bool, bool, float, float, float, float)
    def set_camera_roi(
        self, is_left: bool, enabled: bool, x_min_rel: float, x_max_rel: float, y_min_rel: float, y_max_rel: float
    ) -> None:
        if self._detector:
            self._detector.set_roi(is_left, enabled, x_min_rel, x_max_rel, y_min_rel, y_max_rel)

    @pyqtSlot(bool, bool, bool, int)
    def set_camera_transform(self, is_left: bool, flip_h: bool, flip_v: bool, rotation: int) -> None:
        if self._cam_manager:
            self._cam_manager.set_camera_transform(is_left, flip_h, flip_v, rotation)

    def run(self) -> None:
        logger.info("TrackerWorker szál elindult")
        self._running = True

        try:
            self._cam_manager = CameraManager(self._config)
            self._detector = BallDetector(self._config["detection"])
            self._triangulator = StereoTriangulator(self._config)
            self._predictor = TrajectoryPredictor(self._config)

            cal_file = self._config.get("stereo", {}).get(
                "calibration_file", "data/calibration/stereo_calibration.npz"
            )
            self._triangulator.load_calibration(cal_file)

        except Exception as exc:
            error_msg = f"Komponens inicializálási hiba: {exc}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return

        if not self._cam_manager.open():
            error_msg = "Kamerák megnyitása sikertelen! Ellenőrizd a csatlakozásokat."
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return

        logger.info("TrackerWorker főciklus indítása...")

        try:
            self._main_loop()
        except Exception as exc:
            logger.exception("Váratlan hiba: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            if self._cam_manager:
                self._cam_manager.close()
            logger.info("TrackerWorker szál leállt")
            self.tracker_stopped.emit()

    def _main_loop(self) -> None:
        overlay_cfg = self._config.get("gui", {}).get("overlay", {})

        while self._running:
            pair: StereoPair = self._cam_manager.read_stereo_pair()
            if not pair.success:
                time.sleep(0.005)
                continue

            frame_left = pair.left.image.copy()
            frame_right = pair.right.image.copy()

            detection: StereoBallDetection = self._detector.detect(
                pair.left.image, pair.right.image
            )

            left_det = detection.left
            right_det = detection.right

            if left_det.found:
                lx, ly = self._kalman_left.update(left_det.x, left_det.y)
                left_det.x, left_det.y = lx, ly
            else:
                self._kalman_left.predict()

            if right_det.found:
                rx, ry = self._kalman_right.update(right_det.x, right_det.y)
                right_det.x, right_det.y = rx, ry
            else:
                self._kalman_right.predict()

            pos_3d = None
            if detection.both_found:
                pos_3d = self._triangulator.triangulate(
                    left_point=(left_det.x, left_det.y),
                    right_point=(right_det.x, right_det.y),
                )

            impact: Optional[ImpactPrediction] = None
            if pos_3d is not None:
                self._predictor.add_measurement(
                    x_mm=float(pos_3d[0]),
                    y_mm=float(pos_3d[1]),
                    z_mm=float(pos_3d[2]),
                )
                impact = self._predictor.get_impact_prediction()
            else:
                if not detection.left.found and not detection.right.found:
                    self._predictor.reset()
                    self._kalman_left.reset()
                    self._kalman_right.reset()

            if self._detector:
                self._detector.draw_roi(frame_left, is_left=True)
                self._detector.draw_roi(frame_right, is_left=False)

            if overlay_cfg.get("show_detection_box", True) and self._detector:
                self._detector.draw_detection(frame_left, left_det)
                self._detector.draw_detection(frame_right, right_det)

            now = time.perf_counter()
            if now - self._last_gui_emit >= self._gui_interval:
                self._last_gui_emit = now
                cam_status = self._cam_manager.get_camera_status()
                vx, vy, vz = self._predictor.estimated_velocity_mm_s

                stats = {
                    "cam_fps_left":  cam_status["fps_left"],
                    "cam_fps_right": cam_status["fps_right"],
                    "pair_fps":      cam_status["pair_fps"],
                    "temp_left":     cam_status["temp_left"],
                    "temp_right":    cam_status["temp_right"],
                    "det_fps":       detection.det_fps,
                    "left_found":    left_det.found,
                    "right_found":   right_det.found,
                    "both_found":    detection.both_found,
                    "x_3d": float(pos_3d[0]) if pos_3d is not None else 0.0,
                    "y_3d": float(pos_3d[1]) if pos_3d is not None else 0.0,
                    "z_3d": float(pos_3d[2]) if pos_3d is not None else 0.0,
                    "pos_valid": pos_3d is not None,
                    "vx_mms": vx, "vy_mms": vy, "vz_mms": vz,
                    "speed_ms": np.sqrt(vx**2 + vy**2 + vz**2) / 1000.0,
                    "impact": impact,
                    "calibrated": self._triangulator.is_calibrated,
                }
                self.frames_ready.emit(frame_left, frame_right, stats)


# --------------------------------------------------------------------------- #
# MainWindow (Fehér DEIK Formális GUI)
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    """
    DEIK Robot Foci Kapus Fehér Formális Témájú GUI.
    """

    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._load_gui_settings()
        self._worker: Optional[TrackerWorker] = None
        self._is_running = False

        self.setWindowTitle("DEIK Robot Foci Kapus – Debreceni Egyetem Informatikai Kar")
        self.setMinimumSize(1280, 820)

        app_icon = get_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)

        self._alt_f4_shortcut = QShortcut(QKeySequence("Alt+F4"), self)
        self._alt_f4_shortcut.activated.connect(self.close)

        self._f11_shortcut = QShortcut(QKeySequence("F11"), self)
        self._f11_shortcut.activated.connect(self._toggle_fullscreen)

        self._esc_shortcut = QShortcut(QKeySequence("Esc"), self)
        self._esc_shortcut.activated.connect(self._toggle_fullscreen)

        self.setStyleSheet(LIGHT_DEIK_QSS)

        self._setup_log_handler()
        self._build_ui()

        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_system_status)
        self._status_timer.start(1000)

        logger.info("MainWindow DEIK fehér felület inicializálva")

    def _build_ui(self) -> None:
        self._build_header_toolbar()
        self._build_central_widget()
        self._build_control_dock()
        self._build_log_dock()
        self._build_status_bar()

    def _build_header_toolbar(self) -> None:
        """Letisztult Fejléc DEIK Címerrel és Fő Vezérlőkkel."""
        toolbar = QToolBar("DEIK Fejléc")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # DEIK Címer Logó
        deik_logo_path = "assets/deik_logo.png"
        if os.path.exists(deik_logo_path):
            l_lbl1 = QLabel()
            l_lbl1.setPixmap(QPixmap(deik_logo_path).scaledToHeight(46, Qt.TransformationMode.SmoothTransformation))
            toolbar.addWidget(l_lbl1)

        # RGK System Shield Logó
        rgk_logo_path = "assets/logo.png"
        if os.path.exists(rgk_logo_path):
            l_lbl2 = QLabel()
            l_lbl2.setPixmap(QPixmap(rgk_logo_path).scaledToHeight(46, Qt.TransformationMode.SmoothTransformation))
            toolbar.addWidget(l_lbl2)

        title_lbl = QLabel("DEIK ROBOT FOCI KAPUS")
        title_lbl.setStyleSheet("font-weight: 800; font-size: 15px; color: #0F5132;")
        sub_lbl = QLabel("Debreceni Egyetem Informatikai Kar")
        sub_lbl.setStyleSheet("font-size: 11px; color: #D97706; font-weight: 600;")

        brand_widget = QWidget()
        b_box = QVBoxLayout(brand_widget)
        b_box.setContentsMargins(4, 0, 10, 0)
        b_box.setSpacing(0)
        b_box.addWidget(title_lbl)
        b_box.addWidget(sub_lbl)
        toolbar.addWidget(brand_widget)

        toolbar.addSeparator()

        self._btn_start = QPushButton("INDÍTÁS")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setFixedHeight(36)
        self._btn_start.clicked.connect(self._on_start_stop)
        toolbar.addWidget(self._btn_start)

        toolbar.addSeparator()

        # Nézetváltó Gombok (Összesített, Élő Kamerák, Kapu & Telemetria)
        self._btn_view_comb = QPushButton("Összesített Nézet")
        self._btn_view_cams = QPushButton("Élő Kamerák")
        self._btn_view_goal = QPushButton("Kapu / Telemetria")

        for b in [self._btn_view_comb, self._btn_view_cams, self._btn_view_goal]:
            b.setCheckable(True)
            b.setFixedHeight(34)
            b.setCursor(Qt.CursorShape.PointingHandCursor)

        self._btn_view_comb.setChecked(True)
        self._btn_view_comb.clicked.connect(lambda: self._set_central_view(0))
        self._btn_view_cams.clicked.connect(lambda: self._set_central_view(1))
        self._btn_view_goal.clicked.connect(lambda: self._set_central_view(2))

        toolbar.addWidget(self._btn_view_comb)
        toolbar.addWidget(self._btn_view_cams)
        toolbar.addWidget(self._btn_view_goal)

        toolbar.addSeparator()

        btn_about = QPushButton("Névjegy / Fejlesztők")
        btn_about.setFixedHeight(34)
        btn_about.setStyleSheet("font-weight: 700; color: #0F5132;")
        btn_about.clicked.connect(self._show_about_dialog)
        toolbar.addWidget(btn_about)

        toolbar.addSeparator()

        btn_calibrate = QPushButton("⚙  Kalibrálás")
        btn_calibrate.setFixedHeight(34)
        btn_calibrate.setStyleSheet(
            "QPushButton { font-weight: 800; color: #FFFFFF; background-color: #D97706; "
            "border-radius: 6px; font-size: 12px; border: none; padding: 0 10px; }"
            "QPushButton:hover { background-color: #B45309; }"
        )
        btn_calibrate.setToolTip(
            "Sztereó kalibrációs munkafolyamat megnyitása\n"
            "(sakktáblás képpár rögzítés + OpenCV sztereó kalibrálás)"
        )
        btn_calibrate.clicked.connect(self._on_open_calibration)
        toolbar.addWidget(btn_calibrate)

        toolbar.addSeparator()

        self._pill_sys = QLabel(" INAKTÍV ")
        self._pill_sys.setStyleSheet(get_status_pill_style("info"))
        toolbar.addWidget(self._pill_sys)

        self._pill_gpu = QLabel(" GPU: RTX 3050 ")
        self._pill_gpu.setStyleSheet(get_status_pill_style("ok"))
        toolbar.addWidget(self._pill_gpu)

    def _set_central_view(self, index: int) -> None:
        """Vált a 3 beépített központi nézet mód között."""
        self._central_stack.setCurrentIndex(index)
        self._btn_view_comb.setChecked(index == 0)
        self._btn_view_cams.setChecked(index == 1)
        self._btn_view_goal.setChecked(index == 2)
        self._update_view_btn_styles()

    def _update_view_btn_styles(self) -> None:
        active_s = (
            "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; border-radius: 6px; font-size: 12px; border: 1px solid #0F5132; }"
        )
        inactive_s = (
            "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 6px; font-size: 12px; border: 1px solid #CBD5E1; }"
            "QPushButton:hover { background-color: #E2E8F0; color: #0F5132; }"
        )

        self._btn_view_comb.setStyleSheet(active_s if self._btn_view_comb.isChecked() else inactive_s)
        self._btn_view_cams.setStyleSheet(active_s if self._btn_view_cams.isChecked() else inactive_s)
        self._btn_view_goal.setStyleSheet(active_s if self._btn_view_goal.isChecked() else inactive_s)

    def _build_central_widget(self) -> None:
        """Központi Terület QStackedWidget-tel: 3 Különböző Nézet Mód."""
        central = QWidget()
        central.setObjectName("centralWidget")
        central.setStyleSheet("background-color: #F8FAFC;")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self._central_stack = QStackedWidget()

        # -------------------------------------------------------------
        # PAGE 0: ÖSSZESÍTETT NÉZET (Combined View)
        # -------------------------------------------------------------
        page_comb = QWidget()
        page_comb.setStyleSheet("background-color: #F8FAFC;")
        comb_layout = QVBoxLayout(page_comb)
        comb_layout.setSpacing(10)
        comb_layout.setContentsMargins(4, 4, 4, 4)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        left_grp = QGroupBox("Bal Kamera  [X = -1100 mm]")
        l_box = QVBoxLayout(left_grp)
        l_box.setContentsMargins(6, 18, 6, 6)
        self._cam_label_left = ZoomableLabel()
        self._cam_label_left.setText("Kamera inaktív")
        self._cam_label_left.setStyleSheet("background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; border-radius: 6px;")
        self._cam_label_left.setMinimumSize(320, 220)
        self._cam_label_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        l_box.addWidget(self._cam_label_left)
        top_row.addWidget(left_grp, stretch=3)

        right_grp = QGroupBox("Jobb Kamera  [X = +1100 mm]")
        r_box = QVBoxLayout(right_grp)
        r_box.setContentsMargins(6, 18, 6, 6)
        self._cam_label_right = ZoomableLabel()
        self._cam_label_right.setText("Kamera inaktív")
        self._cam_label_right.setStyleSheet("background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; border-radius: 6px;")
        self._cam_label_right.setMinimumSize(320, 220)
        self._cam_label_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        r_box.addWidget(self._cam_label_right)
        top_row.addWidget(right_grp, stretch=3)

        goal_grp = QGroupBox("Kapu Vizualizáció / Becsapódás")
        g_box = QVBoxLayout(goal_grp)
        g_box.setContentsMargins(6, 18, 6, 6)
        g_box.setSpacing(6)

        self._goal_view = GoalViewWidget(self._config)
        g_box.addWidget(self._goal_view)

        btn_clear = QPushButton("Lövéstörténet Törlése")
        btn_clear.setFixedHeight(30)
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.setStyleSheet(
            "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 6px; font-size: 11px; border: 1px solid #CBD5E1; }"
            "QPushButton:hover { background-color: #FEE2E2; color: #DC2626; border-color: #EF4444; }"
        )
        btn_clear.clicked.connect(self._on_clear_history)
        g_box.addWidget(btn_clear)

        top_row.addWidget(goal_grp, stretch=4)
        comb_layout.addLayout(top_row, stretch=3)

        # 3D Telemetriai Kártyák
        tel_grp = QGroupBox("ÉLŐ 3D TELEMETRIA / VÉDELMI ZÓNA ELŐREJELZÉS")
        tel_box = QHBoxLayout(tel_grp)
        tel_box.setSpacing(12)
        tel_box.setContentsMargins(12, 18, 12, 10)

        font_val = QFont("Consolas", 12, QFont.Weight.Bold)

        def make_card(title: str, color: str = "#0F5132") -> tuple[QLabel, QWidget]:
            w = QWidget()
            w.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; padding: 6px;")
            v = QVBoxLayout(w)
            v.setContentsMargins(6, 4, 6, 4)
            v.setSpacing(2)
            t_lbl = QLabel(title)
            t_lbl.setStyleSheet("font-size: 10px; color: #475569; font-weight: bold;")
            val_lbl = QLabel("—")
            val_lbl.setFont(font_val)
            val_lbl.setStyleSheet(f"color: {color};")
            v.addWidget(t_lbl)
            v.addWidget(val_lbl)
            return val_lbl, w

        self._lbl_x, card_x = make_card("LABDA X (MM)")
        self._lbl_y, card_y = make_card("LABDA Y (MM)")
        self._lbl_z, card_z = make_card("LABDA Z (MM)")
        self._lbl_speed, card_sp = make_card("SEBESSÉG", "#059669")
        self._lbl_impact, card_imp = make_card("BECSAPÓDÁS (X, Y)", "#D97706")
        self._lbl_time, card_time = make_card("IDŐ (MP)", "#D97706")
        self._lbl_zone, card_zone = make_card("VÉDELMI SZEKTOR", "#0F5132")

        for card in [card_x, card_y, card_z, card_sp, card_imp, card_time, card_zone]:
            tel_box.addWidget(card)

        comb_layout.addWidget(tel_grp, stretch=1)
        self._central_stack.addWidget(page_comb)

        # -------------------------------------------------------------
        # PAGE 1: ÉLŐ KAMERÁK NÉZET (Live Stereo Cameras View)
        # -------------------------------------------------------------
        page_cams = QWidget()
        page_cams.setStyleSheet("background-color: #F8FAFC;")
        cams_layout = QHBoxLayout(page_cams)
        cams_layout.setSpacing(12)
        cams_layout.setContentsMargins(4, 4, 4, 4)

        left_grp_full = QGroupBox("Bal Kamera — Nagyfelbontású Élő Videófolyam  [X = -1100 mm]")
        lf_box = QVBoxLayout(left_grp_full)
        lf_box.setContentsMargins(8, 20, 8, 8)
        self._cam_label_left_full = ZoomableLabel()
        self._cam_label_left_full.setText("Kamera inaktív")
        self._cam_label_left_full.setStyleSheet("background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; border-radius: 8px;")
        self._cam_label_left_full.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lf_box.addWidget(self._cam_label_left_full)
        cams_layout.addWidget(left_grp_full, stretch=1)

        right_grp_full = QGroupBox("Jobb Kamera — Nagyfelbontású Élő Videófolyam  [X = +1100 mm]")
        rf_box = QVBoxLayout(right_grp_full)
        rf_box.setContentsMargins(8, 20, 8, 8)
        self._cam_label_right_full = ZoomableLabel()
        self._cam_label_right_full.setText("Kamera inaktív")
        self._cam_label_right_full.setStyleSheet("background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; border-radius: 8px;")
        self._cam_label_right_full.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rf_box.addWidget(self._cam_label_right_full)
        cams_layout.addWidget(right_grp_full, stretch=1)

        self._central_stack.addWidget(page_cams)

        # -------------------------------------------------------------
        # PAGE 2: KAPU & TELEMETRIA NÉZET (Goal & 3D Telemetry Dashboard View)
        # -------------------------------------------------------------
        page_goal = QWidget()
        page_goal.setStyleSheet("background-color: #F8FAFC;")
        goal_layout = QVBoxLayout(page_goal)
        goal_layout.setSpacing(10)
        goal_layout.setContentsMargins(4, 4, 4, 4)

        goal_grp_full = QGroupBox("Nagyfelbontású Kapu Vizualizáció / Becsapódás Előrejelzés")
        gf_box = QVBoxLayout(goal_grp_full)
        gf_box.setContentsMargins(8, 20, 8, 8)
        gf_box.setSpacing(8)

        self._goal_view_full = GoalViewWidget(self._config)
        gf_box.addWidget(self._goal_view_full)

        btn_clear_full = QPushButton("Lövéstörténet Törlése")
        btn_clear_full.setFixedHeight(34)
        btn_clear_full.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear_full.setStyleSheet(
            "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 6px; font-size: 12px; border: 1px solid #CBD5E1; }"
            "QPushButton:hover { background-color: #FEE2E2; color: #DC2626; border-color: #EF4444; }"
        )
        btn_clear_full.clicked.connect(self._on_clear_history)
        gf_box.addWidget(btn_clear_full)

        goal_layout.addWidget(goal_grp_full, stretch=4)

        self._central_stack.addWidget(page_goal)

        main_layout.addWidget(self._central_stack)

    def _build_control_dock(self) -> None:
        """Reszponzív Vezérlő Dock Panel szegmentált gombokkal és QStackedWidget-tel."""
        dock = QDockWidget("Kamera Pozícionálás & ROI Vezérlés", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)

        self._cam_widgets = {}

        dock_widget = QWidget()
        dock_widget.setStyleSheet("background-color: #F8FAFC;")
        dock_layout = QVBoxLayout(dock_widget)
        dock_layout.setContentsMargins(6, 6, 6, 6)
        dock_layout.setSpacing(8)

        # 3 Szegmentált Nézetváltó Gomb
        nav_box = QHBoxLayout()
        nav_box.setSpacing(4)

        btn_left = QPushButton("BAL KAMERA")
        btn_right = QPushButton("JOBB KAMERA")
        btn_sys = QPushButton("RENDSZER ADATOK")

        for b in [btn_left, btn_right, btn_sys]:
            b.setCheckable(True)
            b.setFixedHeight(34)
            b.setCursor(Qt.CursorShape.PointingHandCursor)

        btn_left.setChecked(True)

        nav_box.addWidget(btn_left, stretch=1)
        nav_box.addWidget(btn_right, stretch=1)
        nav_box.addWidget(btn_sys, stretch=1)
        dock_layout.addLayout(nav_box)

        # QStackedWidget az oldalak közötti törésmentes váltáshoz
        stack = QStackedWidget()
        stack.addWidget(self._create_camera_tab(is_left=True))
        stack.addWidget(self._create_camera_tab(is_left=False))
        stack.addWidget(self._create_system_info_tab())

        dock_layout.addWidget(stack)

        def set_page(index: int):
            stack.setCurrentIndex(index)
            btn_left.setChecked(index == 0)
            btn_right.setChecked(index == 1)
            btn_sys.setChecked(index == 2)
            update_nav_styles()

        def update_nav_styles():
            active_style = (
                "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; border-radius: 6px; font-size: 11px; border: 1px solid #0F5132; }"
            )
            inactive_style = (
                "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 6px; font-size: 11px; border: 1px solid #CBD5E1; }"
                "QPushButton:hover { background-color: #E2E8F0; color: #0F5132; border-color: #0F5132; }"
            )

            btn_left.setStyleSheet(active_style if btn_left.isChecked() else inactive_style)
            btn_right.setStyleSheet(active_style if btn_right.isChecked() else inactive_style)
            btn_sys.setStyleSheet(active_style if btn_sys.isChecked() else inactive_style)

        btn_left.clicked.connect(lambda: set_page(0))
        btn_right.clicked.connect(lambda: set_page(1))
        btn_sys.clicked.connect(lambda: set_page(2))

        update_nav_styles()

        dock.setWidget(dock_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _create_camera_tab(self, is_left: bool) -> QWidget:
        side = "left" if is_left else "right"
        side_name = "Bal" if is_left else "Jobb"

        cam_cfg = self._config["camera"].get(side, {})
        def_cfg = self._config["camera"]

        tab_widget = QWidget()
        tab_widget.setStyleSheet("background-color: #F8FAFC;")
        main_vbox = QVBoxLayout(tab_widget)
        main_vbox.setSpacing(8)
        main_vbox.setContentsMargins(6, 6, 6, 6)

        # Pozícionálás (X/Y Offset)
        pos_grp = QGroupBox(f"{side_name} Kamera X / Y Elmozdulás (Offset)")
        pos_layout = QFormLayout(pos_grp)

        init_x = int(cam_cfg.get("offset_x", 0))
        init_y = int(cam_cfg.get("offset_y", 0))

        spin_x = QSpinBox()
        spin_x.setRange(-500, 500)
        spin_x.setValue(init_x)
        spin_x.setSuffix(" px")

        slider_x = QSlider(Qt.Orientation.Horizontal)
        slider_x.setRange(-500, 500)
        slider_x.setValue(init_x)
        slider_x.valueChanged.connect(spin_x.setValue)
        spin_x.valueChanged.connect(slider_x.setValue)

        hx = QHBoxLayout()
        hx.addWidget(slider_x, stretch=3)
        hx.addWidget(spin_x, stretch=1)
        pos_layout.addRow("X Offset:", hx)

        spin_y = QSpinBox()
        spin_y.setRange(-500, 500)
        spin_y.setValue(init_y)
        spin_y.setSuffix(" px")

        slider_y = QSlider(Qt.Orientation.Horizontal)
        slider_y.setRange(-500, 500)
        slider_y.setValue(init_y)
        slider_y.valueChanged.connect(spin_y.setValue)
        spin_y.valueChanged.connect(slider_y.setValue)

        hy = QHBoxLayout()
        hy.addWidget(slider_y, stretch=3)
        hy.addWidget(spin_y, stretch=1)
        pos_layout.addRow("Y Offset:", hy)

        btn_center = QPushButton("X / Y Nullázás (0, 0)")
        btn_center.setFixedHeight(26)
        btn_center.clicked.connect(lambda: (spin_x.setValue(0), spin_y.setValue(0)))
        pos_layout.addRow("", btn_center)

        main_vbox.addWidget(pos_grp)

        # ROI szűrés
        roi_grp = QGroupBox("ROI Szűrés (Region of Interest)")
        roi_layout = QFormLayout(roi_grp)

        init_roi_en = bool(cam_cfg.get("roi_enabled", False))
        chk_roi = QCheckBox("ROI Szűrés Engedélyezése")
        chk_roi.setChecked(init_roi_en)

        spin_xmin = QSpinBox()
        spin_xmin.setRange(0, 99)
        spin_xmin.setValue(int(cam_cfg.get("roi_x_min", 0)))
        spin_xmin.setSuffix(" %")

        spin_xmax = QSpinBox()
        spin_xmax.setRange(1, 100)
        spin_xmax.setValue(int(cam_cfg.get("roi_x_max", 100)))
        spin_xmax.setSuffix(" %")

        h_roi_x = QHBoxLayout()
        h_roi_x.addWidget(QLabel("Min:"))
        h_roi_x.addWidget(spin_xmin)
        h_roi_x.addWidget(QLabel("Max:"))
        h_roi_x.addWidget(spin_xmax)

        spin_ymin = QSpinBox()
        spin_ymin.setRange(0, 99)
        spin_ymin.setValue(int(cam_cfg.get("roi_y_min", 10)))
        spin_ymin.setSuffix(" %")

        spin_ymax = QSpinBox()
        spin_ymax.setRange(1, 100)
        spin_ymax.setValue(int(cam_cfg.get("roi_y_max", 90)))
        spin_ymax.setSuffix(" %")

        h_roi_y = QHBoxLayout()
        h_roi_y.addWidget(QLabel("Min:"))
        h_roi_y.addWidget(spin_ymin)
        h_roi_y.addWidget(QLabel("Max:"))
        h_roi_y.addWidget(spin_ymax)

        chk_roi_zoom = QCheckBox("ROI Zoom – csak a ROI területet mutassa kinagyítva")
        chk_roi_zoom.setChecked(bool(cam_cfg.get("roi_zoom", False)))
        chk_roi_zoom.setToolTip(
            "Ha be van kapcsolva, a kamera nézet kinagyítva mutatja az aktív ROI területet.\n"
            "Ha ki van kapcsolva, a teljes kép látható egy sárga kerettel jelölt ROI-val."
        )

        roi_layout.addRow("Engedélyezve:", chk_roi)
        roi_layout.addRow("X Tartomány:", h_roi_x)
        roi_layout.addRow("Y Tartomány:", h_roi_y)
        roi_layout.addRow("Zoom nézet:", chk_roi_zoom)

        main_vbox.addWidget(roi_grp)

        # Záridő és Erősítés
        exp_grp = QGroupBox("Záridő / Erősítés")
        exp_layout = QFormLayout(exp_grp)

        spin_exp = QSpinBox()
        spin_exp.setRange(100, 10000)
        spin_exp.setSingleStep(100)
        spin_exp.setValue(int(cam_cfg.get("exposure_time_us", def_cfg.get("exposure_time_us", 3000))))
        spin_exp.setSuffix(" µs")

        spin_gain = QDoubleSpinBox()
        spin_gain.setRange(0.0, 24.0)
        spin_gain.setSingleStep(0.5)
        spin_gain.setValue(float(cam_cfg.get("gain_db", def_cfg.get("gain_db", 0.0))))
        spin_gain.setSuffix(" dB")

        exp_layout.addRow("Záridő:", spin_exp)
        exp_layout.addRow("Erősítés:", spin_gain)

        main_vbox.addWidget(exp_grp)

        # Transzformációk
        trans_grp = QGroupBox("Tükrözés / Forgatás")
        trans_layout = QFormLayout(trans_grp)

        chk_fliph = QCheckBox("Flip H (Vízszintes)")
        chk_fliph.setChecked(bool(cam_cfg.get("flip_h", False)))

        chk_flipv = QCheckBox("Flip V (Függőleges)")
        chk_flipv.setChecked(bool(cam_cfg.get("flip_v", False)))

        combo_rot = QComboBox()
        combo_rot.addItems(["0°", "90°", "180°", "270°"])
        rot_map = {0: 0, 90: 1, 180: 2, 270: 3}
        combo_rot.setCurrentIndex(rot_map.get(int(cam_cfg.get("rotation", 0)), 0))

        h_flips = QHBoxLayout()
        h_flips.addWidget(chk_fliph)
        h_flips.addWidget(chk_flipv)

        trans_layout.addRow("Tükrözés:", h_flips)
        trans_layout.addRow("Forgatás:", combo_rot)

        main_vbox.addWidget(trans_grp)

        # Gombok: Mentés és Alapértelmezett
        h_btns = QHBoxLayout()
        h_btns.setSpacing(6)

        btn_save = QPushButton("Beállítások Mentése")
        btn_save.setFixedHeight(34)
        btn_save.setStyleSheet(
            "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; border-radius: 6px; font-size: 12px; border: none; }"
            "QPushButton:hover { background-color: #146C43; }"
        )
        btn_save.clicked.connect(self._save_gui_settings)

        btn_reset = QPushButton("Alapértelmezett")
        btn_reset.setFixedHeight(34)
        btn_reset.setStyleSheet(
            "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 6px; font-size: 12px; border: 1px solid #CBD5E1; }"
            "QPushButton:hover { background-color: #E2E8F0; color: #0F5132; }"
        )
        btn_reset.clicked.connect(self._reset_gui_settings)

        h_btns.addWidget(btn_save, stretch=1)
        h_btns.addWidget(btn_reset, stretch=1)
        main_vbox.addLayout(h_btns)

        main_vbox.addStretch(1)

        ctrls = {
            "spin_x": spin_x, "spin_y": spin_y, "slider_x": slider_x, "slider_y": slider_y,
            "chk_roi": chk_roi, "spin_xmin": spin_xmin, "spin_xmax": spin_xmax,
            "spin_ymin": spin_ymin, "spin_ymax": spin_ymax,
            "chk_roi_zoom": chk_roi_zoom,
            "spin_exp": spin_exp, "spin_gain": spin_gain,
            "chk_fliph": chk_fliph, "chk_flipv": chk_flipv, "combo_rot": combo_rot
        }
        self._cam_widgets[side] = ctrls

        def on_offset_changed():
            ox, oy = spin_x.value(), spin_y.value()
            self._config["camera"].setdefault(side, {})["offset_x"] = ox
            self._config["camera"][side]["offset_y"] = oy
            if self._worker:
                self._worker.set_camera_offset(is_left, ox, oy)

        def on_roi_changed():
            en = chk_roi.isChecked()
            xmin, xmax = spin_xmin.value(), spin_xmax.value()
            ymin, ymax = spin_ymin.value(), spin_ymax.value()
            self._config["camera"].setdefault(side, {})["roi_enabled"] = en
            self._config["camera"][side]["roi_x_min"] = xmin
            self._config["camera"][side]["roi_x_max"] = xmax
            self._config["camera"][side]["roi_y_min"] = ymin
            self._config["camera"][side]["roi_y_max"] = ymax
            if self._worker:
                self._worker.set_camera_roi(is_left, en, xmin / 100.0, xmax / 100.0, ymin / 100.0, ymax / 100.0)

        def on_exp_changed(v):
            self._config["camera"].setdefault(side, {})["exposure_time_us"] = v
            if self._worker:
                self._worker.set_camera_exposure(is_left, v)

        def on_gain_changed(v):
            self._config["camera"].setdefault(side, {})["gain_db"] = v
            if self._worker:
                self._worker.set_camera_gain(is_left, v)

        def on_trans_changed():
            fh, fv = chk_fliph.isChecked(), chk_flipv.isChecked()
            rot_vals = [0, 90, 180, 270]
            rot = rot_vals[combo_rot.currentIndex()] if combo_rot.currentIndex() < 4 else 0
            self._config["camera"].setdefault(side, {})["flip_h"] = fh
            self._config["camera"][side]["flip_v"] = fv
            self._config["camera"][side]["rotation"] = rot
            if self._worker:
                self._worker.set_camera_transform(is_left, fh, fv, rot)

        spin_x.valueChanged.connect(on_offset_changed)
        spin_y.valueChanged.connect(on_offset_changed)
        chk_roi.toggled.connect(on_roi_changed)
        spin_xmin.valueChanged.connect(on_roi_changed)
        spin_xmax.valueChanged.connect(on_roi_changed)
        spin_ymin.valueChanged.connect(on_roi_changed)
        spin_ymax.valueChanged.connect(on_roi_changed)
        spin_exp.valueChanged.connect(on_exp_changed)
        spin_gain.valueChanged.connect(on_gain_changed)
        chk_fliph.toggled.connect(on_trans_changed)
        chk_flipv.toggled.connect(on_trans_changed)
        combo_rot.currentIndexChanged.connect(on_trans_changed)
        # ROI zoom: csak konfig frissítés, a megjelenítés _on_frames_ready-ben kezelt
        chk_roi_zoom.toggled.connect(
            lambda _: self._config["camera"].setdefault(side, {}).update({"roi_zoom": chk_roi_zoom.isChecked()})
        )

        return tab_widget

    def _create_system_info_tab(self) -> QWidget:
        """Létrehozza a Rendszer Adatok & Diagnosztika lapfület."""
        tab = QWidget()
        tab.setStyleSheet("background-color: #F8FAFC;")
        layout = QVBoxLayout(tab)
        layout.setSpacing(10)
        layout.setContentsMargins(8, 8, 8, 8)

        # Kamera & Sztereó Diagnosztika
        cam_grp = QGroupBox("Kamera Telemetria / Hőmérséklet")
        cam_form = QFormLayout(cam_grp)
        cam_form.setSpacing(6)

        self._lbl_diag_fps_l = QLabel("— FPS")
        self._lbl_diag_temp_l = QLabel("— °C")
        self._lbl_diag_fps_r = QLabel("— FPS")
        self._lbl_diag_temp_r = QLabel("— °C")
        self._lbl_diag_fps_pair = QLabel("— FPS")
        self._lbl_diag_fps_det = QLabel("— FPS")
        self._lbl_diag_calib = QLabel("OK (Stereo Calibrated)")
        self._lbl_diag_calib.setStyleSheet("color: #0F5132; font-weight: bold;")

        cam_form.addRow("Bal Kamera FPS:", self._lbl_diag_fps_l)
        cam_form.addRow("Bal Hőmérséklet:", self._lbl_diag_temp_l)
        cam_form.addRow("Jobb Kamera FPS:", self._lbl_diag_fps_r)
        cam_form.addRow("Jobb Hőmérséklet:", self._lbl_diag_temp_r)
        cam_form.addRow("Sztereó Pár FPS:", self._lbl_diag_fps_pair)
        cam_form.addRow("YOLO Detektálás FPS:", self._lbl_diag_fps_det)
        cam_form.addRow("Kalibrációs Státusz:", self._lbl_diag_calib)

        layout.addWidget(cam_grp)

        # Hardver & Erőforrás Használat
        hw_grp = QGroupBox("Hardver Erőforrások (CPU / GPU / RAM)")
        hw_form = QFormLayout(hw_grp)
        hw_form.setSpacing(6)

        self._lbl_diag_gpu = QLabel("NVIDIA RTX 3050 (6.1 GB VRAM)")
        self._lbl_diag_gpu.setStyleSheet("font-weight: bold; color: #0F5132;")
        self._lbl_diag_cpu = QLabel("— %")
        self._lbl_diag_ram = QLabel("— MB / — %")

        hw_form.addRow("Grafikus Kártya (GPU):", self._lbl_diag_gpu)
        hw_form.addRow("CPU Használat:", self._lbl_diag_cpu)
        hw_form.addRow("RAM Használat:", self._lbl_diag_ram)

        layout.addWidget(hw_grp)

        # Feldolgozási Állapot
        proc_grp = QGroupBox("Feldolgozási Állapot")
        proc_form = QFormLayout(proc_grp)
        proc_form.setSpacing(6)

        self._lbl_diag_det_status = QLabel("Nincs detektálás")
        self._lbl_diag_pos3d = QLabel("—")

        proc_form.addRow("Követési Státusz:", self._lbl_diag_det_status)
        proc_form.addRow("Legutóbbi 3D Pozíció:", self._lbl_diag_pos3d)

        layout.addWidget(proc_grp)
        layout.addStretch(1)
        return tab

    def _build_log_dock(self) -> None:
        dock = QDockWidget("Rendszernapló", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)

        self._log_panel = QPlainTextEdit()
        self._log_panel.setObjectName("log_console")
        self._log_panel.setReadOnly(True)
        self._log_panel.setMaximumBlockCount(300)

        dock.setWidget(self._log_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.setMaximumHeight(140)

    def _build_status_bar(self) -> None:
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("DEIK Robot Kapus készenlétben. Kattints az INDÍTÁS gombra!")

        creators_lbl = QLabel(" Készítők: Morvai Roland & Rácz Donát (BSc Mérnökinformatikus) | DEIK v1.0 ")
        creators_lbl.setStyleSheet("color: #0F5132; font-weight: 700; font-size: 11px; padding-right: 8px;")
        self._status_bar.addPermanentWidget(creators_lbl)

    @pyqtSlot()
    def _show_about_dialog(self) -> None:
        """Megnyitja az egységes fehér témájú Névjegy & Készítők ablakot."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Névjegy & Fejlesztési Információk")
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet(
            "QDialog, QWidget { background-color: #FFFFFF; color: #0F172A; font-family: 'Segoe UI', sans-serif; }"
        )

        vbox = QVBoxLayout(dlg)
        vbox.setSpacing(14)
        vbox.setContentsMargins(20, 20, 20, 20)

        # Logók egymás mellett
        logo_box = QHBoxLayout()
        logo_box.setSpacing(20)
        logo_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

        deik_logo_path = "assets/deik_logo.png"
        if os.path.exists(deik_logo_path):
            l1 = QLabel()
            l1.setPixmap(QPixmap(deik_logo_path).scaledToHeight(80, Qt.TransformationMode.SmoothTransformation))
            logo_box.addWidget(l1)

        rgk_logo_path = "assets/logo.png"
        if os.path.exists(rgk_logo_path):
            l2 = QLabel()
            l2.setPixmap(QPixmap(rgk_logo_path).scaledToHeight(80, Qt.TransformationMode.SmoothTransformation))
            logo_box.addWidget(l2)

        vbox.addLayout(logo_box)

        # Cím Fejléc
        title = QLabel("DEIK ROBOT FOCI KAPUS")
        title.setStyleSheet("font-weight: 800; font-size: 19px; color: #0F5132;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title)

        sub = QLabel("Debreceni Egyetem Informatikai Kar")
        sub.setStyleSheet("font-size: 12px; color: #D97706; font-weight: 700;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(sub)

        ver = QLabel("Verzió 1.0.0 (2026)")
        ver.setStyleSheet("background-color: #F1F5F9; color: #475569; font-weight: 600; font-size: 11px; border-radius: 8px; padding: 2px 10px;")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(ver)

        # Egyetlen Kártya Konténer
        card = QWidget()
        card.setStyleSheet(
            "background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;"
        )
        card_vbox = QVBoxLayout(card)
        card_vbox.setSpacing(8)

        lbl_dev = QLabel("FEJLESZTŐK / KÉSZÍTŐK")
        lbl_dev.setStyleSheet("font-weight: 800; font-size: 11px; color: #0F5132; letter-spacing: 0.5px;")
        card_vbox.addWidget(lbl_dev)

        m1 = QLabel("• Morvai Roland – BSc Mérnökinformatikus (DEIK)")
        m1.setStyleSheet("font-weight: 700; color: #0F172A; font-size: 13px;")
        card_vbox.addWidget(m1)

        m2 = QLabel("• Rácz Donát – BSc Mérnökinformatikus (DEIK)")
        m2.setStyleSheet("font-weight: 700; color: #0F172A; font-size: 13px;")
        card_vbox.addWidget(m2)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #E2E8F0;")
        card_vbox.addWidget(line)

        lbl_info = QLabel("PROJEKT & TECHNOLÓGIA")
        lbl_info.setStyleSheet("font-weight: 800; font-size: 11px; color: #0F5132; letter-spacing: 0.5px;")
        card_vbox.addWidget(lbl_info)

        details = QLabel(
            "• <b>Projekt:</b> Valós idejű sztereó látórendszer és trajektória előrejelzés robot kapushoz.<br>"
            "• <b>Szoftver stack:</b> Python 3.12, PyQt6, OpenCV, PyTorch, YOLOv10 (CUDA GPU Acceleration)"
        )
        details.setStyleSheet("color: #334155; font-size: 12px;")
        card_vbox.addWidget(details)

        vbox.addWidget(card)

        # Rendben Gomb
        btn_ok = QPushButton("Rendben")
        btn_ok.setFixedHeight(36)
        btn_ok.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_ok.setStyleSheet(
            "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: bold; border-radius: 6px; font-size: 13px; border: none; }"
            "QPushButton:hover { background-color: #146C43; }"
        )
        btn_ok.clicked.connect(dlg.accept)
        vbox.addWidget(btn_ok, alignment=Qt.AlignmentFlag.AlignCenter)

        dlg.exec()

    # ------------------------------------------------------------------
    # Eseménykezelők
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_open_calibration(self) -> None:
        """Megnyitja a számára tervezett Kalibrációs Munkafolyamat Dialógot."""
        if self._is_running:
            reply = QMessageBox.question(
                self,
                "Kamera foglalt",
                "A követő rendszer jelenleg fut.\n"
                "A kalibrációhoz le kell állítani a követőt (kimeríti a kamerát).\n\n"
                "Leállítsuk automatikusan, és megnyissuk a kalibrációs menütt?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply != QMessageBox.StandardButton.Yes:
                return
            self._stop_tracker()
            # Várjuk meg a leállást
            import time as _time
            _time.sleep(0.5)

        dlg = CalibrationDialog(self._config, parent=self)
        dlg.exec()
        logger.info("Kalibrációs dialog bezárva.")

    @pyqtSlot()
    def _toggle_fullscreen(self) -> None:
        """Vált a teljes képernyős és az ablakos nézet között (F11 / Esc)."""
        if self.isFullScreen():
            self.showNormal()
        else:
            self.showFullScreen()

    @pyqtSlot()
    def _on_start_stop(self) -> None:
        if self._is_running:
            self._stop_tracker()
        else:
            self._start_tracker()

    @pyqtSlot()
    def _on_clear_history(self) -> None:
        if hasattr(self, "_goal_view") and self._goal_view:
            self._goal_view.clear_history()
        if hasattr(self, "_goal_view_full") and self._goal_view_full:
            self._goal_view_full.clear_history()
        logger.info("Lövés történet törölve.")

    @pyqtSlot(np.ndarray, np.ndarray, dict)
    def _on_frames_ready(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        stats: dict
    ) -> None:
        # ROI zoom opció olvasása
        left_roi_zoom = self._get_roi_zoom_frame(frame_left, "left")
        right_roi_zoom = self._get_roi_zoom_frame(frame_right, "right")

        self._display_frame(left_roi_zoom, self._cam_label_left)
        self._display_frame(right_roi_zoom, self._cam_label_right)

        if hasattr(self, "_cam_label_left_full") and self._cam_label_left_full:
            self._display_frame(left_roi_zoom, self._cam_label_left_full)
        if hasattr(self, "_cam_label_right_full") and self._cam_label_right_full:
            self._display_frame(right_roi_zoom, self._cam_label_right_full)

        if stats["pos_valid"]:
            self._lbl_x.setText(f"{stats['x_3d']:+.1f}")
            self._lbl_y.setText(f"{stats['y_3d']:+.1f}")
            self._lbl_z.setText(f"{stats['z_3d']:.1f}")
            speed_kmh = stats["speed_ms"] * 3.6
            self._lbl_speed.setText(f"{stats['speed_ms']:.1f} m/s ({speed_kmh:.1f} km/h)")
        else:
            self._lbl_x.setText("—")
            self._lbl_y.setText("—")
            self._lbl_z.setText("—")
            self._lbl_speed.setText("—")

        impact: Optional[ImpactPrediction] = stats.get("impact")
        if impact and impact.valid:
            self._lbl_impact.setText(f"X:{impact.x_mm:+.0f} Y:{impact.y_mm:.0f} mm")
            self._lbl_time.setText(f"{impact.time_to_impact_s:.3f} mp")
            zone_text = self._determine_defense_zone(impact.x_mm, impact.y_mm)
            self._lbl_zone.setText(zone_text)

            if hasattr(self, "_goal_view") and self._goal_view:
                self._goal_view.update_impact(
                    x_mm=impact.x_mm,
                    y_mm=impact.y_mm,
                    confidence=impact.confidence,
                    time_to_impact_s=impact.time_to_impact_s,
                    in_goal=impact.in_goal,
                )
            if hasattr(self, "_goal_view_full") and self._goal_view_full:
                self._goal_view_full.update_impact(
                    x_mm=impact.x_mm,
                    y_mm=impact.y_mm,
                    confidence=impact.confidence,
                    time_to_impact_s=impact.time_to_impact_s,
                    in_goal=impact.in_goal,
                )
        else:
            self._lbl_impact.setText("—")
            self._lbl_time.setText("—")
            self._lbl_zone.setText("— KÖZÉP —")
            if hasattr(self, "_goal_view") and self._goal_view:
                self._goal_view.update_impact(None, None, 0.0, 0.0)
            if hasattr(self, "_goal_view_full") and self._goal_view_full:
                self._goal_view_full.update_impact(None, None, 0.0, 0.0)

        det_str = "Mindkét kamerában" if stats["both_found"] else (
            "Csak bal kamera" if stats["left_found"] else (
            "Csak jobb kamera" if stats["right_found"] else "Nincs detektálás"
        ))
        self._status_bar.showMessage(f"Labda követés: {det_str} | Sztereó FPS: {stats['pair_fps']:.0f}")

        # Rendszer Adatok & Diagnosztika Lapfül Élő Frissítése
        if hasattr(self, "_lbl_diag_fps_l"):
            self._lbl_diag_fps_l.setText(f"{stats['cam_fps_left']:.1f} FPS")
            self._lbl_diag_temp_l.setText(f"{stats['temp_left']:.1f} °C")
            self._lbl_diag_fps_r.setText(f"{stats['cam_fps_right']:.1f} FPS")
            self._lbl_diag_temp_r.setText(f"{stats['temp_right']:.1f} °C")
            self._lbl_diag_fps_pair.setText(f"{stats['pair_fps']:.1f} FPS")
            self._lbl_diag_fps_det.setText(f"{stats['det_fps']:.1f} FPS")

            cal_str = "OK (Stereo Calibrated)" if stats.get("calibrated", True) else "HIBA (Nincs Kalibrálva)"
            self._lbl_diag_calib.setText(cal_str)

            self._lbl_diag_det_status.setText(det_str)
            if stats["pos_valid"]:
                self._lbl_diag_pos3d.setText(f"X:{stats['x_3d']:+.1f} Y:{stats['y_3d']:+.1f} Z:{stats['z_3d']:.1f} mm")
            else:
                self._lbl_diag_pos3d.setText("— (Nincs 3D detektálás)")

    def _determine_defense_zone(self, x_mm: float, y_mm: float) -> str:
        horiz = "BAL" if x_mm < -600 else ("JOBB" if x_mm > 600 else "KÖZÉP")
        vert = "FELSŐ" if y_mm > 1000 else "ALSÓ"
        return f"{horiz} {vert} SZEKTOR"

    @pyqtSlot(str)
    def _on_worker_error(self, error_msg: str) -> None:
        logger.error("Tracker hiba: %s", error_msg)
        self._stop_tracker()
        QMessageBox.critical(self, "Rendszerhiba", error_msg)

    @pyqtSlot()
    def _on_worker_stopped(self) -> None:
        self._is_running = False
        self._btn_start.setText("INDÍTÁS")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setStyleSheet("")
        self._pill_sys.setText(" INAKTÍV ")
        self._pill_sys.setStyleSheet(get_status_pill_style("info"))
        self._status_bar.showMessage("Rendszer leállítva.")

    @pyqtSlot(str)
    def _on_log_message(self, msg: str) -> None:
        self._log_panel.appendPlainText(msg)
        cursor = self._log_panel.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_panel.setTextCursor(cursor)

    @pyqtSlot()
    def _update_system_status(self) -> None:
        """Másodpercenként frissíti a hardveres CPU és RAM erőforrás telemetriát."""
        if hasattr(self, "_lbl_diag_cpu"):
            try:
                cpu_p = psutil.cpu_percent()
                self._lbl_diag_cpu.setText(f"{cpu_p:.1f} %")

                mem = psutil.virtual_memory()
                mem_mb = mem.used / (1024 * 1024)
                mem_tot = mem.total / (1024 * 1024)
                self._lbl_diag_ram.setText(f"{mem_mb:.0f} MB / {mem_tot:.0f} MB ({mem.percent:.1f} %)")
            except Exception as e:
                logger.debug("Hardware stats update error: %s", e)

    def _load_gui_settings(self) -> None:
        path = "config/gui_settings.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    gui_cfg = json.load(f)
                for side in ["left", "right"]:
                    if side in gui_cfg:
                        if side not in self._config["camera"]:
                            self._config["camera"][side] = {}
                        self._config["camera"][side].update(gui_cfg[side])
            except Exception as e:
                logger.error("GUI beállítások betöltése sikertelen: %s", e)

    @pyqtSlot()
    def _save_gui_settings(self) -> None:
        gui_cfg = {}
        for side in ["left", "right"]:
            if hasattr(self, "_cam_widgets") and side in self._cam_widgets:
                w = self._cam_widgets[side]
                rot_index = w["combo_rot"].currentIndex()
                rot_val = [0, 90, 180, 270][rot_index] if rot_index < 4 else 0

                gui_cfg[side] = {
                    "offset_x": w["spin_x"].value(),
                    "offset_y": w["spin_y"].value(),
                    "roi_enabled": w["chk_roi"].isChecked(),
                    "roi_x_min": w["spin_xmin"].value(),
                    "roi_x_max": w["spin_xmax"].value(),
                    "roi_y_min": w["spin_ymin"].value(),
                    "roi_y_max": w["spin_ymax"].value(),
                    "roi_zoom": w["chk_roi_zoom"].isChecked(),
                    "exposure_time_us": w["spin_exp"].value(),
                    "gain_db": w["spin_gain"].value(),
                    "flip_h": w["chk_fliph"].isChecked(),
                    "flip_v": w["chk_flipv"].isChecked(),
                    "rotation": rot_val,
                }
                if side not in self._config["camera"]:
                    self._config["camera"][side] = {}
                self._config["camera"][side].update(gui_cfg[side])

        try:
            os.makedirs("config", exist_ok=True)
            with open("config/gui_settings.json", "w", encoding="utf-8") as f:
                json.dump(gui_cfg, f, indent=4, ensure_ascii=False)
            QMessageBox.information(self, "Mentés Sikeres", "A kamera beállítások sikeresen elmentve!")
        except Exception as e:
            QMessageBox.critical(self, "Mentés Hiba", f"Nem sikerült menteni:\n{e}")

    @pyqtSlot()
    def _reset_gui_settings(self) -> None:
        reply = QMessageBox.question(
            self, "Alapértelmezések visszaállítása",
            "Biztosan visszaállítod az összes kamera beállítást?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            for side in ["left", "right"]:
                if hasattr(self, "_cam_widgets") and side in self._cam_widgets:
                    w = self._cam_widgets[side]
                    w["spin_x"].setValue(0)
                    w["spin_y"].setValue(0)
                    w["chk_roi"].setChecked(False)
                    w["spin_xmin"].setValue(0)
                    w["spin_xmax"].setValue(100)
                    w["spin_ymin"].setValue(10)
                    w["spin_ymax"].setValue(90)
                    w["chk_roi_zoom"].setChecked(False)
                    w["spin_exp"].setValue(3000)
                    w["spin_gain"].setValue(0.0)
                    w["chk_fliph"].setChecked(False)
                    w["chk_flipv"].setChecked(False)
                    w["combo_rot"].setCurrentIndex(0)

    def _start_tracker(self) -> None:
        logger.info("Tracker indítása...")

        self._worker = TrackerWorker(self._config, parent=self)
        self._worker.frames_ready.connect(self._on_frames_ready)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.tracker_stopped.connect(self._on_worker_stopped)
        self._worker.start()

        self._is_running = True
        self._btn_start.setText("LEÁLLÍTÁS")
        self._btn_start.setObjectName("btn_stop")
        self._btn_start.setStyleSheet("")
        self._pill_sys.setText(" AKTÍV FUTÁS ")
        self._pill_sys.setStyleSheet(get_status_pill_style("ok"))
        self._status_bar.showMessage("Rendszer aktív – sztereó feldolgozás fut...")

    def _stop_tracker(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.wait(5000)
            self._worker = None

    def _display_frame(self, frame: np.ndarray, label: "ZoomableLabel") -> None:
        """Átadja a frame-et a ZoomableLabel-nek megjelenítésre (zoom/pan megőrződik)."""
        label.set_frame(frame)

    def _get_roi_zoom_frame(self, frame: np.ndarray, side: str) -> np.ndarray:
        """
        Ha a ROI Zoom aktív (checkbox be van kapcsolva) és a ROI engedélyezett,
        kivágja a ROI területet a képből, és azt adja vissza megjelenítésre.
        Különben az eredeti (teljes) képet adja vissza.
        """
        if not hasattr(self, "_cam_widgets") or side not in self._cam_widgets:
            return frame
        w = self._cam_widgets[side]
        if not w["chk_roi_zoom"].isChecked() or not w["chk_roi"].isChecked():
            return frame

        # ROI koordináták kiszámítása
        fh, fw = frame.shape[:2]
        xmin_rel = w["spin_xmin"].value() / 100.0
        xmax_rel = w["spin_xmax"].value() / 100.0
        ymin_rel = w["spin_ymin"].value() / 100.0
        ymax_rel = w["spin_ymax"].value() / 100.0

        x1 = max(0, int(fw * xmin_rel))
        x2 = min(fw, int(fw * xmax_rel))
        y1 = max(0, int(fh * ymin_rel))
        y2 = min(fh, int(fh * ymax_rel))

        if x2 <= x1 or y2 <= y1:
            return frame

        cropped = frame[y1:y2, x1:x2].copy()
        # ROI ZOOM felirat rárajzolása
        cv2.putText(
            cropped, "ROI ZOOM",
            (6, min(22, cropped.shape[0] - 4)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2
        )
        return cropped

    def _setup_log_handler(self) -> None:
        self._qt_log_handler = _QtLogHandler()
        formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S")
        self._qt_log_handler.setFormatter(formatter)
        self._qt_log_handler.signals.log_msg.connect(self._on_log_message)
        logging.getLogger().addHandler(self._qt_log_handler)

    def closeEvent(self, event) -> None:
        self._stop_tracker()
        event.accept()
