"""
DEIK Robot Foci Kapus – Fő GUI Ablak (PyQt6)
=============================================

Ez a modul a teljes grafikus felhasználói felületet valósítja meg.

Felépítés:
    - Toolbar: rendszer vezérlése (start/stop, kamera teszt)
    - Bal panel:  Bal kamera élő képe + detektálási overlay
    - Jobb panel: Jobb kamera élő képe + detektálási overlay
    - Alsó panel: Kapu nézet + statisztikák
    - Log panel: Rendszerüzenetek

Feldolgozási pipeline (háttérszálon):
    CameraManager → BallDetector → StereoTriangulator →
    TrajectoryPredictor → GUI frissítés (Qt Signal/Slot)
"""

import logging
import sys
import time
import json
import os
from typing import Any, Dict, Optional

import cv2
import numpy as np
import psutil
from PyQt6.QtCore import (
    QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
)
from PyQt6.QtGui import (
    QAction, QColor, QFont, QImage, QPixmap, QTextCursor
)
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QDockWidget, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QPushButton,
    QSizePolicy, QSpinBox, QStatusBar, QTabWidget,
    QToolBar, QVBoxLayout, QWidget, QSlider, QDoubleSpinBox
)

from camera.camera_manager import CameraManager, StereoPair
from detection.ball_detector import BallDetector, StereoBallDetection
from detection.kalman_tracker import KalmanTracker2D
from stereo.triangulator import StereoTriangulator
from prediction.trajectory_predictor import TrajectoryPredictor, ImpactPrediction
from gui.goal_view import GoalViewWidget

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Qt Log handler – napló üzenetek a GUI-ba
# --------------------------------------------------------------------------- #

class _QtLogSignals(QObject):
    log_msg = pyqtSignal(str)


class _QtLogHandler(logging.Handler):
    """Napló üzeneteket a Qt signal rendszerén keresztül a GUI-ba irányítja."""

    def __init__(self):
        super().__init__()
        self.signals = _QtLogSignals()

    def emit(self, record: logging.LogRecord) -> None:
        msg = self.format(record)
        self.signals.log_msg.emit(msg)


# --------------------------------------------------------------------------- #
# Háttérszál – a teljes vizualizációs pipeline
# --------------------------------------------------------------------------- #

class TrackerWorker(QThread):
    """
    Qt háttérszál: futtatja a teljes képfeldolgozási pipeline-t.

    A GUI szálával a pyqtSignal/pyqtSlot mechanizmuson kommunikál,
    ami szálbiztos és Qt-konform módja az inter-thread kommunikációnak.

    Kibocsátott signalok:
        frames_ready:    Új frame-pár feldolgozva (bal kép, jobb kép, stats)
        error_occurred:  Végzetes hiba történt (hibaüzenet)
        tracker_stopped: A szál normálisan leállt
    """

    frames_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    error_occurred = pyqtSignal(str)
    tracker_stopped = pyqtSignal()

    def __init__(self, config: dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._config = config
        self._running = False

        # Komponensek (a run() metódusban inicializáljuk – szálban kell!)
        self._cam_manager: Optional[CameraManager] = None
        self._detector: Optional[BallDetector] = None
        self._triangulator: Optional[StereoTriangulator] = None
        self._predictor: Optional[TrajectoryPredictor] = None

        # Per-kamera Kalman szűrők (2D simítás)
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

        # GUI frissítési ráta korlát (60 FPS-re)
        self._gui_fps_limit = float(config.get("gui", {}).get("gui_fps_limit", 60))
        self._gui_interval = 1.0 / self._gui_fps_limit
        self._last_gui_emit = 0.0

        logger.debug("TrackerWorker létrehozva")

    def stop(self) -> None:
        """Kéri a szál leállítását (graceful shutdown)."""
        self._running = False

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

    def run(self) -> None:
        """
        A háttérszál fő ciklusa.

        Inicializálja az összes komponenst, majd folyamatosan futtatja:
        1. Sztereó frame-pár olvasás
        2. Labda detektálás (YOLOv10n)
        3. Kalman simítás
        4. 3D háromszögelés
        5. Trajektória előrejelzés
        6. GUI frissítés (rate-limited)
        """
        logger.info("TrackerWorker szál elindult")
        self._running = True

        # --- Komponensek inicializálása ---
        try:
            self._cam_manager = CameraManager(self._config)
            self._detector = BallDetector(self._config["detection"])
            self._triangulator = StereoTriangulator(self._config)
            self._predictor = TrajectoryPredictor(self._config)

            # Kalibrálás betöltése (nem fatális hiba ha nincs)
            cal_file = self._config.get("stereo", {}).get(
                "calibration_file", "data/calibration/stereo_calibration.npz"
            )
            self._triangulator.load_calibration(cal_file)

        except Exception as exc:
            error_msg = f"Komponens inicializálási hiba: {exc}"
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return

        # --- Kamerák megnyitása ---
        if not self._cam_manager.open():
            error_msg = "Kamerák megnyitása sikertelen! Ellenőrizd az USB kapcsolatokat."
            logger.error(error_msg)
            self.error_occurred.emit(error_msg)
            return

        logger.info("TrackerWorker főciklus indítása...")

        # --- Főciklus ---
        try:
            self._main_loop()
        except Exception as exc:
            logger.exception("Váratlan hiba a főciklusban: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            if self._cam_manager:
                self._cam_manager.close()
            logger.info("TrackerWorker szál leállt")
            self.tracker_stopped.emit()

    def _main_loop(self) -> None:
        """A tényleges feldolgozási ciklus."""
        overlay_cfg = self._config.get("gui", {}).get("overlay", {})

        while self._running:
            # 1. Sztereó frame-pár olvasás
            pair: StereoPair = self._cam_manager.read_stereo_pair()

            if not pair.success:
                # Ha nincs frame: rövid várakozás és folytatás
                time.sleep(0.005)
                continue

            frame_left = pair.left.image.copy()
            frame_right = pair.right.image.copy()

            # 2. Labda detektálás (YOLOv10n, GPU)
            detection: StereoBallDetection = self._detector.detect(
                pair.left.image, pair.right.image
            )

            # 3. 2D Kalman simítás mindkét kamera képén
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

            # 4. 3D Háromszögelés (ha mindkét kamerában megtaláltuk a labdát)
            pos_3d = None
            if detection.both_found:
                pos_3d = self._triangulator.triangulate(
                    left_point=(left_det.x, left_det.y),
                    right_point=(right_det.x, right_det.y),
                )

            # 5. Trajektória frissítése és előrejelzés
            impact: Optional[ImpactPrediction] = None
            if pos_3d is not None:
                self._predictor.add_measurement(
                    x_mm=float(pos_3d[0]),
                    y_mm=float(pos_3d[1]),
                    z_mm=float(pos_3d[2]),
                )
                impact = self._predictor.get_impact_prediction()
            else:
                # Ha elvesztettük a labdát: predictor reset
                if not detection.left.found and not detection.right.found:
                    self._predictor.reset()
                    self._kalman_left.reset()
                    self._kalman_right.reset()

            # 6. Overlay rajzolása a képekre
            if overlay_cfg.get("show_detection_box", True):
                self._detector.draw_detection(frame_left, left_det)
                self._detector.draw_detection(frame_right, right_det)

            # 7. GUI frissítés (rate-limited)
            now = time.perf_counter()
            if now - self._last_gui_emit >= self._gui_interval:
                self._last_gui_emit = now

                # Statisztika összegyűjtése
                cam_status = self._cam_manager.get_camera_status()
                vx, vy, vz = self._predictor.estimated_velocity_mm_s

                stats = {
                    # Kamera adatok
                    "cam_fps_left":  cam_status["fps_left"],
                    "cam_fps_right": cam_status["fps_right"],
                    "pair_fps":      cam_status["pair_fps"],
                    "temp_left":     cam_status["temp_left"],
                    "temp_right":    cam_status["temp_right"],
                    # Detektálás adatok
                    "det_fps":       detection.det_fps,
                    "left_found":    left_det.found,
                    "right_found":   right_det.found,
                    "both_found":    detection.both_found,
                    "left_conf":     left_det.confidence,
                    "right_conf":    right_det.confidence,
                    # 3D pozíció
                    "x_3d": float(pos_3d[0]) if pos_3d is not None else 0.0,
                    "y_3d": float(pos_3d[1]) if pos_3d is not None else 0.0,
                    "z_3d": float(pos_3d[2]) if pos_3d is not None else 0.0,
                    "pos_valid": pos_3d is not None,
                    # Sebesség
                    "vx_mms": vx, "vy_mms": vy, "vz_mms": vz,
                    "speed_ms": np.sqrt(vx**2 + vy**2 + vz**2) / 1000.0,
                    # Trajektória előrejelzés
                    "impact": impact,
                    # Kalibrálás
                    "calibrated": self._triangulator.is_calibrated,
                }

                # Signál küldése a GUI szálnak
                self.frames_ready.emit(frame_left, frame_right, stats)


# --------------------------------------------------------------------------- #
# Fő ablak
# --------------------------------------------------------------------------- #

class MainWindow(QMainWindow):
    """
    A fő alkalmazási ablak.

    Koordinálja a GUI komponenseket és a háttérszálat.
    """

    def __init__(self, config: dict):
        super().__init__()
        self._config = config
        self._load_gui_settings()
        self._worker: Optional[TrackerWorker] = None
        self._is_running = False

        # Ablak alap beállítások
        gui_cfg = config.get("gui", {})
        self.setWindowTitle(gui_cfg.get(
            "window_title", "DEIK Robot Foci Kapus"
        ))
        self.setMinimumSize(1200, 750)

        # Sötét téma
        self._apply_dark_theme()

        # Qt Log handler regisztrálása
        self._setup_log_handler()

        # UI felépítése
        self._build_ui()

        # Rendszer státuszfrissítő timer (1 Hz)
        self._status_timer = QTimer(self)
        self._status_timer.timeout.connect(self._update_system_status)
        self._status_timer.start(1000)

        logger.info("MainWindow inicializálva")

    # ------------------------------------------------------------------
    # UI felépítés
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Megépíti a teljes GUI struktúrát."""
        self._build_toolbar()
        self._build_central_widget()
        self._build_camera_settings_dock()
        self._build_stats_dock()
        self._build_log_dock()
        self._build_status_bar()

    def _build_toolbar(self) -> None:
        """Eszköztár létrehozása (start/stop, kamera teszt, stb.)."""
        toolbar = QToolBar("Vezérlés")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # Start / Stop gomb
        self._btn_start = QPushButton("▶  INDÍTÁS")
        self._btn_start.setFixedHeight(36)
        self._btn_start.setStyleSheet(
            "QPushButton { background-color: #2ea043; color: white; font-weight: bold; "
            "border-radius: 6px; padding: 0 20px; font-size: 13px; }"
            "QPushButton:hover { background-color: #3fb950; }"
            "QPushButton:pressed { background-color: #238636; }"
        )
        self._btn_start.clicked.connect(self._on_start_stop)
        toolbar.addWidget(self._btn_start)

        toolbar.addSeparator()

        # Kalibrálás állapot jelző
        self._lbl_calib = QLabel("  ⚠  Kalibrálás szükséges")
        self._lbl_calib.setStyleSheet("color: #f0a500; font-weight: bold;")
        toolbar.addWidget(self._lbl_calib)

        toolbar.addSeparator()

        # FPS jelző
        self._lbl_fps = QLabel("  FPS: —")
        self._lbl_fps.setStyleSheet("color: #8be9fd; font-family: monospace;")
        toolbar.addWidget(self._lbl_fps)

        # Detektálás FPS
        self._lbl_det_fps = QLabel("  Det: —")
        self._lbl_det_fps.setStyleSheet("color: #50fa7b; font-family: monospace;")
        toolbar.addWidget(self._lbl_det_fps)

        toolbar.addSeparator()

        # Törlés gomb
        btn_clear = QPushButton("🗑  Historika törlése")
        btn_clear.setFixedHeight(32)
        btn_clear.setStyleSheet(
            "QPushButton { background-color: #3d3d4d; color: #ccc; "
            "border-radius: 5px; padding: 0 12px; }"
            "QPushButton:hover { background-color: #555; }"
        )
        btn_clear.clicked.connect(self._on_clear_history)
        toolbar.addWidget(btn_clear)

    def _build_central_widget(self) -> None:
        """Fő tartalom terület: kamera nézetek + kapu vizualizátor."""
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # ── Felső sáv: Bal és Jobb kamera képe egymás mellett ─────────
        cameras_layout = QHBoxLayout()
        cameras_layout.setSpacing(6)

        # Bal kamera nézet
        left_group = QGroupBox("Bal kamera  [X = -2450 mm]")
        left_group.setStyleSheet("QGroupBox { color: #bd93f9; font-weight: bold; }")
        left_layout = QVBoxLayout(left_group)
        self._cam_label_left = QLabel("Kamera nem aktív")
        self._cam_label_left.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label_left.setStyleSheet("background-color: #111; color: #666; border-radius: 4px;")
        self._cam_label_left.setMinimumSize(480, 300)
        self._cam_label_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self._cam_label_left)
        cameras_layout.addWidget(left_group)

        # Jobb kamera nézet
        right_group = QGroupBox("Jobb kamera  [X = +2450 mm]")
        right_group.setStyleSheet("QGroupBox { color: #ff79c6; font-weight: bold; }")
        right_layout = QVBoxLayout(right_group)
        self._cam_label_right = QLabel("Kamera nem aktív")
        self._cam_label_right.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._cam_label_right.setStyleSheet("background-color: #111; color: #666; border-radius: 4px;")
        self._cam_label_right.setMinimumSize(480, 300)
        self._cam_label_right.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        right_layout.addWidget(self._cam_label_right)
        cameras_layout.addWidget(right_group)

        main_layout.addLayout(cameras_layout, stretch=3)

        # ── Alsó sáv: Kapu vizualizátor ───────────────────────────────
        bottom_layout = QHBoxLayout()

        # Kapu vizualizátor
        goal_group = QGroupBox("Kapu Nézet — Becsapódási Pont Előrejelzés")
        goal_group.setStyleSheet("QGroupBox { color: #f1fa8c; font-weight: bold; }")
        goal_layout = QVBoxLayout(goal_group)
        self._goal_view = GoalViewWidget(self._config)
        goal_layout.addWidget(self._goal_view)
        bottom_layout.addWidget(goal_group, stretch=2)

        # 3D pozíció kijelző
        pos_group = QGroupBox("3D Pozíció")
        pos_group.setStyleSheet("QGroupBox { color: #8be9fd; font-weight: bold; }")
        pos_layout = QFormLayout(pos_group)
        pos_layout.setSpacing(8)

        font_mono = QFont("Consolas", 11)
        self._lbl_x = QLabel("—")
        self._lbl_y = QLabel("—")
        self._lbl_z = QLabel("—")
        self._lbl_speed = QLabel("—")
        self._lbl_impact_x = QLabel("—")
        self._lbl_impact_y = QLabel("—")
        self._lbl_impact_t = QLabel("—")
        self._lbl_impact_conf = QLabel("—")

        for lbl in [self._lbl_x, self._lbl_y, self._lbl_z, self._lbl_speed,
                    self._lbl_impact_x, self._lbl_impact_y,
                    self._lbl_impact_t, self._lbl_impact_conf]:
            lbl.setFont(font_mono)
            lbl.setStyleSheet("color: #f8f8f2;")

        pos_layout.addRow("X [mm]:", self._lbl_x)
        pos_layout.addRow("Y [mm]:", self._lbl_y)
        pos_layout.addRow("Z [mm]:", self._lbl_z)
        pos_layout.addRow("Sebesség:", self._lbl_speed)
        pos_layout.addRow("— Becsapódás —", QLabel(""))
        pos_layout.addRow("X [mm]:", self._lbl_impact_x)
        pos_layout.addRow("Y [mm]:", self._lbl_impact_y)
        pos_layout.addRow("Idő [s]:", self._lbl_impact_t)
        pos_layout.addRow("Megbízhatóság:", self._lbl_impact_conf)

        bottom_layout.addWidget(pos_group, stretch=1)
        main_layout.addLayout(bottom_layout, stretch=2)

    def _build_camera_settings_dock(self) -> None:
        """Kamera beállítások lebegő panel létrehozása."""
        dock = QDockWidget("Kamera Beállítások", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable | QDockWidget.DockWidgetFeature.DockWidgetFloatable)

        tabs = QTabWidget()
        
        # --- Bal kamera tab ---
        left_tab = QWidget()
        left_layout = QFormLayout(left_tab)
        
        # Exposure
        self._exp_l = QSlider(Qt.Orientation.Horizontal)
        self._exp_l.setRange(100, 10000)
        self._exp_l.setValue(self._config["camera"].get("left", {}).get("exposure_time_us", self._config["camera"].get("exposure_time_us", 3000)))
        self._exp_lbl_l = QLabel(f"{self._exp_l.value()} us")
        self._exp_l.valueChanged.connect(lambda v: self._exp_lbl_l.setText(f"{v} us"))
        self._exp_l.valueChanged.connect(lambda v: self._worker.set_camera_exposure(True, v) if self._worker else None)
        
        # Gain
        self._gain_l = QDoubleSpinBox()
        self._gain_l.setRange(0.0, 15.0)
        self._gain_l.setSingleStep(0.5)
        self._gain_l.setValue(self._config["camera"].get("left", {}).get("gain_db", self._config["camera"].get("gain_db", 0.0)))
        self._gain_l.valueChanged.connect(lambda v: self._worker.set_camera_gain(True, v) if self._worker else None)
        
        # AWB
        self._awb_l = QCheckBox("Automatikus (AWB)")
        self._awb_l.setChecked(self._config["camera"].get("left", {}).get("auto_white_balance", self._config["camera"].get("auto_white_balance", True)))
        self._awb_l.toggled.connect(lambda v: self._worker.set_camera_awb(True, v) if self._worker else None)
        
        left_layout.addRow("Záridő:", self._exp_lbl_l)
        left_layout.addRow("", self._exp_l)
        left_layout.addRow("Erősítés (dB):", self._gain_l)
        left_layout.addRow("Fehéregy.:", self._awb_l)
        tabs.addTab(left_tab, "Bal")

        # --- Jobb kamera tab ---
        right_tab = QWidget()
        right_layout = QFormLayout(right_tab)
        
        # Exposure
        self._exp_r = QSlider(Qt.Orientation.Horizontal)
        self._exp_r.setRange(100, 10000)
        self._exp_r.setValue(self._config["camera"].get("right", {}).get("exposure_time_us", self._config["camera"].get("exposure_time_us", 3000)))
        self._exp_lbl_r = QLabel(f"{self._exp_r.value()} us")
        self._exp_r.valueChanged.connect(lambda v: self._exp_lbl_r.setText(f"{v} us"))
        self._exp_r.valueChanged.connect(lambda v: self._worker.set_camera_exposure(False, v) if self._worker else None)
        
        # Gain
        self._gain_r = QDoubleSpinBox()
        self._gain_r.setRange(0.0, 15.0)
        self._gain_r.setSingleStep(0.5)
        self._gain_r.setValue(self._config["camera"].get("right", {}).get("gain_db", self._config["camera"].get("gain_db", 0.0)))
        self._gain_r.valueChanged.connect(lambda v: self._worker.set_camera_gain(False, v) if self._worker else None)
        
        # AWB
        self._awb_r = QCheckBox("Automatikus (AWB)")
        self._awb_r.setChecked(self._config["camera"].get("right", {}).get("auto_white_balance", self._config["camera"].get("auto_white_balance", True)))
        self._awb_r.toggled.connect(lambda v: self._worker.set_camera_awb(False, v) if self._worker else None)
        
        right_layout.addRow("Záridő:", self._exp_lbl_r)
        right_layout.addRow("", self._exp_r)
        right_layout.addRow("Erősítés (dB):", self._gain_r)
        right_layout.addRow("Fehéregy.:", self._awb_r)
        tabs.addTab(right_tab, "Jobb")

        dock_layout = QVBoxLayout()
        dock_layout.addWidget(tabs)
        
        self._btn_save_settings = QPushButton("💾 Beállítások Mentése")
        self._btn_save_settings.setStyleSheet(
            "QPushButton { background-color: #3d3d4d; color: #cdd6f4; font-weight: bold; padding: 8px; border-radius: 4px; }"
            "QPushButton:hover { background-color: #555; }"
        )
        self._btn_save_settings.clicked.connect(self._save_gui_settings)
        dock_layout.addWidget(self._btn_save_settings)
        
        container = QWidget()
        container.setLayout(dock_layout)
        dock.setWidget(container)
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, dock)

    def _load_gui_settings(self) -> None:
        """Betölti az elmentett GUI kamera beállításokat."""
        path = "config/gui_settings.json"
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    gui_cfg = json.load(f)
                if "left" in gui_cfg:
                    if "left" not in self._config["camera"]:
                        self._config["camera"]["left"] = {}
                    self._config["camera"]["left"].update(gui_cfg["left"])
                if "right" in gui_cfg:
                    if "right" not in self._config["camera"]:
                        self._config["camera"]["right"] = {}
                    self._config["camera"]["right"].update(gui_cfg["right"])
                logger.info("Mentett GUI kamera beállítások betöltve.")
            except Exception as e:
                logger.error("GUI beállítások betöltése sikertelen: %s", e)

    @pyqtSlot()
    def _save_gui_settings(self) -> None:
        """Elmenti az aktuális kamera beállításokat."""
        gui_cfg = {
            "left": {
                "exposure_time_us": self._exp_l.value(),
                "gain_db": self._gain_l.value(),
                "auto_white_balance": self._awb_l.isChecked()
            },
            "right": {
                "exposure_time_us": self._exp_r.value(),
                "gain_db": self._gain_r.value(),
                "auto_white_balance": self._awb_r.isChecked()
            }
        }
        try:
            os.makedirs("config", exist_ok=True)
            with open("config/gui_settings.json", "w") as f:
                json.dump(gui_cfg, f, indent=4)
            logger.info("GUI kamera beállítások sikeresen mentve (config/gui_settings.json).")
            QMessageBox.information(self, "Mentés Sikeres", "A kamera beállítások sikeresen elmentve!\nKövetkező indításnál ezek az értékek töltődnek be.")
        except Exception as e:
            logger.error("GUI beállítások mentése sikertelen: %s", e)
            QMessageBox.critical(self, "Mentés Hiba", f"Nem sikerült elmenteni a beállításokat:\n{e}")

    def _build_stats_dock(self) -> None:
        """Statisztika lebegő panel létrehozása."""
        dock = QDockWidget("Rendszer Statisztikák", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )

        stats_widget = QWidget()
        layout = QFormLayout(stats_widget)
        layout.setSpacing(6)

        font_small = QFont("Consolas", 9)

        def make_stat_label() -> QLabel:
            lbl = QLabel("—")
            lbl.setFont(font_small)
            lbl.setStyleSheet("color: #50fa7b;")
            return lbl

        self._stat_cam_fps_l = make_stat_label()
        self._stat_cam_fps_r = make_stat_label()
        self._stat_temp_l = make_stat_label()
        self._stat_temp_r = make_stat_label()
        self._stat_det_fps = make_stat_label()
        self._stat_cpu = make_stat_label()
        self._stat_ram = make_stat_label()
        self._stat_calib = make_stat_label()

        layout.addRow("Bal kamera FPS:", self._stat_cam_fps_l)
        layout.addRow("Jobb kamera FPS:", self._stat_cam_fps_r)
        layout.addRow("Bal kamera hőm.:", self._stat_temp_l)
        layout.addRow("Jobb kamera hőm.:", self._stat_temp_r)
        layout.addRow("Detektálás FPS:", self._stat_det_fps)
        layout.addRow("CPU:", self._stat_cpu)
        layout.addRow("RAM:", self._stat_ram)
        layout.addRow("Kalibrálás:", self._stat_calib)

        dock.setWidget(stats_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _build_log_dock(self) -> None:
        """Napló megjelenítő panel."""
        dock = QDockWidget("Rendszernapló", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.BottomDockWidgetArea)

        self._log_panel = QPlainTextEdit()
        self._log_panel.setReadOnly(True)
        self._log_panel.setMaximumBlockCount(500)
        font_log = QFont("Consolas", 8)
        self._log_panel.setFont(font_log)
        self._log_panel.setStyleSheet(
            "background-color: #0d1117; color: #8be9fd; border: none;"
        )

        dock.setWidget(self._log_panel)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        dock.setMaximumHeight(180)

    def _build_status_bar(self) -> None:
        """Státuszsor a képernyő alján."""
        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self._status_bar.showMessage("Rendszer kész. Kattints a INDÍTÁS gombra!")

    # ------------------------------------------------------------------
    # Eseménykezelők
    # ------------------------------------------------------------------

    @pyqtSlot()
    def _on_start_stop(self) -> None:
        """Start/Stop gomb kezelője."""
        if self._is_running:
            self._stop_tracker()
        else:
            self._start_tracker()

    @pyqtSlot()
    def _on_clear_history(self) -> None:
        """Historika törlés gomb kezelője."""
        self._goal_view.clear_history()
        logger.info("Lövés historika törölve")

    @pyqtSlot(np.ndarray, np.ndarray, dict)
    def _on_frames_ready(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray,
        stats: dict
    ) -> None:
        """
        Háttérszálból érkező frame-pár és statisztikák feldolgozása.
        Ez a metódus a GUI szálban fut (Qt guarantee).
        """
        # Kamera képek megjelenítése
        self._display_frame(frame_left, self._cam_label_left)
        self._display_frame(frame_right, self._cam_label_right)

        # 3D pozíció feliratok frissítése
        if stats["pos_valid"]:
            self._lbl_x.setText(f"{stats['x_3d']:+.1f}")
            self._lbl_y.setText(f"{stats['y_3d']:+.1f}")
            self._lbl_z.setText(f"{stats['z_3d']:.1f}")
            speed_kmh = stats["speed_ms"] * 3.6
            self._lbl_speed.setText(f"{stats['speed_ms']:.1f} m/s  ({speed_kmh:.1f} km/h)")
        else:
            self._lbl_x.setText("—")
            self._lbl_y.setText("—")
            self._lbl_z.setText("—")
            self._lbl_speed.setText("—")

        # Trajektória előrejelzés frissítése
        impact: Optional[ImpactPrediction] = stats.get("impact")
        if impact and impact.valid:
            self._lbl_impact_x.setText(f"{impact.x_mm:+.1f}")
            self._lbl_impact_y.setText(f"{impact.y_mm:.1f}")
            self._lbl_impact_t.setText(f"{impact.time_to_impact_s:.3f}")
            self._lbl_impact_conf.setText(f"{impact.confidence * 100:.1f}%")
            self._goal_view.update_impact(
                x_mm=impact.x_mm,
                y_mm=impact.y_mm,
                confidence=impact.confidence,
                time_to_impact_s=impact.time_to_impact_s,
                in_goal=impact.in_goal,
            )
        else:
            self._goal_view.update_impact(None, None, 0.0, 0.0)

        # Toolbar FPS frissítése
        self._lbl_fps.setText(f"  FPS: {stats['pair_fps']:.0f}")
        self._lbl_det_fps.setText(f"  Det: {stats['det_fps']:.0f}")

        # Statisztika panel frissítése
        self._stat_cam_fps_l.setText(f"{stats['cam_fps_left']:.1f}")
        self._stat_cam_fps_r.setText(f"{stats['cam_fps_right']:.1f}")
        self._stat_temp_l.setText(f"{stats['temp_left']:.1f}°C")
        self._stat_temp_r.setText(f"{stats['temp_right']:.1f}°C")
        self._stat_det_fps.setText(f"{stats['det_fps']:.1f}")
        self._stat_calib.setText(
            "✓ Kalibrált" if stats.get("calibrated") else "⚠ Nem kalibrált"
        )

        # Státuszsor frissítése
        det_str = "✓ Mindkét kamerában" if stats["both_found"] else (
            "◑ Csak bal" if stats["left_found"] else (
            "◑ Csak jobb" if stats["right_found"] else "✗ Nincs detektálás"
        ))
        self._status_bar.showMessage(f"Labda: {det_str}")

    @pyqtSlot(str)
    def _on_worker_error(self, error_msg: str) -> None:
        """Háttérszál hibájának kezelője."""
        logger.error("Tracker hiba: %s", error_msg)
        self._stop_tracker()
        QMessageBox.critical(self, "Rendszerhiba", error_msg)

    @pyqtSlot()
    def _on_worker_stopped(self) -> None:
        """Háttérszál leállásának kezelője."""
        self._is_running = False
        self._btn_start.setText("▶  INDÍTÁS")
        self._btn_start.setStyleSheet(
            "QPushButton { background-color: #2ea043; color: white; font-weight: bold; "
            "border-radius: 6px; padding: 0 20px; font-size: 13px; }"
        )
        self._status_bar.showMessage("Rendszer megállt.")
        logger.info("Tracker leállt")

    @pyqtSlot(str)
    def _on_log_message(self, msg: str) -> None:
        """Napló üzenet megjelenítése a GUI log panelban."""
        self._log_panel.appendPlainText(msg)
        cursor = self._log_panel.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_panel.setTextCursor(cursor)

    @pyqtSlot()
    def _update_system_status(self) -> None:
        """Rendszer erőforrás monitoring (1 Hz)."""
        cpu = psutil.cpu_percent(interval=None)
        ram = psutil.virtual_memory().percent
        self._stat_cpu.setText(f"{cpu:.0f}%")
        self._stat_ram.setText(f"{ram:.0f}%")

    # ------------------------------------------------------------------
    # Tracker lifecycle
    # ------------------------------------------------------------------

    def _start_tracker(self) -> None:
        """Elindítja a háttérszálat."""
        logger.info("Tracker indítása...")

        self._worker = TrackerWorker(self._config, parent=self)
        self._worker.frames_ready.connect(self._on_frames_ready)
        self._worker.error_occurred.connect(self._on_worker_error)
        self._worker.tracker_stopped.connect(self._on_worker_stopped)
        self._worker.start()

        self._is_running = True
        self._btn_start.setText("⏹  MEGÁLLÍTÁS")
        self._btn_start.setStyleSheet(
            "QPushButton { background-color: #da3633; color: white; font-weight: bold; "
            "border-radius: 6px; padding: 0 20px; font-size: 13px; }"
            "QPushButton:hover { background-color: #f85149; }"
        )
        self._status_bar.showMessage("Rendszer fut…")

    def _stop_tracker(self) -> None:
        """Leállítja a háttérszálat."""
        if self._worker:
            self._worker.stop()
            self._worker.wait(5000)
            self._worker = None

    # ------------------------------------------------------------------
    # Segédek
    # ------------------------------------------------------------------

    def _display_frame(self, frame: np.ndarray, label: QLabel) -> None:
        """
        BGR NumPy képet jeleníti meg egy QLabel-ben.

        Args:
            frame: BGR NumPy tömb
            label: Célzott QLabel widget
        """
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(
            frame.data, w, h, bytes_per_line,
            QImage.Format.Format_BGR888
        )
        pixmap = QPixmap.fromImage(q_img).scaled(
            label.width(), label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        label.setPixmap(pixmap)

    def _setup_log_handler(self) -> None:
        """Regisztrálja a Qt log handlert a Python logging rendszerbe."""
        self._qt_log_handler = _QtLogHandler()
        formatter = logging.Formatter("%(asctime)s %(levelname)s  %(message)s", "%H:%M:%S")
        self._qt_log_handler.setFormatter(formatter)
        self._qt_log_handler.signals.log_msg.connect(self._on_log_message)
        logging.getLogger().addHandler(self._qt_log_handler)

    def _apply_dark_theme(self) -> None:
        """Sötét Dracula-ihletett téma alkalmazása."""
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #1e1e2e;
                color: #cdd6f4;
            }
            QGroupBox {
                border: 1px solid #313244;
                border-radius: 6px;
                margin-top: 8px;
                padding-top: 8px;
                font-size: 11px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 4px;
            }
            QDockWidget {
                color: #cdd6f4;
                font-weight: bold;
            }
            QDockWidget::title {
                background-color: #313244;
                padding: 4px;
            }
            QStatusBar {
                background-color: #181825;
                color: #a6e3a1;
                font-family: monospace;
            }
            QToolBar {
                background-color: #181825;
                border-bottom: 1px solid #313244;
                spacing: 4px;
                padding: 4px;
            }
            QFormLayout QLabel {
                color: #7f849c;
                font-size: 10px;
            }
        """)

    def closeEvent(self, event) -> None:
        """Ablak bezárásakor leállítjuk a háttérszálat."""
        logger.info("Ablak bezárása – rendszer leállítása...")
        self._stop_tracker()
        event.accept()
