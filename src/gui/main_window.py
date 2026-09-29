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
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    QGridLayout, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QMessageBox, QPlainTextEdit, QProgressBar, QPushButton,
    QSizePolicy, QSlider, QSpinBox, QDoubleSpinBox, QStackedWidget, QStatusBar, QTabWidget,
    QToolBar, QVBoxLayout, QWidget
)

from camera.camera_manager import CameraManager, StereoPair
from detection.ball_detector import BallDetection, BallDetector, StereoBallDetection
from detection.kalman_tracker import KalmanTracker2D
from detection.optical_flow_tracker import OpticalFlowTracker
from stereo.triangulator import StereoTriangulator
from stereo.mono_depth_estimator import MonoDepthEstimator
from prediction.trajectory_predictor import TrajectoryPredictor, ImpactPrediction
from gui.goal_view import GoalViewWidget
from gui.calibration_dialog import CalibrationDialog
from gui.actuator_widget import ActuatorControlWidget
from gui.analytics_view import AnalyticsDashboardWidget
from session.session_manager import SessionManager
from gui.theme import (
    LIGHT_DEIK_QSS, DARK_DEIK_QSS, get_status_pill_style, get_hw_pill_style, usage_level,
    get_app_icon, COLOR_DEIK_GREEN, COLOR_DEIK_GOLD
)
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import QMenu

try:
    # pyrefly: ignore [missing-import]
    import pynvml
    pynvml.nvmlInit()
    _nvml_available = True
    _nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception:
    _nvml_available = False
    _nvml_handle = None

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
            cropped = np.ascontiguousarray(frame[y1:y2, x1:x2])

        # Skálázás a label méretére
        ch = cropped.shape[2] if len(cropped.shape) == 3 else 1
        q_img = QImage(
            bytes(cropped.data), cropped.shape[1], cropped.shape[0],
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
# Kis késleltetésű kamera → inferencia → GUI pipeline
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StereoFrameSnapshot:
    """Immutable ownership snapshot shared by the preview and inference stages."""

    sequence: int
    left_image: np.ndarray
    right_image: np.ndarray
    left_frame_id: int
    right_frame_id: int
    timestamp: float
    sync_delta_ms: float = 0.0  # Sztereó szinkron jitter ms-ben (HW triggerrel <1 ms)


@dataclass
class TrackingState:
    """The newest completed inference result, independent from preview cadence."""

    source_sequence: int
    source_timestamp: float
    detection: StereoBallDetection
    pos_3d: Optional[np.ndarray]
    impact: Optional[ImpactPrediction]
    velocity_mm_s: Tuple[float, float, float]
    calibrated: bool
    completed_at: float
    
    # Képre vetített (2D) trajektória pontok a vizualizációhoz
    left_past_2d: Optional[np.ndarray] = None
    right_past_2d: Optional[np.ndarray] = None
    left_future_2d: Optional[np.ndarray] = None
    right_future_2d: Optional[np.ndarray] = None

    # Lövésállapot a kapu-vizualizáció és analitika kizárólagos vezérléséhez.
    shot_confirmed: bool = False
    shot_active: bool = False
    shot_finished: bool = False
    shot_armed: bool = False
    # A megerősítéskori Z-regresszió sebessége (mm/s) – ld. ShotStatus.confirmed_vz_trend_mm_s.
    shot_confirmed_vz_trend_mm_s: float = 0.0
    # Mindkét kamera ténylegesen (YOLO/színfolt) mérte a labdát, és a pozíció valódi
    # sztereó triangulációból jön (nem optikai flow / Kalman-coast / mono tartalék).
    stereo_measured: bool = False


@dataclass(frozen=True)
class ShotStatus:
    """Egy feldolgozott 3D mérés utáni lövésállapot."""

    confirmed: bool = False
    active: bool = False
    finished: bool = False
    armed: bool = False
    # True pontosan azon a frame-en, ahol a valódi mozgás (rúgás) elindult – ekkor
    # a TrajectoryPredictor historikáját is nullázni kell (lásd DetectionWorker.run()),
    # különben az állás közben felgyűlt zajos minták elrontják az impact-predikció
    # sebességbecslését (és ezzel az extrapolált X/Y-t).
    motion_started_event: bool = False
    # A megerősítéskori Z-regresszió sebessége (mm/s) – ez a robusztus, tesztelt
    # sebességjel (lásd kinematics_ok), szemben a TrajectoryPredictor saját, gyors
    # lövésnél gyakran alulbecsült/lassan konvergáló sebességbecslésével.
    confirmed_vz_trend_mm_s: float = 0.0


class ShotDetector:
    """
    Valódi lövést azonosít a 3D trajektória historikából.

    Egy lövés életciklusa:
        DISARMED → ARMED → TRACKING → CONFIRMED → COOLDOWN → DISARMED

    A megerősítéshez az objektumnak az indítózónában kell megjelennie, majd
    kellő távolságot kell konzisztensen, elég nagy sebességgel a kapu felé
    megtennie. A döntés valódi időbélyegekkel készül, nem detektálási FPS-becsléssel.
    """

    DISARMED = "DISARMED"
    ARMED = "ARMED"
    TRACKING = "TRACKING"
    CONFIRMED = "CONFIRMED"
    COOLDOWN = "COOLDOWN"

    def __init__(self, config: Optional[dict] = None) -> None:
        cfg = (config or {}).get("shot_detection", {})
        self._manual_arm_required = bool(cfg.get("manual_arm_required", True))
        self._launch_z_min_mm = float(cfg.get("launch_z_min_mm", 5000.0))
        self._launch_z_max_mm = float(cfg.get("launch_z_max_mm", 12000.0))
        self._min_travel_mm = float(cfg.get("min_travel_mm", 2000.0))
        self._min_samples = int(cfg.get("min_samples", 6))
        self._min_speed_mm_s = float(cfg.get("min_speed_mm_s", 4000.0))
        self._min_r2 = float(cfg.get("min_r2", 0.85))
        self._confirm_z_mm = float(cfg.get("confirm_z_mm", 1500.0))
        self._finish_z_mm = float(cfg.get("finish_z_mm", 700.0))
        self._max_missing_frames = int(cfg.get("max_missing_frames", 4))
        self._cooldown_s = float(cfg.get("cooldown_s", 2.5))
        # Amíg a labda az élesítés után lényegében mozdulatlanul áll az indítózónában
        # (a kezelő még célra áll/hátralép), egy-egy mintavétel közötti Z-elmozdulásnak
        # ennél gyorsabbnak kell lennie (mm/s), hogy "valódi lövés indulásaként" számítson.
        self._motion_start_speed_mm_s = float(cfg.get("motion_start_speed_mm_s", 600.0))
        # Egyetlen zajos képkocka (pl. egy pixelnyi sugárugrás) is adhat ennél nagyobb
        # pillanatnyi sebességet állás közben, ezért ennyi EGYMÁST KÖVETŐ mintának kell
        # a küszöb felett lennie, mielőtt tényleges lövés-indulásként fogadjuk el.
        self._motion_confirm_frames = int(cfg.get("motion_confirm_frames", 3))
        # Az `impact.in_goal` néhány mm-es zajingadozás miatt kereten belülről kívülre
        # villódzhat a kapu vonala közelében; ennyi másodpercig "emlékszünk" az utolsó
        # in_goal=True pillanatra, hogy egy köztes zajos frame ne buktassa el a megerősítést.
        self._in_goal_grace_s = float(cfg.get("in_goal_grace_s", 0.2))
        # A ballisztikus becsapódás-predikció (impact.in_goal) a labda Y-koordinátáján
        # múlik, ami gyors lövésnél a sztereó Y-hiba (epipoláris eltérés, mozgáselmosódás)
        # miatt megbízhatatlan – X és Z stabil, Y sokszor több méterrel elcsúszik.
        # Ezért alapból a lövést a MOZGÁSA (gyors, konzisztens, kapu felé tartó, a kaput
        # ténylegesen elérő) alapján erősítjük meg, NEM a törékeny Y-predikción. Ha true,
        # a régi, szigorú in_goal feltételt is megköveteljük.
        self._require_impact_in_goal = bool(cfg.get("require_impact_in_goal", False))
        self._samples: list[tuple[float, float]] = []
        self._motion_started = False
        self._motion_streak = 0
        self._last_in_goal_true_time = float("-inf")
        self._state = self.ARMED if not self._manual_arm_required else self.DISARMED
        self._last_shot_time = float("-inf")
        self._missing_frames = 0
        logger.debug("ShotDetector inicializálva: kézi élesítés=%s", self._manual_arm_required)

    @property
    def is_armed(self) -> bool:
        return self._state in (self.ARMED, self.TRACKING)

    @property
    def is_active(self) -> bool:
        return self._state == self.CONFIRMED

    def arm_next_shot(self) -> None:
        """Élesíti a következő, indítózónából érkező lövést."""
        self._state = self.ARMED
        self._samples.clear()
        self._motion_started = False
        self._motion_streak = 0
        self._missing_frames = 0
        self._last_in_goal_true_time = float("-inf")
        logger.info("Lövésdetektor ÉLESÍTVE: indítózónából érkező lövésre vár.")

    def disarm(self) -> None:
        """Megszakítja a függőben lévő lövésfigyelést és törli a mintákat."""
        self._state = self.DISARMED
        self._samples.clear()
        self._motion_started = False
        self._motion_streak = 0
        self._missing_frames = 0
        self._last_in_goal_true_time = float("-inf")
        logger.info("Lövésdetektor hatástalanítva.")

    def update(
        self,
        z_mm: Optional[float],
        timestamp_s: float,
        impact: Optional[ImpactPrediction],
        pos_3d_valid: bool,
    ) -> ShotStatus:
        """
        Megvizsgálja az aktuális állapotot és eldönti, hogy éppen lövés történik-e.

        A döntés a Z-historikán alapul: lineáris regresszióval ellenőrzi,
        hogy a labda konzisztensen közeledik-e a kapu felé.

        Returns:
            ShotStatus: a megerősítés, aktivitás, lezárás és élesítettség állapota.
        """
        if self._state == self.COOLDOWN:
            if timestamp_s - self._last_shot_time >= self._cooldown_s:
                self._state = self.ARMED if not self._manual_arm_required else self.DISARMED
            return self._status()

        if self._state == self.DISARMED:
            return self._status()

        if self._state == self.CONFIRMED:
            if pos_3d_valid and z_mm is not None:
                self._missing_frames = 0
                if z_mm > self._finish_z_mm:
                    return self._status()
            else:
                self._missing_frames += 1
                if self._missing_frames <= self._max_missing_frames:
                    return self._status()
            self._state = self.COOLDOWN
            self._last_shot_time = timestamp_s
            return self._status(finished=True)

        if not pos_3d_valid or z_mm is None:
            return self._status()

        if self._state == self.ARMED:
            if not (self._launch_z_min_mm <= z_mm <= self._launch_z_max_mm):
                logger.debug(
                    "Lövésdetektor: Z=%.0f mm kívül esik az indítózónán [%.0f, %.0f] mm",
                    z_mm, self._launch_z_min_mm, self._launch_z_max_mm,
                )
                return self._status()
            self._state = self.TRACKING
            self._samples = [(timestamp_s, z_mm)]
            self._motion_started = False
            self._motion_streak = 0
            self._last_in_goal_true_time = float("-inf")
            logger.debug("Lövésdetektor: TRACKING elindult Z=%.0f mm-nél", z_mm)
            return self._status()

        # TRACKING: amíg a labda lényegében mozdulatlanul áll az indítózónában
        # (élesítés után, a rúgás előtt), az egyetlen bázispontot frissítjük ahelyett,
        # hogy a mozdulatlan mintákat felhalmoznánk. Enélkül ezek a régi, közel nulla
        # sebességű minták bekerülnének a 30-elemes csúszóablakba és összekeverednének
        # a tényleges lövés adataival, jelentősen belassítva/megakadályozva az
        # R²/sebesség-illesztés konvergenciáját, mire a labda eléri a kaput.
        if not self._motion_started:
            prev_t, prev_z = self._samples[-1] if self._samples else (timestamp_s, z_mm)
            dt = timestamp_s - prev_t
            inst_vz = (z_mm - prev_z) / dt if dt > 1e-6 else 0.0
            if inst_vz > -self._motion_start_speed_mm_s:
                self._motion_streak = 0
                self._samples = [(timestamp_s, z_mm)]
                return self._status()
            # A küszöb feletti pillanatnyi sebesség önmagában lehet egyetlen zajos
            # képkocka is (pl. sugárbecslési ugrás állás közben) – csak akkor fogadjuk
            # el valódi lövés-indulásnak, ha ez több egymást követő mintán is fennáll.
            self._samples.append((timestamp_s, z_mm))
            self._motion_streak += 1
            if self._motion_streak < self._motion_confirm_frames:
                return self._status()
            self._motion_started = True
            logger.debug(
                "Lövésdetektor: valódi elmozdulás észlelve (v=%.0f mm/s, Z=%.0f mm), "
                "regresszió indul",
                inst_vz, z_mm,
            )
            return self._status(motion_started_event=True)

        # a ténylegesen mért, időbélyegzett Z-pontokból döntünk.
        self._samples.append((timestamp_s, z_mm))
        self._samples = self._samples[-30:]
        if len(self._samples) < self._min_samples:
            return self._status()

        times = np.array([sample[0] for sample in self._samples], dtype=np.float64)
        z_values = np.array([sample[1] for sample in self._samples], dtype=np.float64)
        elapsed = times - times.mean()
        z_centered = z_values - z_values.mean()
        ss_tt = float(np.dot(elapsed, elapsed))
        if ss_tt < 1e-9:
            return self._status()
        vz_trend = float(np.dot(elapsed, z_centered) / ss_tt)
        z_pred = vz_trend * elapsed + z_values.mean()
        ss_tot = float(np.dot(z_centered, z_centered))
        r2 = 1.0 - float(np.sum((z_values - z_pred) ** 2)) / ss_tot if ss_tot > 1e-9 else 0.0
        traveled_mm = self._samples[0][1] - z_mm

        impact_in_goal_now = bool(impact is not None and impact.valid and impact.in_goal)
        if impact_in_goal_now:
            self._last_in_goal_true_time = timestamp_s
        # Rövid türelmi ablak: a becsapódás-predikció a kapuvonal közelében néhány
        # mm-es zajra is be-kilebeghet a kereten, ezért egy köztes False frame miatt
        # nem utasítjuk el a lövést, ha nagyon nemrég még in_goal=True volt.
        impact_in_goal_recent = impact_in_goal_now or (
            timestamp_s - self._last_in_goal_true_time <= self._in_goal_grace_s
        )

        # --- Megerősítés ---
        # ROBUST KINEMATIKA (X/Z-alapú, gyors lövésnél is megbízható): a labda a kapu
        # felé, elég gyorsan, konzisztensen (magas R²) mozgott, elég nagy utat tett meg,
        # és ténylegesen elérte a kapu közelségét (confirm_z). Ezek a jelek a logokban
        # minden valódi lövésnél teljesülnek.
        kinematics_ok = (
            z_mm <= self._confirm_z_mm and
            vz_trend <= -self._min_speed_mm_s and
            r2 >= self._min_r2 and
            traveled_mm >= self._min_travel_mm
        )
        # A ballisztikus in_goal-predikció a labda Y-koordinátáján múlik, ami gyors
        # lövésnél a sztereó Y-hiba miatt gyakran több méterrel elcsúszik (a logokban
        # X és Z helyes, Y viszont -2000..-45000 mm közé ugrik). Ezért alapból NEM
        # kötjük ehhez a megerősítést; csak ha require_impact_in_goal explicit be van
        # kapcsolva a configban.
        impact_gate_ok = (not self._require_impact_in_goal) or (
            impact is not None and impact.valid and impact_in_goal_recent
        )

        if not (kinematics_ok and impact_gate_ok):
            logger.debug(
                "Lövésdetektor: TRACKING elutasítva Z=%.0f mm, vz=%.0f mm/s (min=%.0f), "
                "R²=%.2f (min=%.2f), dz=%.0f mm (min=%.0f), impact=%s, "
                "impact_xy=(X=%s, Y=%s), in_goal=%s, in_goal_kell=%s, confirm_z=%.0f mm",
                z_mm, vz_trend, self._min_speed_mm_s, r2, self._min_r2,
                traveled_mm, self._min_travel_mm,
                impact.valid if impact is not None else None,
                f"{impact.x_mm:+.0f}" if impact is not None else None,
                f"{impact.y_mm:.0f}" if impact is not None else None,
                impact_in_goal_recent,
                self._require_impact_in_goal,
                self._confirm_z_mm,
            )
            return self._status()

        self._state = self.CONFIRMED
        self._missing_frames = 0
        self._last_shot_time = timestamp_s
        if impact is not None and impact.valid:
            logger.info(
                "LÖVÉS MEGERŐSÍTVE ✓: Z=%0.f→%0.f mm, dz=%0.f mm, vz=%0.f mm/s, R²=%.2f, "
                "impact=(X=%+.0f mm, Y=%.0f mm, t=%.3f s)",
                self._samples[0][1], z_mm, traveled_mm, vz_trend, r2,
                impact.x_mm, impact.y_mm, impact.time_to_impact_s,
            )
        else:
            logger.info(
                "LÖVÉS MEGERŐSÍTVE ✓ (kinematika): Z=%0.f→%0.f mm, dz=%0.f mm, "
                "vz=%0.f mm/s, R²=%.2f (nincs érvényes becsapódás-predikció)",
                self._samples[0][1], z_mm, traveled_mm, vz_trend, r2,
            )
        return self._status(confirmed=True, confirmed_vz_trend_mm_s=vz_trend)

    def _status(
        self,
        confirmed: bool = False,
        finished: bool = False,
        motion_started_event: bool = False,
        confirmed_vz_trend_mm_s: float = 0.0,
    ) -> ShotStatus:
        return ShotStatus(
            confirmed=confirmed,
            active=self._state == self.CONFIRMED,
            finished=finished,
            armed=self.is_armed,
            motion_started_event=motion_started_event,
            confirmed_vz_trend_mm_s=confirmed_vz_trend_mm_s,
        )

    def reset(self) -> None:
        """Visszaállítja az állapotot."""
        self._samples.clear()
        self._motion_started = False
        self._motion_streak = 0
        self._missing_frames = 0
        self._last_in_goal_true_time = float("-inf")
        self._state = self.ARMED if not self._manual_arm_required else self.DISARMED
        self._last_shot_time = float("-inf")


class LatestStereoFrame:
    """One-slot exchange: consumers always receive the newest available pair."""


    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._latest: Optional[StereoFrameSnapshot] = None
        self._stopped = False

    def publish(self, snapshot: StereoFrameSnapshot) -> None:
        with self._condition:
            self._latest = snapshot
            self._condition.notify_all()

    def wait_for_newer(
        self, last_sequence: int, timeout: float = 0.1
    ) -> Optional[StereoFrameSnapshot]:
        with self._condition:
            newer_frame_available = self._condition.wait_for(
                lambda: self._stopped or (
                    self._latest is not None and self._latest.sequence > last_sequence
                ),
                timeout=timeout,
            )
            if self._stopped or not newer_frame_available:
                return None
            return self._latest

    def stop(self) -> None:
        with self._condition:
            self._stopped = True
            self._condition.notify_all()


class LatestTrackingState:
    """Thread-safe holder for the last completed YOLO/Kalman result."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state: Optional[TrackingState] = None

    def publish(self, state: TrackingState) -> None:
        with self._lock:
            self._state = state

    def get(self) -> Optional[TrackingState]:
        with self._lock:
            return self._state


class DetectionWorker(threading.Thread):
    """Runs GPU inference independently and deliberately drops stale camera pairs."""

    def __init__(
        self,
        config: dict,
        frame_exchange: LatestStereoFrame,
        result_exchange: LatestTrackingState,
    ) -> None:
        super().__init__(name="BallDetection", daemon=True)
        self._config = config
        self._frame_exchange = frame_exchange
        self._result_exchange = result_exchange
        self._running = threading.Event()
        self._running.set()
        self._detector_lock = threading.Lock()
        self._kalman_lock = threading.Lock()
        self._detector: Optional[BallDetector] = None
        roi_cfg = config.get("detection", {}).get("roi", {})
        self._roi_settings = {
            True: self._make_roi(roi_cfg),
            False: self._make_roi(roi_cfg),
        }
        self._error_lock = threading.Lock()
        self._error: Optional[str] = None
        self._shot_control_lock = threading.Lock()
        self._arm_shot_requested = False
        self._disarm_shot_requested = False

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

    @property
    def error(self) -> Optional[str]:
        with self._error_lock:
            return self._error

    def stop(self) -> None:
        self._running.clear()
        self._frame_exchange.stop()

    def arm_next_shot(self) -> None:
        """Szálbiztos kérés a következő távoli lövés élesítésére."""
        with self._shot_control_lock:
            self._arm_shot_requested = True
            self._disarm_shot_requested = False

    def disarm_shot(self) -> None:
        """Szálbiztos kérés az aktuális lövésfigyelés megszakítására."""
        with self._shot_control_lock:
            self._disarm_shot_requested = True
            self._arm_shot_requested = False

    def set_roi(
        self,
        is_left: bool,
        enabled: bool,
        x_min_rel: float,
        x_max_rel: float,
        y_min_rel: float,
        y_max_rel: float,
    ) -> None:
        with self._detector_lock:
            self._roi_settings[is_left] = {
                "enabled": bool(enabled),
                "x_min_rel": float(x_min_rel),
                "x_max_rel": float(x_max_rel),
                "y_min_rel": float(y_min_rel),
                "y_max_rel": float(y_max_rel),
            }
            if self._detector:
                self._detector.set_roi(
                    is_left, enabled, x_min_rel, x_max_rel, y_min_rel, y_max_rel
                )

    def run(self) -> None:
        last_sequence = 0
        try:
            detector = BallDetector(self._config["detection"], full_config=self._config)
            triangulator = StereoTriangulator(self._config)
            predictor = TrajectoryPredictor(self._config)
            cal_file = self._config.get("stereo", {}).get(
                "calibration_file", "data/calibration/stereo_calibration.npz"
            )
            triangulator.load_calibration(cal_file)

            # --- Optikai flow tracker-ek (per-kamera) ---
            of_cfg = self._config.get("optical_flow", {})
            of_left = OpticalFlowTracker(of_cfg)
            of_right = OpticalFlowTracker(of_cfg)

            # --- Mono mélység becslő ---
            mono_est = MonoDepthEstimator(self._config)
            # A nyers (torzított) pixelsugárhoz az eredeti K1 fókusz tartozik, nem a rektifikált P1-é.
            if triangulator.is_calibrated:
                mono_est.update_focal_length(triangulator.left_focal_length_px)

            # --- Lövés detektor (valódi lövés elkülönítése a normális mozgástól) ---
            shot_detector = ShotDetector(self._config)

            with self._detector_lock:
                self._detector = detector
                for is_left, roi in self._roi_settings.items():
                    detector.set_roi(is_left, **roi)

            logger.info("Detektáló szál elindult (latest-frame üzemmód + OF + MonoZ)")
            while self._running.is_set():
                with self._shot_control_lock:
                    if self._disarm_shot_requested:
                        shot_detector.disarm()
                        predictor.reset()
                        self._disarm_shot_requested = False
                    elif self._arm_shot_requested:
                        shot_detector.arm_next_shot()
                        predictor.reset()
                        self._arm_shot_requested = False

                snapshot = self._frame_exchange.wait_for_newer(last_sequence)
                if snapshot is None:
                    continue
                last_sequence = snapshot.sequence

                # Szürke képek az optikai flow-hoz
                gray_left = cv2.cvtColor(snapshot.left_image, cv2.COLOR_BGR2GRAY)
                gray_right = cv2.cvtColor(snapshot.right_image, cv2.COLOR_BGR2GRAY)

                # The lock also makes live ROI updates safe while Ultralytics is running.
                with self._detector_lock:
                    detection = detector.detect(snapshot.left_image, snapshot.right_image)
                # Csak a YOLO/színfolt által ténylegesen MÉRT sugár használható mélységbecsléshez;
                # az optikai flow (15 px) és a Kalman-coast (25 px) sugara kitalált érték.
                left_measured = detection.left.found
                right_measured = detection.right.found

                # --- Optikai flow fallback: ha a YOLO nem talált labbát ---
                if detection.left.found:
                    of_left.update_from_yolo(
                        gray_left,
                        detection.left.x, detection.left.y, detection.left.radius
                    )
                else:
                    of_result_left = of_left.track(gray_left)
                    if of_result_left is not None:
                        ox, oy = of_result_left
                        r = detection.left.radius if detection.left.radius > 0 else 15.0
                        detection.left = BallDetection(
                            found=True, x=ox, y=oy, radius=r,
                            confidence=0.55,  # optical flow konfidencia
                            timestamp=snapshot.timestamp,
                        )

                if detection.right.found:
                    of_right.update_from_yolo(
                        gray_right,
                        detection.right.x, detection.right.y, detection.right.radius
                    )
                else:
                    of_result_right = of_right.track(gray_right)
                    if of_result_right is not None:
                        ox, oy = of_result_right
                        r = detection.right.radius if detection.right.radius > 0 else 15.0
                        detection.right = BallDetection(
                            found=True, x=ox, y=oy, radius=r,
                            confidence=0.55,
                            timestamp=snapshot.timestamp,
                        )

                # Optikai folyam visszaállítása ha mindkét kamera elveszíti a labdát
                if not detection.left.found and not detection.right.found:
                    of_left.reset()
                    of_right.reset()

                with self._kalman_lock:
                    left_det, left_valid, left_x, left_y = self._filter_detection(
                        detection.left, self._kalman_left, snapshot.timestamp
                    )
                    right_det, right_valid, right_x, right_y = self._filter_detection(
                        detection.right, self._kalman_right, snapshot.timestamp
                    )
                    # Kalibráció nélkül is látható legyen a mozgás iránya. Ez
                    # csak 2D, egyenes vonalú extrapoláció; kalibráció esetén a
                    # később számolt ballisztikus 3D pálya felülírja.
                    left_future = self._make_2d_prediction(
                        self._kalman_left, snapshot.timestamp
                    )
                    right_future = self._make_2d_prediction(
                        self._kalman_right, snapshot.timestamp
                    )
                detection.both_found = left_det.found and right_det.found

                pos_3d: Optional[np.ndarray] = None
                # True sztereó trianguláció ténye (a labdaméret-alapú mono tartalék NEM az).
                # A ShotDetector Z-trendje ehhez a zászlóhoz kötött, mert a mono Z
                # (lásd lejjebb) néhány pixelnyi sugárzaj mellett is több ezer mm-t
                # ugorhat, ami tönkreteszi az R²/sebesség-illesztést.
                stereo_pos_valid = False
                if left_valid and right_valid:
                    pos_3d = triangulator.triangulate(
                        left_point=(left_x, left_y), right_point=(right_x, right_y)
                    )
                    stereo_pos_valid = pos_3d is not None

                # --- Mono mélység validáció és fuzízió ---
                if pos_3d is not None and detection.left.radius > 0:
                    z_fused, z_valid, z_warn = mono_est.validate_and_fuse_stereo_z(
                        float(pos_3d[2]), detection.left.radius
                    )
                    if z_warn:
                        logger.warning("MonoZ: %s", z_warn)
                    pos_3d[2] = z_fused
                elif (pos_3d is None and left_valid and left_measured
                      and detection.left.radius > 0):
                    # Szoftveres szinkron jitter vagy egykamerás takarás esetén: Mono mélység tartalék
                    z_fallback = mono_est.fallback_z(detection.left.radius)
                    if z_fallback is not None:
                        # Pixel visszavetítés az EREDETI bal kamera keretbe (K1 + torzítás-korrekció),
                        # majd a sztereóval AZONOS világ-transzformáció (kalibrált R_wc vagy tartalék).
                        x_n, y_n = triangulator.left_pixel_to_normalized(left_x, left_y)
                        pos_3d = triangulator.leftcam_original_to_world(
                            (x_n * z_fallback, y_n * z_fallback, z_fallback)
                        )
                        logger.debug("MonoZ 3D tartalék aktiválva: Z=%.0f mm", pos_3d[2])

                if pos_3d is not None:
                    # --- Minőségi kapu a trajektória előrejelzőhöz ---
                    # Elég, ha BÁRMELYIK kamera megbízható, elég nagy labdát talált.
                    # Mérés alapján lövésenként hol a bal, hol a jobb kamera esik
                    # 0.30 konfidencia alá, ezért a "mindkettő kell" szabály a valódi
                    # lövések nagy részét is kizárta (0% egyidejű egyezés több tesztnél).
                    # A radius küszöb (8px) továbbra is kiszűri a kis zajpontokat
                    # (cipő, ruha, pixel-zaj).
                    #
                    # Konfidencia küszöb: 0.30 (YOLO 30%+ vagy fallback blob 0.60)
                    # Sugár küszöb: 8px – ennél kisebb pont nem lehet egy valódi labda
                    QUALITY_MIN_CONF = 0.30
                    QUALITY_MIN_RADIUS = 8.0

                    left_quality_ok = (
                        left_det.found
                        and left_det.confidence >= QUALITY_MIN_CONF
                        and left_det.radius >= QUALITY_MIN_RADIUS
                    )
                    right_quality_ok = (
                        right_det.found
                        and right_det.confidence >= QUALITY_MIN_CONF
                        and right_det.radius >= QUALITY_MIN_RADIUS
                    )

                    if left_quality_ok or right_quality_ok:
                        predictor.add_measurement(
                            x_mm=float(pos_3d[0]),
                            y_mm=float(pos_3d[1]),
                            z_mm=float(pos_3d[2]),
                        )
                    else:
                        # Alacsony minőségű detektálás: nem adjuk hozzá a historikához,
                        # de a prediktort se nullázzuk – várjuk vissza a labdát
                        logger.debug(
                            "Quality gate: mérés KIZÁRVA (L: conf=%.2f r=%.1f, R: conf=%.2f r=%.1f)",
                            left_det.confidence, left_det.radius,
                            right_det.confidence, right_det.radius,
                        )
                    impact = predictor.get_impact_prediction()
                else:
                    # Nincs 3D pozíció: ha a Kalman tracker sem inicializált (nincs labda),
                    # töröljük a historikát – így a régi zaj-mérések nem terhelik a prediktort
                    if not self._kalman_left.is_initialized and not self._kalman_right.is_initialized:
                        predictor.reset()
                    impact = predictor.get_impact_prediction()

                # Trajektória pontok visszavetítése 2D-be (ha van kalibráció)
                left_past, right_past = None, None
                
                if triangulator.is_calibrated:
                    history_3d = predictor.get_trajectory_history_mm()
                    if len(history_3d) >= 2:
                        history_arr = np.array(history_3d)
                        left_past = triangulator.project_to_2d(history_arr, is_left=True)
                        right_past = triangulator.project_to_2d(history_arr, is_left=False)
                        
                    # Folyamatos 3D jövőbeli pálya kiszámítása (bármilyen mozgásirány esetén)
                    future_path_3d = predictor.get_future_path_3d(num_points=20, max_time_s=1.5)
                    if future_path_3d is None and impact is not None and impact.valid:
                        future_path_3d = impact.path_3d

                    if future_path_3d is not None:
                        left_future = triangulator.project_to_2d(future_path_3d, is_left=True)
                        right_future = triangulator.project_to_2d(future_path_3d, is_left=False)

                # --- Lövés detektálás ---
                # Csak valódi lövésnél (gyors, kapu felé tartó labda) hozzuk létre a lövés eseményt.
                # A trajektória-előrejelzőhöz (fentebb) a mono-fallback Z-t is elfogadjuk, hogy a
                # kirajzolt pálya ne szakadjon meg. A ShotDetector Z-trendjéhez viszont KIZÁRÓLAG
                # a valódi sztereó trianguláció Z-jét adjuk tovább (stereo_pos_valid) – a mono
                # tartalék (labda-sugár alapú) Z pár pixelnyi sugárzajra is ezer mm-es ugrásokat
                # produkál (lásd MonoDepthEstimator.fallback_z), ami megöli az R²/sebesség
                # illesztést és soha nem engedi megerősíteni a lövést. Egy ilyen frame egyszerűen
                # kimarad a mintákból (mint egy hiányzó detektálás), nem szakítja meg a TRACKING-et.
                left_ok = left_det.found and left_det.confidence >= 0.30 and left_det.radius >= 8.0
                right_ok = right_det.found and right_det.confidence >= 0.30 and right_det.radius >= 8.0
                quality_ok = pos_3d is not None and stereo_pos_valid and (left_ok or right_ok)
                shot_status = shot_detector.update(
                    z_mm=float(pos_3d[2]) if pos_3d is not None else None,
                    timestamp_s=snapshot.timestamp,
                    impact=impact,
                    pos_3d_valid=quality_ok,
                )
                if shot_status.motion_started_event:
                    # A rúgás előtti állás közben felgyűlt zajos minták kikerülnek a
                    # predikcióból – a sebességbecslés (és vele az impact X/Y) a
                    # valódi lövés első mintájától indul újra, tisztán.
                    predictor.reset()
                    logger.debug("TrajectoryPredictor újraindítva a lövés-indulás pillanatában")

                self._result_exchange.publish(
                    TrackingState(
                        source_sequence=snapshot.sequence,
                        source_timestamp=snapshot.timestamp,
                        detection=detection,
                        pos_3d=pos_3d,
                        impact=impact,
                        velocity_mm_s=predictor.estimated_velocity_mm_s,
                        calibrated=triangulator.is_calibrated,
                        completed_at=time.perf_counter(),
                        left_past_2d=left_past,
                        right_past_2d=right_past,
                        left_future_2d=left_future,
                        right_future_2d=right_future,
                        shot_confirmed=shot_status.confirmed,
                        shot_active=shot_status.active,
                        shot_finished=shot_status.finished,
                        shot_armed=shot_status.armed,
                        shot_confirmed_vz_trend_mm_s=shot_status.confirmed_vz_trend_mm_s,
                        stereo_measured=(
                            stereo_pos_valid and left_measured and right_measured
                        ),
                    )
                )
        except Exception as exc:
            logger.exception("Detektáló szál hiba: %s", exc)
            with self._error_lock:
                self._error = str(exc)
        finally:
            with self._detector_lock:
                self._detector = None
            logger.info("Detektáló szál leállt")

    def _filter_detection(
        self, detection: BallDetection, tracker: KalmanTracker2D, timestamp: float
    ) -> Tuple[BallDetection, bool, float, float]:
        if detection.found:
            x, y = tracker.update(detection.x, detection.y, timestamp)
            detection.x, detection.y = x, y
            return detection, True, x, y

        if tracker.is_initialized:
            x, y = tracker.predict(timestamp)
            if x > 0 and y > 0:
                radius = detection.radius if detection.radius > 0 else 25.0
                detection.found = True
                detection.x, detection.y = x, y
                detection.radius = radius
                detection.bbox = (x - radius, y - radius, x + radius, y + radius)
                return detection, True, x, y

        return detection, False, 0.0, 0.0

    @staticmethod
    def _make_2d_prediction(
        tracker: KalmanTracker2D,
        timestamp: float,
        horizon_s: float = 0.50,
        point_count: int = 12,
    ) -> Optional[np.ndarray]:
        """Kalibráció előtti, képsíkbeli irányjelző pálya."""
        vx, vy = tracker.velocity_pixels_s
        if not tracker.is_initialized or np.hypot(vx, vy) < 25.0:
            return None

        times = np.linspace(0.0, horizon_s, point_count)
        return np.array(
            [tracker.project(timestamp + float(dt), horizon_s) for dt in times],
            dtype=np.float32,
        )

    def project_detection_for_display(
        self,
        detection: BallDetection,
        is_left: bool,
        display_timestamp: float,
        max_horizon_s: float,
    ) -> BallDetection:
        """A korábbi inferenciaeredményt az aktuális preview frame-re vetíti."""
        if not detection.found:
            return detection

        tracker = self._kalman_left if is_left else self._kalman_right
        with self._kalman_lock:
            x, y = tracker.project(display_timestamp, max_horizon_s)
        if x <= 0.0 or y <= 0.0:
            return detection

        radius = detection.radius
        return BallDetection(
            found=True,
            x=x,
            y=y,
            radius=radius,
            confidence=detection.confidence,
            track_id=detection.track_id,
            bbox=(x - radius, y - radius, x + radius, y + radius),
            timestamp=detection.timestamp,
        )

    @staticmethod
    def _make_roi(roi_cfg: dict) -> dict:
        return {
            "enabled": bool(roi_cfg.get("enabled", False)),
            "x_min_rel": float(roi_cfg.get("x_min_rel", 0.0)),
            "x_max_rel": float(roi_cfg.get("x_max_rel", 1.0)),
            "y_min_rel": float(roi_cfg.get("y_min_rel", 0.0)),
            "y_max_rel": float(roi_cfg.get("y_max_rel", 1.0)),
        }


class TrackerWorker(QThread):
    frames_ready = pyqtSignal(np.ndarray, np.ndarray, dict)
    error_occurred = pyqtSignal(str)
    tracker_stopped = pyqtSignal()

    def __init__(self, config: dict, parent: Optional[QObject] = None):
        super().__init__(parent)
        self._config = config
        self._running = False

        self._cam_manager: Optional[CameraManager] = None
        self._frame_exchange = LatestStereoFrame()
        self._result_exchange = LatestTrackingState()
        self._detection_worker: Optional[DetectionWorker] = None
        self._snapshot_sequence = 0
        self._last_frame_ids: Tuple[int, int] = (-1, -1)

        roi_cfg = config.get("detection", {}).get("roi", {})
        self._roi_lock = threading.Lock()
        self._display_roi = {
            "left": self._make_roi(roi_cfg),
            "right": self._make_roi(roi_cfg),
        }

        self._gui_fps_limit = float(config.get("gui", {}).get("gui_fps_limit", 60))
        self._gui_interval = 1.0 / self._gui_fps_limit
        self._last_gui_emit = 0.0
        kalman_cfg = config.get("detection", {}).get("kalman", {})
        self._display_prediction_horizon_s = float(
            kalman_cfg.get("display_prediction_horizon_s", 0.15)
        )

    def stop(self) -> None:
        self._running = False
        self._frame_exchange.stop()
        if self._detection_worker:
            self._detection_worker.stop()

    def arm_next_shot(self) -> None:
        if self._detection_worker:
            self._detection_worker.arm_next_shot()

    def disarm_shot(self) -> None:
        if self._detection_worker:
            self._detection_worker.disarm_shot()

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
        side = "left" if is_left else "right"
        roi = {
            "enabled": bool(enabled),
            "x_min_rel": max(0.0, min(1.0, float(x_min_rel))),
            "x_max_rel": max(0.0, min(1.0, float(x_max_rel))),
            "y_min_rel": max(0.0, min(1.0, float(y_min_rel))),
            "y_max_rel": max(0.0, min(1.0, float(y_max_rel))),
        }
        with self._roi_lock:
            self._display_roi[side] = roi
        if self._detection_worker:
            self._detection_worker.set_roi(
                is_left, enabled, x_min_rel, x_max_rel, y_min_rel, y_max_rel
            )

    @pyqtSlot(bool, bool, bool, int)
    def set_camera_transform(self, is_left: bool, flip_h: bool, flip_v: bool, rotation: int) -> None:
        if self._cam_manager:
            self._cam_manager.set_camera_transform(is_left, flip_h, flip_v, rotation)

    def run(self) -> None:
        logger.info("TrackerWorker szál elindult")
        self._running = True

        try:
            self._cam_manager = CameraManager(self._config)
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

        self._detection_worker = DetectionWorker(
            self._config, self._frame_exchange, self._result_exchange
        )
        self._detection_worker.start()

        logger.info("TrackerWorker főciklus indítása...")

        try:
            self._main_loop()
        except Exception as exc:
            logger.exception("Váratlan hiba: %s", exc)
            self.error_occurred.emit(str(exc))
        finally:
            if self._detection_worker:
                self._detection_worker.stop()
                self._detection_worker.join(timeout=5.0)
                self._detection_worker = None
            if self._cam_manager:
                self._cam_manager.close()
            logger.info("TrackerWorker szál leállt")
            self.tracker_stopped.emit()

    def _main_loop(self) -> None:
        overlay_cfg = self._config.get("gui", {}).get("overlay", {})

        while self._running:
            if self._detection_worker and self._detection_worker.error:
                raise RuntimeError(f"Detektáló szál hiba: {self._detection_worker.error}")

            pair: StereoPair = self._cam_manager.read_stereo_pair()
            if not pair.success:
                time.sleep(0.005)
                continue

            frame_ids = (pair.left.frame_id, pair.right.frame_id)
            if frame_ids == self._last_frame_ids:
                time.sleep(0.001)
                continue
            self._last_frame_ids = frame_ids
            self._snapshot_sequence += 1

            # Ximea SDK buffers can be recycled by acquisition threads. One owned copy is
            # therefore made at the pipeline boundary and shared read-only afterwards.
            snapshot = StereoFrameSnapshot(
                sequence=self._snapshot_sequence,
                left_image=pair.left.image.copy(),
                right_image=pair.right.image.copy(),
                left_frame_id=pair.left.frame_id,
                right_frame_id=pair.right.frame_id,
                timestamp=pair.timestamp,
                sync_delta_ms=pair.sync_delta_ms,
            )
            self._frame_exchange.publish(snapshot)

            now = time.perf_counter()
            if now - self._last_gui_emit >= self._gui_interval:
                self._last_gui_emit = now
                state = self._result_exchange.get()
                frame_left = snapshot.left_image.copy()
                frame_right = snapshot.right_image.copy()
                self._draw_roi(frame_left, is_left=True)
                self._draw_roi(frame_right, is_left=False)

                # A state a korábbi forrásframe-hez tartozik. A Kalman állapot
                # az aktuális preview idejére vetítve megszünteti a látható lagot.
                left_det = (
                    self._detection_worker.project_detection_for_display(
                        state.detection.left,
                        True,
                        snapshot.timestamp,
                        self._display_prediction_horizon_s,
                    )
                    if state and self._detection_worker else BallDetection()
                )
                right_det = (
                    self._detection_worker.project_detection_for_display(
                        state.detection.right,
                        False,
                        snapshot.timestamp,
                        self._display_prediction_horizon_s,
                    )
                    if state and self._detection_worker else BallDetection()
                )
                if overlay_cfg.get("show_detection_box", True):
                    self._draw_detection(frame_left, left_det)
                    self._draw_detection(frame_right, right_det)
                    
                # 3D trajektória (múlt és jövő) vizualizációja
                if state and overlay_cfg.get("show_trajectory", True):
                    self._draw_trajectory(frame_left, state.left_past_2d, state.left_future_2d)
                    self._draw_trajectory(frame_right, state.right_past_2d, state.right_future_2d)

                cam_status = self._cam_manager.get_camera_status()
                vx, vy, vz = state.velocity_mm_s if state else (0.0, 0.0, 0.0)
                pos_3d = state.pos_3d if state else None
                impact = state.impact if state else None
                detection = state.detection if state else StereoBallDetection()

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
                    "shot_confirmed": state.shot_confirmed if state else False,
                    "shot_active": state.shot_active if state else False,
                    "shot_finished": state.shot_finished if state else False,
                    "shot_armed": state.shot_armed if state else False,
                    "shot_confirmed_vz_trend_mm_s": (
                        state.shot_confirmed_vz_trend_mm_s if state else 0.0
                    ),
                    "stereo_measured": state.stereo_measured if state else False,
                    "calibrated": state.calibrated if state else False,
                    "source_sequence": state.source_sequence if state else -1,
                    # Teljes képkori késés: a zöld jelöléshez tartozó bemeneti
                    # frame és az épp kijelzett kamera-frame közti idő.
                    "detection_age_ms": (
                        (snapshot.timestamp - state.source_timestamp) * 1000.0
                        if state else 0.0
                    ),
                    "detection_inference_ms": (
                        (state.completed_at - state.source_timestamp) * 1000.0
                        if state else 0.0
                    ),
                    "detection_frame_lag": self._snapshot_sequence - state.source_sequence if state else 0,
                    "sync_delta_ms": snapshot.sync_delta_ms,
                     "hw_sync_enabled": self._cam_manager.is_hw_sync_enabled(),
                    # Bal/jobb kamera detektálási konfidencia (telemetria rekordhoz)
                    "conf_left": float(left_det.confidence) if left_det.found else 0.0,
                    "conf_right": float(right_det.confidence) if right_det.found else 0.0,
                }
                self.frames_ready.emit(frame_left, frame_right, stats)

    @staticmethod
    def _make_roi(roi_cfg: dict) -> dict:
        return {
            "enabled": bool(roi_cfg.get("enabled", False)),
            "x_min_rel": float(roi_cfg.get("x_min_rel", 0.0)),
            "x_max_rel": float(roi_cfg.get("x_max_rel", 1.0)),
            "y_min_rel": float(roi_cfg.get("y_min_rel", 0.0)),
            "y_max_rel": float(roi_cfg.get("y_max_rel", 1.0)),
        }

    def _draw_roi(self, frame: np.ndarray, is_left: bool) -> None:
        side = "left" if is_left else "right"
        with self._roi_lock:
            roi = self._display_roi[side].copy()
        if not roi["enabled"]:
            return
        height, width = frame.shape[:2]
        x1, x2 = int(width * roi["x_min_rel"]), int(width * roi["x_max_rel"])
        y1, y2 = int(height * roi["y_min_rel"]), int(height * roi["y_max_rel"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(
            frame, "ROI ACTIVE", (x1 + 6, max(y1 + 20, 22)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1,
        )

    @staticmethod
    def _draw_detection(frame: np.ndarray, detection: BallDetection) -> None:
        if not detection.found:
            return
        cx, cy, radius = int(detection.x), int(detection.y), int(detection.radius)
        x1, y1, x2, y2 = [int(value) for value in detection.bbox]
        color = (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        cv2.circle(frame, (cx, cy), radius, color, 2)
        cv2.line(frame, (cx - radius, cy), (cx + radius, cy), color, 1)
        cv2.line(frame, (cx, cy - radius), (cx, cy + radius), color, 1)
        label_parts = [f"{detection.confidence:.2f}"]
        if detection.track_id is not None:
            label_parts.append(f"ID:{detection.track_id}")
        cv2.putText(
            frame, "  ".join(label_parts), (x1, y1 - 8),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2,
        )

    @staticmethod
    def _draw_trajectory(
        frame: np.ndarray,
        past_2d: Optional[np.ndarray],
        future_2d: Optional[np.ndarray]
    ) -> None:
        """Kirajzolja a mért csóvát és a várható röppálya ívét a képre."""

        # Készítünk egy átlátszó réteget (overlay) a finom áttűnésekhez
        overlay = frame.copy()
        
        # 1. Múltbeli pálya rajzolása (zöldeskék csóva, elvékonyodó)
        if past_2d is not None and len(past_2d) >= 2:
            n = len(past_2d)
            for i in range(n - 1):
                x1, y1 = past_2d[i, 0], past_2d[i, 1]
                x2, y2 = past_2d[i+1, 0], past_2d[i+1, 1]
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                    continue
                if np.isinf(x1) or np.isinf(y1) or np.isinf(x2) or np.isinf(y2):
                    continue
                if abs(x1) > 10000 or abs(y1) > 10000 or abs(x2) > 10000 or abs(y2) > 10000:
                    continue
                    
                # Ugrás elleni védelem: ha két pont túl messze van (pl. vak/zajos detektálás), nem kötjük össze!
                if np.hypot(x2 - x1, y2 - y1) > 800.0:
                    continue

                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))
                
                # Vastagság csökken a múltba haladva (i=n-1 a legújabb pont)
                ratio = (i + 1) / n
                color = (0, 255, 200)  # BGR: sárgászöld/világoszöld (citrus)
                thickness = max(1, int(6 * ratio))
                cv2.line(overlay, pt1, pt2, color, thickness, cv2.LINE_AA)
                
        # 2. Jövőbeli röppálya: erős sárga, pontozott ív. A pontok a 3D
        # ballisztikus predikció mintái, a vonal kizárólag a köztük lévő
        # érvényes szakaszokat köti össze.
        if future_2d is not None and len(future_2d) >= 2:
            n = len(future_2d)
            first_valid_pt = None
            for i in range(n - 1):
                x1, y1 = future_2d[i, 0], future_2d[i, 1]
                x2, y2 = future_2d[i+1, 0], future_2d[i+1, 1]
                if np.isnan(x1) or np.isnan(y1) or np.isnan(x2) or np.isnan(y2):
                    continue
                if np.isinf(x1) or np.isinf(y1) or np.isinf(x2) or np.isinf(y2):
                    continue
                if abs(x1) > 10000 or abs(y1) > 10000 or abs(x2) > 10000 or abs(y2) > 10000:
                    continue

                if np.hypot(x2 - x1, y2 - y1) > 800.0:
                    continue

                pt1 = (int(x1), int(y1))
                pt2 = (int(x2), int(y2))

                if first_valid_pt is None:
                    first_valid_pt = pt1
                # A folytonos, vékony alapívhez minden második szakaszon
                # vastagabb sárga jelölés kerül, így gyors mozgásnál is jól
                # olvasható marad és megkülönböztethető a múltbeli csóvától.
                cv2.line(overlay, pt1, pt2, (0, 180, 255), 2, cv2.LINE_AA)
                if i % 2 == 0:
                    cv2.circle(overlay, pt1, 3, (0, 255, 255), -1, cv2.LINE_AA)

            if first_valid_pt is not None:
                cv2.putText(
                    overlay, "PREDIKALT PALYA",
                    (first_valid_pt[0] + 8, max(20, first_valid_pt[1] - 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 255), 1, cv2.LINE_AA,
                )

            # Várható érkezési / landolási pont megjelölése célkereszttel
            last_x, last_y = future_2d[-1, 0], future_2d[-1, 1]
            if not np.isnan(last_x) and not np.isnan(last_y) and abs(last_x) < 10000 and abs(last_y) < 10000:
                end_pt = (int(last_x), int(last_y))
                cv2.circle(overlay, end_pt, 8, (0, 180, 255), 2, cv2.LINE_AA)
                cv2.circle(overlay, end_pt, 3, (0, 255, 255), -1, cv2.LINE_AA)
                
        # Átlátszó (alpha blending) összekeverés az eredeti képpel
        # overlay súlya 75%, eredeti kép súlya 25% -> a vonalak áttetszőek lesznek 75% opacitással.
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)


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
        self._shot_armed_ui = False
        self._shot_active_ui = False
        self._auto_arm_enabled = bool(config.get("shot_detection", {}).get("auto_rearm", False))
        self._last_handled_shot_sequence = -1
        # Kapu-jelölő rögzítése: a megerősítés után legfeljebb ennyi jó sztereó
        # mérés frissítheti, az első rossz/hiányzó mérésnél azonnal befagy.
        self._marker_lock_samples = max(1, int(
            config.get("gui", {}).get("goal_view", {}).get("marker_lock_samples", 5)
        ))
        self._shot_marker_samples = 0
        self._shot_marker_locked = False
        self._last_marker_sequence = -1
        self._shot_arm_time: float = 0.0
        # ── Lövés telemetria követés ─────────────────────────────────────────
        # Az időpont, amikor a ShotDetector megerősítette a lövést
        self._shot_confirm_time: float = 0.0
        # Az aktuátor utolsó parancsvételének időpontja és célpozíciója
        self._last_goalkeeper_cmd_time: float = 0.0
        self._last_goalkeeper_x_mm: float = 0.0
        self._last_goalkeeper_y_mm: float = 0.0
        self._last_goalkeeper_speed_m_s: float = 2.5
        # Lövés-specifikus pipeline idők (az utolsó megerősítésnél tárolt stats-ból)
        self._last_shot_detection_latency_ms: float = 0.0
        self._last_shot_pipeline_ms: float = 0.0
        self._last_shot_vx: float = 0.0
        self._last_shot_vy: float = 0.0
        self._last_shot_vz: float = 0.0
        # ──────────────────────────────────────────────────
        session_dir = config.get("session", {}).get("session_dir", "data/sessions")
        self._session_manager = SessionManager(session_dir=session_dir)

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

        self._setup_log_handler()
        self._is_dark_theme = getattr(self, "_is_dark_theme", False)
        self._telemetry_cards = []
        self._btn_resets = []
        self._build_ui()

        self._apply_theme_to_ui()

        # Auto-élesítés checkbox állapot a config alapján
        if hasattr(self, '_chk_auto_arm'):
            self._chk_auto_arm.setChecked(self._auto_arm_enabled)

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
        """Kiemelt Fejléc: Bal oldalon nagy Hamburger Menü, jobb oldalon Indítás gomb."""
        toolbar = QToolBar("DEIK Fejléc")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        # 1. ≡ Nagy Hamburger Menü Gomb (Bal oldal)
        btn_hamburger = QPushButton("≡   MENÜ")
        self._btn_hamburger = btn_hamburger
        btn_hamburger.setFixedHeight(40)
        btn_hamburger.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_hamburger.setStyleSheet(
            "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 900; "
            "font-size: 14px; border-radius: 8px; border: 1px solid #10B981; padding: 0 18px; letter-spacing: 1px; }"
            "QPushButton:hover { background-color: #146C43; border-color: #34D399; }"
        )

        menu = QMenu(self)
        self._main_menu = menu
        self._apply_menu_stylesheet()

        # Section 1: Nézetek
        lbl_sec1 = menu.addAction("── NÉZETEK & DASHBOARDOK ──")
        lbl_sec1.setEnabled(False)

        act_overview  = menu.addAction("Összesített Nézet (Összes modul)")
        act_cameras   = menu.addAction("Élő Dual Kamerák (HD Stream)")
        act_goal_sim  = menu.addAction("Kapu / Robot Szimuláció (Dőlő kapus)")
        act_analytics = menu.addAction("Analitika / Hőtérkép Dashboard")
        act_actuator  = menu.addAction("Aktuátor Hardver Teszt Panel (E-Stop)")

        menu.addSeparator()

        # Section 2: Eszközök
        lbl_sec2 = menu.addAction("── ESZKÖZÖK & BEÁLLÍTÁSOK ──")
        lbl_sec2.setEnabled(False)

        act_calib = menu.addAction("Sztereó Kalibrációs Munkafolyamat")
        act_theme = menu.addAction("Téma Váltás (Sötét / Világos Mód)")

        menu.addSeparator()

        # Section 3: Információ
        lbl_sec3 = menu.addAction("── INFORMÁCIÓ ──")
        lbl_sec3.setEnabled(False)

        act_about = menu.addAction("Névjegy / Fejlesztők (DEIK v1.0)")

        # Eseménykezelők
        act_overview.triggered.connect(lambda: self._set_central_view(0))
        act_cameras.triggered.connect(lambda: self._set_central_view(1))
        act_goal_sim.triggered.connect(lambda: self._set_central_view(2))
        act_analytics.triggered.connect(lambda: self._set_central_view(3))
        act_actuator.triggered.connect(lambda: self._set_central_view(4))
        act_calib.triggered.connect(self._on_open_calibration)
        act_theme.triggered.connect(self._toggle_theme)
        act_about.triggered.connect(self._show_about_dialog)

        btn_hamburger.setMenu(menu)
        toolbar.addWidget(btn_hamburger)

        toolbar.addSeparator()

        # 2. DEIK Logó & Cím (Középső részen)
        deik_logo_path = "assets/deik_logo.png"
        if os.path.exists(deik_logo_path):
            l_lbl1 = QLabel()
            l_lbl1.setPixmap(QPixmap(deik_logo_path).scaledToHeight(42, Qt.TransformationMode.SmoothTransformation))
            toolbar.addWidget(l_lbl1)

        rgk_logo_path = "assets/logo.png"
        if os.path.exists(rgk_logo_path):
            l_lbl2 = QLabel()
            l_lbl2.setPixmap(QPixmap(rgk_logo_path).scaledToHeight(42, Qt.TransformationMode.SmoothTransformation))
            toolbar.addWidget(l_lbl2)

        title_lbl = QLabel("DEIK ROBOT FOCI KAPUS")
        self._title_lbl = title_lbl
        sub_lbl = QLabel("Debreceni Egyetem Informatikai Kar")
        self._sub_lbl = sub_lbl

        brand_widget = QWidget()
        b_box = QVBoxLayout(brand_widget)
        b_box.setContentsMargins(6, 0, 12, 0)
        b_box.setSpacing(0)
        b_box.addWidget(title_lbl)
        b_box.addWidget(sub_lbl)
        toolbar.addWidget(brand_widget)

        # Rugalmas elválasztó térköz a jobb oldalra toláshoz
        spacer = QWidget()
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)

        # 3. Jobb oldali sáv: Fő Indítás Gomb & Telemetriák
        self._btn_start = QPushButton("▶  INDÍTÁS")
        self._btn_start.setObjectName("btn_start")
        self._btn_start.setFixedHeight(40)
        self._btn_start.setMinimumWidth(130)
        self._btn_start.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_start.clicked.connect(self._on_start_stop)
        toolbar.addWidget(self._btn_start)

        self._btn_shot_arm = QPushButton("LÖVÉS ÉLESÍTÉSE")
        self._btn_shot_arm.setFixedHeight(40)
        self._btn_shot_arm.setMinimumWidth(160)
        self._btn_shot_arm.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_shot_arm.setEnabled(False)
        self._btn_shot_arm.clicked.connect(self._on_arm_next_shot)
        toolbar.addWidget(self._btn_shot_arm)

        self._chk_auto_arm = QCheckBox("AUTO ÉLESÍTÉS")
        self._chk_auto_arm.setFixedHeight(40)
        self._chk_auto_arm.setCursor(Qt.CursorShape.PointingHandCursor)
        self._chk_auto_arm.setToolTip(
            "Bekapcsolva: minden lövés után automatikusan élesít a következőre.\n"
            "Kikapcsolva: manuális élesítés szükséges minden lövés előtt."
        )
        self._chk_auto_arm.toggled.connect(self._on_auto_arm_toggled)
        toolbar.addWidget(self._chk_auto_arm)

        toolbar.addSeparator()

        self._pill_sys = QLabel(" INAKTÍV ")
        self._pill_sys.setStyleSheet(get_status_pill_style("info"))
        toolbar.addWidget(self._pill_sys)

        self._pill_gpu = QLabel(" GPU: RTX 3050 ")
        self._pill_gpu.setStyleSheet(get_status_pill_style("ok"))
        toolbar.addWidget(self._pill_gpu)

    def _apply_menu_stylesheet(self) -> None:
        dark = getattr(self, "_is_dark_theme", False)
        if hasattr(self, "_main_menu"):
            if dark:
                self._main_menu.setStyleSheet(
                    "QMenu { background-color: #0B0F17; color: #F8FAFC; border: 2px solid #26334D; "
                    "font-size: 13px; font-weight: 600; padding: 10px; border-radius: 10px; }"
                    "QMenu::item { padding: 10px 24px; border-radius: 6px; margin: 3px 0; }"
                    "QMenu::item:disabled { color: #10B981; font-weight: 900; font-size: 11px; padding: 8px 16px 4px 16px; background: transparent; }"
                    "QMenu::item:selected { background-color: #0F5132; color: #FFFFFF; font-weight: 800; }"
                    "QMenu::separator { height: 1px; background: #26334D; margin: 8px 4px; }"
                )
            else:
                self._main_menu.setStyleSheet(
                    "QMenu { background-color: #FFFFFF; color: #0F172A; border: 2px solid #CBD5E1; "
                    "font-size: 13px; font-weight: 600; padding: 10px; border-radius: 10px; }"
                    "QMenu::item { padding: 10px 24px; border-radius: 6px; margin: 3px 0; }"
                    "QMenu::item:disabled { color: #0F5132; font-weight: 900; font-size: 11px; padding: 8px 16px 4px 16px; background: transparent; }"
                    "QMenu::item:selected { background-color: #0F5132; color: #FFFFFF; font-weight: 800; }"
                    "QMenu::separator { height: 1px; background: #CBD5E1; margin: 8px 4px; }"
                )

    def _set_central_view(self, index: int) -> None:
        """Vált az 5 beépített központi nézet mód között."""
        self._central_stack.setCurrentIndex(index)

    def _update_view_btn_styles(self) -> None:
        """Téma-frissítés segédfüggvény."""
        self._apply_menu_stylesheet()

    def _build_central_widget(self) -> None:
        """Központi Terület QStackedWidget-tel: 5 Különböző Nézet Mód."""
        central = QWidget()
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setSpacing(6)
        main_layout.setContentsMargins(6, 6, 6, 6)

        self._central_stack = QStackedWidget()

        # -------------------------------------------------------------
        # PAGE 0: ÖSSZESÍTETT NÉZET (Combined View)
        # -------------------------------------------------------------
        page_comb = QWidget()
        comb_layout = QVBoxLayout(page_comb)
        comb_layout.setSpacing(10)
        comb_layout.setContentsMargins(4, 4, 4, 4)

        top_row = QHBoxLayout()
        top_row.setSpacing(10)

        left_grp = QGroupBox("Bal Kamera  [X = -1150 mm]")
        self._left_grp = left_grp
        l_box = QVBoxLayout(left_grp)
        l_box.setContentsMargins(6, 18, 6, 6)
        self._cam_label_left = ZoomableLabel()
        self._cam_label_left.setText("Kamera inaktív")
        self._cam_label_left.setMinimumSize(320, 220)
        self._cam_label_left.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        l_box.addWidget(self._cam_label_left)
        top_row.addWidget(left_grp, stretch=3)

        right_grp = QGroupBox("Jobb Kamera  [X = +1150 mm]")
        self._right_grp = right_grp
        r_box = QVBoxLayout(right_grp)
        r_box.setContentsMargins(6, 18, 6, 6)
        self._cam_label_right = ZoomableLabel()
        self._cam_label_right.setText("Kamera inaktív")
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
        self._btn_clear = btn_clear
        btn_clear.setFixedHeight(30)
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.clicked.connect(self._on_clear_history)
        g_box.addWidget(btn_clear)

        top_row.addWidget(goal_grp, stretch=4)
        comb_layout.addLayout(top_row, stretch=3)

        # 3D Telemetriai Kártyák
        tel_grp = QGroupBox("ÉLŐ 3D TELEMETRIA / VÉDELMI ZÓNA ELŐREJELZÉS")
        tel_box = QGridLayout(tel_grp)
        tel_box.setSpacing(6)
        tel_box.setContentsMargins(12, 18, 12, 10)

        font_val = QFont("Consolas", 11, QFont.Weight.Bold)  # Kicsit kisebb betűméret, hogy biztosan kiférjen

        def make_card(title: str, color: str = "#0F5132") -> tuple[QLabel, QWidget]:
            w = QWidget()
            v = QVBoxLayout(w)
            v.setContentsMargins(6, 4, 6, 4)
            v.setSpacing(2)
            t_lbl = QLabel(title)
            t_lbl.setWordWrap(True)
            # A címkét picit kisebb, vastag betűvel írjuk ki a jobb tördelés miatt
            font_title = QFont("Segoe UI", 9, QFont.Weight.Bold)
            t_lbl.setFont(font_title)
            val_lbl = QLabel("—")
            val_lbl.setFont(font_val)
            val_lbl.setWordWrap(True)
            v.addWidget(t_lbl)
            v.addWidget(val_lbl)
            self._telemetry_cards.append((w, t_lbl, val_lbl, color))
            return val_lbl, w

        self._lbl_x, card_x = make_card("LABDA X (MM)")
        self._lbl_y, card_y = make_card("LABDA Y (MM)")
        self._lbl_z, card_z = make_card("LABDA Z (MM)")
        self._lbl_speed, card_sp = make_card("SEBESSÉG (KM/H)", "#059669")
        self._lbl_impact, card_imp = make_card("BECSAPÓDÁS (X,Y)", "#D97706")
        self._lbl_time, card_time = make_card("IDŐ (MP)", "#D97706")
        self._lbl_zone, card_zone = make_card("SZEKTOR", "#0F5132")

        # 2 sorba rendezzük a kártyákat a Grid-ben (első sor 4 kártya, második sor 3 kártya)
        tel_box.addWidget(card_x, 0, 0)
        tel_box.addWidget(card_y, 0, 1)
        tel_box.addWidget(card_z, 0, 2)
        tel_box.addWidget(card_sp, 0, 3)
        
        tel_box.addWidget(card_imp, 1, 0, 1, 2)  # A becsapódás 2 oszlopnyi helyet foglal el
        tel_box.addWidget(card_time, 1, 2)
        tel_box.addWidget(card_zone, 1, 3)

        comb_layout.addWidget(tel_grp, stretch=1)
        self._central_stack.addWidget(page_comb)

        # -------------------------------------------------------------
        # PAGE 1: ÉLŐ KAMERÁK NÉZET (Live Stereo Cameras View)
        # -------------------------------------------------------------
        page_cams = QWidget()
        cams_layout = QHBoxLayout(page_cams)
        cams_layout.setSpacing(12)
        cams_layout.setContentsMargins(4, 4, 4, 4)

        left_grp_full = QGroupBox("Bal Kamera — Nagyfelbontású Élő Videófolyam  [X = -1150 mm]")
        self._left_grp_full = left_grp_full
        lf_box = QVBoxLayout(left_grp_full)
        lf_box.setContentsMargins(8, 20, 8, 8)
        self._cam_label_left_full = ZoomableLabel()
        self._cam_label_left_full.setText("Kamera inaktív")
        self._cam_label_left_full.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        lf_box.addWidget(self._cam_label_left_full)
        cams_layout.addWidget(left_grp_full, stretch=1)

        right_grp_full = QGroupBox("Jobb Kamera — Nagyfelbontású Élő Videófolyam  [X = +1150 mm]")
        self._right_grp_full = right_grp_full
        rf_box = QVBoxLayout(right_grp_full)
        rf_box.setContentsMargins(8, 20, 8, 8)
        self._cam_label_right_full = ZoomableLabel()
        self._cam_label_right_full.setText("Kamera inaktív")
        self._cam_label_right_full.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        rf_box.addWidget(self._cam_label_right_full)
        cams_layout.addWidget(right_grp_full, stretch=1)

        self._central_stack.addWidget(page_cams)

        # -------------------------------------------------------------
        # PAGE 2: KAPU & SZIMULÁCIÓ NÉZET (Goal & Robot Simulation View)
        # -------------------------------------------------------------
        page_goal = QWidget()
        goal_layout = QVBoxLayout(page_goal)
        goal_layout.setSpacing(10)
        goal_layout.setContentsMargins(4, 4, 4, 4)

        goal_grp_full = QGroupBox("Nagyfelbontású Kapu Vizualizáció / Robot Kapus Szimuláció")
        gf_box = QVBoxLayout(goal_grp_full)
        gf_box.setContentsMargins(8, 20, 8, 8)
        gf_box.setSpacing(8)

        self._goal_view_full = GoalViewWidget(self._config)
        gf_box.addWidget(self._goal_view_full)

        btn_clear_full = QPushButton("Lövéstörténet Törlése")
        self._btn_clear_full = btn_clear_full
        btn_clear_full.setFixedHeight(34)
        btn_clear_full.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear_full.clicked.connect(self._on_clear_history)
        gf_box.addWidget(btn_clear_full)

        goal_layout.addWidget(goal_grp_full, stretch=4)
        self._central_stack.addWidget(page_goal)

        # -------------------------------------------------------------
        # PAGE 3: 📊 ANALITIKA & HŐTÉRKÉP DASHBOARD
        # -------------------------------------------------------------
        self._analytics_view = AnalyticsDashboardWidget(self._config)
        self._central_stack.addWidget(self._analytics_view)

        # -------------------------------------------------------------
        # PAGE 4: ⚙️ AKTUÁTOR HARDVER TESZT PANEL
        # -------------------------------------------------------------
        self._actuator_view = ActuatorControlWidget(self._config)
        self._actuator_view.position_changed.connect(self._on_actuator_manual_pos)
        self._central_stack.addWidget(self._actuator_view)

        main_layout.addWidget(self._central_stack)

        main_layout.addWidget(self._central_stack)

    def _build_control_dock(self) -> None:
        """Reszponzív Vezérlő Dock Panel szegmentált gombokkal és QStackedWidget-tel."""
        dock = QDockWidget("Kamera Pozícionálás & ROI Vezérlés", self)
        dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea | Qt.DockWidgetArea.LeftDockWidgetArea)

        self._cam_widgets = {}

        dock_widget = QWidget()
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
        self._btn_dock_left = btn_left
        self._btn_dock_right = btn_right
        self._btn_dock_sys = btn_sys

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
            self._update_dock_nav_styles()

        btn_left.clicked.connect(lambda: set_page(0))
        btn_right.clicked.connect(lambda: set_page(1))
        btn_sys.clicked.connect(lambda: set_page(2))

        self._update_dock_nav_styles()

        dock.setWidget(dock_widget)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, dock)

    def _update_dock_nav_styles(self) -> None:
        dark = getattr(self, "_is_dark_theme", False)
        if dark:
            active_style = (
                "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; border-radius: 6px; font-size: 11px; border: 1px solid #10B981; }"
            )
            inactive_style = (
                "QPushButton { background-color: #151D2A; color: #94A3B8; font-weight: 700; border-radius: 6px; font-size: 11px; border: 1px solid #26334D; }"
                "QPushButton:hover { background-color: #1E293B; color: #10B981; border-color: #10B981; }"
            )
        else:
            active_style = (
                "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; border-radius: 6px; font-size: 11px; border: 1px solid #0F5132; }"
            )
            inactive_style = (
                "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 6px; font-size: 11px; border: 1px solid #CBD5E1; }"
                "QPushButton:hover { background-color: #E2E8F0; color: #0F5132; border-color: #0F5132; }"
            )

        if hasattr(self, "_btn_dock_left"):
            self._btn_dock_left.setStyleSheet(active_style if self._btn_dock_left.isChecked() else inactive_style)
            self._btn_dock_right.setStyleSheet(active_style if self._btn_dock_right.isChecked() else inactive_style)
            self._btn_dock_sys.setStyleSheet(active_style if self._btn_dock_sys.isChecked() else inactive_style)

    def _create_camera_tab(self, is_left: bool) -> QWidget:
        side = "left" if is_left else "right"
        side_name = "Bal" if is_left else "Jobb"

        cam_cfg = self._config["camera"].get(side, {})
        def_cfg = self._config["camera"]

        tab_widget = QWidget()
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
        spin_ymin.setValue(int(cam_cfg.get("roi_y_min", 0)))
        spin_ymin.setSuffix(" %")

        spin_ymax = QSpinBox()
        spin_ymax.setRange(1, 100)
        spin_ymax.setValue(int(cam_cfg.get("roi_y_max", 100)))
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
        combo_rot.setCurrentIndex(rot_map.get(int(cam_cfg.get("rotation", 90)), 1))

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
        self._btn_resets.append(btn_reset)
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
        self._lbl_diag_sync_mode = QLabel("—")
        self._lbl_diag_sync_delta = QLabel("—")

        cam_form.addRow("Bal Kamera FPS:", self._lbl_diag_fps_l)
        cam_form.addRow("Bal Hőmérséklet:", self._lbl_diag_temp_l)
        cam_form.addRow("Jobb Kamera FPS:", self._lbl_diag_fps_r)
        cam_form.addRow("Jobb Hőmérséklet:", self._lbl_diag_temp_r)
        cam_form.addRow("Sztereó Pár FPS:", self._lbl_diag_fps_pair)
        cam_form.addRow("YOLO Detektálás FPS:", self._lbl_diag_fps_det)
        cam_form.addRow("Kalibrációs Státusz:", self._lbl_diag_calib)
        cam_form.addRow("Szinkronizáció mód:", self._lbl_diag_sync_mode)
        cam_form.addRow("HW Szinkron Jitter:", self._lbl_diag_sync_delta)

        layout.addWidget(cam_grp)

        # Hardver & Erőforrás Használat
        hw_grp = QGroupBox("Hardver Erőforrások (CPU / GPU / RAM)")
        hw_form = QFormLayout(hw_grp)
        hw_form.setSpacing(6)

        self._lbl_diag_gpu = QLabel("NVIDIA RTX 3050 (6.1 GB VRAM)")
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
        self._lbl_diag_inference = QLabel("— ms")
        self._lbl_diag_pipeline = QLabel("— ms")
        self._lbl_diag_frame_lag = QLabel("— frame")

        proc_form.addRow("Követési Státusz:", self._lbl_diag_det_status)
        proc_form.addRow("Legutóbbi 3D Pozíció:", self._lbl_diag_pos3d)
        proc_form.addRow("YOLO Inferencia:", self._lbl_diag_inference)
        proc_form.addRow("Pipeline Késleltetés:", self._lbl_diag_pipeline)
        proc_form.addRow("Frame Lemaradás:", self._lbl_diag_frame_lag)

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

        dark = getattr(self, "_is_dark_theme", False)

        self._pill_cpu = QLabel(" CPU: --% ")
        self._pill_cpu.setStyleSheet(get_hw_pill_style("low", dark))

        self._pill_ram = QLabel(" RAM: --% ")
        self._pill_ram.setStyleSheet(get_hw_pill_style("low", dark))

        self._pill_gpu_usage = QLabel(" GPU: --% ")
        self._pill_gpu_usage.setStyleSheet(get_hw_pill_style("low", dark))

        self._pill_gpu_temp = QLabel(" --°C ")
        self._pill_gpu_temp.setStyleSheet(get_hw_pill_style("low", dark))

        self._status_bar.addPermanentWidget(self._pill_cpu)
        self._status_bar.addPermanentWidget(self._pill_ram)
        self._status_bar.addPermanentWidget(self._pill_gpu_usage)
        self._status_bar.addPermanentWidget(self._pill_gpu_temp)

        creators_lbl = QLabel(" Készítők: Morvai Roland & Rácz Donát (BSc Mérnökinformatikus) | DEIK v1.0 ")
        self._creators_lbl = creators_lbl
        self._status_bar.addPermanentWidget(creators_lbl)

    @pyqtSlot()
    def _show_about_dialog(self) -> None:
        """Megnyitja a témahű Névjegy & Készítők ablakot."""
        dark = getattr(self, "_is_dark_theme", False)
        dlg = QDialog(self)
        dlg.setWindowTitle("Névjegy & Fejlesztési Információk")
        dlg.setMinimumWidth(480)
        dlg.setStyleSheet(
            "QDialog, QWidget { background-color: #0B0F17; color: #F8FAFC; font-family: 'Segoe UI', sans-serif; }"
            if dark else
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
        title.setStyleSheet("font-weight: 800; font-size: 19px; color: #4ADE80;" if dark else "font-weight: 800; font-size: 19px; color: #0F5132;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(title)

        sub = QLabel("Debreceni Egyetem Informatikai Kar")
        sub.setStyleSheet("font-size: 12px; color: #F59E0B; font-weight: 700;" if dark else "font-size: 12px; color: #D97706; font-weight: 700;")
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(sub)

        ver = QLabel("Verzió 1.0.0 (2026)")
        ver.setStyleSheet("background-color: #1E293B; color: #94A3B8; font-weight: 600; font-size: 11px; border-radius: 8px; padding: 2px 10px;" if dark else "background-color: #F1F5F9; color: #475569; font-weight: 600; font-size: 11px; border-radius: 8px; padding: 2px 10px;")
        ver.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vbox.addWidget(ver)

        # Egyetlen Kártya Konténer
        card = QWidget()
        card.setStyleSheet(
            "background-color: #151D2A; border: 1px solid #26334D; border-radius: 8px; padding: 14px;"
            if dark else
            "background-color: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px;"
        )
        card_vbox = QVBoxLayout(card)
        card_vbox.setSpacing(8)

        lbl_dev = QLabel("FEJLESZTŐK / KÉSZÍTŐK")
        lbl_dev.setStyleSheet("font-weight: 800; font-size: 11px; color: #4ADE80; letter-spacing: 0.5px;" if dark else "font-weight: 800; font-size: 11px; color: #0F5132; letter-spacing: 0.5px;")
        card_vbox.addWidget(lbl_dev)

        m1 = QLabel("• Morvai Roland – BSc Mérnökinformatikus (DEIK)")
        m1.setStyleSheet("font-weight: 700; color: #F8FAFC; font-size: 13px;" if dark else "font-weight: 700; color: #0F172A; font-size: 13px;")
        card_vbox.addWidget(m1)

        m2 = QLabel("• Rácz Donát – BSc Mérnökinformatikus (DEIK)")
        m2.setStyleSheet("font-weight: 700; color: #F8FAFC; font-size: 13px;" if dark else "font-weight: 700; color: #0F172A; font-size: 13px;")
        card_vbox.addWidget(m2)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setStyleSheet("color: #26334D;" if dark else "color: #E2E8F0;")
        card_vbox.addWidget(line)

        lbl_info = QLabel("PROJEKT & TECHNOLÓGIA")
        lbl_info.setStyleSheet("font-weight: 800; font-size: 11px; color: #4ADE80; letter-spacing: 0.5px;" if dark else "font-weight: 800; font-size: 11px; color: #0F5132; letter-spacing: 0.5px;")
        card_vbox.addWidget(lbl_info)

        details = QLabel(
            "• <b>Projekt:</b> Valós idejű sztereó látórendszer és trajektória előrejelzés robot kapushoz.<br>"
            "• <b>Szoftver stack:</b> Python 3.12, PyQt6, OpenCV, PyTorch, YOLOv8 (CUDA GPU Acceleration)"
        )
        details.setStyleSheet("color: #CBD5E1; font-size: 12px;" if dark else "color: #334155; font-size: 12px;")
        card_vbox.addWidget(details)

        vbox.addWidget(card)

        # Rendben Gomb
        btn_ok = QPushButton("Rendben")
        btn_ok.setFixedSize(120, 36)
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

        # Szinkronizáljuk a főoldali GUI-ban beállított kamera paramétereket a konfigba
        self._sync_config_from_ui()

        dlg = CalibrationDialog(self._config, is_dark=getattr(self, "_is_dark_theme", False), parent=self)
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
    def _on_auto_arm_toggled(self, checked: bool) -> None:
        """Auto-élesítés be/kikapcsolása."""
        self._auto_arm_enabled = checked
        if checked:
            self._status_bar.showMessage("Auto-élesítés BEKAPCSOLVA: lövések után automatikus élesítés.")
        else:
            self._status_bar.showMessage("Auto-élesítés KIKAPCSOLVA: manuális élesítés szükséges.")

    @pyqtSlot()
    def _on_clear_history(self) -> None:
        if self._worker:
            self._worker.disarm_shot()
        self._set_shot_arm_button(False, False)
        # Munkamenet nullázás
        if hasattr(self, '_session_manager'):
            self._session_manager.clear()
        # A GoalViewWidget a reset_stats() metódust használja a teljes előzmény és statisztika törléséhez
        if hasattr(self, "_goal_view") and hasattr(self._goal_view, "reset_stats"):
            self._goal_view.reset_stats()
        if hasattr(self, "_goal_view_full") and hasattr(self._goal_view_full, "reset_stats"):
            self._goal_view_full.reset_stats()
        if hasattr(self, "_analytics_view") and hasattr(self._analytics_view, "_clear_analytics"):
            self._analytics_view._clear_analytics()
        logger.info("Lövés történet és statisztikák sikeresen törölve.")

    @pyqtSlot()
    def _on_arm_next_shot(self) -> None:
        """Élesíti, illetve második kattintásra megszakítja a következő lövés figyelését."""
        if not self._worker or not self._is_running:
            self._status_bar.showMessage("Előbb indítsd el a kamerakövetést.")
            return
        if self._shot_armed_ui:
            self._worker.disarm_shot()
            self._set_shot_arm_button(False, False)
            self._status_bar.showMessage("Lövésfigyelés megszakítva.")
            return
        self._shot_arm_time = time.time()
        self._worker.arm_next_shot()
        self._set_shot_arm_button(True, False)
        self._status_bar.showMessage("Lövésfigyelés élesítve: várakozás távoli indítópozícióra.")

    def _set_shot_arm_button(
        self, armed: bool, active: bool, preserve_pending: bool = False
    ) -> None:
        """Frissíti a következő-lövés élesítő gomb állapotát."""
        if preserve_pending and self._shot_armed_ui and not active:
            armed = True
        self._shot_armed_ui = armed
        self._shot_active_ui = active
        if active:
            self._btn_shot_arm.setText("LÖVÉS FOLYAMATBAN")
            self._btn_shot_arm.setEnabled(False)
        elif armed:
            self._btn_shot_arm.setText("ÉLESÍTVE - MÉGSE")
            self._btn_shot_arm.setEnabled(True)
        else:
            self._btn_shot_arm.setText("LÖVÉS ÉLESÍTÉSE")
            self._btn_shot_arm.setEnabled(self._is_running)

    def _apply_camera_overlay_info(self, frame: np.ndarray, side: str, fps: float) -> np.ndarray:
        """Kirajzolja az FPS és kamera info overlay-t közvetlenül a képkockára."""
        if frame is None or frame.size == 0:
            return frame
        out = frame.copy()
        cam_cfg = self._config.get("camera", {}).get(side, {})
        exp = cam_cfg.get("exposure_time_us", 3000)

        side_text = "BAL" if side == "left" else "JOBB"
        label_text = f"{side_text}: {fps:.1f} FPS  |  {exp} us"

        cv2.rectangle(out, (10, 10), (270, 42), (15, 23, 42), -1)
        cv2.rectangle(out, (10, 10), (270, 42), (15, 81, 50), 1)

        cv2.circle(out, (24, 26), 5, (74, 222, 128), -1)
        cv2.putText(
            out, label_text, (38, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
        )
        return out

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

        # FPS overlay rárajzolása
        cam_fps_l = stats.get("cam_fps_left", 0.0)
        cam_fps_r = stats.get("cam_fps_right", 0.0)
        left_overlay = self._apply_camera_overlay_info(left_roi_zoom, "left", cam_fps_l)
        right_overlay = self._apply_camera_overlay_info(right_roi_zoom, "right", cam_fps_r)

        self._display_frame(left_overlay, self._cam_label_left)
        self._display_frame(right_overlay, self._cam_label_right)

        if hasattr(self, "_cam_label_left_full") and self._cam_label_left_full:
            self._display_frame(left_overlay, self._cam_label_left_full)
        if hasattr(self, "_cam_label_right_full") and self._cam_label_right_full:
            self._display_frame(right_overlay, self._cam_label_right_full)

        # Dinamikus kamera fejléc frissítés
        if hasattr(self, "_left_grp") and self._left_grp:
            self._left_grp.setTitle(f"Bal Kamera  ●  {cam_fps_l:.0f} FPS  [X = -1150 mm]")
        if hasattr(self, "_right_grp") and self._right_grp:
            self._right_grp.setTitle(f"Jobb Kamera  ●  {cam_fps_r:.0f} FPS  [X = +1150 mm]")
        if hasattr(self, "_left_grp_full") and self._left_grp_full:
            self._left_grp_full.setTitle(f"Bal Kamera — Nagyfelbontású Élő Videófolyam  ●  {cam_fps_l:.0f} FPS")
        if hasattr(self, "_right_grp_full") and self._right_grp_full:
            self._right_grp_full.setTitle(f"Jobb Kamera — Nagyfelbontású Élő Videófolyam  ●  {cam_fps_r:.0f} FPS")

        if stats["pos_valid"]:
            self._lbl_x.setText(f"{stats['x_3d']:+.1f}")
            # Y tengely: ha érvényes ballisztikus impact.y_mm előrejelzés van,
            # azt mutatjuk (pontosabb). Ha nincs, visszaesünk a mért pozícióra.
            _impact_for_y: Optional[ImpactPrediction] = stats.get("impact")
            if _impact_for_y and _impact_for_y.valid:
                self._lbl_y.setText(f"{_impact_for_y.y_mm:+.1f} ⬡")
            else:
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
        shot_confirmed = bool(stats.get("shot_confirmed", False))
        shot_active = bool(stats.get("shot_active", False))
        shot_finished = bool(stats.get("shot_finished", False))
        source_sequence = int(stats.get("source_sequence", -1))
        shot_confirmed_event = shot_confirmed and source_sequence != self._last_handled_shot_sequence
        if shot_confirmed_event:
            self._last_handled_shot_sequence = source_sequence
        self._set_shot_arm_button(
            bool(stats.get("shot_armed", False)),
            shot_active,
            preserve_pending=not shot_finished,
        )
        if impact and impact.valid:
            self._lbl_impact.setText(f"X:{impact.x_mm:+.0f} Y:{impact.y_mm:.0f} mm")
            self._lbl_time.setText(f"{impact.time_to_impact_s:.3f} mp")
            zone_text = self._determine_defense_zone(impact.x_mm, impact.y_mm)
            self._lbl_zone.setText(zone_text)
        else:
            self._lbl_impact.setText("—")
            self._lbl_time.setText("—")
            self._lbl_zone.setText("— KÖZÉP —")

        # --- Lövés-jelölő pozíció a kapu-vizualizációhoz ---
        # A ballisztikus impact.y_mm (magasság) gyors lövésnél megbízhatatlan a sztereó
        # Y-hiba miatt, ezért a magasságot – ha van érvényes MÉRT pozíció – mindig onnan
        # vesszük, nem a ballisztikus előrejelzésből (ami akár 0-ra is klampolódhat).
        # Az X koordinátát a megbízható ballisztikus predikcióból vagy mérésből vesszük.
        goal_w = float(self._config.get("geometry", {}).get("goal_width_mm", 4000.0))
        goal_h = float(self._config.get("geometry", {}).get("goal_height_mm", 2000.0))
        # Az `in_goal` eredményt is az X-alapú, megbízható margóellenőrzéssel számoljuk
        # (nem `impact.in_goal`-ból, ami a törékeny ballisztikus Y-on múlik és gyors
        # lövésnél szinte mindig False-t adna, l. shot_detection repo memória).
        goal_margin_mm = float(self._config.get("geometry", {}).get("goal_margin_mm", 150.0))
        y_off = float(self._config.get("geometry", {}).get("impact_y_offset_mm", 0.0))
        shot_pt: Optional[Tuple[float, float, float, float, bool]] = None
        if impact and impact.valid:
            if stats.get("pos_valid"):
                sy_disp = min(max(float(stats.get("y_3d", impact.y_mm)) + y_off, 0.0), goal_h)
            else:
                sy_disp = min(max(impact.y_mm + y_off, 0.0), goal_h)
            eff_in_goal = abs(impact.x_mm) <= (goal_w / 2.0 + goal_margin_mm)
            shot_pt = (impact.x_mm, sy_disp, impact.confidence,
                       impact.time_to_impact_s, eff_in_goal)
        elif (shot_confirmed_event or shot_active) and stats.get("pos_valid"):
            sx = float(stats.get("x_3d", 0.0))
            sy = min(max(float(stats.get("y_3d", goal_h / 2.0)) + y_off, 0.0), goal_h)
            shot_pt = (sx, sy, 0.5, 0.0, abs(sx) <= goal_w / 2.0)

        # A labda a kapu közelében kiesik a látómezőből / bemozdul, onnan a becslés
        # romlik – ezért a jelölőt csak az első néhány jó sztereó mérés finomítja.
        marker_update = False
        if shot_confirmed_event:
            self._shot_marker_samples = 1
            self._shot_marker_locked = self._marker_lock_samples <= 1
            self._last_marker_sequence = source_sequence
        elif (shot_active and not self._shot_marker_locked
              and source_sequence != self._last_marker_sequence):
            self._last_marker_sequence = source_sequence
            if shot_pt is not None and stats.get("stereo_measured", False):
                marker_update = True
                self._shot_marker_samples += 1
                if self._shot_marker_samples >= self._marker_lock_samples:
                    self._shot_marker_locked = True
            else:
                self._shot_marker_locked = True
            if self._shot_marker_locked:
                logger.info(
                    "Kapu-jelölő rögzítve %d jó mérés után%s",
                    self._shot_marker_samples,
                    "" if marker_update else " (labda elveszett / nem sztereó mérés)",
                )

        if shot_pt is not None:
            px_mm, py_mm, pconf, pt_s, p_in_goal = shot_pt
            for gv in (getattr(self, "_goal_view", None), getattr(self, "_goal_view_full", None)):
                if gv is None:
                    continue
                if shot_confirmed_event:
                    gv.begin_shot(px_mm, py_mm, pconf, pt_s, p_in_goal)
                elif marker_update:
                    gv.update_shot(px_mm, py_mm, pconf, pt_s, p_in_goal)

            if shot_confirmed_event:
                # ── Lövés megerősítés: telemetria gyűjtés és időzítés ──
                self._shot_confirm_time = time.perf_counter()
                # Pipeline és detektálási késleltetés tárolása későbbi használatra
                self._last_shot_detection_latency_ms = stats.get("detection_inference_ms", 0.0)
                self._last_shot_pipeline_ms = stats.get("detection_age_ms", 0.0)
                self._last_shot_vx = stats.get("vx_mms", 0.0)
                self._last_shot_vy = stats.get("vy_mms", 0.0)
                self._last_shot_vz = stats.get("vz_mms", 0.0)

                # ── Automatikus kapus mozgatás: ha van érvényes impact előrejelzés ──
                auto_gk_sent = False
                if impact and impact.valid and hasattr(self, "_actuator_view"):
                    # A kapusnak pontosan oda mozogjon, ahova a labda jön
                    gk_x = float(np.clip(impact.x_mm,
                        -float(self._config.get("geometry", {}).get("goal_width_mm", 4000)) / 2.0,
                         float(self._config.get("geometry", {}).get("goal_width_mm", 4000)) / 2.0))
                    # Y: ugyanaz a (mért, klampolt) magasság, mint a jelölőn – a ballisztikus impact.y_mm gyors lövésnél hibás.
                    gk_y = py_mm
                    self._last_goalkeeper_x_mm = gk_x
                    self._last_goalkeeper_y_mm = gk_y
                    self._last_goalkeeper_speed_m_s = getattr(
                        self._actuator_view, "_speed_m_s", 2.5
                    )
                    self._last_goalkeeper_cmd_time = time.perf_counter()
                    auto_gk_sent = True
                    # Vizualizació frissítése
                    if hasattr(self, "_goal_view") and self._goal_view:
                        self._goal_view.set_goalkeeper_target(gk_x, gk_y)
                    if hasattr(self, "_goal_view_full") and self._goal_view_full:
                        self._goal_view_full.set_goalkeeper_target(gk_x, gk_y)
                    logger.info(
                        "Kapus automatikus küldés: X=%.0f mm (impact pred), Y=%.0f mm",
                        gk_x, gk_y
                    )

                # ── Rekord mentés az analytics-be és session manager-be ──
                if hasattr(self, "_analytics_view") and self._analytics_view:
                    # A ShotDetector saját Z-regressziójának sebessége (vz_trend) a
                    # megerősítés alapja is, ezért robusztus/tesztelt – ellentétben a
                    # TrajectoryPredictor pillanatnyi Kalman-sebességével, ami a lövés
                    # elindulása után még nem konvergált, és jelentősen alábecsülhet.
                    confirmed_vz = stats.get("shot_confirmed_vz_trend_mm_s", 0.0)
                    if confirmed_vz:
                        speed_mm_s = float(np.hypot(
                            np.hypot(self._last_shot_vx, self._last_shot_vy), confirmed_vz
                        ))
                        speed_kmh = speed_mm_s / 1000.0 * 3.6
                    else:
                        speed_kmh = (stats.get("speed_ms", 12.5) or 12.5) * 3.6
                    sector = self._determine_defense_zone(px_mm, py_mm)

                    # Y érték forrása – a magasság a MÉRT pozícióból jön, ha érvényes
                    # (a ballisztikus predikció Y-a gyors lövésnél megbízhatatlan),
                    # csak érvénytelen mérés esetén esünk vissza a predikcióra.
                    y_source = "mért" if stats.get("pos_valid") else "pred"

                    # Kapus reakcióidő: lövés megerősítés → parancs küldés
                    goalkeeper_reaction_ms = 0.0
                    if auto_gk_sent and self._shot_confirm_time > 0:
                        goalkeeper_reaction_ms = (
                            self._last_goalkeeper_cmd_time - self._shot_confirm_time
                        ) * 1000.0
                    elif self._last_goalkeeper_cmd_time > self._shot_confirm_time:
                        # Kézi küldés esetén
                        goalkeeper_reaction_ms = (
                            self._last_goalkeeper_cmd_time - self._shot_confirm_time
                        ) * 1000.0

                    # Kapus becsült mozgási ideje (távolság / sebesség)
                    goalkeeper_travel_ms = 0.0
                    if auto_gk_sent and self._last_goalkeeper_speed_m_s > 0:
                        # Jelenlegi kapus pozícióját olvassuk a goal_view-ból
                        gv = getattr(self, "_goal_view", None)
                        cur_gx, cur_gy = 0.0, 1000.0
                        if gv and hasattr(gv, "_goalkeeper_x") and hasattr(gv, "_goalkeeper_y"):
                            cur_gx = float(gv._goalkeeper_x)
                            cur_gy = float(gv._goalkeeper_y)
                        dist_mm = float(np.hypot(
                            self._last_goalkeeper_x_mm - cur_gx,
                            self._last_goalkeeper_y_mm - cur_gy
                        ))
                        goalkeeper_travel_ms = (dist_mm / 1000.0) / self._last_goalkeeper_speed_m_s * 1000.0

                    # Detektálási módszer: jelenleg kizárólag YOLO TensorRT GPU inferencia
                    det_method = "YOLO"

                    # Z pozíció rögzítés
                    z_now = stats.get("z_3d", 0.0)

                    record = self._analytics_view.add_shot_event(
                        x_mm=px_mm,
                        y_mm=py_mm,
                        conf=pconf,
                        in_goal=p_in_goal,
                        speed_kmh=speed_kmh,
                        time_to_impact_s=pt_s,
                        sector=sector,
                        # ── Bővített telemetria ──────────────────────────────
                        y_source=y_source,
                        detection_latency_ms=self._last_shot_detection_latency_ms,
                        total_pipeline_ms=self._last_shot_pipeline_ms,
                        goalkeeper_reaction_ms=goalkeeper_reaction_ms,
                        goalkeeper_x_cmd_mm=self._last_goalkeeper_x_mm if auto_gk_sent else 0.0,
                        goalkeeper_y_cmd_mm=self._last_goalkeeper_y_mm if auto_gk_sent else 0.0,
                        goalkeeper_travel_ms=goalkeeper_travel_ms,
                        vx_mm_s=self._last_shot_vx,
                        vy_mm_s=self._last_shot_vy,
                        vz_mm_s=self._last_shot_vz,
                        z_start_mm=0.0,   # Lövés induló Z-pozíciója (ShotDetector.samples[0][1] alapján töltendő be)
                        z_end_mm=z_now,
                        det_method=det_method,
                        conf_left=stats.get("conf_left", 0.0),
                        conf_right=stats.get("conf_right", 0.0),
                    )
                    # Munkamenet perzisztencia: automatikus mentés
                    if record and hasattr(self, '_session_manager'):
                        self._session_manager.save_shot(record)
                        logger.info(
                            "TELEMETRIA MENTVE: X=%.0f Y=%.0f (forrás: %s) "
                            "v=(%.0f, %.0f, %.0f) mm/s | det_lat=%.1f ms | pipeline=%.1f ms | "
                            "kapus_reakció=%.1f ms | kapus_utazás=%.1f ms",
                            px_mm, py_mm, y_source,
                            self._last_shot_vx, self._last_shot_vy, self._last_shot_vz,
                            self._last_shot_detection_latency_ms, self._last_shot_pipeline_ms,
                            goalkeeper_reaction_ms, goalkeeper_travel_ms,
                        )

        if shot_finished:
            if hasattr(self, "_goal_view") and self._goal_view:
                self._goal_view.finish_shot()
            if hasattr(self, "_goal_view_full") and self._goal_view_full:
                self._goal_view_full.finish_shot()
            # Auto-élesítés mód: ha be van kapcsolva, automatikusan élesítjük a következő lövést
            if self._auto_arm_enabled and self._worker and self._is_running:
                self._shot_arm_time = time.time()
                self._worker.arm_next_shot()
                self._set_shot_arm_button(True, False)
                self._status_bar.showMessage("Auto-élesítés: következő lövés élesítve.")
            else:
                # Manual arm módban: a lövés lezárása után visszaváltunk DISARMED-ra,
                # így a felhasználónak újra kell élesítenie a következő lövést.
                self._set_shot_arm_button(False, False)

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

            # Szinkron mód és jitter megjelenítése
            delta_ms = stats.get("sync_delta_ms", 0.0)
            hw_sync  = stats.get("hw_sync_enabled", False)

            if hasattr(self, "_lbl_diag_sync_mode"):
                if hw_sync:
                    mode_color = "#00e676"
                    mode_text  = "⚡ HW GPIO trigger (opto-izolált)"
                else:
                    mode_color = "#ffeb3b"
                    mode_text  = "💻 Szoftver szinkron"
                self._lbl_diag_sync_mode.setText(
                    f"<span style='color:{mode_color}; font-weight:bold;'>{mode_text}</span>"
                )
                self._lbl_diag_sync_mode.setTextFormat(Qt.TextFormat.RichText)

            if abs(delta_ms - (-2.0)) < 0.01:
                # Frame mismatch sentinel: ring buffer aszinkron olvasás
                # A HW trigger rendben van, csak a frame_id-k nem egyeznek
                self._lbl_diag_sync_delta.setText(
                    "<span style='color:#90a4ae; font-weight:bold;'>"
                    "↔ Frame illesztés... (ring buffer)</span>"
                )
            elif delta_ms < 0.0:
                # Sentinel: HW GPIO clock-offset kalibráció folyamatban (első 30 frame)
                self._lbl_diag_sync_delta.setText(
                    "<span style='color:#ffeb3b; font-weight:bold;'>"
                    "⏳ Kalibrálás... (első 30 illesztett frame)</span>"
                )
            elif hw_sync:
                # HW GPIO szinkron – µs-ban mutatjuk, mért referencia: STD=10.8µs
                delta_us = delta_ms * 1000.0
                if delta_us < 100.0:
                    color = "#00e676"
                    icon  = "✓ HW SYNC KIVÁLÓ"
                elif delta_us < 500.0:
                    color = "#69f0ae"
                    icon  = "✓ HW SYNC JÓ"
                elif delta_ms < 2.0:
                    color = "#ffeb3b"
                    icon  = "⚠ USB LATENCIA ZAJ"
                else:
                    color = "#ff5252"
                    icon  = "✗ SZINKRON HIBA"
                self._lbl_diag_sync_delta.setText(
                    f"<span style='color:{color}; font-weight:bold;'>"
                    f"{delta_us:.1f} µs – {icon}</span>"
                )
            elif delta_ms < 5.0:
                color = "#ffeb3b"
                self._lbl_diag_sync_delta.setText(
                    f"<span style='color:{color}; font-weight:bold;'>"
                    f"{delta_ms:.3f} ms – ⚠ Szoftver szinkron</span>"
                )
            else:
                color = "#ff5252"
                self._lbl_diag_sync_delta.setText(
                    f"<span style='color:{color}; font-weight:bold;'>"
                    f"{delta_ms:.3f} ms – ✗ JITTER HIBA</span>"
                )
            self._lbl_diag_sync_delta.setTextFormat(Qt.TextFormat.RichText)




            self._lbl_diag_det_status.setText(det_str)
            if stats["pos_valid"]:
                self._lbl_diag_pos3d.setText(f"X:{stats['x_3d']:+.1f} Y:{stats['y_3d']:+.1f} Z:{stats['z_3d']:.1f} mm")
            else:
                self._lbl_diag_pos3d.setText("— (Nincs 3D detektálás)")

            # Pipeline késleltetés telemetria
            inf_ms = stats.get("detection_inference_ms", 0.0)
            pipe_ms = stats.get("detection_age_ms", 0.0)
            frame_lag = stats.get("detection_frame_lag", 0)

            def _latency_color(ms: float) -> str:
                if ms < 20.0:
                    return "#4ADE80"  # Zöld
                elif ms < 50.0:
                    return "#FBBF24"  # Sárga
                return "#EF4444"  # Piros

            self._lbl_diag_inference.setText(f"{inf_ms:.1f} ms")
            self._lbl_diag_inference.setStyleSheet(f"color: {_latency_color(inf_ms)}; font-weight: bold;")

            self._lbl_diag_pipeline.setText(f"{pipe_ms:.1f} ms")
            self._lbl_diag_pipeline.setStyleSheet(f"color: {_latency_color(pipe_ms)}; font-weight: bold;")

            lag_color = "#4ADE80" if frame_lag <= 1 else ("#FBBF24" if frame_lag <= 3 else "#EF4444")
            self._lbl_diag_frame_lag.setText(f"{frame_lag} frame")
            self._lbl_diag_frame_lag.setStyleSheet(f"color: {lag_color}; font-weight: bold;")

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
        self._set_shot_arm_button(False, False)
        if hasattr(self, "_goal_view") and self._goal_view:
            self._goal_view.cancel_active_shot()
        if hasattr(self, "_goal_view_full") and self._goal_view_full:
            self._goal_view_full.cancel_active_shot()

        # Visszaállítjuk az alapértelmezett GroupBox címeket
        if hasattr(self, "_left_grp") and self._left_grp:
            self._left_grp.setTitle("Bal Kamera  [X = -1150 mm]")
        if hasattr(self, "_right_grp") and self._right_grp:
            self._right_grp.setTitle("Jobb Kamera  [X = +1150 mm]")
        if hasattr(self, "_left_grp_full") and self._left_grp_full:
            self._left_grp_full.setTitle("Bal Kamera — Nagyfelbontású Élő Videófolyam  [X = -1150 mm]")
        if hasattr(self, "_right_grp_full") and self._right_grp_full:
            self._right_grp_full.setTitle("Jobb Kamera — Nagyfelbontású Élő Videófolyam  [X = +1150 mm]")

    @pyqtSlot(str)
    def _on_log_message(self, msg: str) -> None:
        self._log_panel.appendPlainText(msg)
        cursor = self._log_panel.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        self._log_panel.setTextCursor(cursor)

    def _toggle_theme(self) -> None:
        """Vált a világos (DEIK Fehér) és sötét (Dark Mode) téma között."""
        self._is_dark_theme = not getattr(self, "_is_dark_theme", False)
        self._apply_theme_to_ui()

    def _apply_theme_to_ui(self) -> None:
        """Alkalmazza az aktuális (sötét vagy világos) témát az összes felületi elemre."""
        dark = getattr(self, "_is_dark_theme", False)

        # 1. QSS Alkalmazás a főablakra
        self.setStyleSheet(DARK_DEIK_QSS if dark else LIGHT_DEIK_QSS)

        # 2. Fejléc elemek
        if hasattr(self, "_title_lbl"):
            self._title_lbl.setStyleSheet("font-weight: 800; font-size: 15px; color: #4ADE80;" if dark else "font-weight: 800; font-size: 15px; color: #0F5132;")
        if hasattr(self, "_sub_lbl"):
            self._sub_lbl.setStyleSheet("font-size: 11px; color: #F59E0B; font-weight: 600;" if dark else "font-size: 11px; color: #D97706; font-weight: 600;")
        if hasattr(self, "_btn_about"):
            self._btn_about.setStyleSheet("font-weight: 700; color: #4ADE80;" if dark else "font-weight: 700; color: #0F5132;")

        # Theme toggle button text and style
        if hasattr(self, "_btn_theme_toggle"):
            if dark:
                self._btn_theme_toggle.setText("☀️ Világos Mód")
                self._btn_theme_toggle.setStyleSheet(
                    "QPushButton { font-weight: 800; color: #F8FAFC; background-color: #151D2A; "
                    "border-radius: 6px; font-size: 12px; border: 1px solid #26334D; padding: 0 10px; }"
                    "QPushButton:hover { background-color: #1E293B; border-color: #10B981; color: #10B981; }"
                )
            else:
                self._btn_theme_toggle.setText("🌙 Sötét Mód")
                self._btn_theme_toggle.setStyleSheet(
                    "QPushButton { font-weight: 800; color: #334155; background-color: #F1F5F9; "
                    "border-radius: 6px; font-size: 12px; border: 1px solid #CBD5E1; padding: 0 10px; }"
                    "QPushButton:hover { background-color: #E2E8F0; border-color: #0F5132; color: #0F5132; }"
                )

        # 3. Nézetváltó és Dock Navigációs gombok
        self._update_view_btn_styles()
        self._update_dock_nav_styles()

        # 4. Telemetriai kártyák
        for w, t_lbl, val_lbl, default_color in getattr(self, "_telemetry_cards", []):
            if dark:
                w.setStyleSheet("background-color: #151D2A; border: 1px solid #26334D; border-radius: 6px; padding: 6px;")
                t_lbl.setStyleSheet("font-size: 10px; color: #94A3B8; font-weight: bold; background: transparent;")
            else:
                w.setStyleSheet("background-color: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; padding: 6px;")
                t_lbl.setStyleSheet("font-size: 10px; color: #475569; font-weight: bold; background: transparent;")

        # 6. GoalView, Analytics and Actuator widgets theme propagation
        if hasattr(self, "_goal_view") and self._goal_view:
            self._goal_view.set_dark(dark)
        if hasattr(self, "_goal_view_full") and self._goal_view_full:
            self._goal_view_full.set_dark(dark)
        if hasattr(self, "_analytics_view") and self._analytics_view:
            self._analytics_view.set_dark(dark)
        if hasattr(self, "_actuator_view") and self._actuator_view:
            self._actuator_view.set_dark(dark)

        # 7. Kamera Placeholderek
        cam_ph_style = (
            "background-color: #151D2A; color: #94A3B8; border: 1px solid #26334D; border-radius: 6px;"
            if dark else
            "background-color: #F1F5F9; color: #475569; border: 1px solid #CBD5E1; border-radius: 6px;"
        )
        for attr in ["_cam_label_left", "_cam_label_right", "_cam_label_left_full", "_cam_label_right_full"]:
            lbl = getattr(self, attr, None)
            if lbl:
                lbl.setStyleSheet(cam_ph_style)

        # 8. Törlés és Alapértelmezett Gombok
        btn_clear_style = (
            "QPushButton { background-color: #151D2A; color: #94A3B8; font-weight: 700; border-radius: 6px; font-size: 11px; border: 1px solid #26334D; }"
            "QPushButton:hover { background-color: #450A0A; color: #FCA5A5; border-color: #EF4444; }"
            if dark else
            "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 6px; font-size: 11px; border: 1px solid #CBD5E1; }"
            "QPushButton:hover { background-color: #FEE2E2; color: #DC2626; border-color: #EF4444; }"
        )
        if hasattr(self, "_btn_clear") and self._btn_clear:
            self._btn_clear.setStyleSheet(btn_clear_style)
        if hasattr(self, "_btn_clear_full") and self._btn_clear_full:
            self._btn_clear_full.setStyleSheet(btn_clear_style)

        btn_reset_style = (
            "QPushButton { background-color: #151D2A; color: #94A3B8; font-weight: 700; border-radius: 6px; font-size: 12px; border: 1px solid #26334D; }"
            "QPushButton:hover { background-color: #1E293B; color: #4ADE80; border-color: #4ADE80; }"
            if dark else
            "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; border-radius: 6px; font-size: 12px; border: 1px solid #CBD5E1; }"
            "QPushButton:hover { background-color: #E2E8F0; color: #0F5132; }"
        )
        for b in getattr(self, "_btn_resets", []):
            b.setStyleSheet(btn_reset_style)

        # 9. Kapu vizualizátorok dark állapota
        if hasattr(self, "_goal_view") and self._goal_view:
            self._goal_view.set_dark(dark)
        if hasattr(self, "_goal_view_full") and self._goal_view_full:
            self._goal_view_full.set_dark(dark)

        # 10. Diagnosztikai címkék
        if hasattr(self, "_lbl_diag_calib") and self._lbl_diag_calib:
            self._lbl_diag_calib.setStyleSheet("color: #4ADE80; font-weight: bold;" if dark else "color: #0F5132; font-weight: bold;")
        if hasattr(self, "_lbl_diag_gpu") and self._lbl_diag_gpu:
            self._lbl_diag_gpu.setStyleSheet("font-weight: bold; color: #4ADE80;" if dark else "font-weight: bold; color: #0F5132;")
        if hasattr(self, "_creators_lbl") and self._creators_lbl:
            self._creators_lbl.setStyleSheet("color: #4ADE80; font-weight: 700; font-size: 11px; padding-right: 8px;" if dark else "color: #0F5132; font-weight: 700; font-size: 11px; padding-right: 8px;")

        # 11. Státusz pill-ek
        self._update_system_status()

    @pyqtSlot(float, float, float)
    def _on_actuator_manual_pos(self, x_mm: float, y_mm: float, speed_m_s: float) -> None:
        # Időbélyeg és célpozíció tárolása a reakcióidő számításához
        self._last_goalkeeper_cmd_time = time.perf_counter()
        self._last_goalkeeper_x_mm = x_mm
        self._last_goalkeeper_y_mm = y_mm
        self._last_goalkeeper_speed_m_s = speed_m_s

        # Kézi küldés esetén is loggoljuk
        if self._shot_confirm_time > 0:
            reaction_ms = (self._last_goalkeeper_cmd_time - self._shot_confirm_time) * 1000.0
            logger.info(
                "Kapus kézi parancs: X=%.0f mm, Y=%.0f mm | Reakcióidő: %.1f ms",
                x_mm, y_mm, reaction_ms
            )
        else:
            logger.info("Kapus kézi parancs: X=%.0f mm, Y=%.0f mm (lövés nélkül)", x_mm, y_mm)

        if hasattr(self, "_goal_view") and self._goal_view:
            self._goal_view.set_goalkeeper_target(x_mm, y_mm)
        if hasattr(self, "_goal_view_full") and self._goal_view_full:
            self._goal_view_full.set_goalkeeper_target(x_mm, y_mm)

    @pyqtSlot()
    def _update_system_status(self) -> None:
        """Másodpercenként frissíti a hardveres CPU, RAM és GPU erőforrás telemetriát."""
        dark = getattr(self, "_is_dark_theme", False)
        try:
            cpu_p = psutil.cpu_percent()
            if hasattr(self, "_lbl_diag_cpu"):
                self._lbl_diag_cpu.setText(f"{cpu_p:.1f} %")
            if hasattr(self, "_pill_cpu"):
                self._pill_cpu.setText(f" CPU: {cpu_p:.0f}% ")
                self._pill_cpu.setStyleSheet(get_hw_pill_style(usage_level(cpu_p), dark))

            mem = psutil.virtual_memory()
            mem_mb = mem.used / (1024 * 1024)
            mem_tot = mem.total / (1024 * 1024)
            if hasattr(self, "_lbl_diag_ram"):
                self._lbl_diag_ram.setText(f"{mem_mb:.0f} MB / {mem_tot:.0f} MB ({mem.percent:.1f} %)")
            if hasattr(self, "_pill_ram"):
                self._pill_ram.setText(f" RAM: {mem.percent:.0f}% ")
                self._pill_ram.setStyleSheet(get_hw_pill_style(usage_level(mem.percent), dark))

            # GPU Telemetria NVML-lel
            if _nvml_available and _nvml_handle:
                util = pynvml.nvmlDeviceGetUtilizationRates(_nvml_handle)
                temp = pynvml.nvmlDeviceGetTemperature(_nvml_handle, pynvml.NVML_TEMPERATURE_GPU)
                gpu_p = util.gpu
                if hasattr(self, "_pill_gpu_usage"):
                    self._pill_gpu_usage.setText(f" GPU: {gpu_p}% ")
                    self._pill_gpu_usage.setStyleSheet(get_hw_pill_style(usage_level(gpu_p), dark))
                if hasattr(self, "_pill_gpu_temp"):
                    self._pill_gpu_temp.setText(f" {temp}°C ")
                    temp_lvl = "low" if temp < 70 else ("medium" if temp < 85 else "high")
                    self._pill_gpu_temp.setStyleSheet(get_hw_pill_style(temp_lvl, dark))
                if hasattr(self, "_lbl_diag_gpu"):
                    self._lbl_diag_gpu.setText(f"NVIDIA RTX 3050 ({gpu_p}% util · {temp}°C)")
        except Exception as e:
            logger.debug("Hardware stats update error: %s", e)

    def _sync_config_from_ui(self) -> None:
        """Frissíti a belső self._config szótárt a főoldali GUI vezérlők aktuális értékeivel."""
        for side in ["left", "right"]:
            if hasattr(self, "_cam_widgets") and side in self._cam_widgets:
                w = self._cam_widgets[side]
                rot_index = w["combo_rot"].currentIndex()
                rot_val = [0, 90, 180, 270][rot_index] if rot_index < 4 else 0

                cam_dict = self._config.setdefault("camera", {}).setdefault(side, {})
                cam_dict["offset_x"] = w["spin_x"].value()
                cam_dict["offset_y"] = w["spin_y"].value()
                cam_dict["roi_enabled"] = w["chk_roi"].isChecked()
                cam_dict["roi_x_min"] = w["spin_xmin"].value()
                cam_dict["roi_x_max"] = w["spin_xmax"].value()
                cam_dict["roi_y_min"] = w["spin_ymin"].value()
                cam_dict["roi_y_max"] = w["spin_ymax"].value()
                cam_dict["roi_zoom"] = w["chk_roi_zoom"].isChecked()
                cam_dict["exposure_time_us"] = w["spin_exp"].value()
                cam_dict["gain_db"] = w["spin_gain"].value()
                cam_dict["flip_h"] = w["chk_fliph"].isChecked()
                cam_dict["flip_v"] = w["chk_flipv"].isChecked()
                cam_dict["rotation"] = rot_val

    def _load_gui_settings(self) -> None:
        path = "config/gui_settings.json"
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    gui_cfg = json.load(f)
                if "theme" in gui_cfg:
                    self._is_dark_theme = (gui_cfg["theme"] == "dark")
                for side in ["left", "right"]:
                    if side in gui_cfg:
                        if side not in self._config["camera"]:
                            self._config["camera"][side] = {}
                        self._config["camera"][side].update(gui_cfg[side])
            except Exception as e:
                logger.error("GUI beállítások betöltése sikertelen: %s", e)

    @pyqtSlot()
    def _save_gui_settings(self) -> None:
        self._sync_config_from_ui()
        gui_cfg = {}
        gui_cfg["theme"] = "dark" if getattr(self, "_is_dark_theme", False) else "light"
        for side in ["left", "right"]:
            if side in self._config["camera"]:
                gui_cfg[side] = self._config["camera"][side]

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
                    w["combo_rot"].setCurrentIndex(1)

    def _start_tracker(self) -> None:
        logger.info("Tracker indítása...")
        self._last_handled_shot_sequence = -1

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
        self._set_shot_arm_button(False, False)

    def _stop_tracker(self) -> None:
        if self._worker:
            self._worker.disarm_shot()
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
