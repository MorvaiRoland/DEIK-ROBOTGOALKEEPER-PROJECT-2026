"""
DEIK Robot Foci Kapus – Aktuátor Vezérlő & Tesztelő Panel (PyQt6)
==================================================================

Kézi vezérlő és teszt panel a szervo mozgató mechanika teszteléséhez:
  - Preset mozgások (Balra, Jobbra, Középre, Felső sarok)
  - Sürgősségi leállítás (E-STOP)
  - Pozíció csúszkák (-2000 mm .. +2000 mm)
  - Sebesség korlátozó csúszka (0.5 m/s .. 5.0 m/s)
  - Téma-tudatos (Dark Mode & Light Mode)
"""

import logging
from typing import Optional

# pyrefly: ignore [missing-import]
from PyQt6.QtCore import Qt, pyqtSignal, pyqtSlot
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QPushButton, QSlider, QSpinBox, QDoubleSpinBox,
    QProgressBar, QMessageBox, QFrame, QScrollArea
)

logger = logging.getLogger(__name__)


class ActuatorControlWidget(QWidget):
    """
    Hardveres szervo mozgató teszt panel manuális felülbírálással és E-STOP funkcióval.
    """

    position_changed = pyqtSignal(float, float, float)  # (x_mm, y_mm, speed_m_s)
    estop_triggered  = pyqtSignal()

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self._dark   = False

        geo_cfg = config.get("geometry", {})
        self._max_x_mm = float(geo_cfg.get("goal_width_mm", 4000.0)) / 2.0
        self._max_y_mm = float(geo_cfg.get("goal_height_mm", 2000.0))

        self._current_x = 0.0
        self._current_y = 1000.0
        self._target_x  = 0.0
        self._target_y  = 1000.0
        self._speed_m_s = 2.5
        self._estop     = False

        self._build_ui()

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self._apply_theme()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # 1. Fejléc & Státusz sáv
        hdr_grp = QGroupBox("Aktuátor Kommunikáció & Hardver Státusz")
        hdr_box = QVBoxLayout(hdr_grp)
        hdr_box.setSpacing(8)

        status_row = QHBoxLayout()
        self._lbl_comm_status = QLabel("UDP PROTOKOLL AKTÍV (Szimulált Aktuátor)")
        self._lbl_comm_status.setStyleSheet(
            "background: #064E3B; color: #4ADE80; font-weight: 800; "
            "border: 1px solid #10B981; border-radius: 6px; padding: 6px 12px; font-size: 12px;"
        )

        self._lbl_pos_readout = QLabel("Pozíció: X = 0 mm | Y = 1000 mm")
        self._lbl_pos_readout.setStyleSheet(
            "font-weight: 800; font-size: 13px; font-family: Consolas, monospace; "
            "color: #10B981; padding: 4px 10px;"
        )

        status_row.addWidget(self._lbl_comm_status, stretch=1)
        status_row.addWidget(self._lbl_pos_readout)
        hdr_box.addLayout(status_row)

        main_layout.addWidget(hdr_grp)

        # 2. Sürgősségi Leállítás (E-STOP) Gomb
        estop_grp = QGroupBox("Sürgősségi Vezérlés")
        estop_box = QHBoxLayout(estop_grp)

        self._btn_estop = QPushButton("SÜRGŐSSÉGI LEÁLLÍTÁS (E-STOP)")
        self._btn_estop.setFixedHeight(48)
        self._btn_estop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_estop.setStyleSheet(
            "QPushButton { background-color: #DC2626; color: #FFFFFF; font-weight: 900; "
            "font-size: 15px; border-radius: 8px; border: 2px solid #991B1B; letter-spacing: 1px; }"
            "QPushButton:hover { background-color: #B91C1C; }"
            "QPushButton:pressed { background-color: #7F1D1D; }"
        )
        self._btn_estop.clicked.connect(self._on_estop_click)

        self._btn_reset_estop = QPushButton("Leállítás Feloldása")
        self._btn_reset_estop.setFixedHeight(48)
        self._btn_reset_estop.setCursor(Qt.CursorShape.PointingHandCursor)
        self._btn_reset_estop.setStyleSheet(
            "QPushButton { background-color: #1E293B; color: #94A3B8; font-weight: 700; "
            "font-size: 13px; border-radius: 8px; border: 1px solid #334155; }"
            "QPushButton:hover { background-color: #334155; color: #F8FAFC; }"
        )
        self._btn_reset_estop.clicked.connect(self._on_reset_estop)

        estop_box.addWidget(self._btn_estop, stretch=3)
        estop_box.addWidget(self._btn_reset_estop, stretch=1)
        main_layout.addWidget(estop_grp)

        # 3. Preset Teszt Mozgások
        preset_grp = QGroupBox("Gyors Teszt Pozíciók (Preset Actions)")
        preset_grid = QHBoxLayout(preset_grp)
        preset_grid.setSpacing(8)

        btn_left = QPushButton("Balra Vetődés\n(-1500 mm)")
        btn_center = QPushButton("Alaphelyzet\n(0 mm)")
        btn_right = QPushButton("Jobbra Vetődés\n(+1500 mm)")
        btn_top_l = QPushButton("Bal Felső\n(-1400, 1600 mm)")
        btn_top_r = QPushButton("Jobb Felső\n(+1400, 1600 mm)")

        for b in [btn_left, btn_center, btn_right, btn_top_l, btn_top_r]:
            b.setFixedHeight(46)
            b.setCursor(Qt.CursorShape.PointingHandCursor)
            b.setStyleSheet(
                "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; "
                "font-size: 11px; border-radius: 6px; border: none; }"
                "QPushButton:hover { background-color: #146C43; }"
            )

        btn_left.clicked.connect(lambda: self._set_preset(-1500.0, 800.0))
        btn_center.clicked.connect(lambda: self._set_preset(0.0, 1000.0))
        btn_right.clicked.connect(lambda: self._set_preset(1500.0, 800.0))
        btn_top_l.clicked.connect(lambda: self._set_preset(-1400.0, 1600.0))
        btn_top_r.clicked.connect(lambda: self._set_preset(1400.0, 1600.0))

        preset_grid.addWidget(btn_left)
        preset_grid.addWidget(btn_center)
        preset_grid.addWidget(btn_right)
        preset_grid.addWidget(btn_top_l)
        preset_grid.addWidget(btn_top_r)

        main_layout.addWidget(preset_grp)

        # 4. Manuális Csúszka & Finomhangolás
        manual_grp = QGroupBox("Manuális Pozíció & Sebesség Hangolás")
        manual_form = QFormLayout(manual_grp)
        manual_form.setSpacing(12)

        # X Csúszka (-2000 mm .. +2000 mm)
        self._spin_x = QSpinBox()
        self._spin_x.setRange(int(-self._max_x_mm), int(self._max_x_mm))
        self._spin_x.setValue(0)
        self._spin_x.setSuffix(" mm")
        self._spin_x.setSingleStep(50)

        self._slider_x = QSlider(Qt.Orientation.Horizontal)
        self._slider_x.setRange(int(-self._max_x_mm), int(self._max_x_mm))
        self._slider_x.setValue(0)
        self._slider_x.valueChanged.connect(self._spin_x.setValue)
        self._spin_x.valueChanged.connect(self._slider_x.setValue)

        hx = QHBoxLayout()
        hx.addWidget(self._slider_x, stretch=3)
        hx.addWidget(self._spin_x, stretch=1)
        manual_form.addRow("Vízszintes X:", hx)

        # Y Csúszka (200 mm .. 1800 mm)
        self._spin_y = QSpinBox()
        self._spin_y.setRange(200, int(self._max_y_mm))
        self._spin_y.setValue(1000)
        self._spin_y.setSuffix(" mm")
        self._spin_y.setSingleStep(50)

        self._slider_y = QSlider(Qt.Orientation.Horizontal)
        self._slider_y.setRange(200, int(self._max_y_mm))
        self._slider_y.setValue(1000)
        self._slider_y.valueChanged.connect(self._spin_y.setValue)
        self._spin_y.valueChanged.connect(self._slider_y.setValue)

        hy = QHBoxLayout()
        hy.addWidget(self._slider_y, stretch=3)
        hy.addWidget(self._spin_y, stretch=1)
        manual_form.addRow("Magasság Y:", hy)

        # Sebesség (0.5 m/s .. 5.0 m/s)
        self._spin_speed = QDoubleSpinBox()
        self._spin_speed.setRange(0.5, 5.0)
        self._spin_speed.setSingleStep(0.2)
        self._spin_speed.setValue(2.5)
        self._spin_speed.setSuffix(" m/s")
        manual_form.addRow("Mozgatási Sebesség:", self._spin_speed)

        # Mozgatás Parancs Küldése Gomb
        btn_send = QPushButton("Pozíció Parancs Küldése az Aktuátornak")
        btn_send.setFixedHeight(38)
        btn_send.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_send.setStyleSheet(
            "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; "
            "font-size: 13px; border-radius: 6px; border: none; }"
            "QPushButton:hover { background-color: #146C43; }"
        )
        btn_send.clicked.connect(self._on_send_click)

        manual_form.addRow("", btn_send)
        main_layout.addWidget(manual_grp)

        main_layout.addStretch(1)
        self._apply_theme()

    def _set_preset(self, x_mm: float, y_mm: float) -> None:
        if self._estop:
            QMessageBox.warning(self, "E-STOP Aktív", "A leállítás feloldásáig nem küldhető pozíció parancs!")
            return
        self._spin_x.setValue(int(x_mm))
        self._spin_y.setValue(int(y_mm))
        self._on_send_click()

    @pyqtSlot()
    def _on_send_click(self) -> None:
        if self._estop:
            QMessageBox.warning(self, "E-STOP Aktív", "A leállítás feloldásáig nem küldhető pozíció parancs!")
            return
        x = float(self._spin_x.value())
        y = float(self._spin_y.value())
        spd = float(self._spin_speed.value())
        self._target_x = x
        self._target_y = y
        self._speed_m_s = spd

        self._lbl_pos_readout.setText(f"Pozíció: X = {x:+.0f} mm | Y = {y:.0f} mm ({spd:.1f} m/s)")
        logger.info("Aktuátor parancs elküldve: X=%.1f, Y=%.1f, v=%.1f m/s", x, y, spd)
        self.position_changed.emit(x, y, spd)

    @pyqtSlot()
    def _on_estop_click(self) -> None:
        self._estop = True
        self._lbl_comm_status.setText("SÜRGŐSSÉGI LEÁLLÍTÁS (E-STOP) AKTÍV!")
        self._lbl_comm_status.setStyleSheet(
            "background: #450A0A; color: #FCA5A5; font-weight: 900; "
            "border: 1px solid #EF4444; border-radius: 6px; padding: 6px 12px; font-size: 12px;"
        )
        logger.critical("E-STOP TRIGGERED ON ACTUATOR CONTROL PANEL!")
        self.estop_triggered.emit()

    @pyqtSlot()
    def _on_reset_estop(self) -> None:
        self._estop = False
        self._lbl_comm_status.setText("UDP PROTOKOLL AKTÍV (Szimulált Aktuátor)")
        self._lbl_comm_status.setStyleSheet(
            "background: #064E3B; color: #4ADE80; font-weight: 800; "
            "border: 1px solid #10B981; border-radius: 6px; padding: 6px 12px; font-size: 12px;"
        )
        logger.info("E-STOP reseted.")

    def _apply_theme(self) -> None:
        dark = self._dark
        bg_style = "background-color: #0B0F17; color: #F8FAFC;" if dark else "background-color: #F8FAFC; color: #0F172A;"
        self.setStyleSheet(bg_style)
