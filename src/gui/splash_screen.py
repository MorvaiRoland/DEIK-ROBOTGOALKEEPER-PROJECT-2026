"""
DEIK Robot Foci Kapus – Betöltő Képernyő (SplashScreen)
=======================================================

Ez a modul a szoftver indításakor megjelenő, professzionális betöltő animációs
ablakot (SplashScreen) valósítja meg a két hivatalos logóval (DEIK Címer + RGK System),
progress barral és lépcsőzetes inicializálással.
"""

import os
import time
from typing import Optional

# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtCore import QPoint, QRectF, QTimer, Qt
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import QColor, QFont, QIcon, QPixmap
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout, QWidget
)


from gui.theme import get_app_icon


class SplashScreen(QWidget):
    """
    Keret nélküli, animált betöltő ablak a két logóval és állapotjelzővel.
    """

    def __init__(self, parent: Optional[QWidget] = None):
        super().__init__(parent, Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint | Qt.WindowType.SplashScreen)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        app_icon = get_app_icon()
        if not app_icon.isNull():
            self.setWindowIcon(app_icon)

        self.setFixedSize(560, 410)
        self._center_on_screen()

        self._setup_ui()

    def _center_on_screen(self) -> None:
        screen = QApplication.primaryScreen()
        if screen:
            geo = screen.availableGeometry()
            x = (geo.width() - self.width()) // 2
            y = (geo.height() - self.height()) // 2
            self.move(x, y)

    def _setup_ui(self) -> None:
        self.setStyleSheet("""
            QWidget#splash_bg {
                background-color: #FFFFFF;
                border: 2px solid #0F5132;
                border-radius: 12px;
            }
        """)
        self.setObjectName("splash_bg")

        main_box = QVBoxLayout(self)
        main_box.setContentsMargins(24, 24, 24, 20)
        main_box.setSpacing(12)

        # 1. LOGÓK SZEKCIÓ: Két logó egymás mellett
        logo_box = QHBoxLayout()
        logo_box.setSpacing(24)
        logo_box.setAlignment(Qt.AlignmentFlag.AlignCenter)

        # DEIK Címer Logó
        deik_logo_path = "assets/deik_logo.png"
        if os.path.exists(deik_logo_path):
            l1 = QLabel()
            pix1 = QPixmap(deik_logo_path).scaledToHeight(95, Qt.TransformationMode.SmoothTransformation)
            l1.setPixmap(pix1)
            logo_box.addWidget(l1)

        # RGK System Hexagon Shield Logó
        rgk_logo_path = "assets/logo.png"
        if os.path.exists(rgk_logo_path):
            l2 = QLabel()
            pix2 = QPixmap(rgk_logo_path).scaledToHeight(95, Qt.TransformationMode.SmoothTransformation)
            l2.setPixmap(pix2)
            logo_box.addWidget(l2)

        main_box.addLayout(logo_box)

        # 2. CÍMSOR SZEKCIÓ
        title_lbl = QLabel("DEIK ROBOT FOCI KAPUS")
        title_lbl.setStyleSheet("font-weight: 800; font-size: 20px; color: #0F5132;")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_box.addWidget(title_lbl)

        sub_lbl = QLabel("Debreceni Egyetem Informatikai Kar  |  RGK System")
        sub_lbl.setStyleSheet("font-weight: 700; font-size: 12px; color: #D97706;")
        sub_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_box.addWidget(sub_lbl)

        ver_lbl = QLabel("Verzió 1.0.0 (2026)")
        ver_lbl.setStyleSheet("background-color: #F1F5F9; color: #475569; font-weight: 700; font-size: 11px; border-radius: 8px; padding: 2px 12px;")
        ver_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_box.addWidget(ver_lbl)

        main_box.addSpacing(6)

        # 3. KORDA / PROGRESS BAR SZEKCIÓ
        self._status_lbl = QLabel("Rendszer inicializálása...")
        self._status_lbl.setStyleSheet("color: #334155; font-weight: 600; font-size: 12px;")
        self._status_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_box.addWidget(self._status_lbl)

        self._progress_bar = QProgressBar()
        self._progress_bar.setFixedHeight(8)
        self._progress_bar.setTextVisible(False)
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setValue(0)
        self._progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #E2E8F0;
                border-radius: 4px;
                border: none;
            }
            QProgressBar::chunk {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #0F5132, stop:1 #146C43);
                border-radius: 4px;
            }
        """)
        main_box.addWidget(self._progress_bar)

        # 4. KÉSZÍTŐK FOOTER
        dev_lbl = QLabel("Fejlesztők: Morvai Roland & Rácz Donát (BSc Mérnökinformatikus)")
        dev_lbl.setStyleSheet("color: #64748B; font-weight: 600; font-size: 11px;")
        dev_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_box.addWidget(dev_lbl)

    def set_progress(self, percent: int, status_text: str) -> None:
        """Frissíti a betöltési százalékot és a státuszüzenetet."""
        self._progress_bar.setValue(percent)
        self._status_lbl.setText(status_text)
        QApplication.processEvents()

    def run_loading_sequence(self) -> None:
        """Végrehajtja a látványos lépcsőzetes indítási animációt."""
        steps = [
            (15, "Konfigurációs fájlok és architektúra betöltése..."),
            (35, "Kamera illesztőprogramok (Ximea CMOS) ellenőrzése..."),
            (55, "YOLOv10 AI detektor inicializálása CUDA GPU-n..."),
            (75, "Sztereó optikai kalibráció és Kalman-szűrő beállítása..."),
            (95, "DEIK Grafikus felület felépítése..."),
            (100, "DEIK Robot Foci Kapus készen áll!"),
        ]

        for percent, text in steps:
            self.set_progress(percent, text)
            time.sleep(0.35)
