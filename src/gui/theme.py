"""
DEIK Robot Foci Kapus – Fehér / Világos DEIK Arculati Témacsomag & QSS
======================================================================

Ez a modul a Debreceni Egyetem Informatikai Karának (DEIK) hivatalos arculati
színeire épülő, letisztult, fehér / világos minimalista téma QSS stíluslapját biztosítja.
"""

COLOR_DEIK_GREEN = "#0F5132"
COLOR_DEIK_GREEN_LIGHT = "#146C43"
COLOR_DEIK_GOLD = "#D97706"
COLOR_DEIK_GOLD_LIGHT = "#F59E0B"

COLOR_BG_LIGHT = "#F8FAFC"
COLOR_CARD_BG = "#FFFFFF"
COLOR_PANEL_ALT = "#F1F5F9"
COLOR_BORDER = "#CBD5E1"
COLOR_BORDER_DARK = "#94A3B8"

COLOR_TEXT_MAIN = "#0F172A"
COLOR_TEXT_MUTED = "#334155"

LIGHT_DEIK_QSS = f"""
/* ===================================================================
   DEIK ROBOT KAPUS – FEHÉR MINIMALISTA DEIK QSS
   =================================================================== */

QMainWindow, QDialog, QWidget#centralWidget, QDockWidget, QDockWidget > QWidget {{
    background-color: {COLOR_BG_LIGHT};
    color: {COLOR_TEXT_MAIN};
    font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
}}

QWidget {{
    color: {COLOR_TEXT_MAIN};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}}

QMainWindow::separator {{
    background-color: {COLOR_BG_LIGHT};
    width: 6px;
    height: 6px;
}}

/* ── Fejléc & ToolBar ────────────────────────────────────────────── */
QToolBar {{
    background-color: {COLOR_CARD_BG};
    border-bottom: 3px solid {COLOR_DEIK_GREEN};
    padding: 8px 16px;
    spacing: 12px;
}}

/* ── GroupBox (Kártya konténerek) ───────────────────────────────── */
QGroupBox {{
    background-color: {COLOR_CARD_BG};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    margin-top: 22px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    top: 2px;
    padding: 4px 12px;
    background-color: {COLOR_DEIK_GREEN};
    color: #FFFFFF;
    border-radius: 4px;
    font-size: 12px;
    font-weight: 700;
}}

/* ── DockWidget (Kamera Beállítások & Rendszernapló Fejlécek) ────── */
QDockWidget {{
    font-weight: bold;
    color: #FFFFFF;
    background-color: {COLOR_BG_LIGHT};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
}}

QDockWidget::title {{
    text-align: left;
    background-color: {COLOR_DEIK_GREEN};
    color: #FFFFFF;
    padding: 8px 12px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    font-weight: 700;
    font-size: 13px;
}}

/* ── TabWidget (Lapfülek) ────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    background-color: {COLOR_CARD_BG};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {COLOR_PANEL_ALT};
    color: #0F172A;
    border: 1px solid {COLOR_BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    margin-right: 2px;
    font-weight: 700;
}}

QTabBar::tab:selected {{
    background-color: {COLOR_CARD_BG};
    color: {COLOR_DEIK_GREEN};
    border-top: 3px solid {COLOR_DEIK_GREEN};
}}

QTabBar::tab:hover:!selected {{
    background-color: #E2E8F0;
    color: {COLOR_DEIK_GREEN};
}}

/* ── Gombok (PushButtons) ───────────────────────────────────────── */
QPushButton {{
    background-color: {COLOR_CARD_BG};
    color: {COLOR_TEXT_MAIN};
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 13px;
}}

QPushButton:hover {{
    background-color: {COLOR_PANEL_ALT};
    border-color: {COLOR_DEIK_GREEN};
    color: {COLOR_DEIK_GREEN};
}}

QPushButton:pressed {{
    background-color: #E2E8F0;
}}

QPushButton#btn_start {{
    background-color: {COLOR_DEIK_GREEN};
    color: #FFFFFF;
    border: 1px solid {COLOR_DEIK_GREEN_LIGHT};
    font-weight: 700;
    font-size: 13px;
}}

QPushButton#btn_start:hover {{
    background-color: {COLOR_DEIK_GREEN_LIGHT};
    border-color: {COLOR_DEIK_GOLD};
}}

QPushButton#btn_stop {{
    background-color: #DC2626;
    color: #FFFFFF;
    border: 1px solid #B91C1C;
    font-weight: 700;
    font-size: 13px;
}}

QPushButton#btn_stop:hover {{
    background-color: #B91C1C;
}}

/* ── Csúszkák (QSlider) ─────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 6px;
    background: #E2E8F0;
    border: 1px solid {COLOR_BORDER};
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {COLOR_DEIK_GREEN};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLOR_DEIK_GOLD};
    border: 2px solid #FFFFFF;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

/* ── SpinBox, DoubleSpinBox & ComboBox ───────────────────────────── */
QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {COLOR_CARD_BG};
    color: {COLOR_TEXT_MAIN};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    padding: 4px 6px;
    font-weight: 600;
}}

QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover {{
    border-color: {COLOR_DEIK_GREEN};
}}

/* ── Log Konzol ─────────────────────────────────────────────────── */
QPlainTextEdit#log_console {{
    background-color: #F1F5F9;
    color: #0F172A;
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 6px;
    font-weight: bold;
}}

/* ── Státusz sor ────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {COLOR_CARD_BG};
    border-top: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT_MUTED};
    font-size: 12px;
    padding: 4px 8px;
}}
"""


def get_status_pill_style(state: str) -> str:
    if state == "ok":
        bg, text, border = "#D1FAE5", "#065F46", "#10B981"
    elif state == "warning":
        bg, text, border = "#FEF3C7", "#92400E", "#F59E0B"
    elif state == "error":
        bg, text, border = "#FEE2E2", "#991B1B", "#EF4444"
    else:  # info
        bg, text, border = "#E2E8F0", "#334155", "#94A3B8"

    return (
        f"background-color: {bg}; "
        f"color: {text}; "
        f"border: 1px solid {border}; "
        f"border-radius: 10px; "
        f"padding: 3px 10px; "
        f"font-weight: 700; "
        f"font-size: 11px;"
    )


def get_app_icon():
    """
    Létrehoz egy több felbontású (16, 24, 32, 48, 64, 128, 256, 512 px) QIcon példányt,
    amely garantálja, hogy a Linux tálca (GNOME/KDE/XFCE dock) és a címsorok
    azonnal és hibátlanul megjelenítik a beállított azonosító ikont.
    """
    import os
    # pyrefly: ignore [missing-import]
    # type: ignore
    from PyQt6.QtCore import Qt
    # pyrefly: ignore [missing-import]
    from PyQt6.QtGui import QIcon, QPixmap

    icon = QIcon()
    logo_path = os.path.abspath("assets/logo.png")
    if os.path.exists(logo_path):
        base_pix = QPixmap(logo_path)
        if not base_pix.isNull():
            for sz in [16, 24, 32, 48, 64, 96, 128, 256, 512]:
                scaled_pix = base_pix.scaled(
                    sz, sz,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                icon.addPixmap(scaled_pix)
    return icon
