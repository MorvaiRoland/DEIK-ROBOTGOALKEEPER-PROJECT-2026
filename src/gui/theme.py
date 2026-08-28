"""
DEIK Robot Foci Kapus – DEIK Arculati Témacsomag & QSS
======================================================================

Két téma:
  LIGHT_DEIK_QSS  – Fehér / Világos (alapértelmezett)
  DARK_DEIK_QSS   – Sötét / Dark mode (csarnoki körülményekhez, pro felület)
"""

# ── Közös Színek ────────────────────────────────────────────────────────────
COLOR_DEIK_GREEN       = "#0F5132"
COLOR_DEIK_GREEN_LIGHT = "#146C43"
COLOR_DEIK_GOLD        = "#D97706"
COLOR_DEIK_GOLD_LIGHT  = "#F59E0B"

# LIGHT Paletta
COLOR_BG_LIGHT   = "#F8FAFC"
COLOR_CARD_BG    = "#FFFFFF"
COLOR_PANEL_ALT  = "#F1F5F9"
COLOR_BORDER     = "#CBD5E1"
COLOR_BORDER_DARK= "#94A3B8"
COLOR_TEXT_MAIN  = "#0F172A"
COLOR_TEXT_MUTED = "#334155"

# DARK Paletta
DARK_BG          = "#0B0F17"
DARK_CARD_BG     = "#151D2A"
DARK_PANEL_ALT   = "#1E293B"
DARK_BORDER      = "#26334D"
DARK_TEXT_MAIN   = "#F8FAFC"
DARK_TEXT_MUTED  = "#CBD5E1"
DARK_EMERALD     = "#10B981"

# ── Light QSS ───────────────────────────────────────────────────────────────
LIGHT_DEIK_QSS = f"""
/* ===================================================================
   DEIK ROBOT KAPUS – FEHÉR MINIMALISTA DEIK QSS
   =================================================================== */

QMainWindow, QDialog, QWidget#centralWidget, QDockWidget, QDockWidget > QWidget, QStackedWidget {{
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
    color: {COLOR_TEXT_MAIN};
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

/* ── DockWidget ──────────────────────────────────────────────────── */
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

/* ── TabWidget ───────────────────────────────────────────────────── */
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

/* ── SpinBox, DoubleSpinBox, QLineEdit & ComboBox ───────────────── */
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QAbstractSpinBox {{
    background-color: {COLOR_CARD_BG};
    color: {COLOR_TEXT_MAIN};
    border: 1px solid {COLOR_BORDER};
    border-radius: 5px;
    padding: 4px 6px;
    font-weight: 600;
}}

QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QAbstractSpinBox QLineEdit {{
    background-color: transparent;
    color: {COLOR_TEXT_MAIN};
    border: none;
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

# ── Dark QSS ────────────────────────────────────────────────────────────────
DARK_DEIK_QSS = f"""
/* ===================================================================
   DEIK ROBOT KAPUS – SÖTÉT DARK MODE QSS (ULTRA-MODERN SLATE)
   =================================================================== */

QMainWindow, QDialog, QWidget#centralWidget, QDockWidget, QDockWidget > QWidget, QStackedWidget {{
    background-color: {DARK_BG};
    color: {DARK_TEXT_MAIN};
    font-family: 'Segoe UI', 'Inter', -apple-system, sans-serif;
}}

QWidget {{
    color: {DARK_TEXT_MAIN};
    font-family: 'Segoe UI', 'Inter', sans-serif;
    font-size: 13px;
}}

QMainWindow::separator {{
    background-color: {DARK_BORDER};
    width: 6px;
    height: 6px;
}}

/* ── Fejléc & ToolBar ────────────────────────────────────────────── */
QToolBar {{
    background-color: {DARK_CARD_BG};
    border-bottom: 3px solid {COLOR_DEIK_GREEN};
    padding: 8px 16px;
    spacing: 12px;
}}

/* ── GroupBox ────────────────────────────────────────────────────── */
QGroupBox {{
    background-color: {DARK_CARD_BG};
    border: 1px solid {DARK_BORDER};
    border-radius: 8px;
    margin-top: 22px;
    padding: 14px 10px 10px 10px;
    font-weight: bold;
    color: {DARK_TEXT_MAIN};
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

/* ── DockWidget ──────────────────────────────────────────────────── */
QDockWidget {{
    font-weight: bold;
    color: {DARK_TEXT_MAIN};
    background-color: {DARK_BG};
    border: 1px solid {DARK_BORDER};
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

/* ── TabWidget ───────────────────────────────────────────────────── */
QTabWidget::pane {{
    border: 1px solid {DARK_BORDER};
    border-radius: 6px;
    background-color: {DARK_CARD_BG};
    top: -1px;
}}

QTabBar::tab {{
    background-color: {DARK_BG};
    color: {DARK_TEXT_MUTED};
    border: 1px solid {DARK_BORDER};
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    padding: 8px 16px;
    margin-right: 2px;
    font-weight: 700;
}}

QTabBar::tab:selected {{
    background-color: {DARK_CARD_BG};
    color: {DARK_EMERALD};
    border-top: 3px solid {COLOR_DEIK_GREEN};
}}

QTabBar::tab:hover:!selected {{
    background-color: {DARK_PANEL_ALT};
    color: {DARK_EMERALD};
}}

/* ── Gombok ─────────────────────────────────────────────────────── */
QPushButton {{
    background-color: {DARK_PANEL_ALT};
    color: {DARK_TEXT_MAIN};
    border: 1px solid {DARK_BORDER};
    border-radius: 6px;
    padding: 7px 14px;
    font-weight: 600;
    font-size: 13px;
}}

QPushButton:hover {{
    background-color: {DARK_BORDER};
    border-color: {DARK_EMERALD};
    color: {DARK_EMERALD};
}}

QPushButton:pressed {{
    background-color: {DARK_BG};
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

/* ── Csúszkák ────────────────────────────────────────────────────── */
QSlider::groove:horizontal {{
    height: 6px;
    background: {DARK_BORDER};
    border: 1px solid {DARK_BORDER};
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: {COLOR_DEIK_GREEN};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLOR_DEIK_GOLD};
    border: 2px solid {DARK_CARD_BG};
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}}

/* ── SpinBox, DoubleSpinBox, QLineEdit & ComboBox ───────────────── */
QSpinBox, QDoubleSpinBox, QLineEdit, QComboBox, QAbstractSpinBox {{
    background-color: {DARK_PANEL_ALT};
    color: {DARK_TEXT_MAIN};
    border: 1px solid {DARK_BORDER};
    border-radius: 5px;
    padding: 4px 6px;
    font-weight: 600;
}}

QSpinBox QLineEdit, QDoubleSpinBox QLineEdit, QAbstractSpinBox QLineEdit {{
    background-color: transparent;
    color: {DARK_TEXT_MAIN};
    border: none;
    selection-background-color: {COLOR_DEIK_GREEN};
    selection-color: #FFFFFF;
}}

QSpinBox::up-button, QSpinBox::down-button,
QDoubleSpinBox::up-button, QDoubleSpinBox::down-button {{
    background-color: {DARK_BORDER};
    border: none;
    width: 16px;
}}

QSpinBox::up-button:hover, QSpinBox::down-button:hover,
QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
    background-color: {DARK_EMERALD};
}}

QSpinBox:hover, QDoubleSpinBox:hover, QComboBox:hover, QLineEdit:hover {{
    border-color: {DARK_EMERALD};
}}

QComboBox QAbstractItemView {{
    background-color: {DARK_CARD_BG};
    color: {DARK_TEXT_MAIN};
    border: 1px solid {DARK_BORDER};
    selection-background-color: {COLOR_DEIK_GREEN};
    selection-color: #FFFFFF;
}}

/* ── Checkbox ────────────────────────────────────────────────────── */
QCheckBox {{
    color: {DARK_TEXT_MAIN};
    spacing: 6px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 2px solid {DARK_BORDER};
    border-radius: 3px;
    background-color: {DARK_BG};
}}

QCheckBox::indicator:checked {{
    background-color: {COLOR_DEIK_GREEN};
    border-color: {COLOR_DEIK_GREEN};
}}

/* ── Log Konzol ─────────────────────────────────────────────────── */
QPlainTextEdit#log_console {{
    background-color: #020617;
    color: {DARK_EMERALD};
    border: 1px solid {DARK_BORDER};
    border-radius: 6px;
    font-family: 'Consolas', 'Courier New', monospace;
    font-size: 12px;
    padding: 6px;
    font-weight: bold;
}}

/* ── Státusz sor ────────────────────────────────────────────────── */
QStatusBar {{
    background-color: {DARK_CARD_BG};
    border-top: 1px solid {DARK_BORDER};
    color: {DARK_TEXT_MUTED};
    font-size: 12px;
    padding: 4px 8px;
}}

/* ── ScrollBar ───────────────────────────────────────────────────── */
QScrollBar:vertical {{
    background: {DARK_BG};
    width: 8px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical {{
    background: {DARK_BORDER};
    border-radius: 4px;
    min-height: 20px;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* ── Label override ──────────────────────────────────────────────── */
QLabel {{
    color: {DARK_TEXT_MAIN};
}}

/* ── FormLayout label fix ────────────────────────────────────────── */
QFormLayout QLabel {{
    color: {DARK_TEXT_MUTED};
    font-weight: 600;
}}
"""


def get_status_pill_style(state: str, dark: bool = False) -> str:
    """
    Visszaadja a státusz pill QSS stílusát.

    Args:
        state: 'ok' | 'warning' | 'error' | 'info'
        dark:  True ha dark módban vagyunk
    """
    if state == "ok":
        bg, text, border = "#D1FAE5", "#065F46", "#10B981"
        if dark:
            bg, text, border = "#064E3B", "#4ADE80", "#10B981"
    elif state == "warning":
        bg, text, border = "#FEF3C7", "#92400E", "#F59E0B"
        if dark:
            bg, text, border = "#451A03", "#FCD34D", "#F59E0B"
    elif state == "error":
        bg, text, border = "#FEE2E2", "#991B1B", "#EF4444"
        if dark:
            bg, text, border = "#450A0A", "#FCA5A5", "#EF4444"
    else:  # info
        bg, text, border = "#E2E8F0", "#334155", "#94A3B8"
        if dark:
            bg, text, border = "#1E293B", "#94A3B8", "#26334D"

    return (
        f"background-color: {bg}; "
        f"color: {text}; "
        f"border: 1px solid {border}; "
        f"border-radius: 10px; "
        f"padding: 3px 10px; "
        f"font-weight: 700; "
        f"font-size: 11px;"
    )


def get_hw_pill_style(level: str, dark: bool = False) -> str:
    """
    Hardver erőforrás szinthez illő pill stílus.
    level: 'low' (<70%) | 'medium' (70–89%) | 'high' (≥90%)
    """
    if level == "low":
        return get_status_pill_style("ok", dark)
    elif level == "medium":
        return get_status_pill_style("warning", dark)
    else:
        return get_status_pill_style("error", dark)


def usage_level(percent: float) -> str:
    """Percent → 'low' | 'medium' | 'high'"""
    if percent >= 90:
        return "high"
    elif percent >= 70:
        return "medium"
    return "low"


def get_app_icon():
    """
    Létrehoz egy több felbontású QIcon példányt.
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
