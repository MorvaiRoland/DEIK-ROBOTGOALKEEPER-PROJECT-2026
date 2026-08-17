"""
DEIK Robot Foci Kapus – Fehér Témájú 2D Kapu Vizualizátor Widget
================================================================

Ez a widget a kapu 2D vetületét jeleníti meg letisztult, világos (fehér) témában:
    - DEIK Zöld / Fehér kapukeret világos felületen
    - Sárga / Zöld / Piros becsapódási pontok
    - Éles, jól olvasható sötét skálamérő koordináták
    - Korábbi lövések átlátható históriája
"""

import logging
import math
from typing import List, Optional, Tuple

# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtCore import QPointF, QRectF, QTimer, Qt
# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QPainter, QPen, QPolygonF, QRadialGradient
)
# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtWidgets import QSizePolicy, QWidget

logger = logging.getLogger(__name__)


class GoalViewWidget(QWidget):
    """
    Fehér témájú 2D kapu-vizualizátor (letisztult, formális).
    """

    _ANIMATION_INTERVAL_MS = 40

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        geo_cfg = config.get("geometry", {})
        self._goal_width_mm = float(geo_cfg.get("goal_width_mm", 4000.0))
        self._goal_height_mm = float(geo_cfg.get("goal_height_mm", 2000.0))

        gui_cfg = config.get("gui", {}).get("goal_view", {})
        self._max_history = int(gui_cfg.get("max_shot_history", 20))

        self._impact_x_mm: Optional[float] = None
        self._impact_y_mm: Optional[float] = None
        self._impact_conf: float = 0.0
        self._time_to_impact_s: float = 0.0
        self._in_goal: bool = False

        self._shot_history: List[Tuple[float, float, float, bool]] = []
        self._anim_phase: float = 0.0

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start(self._ANIMATION_INTERVAL_MS)

        self.setMinimumSize(380, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def update_impact(
        self,
        x_mm: Optional[float],
        y_mm: Optional[float],
        confidence: float,
        time_to_impact_s: float,
        in_goal: bool = False,
    ) -> None:
        if x_mm is not None and time_to_impact_s >= 0.0:
            self._impact_x_mm = x_mm
            self._impact_y_mm = y_mm
            self._impact_conf = confidence
            self._time_to_impact_s = time_to_impact_s
            self._in_goal = in_goal
        else:
            if self._impact_x_mm is not None:
                self._save_to_history()
            self._impact_x_mm = None
            self._impact_y_mm = None
            self._impact_conf = 0.0
            self._time_to_impact_s = 0.0

        self.update()

    def clear_history(self) -> None:
        self._shot_history.clear()
        self._impact_x_mm = None
        self._impact_y_mm = None
        self.update()

    def _save_to_history(self) -> None:
        if self._impact_x_mm is not None and self._impact_y_mm is not None:
            self._shot_history.append((
                self._impact_x_mm,
                self._impact_y_mm,
                self._impact_conf,
                self._in_goal
            ))
            if len(self._shot_history) > self._max_history:
                self._shot_history.pop(0)

    def _on_animation_tick(self) -> None:
        self._anim_phase = (self._anim_phase + 0.15) % (2 * math.pi)
        if self._impact_x_mm is not None:
            self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QBrush(QColor("#FFFFFF")))

        margin_x = 45
        margin_top = 35
        margin_bottom = 35

        draw_w = w - 2 * margin_x
        draw_h = h - margin_top - margin_bottom

        aspect_goal = self._goal_width_mm / self._goal_height_mm
        if draw_w / draw_h > aspect_goal:
            rect_h = draw_h
            rect_w = rect_h * aspect_goal
        else:
            rect_w = draw_w
            rect_h = rect_w / aspect_goal

        rect_x = (w - rect_w) / 2.0
        rect_y = margin_top + (draw_h - rect_h) / 2.0

        goal_rect = QRectF(rect_x, rect_y, rect_w, rect_h)

        self._draw_net_and_frame(painter, goal_rect)
        self._draw_history(painter, goal_rect)

        if self._impact_x_mm is not None and self._impact_y_mm is not None:
            self._draw_active_target(painter, goal_rect)

        self._draw_hud_overlay_text(painter, goal_rect, w, h)

    def _draw_net_and_frame(self, painter: QPainter, goal_rect: QRectF) -> None:
        x, y, w, h = goal_rect.x(), goal_rect.y(), goal_rect.width(), goal_rect.height()

        pen_net = QPen(QColor(203, 213, 225), 1)
        painter.setPen(pen_net)
        cols, rows = 12, 6
        for i in range(1, cols):
            nx = x + i * (w / cols)
            painter.drawLine(QPointF(nx, y), QPointF(nx, y + h))
        for j in range(1, rows):
            ny = y + j * (h / rows)
            painter.drawLine(QPointF(x, ny), QPointF(x + w, ny))

        painter.setPen(QPen(QColor("#0F5132"), 4))
        painter.drawRect(goal_rect)

        painter.setPen(QPen(QColor("#D97706"), 3))
        painter.drawLine(QPointF(x - 20, y + h), QPointF(x + w + 20, y + h))

        font_ticks = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(font_ticks)
        painter.setPen(QPen(QColor("#334155")))

        half_w = self._goal_width_mm / 2.0
        for val in [-half_w, -half_w / 2, 0, half_w / 2, half_w]:
            px = x + (val + half_w) / self._goal_width_mm * w
            painter.drawLine(QPointF(px, y + h), QPointF(px, y + h + 5))
            txt = f"{val:+.0f}" if val != 0 else "0"
            painter.drawText(QRectF(px - 30, y + h + 6, 60, 16), Qt.AlignmentFlag.AlignCenter, txt)

    def _draw_history(self, painter: QPainter, goal_rect: QRectF) -> None:
        x, y, w, h = goal_rect.x(), goal_rect.y(), goal_rect.width(), goal_rect.height()
        half_w = self._goal_width_mm / 2.0

        n_shots = len(self._shot_history)
        for idx, (sx, sy, conf, in_g) in enumerate(self._shot_history):
            px = x + (sx + half_w) / self._goal_width_mm * w
            py = y + h - (sy / self._goal_height_mm * h)

            alpha = int(80 + (idx + 1) / n_shots * 160)
            color = QColor("#059669") if in_g else QColor("#DC2626")
            color.setAlpha(alpha)

            painter.setPen(QPen(color, 1.5))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), int(alpha * 0.3))))
            painter.drawEllipse(QPointF(px, py), 5, 5)

    def _draw_yellow_ball_icon(self, painter: QPainter, px: float, py: float, radius: float = 14.0) -> None:
        """
        Kirajzol egy látványos sárga focilabda ikont a becsapódási pontra.
        """
        # 1. Külső sárga lüktető fényudvar (Pulsing Yellow Glow)
        glow_r = radius * 2.2 + 3.0 * math.sin(self._anim_phase)
        glow_grad = QRadialGradient(QPointF(px, py), glow_r)
        glow_grad.setColorAt(0.0, QColor(254, 240, 138, 220))  # Ragyogó sárga mag
        glow_grad.setColorAt(0.6, QColor(234, 179, 8, 120))   # Aranysárga közép
        glow_grad.setColorAt(1.0, QColor(234, 179, 8, 0))     # Átlátszó szél

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow_grad))
        painter.drawEllipse(QPointF(px, py), glow_r, glow_r)

        # 2. Labda teste (Sárga QRadialGradient 3D hatás)
        ball_grad = QRadialGradient(QPointF(px - radius * 0.3, py - radius * 0.3), radius * 1.5)
        ball_grad.setColorAt(0.0, QColor("#FEF08A"))  # Csúcsfény
        ball_grad.setColorAt(0.6, QColor("#FACC15"))  # Élénksárga
        ball_grad.setColorAt(1.0, QColor("#EAB308"))  # Árnyékos sárga

        painter.setPen(QPen(QColor("#713F12"), 1.8))  # Sötétbarna/arany körvonal
        painter.setBrush(QBrush(ball_grad))
        painter.drawEllipse(QPointF(px, py), radius, radius)

        # 3. Focilabda ötszög mintázat (varrás)
        pen_pattern = QPen(QColor("#451A03"), 1.5)
        painter.setPen(pen_pattern)

        r_p = radius * 0.42
        pts_inner = []
        for k in range(5):
            angle = -math.pi / 2.0 + k * (2.0 * math.pi / 5.0)
            pts_inner.append(QPointF(px + r_p * math.cos(angle), py + r_p * math.sin(angle)))

        painter.setBrush(QBrush(QColor("#713F12")))  # Sötétbarna ötszög
        painter.drawPolygon(QPolygonF(pts_inner))

        for k in range(5):
            angle = -math.pi / 2.0 + k * (2.0 * math.pi / 5.0)
            outer_p = QPointF(px + radius * math.cos(angle), py + radius * math.sin(angle))
            painter.drawLine(pts_inner[k], outer_p)

    def _draw_active_target(self, painter: QPainter, goal_rect: QRectF) -> None:
        x, y, w, h = goal_rect.x(), goal_rect.y(), goal_rect.width(), goal_rect.height()
        half_w = self._goal_width_mm / 2.0

        px = x + (self._impact_x_mm + half_w) / self._goal_width_mm * w
        py = y + h - (self._impact_y_mm / self._goal_height_mm * h)

        # Sárga focilabda ikon kirajzolása
        self._draw_yellow_ball_icon(painter, px, py, radius=14.0)

        # Célkereszt
        pen_cross = QPen(QColor("#CA8A04"), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(pen_cross)
        painter.drawLine(QPointF(px - 22, py), QPointF(px + 22, py))
        painter.drawLine(QPointF(px, py - 22), QPointF(px, py + 22))

        # Koordináta kártya
        font_txt = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(font_txt)
        tag = f"⚽ X:{self._impact_x_mm:+.0f} Y:{self._impact_y_mm:.0f}mm ({self._time_to_impact_s:.2f}s)"

        lbl_r = QRectF(px + 18, py - 12, 175, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(254, 240, 138, 240)))
        painter.drawRoundedRect(lbl_r, 5, 5)
        painter.setPen(QPen(QColor("#854D0E"), 1.5))
        painter.drawRoundedRect(lbl_r, 5, 5)
        painter.setPen(QPen(QColor("#713F12")))
        painter.drawText(lbl_r, Qt.AlignmentFlag.AlignCenter, tag)

    def _draw_hud_overlay_text(self, painter: QPainter, goal_rect: QRectF, w: int, h: int) -> None:
        font_hud = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font_hud)

        if self._impact_x_mm is not None:
            st_txt = "DETEKTÁLT BECSAPÓDÁS"
            st_color = QColor("#0F5132")
        else:
            st_txt = "PÁLYAKÖVETÉS AKTÍV"
            st_color = QColor("#475569")

        painter.setPen(QPen(st_color))
        painter.drawText(14, 20, st_txt)

        hist_txt = f"LÖVÉSEK: {len(self._shot_history)} DB"
        painter.setPen(QPen(QColor("#475569")))
        painter.drawText(w - 140, 20, hist_txt)
