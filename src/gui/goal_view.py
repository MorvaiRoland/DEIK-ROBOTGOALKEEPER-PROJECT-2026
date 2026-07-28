"""
DEIK Robot Foci Kapus – Kapu Vizualizátor Widget
=================================================

Ez a PyQt6 widget a kapu 2D nézetét rajzolja:
    - Fehér kapu keret fekete háttéren
    - Aktív becsapódási pont (sárga, pulzáló)
    - Korábbi lövések história (halvány szürkék)
    - Konfidencia-alapú színezés
    - Robot kapus pozíció jelzője (kék csík)

Koordináta-rendszer:
    X = [-2000, +2000] mm (bal–jobb, kapu közepétől)
    Y = [0, 2000] mm     (alul–felül, talajtól)
"""

import logging
import math
from typing import List, Optional, Tuple

from PyQt6.QtCore import QPointF, QRectF, QTimer, Qt
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter,
    QPen, QRadialGradient
)
from PyQt6.QtWidgets import QSizePolicy, QWidget

logger = logging.getLogger(__name__)


class GoalViewWidget(QWidget):
    """
    2D kapu-vizualizátor widget.

    Megjeleníti:
        - A focilabda prediktált becsapódási pontját
        - A korábbi lövések historikáját
        - A kapu keretét, rácsvonalakkal
        - A konfidenciát színkódolva

    Példa (PyQt6):
        goal_widget = GoalViewWidget(config)
        goal_widget.update_impact(x_mm=500, y_mm=800, conf=0.85, t_s=0.45)
    """

    # Animációs frissítési ráta (ms) – 20 Hz a pulzáló hatáshoz
    _ANIMATION_INTERVAL_MS = 50

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        """
        Args:
            config: A system_config.yaml "geometry" szekciója
            parent: Szülő widget
        """
        super().__init__(parent)

        # Kapu méretei
        geo_cfg = config.get("geometry", {})
        self._goal_width_mm = float(geo_cfg.get("goal_width_mm", 4000.0))
        self._goal_height_mm = float(geo_cfg.get("goal_height_mm", 2000.0))

        # GUI beállítások
        gui_cfg = config.get("gui", {}).get("goal_view", {})
        self._max_history = int(gui_cfg.get("max_shot_history", 15))

        # Aktív becsapódási pont adatai
        self._impact_x_mm: Optional[float] = None  # mm, kapu közepétől
        self._impact_y_mm: Optional[float] = None  # mm, talajtól
        self._impact_conf: float = 0.0             # 0.0 – 1.0
        self._time_to_impact_s: float = 0.0        # másodperc
        self._in_goal: bool = False                # a kapun belül van-e

        # Korábbi lövések historikája: lista (x_mm, y_mm, conf, in_goal) tupleokból
        self._shot_history: List[Tuple[float, float, float, bool]] = []

        # Animációs fázis (pulzáláshoz)
        self._anim_phase: float = 0.0

        # Animációs timer
        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start(self._ANIMATION_INTERVAL_MS)

        # Widget méretpolitika
        self.setMinimumSize(400, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        logger.debug("GoalViewWidget inicializálva: %.0f×%.0f mm kapu",
                     self._goal_width_mm, self._goal_height_mm)

    # ------------------------------------------------------------------
    # Publikus API
    # ------------------------------------------------------------------

    def update_impact(
        self,
        x_mm: Optional[float],
        y_mm: Optional[float],
        confidence: float,
        time_to_impact_s: float,
        in_goal: bool = False,
    ) -> None:
        """
        Frissíti a megjelenített becsapódási pontot.

        Args:
            x_mm:             X koordináta mm-ben (kapu közepétől, negatív = bal)
            y_mm:             Y koordináta mm-ben (talajtól)
            confidence:       Predikció megbízhatósága [0.0 – 1.0]
            time_to_impact_s: Hány másodperc múlva ér a kapuhoz
            in_goal:          True ha a kapun belülre jósolt
        """
        if x_mm is not None and time_to_impact_s > 0.0:
            # Új aktív pont
            self._impact_x_mm = x_mm
            self._impact_y_mm = y_mm
            self._impact_conf = confidence
            self._time_to_impact_s = time_to_impact_s
            self._in_goal = in_goal
        else:
            # Nincs aktív predikció: az előző pontot átrakjuk a historikába
            if self._impact_x_mm is not None:
                self._save_to_history()
            self._impact_x_mm = None
            self._impact_y_mm = None
            self._impact_conf = 0.0
            self._time_to_impact_s = 0.0

        self.update()  # Widget újrarajzolás kérése

    def clear_history(self) -> None:
        """Törli a lövés historikát és az aktív pontot."""
        self._shot_history.clear()
        self._impact_x_mm = None
        self._impact_y_mm = None
        self.update()

    # ------------------------------------------------------------------
    # Animáció
    # ------------------------------------------------------------------

    def _on_animation_tick(self) -> None:
        """Animációs timer callback – frissíti a pulzálási fázist."""
        self._anim_phase = (self._anim_phase + 0.18) % (2 * math.pi)
        if self._impact_x_mm is not None:
            self.update()

    # ------------------------------------------------------------------
    # Qt Paint Event – a teljes widget újrarajzolása
    # ------------------------------------------------------------------

    def paintEvent(self, _event) -> None:
        """A Qt főrajzolási metódusa – minden frame-en meghívódik."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        W = self.width()
        H = self.height()

        # ── 1. Sötét háttér ───────────────────────────────────────────
        painter.fillRect(0, 0, W, H, QColor(18, 18, 24))

        # ── 2. Koordináta-transzformáció ──────────────────────────────
        # Margó a feliratoknak
        margin_top = 35
        margin_bottom = 20
        margin_left = 50
        margin_right = 20

        draw_w = W - margin_left - margin_right
        draw_h = H - margin_top - margin_bottom

        # Egységes skála (torzítás nélkül)
        scale = min(draw_w / self._goal_width_mm, draw_h / self._goal_height_mm)
        goal_px_w = self._goal_width_mm * scale
        goal_px_h = self._goal_height_mm * scale

        # Kapu bal-felső sarka képernyő-koordinátában
        gx = margin_left + (draw_w - goal_px_w) / 2
        gy = margin_top + (draw_h - goal_px_h) / 2

        def mm_to_px(x_mm: float, y_mm: float) -> QPointF:
            """Kapu-mm koordinátát widget pixel koordinátává alakít."""
            px = gx + (x_mm + self._goal_width_mm / 2) * scale
            py = gy + (self._goal_height_mm - y_mm) * scale  # Y felfelé pozitív
            return QPointF(px, py)

        # ── 3. Fejléc ─────────────────────────────────────────────────
        self._draw_header(painter, W)

        # ── 4. Rács ───────────────────────────────────────────────────
        self._draw_grid(painter, gx, gy, goal_px_w, goal_px_h)

        # ── 5. Kapu keret ─────────────────────────────────────────────
        self._draw_goal_frame(painter, gx, gy, goal_px_w, goal_px_h)

        # ── 6. Méretek feliratozása ────────────────────────────────────
        self._draw_dimension_labels(painter, gx, gy, goal_px_w, goal_px_h)

        # ── 7. Historika (korábbi lövések) ────────────────────────────
        self._draw_shot_history(painter, mm_to_px, scale)

        # ── 8. Aktív becsapódási pont ─────────────────────────────────
        if self._impact_x_mm is not None:
            self._draw_active_impact(painter, mm_to_px, W, H)
        elif not self._shot_history:
            self._draw_waiting_message(painter, W, H)

        painter.end()

    # ------------------------------------------------------------------
    # Rajzoló segéd-metódusok
    # ------------------------------------------------------------------

    def _draw_header(self, painter: QPainter, W: int) -> None:
        """Fejléc szöveg rajzolása."""
        font = QFont("Arial", 10, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QPen(QColor(180, 180, 200)))
        painter.drawText(
            QRectF(0, 4, W, 28),
            Qt.AlignmentFlag.AlignHCenter,
            "KAPU NÉZET  —  Becsapódás Előrejelző"
        )

    def _draw_grid(
        self, painter: QPainter,
        gx: float, gy: float, pw: float, ph: float
    ) -> None:
        """Belső rácsvonalak rajzolása."""
        grid_pen = QPen(QColor(40, 42, 54), 1)
        painter.setPen(grid_pen)
        n = 4
        for i in range(1, n):
            # Függőleges vonalak
            x = gx + i * pw / n
            painter.drawLine(QPointF(x, gy), QPointF(x, gy + ph))
            # Vízszintes vonalak
            y = gy + i * ph / n
            painter.drawLine(QPointF(gx, y), QPointF(gx + pw, y))

    def _draw_goal_frame(
        self, painter: QPainter,
        gx: float, gy: float, pw: float, ph: float
    ) -> None:
        """Fehér kapu keret rajzolása."""
        # Kapu mögötti háttér (sötétebb téglalappal)
        painter.fillRect(QRectF(gx, gy, pw, ph), QColor(8, 8, 14))

        # Kapu keret (fehér, 3 px vastag)
        frame_pen = QPen(QColor(240, 240, 255), 3)
        painter.setPen(frame_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(QRectF(gx, gy, pw, ph))

        # Középvonal (szaggatott, halvány)
        mid_pen = QPen(QColor(80, 80, 100), 1, Qt.PenStyle.DashLine)
        painter.setPen(mid_pen)
        cx = gx + pw / 2
        painter.drawLine(QPointF(cx, gy), QPointF(cx, gy + ph))

    def _draw_dimension_labels(
        self, painter: QPainter,
        gx: float, gy: float, pw: float, ph: float
    ) -> None:
        """Méretek feliratozása (mm / méter)."""
        font = QFont("Consolas", 7)
        painter.setFont(font)
        painter.setPen(QPen(QColor(90, 90, 110)))

        # Szélesség (felül)
        painter.drawText(
            QRectF(gx, gy - 18, pw, 16),
            Qt.AlignmentFlag.AlignHCenter,
            f"← {self._goal_width_mm / 1000:.1f} m →"
        )

        # Magasság (bal oldal, elforgatott)
        painter.save()
        painter.translate(gx - 38, gy + ph / 2)
        painter.rotate(-90)
        painter.drawText(
            QRectF(-ph / 2, -12, ph, 16),
            Qt.AlignmentFlag.AlignHCenter,
            f"↑ {self._goal_height_mm / 1000:.1f} m ↓"
        )
        painter.restore()

    def _draw_shot_history(
        self, painter: QPainter, mm_to_px, scale: float
    ) -> None:
        """Korábbi lövések halvány körökkel."""
        n = len(self._shot_history)
        for idx, (hx, hy, hconf, hin_goal) in enumerate(self._shot_history):
            # Régebbi lövések halványabbak
            alpha = int(40 + 140 * idx / max(n, 1))
            color = QColor(255, 100, 100, alpha) if not hin_goal else QColor(100, 200, 100, alpha)

            pt = mm_to_px(hx, hy)
            r = max(5.0, 12.0 * scale * 0.11)
            painter.setPen(QPen(color, 1))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), alpha // 3)))
            painter.drawEllipse(pt, r, r)

    def _draw_active_impact(
        self, painter: QPainter, mm_to_px, W: int, H: int
    ) -> None:
        """Az aktív becsapódási pont megjelenítése pulzálással."""
        pt = mm_to_px(self._impact_x_mm, self._impact_y_mm)

        # Konfidencia-alapú szín
        if self._impact_conf >= 0.75:
            base_color = QColor(50, 255, 120)     # Zöld – nagy megbízhatóság
        elif self._impact_conf >= 0.50:
            base_color = QColor(255, 220, 0)      # Sárga – közepes
        elif self._impact_conf >= 0.30:
            base_color = QColor(255, 140, 0)      # Narancssárga – gyenge
        else:
            base_color = QColor(255, 60, 60)      # Piros – kis megbízhatóság

        # Pulzáló külső gyűrű
        pulse_r = 22 + 7 * math.sin(self._anim_phase)
        ring_pen = QPen(base_color, 2, Qt.PenStyle.DashLine)
        painter.setPen(ring_pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(pt, pulse_r, pulse_r)

        # Belső kör (radial gradient)
        inner_r = 11.0
        grad = QRadialGradient(pt, inner_r)
        grad.setColorAt(0.0, QColor(255, 255, 240, 230))
        grad.setColorAt(1.0, QColor(base_color.red(), base_color.green(), base_color.blue(), 200))
        painter.setPen(QPen(base_color, 2))
        painter.setBrush(QBrush(grad))
        painter.drawEllipse(pt, inner_r, inner_r)

        # Kereszthajó
        ch_len = 25
        ch_pen = QPen(QColor(255, 255, 255, 160), 1)
        painter.setPen(ch_pen)
        painter.drawLine(
            QPointF(pt.x() - ch_len, pt.y()),
            QPointF(pt.x() + ch_len, pt.y())
        )
        painter.drawLine(
            QPointF(pt.x(), pt.y() - ch_len),
            QPointF(pt.x(), pt.y() + ch_len)
        )

        # Koordináta-felirat
        label_font = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(label_font)
        painter.setPen(QPen(base_color))

        goal_side = "KAPUN BELÜL" if self._in_goal else "KAPUN KÍVÜL"
        lines = [
            f"X: {self._impact_x_mm:+.0f} mm",
            f"Y: {self._impact_y_mm:+.0f} mm",
            f"T: {self._time_to_impact_s:.3f} s",
            f"Conf: {self._impact_conf * 100:.0f}%  {goal_side}",
        ]
        lx = pt.x() + 16
        ly = pt.y() - 22
        if lx + 170 > W:
            lx = pt.x() - 185
        for i, line in enumerate(lines):
            painter.drawText(QPointF(lx, ly + i * 14), line)

    def _draw_waiting_message(self, painter: QPainter, W: int, H: int) -> None:
        """'Várakozás' üzenet ha nincs mérés."""
        painter.setPen(QPen(QColor(70, 70, 90)))
        font = QFont("Arial", 12)
        painter.setFont(font)
        painter.drawText(
            QRectF(0, 0, W, H),
            Qt.AlignmentFlag.AlignCenter,
            "Várakozás lövésre…\n(Nincs aktív becsapódás-predikció)"
        )

    # ------------------------------------------------------------------
    # Belső segédek
    # ------------------------------------------------------------------

    def _save_to_history(self) -> None:
        """Az aktív pontot áthelyezi a historikába."""
        if self._impact_x_mm is not None:
            self._shot_history.append((
                self._impact_x_mm,
                self._impact_y_mm,
                self._impact_conf,
                self._in_goal,
            ))
            if len(self._shot_history) > self._max_history:
                self._shot_history.pop(0)
