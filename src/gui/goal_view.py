"""
DEIK Robot Foci Kapus – 2D Kapu Vizualizátor Widget
================================================================

Ez a widget a kapu 2D vetületét jeleníti meg:
    - DEIK Zöld / Fehér kapukeret
    - Sárga / Zöld / Piros becsapódási pontok
    - Éles, jól olvasható koordináta skálázás
    - Korábbi lövések história + statisztika
    - Konfidencia sáv a predikció megbízhatóságára
"""

import logging
import math
from typing import Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtCore import QPointF, QRectF, QTimer, Qt
# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPen, QPolygonF, QRadialGradient
)
# pyrefly: ignore [missing-import]
# type: ignore
from PyQt6.QtWidgets import QHBoxLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

logger = logging.getLogger(__name__)


class GoalViewWidget(QWidget):
    """
    2D kapu-vizualizátor konfidencia sávval és lövési statisztikával.
    Téma-tudatos: is_dark property-vel átkapcsolható dark módba.
    """

    _ANIMATION_INTERVAL_MS = 40

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)

        geo_cfg = config.get("geometry", {})
        self._goal_width_mm  = float(geo_cfg.get("goal_width_mm",  4000.0))
        self._goal_height_mm = float(geo_cfg.get("goal_height_mm", 2000.0))

        gui_cfg = config.get("gui", {}).get("goal_view", {})
        self._max_history = int(gui_cfg.get("max_shot_history", 20))

        self._impact_x_mm:       Optional[float] = None
        self._impact_y_mm:       Optional[float] = None
        self._impact_conf:       float = 0.0
        self._time_to_impact_s:  float = 0.0
        self._in_goal:           bool  = False

        # Lövés história: (x_mm, y_mm, conf, in_goal)
        self._shot_history: List[Tuple[float, float, float, bool]] = []
        self._anim_phase:   float = 0.0
        # Statisztika számlálók
        self._total_shots:  int = 0
        self._in_goal_shots: int = 0
        self._saved_shots:  int = 0
        self._left_shots:   int = 0
        self._center_shots: int = 0
        self._right_shots:  int = 0

        # Robot Kapus Szimuláció (Középen álló, balra/jobbra dőlő mechanika)
        self._gk_tilt_angle_deg: float = 0.0      # Aktuális dőlésszög fokban (-55° .. +55°)
        self._gk_target_tilt_deg: float = 0.0     # Cél dőlésszög fokban
        self._gk_reach_width_mm: float = 800.0    # Védési pajzs szélessége (mm)
        self._gk_state: str = "IDLE"              # "IDLE" | "MOVING" | "DEFENDED" | "MISSED"

        # Dark mód & 2D / 3D Nézet mód
        self._dark: bool = False
        self._mode_3d: bool = False

        # Status QLabel a bal felső sarokban
        self._lbl_hud_status = QLabel("PÁLYAKÖVETÉS AKTÍV | LÖVÉSEK: 0 DB")

        # Valódi Natív QPushButtonek a 2D / 3D váltáshoz
        self._btn_mode_2d = QPushButton("2D KAPU", self)
        self._btn_mode_3d = QPushButton("3D PÁLYA", self)
        for b in [self._btn_mode_2d, self._btn_mode_3d]:
            b.setFixedSize(72, 24)
            b.setCursor(Qt.CursorShape.PointingHandCursor)

        self._btn_mode_2d.clicked.connect(lambda _=False: self.set_mode_3d(False))
        self._btn_mode_3d.clicked.connect(lambda _=False: self.set_mode_3d(True))

        top_layout = QHBoxLayout()
        top_layout.setContentsMargins(8, 4, 8, 0)
        top_layout.setSpacing(4)
        top_layout.addWidget(self._lbl_hud_status, stretch=1)
        top_layout.addWidget(self._btn_mode_2d)
        top_layout.addWidget(self._btn_mode_3d)

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        main_layout.addLayout(top_layout)
        main_layout.addStretch()

        self._update_mode_button_styles()
        self._update_hud_status_label()

        self._anim_timer = QTimer(self)
        self._anim_timer.timeout.connect(self._on_animation_tick)
        self._anim_timer.start(self._ANIMATION_INTERVAL_MS)

        self.setMinimumSize(380, 260)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    # ── Public API ───────────────────────────────────────────────────────────

    def set_mode_3d(self, mode_3d: bool) -> None:
        """Vált a 2D Kapu és 3D Pálya nézet között."""
        self._mode_3d = bool(mode_3d)
        self._update_mode_button_styles()
        self.update()

    def set_dark(self, dark: bool) -> None:
        """Téma váltás."""
        self._dark = dark
        self._update_mode_button_styles()
        self._update_hud_status_label()
        self.update()

    def _update_mode_button_styles(self) -> None:
        """Gombok stílusának frissítése téma és aktív módtól függően."""
        if not hasattr(self, "_btn_mode_2d"):
            return
        dark = getattr(self, "_dark", False)
        mode3d = getattr(self, "_mode_3d", False)

        active_s = (
            "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; "
            "font-size: 11px; border-radius: 5px; border: 1.5px solid #10B981; padding: 0 4px; }"
            "QPushButton:hover { background-color: #146C43; color: #FFFFFF; }"
        )
        inactive_s = (
            "QPushButton { background-color: #151D2A; color: #94A3B8; font-weight: 700; "
            "font-size: 11px; border-radius: 5px; border: 1px solid #26334D; padding: 0 4px; }"
            "QPushButton:hover { background-color: #1E293B; color: #4ADE80; border-color: #4ADE80; }"
            if dark else
            "QPushButton { background-color: #F1F5F9; color: #334155; font-weight: 700; "
            "font-size: 11px; border-radius: 5px; border: 1px solid #CBD5E1; padding: 0 4px; }"
            "QPushButton:hover { background-color: #E2E8F0; color: #0F5132; }"
        )

        self._btn_mode_2d.setStyleSheet(active_s if not mode3d else inactive_s)
        self._btn_mode_3d.setStyleSheet(active_s if mode3d else inactive_s)

    def _update_hud_status_label(self) -> None:
        """Frissíti a bal felső QLabel státusz szövegét és téma szerinti színét."""
        if not hasattr(self, "_lbl_hud_status"):
            return
        dark = getattr(self, "_dark", False)
        stats = self.get_stats()
        if stats["total"] > 0:
            hist_str = f" | LÖVÉS: {stats['total']:.0f} (GÓL: {stats['in_goal']:.0f})"
        else:
            hist_str = f" | LÖVÉSEK: {len(self._shot_history)} DB"

        if self._impact_x_mm is not None:
            st_txt = f"BECSAPÓDÁS DETEKTÁLVA{hist_str}"
            color = "#4ADE80" if dark else "#0F5132"
        else:
            st_txt = f"PÁLYAKÖVETÉS AKTÍV{hist_str}"
            color = "#94A3B8" if dark else "#475569"

        self._lbl_hud_status.setText(st_txt)
        self._lbl_hud_status.setStyleSheet(f"font-size: 11px; font-weight: 800; color: {color}; background: transparent;")

    def set_goalkeeper_target(self, x_mm: float, y_mm: float = 1000.0) -> None:
        """Beállítja a dőlési célpozíciót az X koordináta alapján (középről balra/jobbra dőlés)."""
        half_w = self._goal_width_mm / 2.0
        norm_x = max(-1.0, min(1.0, x_mm / half_w))
        self._gk_target_tilt_deg = norm_x * 55.0  # Max ±55 fokos dőlés

    def update_impact(
        self,
        x_mm: Optional[float],
        y_mm: Optional[float],
        confidence: float,
        time_to_impact_s: float,
        in_goal: bool = False,
    ) -> None:
        if x_mm is not None and time_to_impact_s >= 0.0:
            self._impact_x_mm       = x_mm
            self._impact_y_mm       = y_mm
            self._impact_conf       = confidence
            self._time_to_impact_s  = time_to_impact_s
            self._in_goal           = in_goal

            # Robot kapus cél dőlésszöge az előrejelzett becsapódásra
            half_w = self._goal_width_mm / 2.0
            norm_x = max(-1.0, min(1.0, x_mm / half_w))
            self._gk_target_tilt_deg = norm_x * 55.0
            self._gk_state = "MOVING"
        else:
            if self._impact_x_mm is not None:
                self._save_to_history()
            self._impact_x_mm      = None
            self._impact_y_mm      = None
            self._impact_conf      = 0.0
            self._time_to_impact_s = 0.0
            self._in_goal          = False
        
        self._update_hud_status_label()
        self.update()

    def reset_stats(self) -> None:
        """Szerver / GUI által meghívható statisztika törlés."""
        self._shot_history.clear()
        self._total_shots  = 0
        self._in_goal_shots = 0
        self._saved_shots  = 0
        self._left_shots   = 0
        self._center_shots = 0
        self._right_shots  = 0
        self._gk_tilt_angle_deg = 0.0
        self._gk_target_tilt_deg = 0.0
        self._gk_state     = "IDLE"
        self.update()

    def get_stats(self) -> Dict[str, float]:
        total = max(1, self._total_shots)
        return {
            "total": float(self._total_shots),
            "in_goal": float(self._in_goal_shots),
            "saved": float(self._saved_shots),
            "save_pct": (self._saved_shots / total) * 100.0,
            "in_goal_pct": (self._in_goal_shots / total) * 100.0,
            "left": float(self._left_shots),
            "center": float(self._center_shots),
            "right": float(self._right_shots),
        }

    # ── Belső logika ─────────────────────────────────────────────────────────

    def _save_to_history(self) -> None:
        if self._impact_x_mm is not None and self._impact_y_mm is not None:
            # Ellenőrizzük a dőléssel elért pozíciót a védés szempontjából
            half_w = self._goal_width_mm / 2.0
            reach_x = (self._gk_tilt_angle_deg / 55.0) * (half_w * 0.85)
            dist_to_gk = abs(self._impact_x_mm - reach_x)

            is_saved = dist_to_gk <= (self._gk_reach_width_mm / 2.0)
            if is_saved:
                self._saved_shots += 1
                self._gk_state = "DEFENDED"
            else:
                self._gk_state = "MISSED"

            self._shot_history.append((
                self._impact_x_mm,
                self._impact_y_mm,
                self._impact_conf,
                self._in_goal
            ))
            if len(self._shot_history) > self._max_history:
                self._shot_history.pop(0)

            # Statisztika frissítés
            self._total_shots += 1
            if self._in_goal:
                self._in_goal_shots += 1
            zone_w = half_w * 0.4   # ±40% közép sáv
            if self._impact_x_mm < -zone_w:
                self._left_shots += 1
            elif self._impact_x_mm > zone_w:
                self._right_shots += 1
            else:
                self._center_shots += 1

    def _on_animation_tick(self) -> None:
        self._anim_phase = (self._anim_phase + 0.15) % (2 * math.pi)

        # Robot Kapus dőlésszögének simított interpolációja (22% per tick)
        dtilt = self._gk_target_tilt_deg - self._gk_tilt_angle_deg
        if abs(dtilt) > 0.5:
            self._gk_tilt_angle_deg += dtilt * 0.22
            if self._gk_state != "DEFENDED":
                self._gk_state = "MOVING"
        else:
            if self._gk_state == "MOVING":
                self._gk_state = "IDLE"

        self.update()

    # ── Public API ───────────────────────────────────────────────────────────

    def set_mode_3d(self, mode_3d: bool) -> None:
        """Vált a 2D Kapu és 3D Pálya nézet között."""
        self._mode_3d = mode_3d
        self.update()

    def set_dark(self, dark: bool) -> None:
        """Téma váltás."""
        self._dark = dark
        self.update()

    # ── Paint ────────────────────────────────────────────────────────────────

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)

        w, h = self.width(), self.height()

        bg_color = QColor("#0F172A") if self._dark else QColor("#F8FAFC")
        painter.fillRect(0, 0, w, h, QBrush(bg_color))

        margin_bottom = 52   # plusz hely: konfidencia sáv

        if self._mode_3d:
            self._draw_3d_pitch_view(painter, w, h)
        else:
            margin_x   = 45
            margin_top = 35

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
            self._draw_robot_goalkeeper(painter, goal_rect)

            if self._impact_x_mm is not None and self._impact_y_mm is not None:
                self._draw_active_target(painter, goal_rect)
        self._draw_confidence_bar(painter, w, h, margin_bottom)

    def _project_3d(self, x_mm: float, y_mm: float, z_mm: float, w: int, h: int) -> QPointF:
        """3D perspektivikus koordináta leképezés (X, Y, Z mm -> képernyő pixel)."""
        # Y_mm: 0 mm (Kapuvonal) .. 10000 mm (10-méteres rúgó pont)
        # Z_mm: 0 mm (Talaj) .. 2000 mm (Kapumagasság)
        norm_y = max(0.0, min(1.0, y_mm / 10000.0))

        # Perspektivikus skála a távolság függvényében (közel=nagy, távol=kicsi)
        scale = 1.0 - norm_y * 0.58

        center_x = w / 2.0
        # Kapuvonal Y pixele a képernyőn (felső harmad)
        goal_y_px = h * 0.38
        # Rúgó pont Y pixele (alsó részen)
        kick_y_px = h * 0.88

        py = goal_y_px + norm_y * (kick_y_px - goal_y_px)
        px = center_x + (x_mm / (self._goal_width_mm / 2.0)) * (w * 0.28) * scale

        # Magasság (Z) emelkedés eltolása felfelé
        pz = py - (z_mm / self._goal_height_mm) * (h * 0.26) * scale
        return QPointF(px, pz)

    def _draw_3d_pitch_view(self, painter: QPainter, w: int, h: int) -> None:
        """3D Perspektivikus 10 méteres Focipálya & Trajektória Kirajzolás."""
        # 1. 3D Zöld Focipálya Gradiens Talaj
        p_top_l = self._project_3d(-self._goal_width_mm / 2.0 - 1500, 0, 0, w, h)
        p_top_r = self._project_3d(self._goal_width_mm / 2.0 + 1500, 0, 0, w, h)
        p_bot_r = self._project_3d(self._goal_width_mm / 2.0 + 2500, 10000, 0, w, h)
        p_bot_l = self._project_3d(-self._goal_width_mm / 2.0 - 2500, 10000, 0, w, h)

        pitch_poly = QPolygonF([p_top_l, p_top_r, p_bot_r, p_bot_l])

        pitch_grad = QLinearGradient(0, p_top_l.y(), 0, p_bot_l.y())
        pitch_grad.setColorAt(0.0, QColor("#064E3B") if self._dark else QColor("#15803D"))
        pitch_grad.setColorAt(1.0, QColor("#022C22") if self._dark else QColor("#166534"))

        painter.setPen(QPen(QColor("#10B981"), 2))
        painter.setBrush(QBrush(pitch_grad))
        painter.drawPolygon(pitch_poly)

        # 3D Pálya fehéredő vonalai (10m büntető sáv, kapuvonal, csíkozás)
        pen_lines = QPen(QColor(255, 255, 255, 180), 2)
        painter.setPen(pen_lines)

        # Kapuvonal
        p_g_l = self._project_3d(-self._goal_width_mm / 2.0, 0, 0, w, h)
        p_g_r = self._project_3d(self._goal_width_mm / 2.0, 0, 0, w, h)
        painter.drawLine(p_g_l, p_g_r)

        # 10m Büntető rúgó pont (Középen alul)
        p_kick = self._project_3d(0, 10000, 0, w, h)
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawEllipse(p_kick, 6, 4)

        # 10m Büntető ív / kör
        p_kick_l = self._project_3d(-1000, 10000, 0, w, h)
        p_kick_r = self._project_3d(1000, 10000, 0, w, h)
        painter.drawLine(p_kick_l, p_kick_r)

        # 2. 3D Kapu keret (Post & Crossbar)
        p_goal_bl = self._project_3d(-self._goal_width_mm / 2.0, 0, 0, w, h)
        p_goal_br = self._project_3d(self._goal_width_mm / 2.0, 0, 0, w, h)
        p_goal_tl = self._project_3d(-self._goal_width_mm / 2.0, 0, self._goal_height_mm, w, h)
        p_goal_tr = self._project_3d(self._goal_width_mm / 2.0, 0, self._goal_height_mm, w, h)

        # Hátsó 3D háló keret eltolása (Z=0, Y=-1000 mm mélység)
        p_net_tl = self._project_3d(-self._goal_width_mm / 2.0, -800, self._goal_height_mm, w, h)
        p_net_tr = self._project_3d(self._goal_width_mm / 2.0, -800, self._goal_height_mm, w, h)
        p_net_bl = self._project_3d(-self._goal_width_mm / 2.0, -800, 0, w, h)
        p_net_br = self._project_3d(self._goal_width_mm / 2.0, -800, 0, w, h)

        # Háló mélységi vonalak
        net_pen = QPen(QColor(148, 163, 184, 120) if self._dark else QColor(100, 116, 139, 140), 1, Qt.PenStyle.DashLine)
        painter.setPen(net_pen)
        painter.drawLine(p_goal_tl, p_net_tl)
        painter.drawLine(p_goal_tr, p_net_tr)
        painter.drawLine(p_goal_bl, p_net_bl)
        painter.drawLine(p_goal_br, p_net_br)
        painter.drawLine(p_net_tl, p_net_tr)
        painter.drawLine(p_net_bl, p_net_br)
        painter.drawLine(p_net_tl, p_net_bl)
        painter.drawLine(p_net_tr, p_net_br)

        # Kapu Keret (Fehér vastag léc)
        pen_frame = QPen(QColor("#FFFFFF"), 4)
        painter.setPen(pen_frame)
        painter.drawLine(p_goal_bl, p_goal_tl)
        painter.drawLine(p_goal_tl, p_goal_tr)
        painter.drawLine(p_goal_tr, p_goal_br)

        # 3. 3D Dőlő Robot Kapus kirajzolása a kapuvonal közepén
        p_pivot = self._project_3d(0, 0, 0, w, h)
        painter.save()
        painter.translate(p_pivot)
        painter.rotate(self._gk_tilt_angle_deg)

        gk_color = QColor(16, 185, 129, 230) if self._gk_state == "DEFENDED" else QColor(59, 130, 246, 210)
        painter.setPen(QPen(QColor("#34D399"), 2))
        painter.setBrush(QBrush(gk_color))

        # Dőlő kapus figura a 3D nézetben
        torso_h_3d = (p_goal_bl.y() - p_goal_tl.y()) * 0.6
        painter.drawRoundedRect(QRectF(-12, -torso_h_3d, 24, torso_h_3d), 4, 4)
        painter.drawRoundedRect(QRectF(-35, -torso_h_3d - 16, 70, 24), 6, 6)
        painter.restore()

        # 4. 3D Parabola Labda Trajektória Görbe (Csak aktív detektálás / mérés esetén!)
        if self._impact_x_mm is not None and self._impact_y_mm is not None:
            target_x = self._impact_x_mm
            target_z = self._impact_y_mm

            traj_pts: List[QPointF] = []
            steps = 40
            for i in range(steps + 1):
                t = i / float(steps)  # t=0: rúgó pont (Y=10000), t=1: kapuvonal (Y=0)
                cur_y = 10000.0 * (1.0 - t)
                cur_x = target_x * t
                # Parabola ív emelkedés (csúcs magasság ~1.8m a pálya közepén)
                cur_z = 4.0 * 1800.0 * t * (1.0 - t) + target_z * (t ** 2)

                pt_3d = self._project_3d(cur_x, cur_y, cur_z, w, h)
                traj_pts.append(pt_3d)

            # Fénylő Zöld -> Arany 3D Trajektória Cső
            for i in range(len(traj_pts) - 1):
                t_norm = i / float(steps)
                c_line = QColor(16, 185, 129) if t_norm < 0.5 else QColor(245, 158, 11)
                painter.setPen(QPen(c_line, 3.5, Qt.PenStyle.SolidLine))
                painter.drawLine(traj_pts[i], traj_pts[i + 1])

            # Animált 3D Labda Pozíció a Parabola Íven
            ball_t = (math.sin(self._anim_phase * 0.8) + 1.0) / 2.0  # 0.0 .. 1.0 folyamatos mozgás
            ball_idx = int(ball_t * steps)
            ball_pt = traj_pts[min(steps, max(0, ball_idx))]

            self._draw_yellow_ball_icon(painter, ball_pt.x(), ball_pt.y(), radius=12.0)
        else:
            # Nincs aktív lövés: tiszta pálya felirat
            font_idle = QFont("Segoe UI", 9, QFont.Weight.Bold)
            painter.setFont(font_idle)
            painter.setPen(QPen(QColor("#94A3B8") if self._dark else QColor("#64748B")))
            painter.drawText(QRectF(15, h * 0.74, w - 30, 24), Qt.AlignmentFlag.AlignCenter, "Nincs aktív predikció – várakozás labda detektálásra...")

        # 3D Címke / HUD
        font_hud3d = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font_hud3d)
        painter.setPen(QPen(QColor("#10B981") if self._dark else QColor("#0F5132")))
        painter.drawText(QRectF(12, 30, 280, 20), Qt.AlignmentFlag.AlignLeft, "ÉLŐ 3D PARABOLA TRAJEKTÓRIA (10 MÉTER)")

    def _draw_robot_goalkeeper(self, painter: QPainter, goal_rect: QRectF) -> None:
        """Kirajzolja a középen álló, balra/jobbra dőlő robot kapus szimulációt."""
        x, y, rw, rh = goal_rect.x(), goal_rect.y(), goal_rect.width(), goal_rect.height()

        # Alap pivot csukló: a kapu alsó vonalának közepe (X=0, Y=0 ground line)
        pivot_px = x + rw / 2.0
        pivot_py = y + rh

        painter.save()
        # Áthelyezzük az origót az alsó középső csuklóra
        painter.translate(pivot_px, pivot_py)
        # Elforgatjuk a vásznat a kapus dőlésszögével
        painter.rotate(self._gk_tilt_angle_deg)

        # Állapotfüggő színek
        if self._gk_state == "DEFENDED":
            body_color = QColor(16, 185, 129, 220)  # Smaragdzöld (Védve)
            border_color = QColor("#34D399")
            status_text = "VÉDVE!"
        elif self._gk_state == "MOVING":
            body_color = QColor(245, 158, 11, 210)  # Borostyán (Dőlésben)
            border_color = QColor("#FBBF24")
            status_text = f"DŐLÉS: {self._gk_tilt_angle_deg:+.0f}°"
        elif self._gk_state == "MISSED":
            body_color = QColor(239, 68, 68, 210)   # Piros (Kimaradt)
            border_color = QColor("#FCA5A5")
            status_text = "KIMARADT"
        else:
            body_color = QColor(59, 130, 246, 190)  # Kék (Közép alaphelyzet)
            border_color = QColor("#60A5FA")
            status_text = "KÖZÉP KÉSZ"

        # 1. Alsó mechanikus forgócsukló / Talp
        base_w = 46.0
        base_h = 16.0
        painter.setPen(QPen(QColor("#0F5132"), 1.5))
        painter.setBrush(QBrush(QColor("#10B981") if self._dark else QColor("#0F5132")))
        painter.drawRoundedRect(QRectF(-base_w / 2.0, -base_h, base_w, base_h), 4, 4)

        # 2. Dőlő teleszkópos robot torzó / kar
        torso_h = rh * 0.62   # A kapu magasságának ~62%-a
        torso_w = 26.0

        painter.setPen(QPen(QColor("#475569") if self._dark else QColor("#64748B"), 2))
        painter.setBrush(QBrush(QColor("#1E293B") if self._dark else QColor("#E2E8F0")))
        painter.drawRoundedRect(QRectF(-torso_w / 2.0, -base_h - torso_h, torso_w, torso_h), 6, 6)

        # 3. Felső Védőblokk / Kesztyűk (Shield barrier at the top of torso)
        shield_w = (self._gk_reach_width_mm / self._goal_width_mm) * rw * 0.85
        shield_h = 36.0
        shield_y = -base_h - torso_h - shield_h / 2.0
        shield_rect = QRectF(-shield_w / 2.0, shield_y, shield_w, shield_h)

        painter.setPen(QPen(border_color, 2.5))
        painter.setBrush(QBrush(body_color))
        painter.drawRoundedRect(shield_rect, 8, 8)

        # Bal és jobb kinyúló kapus kesztyűk
        glove_r = 12.0
        painter.drawEllipse(QPointF(-shield_w / 2.0 - 4, shield_y + shield_h / 2.0), glove_r, glove_r)
        painter.drawEllipse(QPointF(shield_w / 2.0 + 4, shield_y + shield_h / 2.0), glove_r, glove_r)

        # Status LED fénye
        led_c = QColor("#4ADE80") if self._gk_state == "DEFENDED" else QColor("#FACC15")
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(led_c))
        painter.drawEllipse(QPointF(0, shield_y + 8), 5, 5)

        # Felirat a pajzson
        font_gk = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(font_gk)
        painter.setPen(QPen(QColor("#FFFFFF")))
        painter.drawText(shield_rect, Qt.AlignmentFlag.AlignCenter, f"ROBOT\n{status_text}")

        painter.restore()

    def _draw_net_and_frame(self, painter: QPainter, goal_rect: QRectF) -> None:
        x, y, rw, rh = goal_rect.x(), goal_rect.y(), goal_rect.width(), goal_rect.height()

        net_color = QColor(50, 65, 90) if self._dark else QColor(203, 213, 225)
        pen_net = QPen(net_color, 1)
        painter.setPen(pen_net)
        cols, rows = 12, 6
        for i in range(1, cols):
            nx = x + i * (rw / cols)
            painter.drawLine(QPointF(nx, y), QPointF(nx, y + rh))
        for j in range(1, rows):
            ny = y + j * (rh / rows)
            painter.drawLine(QPointF(x, ny), QPointF(x + rw, ny))

        painter.setPen(QPen(QColor("#0F5132"), 4))
        painter.drawRect(goal_rect)

        painter.setPen(QPen(QColor("#D97706"), 3))
        painter.drawLine(QPointF(x - 20, y + rh), QPointF(x + rw + 20, y + rh))

        tick_color = QColor("#94A3B8") if self._dark else QColor("#334155")
        font_ticks = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(font_ticks)
        painter.setPen(QPen(tick_color))

        half_w = self._goal_width_mm / 2.0
        for val in [-half_w, -half_w / 2, 0, half_w / 2, half_w]:
            px = x + (val + half_w) / self._goal_width_mm * rw
            painter.drawLine(QPointF(px, y + rh), QPointF(px, y + rh + 5))
            txt = f"{val:+.0f}" if val != 0 else "0"
            painter.drawText(QRectF(px - 30, y + rh + 6, 60, 16), Qt.AlignmentFlag.AlignCenter, txt)

    def _draw_history(self, painter: QPainter, goal_rect: QRectF) -> None:
        x, y, rw, rh = goal_rect.x(), goal_rect.y(), goal_rect.width(), goal_rect.height()
        half_w = self._goal_width_mm / 2.0

        n_shots = len(self._shot_history)
        for idx, (sx, sy, conf, in_g) in enumerate(self._shot_history):
            px = x + (sx + half_w) / self._goal_width_mm * rw
            py = y + rh - (sy / self._goal_height_mm * rh)

            alpha = int(80 + (idx + 1) / n_shots * 160)
            color = QColor("#059669") if in_g else QColor("#DC2626")
            color.setAlpha(alpha)

            painter.setPen(QPen(color, 1.5))
            painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), int(alpha * 0.3))))
            painter.drawEllipse(QPointF(px, py), 5, 5)

    def _draw_yellow_ball_icon(self, painter: QPainter, px: float, py: float, radius: float = 14.0) -> None:
        # 1. Lüktető sárga fényudvar
        glow_r = radius * 2.2 + 3.0 * math.sin(self._anim_phase)
        glow_grad = QRadialGradient(QPointF(px, py), glow_r)
        glow_grad.setColorAt(0.0, QColor(254, 240, 138, 220))
        glow_grad.setColorAt(0.6, QColor(234, 179, 8, 120))
        glow_grad.setColorAt(1.0, QColor(234, 179, 8, 0))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(glow_grad))
        painter.drawEllipse(QPointF(px, py), glow_r, glow_r)

        # 2. Labda test
        ball_grad = QRadialGradient(QPointF(px - radius * 0.3, py - radius * 0.3), radius * 1.5)
        ball_grad.setColorAt(0.0, QColor("#FEF08A"))
        ball_grad.setColorAt(0.6, QColor("#FACC15"))
        ball_grad.setColorAt(1.0, QColor("#EAB308"))

        painter.setPen(QPen(QColor("#713F12"), 1.8))
        painter.setBrush(QBrush(ball_grad))
        painter.drawEllipse(QPointF(px, py), radius, radius)

        # 3. Focilabda ötszög mintázat
        pen_pattern = QPen(QColor("#451A03"), 1.5)
        painter.setPen(pen_pattern)

        r_p = radius * 0.42
        pts_inner = []
        for k in range(5):
            angle = -math.pi / 2.0 + k * (2.0 * math.pi / 5.0)
            pts_inner.append(QPointF(px + r_p * math.cos(angle), py + r_p * math.sin(angle)))

        painter.setBrush(QBrush(QColor("#713F12")))
        painter.drawPolygon(QPolygonF(pts_inner))

        for k in range(5):
            angle = -math.pi / 2.0 + k * (2.0 * math.pi / 5.0)
            outer_p = QPointF(px + radius * math.cos(angle), py + radius * math.sin(angle))
            painter.drawLine(pts_inner[k], outer_p)

    def _draw_active_target(self, painter: QPainter, goal_rect: QRectF) -> None:
        x, y, rw, rh = goal_rect.x(), goal_rect.y(), goal_rect.width(), goal_rect.height()
        half_w = self._goal_width_mm / 2.0

        px = x + (self._impact_x_mm + half_w) / self._goal_width_mm * rw
        py = y + rh - (self._impact_y_mm / self._goal_height_mm * rh)

        self._draw_yellow_ball_icon(painter, px, py, radius=14.0)

        pen_cross = QPen(QColor("#CA8A04"), 1.2, Qt.PenStyle.DashLine)
        painter.setPen(pen_cross)
        painter.drawLine(QPointF(px - 22, py), QPointF(px + 22, py))
        painter.drawLine(QPointF(px, py - 22), QPointF(px, py + 22))

        font_txt = QFont("Consolas", 8, QFont.Weight.Bold)
        painter.setFont(font_txt)
        tag = f"X:{self._impact_x_mm:+.0f} Y:{self._impact_y_mm:.0f}mm ({self._time_to_impact_s:.2f}s)"

        lbl_r = QRectF(px + 18, py - 12, 185, 22)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(QColor(254, 240, 138, 240)))
        painter.drawRoundedRect(lbl_r, 5, 5)
        painter.setPen(QPen(QColor("#854D0E"), 1.5))
        painter.drawRoundedRect(lbl_r, 5, 5)
        painter.setPen(QPen(QColor("#713F12")))
        painter.drawText(lbl_r, Qt.AlignmentFlag.AlignCenter, tag)

    def _draw_hud_overlay_text(self, painter: QPainter, goal_rect: QRectF, w: int, h: int) -> None:
        font_hud = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_hud)

        text_color = QColor("#94A3B8") if self._dark else QColor("#475569")

        stats = self.get_stats()
        if stats["total"] > 0:
            pct = stats["in_goal_pct"]
            hist_str = f" | LÖVÉS: {stats['total']} (GÓL: {stats['in_goal']})"
        else:
            hist_str = f" | LÖVÉSEK: {len(self._shot_history)} DB"

        if self._impact_x_mm is not None:
            st_txt   = f"BECSAPÓDÁS DETEKTÁLVA{hist_str}"
            st_color = QColor("#4ADE80") if self._dark else QColor("#0F5132")
        else:
            st_txt   = f"PÁLYAKÖVETÉS AKTÍV{hist_str}"
            st_color = text_color

        painter.setPen(QPen(st_color))
        painter.drawText(12, 18, st_txt)

    def _draw_confidence_bar(self, painter: QPainter, w: int, h: int, margin_bottom: int) -> None:
        """Konfidencia sáv a kapu rajz alatt."""
        bar_h      = 12
        bar_margin = 10
        bar_y      = h - margin_bottom + 8
        bar_x      = bar_margin + 45
        bar_w      = w - 2 * bar_x

        text_color = QColor("#94A3B8") if self._dark else QColor("#475569")
        bg_color   = QColor("#1E293B") if self._dark else QColor("#E2E8F0")
        border_c   = QColor("#334155") if self._dark else QColor("#CBD5E1")

        font_conf = QFont("Segoe UI", 8, QFont.Weight.Bold)
        painter.setFont(font_conf)

        if self._impact_x_mm is not None and self._impact_conf > 0:
            conf_pct = min(100.0, self._impact_conf * 100.0)

            # Szín a szint alapján
            if conf_pct >= 70:
                bar_color_start = QColor("#10B981")
                bar_color_end   = QColor("#059669")
            elif conf_pct >= 40:
                bar_color_start = QColor("#F59E0B")
                bar_color_end   = QColor("#D97706")
            else:
                bar_color_start = QColor("#EF4444")
                bar_color_end   = QColor("#DC2626")

            # Háttér sáv
            painter.setPen(QPen(border_c, 1))
            painter.setBrush(QBrush(bg_color))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 5, 5)

            # Kitöltött rész – lineáris gradiens
            filled_w = max(16.0, bar_w * conf_pct / 100.0)
            grad = QLinearGradient(bar_x, 0, bar_x + filled_w, 0)
            grad.setColorAt(0.0, bar_color_start)
            grad.setColorAt(1.0, bar_color_end)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(grad))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, filled_w, bar_h), 5, 5)

            # Felirat
            painter.setPen(QPen(text_color))
            label_y = bar_y + bar_h + 14
            label_txt = f"PREDIKCIÓ KONFIDENCIA: {conf_pct:.0f}%"
            painter.drawText(
                QRectF(bar_x, label_y, bar_w, 16),
                Qt.AlignmentFlag.AlignCenter, label_txt
            )

            # Zóna statisztika sor
            stats = self.get_stats()
            zone_txt = (
                f"BAL: {stats['left']}  |  KÖZÉP: {stats['center']}  |  JOBB: {stats['right']}"
            )
            painter.drawText(
                QRectF(bar_x, label_y + 15, bar_w, 14),
                Qt.AlignmentFlag.AlignCenter, zone_txt
            )

        else:
            # Nincs aktív predikció – üres sáv + szöveg
            painter.setPen(QPen(border_c, 1))
            painter.setBrush(QBrush(bg_color))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 5, 5)

            painter.setPen(QPen(text_color))
            label_y = bar_y + bar_h + 14
            stats = self.get_stats()
            if stats["total"] > 0:
                lbl = (
                    f"Lövések összesen: {stats['total']} | Kapura: {stats['in_goal']} "
                    f"({stats['in_goal_pct']:.0f}%)  ·  "
                    f"BAL: {stats['left']}  KÖZÉP: {stats['center']}  JOBB: {stats['right']}"
                )
            else:
                lbl = "Nincs aktív predikció – várakozás labda detektálásra..."
            painter.drawText(
                QRectF(bar_x, label_y, bar_w, 16),
                Qt.AlignmentFlag.AlignCenter, lbl
            )
