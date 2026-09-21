"""
DEIK Robot Foci Kapus – Analitika & Vizuális Grafikonok Dashboard (PyQt6)
========================================================================

Bővített statisztikai és elemző felület:
  - Labda Sebesség & Trajektória Grafikon (km/h, Z magasság)
  - 2D Lövési Hőtérkép (Goal 2D Heatmap)
  - Szektoros eloszlási mutatók (Bal felső, Jobb felső, Bal alsó, Jobb alsó, Közép)
  - Részletes lövési lista (sebesség, becsapódási idő, szektor)
  - Munkamenet Exportálás (CSV & HTML Riport)
  - Munkamenet Perzisztencia (automatikus JSON Lines mentés)
  - Téma-tudatos (Dark Mode & Light Mode)
"""

import base64
import csv
import logging
import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# pyrefly: ignore [missing-import]
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal, pyqtSlot
# pyrefly: ignore [missing-import]
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QLinearGradient, QPainter, QPen, QRadialGradient
)
# pyrefly: ignore [missing-import]
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QFormLayout,
    QLabel, QPushButton, QTableWidget, QTableWidgetItem, QHeaderView,
    QFileDialog, QMessageBox, QFrame, QSplitter
)

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# 2D Kapu Hőtérkép Widget
# --------------------------------------------------------------------------- #

class GoalHeatmapWidget(QWidget):
    """
    2D Kapu Hőtérkép rajzoló widget (10x5 rács sűrűségű hőtérkép kinyerésével).
    """

    GRID_COLS = 10
    GRID_ROWS = 5

    def __init__(self, goal_w_mm: float = 4000.0, goal_h_mm: float = 2000.0, parent=None):
        super().__init__(parent)
        self._goal_w = goal_w_mm
        self._goal_h = goal_h_mm
        self._dark = False
        self._grid = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]
        self._shots: List[dict] = []
        self.setMinimumSize(360, 220)

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self.update()

    def set_shots(self, shots: List[dict]) -> None:
        """Frissíti a hőtérkép rácsát a kapott lövés listával."""
        self._shots = list(shots)
        self._grid = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

        half_w = self._goal_w / 2.0
        for shot in self._shots:
            sx = shot.get("x_mm", 0.0)
            sy = shot.get("y_mm", 0.0)
            norm_x = (sx + half_w) / self._goal_w
            norm_y = sy / self._goal_h

            col = int(math.floor(norm_x * self.GRID_COLS))
            row = int(math.floor((1.0 - norm_y) * self.GRID_ROWS))

            col = max(0, min(self.GRID_COLS - 1, col))
            row = max(0, min(self.GRID_ROWS - 1, row))

            self._grid[row][col] += 1

        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        bg_c = QColor("#151D2A") if self._dark else QColor("#FFFFFF")
        painter.fillRect(0, 0, w, h, QBrush(bg_c))

        margin = 35
        gw = w - 2 * margin
        gh = h - 2 * margin

        rect = QRectF(margin, margin, gw, gh)
        cell_w = gw / self.GRID_COLS
        cell_h = gh / self.GRID_ROWS

        max_val = max(1, max(max(row) for row in self._grid))

        # Hő-cellák kirajzolása
        for r in range(self.GRID_ROWS):
            for c in range(self.GRID_COLS):
                count = self._grid[r][c]
                val_norm = count / float(max_val) if count > 0 else 0.0

                cell_r = QRectF(margin + c * cell_w, margin + r * cell_h, cell_w, cell_h)

                if count == 0:
                    fill_c = QColor("#1E293B") if self._dark else QColor("#F1F5F9")
                else:
                    # Kék -> Zöld -> Sárga -> Piros gadiens
                    if val_norm < 0.33:
                        fill_c = QColor(30, 144, 255, int(100 + val_norm * 450))
                    elif val_norm < 0.66:
                        fill_c = QColor(255, 215, 0, int(150 + val_norm * 150))
                    else:
                        fill_c = QColor(239, 68, 68, int(180 + val_norm * 75))

                painter.setPen(QPen(QColor("#26334D") if self._dark else QColor("#CBD5E1"), 1))
                painter.setBrush(QBrush(fill_c))
                painter.drawRect(cell_r)

                if count > 0:
                    painter.setFont(QFont("Consolas", 8, QFont.Weight.Bold))
                    painter.setPen(QPen(QColor("#FFFFFF") if (self._dark or val_norm > 0.3) else QColor("#0F172A")))
                    painter.drawText(cell_r, Qt.AlignmentFlag.AlignCenter, str(count))

        # Keret & tengely feliratok
        painter.setPen(QPen(QColor("#0F5132"), 3))
        painter.drawRect(rect)

        font_lbl = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font_lbl)
        painter.setPen(QPen(QColor("#4ADE80") if self._dark else QColor("#0F5132")))
        painter.drawText(QRectF(margin, 8, gw, 20), Qt.AlignmentFlag.AlignCenter, "2D LÖVÉSI HŐTÉRKÉP (LÖVÉSEK SŰRŰSÉGE)")


# --------------------------------------------------------------------------- #
# Sebesség & Trajektória Grafikon Widget
# --------------------------------------------------------------------------- #

class SpeedPlotWidget(QWidget):
    """
    Labda sebesség (km/h) és Z magasság (mm) valós idejű trajektória grafikonja.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dark = False
        self._speed_history: List[float] = []
        self._height_history: List[float] = []
        self.setMinimumSize(360, 220)

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self.update()

    def add_data_point(self, speed_kmh: float, height_mm: float) -> None:
        self._speed_history.append(speed_kmh)
        self._height_history.append(height_mm)
        if len(self._speed_history) > 60:
            self._speed_history.pop(0)
            self._height_history.pop(0)
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        bg_c = QColor("#151D2A") if self._dark else QColor("#FFFFFF")
        painter.fillRect(0, 0, w, h, QBrush(bg_c))

        margin = 35
        gw = w - 2 * margin
        gh = h - 2 * margin

        # Cím
        painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#4ADE80") if self._dark else QColor("#0F5132")))
        painter.drawText(QRectF(margin, 8, gw, 20), Qt.AlignmentFlag.AlignCenter, "SEBESSÉG (KM/H) ÉS LABDAMAGASSÁG Z (MM)")

        # Keret
        rect = QRectF(margin, margin, gw, gh)
        painter.setPen(QPen(QColor("#26334D") if self._dark else QColor("#CBD5E1"), 1))
        painter.drawRect(rect)

        if not self._speed_history:
            painter.setPen(QPen(QColor("#94A3B8")))
            painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, "Nincs mért trajektória adat – várakozás lövésre...")
            return

        n = len(self._speed_history)
        max_spd = max(60.0, max(self._speed_history))
        dx = gw / max(1, n - 1)

        # 1. Sebesség Görbe (Zöld)
        pen_spd = QPen(QColor("#10B981"), 2.5)
        painter.setPen(pen_spd)
        for i in range(n - 1):
            x1 = margin + i * dx
            y1 = margin + gh - (self._speed_history[i] / max_spd * gh)
            x2 = margin + (i + 1) * dx
            y2 = margin + gh - (self._speed_history[i + 1] / max_spd * gh)
            painter.drawLine(QPointF(x1, y1), QPointF(x2, y2))

        # Legutóbbi Érték Kiírása
        curr_spd = self._speed_history[-1]
        painter.setFont(QFont("Consolas", 9, QFont.Weight.Bold))
        painter.setPen(QPen(QColor("#34D399")))
        painter.drawText(int(margin + gw - 120), int(margin + 20), f"Sebesség: {curr_spd:.1f} km/h")


# --------------------------------------------------------------------------- #
# Fő Analitika Dashboard Widget
# --------------------------------------------------------------------------- #

class AnalyticsDashboardWidget(QWidget):
    """
    Összesített Analitika Dashboard: Hőtérkép, Görbék, Statisztikai Táblázat, CSV/HTML Export.
    Bővített adatmodell: sebesség, becsapódási idő, szektor rögzítése lövésenként.
    """

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self._dark = False

        geo_cfg = config.get("geometry", {})
        self._goal_w = float(geo_cfg.get("goal_width_mm", 4000.0))
        self._goal_h = float(geo_cfg.get("goal_height_mm", 2000.0))

        # Bővített lövési rekordok (dict lista)
        self._shot_records: List[dict] = []
        self._build_ui()

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self._heatmap.set_dark(dark)
        self._speed_plot.set_dark(dark)
        self._apply_theme()

    def add_shot_event(
        self,
        x_mm: float,
        y_mm: float,
        conf: float,
        in_goal: bool,
        speed_kmh: float = 0.0,
        time_to_impact_s: float = 0.0,
        sector: str = "",
        # ── Bővített telemetria mezők ────────────────────────────────────
        y_source: str = "mért",            # "pred" = impact.y_mm, "mért" = pos_3d[1]
        detection_latency_ms: float = 0.0, # kamera frame → YOLO inferencia befejezése
        total_pipeline_ms: float = 0.0,    # frame → GUI megjelenítés (teljes pipeline)
        goalkeeper_reaction_ms: float = 0.0,  # lövés megerősítés → kapus parancs küldés
        goalkeeper_x_cmd_mm: float = 0.0,  # merre lett küldve a kapus (X)
        goalkeeper_y_cmd_mm: float = 0.0,  # merre lett küldve a kapus (Y)
        goalkeeper_travel_ms: float = 0.0, # becsült kapus mozgási idő (távolság/sebesség)
        vx_mm_s: float = 0.0,             # sebességvektor X komponens
        vy_mm_s: float = 0.0,             # sebességvektor Y komponens
        vz_mm_s: float = 0.0,             # sebességvektor Z komponens (negatív = kapu felé)
        z_start_mm: float = 0.0,          # Z pozíció lövés kezdetén
        z_end_mm: float = 0.0,            # Z pozíció lövés végén (kapu közelében)
        det_method: str = "YOLO",         # detektálási módszer: YOLO / HSV / OptFlow
        conf_left: float = 0.0,           # bal kamera detektálási konfidencia
        conf_right: float = 0.0,          # jobb kamera detektálási konfidencia
    ) -> dict:
        """Hozzáad egy új lövés eseményt a bővített telemetria gyűjteményhez."""
        if not sector:
            sector = self._determine_sector(x_mm, y_mm)

        record = {
            # ── Alap azonosítók ─────────────────────────────────────────
            "timestamp": time.strftime("%H:%M:%S"),
            "timestamp_full": time.strftime("%Y-%m-%d %H:%M:%S"),
            # ── Becsapódási pozíció ──────────────────────────────────────
            "x_mm": round(x_mm, 1),
            "y_mm": round(y_mm, 1),
            "y_source": y_source,          # honnan jön az Y érték
            "conf": round(conf, 3),
            "in_goal": in_goal,
            "sector": sector,
            # ── Labda fizika ─────────────────────────────────────────────
            "speed_kmh": round(speed_kmh, 2),
            "time_to_impact_s": round(time_to_impact_s, 4),
            "vx_mm_s": round(vx_mm_s, 1),
            "vy_mm_s": round(vy_mm_s, 1),
            "vz_mm_s": round(vz_mm_s, 1),
            "z_start_mm": round(z_start_mm, 1),
            "z_end_mm": round(z_end_mm, 1),
            # ── Pipeline időzítés ────────────────────────────────────────
            "detection_latency_ms": round(detection_latency_ms, 2),
            "total_pipeline_ms": round(total_pipeline_ms, 2),
            # ── Kapus reakció ────────────────────────────────────────────
            "goalkeeper_reaction_ms": round(goalkeeper_reaction_ms, 2),
            "goalkeeper_x_cmd_mm": round(goalkeeper_x_cmd_mm, 1),
            "goalkeeper_y_cmd_mm": round(goalkeeper_y_cmd_mm, 1),
            "goalkeeper_travel_ms": round(goalkeeper_travel_ms, 2),
            # ── Detektálás minősége ──────────────────────────────────────
            "det_method": det_method,
            "conf_left": round(conf_left, 3),
            "conf_right": round(conf_right, 3),
        }
        self._shot_records.append(record)
        self._heatmap.set_shots(self._shot_records)
        self._speed_plot.add_data_point(speed_kmh, y_mm)
        self._update_table_and_stats()
        return record  # Visszaadjuk a session manager számára

    def get_shot_records(self) -> List[dict]:
        """Visszaadja az összes lövési rekordot."""
        return list(self._shot_records)

    def _determine_sector(self, x_mm: float, y_mm: float) -> str:
        """Meghatározza a becsapódási szektort az X,Y koordináták alapján."""
        horiz = "BAL" if x_mm < -600 else ("JOBB" if x_mm > 600 else "KÖZÉP")
        vert = "FELSŐ" if y_mm > 1000 else "ALSÓ"
        return f"{horiz} {vert}"

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # 1. Fejléc gombok (Exportálás & Törlés)
        top_bar = QHBoxLayout()
        lbl_title = QLabel("Elemzés & Munkamenet Analitika")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: 900; color: #10B981;")

        btn_export_csv = QPushButton("Export CSV")
        btn_export_csv.setFixedHeight(34)
        btn_export_csv.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_export_csv.setStyleSheet(
            "QPushButton { background-color: #0F5132; color: #FFFFFF; font-weight: 800; "
            "border-radius: 6px; padding: 0 14px; border: none; }"
            "QPushButton:hover { background-color: #146C43; }"
        )
        btn_export_csv.clicked.connect(self._export_csv)

        btn_export_html = QPushButton("Export HTML Riport")
        btn_export_html.setFixedHeight(34)
        btn_export_html.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_export_html.setStyleSheet(
            "QPushButton { background-color: #D97706; color: #FFFFFF; font-weight: 800; "
            "border-radius: 6px; padding: 0 14px; border: none; }"
            "QPushButton:hover { background-color: #B45309; }"
        )
        btn_export_html.clicked.connect(self._export_html_report)

        btn_clear = QPushButton("Statisztika Nullázás")
        btn_clear.setFixedHeight(34)
        btn_clear.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_clear.setStyleSheet(
            "QPushButton { background-color: #1E293B; color: #94A3B8; font-weight: 700; "
            "border-radius: 6px; padding: 0 14px; border: 1px solid #334155; }"
            "QPushButton:hover { background-color: #DC2626; color: #FFFFFF; border-color: #EF4444; }"
        )
        btn_clear.clicked.connect(self._clear_analytics)

        top_bar.addWidget(lbl_title, stretch=1)
        top_bar.addWidget(btn_export_csv)
        top_bar.addWidget(btn_export_html)
        top_bar.addWidget(btn_clear)
        main_layout.addLayout(top_bar)

        # 2. Vizuális Grafikonok (Hőtérkép & Sebesség Görbe)
        visual_box = QHBoxLayout()
        visual_box.setSpacing(12)

        self._heatmap = GoalHeatmapWidget(self._goal_w, self._goal_h)
        self._speed_plot = SpeedPlotWidget()

        visual_box.addWidget(self._heatmap, stretch=1)
        visual_box.addWidget(self._speed_plot, stretch=1)
        main_layout.addLayout(visual_box)

        # 3. Statisztikai Összesítő Kártyák & Táblázat
        details_box = QHBoxLayout()
        details_box.setSpacing(12)

        # Statisztikai Mutatók Kártya
        stats_grp = QGroupBox("Összesített Mutatók")
        stats_form = QFormLayout(stats_grp)

        self._lbl_stat_total = QLabel("0 db")
        self._lbl_stat_ingoal = QLabel("0 db (0%)")
        self._lbl_stat_avg_speed = QLabel("— km/h")
        self._lbl_stat_max_speed = QLabel("— km/h")
        self._lbl_stat_avg_time = QLabel("— mp")
        self._lbl_stat_sectors = QLabel("BAL: 0 | KÖZÉP: 0 | JOBB: 0")
        self._lbl_stat_session = QLabel("Munkamenet: 0 perc")

        stats_form.addRow("Összes Lövés:", self._lbl_stat_total)
        stats_form.addRow("Kaput Talált:", self._lbl_stat_ingoal)
        stats_form.addRow("Átlag Sebesség:", self._lbl_stat_avg_speed)
        stats_form.addRow("Max Sebesség:", self._lbl_stat_max_speed)
        stats_form.addRow("Átlag Becsap. Idő:", self._lbl_stat_avg_time)
        stats_form.addRow("Zóna Eloszlás:", self._lbl_stat_sectors)
        stats_form.addRow("Munkamenet:", self._lbl_stat_session)

        # Lövési Lista Táblázat
        table_grp = QGroupBox("Legutóbbi Lövések Részletes Listája – Teljes Telemetria")
        table_box = QVBoxLayout(table_grp)

        self._table = QTableWidget(0, 12)
        self._table.setHorizontalHeaderLabels([
            "Időpont", "X (mm)", "Y (mm)", "Sebesség",
            "Becsap. Idő", "Det. Latencia", "Pipeline",
            "Kapus Reakció", "Kapus Cél", "Det. Módszer",
            "Szektor", "Eredmény"
        ])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        self._table.verticalHeader().setVisible(False)
        self._table.setAlternatingRowColors(True)
        table_box.addWidget(self._table)

        details_box.addWidget(stats_grp, stretch=1)
        details_box.addWidget(table_grp, stretch=2)
        main_layout.addLayout(details_box)

        self._apply_theme()

    def _update_table_and_stats(self) -> None:
        n = len(self._shot_records)
        in_g = sum(1 for r in self._shot_records if r.get("in_goal", False))
        pct = (in_g / n * 100.0) if n > 0 else 0.0

        # Sebességek és becsapódási idők
        speeds = [r.get("speed_kmh", 0.0) for r in self._shot_records if r.get("speed_kmh", 0.0) > 0]
        times = [r.get("time_to_impact_s", 0.0) for r in self._shot_records if r.get("time_to_impact_s", 0.0) > 0]
        reactions = [r.get("goalkeeper_reaction_ms", 0.0) for r in self._shot_records if r.get("goalkeeper_reaction_ms", 0.0) > 0]
        pipelines = [r.get("total_pipeline_ms", 0.0) for r in self._shot_records if r.get("total_pipeline_ms", 0.0) > 0]

        # Szektor statisztika
        left = sum(1 for r in self._shot_records if "BAL" in r.get("sector", ""))
        center = sum(1 for r in self._shot_records if "KÖZÉP" in r.get("sector", ""))
        right = sum(1 for r in self._shot_records if "JOBB" in r.get("sector", ""))

        self._lbl_stat_total.setText(f"{n} db")
        self._lbl_stat_ingoal.setText(f"{in_g} db ({pct:.0f}%)")

        if speeds:
            self._lbl_stat_avg_speed.setText(f"{sum(speeds)/len(speeds):.1f} km/h")
            self._lbl_stat_max_speed.setText(f"{max(speeds):.1f} km/h")
        else:
            self._lbl_stat_avg_speed.setText("— km/h")
            self._lbl_stat_max_speed.setText("— km/h")

        if times:
            self._lbl_stat_avg_time.setText(f"{sum(times)/len(times):.3f} mp")
        else:
            self._lbl_stat_avg_time.setText("— mp")

        self._lbl_stat_sectors.setText(f"BAL: {left} | KÖZÉP: {center} | JOBB: {right}")

        # Munkamenet infó bővítése
        extra_parts = []
        if reactions:
            extra_parts.append(f"Átl. reakció: {sum(reactions)/len(reactions):.0f} ms")
        if pipelines:
            extra_parts.append(f"Átl. pipeline: {sum(pipelines)/len(pipelines):.0f} ms")
        if extra_parts and hasattr(self, "_lbl_stat_session"):
            self._lbl_stat_session.setText("  |  ".join(extra_parts))

        # Táblázat frissítés (12 oszlop)
        self._table.setRowCount(0)
        for i, rec in enumerate(reversed(self._shot_records)):
            self._table.insertRow(i)
            ig = rec.get("in_goal", False)
            res_str = "KAPUBAN ✓" if ig else "MELLÉ ✕"

            # 0: Időpont
            self._table.setItem(i, 0, QTableWidgetItem(rec.get("timestamp", "—")))

            # 1: X (mm)
            self._table.setItem(i, 1, QTableWidgetItem(f"{rec.get('x_mm', 0):+.0f}"))

            # 2: Y (mm) – forrás jelzéssel
            y_src = rec.get("y_source", "")
            y_lbl = f"{rec.get('y_mm', 0):.0f}"
            if y_src == "pred":
                y_lbl += " ⬡"
            item_y = QTableWidgetItem(y_lbl)
            item_y.setToolTip("⬡ = ballisztikus előrejelzés (impact.y_mm)" if y_src == "pred"
                              else "Mért sztereó Y pozíció")
            self._table.setItem(i, 2, item_y)

            # 3: Sebesség
            spd = rec.get("speed_kmh", 0.0)
            self._table.setItem(i, 3, QTableWidgetItem(f"{spd:.1f} km/h" if spd > 0 else "—"))

            # 4: Becsapódási idő
            tti = rec.get("time_to_impact_s", 0.0)
            self._table.setItem(i, 4, QTableWidgetItem(f"{tti:.3f} s" if tti > 0 else "—"))

            # 5: Detektálási latencia
            det_lat = rec.get("detection_latency_ms", 0.0)
            self._table.setItem(i, 5, QTableWidgetItem(f"{det_lat:.1f} ms" if det_lat > 0 else "—"))

            # 6: Teljes pipeline
            pipe = rec.get("total_pipeline_ms", 0.0)
            self._table.setItem(i, 6, QTableWidgetItem(f"{pipe:.1f} ms" if pipe > 0 else "—"))

            # 7: Kapus reakcióidő
            react = rec.get("goalkeeper_reaction_ms", 0.0)
            item_react = QTableWidgetItem(f"{react:.1f} ms" if react > 0 else "—")
            if 0 < react <= 50:
                item_react.setForeground(QColor("#4ADE80"))  # Zöld: kiváló
            elif 0 < react <= 100:
                item_react.setForeground(QColor("#FCD34D"))  # Sárga: jó
            elif react > 100:
                item_react.setForeground(QColor("#F87171"))  # Piros: lassú
            self._table.setItem(i, 7, item_react)

            # 8: Kapus célpozíció
            gx = rec.get("goalkeeper_x_cmd_mm", 0.0)
            gy = rec.get("goalkeeper_y_cmd_mm", 0.0)
            if gx != 0.0 or gy != 0.0:
                self._table.setItem(i, 8, QTableWidgetItem(f"X:{gx:+.0f} Y:{gy:.0f}"))
            else:
                self._table.setItem(i, 8, QTableWidgetItem("—"))

            # 9: Detektálási módszer
            self._table.setItem(i, 9, QTableWidgetItem(rec.get("det_method", "—")))

            # 10: Szektor
            self._table.setItem(i, 10, QTableWidgetItem(rec.get("sector", "—")))

            # 11: Eredmény
            item_res = QTableWidgetItem(res_str)
            item_res.setForeground(QColor("#10B981") if ig else QColor("#EF4444"))
            self._table.setItem(i, 11, item_res)

    @pyqtSlot()
    def _clear_analytics(self) -> None:
        self._shot_records.clear()
        self._heatmap.set_shots([])
        self._update_table_and_stats()

    @pyqtSlot()
    def _export_csv(self) -> None:
        if not self._shot_records:
            QMessageBox.warning(self, "Nincs Adat", "Nincs menthető lövési adat!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "Lövési Statisztika Export (CSV)", "deik_shots.csv", "CSV (*.csv)")
        if path:
            try:
                with open(path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        "Timestamp", "X_mm", "Y_mm", "Y_Source", "Confidence", "InGoal",
                        "Speed_kmh", "VX_mm_s", "VY_mm_s", "VZ_mm_s",
                        "TimeToImpact_s", "Z_Start_mm", "Z_End_mm",
                        "Detection_Latency_ms", "Total_Pipeline_ms",
                        "Goalkeeper_Reaction_ms", "Goalkeeper_X_cmd_mm", "Goalkeeper_Y_cmd_mm",
                        "Goalkeeper_Travel_ms",
                        "Det_Method", "Conf_Left", "Conf_Right",
                        "Sector"
                    ])
                    for rec in self._shot_records:
                        writer.writerow([
                            rec.get("timestamp_full", ""),
                            rec.get("x_mm", 0.0),
                            rec.get("y_mm", 0.0),
                            rec.get("y_source", "mért"),
                            rec.get("conf", 0.0),
                            rec.get("in_goal", False),
                            rec.get("speed_kmh", 0.0),
                            rec.get("vx_mm_s", 0.0),
                            rec.get("vy_mm_s", 0.0),
                            rec.get("vz_mm_s", 0.0),
                            rec.get("time_to_impact_s", 0.0),
                            rec.get("z_start_mm", 0.0),
                            rec.get("z_end_mm", 0.0),
                            rec.get("detection_latency_ms", 0.0),
                            rec.get("total_pipeline_ms", 0.0),
                            rec.get("goalkeeper_reaction_ms", 0.0),
                            rec.get("goalkeeper_x_cmd_mm", 0.0),
                            rec.get("goalkeeper_y_cmd_mm", 0.0),
                            rec.get("goalkeeper_travel_ms", 0.0),
                            rec.get("det_method", "YOLO"),
                            rec.get("conf_left", 0.0),
                            rec.get("conf_right", 0.0),
                            rec.get("sector", ""),
                        ])
                QMessageBox.information(self, "Export Sikeres", f"Lövési adatok mentve:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Hiba", str(e))

    @pyqtSlot()
    def _export_html_report(self) -> None:
        if not self._shot_records:
            QMessageBox.warning(self, "Nincs Adat", "Nincs menthető lövési adat!")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Profi HTML Riport Mentése",
            "deik_session_report.html", "HTML (*.html)"
        )
        if not path:
            return
        try:
            html = self._build_html_report()
            with open(path, "w", encoding="utf-8") as f:
                f.write(html)
            QMessageBox.information(self, "Export Sikeres", f"Profi HTML Riport mentve:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Export Hiba", str(e))

    def _build_html_report(self) -> str:
        """Profi, hivatkos HTML riportot generál logóval, KPI kártyákkal és teljes telemetriával."""
        # ── Logók base64 beágyazása ─────────────────────────────────────────
        assets_dir = Path(__file__).parent.parent.parent / "assets"
        def _img_b64(fname: str) -> str:
            p = assets_dir / fname
            if p.exists():
                with open(p, "rb") as f:
                    return "data:image/png;base64," + base64.b64encode(f.read()).decode()
            return ""
        deik_img = _img_b64("deik_logo.png")
        rgk_img  = _img_b64("logo.png")

        # ── Statisztikák ────────────────────────────────────────────────────
        recs = self._shot_records
        n = len(recs)
        in_g = sum(1 for r in recs if r.get("in_goal", False))
        pct = in_g / n * 100 if n > 0 else 0.0
        speeds = [r.get("speed_kmh", 0.0) for r in recs if r.get("speed_kmh", 0.0) > 0]
        avg_spd = sum(speeds) / len(speeds) if speeds else 0.0
        max_spd = max(speeds) if speeds else 0.0
        min_spd = min(speeds) if speeds else 0.0
        ttis = [r.get("time_to_impact_s", 0.0) for r in recs if r.get("time_to_impact_s", 0.0) > 0]
        avg_tti = sum(ttis) / len(ttis) if ttis else 0.0
        reacts = [r.get("goalkeeper_reaction_ms", 0.0) for r in recs if r.get("goalkeeper_reaction_ms", 0.0) > 0]
        avg_react = sum(reacts) / len(reacts) if reacts else 0.0
        pipes = [r.get("total_pipeline_ms", 0.0) for r in recs if r.get("total_pipeline_ms", 0.0) > 0]
        avg_pipe = sum(pipes) / len(pipes) if pipes else 0.0
        lats = [r.get("detection_latency_ms", 0.0) for r in recs if r.get("detection_latency_ms", 0.0) > 0]
        avg_lat = sum(lats) / len(lats) if lats else 0.0

        sector_counts: dict = {}
        for r in recs:
            s = r.get("sector", "—")
            sector_counts[s] = sector_counts.get(s, 0) + 1

        now_str = time.strftime("%Y. %m. %d. %H:%M:%S")

        def rcol(ms: float) -> str:
            if ms <= 0:  return "#94A3B8"
            if ms <= 5:  return "#10B981"
            if ms <= 10: return "#F59E0B"
            return "#EF4444"

        def scol(k: float) -> str:
            if k >= 90: return "#EF4444"
            if k >= 65: return "#F59E0B"
            return "#10B981"

        # ── Hőtérkép SVG generálás ──────────────────────────────────────────
        GW, GH, GC, GR = 4000.0, 2000.0, 8, 4
        grid = [[0]*GC for _ in range(GR)]
        for r in recs:
            col_i = max(0, min(GC-1, int((r.get("x_mm",0) + GW/2) / GW * GC)))
            row_i = max(0, min(GR-1, int((1 - r.get("y_mm",0)/GH) * GR)))
            grid[row_i][col_i] += 1
        mg_val = max(max(row) for row in grid) or 1
        SVG_W, SVG_H, MG = 520, 240, 24
        gw_px = SVG_W - 2*MG; gh_px = SVG_H - 2*MG
        cw = gw_px / GC; ch = gh_px / GR

        def heat_col(v: int):
            if v == 0: return "#060C14", 0.0
            p = v / mg_val
            if p < 0.33: return "rgb(30,144,255)", 0.4 + p*1.5
            if p < 0.66: return "rgb(255,200,0)", 0.5 + p
            return "rgb(239,68,68)", 0.7 + p*0.3

        cells = ""
        for ri in range(GR):
            for ci in range(GC):
                cnt = grid[ri][ci]; cf, op = heat_col(cnt)
                x0 = MG + ci*cw; y0 = MG + ri*ch
                cells += (f'<rect x="{x0:.1f}" y="{y0:.1f}" width="{cw:.1f}" height="{ch:.1f}" '
                          f'fill="{cf}" opacity="{op:.2f}" stroke="#1A3526" stroke-width="0.5"/>')
                if cnt > 0:
                    cells += (f'<text x="{x0+cw/2:.1f}" y="{y0+ch/2+5:.1f}" '
                               f'text-anchor="middle" font-size="14" font-weight="bold" fill="white">{cnt}</text>')

        dots = ""
        for r in recs:
            sx = MG + (r.get("x_mm",0) + GW/2) / GW * gw_px
            sy = MG + (1 - r.get("y_mm",0)/GH) * gh_px
            color = "#10B981" if r.get("in_goal") else "#EF4444"
            dots += f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5.5" fill="{color}" stroke="white" stroke-width="1.5" opacity="0.9"/>'

        # ── Szektor sávok ───────────────────────────────────────────────────
        sector_bars = ""
        for sec, cnt in sorted(sector_counts.items(), key=lambda x: -x[1]):
            p = cnt / n * 100 if n > 0 else 0
            sector_bars += (f'<div class="sr"><span class="sn">{sec}</span>'
                            f'<div class="sbw"><div class="sb" style="width:{p:.0f}%"></div></div>'
                            f'<span class="sc">{cnt} ({p:.0f}%)</span></div>')

        # ── Sebességprofil bárok ────────────────────────────────────────────
        speed_bars = ""
        for idx, r in enumerate(recs, 1):
            spd = r.get("speed_kmh", 0.0)
            pw = min(100, spd / 120 * 100)
            if spd >= 90:   bc = "linear-gradient(90deg,#DC2626,#EF4444)"
            elif spd >= 65: bc = "linear-gradient(90deg,#D97706,#F59E0B)"
            else:           bc = "linear-gradient(90deg,#059669,#10B981)"
            mk = " ✓" if r.get("in_goal") else " ✕"
            speed_bars += (f'<div class="si"><span class="snum">#{idx}</span>'
                           f'<div class="sbw"><div class="spb" style="width:{pw:.0f}%;background:{bc};">'
                           f'<span>{spd:.1f} km/h{mk}</span></div></div>'
                           f'<span class="sts">{r.get("timestamp","")}</span></div>')

        # ── Lövési tábla sorok ──────────────────────────────────────────────
        rows_html = ""
        for idx, r in enumerate(recs, 1):
            ig = r.get("in_goal", False)
            res_cls = "in-goal" if ig else "missed"
            res_txt = "KAPUBAN ✓" if ig else "MELLÉ ✕"
            y_src = r.get("y_source", "")
            yb = ('<span class="badge badge-pred">⬡ pred</span>' if y_src == "pred"
                  else '<span class="badge badge-mert">mért</span>')
            spd  = r.get("speed_kmh", 0.0)
            tti  = r.get("time_to_impact_s", 0.0)
            dlat = r.get("detection_latency_ms", 0.0)
            pipe = r.get("total_pipeline_ms", 0.0)
            react= r.get("goalkeeper_reaction_ms", 0.0)
            gx   = r.get("goalkeeper_x_cmd_mm", 0.0)
            gy   = r.get("goalkeeper_y_cmd_mm", 0.0)
            rows_html += (
                f'<tr><td class="num">{idx}</td><td>{r.get("timestamp","")}</td>'
                f'<td class="mono" style="color:#60A5FA;">{r.get("x_mm",0):+.0f}</td>'
                f'<td class="mono">{r.get("y_mm",0):.0f} {yb}</td>'
                f'<td class="mono" style="color:{scol(spd)};font-weight:700;">{spd:.1f} km/h</td>'
                f'<td class="mono">{tti:.3f} s</td>'
                f'<td class="mono">{dlat:.1f} ms</td>'
                f'<td class="mono">{pipe:.1f} ms</td>'
                f'<td class="mono" style="color:{rcol(react)};font-weight:700;">{react:.1f} ms</td>'
                f'<td class="mono">{gx:+.0f} / {gy:.0f}</td>'
                f'<td>{r.get("det_method","—")}</td>'
                f'<td>{r.get("sector","—")}</td>'
                f'<td class="{res_cls}">{res_txt}</td></tr>'
            )

        # ── Logó HTML ───────────────────────────────────────────────────────
        deik_tag = f'<img src="{deik_img}" alt="DE logó">' if deik_img else \
                   '<div style="width:88px;height:88px;background:#1E3A2F;border-radius:8px;"></div>'
        rgk_tag  = f'<img src="{rgk_img}" alt="RGK logó" style="height:78px;">' if rgk_img else ""
        ftr_logo = f'<img src="{deik_img}" alt="DEIK" class="ftr-logo">' if deik_img else ""

        # ── Összefűzés ──────────────────────────────────────────────────────
        return f"""<!DOCTYPE html>
<html lang="hu">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DEIK Robot Kapus – Edzés Munkamenet Riport</title>
<style>
:root{{--bg:#080E19;--sf:#0F1A27;--sf2:#172030;--bd:#1A3528;--gp:#10B981;--glt:#4ADE80;
--gd:#F59E0B;--gdl:#FCD34D;--bl:#60A5FA;--rd:#EF4444;--tx:#F1F5F9;--tm:#94A3B8;--td:#475569;--r:12px;}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0;}}
body{{font-family:'Segoe UI',system-ui,sans-serif;background:var(--bg);color:var(--tx);}}
.hdr{{background:linear-gradient(135deg,#071510 0%,#0C2016 60%,#050E0A 100%);border-bottom:3px solid var(--gp);}}
.hdr-i{{max-width:1400px;margin:0 auto;padding:30px 48px;display:flex;align-items:center;gap:32px;}}
.hdr-logos{{display:flex;align-items:center;gap:18px;flex-shrink:0;}}
.hdr-logos img{{height:88px;width:auto;}}
.ldiv{{width:2px;height:68px;background:linear-gradient(to bottom,transparent,#10B981,transparent);opacity:.45;}}
.hdr-t{{flex:1;}}.hdr-t .sup{{font-size:10px;font-weight:700;letter-spacing:3.5px;text-transform:uppercase;color:var(--glt);opacity:.75;margin-bottom:5px;}}
.hdr-t h1{{font-size:26px;font-weight:900;color:var(--tx);line-height:1.15;margin-bottom:8px;}}
.hdr-t h1 span{{color:var(--gp);}}.hdr-t .sub{{font-size:13px;color:var(--tm);}}
.hdr-m{{display:flex;flex-direction:column;align-items:flex-end;gap:7px;flex-shrink:0;}}
.chip{{background:rgba(16,185,129,.12);border:1px solid rgba(16,185,129,.3);border-radius:20px;padding:4px 14px;font-size:11px;color:var(--glt);font-weight:700;}}
.mdate{{font-size:12px;color:var(--tm);font-family:'Courier New',monospace;}}
.msess{{font-size:10px;color:var(--td);font-family:'Courier New',monospace;}}
.sbadge{{background:rgba(16,185,129,.08);border:1px solid rgba(16,185,129,.2);border-radius:6px;padding:4px 10px;font-size:10px;color:var(--gp);font-family:'Courier New',monospace;font-weight:700;}}
.con{{max-width:1400px;margin:0 auto;padding:40px 48px;}}
.stitle{{font-size:10px;font-weight:700;letter-spacing:3px;text-transform:uppercase;color:var(--gp);margin-bottom:18px;display:flex;align-items:center;gap:12px;}}
.stitle::after{{content:'';flex:1;height:1px;background:linear-gradient(to right,var(--bd),transparent);}}
.kpi{{display:grid;grid-template-columns:repeat(6,1fr);gap:14px;margin-bottom:36px;}}
.kc{{background:var(--sf);border:1px solid var(--bd);border-radius:var(--r);padding:18px 14px;text-align:center;position:relative;overflow:hidden;}}
.kc::before{{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:var(--r) var(--r) 0 0;}}
.kc.g::before{{background:var(--gp);}}.kc.a::before{{background:var(--gd);}}.kc.b::before{{background:var(--bl);}}
.kc.r::before{{background:var(--rd);}}.kc.t::before{{background:#06B6D4;}}.kc.p::before{{background:#A78BFA;}}
.ki{{font-size:20px;margin-bottom:7px;display:block;}}.kv{{font-size:24px;font-weight:900;line-height:1;margin-bottom:4px;font-family:'Courier New',monospace;}}
.kc.g .kv{{color:var(--glt);}}.kc.a .kv{{color:var(--gdl);}}.kc.b .kv{{color:var(--bl);}}
.kc.r .kv{{color:#FCA5A5;}}.kc.t .kv{{color:#67E8F9;}}.kc.p .kv{{color:#C4B5FD;}}
.kl{{font-size:9px;font-weight:700;letter-spacing:1.5px;text-transform:uppercase;color:var(--td);}}
.tc{{display:grid;grid-template-columns:1fr 1fr;gap:22px;margin-bottom:36px;}}
.pnl{{background:var(--sf);border:1px solid var(--bd);border-radius:var(--r);padding:26px;}}
.pnl-t{{font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--gp);margin-bottom:18px;padding-bottom:10px;border-bottom:1px solid var(--bd);}}
.hmw{{border-radius:8px;overflow:hidden;background:#060C14;border:1px solid var(--bd);}}
.hml{{font-size:9px;color:var(--tm);text-align:center;padding:5px 0 3px;font-weight:700;letter-spacing:1px;text-transform:uppercase;}}
.hmgl{{display:flex;justify-content:space-between;padding:2px 24px;font-size:9px;color:var(--td);font-family:'Courier New',monospace;}}
.hmleg{{margin-top:9px;display:flex;gap:16px;font-size:11px;}}
.sr{{display:flex;align-items:center;gap:10px;margin-bottom:9px;}}
.sn{{font-size:11px;color:var(--tm);width:95px;flex-shrink:0;font-weight:600;}}
.sbw{{flex:1;height:17px;background:var(--sf2);border-radius:3px;overflow:hidden;}}
.sb{{height:100%;background:linear-gradient(90deg,#10B981,#4ADE80);border-radius:3px;}}
.sc{{font-size:11px;color:var(--glt);font-family:'Courier New',monospace;width:58px;text-align:right;font-weight:700;}}
.stat-t{{width:100%;border-collapse:collapse;}}
.stat-t td{{padding:7px 4px;border-bottom:1px solid var(--bd);font-size:12px;}}
.stat-t td:first-child{{color:var(--tm);font-size:11px;}}.stat-t td:last-child{{text-align:right;font-family:'Courier New',monospace;font-weight:700;color:var(--glt);}}
.spd-pnl{{background:var(--sf);border:1px solid var(--bd);border-radius:var(--r);padding:26px;margin-bottom:36px;}}
.si{{display:flex;align-items:center;gap:10px;margin-bottom:8px;}}
.snum{{font-size:10px;color:var(--td);font-family:'Courier New',monospace;width:26px;text-align:right;}}
.spb{{height:22px;border-radius:4px;display:flex;align-items:center;padding-left:8px;}}
.spb span{{font-size:11px;font-weight:700;color:white;font-family:'Courier New',monospace;}}
.sts{{font-size:10px;color:var(--td);width:75px;text-align:right;font-family:'Courier New',monospace;}}
.tw{{background:var(--sf);border:1px solid var(--bd);border-radius:var(--r);overflow:hidden;margin-bottom:40px;}}
.th{{padding:18px 22px 14px;border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;}}
.th-t{{font-size:12px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:var(--gp);}}
.tbadge{{background:rgba(16,185,129,.15);border:1px solid rgba(16,185,129,.3);border-radius:12px;padding:3px 12px;font-size:10px;color:var(--glt);font-weight:700;}}
table.sht{{width:100%;border-collapse:collapse;}}
table.sht thead tr{{background:#0A1610;}}
table.sht th{{padding:9px 11px;text-align:left;font-size:9px;font-weight:700;letter-spacing:1.2px;text-transform:uppercase;color:var(--td);border-bottom:1px solid var(--bd);white-space:nowrap;}}
table.sht td{{padding:9px 11px;font-size:11px;border-bottom:1px solid rgba(26,53,40,.5);vertical-align:middle;}}
table.sht tr:last-child td{{border-bottom:none;}}
table.sht tr:hover td{{background:rgba(16,185,129,.04);}}
table.sht tr:nth-child(even) td{{background:rgba(8,12,20,.5);}}
table.sht .num{{color:var(--td);font-family:'Courier New',monospace;text-align:center;}}
table.sht .mono{{font-family:'Courier New',monospace;}}
table.sht .in-goal{{color:var(--glt);font-weight:700;}}
table.sht .missed{{color:var(--rd);font-weight:700;}}
.badge{{display:inline-block;font-size:8px;font-weight:700;letter-spacing:.5px;border-radius:3px;padding:1px 5px;vertical-align:middle;margin-left:3px;text-transform:uppercase;}}
.badge-pred{{background:rgba(16,185,129,.22);color:#4ADE80;border:1px solid rgba(74,222,128,.25);}}
.badge-mert{{background:rgba(148,163,184,.12);color:#94A3B8;border:1px solid rgba(148,163,184,.2);}}
.ftr{{border-top:1px solid var(--bd);background:#050A12;padding:26px 48px;}}
.ftr-i{{max-width:1400px;margin:0 auto;display:flex;align-items:center;justify-content:space-between;gap:20px;}}
.ftr-l{{font-size:12px;color:var(--td);line-height:1.7;}}.ftr-l strong{{color:var(--tm);}}
.ftr-r{{font-size:10px;color:var(--td);text-align:right;font-family:'Courier New',monospace;line-height:1.7;}}
.ftr-logo{{height:34px;opacity:.45;}}
</style>
</head>
<body>
<header class="hdr">
  <div class="hdr-i">
    <div class="hdr-logos">
      {deik_tag}
      <div class="ldiv"></div>
      {rgk_tag}
    </div>
    <div class="hdr-t">
      <div class="sup">Debreceni Egyetem Informatikai Kar</div>
      <h1>Robot Kapus Rendszer<br><span>Edzés Munkamenet Riport</span></h1>
      <div class="sub">Valós idejű sztereó látórendszer &nbsp;·&nbsp; YOLOv8 &nbsp;·&nbsp; Ballisztikus trajektória előrejelzés</div>
    </div>
    <div class="hdr-m">
      <div class="chip">⚡ AUTOMATIKUS GENERÁLÁS</div>
      <div class="mdate">{now_str}</div>
      <div class="msess">{n} lövés rögzítve</div>
      <div class="sbadge">🤖 DEIK-RGK v1.0</div>
    </div>
  </div>
</header>
<main class="con">
  <p class="stitle">Munkamenet Összesítő</p>
  <div class="kpi">
    <div class="kc g"><span class="ki">⚽</span><div class="kv">{n}</div><div class="kl">Összes Lövés</div></div>
    <div class="kc a"><span class="ki">🎯</span><div class="kv">{in_g} <span style="font-size:13px">({pct:.0f}%)</span></div><div class="kl">Kaput Talált</div></div>
    <div class="kc b"><span class="ki">💨</span><div class="kv">{avg_spd:.1f}<span style="font-size:12px"> km/h</span></div><div class="kl">Átlag Sebesség</div></div>
    <div class="kc r"><span class="ki">🚀</span><div class="kv">{max_spd:.1f}<span style="font-size:12px"> km/h</span></div><div class="kl">Max Sebesség</div></div>
    <div class="kc t"><span class="ki">⏱</span><div class="kv">{avg_react:.1f}<span style="font-size:12px"> ms</span></div><div class="kl">Átl. Kapus Reakció</div></div>
    <div class="kc p"><span class="ki">🔬</span><div class="kv">{avg_pipe:.1f}<span style="font-size:12px"> ms</span></div><div class="kl">Átl. Pipeline</div></div>
  </div>
  <div class="tc">
    <div class="pnl">
      <div class="pnl-t">🟥 Lövési Hőtérkép – Kapu Rácsanalízis</div>
      <div class="hmw">
        <div class="hml">KAPU NÉZET (elölről)</div>
        <svg width="100%" viewBox="0 0 {SVG_W} {SVG_H}" xmlns="http://www.w3.org/2000/svg">
          <rect width="{SVG_W}" height="{SVG_H}" fill="#060C14"/>
          {cells}
          <rect x="{MG}" y="{MG}" width="{gw_px}" height="{gh_px}" fill="none" stroke="#10B981" stroke-width="2.5"/>
          <line x1="{MG}" y1="{MG}" x2="{MG}" y2="{MG+gh_px}" stroke="#4ADE80" stroke-width="5"/>
          <line x1="{MG+gw_px}" y1="{MG}" x2="{MG+gw_px}" y2="{MG+gh_px}" stroke="#4ADE80" stroke-width="5"/>
          <line x1="{MG}" y1="{MG}" x2="{MG+gw_px}" y2="{MG}" stroke="#4ADE80" stroke-width="5"/>
          <line x1="{MG}" y1="{MG+gh_px/2:.1f}" x2="{MG+gw_px}" y2="{MG+gh_px/2:.1f}" stroke="#1A3526" stroke-width="1" stroke-dasharray="5,5"/>
          <line x1="{MG+gw_px/2:.1f}" y1="{MG}" x2="{MG+gw_px/2:.1f}" y2="{MG+gh_px}" stroke="#1A3526" stroke-width="1" stroke-dasharray="5,5"/>
          {dots}
          <text x="{MG+8}" y="{SVG_H-6}" font-size="9" fill="#475569" font-family="monospace">BAL</text>
          <text x="{MG+gw_px-30}" y="{SVG_H-6}" font-size="9" fill="#475569" font-family="monospace">JOBB</text>
          <text x="{MG+gw_px/2:.1f}" y="{SVG_H-6}" font-size="9" fill="#475569" font-family="monospace" text-anchor="middle">KÖZÉP</text>
        </svg>
        <div class="hmgl"><span>−2000 mm</span><span>0</span><span>+2000 mm</span></div>
      </div>
      <div class="hmleg"><span style="color:#10B981;">● Kapuban</span><span style="color:#EF4444;">● Mellé</span><span style="color:#94A3B8;font-size:10px;">⬡ = ballisztikus pred.</span></div>
    </div>
    <div class="pnl">
      <div class="pnl-t">📊 Szektor Eloszlás</div>
      {sector_bars}
      <div style="margin-top:22px;">
        <div class="pnl-t" style="margin-top:0;">📈 Teljesítmény Összesítő</div>
        <table class="stat-t">
          <tr><td>Min. sebesség</td><td>{min_spd:.1f} km/h</td></tr>
          <tr><td>Max. sebesség</td><td>{max_spd:.1f} km/h</td></tr>
          <tr><td>Átl. becsapódási idő</td><td>{avg_tti:.3f} s</td></tr>
          <tr><td>Átl. YOLO latencia</td><td>{avg_lat:.1f} ms</td></tr>
          <tr><td>Átl. pipeline késleltetés</td><td>{avg_pipe:.1f} ms</td></tr>
          <tr><td>Átl. kapus reakcióidő</td><td>{avg_react:.1f} ms</td></tr>
          <tr><td>Kapuban talált</td><td>{in_g} / {n} ({pct:.0f}%)</td></tr>
          <tr><td>Mellé ment</td><td>{n-in_g} / {n} ({100-pct:.0f}%)</td></tr>
        </table>
      </div>
    </div>
  </div>
  <div class="spd-pnl">
    <div class="pnl-t">💨 Lövési Sebességprofil – Munkamenet Kronológia</div>
    {speed_bars}
  </div>
  <p class="stitle">Részletes Lövési Napló – Teljes Telemetria</p>
  <div class="tw">
    <div class="th"><span class="th-t">📋 Lövés-szintű Telemetria Adatok</span><span class="tbadge">{n} bejegyzés</span></div>
    <div style="overflow-x:auto;">
    <table class="sht">
      <thead><tr>
        <th>#</th><th>Időpont</th><th>X (mm)</th><th>Y (mm)</th><th>Sebesség</th>
        <th>Becsap. Idő</th><th>Det. Latencia</th><th>Pipeline</th><th>Kapus Reakció</th>
        <th>Kapus Cél (X/Y)</th><th>Módszer</th><th>Szektor</th><th>Eredmény</th>
      </tr></thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
  </div>
</main>
<footer class="ftr">
  <div class="ftr-i">
    <div class="ftr-l">
      <strong>DEIK Robot Foci Kapus Projekt</strong> – Debreceni Egyetem Informatikai Kar (2026)<br>
      Fejlesztők: <strong>Morvai Roland</strong> &nbsp;·&nbsp; <strong>Rácz Donát</strong> – BSc Mérnökinformatikus<br>
      Rendszer: YOLOv8 &nbsp;·&nbsp; Python 3.12 &nbsp;·&nbsp; PyQt6 &nbsp;·&nbsp; OpenCV &nbsp;·&nbsp; CUDA (RTX 3050)
    </div>
    {ftr_logo}
    <div class="ftr-r">
      Generálva: {now_str}<br>
      {n} lövés rögzítve<br>
      <span style="color:#10B981;">● DEIK-RGK v1.0</span>
    </div>
  </div>
</footer>
</body>
</html>"""



    def _apply_theme(self) -> None:
        dark = self._dark
        bg_style = "background-color: #0B0F17; color: #F8FAFC;" if dark else "background-color: #F8FAFC; color: #0F172A;"
        self.setStyleSheet(bg_style)
