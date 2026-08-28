"""
DEIK Robot Foci Kapus – Analitika & Vizuális Grafikonok Dashboard (PyQt6)
========================================================================

Statisztikai és elemző felület:
  - Labda Sebesség & Trajektória Grafikon (km/h, Z magasság)
  - 2D Lövési Hőtérkép (Goal 2D Heatmap)
  - Szektoros eloszlási mutatók (Bal felső, Jobb felső, Bal alsó, Jobb alsó, Közép)
  - Munkamenet Exportálás (CSV & HTML Riport)
  - Téma-tudatos (Dark Mode & Light Mode)
"""

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
        self._shots: List[Tuple[float, float, float, bool]] = []
        self.setMinimumSize(360, 220)

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self.update()

    def set_shots(self, shots: List[Tuple[float, float, float, bool]]) -> None:
        """Frissíti a hőtérkép rácsát a kapott lövés listával."""
        self._shots = list(shots)
        self._grid = [[0 for _ in range(self.GRID_COLS)] for _ in range(self.GRID_ROWS)]

        half_w = self._goal_w / 2.0
        for sx, sy, conf, in_g in self._shots:
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
    """

    def __init__(self, config: dict, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self._config = config
        self._dark = False

        geo_cfg = config.get("geometry", {})
        self._goal_w = float(geo_cfg.get("goal_width_mm", 4000.0))
        self._goal_h = float(geo_cfg.get("goal_height_mm", 2000.0))

        self._shot_records: List[Tuple[float, float, float, bool]] = []
        self._build_ui()

    def set_dark(self, dark: bool) -> None:
        self._dark = dark
        self._heatmap.set_dark(dark)
        self._speed_plot.set_dark(dark)
        self._apply_theme()

    def add_shot_event(self, x_mm: float, y_mm: float, conf: float, in_goal: bool, speed_kmh: float = 45.0) -> None:
        """Hozzáad egy új lövés eseményt az analitikai gyűjteményhez."""
        self._shot_records.append((x_mm, y_mm, conf, in_goal))
        self._heatmap.set_shots(self._shot_records)
        self._speed_plot.add_data_point(speed_kmh, y_mm)
        self._update_table_and_stats()

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
        self._lbl_stat_saved = QLabel("0 db (0%)")
        self._lbl_stat_sectors = QLabel("BAL: 0 | KÖZÉP: 0 | JOBB: 0")

        stats_form.addRow("Összes Lövés:", self._lbl_stat_total)
        stats_form.addRow("Kaput Talált:", self._lbl_stat_ingoal)
        stats_form.addRow("Robot Védések:", self._lbl_stat_saved)
        stats_form.addRow("Zóna Eloszlás:", self._lbl_stat_sectors)

        # Lövési Lista Táblázat
        table_grp = QGroupBox("Legutóbbi Lövések Részletes Listája")
        table_box = QVBoxLayout(table_grp)

        self._table = QTableWidget(0, 5)
        self._table.setHorizontalHeaderLabels(["Időpont", "X (mm)", "Y (mm)", "Konfidencia", "Eredmény"])
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self._table.verticalHeader().setVisible(False)
        table_box.addWidget(self._table)

        details_box.addWidget(stats_grp, stretch=1)
        details_box.addWidget(table_grp, stretch=2)
        main_layout.addLayout(details_box)

        self._apply_theme()

    def _update_table_and_stats(self) -> None:
        n = len(self._shot_records)
        in_g = sum(1 for _, _, _, ig in self._shot_records if ig)
        pct = (in_g / n * 100.0) if n > 0 else 0.0

        self._lbl_stat_total.setText(f"{n} db")
        self._lbl_stat_ingoal.setText(f"{in_g} db ({pct:.0f}%)")

        # Frissítjük a táblázatot
        self._table.setRowCount(0)
        for i, (sx, sy, conf, ig) in enumerate(reversed(self._shot_records)):
            self._table.insertRow(i)
            t_str = time.strftime("%H:%M:%S")
            res_str = "KAPUBAN ✓" if ig else "MELLÉ ✕"

            self._table.setItem(i, 0, QTableWidgetItem(t_str))
            self._table.setItem(i, 1, QTableWidgetItem(f"{sx:+.0f}"))
            self._table.setItem(i, 2, QTableWidgetItem(f"{sy:.0f}"))
            self._table.setItem(i, 3, QTableWidgetItem(f"{conf * 100:.0f}%"))

            item_res = QTableWidgetItem(res_str)
            item_res.setForeground(QColor("#10B981") if ig else QColor("#EF4444"))
            self._table.setItem(i, 4, item_res)

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
                    writer.writerow(["Timestamp", "X_mm", "Y_mm", "Confidence", "InGoal"])
                    for sx, sy, conf, ig in self._shot_records:
                        writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), sx, sy, conf, ig])
                QMessageBox.information(self, "Export Sikeres", f"Lövési adatok mentve:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Hiba", str(e))

    @pyqtSlot()
    def _export_html_report(self) -> None:
        if not self._shot_records:
            QMessageBox.warning(self, "Nincs Adat", "Nincs menthető lövési adat!")
            return
        path, _ = QFileDialog.getSaveFileName(self, "HTML Jelentés Mentése", "deik_session_report.html", "HTML (*.html)")
        if path:
            try:
                n = len(self._shot_records)
                in_g = sum(1 for _, _, _, ig in self._shot_records if ig)
                pct = (in_g / n * 100.0) if n > 0 else 0.0

                html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>DEIK Robot Kapus – Munkamenet Riport</title>
    <style>
        body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #0B0F17; color: #F8FAFC; margin: 20px; }}
        h1 {{ color: #4ADE80; border-bottom: 2px solid #10B981; padding-bottom: 8px; }}
        .card {{ background: #151D2A; border: 1px solid #26334D; padding: 15px; border-radius: 8px; margin-bottom: 15px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
        th, td {{ padding: 8px 12px; border: 1px solid #26334D; text-align: left; }}
        th {{ background: #0F5132; color: white; }}
        tr:nth-child(even) {{ background: #1E293B; }}
    </style>
</head>
<body>
    <h1>⚽ DEIK Robot Kapus – Edzés & Munkamenet Riport</h1>
    <div class="card">
        <h3>Dátum: {time.strftime('%Y-%m-%d %H:%M:%S')}</h3>
        <p><b>Összes Lövés:</b> {n} db</p>
        <p><b>Kaput Talált:</b> {in_g} db ({pct:.1f}%)</p>
    </div>
    <div class="card">
        <h3>Lövési Lista</h3>
        <table>
            <tr><th>#</th><th>X (mm)</th><th>Y (mm)</th><th>Konfidencia</th><th>Eredmény</th></tr>
"""
                for idx, (sx, sy, conf, ig) in enumerate(self._shot_records, 1):
                    res_txt = "KAPUBAN" if ig else "MELLÉ"
                    html += f"<tr><td>{idx}</td><td>{sx:+.0f}</td><td>{sy:.0f}</td><td>{conf*100:.0f}%</td><td>{res_txt}</td></tr>\n"

                html += """        </table>
    </div>
</body>
</html>"""
                with open(path, "w", encoding="utf-8") as f:
                    f.write(html)
                QMessageBox.information(self, "Export Sikeres", f"HTML Riport mentve:\n{path}")
            except Exception as e:
                QMessageBox.critical(self, "Export Hiba", str(e))

    def _apply_theme(self) -> None:
        dark = self._dark
        bg_style = "background-color: #0B0F17; color: #F8FAFC;" if dark else "background-color: #F8FAFC; color: #0F172A;"
        self.setStyleSheet(bg_style)
