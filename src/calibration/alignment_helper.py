"""
DEIK Robot Foci Kapus – Kamera Pozíció Beállítási Segéd
======================================================

Ez a modul kezeli a két sztereó kamera fizikai beállításának és pozícionálásának
elemzését a pálya közepére helyezett sakktáblás kalibrációs tábla alapján.

Analizált geometriai tényezők:
  - Függőleges magasság és dőlés különbség (Pitch / Height disparity)
  - Vízszintes elfordulási szimmetria (Pan / Yaw symmetry)
  - Optikai tengely körüli elforgatás (Roll angle)
  - Relatív távolság / skála eltérés (Distance / Scale ratio)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
# pyrefly: ignore [missing-import]
import numpy as np


@dataclass
class AlignmentInstruction:
    """
    Egyes kameráknak szóló konkrét beállítási utasítás.

    Attributes:
        category:  'pitch' | 'pan' | 'roll' | 'scale' | 'general'
        action:    'UP' | 'DOWN' | 'LEFT' | 'RIGHT' | 'CW' | 'CCW' | 'OK'
        icon:      Ikon a GUI megjelenítéshez (pl. '⬆', '⬇', '🔄', '✔')
        text:      Magyar nyelvű részletes utasítás
        severity:  'ok' | 'warning' | 'error'
        value_str: Rövid érték jelzés (pl. "14.2 px", "+2.5°", "OK")
    """
    category: str
    action: str
    icon: str
    text: str
    severity: str
    value_str: str


@dataclass
class AlignmentResult:
    """
    Kamera pozicionálás teljes elemzési eredménye.
    """
    score: float                                        # 0.0 - 100.0 %
    left_found: bool
    right_found: bool
    both_found: bool
    center_l: Optional[Tuple[float, float]] = None
    center_r: Optional[Tuple[float, float]] = None
    delta_y_px: float = 0.0
    asym_x_px: float = 0.0
    roll_l_deg: float = 0.0
    roll_r_deg: float = 0.0
    scale_ratio: float = 1.0
    left_instructions: List[AlignmentInstruction] = field(default_factory=list)
    right_instructions: List[AlignmentInstruction] = field(default_factory=list)
    general_summary: str = ""
    status_color: str = "#CBD5E1"                       # Hex szín a GUI kártyákhoz


def _calculate_roll_angle(
    corners: np.ndarray,
    pattern_size: Tuple[int, int],
    is_portrait: bool = False
) -> float:
    """
    Kiszámítja a sakktábla felső pontsorának dőlésszög-eltérését fokban (Roll).
    
    Fekvő képeknél (w >= h) az ideális sakktábla sor vízszintes (~0°).
    Álló képeknél (h > w) az ideális sakktábla sor a képen függőlegesen fut (~±90°).
    """
    cols, rows = pattern_size
    pts = corners.reshape(-1, 2)
    if len(pts) < cols:
        return 0.0
    p0 = pts[0]
    p1 = pts[cols - 1]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    if abs(dx) < 1e-6:
        raw_angle = 90.0 if dy > 0 else -90.0
    else:
        angle_rad = math.atan2(dy, dx)
        raw_angle = math.degrees(angle_rad)

    if is_portrait:
        # Álló mód: a vízszintes világ-egyenes a képen függőlegesen jelenik meg (~90° vagy -90°)
        target = 90.0 if raw_angle >= 0 else -90.0
        return raw_angle - target
    else:
        # Fekvő mód: a vízszintes világ-egyenes a képen vízszintesen jelenik meg (~0°)
        return raw_angle


def calculate_camera_alignment(
    corners_l: Optional[np.ndarray],
    corners_r: Optional[np.ndarray],
    image_size: Tuple[int, int],
    pattern_size: Tuple[int, int] = (11, 8),
) -> AlignmentResult:
    """
    Kiszámítja a kamerák illeszkedési paramétereit és legyártja az utasításokat.

    Args:
        corners_l:    Bal kamera detektált sakktábla sarokpontjai ((N, 1, 2) tömb) vagy None
        corners_r:    Jobb kamera detektált sakktábla sarokpontjai ((N, 1, 2) tömb) vagy None
        image_size:   (szélesség, magasság) pixelben
        pattern_size: Sakktábla belső sarokpontjai (belső_sarkok_x, belső_sarkok_y)

    Returns:
        AlignmentResult objektum a részletes adatokkal és magyar utasításokkal.
    """
    left_found = corners_l is not None and len(corners_l) > 0
    right_found = corners_r is not None and len(corners_r) > 0
    both_found = left_found and right_found

    w, h = image_size
    is_portrait = (h > w)
    mid_x = w / 2.0
    mid_y = h / 2.0

    left_instrs: List[AlignmentInstruction] = []
    right_instrs: List[AlignmentInstruction] = []

    if not left_found and not right_found:
        return AlignmentResult(
            score=0.0,
            left_found=False,
            right_found=False,
            both_found=False,
            general_summary="⚠ Sakktábla nem látható egyik kamerában sem! Helyezd a táblát a pálya közepére.",
            status_color="#EF4444",
            left_instructions=[
                AlignmentInstruction(
                    category="general", action="NONE", icon="❓",
                    text="Sakktábla keresése... Helyezd a táblát a látómezőbe.",
                    severity="error", value_str="Nincs detektálva"
                )
            ],
            right_instructions=[
                AlignmentInstruction(
                    category="general", action="NONE", icon="❓",
                    text="Sakktábla keresése... Helyezd a táblát a látómezőbe.",
                    severity="error", value_str="Nincs detektálva"
                )
            ]
        )

    # Egyoldali detektálás kezelése
    if left_found and not right_found:
        pts_l = corners_l.reshape(-1, 2)
        cx_l, cy_l = float(np.mean(pts_l[:, 0])), float(np.mean(pts_l[:, 1]))
        roll_l = _calculate_roll_angle(corners_l, pattern_size, is_portrait)

        left_instrs.append(AlignmentInstruction(
            category="general", action="OK", icon="✔",
            text=f"Sakktábla detektálva (Középpont: {cx_l:.0f}, {cy_l:.0f})",
            severity="ok", value_str="OK"
        ))
        if abs(roll_l) > 1.5:
            act = "CW" if roll_l > 1.5 else "CCW"
            ico = "🔄"
            txt = f"Forgasd az óramutatóval {'megegyezően' if roll_l > 1.5 else 'ellentétesen'}"
            left_instrs.append(AlignmentInstruction(
                category="roll", action=act, icon=ico, text=txt, severity="warning", value_str=f"{roll_l:+.1f}°"
            ))

        right_instrs.append(AlignmentInstruction(
            category="general", action="NONE", icon="❌",
            text="Sakktábla NEM látható! Állítsd be a jobb kamerát a pálya közepe felé.",
            severity="error", value_str="Nincs kép"
        ))

        return AlignmentResult(
            score=30.0,
            left_found=True, right_found=False, both_found=False,
            center_l=(cx_l, cy_l), roll_l_deg=roll_l,
            left_instructions=left_instrs, right_instructions=right_instrs,
            general_summary="⚠ Csak a BAL kamera látja a sakktáblát! Állítsd be a JOBB kamerát is.",
            status_color="#F59E0B"
        )

    if right_found and not left_found:
        pts_r = corners_r.reshape(-1, 2)
        cx_r, cy_r = float(np.mean(pts_r[:, 0])), float(np.mean(pts_r[:, 1]))
        roll_r = _calculate_roll_angle(corners_r, pattern_size, is_portrait)

        left_instrs.append(AlignmentInstruction(
            category="general", action="NONE", icon="❌",
            text="Sakktábla NEM látható! Állítsd be a bal kamerát a pálya közepe felé.",
            severity="error", value_str="Nincs kép"
        ))

        right_instrs.append(AlignmentInstruction(
            category="general", action="OK", icon="✔",
            text=f"Sakktábla detektálva (Középpont: {cx_r:.0f}, {cy_r:.0f})",
            severity="ok", value_str="OK"
        ))
        if abs(roll_r) > 1.5:
            act = "CW" if roll_r > 1.5 else "CCW"
            ico = "🔄"
            txt = f"Forgasd az óramutatóval {'megegyezően' if roll_r > 1.5 else 'ellentétesen'}"
            right_instrs.append(AlignmentInstruction(
                category="roll", action=act, icon=ico, text=txt, severity="warning", value_str=f"{roll_r:+.1f}°"
            ))

        return AlignmentResult(
            score=30.0,
            left_found=False, right_found=True, both_found=False,
            center_r=(cx_r, cy_r), roll_r_deg=roll_r,
            left_instructions=left_instrs, right_instructions=right_instrs,
            general_summary="⚠ Csak a JOBB kamera látja a sakktáblát! Állítsd be a BAL kamerát is.",
            status_color="#F59E0B"
        )

    # ------------------------------------------------------------------ #
    # Mindkét kamera látja a táblát: Részletes illeszkedés elemzés
    # ------------------------------------------------------------------ #
    pts_l = corners_l.reshape(-1, 2)
    pts_r = corners_r.reshape(-1, 2)

    cx_l, cy_l = float(np.mean(pts_l[:, 0])), float(np.mean(pts_l[:, 1]))
    cx_r, cy_r = float(np.mean(pts_r[:, 0])), float(np.mean(pts_r[:, 1]))

    roll_l = _calculate_roll_angle(corners_l, pattern_size, is_portrait)
    roll_r = _calculate_roll_angle(corners_r, pattern_size, is_portrait)

    # 1. Függőleges eltérés (Pitch / Height disparity)
    delta_y = cy_l - cy_r

    # 2. Vízszintes szimmetria (Pan / Yaw)
    dx_l = cx_l - mid_x
    dx_r = cx_r - mid_x
    asym_x = dx_l + dx_r

    # 3. Skála / Távolság arány
    diag_l = float(np.linalg.norm(pts_l[0] - pts_l[-1]))
    diag_r = float(np.linalg.norm(pts_r[0] - pts_r[-1]))
    scale_ratio = diag_l / diag_r if diag_r > 1e-3 else 1.0

    # ------------------------------------------------------------------ #
    # Utasítások generálása – BAL KAMERA
    # ------------------------------------------------------------------ #
    # Függőleges (Pitch)
    if abs(delta_y) <= 8.0:
        left_instrs.append(AlignmentInstruction(
            category="pitch", action="OK", icon="✔",
            text="Függőleges illeszkedés tökéletes", severity="ok", value_str="0 px"
        ))
    elif delta_y > 8.0:
        left_instrs.append(AlignmentInstruction(
            category="pitch", action="DOWN", icon="⬇",
            text=f"Döntsd LE a bal kamerát ({abs(delta_y):.0f} px eltérés)",
            severity="warning" if delta_y < 25.0 else "error",
            value_str=f"-{abs(delta_y):.0f} px"
        ))
    else:
        left_instrs.append(AlignmentInstruction(
            category="pitch", action="UP", icon="⬆",
            text=f"Döntsd FEL a bal kamerát ({abs(delta_y):.0f} px eltérés)",
            severity="warning" if abs(delta_y) < 25.0 else "error",
            value_str=f"+{abs(delta_y):.0f} px"
        ))

    # Vízszintes szimmetria (Pan)
    if abs(asym_x) <= 15.0:
        left_instrs.append(AlignmentInstruction(
            category="pan", action="OK", icon="✔",
            text="Vízszintes szimmetria megfelelő", severity="ok", value_str="Szimmetrikus"
        ))
    elif asym_x > 15.0:
        left_instrs.append(AlignmentInstruction(
            category="pan", action="RIGHT", icon="➡",
            text="Fordítsd JOBBRA a bal kamerát (pályaközép jobbra tolódott)",
            severity="warning", value_str="Jobbra"
        ))
    else:
        left_instrs.append(AlignmentInstruction(
            category="pan", action="LEFT", icon="⬅",
            text="Fordítsd BALRA a bal kamerát (pályaközép balra tolódott)",
            severity="warning", value_str="Balra"
        ))

    # Elforgatás (Roll)
    if abs(roll_l) <= 1.2:
        left_instrs.append(AlignmentInstruction(
            category="roll", action="OK", icon="✔",
            text="Kamera dőlésszöge vízszintes", severity="ok", value_str=f"{roll_l:+.1f}°"
        ))
    elif roll_l > 1.2:
        left_instrs.append(AlignmentInstruction(
            category="roll", action="CW", icon="🔄",
            text=f"Forgasd az ÓRAMUTATÓVAL MEGEGYEZŐEN (dőlés: {roll_l:+.1f}°)",
            severity="warning", value_str=f"{roll_l:+.1f}°"
        ))
    else:
        left_instrs.append(AlignmentInstruction(
            category="roll", action="CCW", icon="🔄",
            text=f"Forgasd az ÓRAMUTATÓVAL ELLENTÉTESEN (dőlés: {roll_l:+.1f}°)",
            severity="warning", value_str=f"{roll_l:+.1f}°"
        ))

    # ------------------------------------------------------------------ #
    # Utasítások generálása – JOBB KAMERA
    # ------------------------------------------------------------------ #
    # Függőleges (Pitch)
    if abs(delta_y) <= 8.0:
        right_instrs.append(AlignmentInstruction(
            category="pitch", action="OK", icon="✔",
            text="Függőleges illeszkedés tökéletes", severity="ok", value_str="0 px"
        ))
    elif delta_y > 8.0:
        right_instrs.append(AlignmentInstruction(
            category="pitch", action="UP", icon="⬆",
            text=f"Döntsd FEL a jobb kamerát ({abs(delta_y):.0f} px eltérés)",
            severity="warning" if delta_y < 25.0 else "error",
            value_str=f"+{abs(delta_y):.0f} px"
        ))
    else:
        right_instrs.append(AlignmentInstruction(
            category="pitch", action="DOWN", icon="⬇",
            text=f"Döntsd LE a jobb kamerát ({abs(delta_y):.0f} px eltérés)",
            severity="warning" if abs(delta_y) < 25.0 else "error",
            value_str=f"-{abs(delta_y):.0f} px"
        ))

    # Vízszintes szimmetria (Pan)
    if abs(asym_x) <= 15.0:
        right_instrs.append(AlignmentInstruction(
            category="pan", action="OK", icon="✔",
            text="Vízszintes szimmetria megfelelő", severity="ok", value_str="Szimmetrikus"
        ))
    elif asym_x > 15.0:
        right_instrs.append(AlignmentInstruction(
            category="pan", action="RIGHT", icon="➡",
            text="Fordítsd JOBBRA a jobb kamerát (pályaközép jobbra tolódott)",
            severity="warning", value_str="Jobbra"
        ))
    else:
        right_instrs.append(AlignmentInstruction(
            category="pan", action="LEFT", icon="⬅",
            text="Fordítsd BALRA a jobb kamerát (pályaközép balra tolódott)",
            severity="warning", value_str="Balra"
        ))

    # Elforgatás (Roll)
    if abs(roll_r) <= 1.2:
        right_instrs.append(AlignmentInstruction(
            category="roll", action="OK", icon="✔",
            text="Kamera dőlésszöge vízszintes", severity="ok", value_str=f"{roll_r:+.1f}°"
        ))
    elif roll_r > 1.2:
        right_instrs.append(AlignmentInstruction(
            category="roll", action="CW", icon="🔄",
            text=f"Forgasd az ÓRAMUTATÓVAL MEGEGYEZŐEN (dőlés: {roll_r:+.1f}°)",
            severity="warning", value_str=f"{roll_r:+.1f}°"
        ))
    else:
        right_instrs.append(AlignmentInstruction(
            category="roll", action="CCW", icon="🔄",
            text=f"Forgasd az ÓRAMUTATÓVAL ELLENTÉTESEN (dőlés: {roll_r:+.1f}°)",
            severity="warning", value_str=f"{roll_r:+.1f}°"
        ))

    # ------------------------------------------------------------------ #
    # Pontszám és összefoglaló
    # ------------------------------------------------------------------ #
    v_err = min(1.0, abs(delta_y) / 60.0)
    h_err = min(1.0, abs(asym_x) / 100.0)
    r_err = min(1.0, (abs(roll_l) + abs(roll_r)) / 8.0)

    avg_y = (cy_l + cy_r) / 2.0
    c_err = min(1.0, abs(avg_y - mid_y) / 150.0)

    penalty = 45.0 * v_err + 25.0 * h_err + 15.0 * r_err + 15.0 * c_err
    score = max(0.0, round(100.0 - penalty, 1))

    if score >= 90.0:
        summary = "✅ Tökéletes kamera pozícionálás! A kamerák pontosan párhuzamosak és szimmetrikusak."
        status_color = "#10B981"
    elif score >= 70.0:
        summary = "⚡ Jó pozicionálás, de apró korrekció szükséges a pontosabb sztereó kalibrációhoz."
        status_color = "#F59E0B"
    else:
        summary = "⚠ Jelentős eltérés a kamerák között! Kövesd az alábbi utasításokat a kamerák beállításához."
        status_color = "#EF4444"

    return AlignmentResult(
        score=score,
        left_found=True, right_found=True, both_found=True,
        center_l=(cx_l, cy_l), center_r=(cx_r, cy_r),
        delta_y_px=delta_y, asym_x_px=asym_x,
        roll_l_deg=roll_l, roll_r_deg=roll_r,
        scale_ratio=scale_ratio,
        left_instructions=left_instrs,
        right_instructions=right_instrs,
        general_summary=summary,
        status_color=status_color
    )


def draw_alignment_hud(
    img: np.ndarray,
    corners: Optional[np.ndarray],
    is_left: bool,
    result: AlignmentResult,
) -> np.ndarray:
    """
    Kirajzolja a vizuális HUD elemeket (célkereszt,Referenciasáv, detektált pontok, nyilak)
    a megadott kameraképre.

    Args:
        img:     BGR képkocka
        corners: Detektált sakktábla sarokpontok (vagy None)
        is_left: True ha a bal kamera képe, False ha a jobb
        result:  Kiszámított AlignmentResult objektum

    Returns:
        HUD elemekkel ellátott BGR képkocka
    """
    out = img.copy()
    h, w = out.shape[:2]
    mid_x, mid_y = w // 2, h // 2

    # 1. Képközéppont célkereszt (Cian szaggatott vonal jellegű)
    grid_color = (180, 180, 0)
    cv2.line(out, (mid_x, 0), (mid_x, h), grid_color, 1, cv2.LINE_AA)
    cv2.line(out, (0, mid_y), (w, mid_y), grid_color, 1, cv2.LINE_AA)
    cv2.circle(out, (mid_x, mid_y), 6, grid_color, 1, cv2.LINE_AA)

    if corners is not None and len(corners) > 0:
        pts = corners.reshape(-1, 2)
        cx, cy = int(np.mean(pts[:, 0])), int(np.mean(pts[:, 1]))

        # Célpont megjelenítés (Sárga korong + célkereszt)
        cv2.circle(out, (cx, cy), 12, (0, 235, 255), 2, cv2.LINE_AA)
        cv2.circle(out, (cx, cy), 3, (0, 235, 255), -1, cv2.LINE_AA)
        cv2.line(out, (cx - 16, cy), (cx + 16, cy), (0, 235, 255), 1, cv2.LINE_AA)
        cv2.line(out, (cx, cy - 16), (cx, cy + 16), (0, 235, 255), 1, cv2.LINE_AA)

        cam_tag = "BAL" if is_left else "JOBB"
        label = f"{cam_tag} C:: ({cx}, {cy})"
        cv2.putText(out, label, (cx + 18, cy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        # Ha mindkét kamera látja a táblát: Vízszintes szintvonal rajzolása
        if result.both_found:
            # Függőleges szintjelző vonal
            line_color = (0, 255, 0) if abs(result.delta_y_px) <= 8.0 else (0, 0, 255)
            cv2.line(out, (0, cy), (w, cy), line_color, 1, cv2.LINE_AA)

    # 2. Fejléc overlay jelzés (Kamera neve + állapota)
    cam_name = "BAL KAMERA" if is_left else "JOBB KAMERA"
    cv2.rectangle(out, (10, 10), (220, 36), (15, 23, 42), -1)
    cv2.rectangle(out, (10, 10), (220, 36), (51, 65, 85), 1)
    cv2.putText(out, cam_name, (18, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

    # Kiemelt figyelmeztető nyíl rajzolása a kép közepére, ha nagy az eltérés
    instrs = result.left_instructions if is_left else result.right_instructions
    for ins in instrs:
        if ins.severity in ("warning", "error") and ins.category in ("pitch", "pan"):
            arrow_txt = f"{ins.icon} {ins.action}"
            cv2.putText(out, arrow_txt, (mid_x - 40, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2, cv2.LINE_AA)
            break

    return out
