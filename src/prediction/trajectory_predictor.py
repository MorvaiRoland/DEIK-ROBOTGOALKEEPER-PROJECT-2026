"""
DEIK Robot Foci Kapus – Trajektória Előrejelző
===============================================

Ez a modul a labda repülési pályájának előrejelzéséért felelős.
A cél: meghatározni, hogy a labda hol (X, Y) fogja keresztezni
a kapu síkját (Z = 0), és mikor ér oda (t_impact).

Fizikai modell:
    A labdára ható erők:
    1. Gravitáció: F_g = m × g, lefelé (Y tengely negatív iránya)
    2. Légellenállás: F_d = ½ × ρ × C_d × A × v², ellentétes a mozgással
       ahol: ρ = 1.225 kg/m³ (levegő sűrűsége)
              C_d ≈ 0.47 (gömb légellenállási koefficiense)
              A = π × r² (labda keresztmetszetű területe)

    5-ös focilabda paraméterei:
        - Átmérő: ~220 mm → sugár r = 0.11 m
        - Tömeg: ~430 g (FIFA specifikáció)
        - A = π × 0.11² ≈ 0.038 m²
        - C_d ≈ 0.47 (sima gömb)

Előrejelzési módszer:
    1. 3D Kalman szűrő a mért pozíciókból → sebesség és pozíció becslése
    2. Runge-Kutta 4. rendű integráció a fizikai egyenletekre
    3. Z = 0 sík metszéspontjának meghatározása (kapu síkja)

Koordináta-rendszer (azonos a triangulator.py-ban definiálttal):
    X: vízszintes (bal → jobb)  [mm]
    Y: függőleges (talaj → fel) [mm]
    Z: mélység (kapu → pálya)   [mm]
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, List, Optional, Tuple

import numpy as np
# pyrefly: ignore [missing-import]
from scipy.integrate import solve_ivp

logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Konstansok
# --------------------------------------------------------------------------- #

# Levegő sűrűsége (kg/m³) – szobahőmérsékleten, tengerszinten
AIR_DENSITY_KG_M3 = 1.225

# 4-es Kipsta focilabda fizikai paraméterei (alapértelmezett)
# Ezeket a config.yaml "ball" szekciója felülírhatja!
DEFAULT_BALL_DIAMETER_MM = 210    # 4-es méret: ~210mm átmérő
DEFAULT_BALL_MASS_KG = 0.340     # 4-es méret: ~340g tömeg
DEFAULT_BALL_DRAG_COEFF = 0.47   # Gömb légellenállási koefficiense


def compute_drag_factor(diameter_mm: float, mass_kg: float, drag_coeff: float) -> float:
    """
    Kiszámítja a légellenállási faktor értékét a labda paraméterei alapján.

    F_d = k_d × |v|² × v_egység
    k_d = 0.5 × ρ × C_d × A / m

    Args:
        diameter_mm: Labda átmérő mm-ben
        mass_kg:     Labda tömeg kg-ban
        drag_coeff:  Légellenállási koefficiens (dimenzió nélküli)

    Returns:
        Drag factor (m^-1 egységben)
    """
    radius_m = (diameter_mm / 1000.0) / 2.0
    cross_area_m2 = np.pi * radius_m ** 2
    return 0.5 * AIR_DENSITY_KG_M3 * drag_coeff * cross_area_m2 / mass_kg


# --------------------------------------------------------------------------- #
# Adatstruktúrák
# --------------------------------------------------------------------------- #

@dataclass
class TrajectoryPoint:
    """
    Egy 3D trajektória pontot tárol.

    Attributes:
        x, y, z:   Pozíció mm-ben
        timestamp: Időbélyeg (perf_counter)
    """
    x: float
    y: float
    z: float
    timestamp: float = field(default_factory=time.perf_counter)


@dataclass
class ImpactPrediction:
    """
    A kapu síkján lévő becsapódási pontot tartalmazza.

    Attributes:
        x_mm:        Vízszintes becsapódási pozíció [mm], kapu közepétől
        y_mm:        Magassági becsapódási pozíció [mm], talajtól
        time_to_impact_s: Hány másodperc múlva ér a kapuhoz
        confidence:  Megbízhatóság [0.0 – 1.0]
        in_goal:     True ha a kapu keretén belül csapódik be
        valid:       True ha az előrejelzés érvényes
    """
    x_mm: float = 0.0
    y_mm: float = 0.0
    time_to_impact_s: float = 0.0
    confidence: float = 0.0
    in_goal: bool = False
    valid: bool = False


# --------------------------------------------------------------------------- #
# Fő előrejelző osztály
# --------------------------------------------------------------------------- #

class TrajectoryPredictor:
    """
    Fizika-alapú trajektória előrejelző 3D Kalman szűrővel.

    Pipeline:
        1. add_measurement() → 3D pont hozzáadása a historikához
        2. 3D Kalman szűrő frissíti a pozíció és sebesség becslést
        3. get_impact_prediction() → Fizikai szimulációval megjósolja
           hol érinti a labda a kapu síkját (Z = 0)

    Example:
        predictor = TrajectoryPredictor(config)

        # Minden detektált 3D pozíciónál:
        predictor.add_measurement(x_mm, y_mm, z_mm)

        # Előrejelzés lekérése:
        pred = predictor.get_impact_prediction()
        if pred.valid:
            print(f"Becsapódás: X={pred.x_mm:.0f} mm, t={pred.time_to_impact_s:.3f} s")
    """

    def __init__(self, config: dict):
        """
        Args:
            config: A system_config.yaml teljes tartalma
        """
        self._pred_cfg = config.get("prediction", {})

        # Fizikai konstansok
        self._gravity_mm_s2 = float(self._pred_cfg.get("gravity_mm_s2", 9810.0))
        self._drag_coeff_cfg = float(self._pred_cfg.get("drag_coefficient", 0.0005))

        # ----- Labda fizikai paraméterei (config "ball" szekciójából) -----
        ball_cfg = config.get("ball", {})
        self._ball_diameter_mm = float(ball_cfg.get("diameter_mm", DEFAULT_BALL_DIAMETER_MM))
        self._ball_mass_kg = float(ball_cfg.get("mass_kg", DEFAULT_BALL_MASS_KG))
        self._ball_drag_coeff = float(ball_cfg.get("drag_coefficient", DEFAULT_BALL_DRAG_COEFF))

        # Légellenállási faktor kiszámítása a labda paramétereiből
        self._drag_factor = compute_drag_factor(
            self._ball_diameter_mm,
            self._ball_mass_kg,
            self._ball_drag_coeff,
        )

        # Kapu paraméterei
        geo_cfg = config.get("geometry", {})
        self._goal_width_mm = float(geo_cfg.get("goal_width_mm", 4000.0))
        self._goal_height_mm = float(geo_cfg.get("goal_height_mm", 2000.0))

        # Előrejelzési beállítások
        self._min_points = int(self._pred_cfg.get("min_points_for_prediction", 3))
        self._history_size = int(self._pred_cfg.get("trajectory_history_size", 30))
        self._min_confidence = float(self._pred_cfg.get("min_confidence_threshold", 0.4))

        # Kalman 3D szűrő paraméterei
        kalman_cfg = self._pred_cfg.get("kalman_3d", {})
        self._q_noise = float(kalman_cfg.get("process_noise", 0.1))
        self._r_noise = float(kalman_cfg.get("measurement_noise", 5.0))

        # Trajektória historika (deque: automatikusan törlődnek a régiek)
        self._history: Deque[TrajectoryPoint] = deque(maxlen=self._history_size)

        # 3D Kalman szűrő állapota: [x, y, z, vx, vy, vz]
        self._kalman_state = np.zeros(6, dtype=np.float64)
        self._kalman_P = np.eye(6, dtype=np.float64) * 1e6  # Nagy kezdeti bizonytalanság
        self._kalman_initialized = False

        # Legutóbb érvényes előrejelzés (cache)
        self._last_prediction: Optional[ImpactPrediction] = None

        logger.info(
            "TrajectoryPredictor kész: g=%.0f mm/s², kapu=%dx%d mm, "
            "labda=%.0fmm/%.0fg, drag_factor=%.5f",
            self._gravity_mm_s2,
            self._goal_width_mm,
            self._goal_height_mm,
            self._ball_diameter_mm,
            self._ball_mass_kg * 1000,
            self._drag_factor,
        )

    # ------------------------------------------------------------------
    # Mérés hozzáadása
    # ------------------------------------------------------------------

    def add_measurement(self, x_mm: float, y_mm: float, z_mm: float) -> None:
        """
        Hozzáad egy új 3D pozíció-mérést a historikához.

        A Kalman szűrő azonnal frissíti a sebesség- és pozíciobecslést.

        Args:
            x_mm: Vízszintes pozíció mm-ben
            y_mm: Magassági pozíció mm-ben
            z_mm: Mélység mm-ben (kaputól)
        """
        now = time.perf_counter()
        point = TrajectoryPoint(x=x_mm, y=y_mm, z=z_mm, timestamp=now)
        self._history.append(point)

        # 3D Kalman szűrő frissítése
        self._kalman_update(x_mm, y_mm, z_mm, now)

    def reset(self) -> None:
        """
        Nullázza a trajektória historikát és a Kalman szűrőt.

        Hívni kell, ha a labdát elvesztjük (nem detektálható).
        """
        self._history.clear()
        self._kalman_initialized = False
        self._kalman_state = np.zeros(6, dtype=np.float64)
        self._kalman_P = np.eye(6, dtype=np.float64) * 1e6
        self._last_prediction = None
        logger.debug("TrajectoryPredictor reset")

    # ------------------------------------------------------------------
    # Előrejelzés
    # ------------------------------------------------------------------

    def get_impact_prediction(self) -> ImpactPrediction:
        """
        Megjósolja a labda kapu síkján lévő becsapódási pontját.

        Legalább min_points_for_prediction mérés szükséges.
        A fizikai szimulációhoz a Kalman szűrő által becsült
        aktuális pozíciót és sebességet vesszük alapul.

        Returns:
            ImpactPrediction: Az előrejelzési eredmény.
                              Ha nincs elég adat: valid=False.
        """
        if len(self._history) < self._min_points:
            return ImpactPrediction(valid=False)

        if not self._kalman_initialized:
            return ImpactPrediction(valid=False)

        # Kalman szűrőből: aktuális pozíció és sebesség
        x0 = float(self._kalman_state[0])
        y0 = float(self._kalman_state[1])
        z0 = float(self._kalman_state[2])
        vx0 = float(self._kalman_state[3])
        vy0 = float(self._kalman_state[4])
        vz0 = float(self._kalman_state[5])

        # Ha a labda már a kapunál van vagy mozdulatlan: nincs értelmes előrejelzés
        if z0 < 10.0:  # 10mm küszöb
            return ImpactPrediction(valid=False)

        # Ha a labda nem közeledik (vz nem negatív = nem a kapu felé tart):
        # Megjegyzés: Z pozitív a pálya felé → ha a labda a kapu felé tart, vz < 0
        if vz0 >= 0:
            return ImpactPrediction(valid=False)

        # Fizikai szimulációval meghatározzuk a becsapódási pontot
        pred = self._simulate_to_goal(x0, y0, z0, vx0, vy0, vz0)

        # Megbízhatóság számítása
        pred.confidence = self._compute_confidence(pred)
        pred.valid = pred.confidence >= self._min_confidence

        # Kapu keretén belül?
        if pred.valid:
            pred.in_goal = (
                abs(pred.x_mm) <= self._goal_width_mm / 2.0 and
                0 <= pred.y_mm <= self._goal_height_mm
            )

        self._last_prediction = pred
        return pred

    # ------------------------------------------------------------------
    # Fizikai szimuláció
    # ------------------------------------------------------------------

    def _simulate_to_goal(
        self,
        x0: float, y0: float, z0: float,
        vx0: float, vy0: float, vz0: float
    ) -> ImpactPrediction:
        """
        Szimulálja a labda pályáját Z=0-ig (kapu síkja).

        A scipy.integrate.solve_ivp() Runge-Kutta 4-5 módszerével
        numerikusan integrálja a mozgásegyenleteket.

        Args:
            x0, y0, z0:   Kiindulási pozíció mm-ben
            vx0, vy0, vz0: Kiindulási sebesség mm/s-ban

        Returns:
            ImpactPrediction: A kapu síkján lévő becsapódási pont
        """
        # mm-ből m-be konvertálás (a fizika m/s-ban számolja)
        MM_TO_M = 1e-3
        M_TO_MM = 1e3

        pos0_m = np.array([x0, y0, z0]) * MM_TO_M
        vel0_ms = np.array([vx0, vy0, vz0]) * MM_TO_M  # mm/s → m/s

        g_ms2 = self._gravity_mm_s2 * MM_TO_M  # mm/s² → m/s²
        drag_factor = self._drag_factor  # Instance-szintű drag faktor (labda paraméterekből)

        def equations_of_motion(t, state):
            """
            A labda mozgásegyenletei (a scipy által hívott függvény).

            Állapotvektor: [x, y, z, vx, vy, vz] SI egységben (m, m/s)

            Erők:
                - Gravitáció: (0, -g, 0) m/s²
                - Légellenállás: -k_d × |v| × v
            """
            _, _, _, vx, vy, vz = state
            v = np.array([vx, vy, vz])
            v_norm = np.linalg.norm(v)

            # Légellenállás gyorsulás (m/s²)
            drag_accel = -drag_factor * v_norm * v if v_norm > 1e-10 else np.zeros(3)

            # Összesített gyorsulás
            ax = drag_accel[0]
            ay = -g_ms2 + drag_accel[1]  # Gravitáció + légellenállás
            az = drag_accel[2]

            return [vx, vy, vz, ax, ay, az]

        def goal_plane_event(t, state):
            """
            Esemény: a labda eléri a Z=0 síkot (kapu síkja).
            scipy solve_ivp() leáll, ha ez nulla lesz.
            """
            return state[2]  # z koordináta

        # Az esemény "terminal" = leállunk, ha eléri a kapuvonalat
        goal_plane_event.terminal = True
        goal_plane_event.direction = -1  # Csak negatív irányban (közeledik)

        # Integráció (max 5 másodperc – ennyi alatt biztosan megérkezik)
        initial_state = [
            pos0_m[0], pos0_m[1], pos0_m[2],
            vel0_ms[0], vel0_ms[1], vel0_ms[2]
        ]

        try:
            sol = solve_ivp(
                equations_of_motion,
                t_span=(0.0, 5.0),
                y0=initial_state,
                method="RK45",
                events=goal_plane_event,
                dense_output=False,
                rtol=1e-4,   # Relatív tolerancia (pontosság vs. sebesség kompromisszum)
                atol=1e-6,   # Abszolút tolerancia
            )

            if sol.t_events[0].size > 0:
                # Megtaláltuk a metszéspontot
                t_impact = float(sol.t_events[0][0])
                x_impact_m = float(sol.y_events[0][0][0])
                y_impact_m = float(sol.y_events[0][0][1])

                return ImpactPrediction(
                    x_mm=x_impact_m * M_TO_MM,
                    y_mm=y_impact_m * M_TO_MM,
                    time_to_impact_s=t_impact,
                    confidence=0.0,  # Confidence-t majd kívül számítjuk
                    valid=True,
                )
            else:
                # Nem érte el a kapuvonalat (pl. mellé megy)
                logger.debug("Szimuláció: labda nem éri el a kapuvonalat")
                return ImpactPrediction(valid=False)

        except Exception as exc:
            logger.error("Trajektória szimulációs hiba: %s", exc)
            return ImpactPrediction(valid=False)

    # ------------------------------------------------------------------
    # 3D Kalman szűrő
    # ------------------------------------------------------------------

    def _kalman_update(self, x: float, y: float, z: float, t: float) -> None:
        """
        Frissíti a 3D Kalman szűrőt az új méréssel.

        Állapotvektor: [x, y, z, vx, vy, vz] (pozíció + sebesség mm-ben, mm/s-ban)
        Mérési vektor: [x, y, z] (csak pozíciót mérünk)

        Args:
            x, y, z: Mért 3D pozíció mm-ben
            t:       Mérés időbélyege
        """
        if not self._kalman_initialized:
            # Első mérés: állapot inicializálása
            self._kalman_state = np.array([x, y, z, 0.0, 0.0, 0.0], dtype=np.float64)
            self._kalman_P = np.eye(6, dtype=np.float64) * 1000.0
            self._kalman_initialized = True
            self._last_kalman_time = t
            return

        # Időlépés (dt)
        dt = t - self._last_kalman_time
        if dt <= 0 or dt > 1.0:  # Érvénytelen dt esetén kihagyjuk
            self._last_kalman_time = t
            return
        self._last_kalman_time = t

        # --- Állapot-átmeneti mátrix (konstans sebesség modell) ---
        # [x, y, z, vx, vy, vz]^T
        F = np.eye(6, dtype=np.float64)
        F[0, 3] = dt   # x += vx * dt
        F[1, 4] = dt   # y += vy * dt
        F[2, 5] = dt   # z += vz * dt

        # --- Folyamatzaj mátrix ---
        # A sebesség komponensek nagyobb zaját feltételezzük (gyorsuló labda).
        # A q_noise * dt^2 skálázás biztosítja, hogy a sebesség becslés
        # gyorsan alkalmazkodjon a valós mozgáshoz, függetlenül a mintavételezési rátától.
        q = self._q_noise
        dt2 = dt * dt
        Q = np.diag([
            q * dt2,       q * dt2,       q * dt2,        # Pozíció zaj
            q * 1e8 * dt2, q * 1e8 * dt2, q * 1e8 * dt2  # Sebesség zaj (nagy → gyors adaptáció)
        ])

        # --- Mérési mátrix: csak pozíciót mérünk ---
        H = np.zeros((3, 6), dtype=np.float64)
        H[0, 0] = 1.0  # x
        H[1, 1] = 1.0  # y
        H[2, 2] = 1.0  # z

        # --- Mérési zaj mátrix ---
        r = self._r_noise
        R = np.eye(3, dtype=np.float64) * r

        # --- PREDICT lépés ---
        x_pred = F @ self._kalman_state
        P_pred = F @ self._kalman_P @ F.T + Q

        # --- CORRECT lépés ---
        z_meas = np.array([x, y, z], dtype=np.float64)
        innov = z_meas - H @ x_pred
        S = H @ P_pred @ H.T + R
        K = P_pred @ H.T @ np.linalg.inv(S)

        self._kalman_state = x_pred + K @ innov
        self._kalman_P = (np.eye(6) - K @ H) @ P_pred

    # ------------------------------------------------------------------
    # Megbízhatóság számítás
    # ------------------------------------------------------------------

    def _compute_confidence(self, pred: ImpactPrediction) -> float:
        """
        Kiszámítja az előrejelzés megbízhatóságát [0.0 – 1.0].

        Figyelembe veszi:
            - A rendelkezésre álló mérési pontok számát
            - A becsapódásig hátralévő időt
            - A sebesség konzisztenciáját

        Args:
            pred: Az előrejelzési eredmény

        Returns:
            Megbízhatóság 0.0 és 1.0 között.
        """
        if not pred.valid:
            return 0.0

        # 1. Pontszám: elegendő mérési pont?
        n_pts = len(self._history)
        points_score = min(n_pts / 5.0, 1.0)  # Teljes pontszám 5 pontnál

        # 2. Pontszám: nem túl messze van még?
        if pred.time_to_impact_s <= 0:
            return 0.0
        # 2 másodpercnél közelebbi becsapódás → nagyobb megbízhatóság
        time_score = max(0.0, 1.0 - pred.time_to_impact_s / 3.0)

        # 3. Sebesség pontszám: reális focilabda sebesség?
        vz = float(self._kalman_state[5]) if self._kalman_initialized else 0.0
        # A z irányú sebesség abszolút értéke: tipikusan 5000-30000 mm/s (5-30 m/s)
        # A Kalman szűrő az első néhány mérésnél alulbecsülheti a sebességet,
        # ezért a küszöböt 100 mm/s-ra csökkentjük (csak a kapu felé mozgást ellenőrizzük)
        speed_ok = 100 <= abs(vz) <= 40000
        speed_score = 1.0 if speed_ok else 0.3

        # Súlyozott összesítés
        confidence = (0.4 * points_score + 0.4 * time_score + 0.2 * speed_score)
        return min(max(confidence, 0.0), 1.0)

    # ------------------------------------------------------------------
    # Property-k és lekérdezők
    # ------------------------------------------------------------------

    @property
    def measurement_count(self) -> int:
        """Az eddig hozzáadott mérési pontok száma."""
        return len(self._history)

    @property
    def last_prediction(self) -> Optional[ImpactPrediction]:
        """Az utoljára kiszámított előrejelzés (cache)."""
        return self._last_prediction

    @property
    def estimated_velocity_mm_s(self) -> Tuple[float, float, float]:
        """
        A Kalman szűrő által becsült aktuális sebesség mm/s-ban.

        Returns:
            Tuple (vx, vy, vz) mm/s-ban
        """
        if not self._kalman_initialized:
            return 0.0, 0.0, 0.0
        return (
            float(self._kalman_state[3]),
            float(self._kalman_state[4]),
            float(self._kalman_state[5]),
        )

    def get_trajectory_history_mm(self) -> List[Tuple[float, float, float]]:
        """
        Visszaadja a 3D trajektória historikát listában.

        Returns:
            Lista (x_mm, y_mm, z_mm) tupleokból
        """
        return [(p.x, p.y, p.z) for p in self._history]
