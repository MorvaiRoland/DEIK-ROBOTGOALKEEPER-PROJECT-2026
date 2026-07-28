"""
DEIK Robot Foci Kapus – 2D Kalman Szűrő (Per-kamera simítás)
=============================================================

Ez a modul egy egyszerű 2D Kalman szűrőt valósít meg a labda
2D képkoordináta-simítására kamera képenként.

Miért kell Kalman szűrő a YOLO mellé?
    A YOLO ~60 Hz-en detektál (az inferencia sebességétől függően).
    Ha a kamera 100 FPS-en fut, de a YOLO csak 60 Hz-en, akkor bizonyos
    frame-eken nincs friss detektálás. A Kalman szűrő ilyenkor
    a fizikai mozgásmodell alapján "megjósolja" a labda pozícióját,
    így a trajectory predictor folyamatos bemenetet kap.

Állapotvektort (4D):
    [x, y, vx, vy]  ahol (x, y) = képkoordináta, (vx, vy) = pixeles sebesség

Mérési modell:
    z = [x, y]  → csak pozíciót mérünk (a YOLO ezt adja)
"""

import logging
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KalmanTracker2D:
    """
    Konstans sebesség modellű 2D Kalman szűrő labda-követéshez.

    Állapotvektor: [x, y, vx, vy]
        - x, y:   Középpont koordináták pixelben
        - vx, vy: Sebesség pixelben/frame

    Mérési vektor: [x, y]
        - Csak a pozíciót mérjük (YOLO bounding box közép)

    Example:
        tracker = KalmanTracker2D(process_noise=0.03, measurement_noise=0.1)
        tracker.init(cx, cy)

        # Detektálás esetén:
        x_filt, y_filt = tracker.update(cx, cy)

        # Kihagyott frame esetén (csak predikció):
        x_pred, y_pred = tracker.predict()
    """

    def __init__(
        self,
        process_noise: float = 0.03,
        measurement_noise: float = 0.1,
        max_coast_frames: int = 10,
    ):
        """
        Args:
            process_noise:      Folyamatzaj (Q mátrix diagonálisa)
                                Kis érték → simább, de lassabban követi a valóságot
            measurement_noise:  Mérési zaj (R mátrix diagonálisa)
                                Nagy érték → inkább bíz a predikciós modellben
            max_coast_frames:   Ennyi frame után detektálás nélkül reseteljük a szűrőt
        """
        self._process_noise = process_noise
        self._measurement_noise = measurement_noise
        self._max_coast_frames = max_coast_frames

        self._is_initialized = False
        self._coast_frames = 0

        # Kalman filter mátrixok inicializálása
        self._init_matrices()

        logger.debug(
            "KalmanTracker2D: Q=%.3f, R=%.3f, max_coast=%d",
            process_noise, measurement_noise, max_coast_frames
        )

    def _init_matrices(self) -> None:
        """
        Inicializálja a Kalman szűrő mátrixait.

        Állapotvektor mérete: 4 (x, y, vx, vy)
        Mérési vektor mérete: 2 (x, y)
        """
        # Állapotvektor (oszlopvektor): [x, y, vx, vy]^T
        self._state = np.zeros((4, 1), dtype=np.float64)

        # Kovariancia mátrix (kezdetben nagy bizonytalanság)
        self._P = np.eye(4, dtype=np.float64) * 1000.0

        # Állapot-átmeneti mátrix (konstans sebesség modell, dt=1 frame)
        # x_new = x + vx*dt, y_new = y + vy*dt
        # vx_new = vx, vy_new = vy
        self._F = np.array([
            [1, 0, 1, 0],  # x  = x  + vx
            [0, 1, 0, 1],  # y  = y  + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1],  # vy = vy
        ], dtype=np.float64)

        # Mérési mátrix: csak x és y mérjük
        # z = H * state = [x, y]
        self._H = np.array([
            [1, 0, 0, 0],  # Mérés: x
            [0, 1, 0, 0],  # Mérés: y
        ], dtype=np.float64)

        # Folyamatzaj mátrix (Q)
        # Nagyobb vx, vy kovarianciával (sebesség bizonytalanabb mint pozíció)
        q = self._process_noise
        self._Q = np.diag([q, q, q * 4, q * 4])

        # Mérési zaj mátrix (R)
        r = self._measurement_noise
        self._R = np.eye(2, dtype=np.float64) * r

    # ------------------------------------------------------------------
    # Kalman szűrő lépések
    # ------------------------------------------------------------------

    def init(self, x: float, y: float) -> None:
        """
        Inicializálja a szűrőt egy mért pozícióval.

        Ezt kell hívni, ha új labdát kezdünk követni, vagy
        reset után az első mérésnél.

        Args:
            x: Kezdeti X koordináta (pixelben)
            y: Kezdeti Y koordináta (pixelben)
        """
        self._state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 100.0
        self._is_initialized = True
        self._coast_frames = 0
        logger.debug("KalmanTracker2D inicializálva: (%.1f, %.1f)", x, y)

    def update(self, x: float, y: float) -> Tuple[float, float]:
        """
        Frissíti a szűrőt egy új méréssel és visszaadja a simított pozíciót.

        Ez a teljes predict + correct lépés egy mérés esetén.

        Args:
            x: Mért X koordináta (YOLO detektálás, pixelben)
            y: Mért Y koordináta (YOLO detektálás, pixelben)

        Returns:
            Tuple: (x_simított, y_simított) pixelben
        """
        if not self._is_initialized:
            self.init(x, y)
            return x, y

        self._coast_frames = 0

        # --- Predict lépés ---
        x_pred, P_pred = self._predict_step()

        # --- Correct (update) lépés ---
        z = np.array([[x], [y]], dtype=np.float64)

        # Innováció: különbség a mérés és predikció között
        y_innov = z - self._H @ x_pred

        # Innováció kovariancia
        S = self._H @ P_pred @ self._H.T + self._R

        # Kalman gain
        K = P_pred @ self._H.T @ np.linalg.inv(S)

        # Állapot frissítése
        self._state = x_pred + K @ y_innov

        # Kovariancia frissítése (Joseph forma – numerikusan stabilabb)
        I = np.eye(4)
        self._P = (I - K @ self._H) @ P_pred

        return float(self._state[0, 0]), float(self._state[1, 0])

    def predict(self) -> Tuple[float, float]:
        """
        Elvégzi a predikciós lépést mérés nélkül (kihagyott detektálás esetén).

        Ha túl sok frame telik el mérés nélkül (> max_coast_frames),
        a szűrő resetelődik.

        Returns:
            Tuple: (x_prediktált, y_prediktált) pixelben.
                   Ha reset történt: (0.0, 0.0) és is_initialized = False.
        """
        if not self._is_initialized:
            return 0.0, 0.0

        self._coast_frames += 1

        # Ellenőrzés: nem mentük-e már túl sokat?
        if self._coast_frames > self._max_coast_frames:
            logger.debug(
                "KalmanTracker2D: %d frame kihagyva → reset",
                self._coast_frames
            )
            self.reset()
            return 0.0, 0.0

        # Predikciós lépés (correction nélkül)
        x_pred, self._P = self._predict_step()
        self._state = x_pred

        return float(self._state[0, 0]), float(self._state[1, 0])

    def _predict_step(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Elvégzi a Kalman predikciós lépést.

        Returns:
            Tuple: (x_prediktált állapot, P_prediktált kovariancia)
        """
        x_pred = self._F @ self._state
        P_pred = self._F @ self._P @ self._F.T + self._Q
        return x_pred, P_pred

    def reset(self) -> None:
        """Nullázza a szűrőt (elveszett labda esetén)."""
        self._is_initialized = False
        self._coast_frames = 0
        self._state = np.zeros((4, 1), dtype=np.float64)
        self._P = np.eye(4, dtype=np.float64) * 1000.0

    # ------------------------------------------------------------------
    # Property-k
    # ------------------------------------------------------------------

    @property
    def is_initialized(self) -> bool:
        """True ha a szűrő aktív (volt már mérés)."""
        return self._is_initialized

    @property
    def coast_frames(self) -> int:
        """Az utolsó mérés óta eltelt frame-ek száma."""
        return self._coast_frames

    @property
    def velocity(self) -> Tuple[float, float]:
        """Becsült sebesség pixelben/frame egységben."""
        if not self._is_initialized:
            return 0.0, 0.0
        return float(self._state[2, 0]), float(self._state[3, 0])

    @property
    def position(self) -> Tuple[float, float]:
        """Jelenlegi becsült pozíció pixelben."""
        if not self._is_initialized:
            return 0.0, 0.0
        return float(self._state[0, 0]), float(self._state[1, 0])
