"""
DEIK Robot Foci Kapus – Sztereó Háromszögelés (3D Pozíció Számítás)
====================================================================

Ez a modul a sztereó kamerarendszerből érkező 2D detektálásokat
3D térkoordinátákká alakítja vissza.

Koordináta-rendszer:
    Origó: a kapu közepének talajon lévő pontja
    X tengely: vízszintes (bal → jobb, pozitív jobbra)  [mm]
    Y tengely: függőleges (talaj → fel, pozitív felfelé)  [mm]
    Z tengely: mélység (kapu → pálya, pozitív a pályára)  [mm]

Kamerarendszer geometriája (Fujifilm CF8ZA-1S, 8mm, Sony IMX174):
    Bal kamera:  pozíció = (-2450, 2800, 0) mm
    Jobb kamera: pozíció = (+2450, 2800, 0) mm
    Baseline: 4900 mm
    Fókusztávolság: ~1365 px (a kalibrálás adja meg pontosan)

Sztereó háromszögelési algoritmus:
    OpenCV triangulatePoints() függvénye lineáris LS megoldást alkalmaz.
    Bemenet:  Bal és jobb kamera kalibrált projekciós mátrixai (P1, P2)
              + a labda 2D képkoordinátái mindkét képen (u_L, v_L) és (u_R, v_R)
    Kimenet:  Homogén 4D vektor → 3D pont (X, Y, Z) mm-ben

Hivatkozás:
    Hartley, R. & Zisserman, A. "Multiple View Geometry in Computer Vision"
    OpenCV dokumentáció: cv2.triangulatePoints
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StereoTriangulator:
    """
    Sztereó háromszögelést végző osztály.

    Betölti a kalibrálási adatokat (K1, K2, D1, D2, R, T)
    és elvégzi a 3D rekonstrukciót minden labda detektálásnál.

    A kalibrálást a `scripts/calibrate_stereo.py` szkript végzi el,
    és a `data/calibration/stereo_calibration.npz` fájlba menti.

    Example:
        triangulator = StereoTriangulator(config)
        if triangulator.load_calibration("data/calibration/stereo_calibration.npz"):
            # 2D pontok mindkét képen (pl. a labda középpontja)
            pos_3d = triangulator.triangulate(
                left_point=(cxL, cyL),
                right_point=(cxR, cyR)
            )
            if pos_3d is not None:
                X, Y, Z = pos_3d  # mm-ben, kapu-koordináta-rendszerben
    """

    def __init__(self, config: dict):
        """
        Args:
            config: A system_config.yaml teljes tartalma
        """
        self._config = config
        self._stereo_cfg = config.get("stereo", {})
        self._geo_cfg = config.get("geometry", {})

        # Kalibrálás elérési útja
        self._calibration_file = Path(
            self._stereo_cfg.get("calibration_file", "data/calibration/stereo_calibration.npz")
        )

        # Kamera belső paraméterek (becslések kalibrálás előtt)
        # Ezeket a kalibrálás pontosítja!
        f_px = self._geo_cfg.get("focal_length_px", 1365.2)
        cx = self._geo_cfg.get("principal_point_x", 968.0)
        cy = self._geo_cfg.get("principal_point_y", 608.0)

        # Becsült belső mátrix (kalibrálás előtti placeholder)
        self._K_est = np.array([
            [f_px,  0.0,  cx],
            [ 0.0, f_px,  cy],
            [ 0.0,  0.0, 1.0],
        ], dtype=np.float64)

        # Kalibrálás utáni pontos mátrixok
        self._K1: Optional[np.ndarray] = None   # Bal kamera belső mátrix
        self._K2: Optional[np.ndarray] = None   # Jobb kamera belső mátrix
        self._D1: Optional[np.ndarray] = None   # Bal kamera torzítási koefficiensek
        self._D2: Optional[np.ndarray] = None   # Jobb kamera torzítási koefficiensek
        self._R: Optional[np.ndarray] = None    # Forgásmátrix (bal → jobb)
        self._T: Optional[np.ndarray] = None    # Eltolásvektor (bal → jobb) [mm]
        self._P1: Optional[np.ndarray] = None   # Bal projekciós mátrix (rektifikált)
        self._P2: Optional[np.ndarray] = None   # Jobb projekciós mátrix (rektifikált)
        self._Q: Optional[np.ndarray] = None    # Diszparitás → mélység mátrix

        # Torzítás-korrekciós térképek (cv2.remap()-hez)
        self._map1_L: Optional[np.ndarray] = None
        self._map2_L: Optional[np.ndarray] = None
        self._map1_R: Optional[np.ndarray] = None
        self._map2_R: Optional[np.ndarray] = None

        # Baseline (fallback, ha még nincs kalibrálás)
        self._baseline_mm = float(self._geo_cfg.get("baseline_mm", 4900.0))

        # Kalibrált-e a rendszer?
        self._is_calibrated = False

        # Fizikai kamera pozíciók (config-ból, fizikailag mért értékek)
        self._left_cam_x_mm  = float(self._geo_cfg.get("left_camera_x_mm",  -1070.0))
        self._cam_height_mm  = float(self._geo_cfg.get("camera_height_mm",   2900.0))
        self._cam_z_offset_mm = float(self._geo_cfg.get("camera_z_offset_mm", -900.0))

    # ------------------------------------------------------------------
    # Kalibrálás betöltése
    # ------------------------------------------------------------------

    def load_calibration(self, calibration_file: Optional[str] = None) -> bool:
        """
        Betölti a sztereó kalibrálási adatokat egy .npz fájlból.

        Args:
            calibration_file: A .npz fájl elérési útja.
                              Ha None, a konfig-ban megadott utat használja.

        Returns:
            True ha a betöltés sikeres, False ha a fájl nem létezik.
        """
        cal_path = Path(calibration_file) if calibration_file else self._calibration_file

        if not cal_path.exists():
            logger.warning(
                "Kalibrálási fájl nem található: '%s'\n"
                "Futtasd: python scripts/calibrate_stereo.py\n"
                "Addig becsült paraméterekkel dolgozom (pontatlan!)",
                cal_path
            )
            self._setup_uncalibrated_fallback()
            return False

        try:
            logger.info("Kalibrálási adatok betöltése: %s", cal_path)
            data = np.load(str(cal_path))

            # Belső mátrixok
            self._K1 = data["K1"]
            self._K2 = data["K2"]
            self._D1 = data["D1"]
            self._D2 = data["D2"]

            # Külső paraméterek
            self._R = data["R"]    # Forgásmátrix
            self._T = data["T"]    # Eltolásvektor

            # Projekciós mátrixok (rektifikált)
            self._P1 = data["P1"]
            self._P2 = data["P2"]
            self._Q = data["Q"]    # Diszparitás → 3D

            # Képméret a rektifikációhoz
            img_width = int(data["image_width"])
            img_height = int(data["image_height"])

            # Torzítás-korrekciós térképek kiszámítása
            self._compute_rectification_maps(img_width, img_height)

            self._is_calibrated = True
            rmse = float(data.get("rmse", -1.0))
            logger.info(
                "✓ Kalibrálás betöltve: RMSE=%.3f px, baseline=%.1f mm",
                rmse, np.linalg.norm(self._T)
            )
            return True

        except Exception as exc:
            logger.error("Kalibrálás betöltési hiba: %s", exc)
            self._setup_uncalibrated_fallback()
            return False

    def _setup_uncalibrated_fallback(self) -> None:
        """
        Becsült paraméterek beállítása kalibrálás nélküli módhoz.

        FIGYELEM: Ez csak közelítő értékeket ad! A pontos 3D pozícióhoz
        kalibrálás szükséges. Fejlesztési/tesztelési célra elegendő.
        """
        logger.warning("Kalibrálás nélküli mód: becsült paraméterekkel dolgozom!")

        B = self._baseline_mm   # 4900 mm
        f = self._K_est[0, 0]  # ~1365 px
        cx = self._K_est[0, 2]  # ~968 px
        cy = self._K_est[1, 2]  # ~608 px

        # Bal kamera projekciós mátrix (ideális, torzítás nélkül)
        # P1 = K * [I | 0]
        self._P1 = np.array([
            [f,  0,  cx,  0],
            [0,  f,  cy,  0],
            [0,  0,   1,  0],
        ], dtype=np.float64)

        # Jobb kamera projekciós mátrix (baseline eltolással)
        # P2 = K * [I | -B * e_x]
        # A negatív jobb oldali X eltolás a baseline irányából ered
        self._P2 = np.array([
            [f,  0,  cx,  -f * B],   # -f*B a baseline eltolás
            [0,  f,  cy,   0    ],
            [0,  0,   1,   0    ],
        ], dtype=np.float64)

        self._K1 = self._K_est.copy()
        self._K2 = self._K_est.copy()
        self._D1 = np.zeros((5, 1), dtype=np.float64)
        self._D2 = np.zeros((5, 1), dtype=np.float64)
        self._is_calibrated = False

    def _compute_rectification_maps(self, width: int, height: int) -> None:
        """
        Kiszámítja a rektifikációs leképezési térképeket.

        Ezek a térképek szükségesek a kameraképek torzítás-korrekciójához
        és sztereó rektifikációjához (cv2.remap()).

        Args:
            width:  Kép szélessége pixelben
            height: Kép magassága pixelben
        """
        # Rektifikációs mátrixok kiszámítása
        R1, R2, P1_rect, P2_rect, Q, roi1, roi2 = cv2.stereoRectify(
            self._K1, self._D1,
            self._K2, self._D2,
            (width, height),
            self._R, self._T,
            flags=cv2.CALIB_ZERO_DISPARITY,
            alpha=0.0   # 0 = minimális fekete szél, 1 = teljes szenzor
        )

        # Bal kamera rektifikációs térképek
        self._map1_L, self._map2_L = cv2.initUndistortRectifyMap(
            self._K1, self._D1, R1, P1_rect, (width, height), cv2.CV_32FC1
        )

        # Jobb kamera rektifikációs térképek
        self._map1_R, self._map2_R = cv2.initUndistortRectifyMap(
            self._K2, self._D2, R2, P2_rect, (width, height), cv2.CV_32FC1
        )

        # Frissített projekciós mátrixok
        self._P1 = P1_rect
        self._P2 = P2_rect
        self._Q = Q

        logger.debug("Rektifikációs térképek kiszámítva: %dx%d", width, height)

    # ------------------------------------------------------------------
    # Főbb algoritmus: 3D háromszögelés
    # ------------------------------------------------------------------

    def triangulate(
        self,
        left_point: Tuple[float, float],
        right_point: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """
        Kiszámítja a labda 3D pozícióját a két kameraképbeli 2D pontokból.

        Az OpenCV triangulatePoints() lineáris legkisebb négyzetek módszerét
        alkalmazza homogén koordinátákban.

        Args:
            left_point:  A labda középpontja a bal képen (u, v) pixelben
            right_point: A labda középpontja a jobb képen (u, v) pixelben

        Returns:
            NumPy tömb [X, Y, Z] mm-ben (kapu-koordináta-rendszerben),
            vagy None ha a háromszögelés sikertelen (pl. kalibrálás hiánya,
            negatív Z, stb.)
        """
        if self._P1 is None or self._P2 is None:
            logger.warning("triangulate() hívás kalibrálás nélkül!")
            return None

        # 2D pontok (2×N mátrix, N=1 pont)
        pts_L = np.array([[left_point[0]], [left_point[1]]], dtype=np.float64)
        pts_R = np.array([[right_point[0]], [right_point[1]]], dtype=np.float64)

        # Ha van kalibrálás: torzítás-korrekciót alkalmazzunk
        if self._is_calibrated and self._D1 is not None:
            # Torzítás-korrekció és normalizálás
            pts_L = cv2.undistortPoints(
                pts_L.T.reshape(-1, 1, 2), self._K1, self._D1, P=self._P1
            ).reshape(2, -1)
            pts_R = cv2.undistortPoints(
                pts_R.T.reshape(-1, 1, 2), self._K2, self._D2, P=self._P2
            ).reshape(2, -1)

            # Epipoláris Y-illeszkedés ellenőrzése (rektifikált képeken a Y koordinátáknak kb. egyezniük kell)
            y_L_rect = pts_L[1, 0]
            y_R_rect = pts_R[1, 0]
            if abs(y_L_rect - y_R_rect) > 150.0:  # 150 pixel feletti eltolás esetén nem ugyanaz az objektum!
                logger.debug("triangulate: Epipoláris Y eltérés túl nagy (L_y=%.1f, R_y=%.1f)", y_L_rect, y_R_rect)
                return None

        # Háromszögelés (Hartley-Sturm lineáris módszer)
        # Eredmény: 4×N homogén koordináták
        pts_4d = cv2.triangulatePoints(self._P1, self._P2, pts_L, pts_R)

        # Homogén → euklideszi koordináták
        W = pts_4d[3, 0]
        if abs(W) < 1e-10:
            logger.debug("triangulate: W ≈ 0, érvénytelen pont")
            return None

        X = pts_4d[0, 0] / W
        Y = pts_4d[1, 0] / W
        Z = pts_4d[2, 0] / W

        # Ellenőrzés: Z pozitívnak kell lennie (a kamera előtt van)
        if Z < 0:
            logger.debug("triangulate: negatív Z=%.1f (a kamera mögött)", Z)
            return None

        # Ellenőrzés: fizikailag értelmező tartomány (0–15 m)
        if Z > 15000.0:
            logger.debug("triangulate: Z=%.1f mm túl messze (>15m)", Z)
            return None

        # ── Koordináta-rendszer átalakítás ─────────────────────────────────
        # A háromszögelés a BAL KAMERA koordináta-rendszerében számol:
        #   X_cam: viszszintes (bal kamera optikai tengelytől számolva)
        #   Y_cam: lefelé pozitív (kamera konvenció)
        #   Z_cam: a kamera előtt pozitív (mélység)
        #
        # Konvertálás kapu-koordináta-rendszerbe:
        #   X_goal = X_cam + left_cam_x_mm
        #           (bal kamera X = -1070mm a kapu közepétől, tehát: X_cam+(-1070))
        #   Y_goal = camera_height_mm - Y_cam
        #           (kamera Y lefelé nő, kapu Y felfelé; talaj = 0)
        #   Z_goal = Z_cam + camera_z_offset_mm
        #           (kamera 900mm-rel a gólvonal mögött: Z_cam-900 → Z=0 a gólvonal)
        x_goal = X + self._left_cam_x_mm            # bal kamera X offsetje
        y_goal = self._cam_height_mm - Y            # Y megfordítása, talaj=0
        z_goal = Z + self._cam_z_offset_mm          # Z offset: gólvonal = 0

        logger.debug(
            "triangulate: cam=(%.0f, %.0f, %.0f) → goal=(%.0f, %.0f, %.0f) mm",
            X, Y, Z, x_goal, y_goal, z_goal
        )

        return np.array([x_goal, y_goal, z_goal], dtype=np.float64)

    def rectify_pair(
        self,
        frame_left: np.ndarray,
        frame_right: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sztereó rektifikációt alkalmaz mindkét képre.

        A rektifikált képeken az epipoláris vonalak vízszintesek,
        ami megkönnyíti a diszparitás-keresést.

        Args:
            frame_left:  Bal kamera nyers képe
            frame_right: Jobb kamera nyers képe

        Returns:
            Tuple: (rektifikált bal kép, rektifikált jobb kép)
        """
        if self._map1_L is None or not self._is_calibrated:
            # Ha nincs kalibrálás: változatlanul adjuk vissza
            return frame_left, frame_right

        rect_left = cv2.remap(frame_left, self._map1_L, self._map2_L, cv2.INTER_LINEAR)
        rect_right = cv2.remap(frame_right, self._map1_R, self._map2_R, cv2.INTER_LINEAR)

        return rect_left, rect_right

    # ------------------------------------------------------------------
    # 2D Visszavetítés (Rajzoláshoz)
    # ------------------------------------------------------------------

    def project_to_2d(self, pts_3d_goal: np.ndarray, is_left: bool) -> Optional[np.ndarray]:
        """
        Kapu-koordinátarendszerben lévő 3D pontokat (N, 3) vetít vissza a 
        kamera nyers 2D képére (N, 2).

        Args:
            pts_3d_goal: (N, 3) alakú NumPy tömb (X, Y, Z mm-ben)
            is_left:     True = Bal kamera, False = Jobb kamera

        Returns:
            (N, 2) alakú NumPy tömb (x, y pixel koordináták), 
            vagy None, ha nincs kalibrálva / hiba történt.
        """
        if not self._is_calibrated or pts_3d_goal is None or len(pts_3d_goal) == 0:
            return None
            
        pts_3d_goal = np.atleast_2d(pts_3d_goal)
            
        # 1. Konvertálás a BAL kamera koordinátarendszerébe
        X_cam = pts_3d_goal[:, 0] - self._left_cam_x_mm
        Y_cam = self._cam_height_mm - pts_3d_goal[:, 1]
        Z_cam = pts_3d_goal[:, 2] - self._cam_z_offset_mm
        pts_3d_cam = np.column_stack((X_cam, Y_cam, Z_cam)).astype(np.float64)
        
        # 2. Vetítés a megfelelő kamerára
        if is_left:
            # Bal kamera a referencia, tehát az rvec és tvec 0
            rvec = np.zeros((3, 1), dtype=np.float64)
            tvec = np.zeros((3, 1), dtype=np.float64)
            K = self._K1
            D = self._D1
        else:
            # Jobb kamerához a bal-jobb transzformációs mátrix (R, T) kell
            rvec, _ = cv2.Rodrigues(self._R)
            tvec = self._T
            K = self._K2
            D = self._D2
            
        try:
            pts_2d, _ = cv2.projectPoints(pts_3d_cam, rvec, tvec, K, D)
            return pts_2d.reshape(-1, 2)
        except Exception as exc:
            logger.debug("Hiba a visszavetítés során: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Property-k
    # ------------------------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        """True ha a kalibrálás sikeresen betöltve."""
        return self._is_calibrated

    @property
    def baseline_mm(self) -> float:
        """A két kamera közötti baseline távolság mm-ben."""
        if self._T is not None:
            return float(np.linalg.norm(self._T))
        return self._baseline_mm

    @property
    def focal_length_px(self) -> float:
        """Bal kamera fókusztávolsága pixelben."""
        if self._K1 is not None:
            return float(self._K1[0, 0])
        return float(self._K_est[0, 0])
