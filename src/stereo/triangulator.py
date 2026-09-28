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
        self._epipolar_y_threshold = float(self._stereo_cfg.get("epipolar_y_threshold_px", 45.0))

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
        self._R1: Optional[np.ndarray] = None   # Bal rektifikációs forgatás (stereoRectify)
        self._R2: Optional[np.ndarray] = None   # Jobb rektifikációs forgatás (stereoRectify)
        self._P1: Optional[np.ndarray] = None   # Bal projekciós mátrix (rektifikált)
        self._P2: Optional[np.ndarray] = None   # Jobb projekciós mátrix (rektifikált)
        self._Q: Optional[np.ndarray] = None    # Diszparitás → mélység mátrix

        # Torzítás-korrekciós térképek (cv2.remap()-hez)
        self._map1_L: Optional[np.ndarray] = None
        self._map2_L: Optional[np.ndarray] = None
        self._map1_R: Optional[np.ndarray] = None
        self._map2_R: Optional[np.ndarray] = None

        # Baseline (fallback, ha még nincs kalibrálás)
        # Fontos: A default értékek a fizikailag mért kamera-geometriát tükrözik.
        # Baseline = 2300 mm (mérve), ne változtasd config nélkül!
        self._baseline_mm = float(self._geo_cfg.get("baseline_mm", 2300.0))

        # Kalibrált-e a rendszer?
        self._is_calibrated = False

        # Fizikai kamera pozíciók (config-ból, fizikailag mért értékek)
        # Fontos: ezeket CSAK a config-alapú tartalék world-transzformáció használja,
        # amikor nincs kalibrált merev transzformáció (R_world_cam/t_world_cam) a .npz-ben.
        self._left_cam_x_mm  = float(self._geo_cfg.get("left_camera_x_mm",  -1150.0))
        self._cam_height_mm  = float(self._geo_cfg.get("camera_height_mm",   2800.0))
        self._cam_z_offset_mm = float(self._geo_cfg.get("camera_z_offset_mm", -900.0))
        self._pitch_deg = float(self._geo_cfg.get("camera_pitch_deg", 0.0))

        # Kalibrált merev világ-transzformáció (rektifikált BAL kamera keret → kapu-koord.).
        # A `scripts/calibrate_world.py` állítja elő ismert talajpontokból (Kabsch), és a
        # kalibrációs .npz-be menti. Ha jelen van, ez FELÜLÍRJA a fenti config-alapú képletet.
        self._R_wc: Optional[np.ndarray] = None   # 3×3 forgatás (rektifikált cam → world)
        self._t_wc: Optional[np.ndarray] = None   # 3× eltolás [mm]

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
        if not cal_path.is_absolute() and not cal_path.exists():
            project_relative_path = Path(__file__).resolve().parents[2] / cal_path
            if project_relative_path.exists():
                cal_path = project_relative_path

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

            # Figyelmeztetés, ha a kalibrálás felbontása eltér a menet közbeni
            # kamerafelbontástól: a pixel-koordináták más képkeretben vannak, ami
            # rendszerhibát okoz a trianguláció X/Y/Z értékeiben. Ilyenkor a
            # kalibrálást a jelenlegi felbontáson kell újra elvégezni.
            cap_res = self._config.get("camera", {}).get("resolution", {})
            cap_w = int(cap_res.get("width", img_width))
            cap_h = int(cap_res.get("height", img_height))
            if (cap_w, cap_h) != (img_width, img_height):
                logger.warning(
                    "FELBONTÁS ELTÉRÉS: kalibrálás=%dx%d, kamera=%dx%d px. "
                    "A pixelkoordináták más képkeretben vannak → pontatlan 3D! "
                    "Kalibrálj újra a jelenlegi felbontáson: python scripts/calibrate_stereo.py",
                    img_width, img_height, cap_w, cap_h
                )

            # Torzítás-korrekciós térképek kiszámítása
            self._compute_rectification_maps(img_width, img_height)

            # Kalibrált merev világ-transzformáció (opcionális, scripts/calibrate_world.py).
            # Ha jelen van, ez adja a pontos kapu-koordinátákat a config-alapú
            # pitch/magasság képlet helyett.
            if "R_world_cam" in data.files and "t_world_cam" in data.files:
                self._R_wc = np.asarray(data["R_world_cam"], dtype=np.float64).reshape(3, 3)
                self._t_wc = np.asarray(data["t_world_cam"], dtype=np.float64).reshape(3)
                logger.info("✓ Kalibrált világ-transzformáció betöltve (merev R|t, talajpont-illesztés).")
            else:
                self._R_wc = None
                self._t_wc = None
                logger.warning(
                    "Nincs kalibrált világ-transzformáció a .npz-ben → config-alapú "
                    "(pitch=%.1f°, magasság=%.0fmm) TARTALÉK. Pontos X/Y/Z-hez futtasd: "
                    "python scripts/calibrate_world.py", self._pitch_deg, self._cam_height_mm
                )

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

        Figyelem: Ez csak közelítő értékeket ad! A pontos 3D pozícióhoz
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

        # Frissített projekciós mátrixok + rektifikációs forgatások
        # Fontos: R1/R2 kell a triangulációhoz (undistortPoints R paramétere),
        # különben a 2D pontok nem a rektifikált keretben vannak → hibás 3D.
        self._R1 = R1
        self._R2 = R2
        self._P1 = P1_rect
        self._P2 = P2_rect
        self._Q = Q

        logger.debug("Rektifikációs térképek kiszámítva: %dx%d", width, height)

    # ------------------------------------------------------------------
    # Főbb algoritmus: 3D háromszögelés
    # ------------------------------------------------------------------

    def triangulate_rect(
        self,
        left_point: Tuple[float, float],
        right_point: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """
        A labda 3D pozícióját adja a REKTIFIKÁLT BAL kamera koordináta-rendszerében.

        Ez a nyers háromszögelés kimenete (világ-transzformáció NÉLKÜL). A
        `triangulate()` ezt hívja, majd a kapu-koordinátarendszerbe konvertál.
        A world-kalibráció (Kabsch) is erre a keretre illeszti a merev transzformációt.

        Returns:
            NumPy [X, Y, Z] mm (rektifikált cam keret), vagy None ha sikertelen.
        """
        if self._P1 is None or self._P2 is None:
            logger.warning("triangulate() hívás kalibrálás nélkül!")
            return None

        # 2D pontok (2×N mátrix, N=1 pont)
        pts_L = np.array([[left_point[0]], [left_point[1]]], dtype=np.float64)
        pts_R = np.array([[right_point[0]], [right_point[1]]], dtype=np.float64)

        # Ha van kalibrálás: torzítás-korrekció + REKTIFIKÁCIÓS FORGATÁS.
        # A triangulatePoints() a rektifikált P1/P2 mátrixokat használja, ezért a
        # 2D pontokat is a rektifikált keretbe kell hozni: undistortPoints R=R1/R2.
        if self._is_calibrated and self._D1 is not None:
            pts_L = cv2.undistortPoints(
                pts_L.T.reshape(-1, 1, 2), self._K1, self._D1,
                R=self._R1, P=self._P1
            ).reshape(2, -1)
            pts_R = cv2.undistortPoints(
                pts_R.T.reshape(-1, 1, 2), self._K2, self._D2,
                R=self._R2, P=self._P2
            ).reshape(2, -1)

            # Epipoláris Y-illeszkedés ellenőrzése (rektifikált képeken a Y-oknak kb. egyezniük kell)
            y_L_rect = pts_L[1, 0]
            y_R_rect = pts_R[1, 0]
            if abs(y_L_rect - y_R_rect) > self._epipolar_y_threshold:
                logger.debug("triangulate: Epipoláris Y eltérés túl nagy (L_y=%.1f, R_y=%.1f, limit=%.1f)", y_L_rect, y_R_rect, self._epipolar_y_threshold)
                return None

        # Háromszögelés (Hartley-Sturm lineáris módszer) → 4×N homogén koordináták
        pts_4d = cv2.triangulatePoints(self._P1, self._P2, pts_L, pts_R)

        W = pts_4d[3, 0]
        if abs(W) < 1e-10:
            logger.debug("triangulate: W ≈ 0, érvénytelen pont")
            return None

        X = pts_4d[0, 0] / W
        Y = pts_4d[1, 0] / W
        Z = pts_4d[2, 0] / W

        # Ellenőrzés: Z pozitív (a kamera előtt) és fizikailag értelmes tartomány
        if Z < 0:
            logger.debug("triangulate: negatív Z=%.1f (a kamera mögött)", Z)
            return None
        if Z > 15000.0:
            logger.debug("triangulate: Z=%.1f mm túl messze (>15m)", Z)
            return None

        return np.array([X, Y, Z], dtype=np.float64)

    def triangulate(
        self,
        left_point: Tuple[float, float],
        right_point: Tuple[float, float],
    ) -> Optional[np.ndarray]:
        """
        Kiszámítja a labda 3D pozícióját KAPU-koordinátarendszerben (mm).

        A nyers (rektifikált kamera-keretbeli) háromszögelést a `triangulate_rect()`
        adja; itt csak a világ-transzformációt alkalmazzuk rá.

        Returns:
            NumPy [X, Y, Z] mm (kapu-koord.), vagy None ha a háromszögelés sikertelen.
        """
        p_rect = self.triangulate_rect(left_point, right_point)
        if p_rect is None:
            return None
        return self._rect_to_world(p_rect)

    # ------------------------------------------------------------------
    # Világ-koordináta transzformáció (rektifikált cam keret <-> kapu-koord.)
    # ------------------------------------------------------------------

    def _rect_to_world(self, p_rect: np.ndarray) -> np.ndarray:
        """
        Rektifikált BAL kamera keretbeli pontot (X, Y, Z) kapu-koordinátává alakít.

        1) Ha van KALIBRÁLT merev transzformáció (R_wc, t_wc) → azt használja (pontos).
        2) Különben a config-alapú tartalék: rektifikált → eredeti BAL kamera keret
           (R1-gyel), majd pitch-forgatás + magasság/offset (közelítő).
        """
        p_rect = np.asarray(p_rect, dtype=np.float64).reshape(3)

        if self._R_wc is not None and self._t_wc is not None:
            return self._R_wc @ p_rect + self._t_wc

        # --- Config-alapú tartalék ---
        # A rektifikált keretet vissza kell forgatni az EREDETI bal kamera optikai
        # keretébe (R1^T), mert a config pitch/magasság az eredeti kamerára vonatkozik.
        p_cam = self._R1.T @ p_rect if self._R1 is not None else p_rect
        X, Yc, Zc = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])

        rad = np.radians(self._pitch_deg)
        cos_p, sin_p = np.cos(rad), np.sin(rad)
        y_down = Yc * cos_p + Zc * sin_p
        z_fwd  = -Yc * sin_p + Zc * cos_p

        x_goal = X + self._left_cam_x_mm
        y_goal = self._cam_height_mm - y_down
        z_goal = z_fwd + self._cam_z_offset_mm
        return np.array([x_goal, y_goal, z_goal], dtype=np.float64)

    @property
    def left_focal_length_px(self) -> float:
        """Az EREDETI (nem rektifikált) bal kamera fókusztávolsága pixelben (K1)."""
        K = self._K1 if self._K1 is not None else self._K_est
        return float(K[0, 0])

    def left_pixel_to_normalized(self, u: float, v: float) -> Tuple[float, float]:
        """Nyers bal kamera pixel → torzításmentes normalizált (X/Z, Y/Z) az eredeti bal kamera keretben."""
        K = self._K1 if self._K1 is not None else self._K_est
        D = self._D1 if self._D1 is not None else np.zeros(5, dtype=np.float64)
        pt = cv2.undistortPoints(np.array([[[u, v]]], dtype=np.float64), K, D)
        return float(pt[0, 0, 0]), float(pt[0, 0, 1])

    def leftcam_original_to_world(self, p_cam: np.ndarray) -> np.ndarray:
        """
        EREDETI BAL kamera optikai keretbeli pontot (X, Y, Z) kapu-koordinátává alakít.

        Az egykamerás (mono-mélység) tartalékhoz: a pixel-visszavetítés az eredeti bal
        kamera keretében ad pontot; ez a metódus konzisztensen ugyanazt a világ-
        transzformációt alkalmazza, mint a sztereó út (kalibrált R_wc, vagy config-tartalék).
        """
        p_cam = np.asarray(p_cam, dtype=np.float64).reshape(3)
        if self._R_wc is not None and self._t_wc is not None:
            # eredeti bal cam → rektifikált:  R1 @ p_cam
            p_rect = self._R1 @ p_cam if self._R1 is not None else p_cam
            return self._R_wc @ p_rect + self._t_wc
        # config-alapú tartalék: az eredeti keretben már közvetlenül alkalmazható
        X, Yc, Zc = float(p_cam[0]), float(p_cam[1]), float(p_cam[2])
        rad = np.radians(self._pitch_deg)
        cos_p, sin_p = np.cos(rad), np.sin(rad)
        y_down = Yc * cos_p + Zc * sin_p
        z_fwd = -Yc * sin_p + Zc * cos_p
        return np.array([
            X + self._left_cam_x_mm,
            self._cam_height_mm - y_down,
            z_fwd + self._cam_z_offset_mm,
        ], dtype=np.float64)

    def _world_to_leftcam(self, pts_world: np.ndarray) -> np.ndarray:
        """
        Kapu-koordinátabeli (N, 3) pontok → EREDETI BAL kamera optikai keret (N, 3).

        A `_rect_to_world` egzakt inverze (a rektifikációs R1-et is beleértve), így a
        `project_to_2d` visszavetítés konzisztens a `triangulate` előreszámítással.
        """
        pts_world = np.atleast_2d(np.asarray(pts_world, dtype=np.float64))

        if self._R_wc is not None and self._t_wc is not None:
            # world → rektifikált cam:  R_wc^T @ (w - t)   (soronként: (w - t) @ R_wc)
            p_rect = (pts_world - self._t_wc) @ self._R_wc
            # rektifikált → eredeti bal cam:  R1^T @ p_rect  (soronként: p_rect @ R1)
            if self._R1 is not None:
                return p_rect @ self._R1
            return p_rect

        # --- Config-alapú tartalék inverze ---
        Xw, Yw, Zw = pts_world[:, 0], pts_world[:, 1], pts_world[:, 2]
        rad = np.radians(self._pitch_deg)
        cos_p, sin_p = np.cos(rad), np.sin(rad)
        y_down = self._cam_height_mm - Yw
        z_fwd  = Zw - self._cam_z_offset_mm
        Yc = y_down * cos_p - z_fwd * sin_p
        Zc = y_down * sin_p + z_fwd * cos_p
        Xc = Xw - self._left_cam_x_mm
        return np.column_stack((Xc, Yc, Zc)).astype(np.float64)

    @staticmethod
    def _rigid_transform_3d(
        src: np.ndarray,
        dst: np.ndarray,
        allow_reflection: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Kabsch–Umeyama merev illesztés (skálázás nélkül): dst ≈ R @ src + t.

        Args:
            src: (N, 3) forráspontok (rektifikált kamera keret)
            dst: (N, 3) célpontok (kapu-koordináta, mért)
            allow_reflection: Ha True, megengedi a det(R) = -1 ortogonális
                transzformációt (szükséges, ha src és dst ellenkező kezességű,
                pl. OpenCV kamera keret Y-lefelé vs kapu-keret Y-felfelé).

        Returns:
            (R 3×3 forgatás/ortogonális mátrix, t 3× eltolás)
        """
        src = np.asarray(src, dtype=np.float64).reshape(-1, 3)
        dst = np.asarray(dst, dtype=np.float64).reshape(-1, 3)

        centroid_src = src.mean(axis=0)
        centroid_dst = dst.mean(axis=0)
        src_c = src - centroid_src
        dst_c = dst - centroid_dst

        H = src_c.T @ dst_c
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        # Tükrözés kiszűrése csak akkor, ha nem engedélyezett (pl. azonos kezességű rendszereknél)
        if not allow_reflection and np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T
        t = centroid_dst - R @ centroid_src
        return R, t

    def calibrate_world_transform(
        self,
        cam_points_rect: np.ndarray,
        world_points: np.ndarray,
    ) -> dict:
        """
        Kiszámítja és beállítja a merev világ-transzformációt (R_wc, t_wc) ismert
        talajpontokból, hogy a trianguláció pontos kapu-koordinátákat adjon.

        Args:
            cam_points_rect: (N, 3) `triangulate_rect()`-tel kapott pontok
            world_points:    (N, 3) ugyanezekhez mért valós kapu-koordináták [mm]

        Returns:
            dict: {R_world_cam, t_world_cam, rms_mm, max_mm}
        """
        cam = np.asarray(cam_points_rect, dtype=np.float64).reshape(-1, 3)
        wld = np.asarray(world_points, dtype=np.float64).reshape(-1, 3)
        if len(cam) < 3 or len(cam) != len(wld):
            raise ValueError(
                f"Legalább 3, azonos számú pontpár kell (kapott: cam={len(cam)}, world={len(wld)})"
            )

        # OpenCV (Y-lefelé, jobbkezes) és a kapu-keret (Y-felfelé, balkezes)
        # ellenkező kezességű, így az ortogonális transzformáció determinánsa -1.
        R, t = self._rigid_transform_3d(cam, wld, allow_reflection=True)

        # Ha a talajpontok (Y=0) miatt az Y-tengely normálisa határozatlan lenne az SVD-ben:
        # A fizikai kényszert alkalmazzuk:
        # - A kamera a talaj felett van: t_y > 0
        # - A lefelé dőlő kamera képen a lefelé mozgás a világban is lefelé irányul: R[1, 1] < 0.
        is_coplanar_ground = np.std(wld[:, 1]) < 10.0
        if is_coplanar_ground:
            if R[1, 1] > 0 or t[1] < 0:
                R[1, :] *= -1.0
                t[1] = wld.mean(axis=0)[1] - float(R[1, :] @ cam.mean(axis=0))

        self._R_wc = R
        self._t_wc = t

        pred = (R @ cam.T).T + t
        res = np.linalg.norm(pred - wld, axis=1)
        return {
            "R_world_cam": R,
            "t_world_cam": t,
            "rms_mm": float(np.sqrt(np.mean(res ** 2))),
            "max_mm": float(res.max()),
        }

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

        # 1. Kapu-koordináta → EREDETI BAL kamera optikai keret.
        #    Ez a triangulate() világ-transzformációjának egzakt inverze (R1-gyel),
        #    így a rajzolt trajektória fedésben marad a detektálással.
        pts_3d_cam = self._world_to_leftcam(pts_3d_goal)

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
