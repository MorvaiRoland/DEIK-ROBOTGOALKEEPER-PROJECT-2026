"""
DEIK Robot Foci Kapus – Mono Mélység Becslő
============================================

A labda látszólagos pixelsugarából becsüli a Z mélységet.

Képlet:
    Z = f_px × D_ball_mm / (2 × r_px)

ahol:
    f_px      = kamera fókusztávolsága pixelben (kalibrálásból)
    D_ball_mm = labda fizikai átmérője mm-ben (konfig)
    r_px      = labda látszólagos sugara pixelben (YOLO detektálásból)

Felhasználás:
    1. Sztereó Z validálás: ha |Z_stereo - Z_mono| > threshold → figyelmeztetés
    2. Tartalék Z-forrás ha csak 1 kamera lát
    3. Súlyozott fúzió: Z_fused = w_s * Z_stereo + w_m * Z_mono
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


class MonoDepthEstimator:
    """
    Labdaméret-alapú monokularis Z-mélység becslő.

    Fizikai alap:
        A kamera perspektív vetítési modelljéből:
            Z [mm] = f_px [px] × D_ball [mm] / (2 × r_px [px])
        ahol f_px a kalibrált fókusztávolság, D_ball a labda valódi
        átmérője, r_px a detektált pixelsugar.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: A teljes system_config.yaml
        """
        mono_cfg = config.get("mono_depth", {})
        self._enabled = bool(mono_cfg.get("enabled", True))
        self._min_radius_px = float(mono_cfg.get("min_radius_px", 8.0))
        self._weight_stereo = float(mono_cfg.get("weight_stereo", 0.85))
        self._weight_mono = float(mono_cfg.get("weight_mono", 0.15))
        self._alert_threshold_mm = float(mono_cfg.get("alert_threshold_mm", 300.0))

        # Labda fizikai átmérő (mm)
        ball_cfg = config.get("ball", {})
        self._ball_diameter_mm = float(ball_cfg.get("diameter_mm", 210.0))

        # Fókusztávolság (px) – kalibráció frissítheti update_focal_length()-vel
        geo_cfg = config.get("geometry", {})
        self._f_px = float(geo_cfg.get("focal_length_px", 1365.2))

        logger.debug(
            "MonoDepthEstimator kész: enabled=%s, D=%.0f mm, f=%.1f px, w_s=%.2f, w_m=%.2f",
            self._enabled, self._ball_diameter_mm, self._f_px,
            self._weight_stereo, self._weight_mono
        )

    def update_focal_length(self, f_px: float) -> None:
        """Frissíti a fókusztávolságot kalibrálás után."""
        self._f_px = f_px
        logger.info("MonoDepth: fókusztávolság frissítve → %.1f px", f_px)

    @property
    def focal_length_px(self) -> float:
        """Aktuális fókusztávolság pixelben."""
        return self._f_px

    def estimate_z(self, radius_px: float) -> Optional[float]:
        """
        Becsüli a labda Z-mélységét a pixelsugar alapján.

        Args:
            radius_px: Labda sugara pixelben (YOLO / color-blob detektálásból)

        Returns:
            Z_mm (float), vagy None ha a sugár érvénytelen
        """
        if not self._enabled:
            return None
        if radius_px < self._min_radius_px:
            logger.debug(
                "MonoDepth: sugár túl kicsi (%.1f px < %.1f px)", radius_px, self._min_radius_px
            )
            return None

        z_mm = self._f_px * self._ball_diameter_mm / (2.0 * radius_px)

        # Fizikai tartomány szűrés: 0.5 m – 15 m között érvényes
        if z_mm < 500.0 or z_mm > 15000.0:
            logger.debug("MonoDepth: Z tartományon kívül: %.0f mm", z_mm)
            return None

        return z_mm

    def validate_and_fuse_stereo_z(
        self,
        z_stereo_mm: float,
        radius_px: float,
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Validálja a sztereó Z-t a mono becslés alapján, majd fúzióval finomítja.

        Args:
            z_stereo_mm: Sztereo triangulációból kapott Z (mm)
            radius_px:   Labda sugara pixelben

        Returns:
            Tuple: (z_fused_mm, valid, warning_message_or_None)
                - z_fused_mm: Súlyozott fúzió eredménye (ha valid), különben z_stereo_mm
                - valid:      False ha az eltérés > alert_threshold (kalibrációs hiba jele)
                - warning:    Szöveges figyelmeztetés vagy None
        """
        z_mono = self.estimate_z(radius_px)

        if z_mono is None:
            # Nem tudunk validálni – elfogadjuk a sztereo értéket
            return z_stereo_mm, True, None

        diff = abs(z_stereo_mm - z_mono)

        if diff > self._alert_threshold_mm:
            warning = (
                f"Sztereó-Mono Z eltérés NAGY: "
                f"stereo={z_stereo_mm:.0f} mm, mono={z_mono:.0f} mm, "
                f"diff={diff:.0f} mm (küszöb: {self._alert_threshold_mm:.0f} mm) "
                f"→ kalibrációs hiba gyanú!"
            )
            logger.debug(warning)
            return z_stereo_mm, False, None

        # Amikor a sztereo trianguláció kalibrált és érvényes, a sztereo Z a legpontosabb.
        # Megtartjuk a sztereó Z értéket, a mono csak ellenőrzésre szolgál.
        return z_stereo_mm, True, None

    def fallback_z(self, radius_px: float) -> Optional[float]:
        """
        Tartalék Z-becslés ha a sztereó triangulálás nem elérhető
        (pl. csak 1 kamera látja a labdát).

        Args:
            radius_px: Labda sugara pixelben

        Returns:
            Z_mm becslés, vagy None ha nem megbízható
        """
        z_mono = self.estimate_z(radius_px)
        if z_mono is not None:
            logger.info("MonoDepth: tartalék Z=%.0f mm (r=%.1f px)", z_mono, radius_px)
        return z_mono

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def focal_length_px(self) -> float:
        return self._f_px

    @property
    def ball_diameter_mm(self) -> float:
        return self._ball_diameter_mm
