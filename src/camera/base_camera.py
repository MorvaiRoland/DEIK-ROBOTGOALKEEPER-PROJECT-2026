"""
DEIK Robot Foci Kapus – Kamera alap interfész (Abstract Base Class)
====================================================================

Ez a modul definiálja az összes kamera implementáció közös interfészét.
Az ABC (Abstract Base Class) minta alkalmazásával garantáljuk, hogy
minden kamera típus (Ximea, Mock, Webcam) ugyanazon metódusokat valósítja meg.

Tervezési elvek:
    - Interface segregation: csak a szükséges metódusokat definiáljuk
    - Liskov substitution: bármelyik kamera felcserélhető a másikkal
    - Single responsibility: csak a kamera életciklust kezeli
"""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Tuple

# pyrefly: ignore [missing-import]
import cv2
import numpy as np

# Modul szintű napló
logger = logging.getLogger(__name__)


@dataclass
class CameraInfo:
    """
    Kamera metaadatait tároló adatosztály.

    Attributes:
        name:        Kamera neve / azonosítója
        width:       Kép szélessége pixelben
        height:      Kép magassága pixelben
        fps:         Célzott frame rate
        is_left:     True = bal oldali kamera, False = jobb oldali
        serial_num:  Gyártói sorozatszám (ha elérhető)
    """
    name: str = "Ismeretlen kamera"
    width: int = 0
    height: int = 0
    fps: float = 0.0
    is_left: bool = True
    serial_num: str = "N/A"


@dataclass
class CameraFrame:
    """
    Egy kamera frame-et reprezentáló adatosztály.

    Attributes:
        image:     BGR formátumú NumPy tömb, vagy None ha a leolvasás sikertelen
        timestamp: Frame megszerzésének ideje (UNIX timestamp, másodpercben)
        frame_id:  Monoton növekvő frame azonosító
        success:   True ha a frame sikeresen megszerzett
    """
    image: Optional[np.ndarray] = None
    timestamp: float = 0.0
    frame_id: int = 0
    success: bool = False


class BaseCamera(ABC):
    """
    Absztrakt alap kamera osztály.

    Minden konkrét kamera implementációnak (XimeaCamera, MockCamera stb.)
    ebből kell örökölnie és az összes @abstractmethod metódust meg kell
    valósítania.

    Example:
        class MyCamera(BaseCamera):
            def open(self) -> bool:
                # Kapcsolódás a kamerához
                ...
    """

    def __init__(self, info: CameraInfo):
        """
        Args:
            info: A kamera metaadatai (név, felbontás, stb.)
        """
        self._info = info
        self._is_open = False
        self._frame_count = 0

        # Kép transzformációs és beállítási paraméterek
        self._offset_x: int = 0
        self._offset_y: int = 0
        self._flip_h: bool = False
        self._flip_v: bool = False
        self._rotation: int = 0  # Landscape mód: kamerák vízszintesen szerelve
        self._exposure_us: int = 3000
        self._gain_db: float = 0.0
        self._auto_wb: bool = True
        self._wb_kr: float = 1.8
        self._wb_kg: float = 1.0
        self._wb_kb: float = 2.1

        logger.debug("Kamera objektum létrehozva: %s", info.name)

    # ------------------------------------------------------------------
    # Absztrakt metódusok – minden implementációban kötelező
    # ------------------------------------------------------------------

    @abstractmethod
    def open(self) -> bool:
        """
        Megnyitja a kamera kapcsolatot és megkezdi a képszerzést.

        Returns:
            True ha a megnyitás sikeres, False ha hiba lépett fel.
        """
        ...

    @abstractmethod
    def close(self) -> None:
        """Leállítja a képszerzést és bezárja a kamera kapcsolatot."""
        ...

    @abstractmethod
    def read(self) -> CameraFrame:
        """
        Olvas egy frame-et a kamerából.

        Returns:
            CameraFrame: A legfrissebb frame és metaadatai.
                         Ha a leolvasás sikertelen, a CameraFrame.success = False.
        """
        ...

    @abstractmethod
    def set_exposure(self, exposure_us: int) -> None:
        """
        Beállítja a zársebességet.

        Args:
            exposure_us: Zársebesség mikroszekundumban (pl. 3000 = 3 ms)
        """
        ...

    @abstractmethod
    def set_gain(self, gain_db: float) -> None:
        """
        Beállítja az erősítést.

        Args:
            gain_db: Erősítés dB-ben (0.0 = nincs erősítés)
        """
        ...

    @abstractmethod
    def get_fps(self) -> float:
        """
        Visszaadja a mért valódi frame rate-t.

        Returns:
            Mért FPS (frame per másodperc)
        """
        ...

    # ------------------------------------------------------------------
    # Kép elmozdulás (X/Y offset) és transzformációk
    # ------------------------------------------------------------------

    def set_offset(self, offset_x: int, offset_y: int) -> None:
        """Beállítja a kép X és Y tengely menti elmozdulását pixelben."""
        self._offset_x = int(offset_x)
        self._offset_y = int(offset_y)
        logger.debug("Offset beállítva: X=%d, Y=%d (%s)", self._offset_x, self._offset_y, self._info.name)

    def set_flip(self, flip_h: bool, flip_v: bool) -> None:
        """Beállítja a vízszintes és függőleges tükrözést."""
        self._flip_h = bool(flip_h)
        self._flip_v = bool(flip_v)

    def set_rotation(self, rotation: int) -> None:
        """Beállítja a kép elforgatását fokban (0, 90, 180, 270)."""
        if rotation in (0, 90, 180, 270):
            self._rotation = rotation

    def set_awb(self, enabled: bool) -> None:
        """Beállítja az automatikus fehéregyensúlyt."""
        self._auto_wb = bool(enabled)

    def set_wb(self, kr: float, kg: float, kb: float) -> None:
        """Beállítja a manuális fehéregyensúly RGB erősítéseit."""
        self._wb_kr = float(kr)
        self._wb_kg = float(kg)
        self._wb_kb = float(kb)

    def apply_image_transformations(self, image: Optional[np.ndarray]) -> Optional[np.ndarray]:
        """
        Alkalmazza a beállított X/Y elmozdulást, tükrözést és forgatást a képre.

        Args:
            image: BGR NumPy kép vagy None

        Returns:
            Transzformált BGR NumPy kép
        """
        if image is None:
            return None

        h, w = image.shape[:2]

        # 0. Szoftveres záridő / erősítés szimuláció (Mock / OpenCV kamera esetén)
        if not hasattr(self, "_cam") or self._cam is None:
            exp_us = getattr(self, "_exposure_us", 3000)
            gain_db = getattr(self, "_gain_db", 0.0)
            exp_ratio = exp_us / 3000.0
            gain_scale = 10.0 ** (gain_db / 20.0)
            total_alpha = exp_ratio * gain_scale
            if abs(total_alpha - 1.0) > 0.01:
                image = cv2.convertScaleAbs(image, alpha=total_alpha)

        # 1. X / Y elmozdulás (Offset / Shift) az X és Y tengelyen
        if self._offset_x != 0 or self._offset_y != 0:
            M = np.float32([[1, 0, self._offset_x], [0, 1, self._offset_y]])
            image = cv2.warpAffine(
                image, M, (w, h),
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0)
            )

        # 2. Tükrözések
        if self._flip_h and self._flip_v:
            image = cv2.flip(image, -1)
        elif self._flip_h:
            image = cv2.flip(image, 1)
        elif self._flip_v:
            image = cv2.flip(image, 0)

        # 3. Forgatás
        if self._rotation == 90:
            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        elif self._rotation == 180:
            image = cv2.rotate(image, cv2.ROTATE_180)
        elif self._rotation == 270:
            image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

        return image

    # ------------------------------------------------------------------
    # Opcionális metódusok – alapértelmezett implementációval
    # ------------------------------------------------------------------

    def get_temperature(self) -> float:
        """
        Visszaadja a szenzor hőmérsékletét (ha a kamera támogatja).

        Returns:
            Hőmérséklet Celsius fokban, vagy 0.0 ha nem támogatott.
        """
        return 0.0

    def set_roi(self, x: int, y: int, width: int, height: int) -> None:
        """
        Beállítja a Region of Interest (érdeklődési terület) ablakot.
        Alapértelmezetten nem csinál semmit – az implementáció dönti el.

        Args:
            x:      ROI bal felső sarkának X koordinátája (pixelben)
            y:      ROI bal felső sarkának Y koordinátája (pixelben)
            width:  ROI szélessége (pixelben)
            height: ROI magassága (pixelben)
        """
        logger.debug("set_roi() nem implementált ebben a kamerában: %s", self._info.name)

    # ------------------------------------------------------------------
    # Property-k – közvetlenül elérhető attribútumok
    # ------------------------------------------------------------------

    @property
    def info(self) -> CameraInfo:
        """A kamera metaadatai (csak olvasható)."""
        return self._info

    @property
    def is_open(self) -> bool:
        """True ha a kamera kapcsolat aktív."""
        return self._is_open

    @property
    def frame_count(self) -> int:
        """Az eddig megszerzett frame-ek száma."""
        return self._frame_count

    # ------------------------------------------------------------------
    # Context manager támogatás (with ... as cam: ...)
    # ------------------------------------------------------------------

    def __enter__(self) -> "BaseCamera":
        """Context manager belépés – automatikusan megnyitja a kamerát."""
        if not self.open():
            raise RuntimeError(f"Nem sikerült megnyitni a kamerát: {self._info.name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager kilépés – automatikusan bezárja a kamerát."""
        self.close()

    def __repr__(self) -> str:
        status = "NYITVA" if self._is_open else "ZÁRVA"
        return (
            f"{self.__class__.__name__}("
            f"name='{self._info.name}', "
            f"res={self._info.width}×{self._info.height}, "
            f"fps={self._info.fps}, "
            f"status={status})"
        )

