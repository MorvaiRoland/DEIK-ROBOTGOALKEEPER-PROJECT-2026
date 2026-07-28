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
