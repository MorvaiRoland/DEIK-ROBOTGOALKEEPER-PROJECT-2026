"""
DEIK Robot Foci Kapus – Lucas-Kanade Optikai Flow Tracker
==========================================================

Sparse optical flow-alapú labda tracker a YOLO detektálások között.
A YOLO ~60 Hz-en fut, de a kamera 100 FPS-en. A közbülső frame-eken
ez a modul pixel-pontosan követi a labdát LK optical flow-val.

Miért Lucas-Kanade?
    - CPU-n ~0.5-1 ms/frame → elhanyagolható overhead
    - Sub-pixel pontosság (piramis alapú, 4 szint = 16× keresési tartomány)
    - Forward-backward konzisztencia ellenőrzéssel
    - Automatikus reset ha a flow divergál (YOLO átveszi)

Pipeline:
    1. YOLO detektál → update_from_yolo(frame_gray, cx, cy, radius)
    2. Következő frame → track(curr_gray) → sub-pixel center
    3. Ha YOLO ismét detektál → update_from_yolo(...)  → korrekció
"""

import logging
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class OpticalFlowTracker:
    """
    Lucas-Kanade sparse optical flow tracker labdakövetéshez.

    Forward-backward error ellenőrzéssel: ha a flow nem visszafordítható
    (error > threshold), a tracker invalidálódik és a következő YOLO-t vár.
    """

    def __init__(self, config: dict):
        """
        Args:
            config: A system_config.yaml "optical_flow" szekciója
        """
        self._enabled = bool(config.get("enabled", True))
        win_size = int(config.get("window_size", 21))
        max_level = int(config.get("max_pyramid_levels", 4))
        self._fb_threshold = float(config.get("fb_error_threshold_px", 2.5))
        self._max_disp = float(config.get("max_displacement_px", 250.0))
        self._max_frames_lost = int(config.get("max_frames_without_yolo", 12))

        self._lk_params = dict(
            winSize=(win_size, win_size),
            maxLevel=max_level,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.01),
        )

        self._point: Optional[np.ndarray] = None
        self._prev_gray: Optional[np.ndarray] = None
        self._radius: float = 20.0
        self._is_tracking: bool = False
        self._frames_since_yolo: int = 0

        logger.debug(
            "OpticalFlowTracker kész: enabled=%s, win=%dx%d, lvl=%d, fb_thresh=%.1f px",
            self._enabled, win_size, win_size, max_level, self._fb_threshold
        )

    def update_from_yolo(
        self,
        frame_gray: np.ndarray,
        cx: float,
        cy: float,
        radius: float,
    ) -> None:
        """
        YOLO detektálás után frissíti a tracker állapotát.

        Args:
            frame_gray: Szürkeárnyalatos kép (np.ndarray, uint8)
            cx, cy:     Labda középpont pixelben
            radius:     Labda sugara pixelben
        """
        if not self._enabled:
            return
        self._point = np.array([[[cx, cy]]], dtype=np.float32)
        self._prev_gray = frame_gray.copy()
        self._radius = radius
        self._is_tracking = True
        self._frames_since_yolo = 0
        logger.debug("OpticalFlow: YOLO korrekció (%.1f, %.1f) r=%.1f", cx, cy, radius)

    def track(self, curr_gray: np.ndarray) -> Optional[Tuple[float, float]]:
        """
        Követi a labdát az előző frame-ről az aktuálisra.

        Args:
            curr_gray: Aktuális szürkeárnyalatos kép (np.ndarray, uint8)

        Returns:
            (cx, cy) sub-pixel pontossággal, vagy None ha érvénytelen
        """
        if not self._enabled or not self._is_tracking or self._point is None:
            return None
        if self._prev_gray is None:
            return None
        if self._frames_since_yolo >= self._max_frames_lost:
            logger.debug("OpticalFlow: max_frames_lost → reset")
            self._is_tracking = False
            return None

        # Forward pass
        next_pt, status_fwd, _ = cv2.calcOpticalFlowPyrLK(
            self._prev_gray, curr_gray, self._point, None, **self._lk_params
        )
        if next_pt is None or status_fwd is None or status_fwd[0, 0] == 0:
            logger.debug("OpticalFlow: forward pass FAILED")
            self._is_tracking = False
            return None

        # Backward pass (forward-backward konzisztencia)
        back_pt, status_bwd, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray, self._prev_gray, next_pt, None, **self._lk_params
        )
        if back_pt is None or status_bwd is None or status_bwd[0, 0] == 0:
            logger.debug("OpticalFlow: backward pass FAILED")
            self._is_tracking = False
            return None

        fb_error = float(np.linalg.norm(self._point[0, 0] - back_pt[0, 0]))
        if fb_error > self._fb_threshold:
            logger.debug("OpticalFlow: FB error=%.2f px > %.2f px", fb_error, self._fb_threshold)
            self._is_tracking = False
            return None

        disp = float(np.linalg.norm(next_pt[0, 0] - self._point[0, 0]))
        if disp > self._max_disp:
            logger.debug("OpticalFlow: elmozdulás=%.1f px > %.1f px", disp, self._max_disp)
            self._is_tracking = False
            return None

        cx = float(next_pt[0, 0, 0])
        cy = float(next_pt[0, 0, 1])
        self._point = next_pt
        self._prev_gray = curr_gray.copy()
        self._frames_since_yolo += 1

        logger.debug(
            "OpticalFlow OK: (%.2f, %.2f) disp=%.1f, fb_err=%.2f, since_yolo=%d",
            cx, cy, disp, fb_error, self._frames_since_yolo
        )
        return cx, cy

    def reset(self) -> None:
        """Nullázza a tracker állapotát."""
        self._is_tracking = False
        self._point = None
        self._prev_gray = None
        self._frames_since_yolo = 0

    @property
    def is_tracking(self) -> bool:
        return self._is_tracking and self._enabled

    @property
    def frames_since_yolo(self) -> int:
        return self._frames_since_yolo

    @property
    def enabled(self) -> bool:
        return self._enabled
