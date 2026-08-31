import numpy as np
import cv2
from detection.optical_flow_tracker import OpticalFlowTracker

def test_optical_flow_basic():
    config = {
        "enabled": True,
        "window_size": 21,
        "max_pyramid_levels": 4,
        "fb_error_threshold_px": 2.5,
        "max_displacement_px": 250.0,
        "max_frames_without_yolo": 5,
    }
    tracker = OpticalFlowTracker(config)
    assert not tracker.is_tracking

    # Készítünk két szürke képet egy elmozdult fehér körrel
    frame1 = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(frame1, (100, 100), 15, 255, -1)

    frame2 = np.zeros((400, 400), dtype=np.uint8)
    cv2.circle(frame2, (105, 108), 15, 255, -1)

    tracker.update_from_yolo(frame1, 100.0, 100.0, 15.0)
    assert tracker.is_tracking

    res = tracker.track(frame2)
    assert res is not None
    cx, cy = res
    assert abs(cx - 105.0) < 2.0
    assert abs(cy - 108.0) < 2.0
    assert tracker.frames_since_yolo == 1
