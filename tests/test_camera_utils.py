"""
DEIK Robot Foci Kapus – Tesztek: Camera Utils (test_camera_utils.py)
"""

from camera.camera_utils import ensure_usbfs_memory_mb


def test_ensure_usbfs_memory_mb():
    result = ensure_usbfs_memory_mb(0)
    assert isinstance(result, bool)
