"""
Tests for CameraManager sync configuration and role assignment.
"""
import pytest
from camera.camera_manager import CameraManager

def test_camera_sync_config_roles():
    config = {
        "camera": {
            "type": "mock",
            "resolution": {"width": 1936, "height": 1216},
            "fps": 100,
            "exposure_time_us": 3000,
            "gain_db": 0.0,
            "bandwidth_mode": "unlimited",
            "sync": {
                "enabled": True,
                "master_side": "right",
                "gpo_selector": "XI_GPO_PORT2",
                "gpo_mode": "XI_GPO_EXPOSURE_ACTIVE",
                "gpi_selector": "XI_GPI_PORT2",
                "gpi_mode": "XI_GPI_TRIGGER",
                "trigger_source": "XI_TRG_EDGE_RISING",
                "fallback_to_software_sync": True,
            },
            "left": {"serial_number": "CACAU2546000"},
            "right": {"serial_number": "CACAU2517001"},
            "mock": {"left_source": 0, "right_source": 1}
        }
    }

    manager = CameraManager(config)
    assert manager._hw_sync_enabled is True
    assert manager._master_side == "right"

    # Test camera creation role logic
    cam_left = manager._create_camera(is_left=True)
    cam_right = manager._create_camera(is_left=False)

    assert cam_left is not None
    assert cam_right is not None
