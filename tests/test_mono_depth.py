from stereo.mono_depth_estimator import MonoDepthEstimator

def test_mono_depth_estimation():
    config = {
        "mono_depth": {
            "enabled": True,
            "min_radius_px": 5.0,
            "weight_stereo": 0.85,
            "weight_mono": 0.15,
            "alert_threshold_mm": 300.0,
        },
        "ball": {
            "diameter_mm": 210.0,
        },
        "geometry": {
            "focal_length_px": 1365.2,
        }
    }
    mono = MonoDepthEstimator(config)
    assert mono.enabled

    # Z = f * D / (2 * r)
    # ha r = 14.335 px -> Z = 1365.2 * 210 / (2 * 14.335) = 10000 mm (10 méter)
    z = mono.estimate_z(14.335)
    assert z is not None
    assert abs(z - 10000.0) < 50.0

    # Sztereo Z megőrzése tesztelése (a kalibrált sztereó Z marad tekintettel)
    z_fused, valid, warn = mono.validate_and_fuse_stereo_z(9900.0, 14.335)
    assert valid
    assert warn is None
    assert z_fused == 9900.0

    # Túl nagy eltérés (kalibrációs hiba jele)
    _, valid_bad, _ = mono.validate_and_fuse_stereo_z(8500.0, 14.335)
    assert not valid_bad
