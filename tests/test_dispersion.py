import numpy as np
from src.physics.dispersion import evaluate_gaussian_plume


def test_upwind_returns_zero():
    # Source south of receptor, wind from south (180 FROM -> blows north) => receptor is downwind? Need upwind case
    # Wind FROM north (0 deg, blows south): source north of receptor is upwind? Let's test receptor south of source with north wind = downwind positive, opposite = zero
    c_down = evaluate_gaussian_plume(0.5, 101.0, 0.0, 101.0, Q=2.0, wind_speed=5.0, wind_dir_deg=0, precip_rate=0)
    c_up = evaluate_gaussian_plume(0.0, 101.0, 0.5, 101.0, Q=2.0, wind_speed=5.0, wind_dir_deg=0, precip_rate=0)
    assert c_down >= 0
    assert c_up == 0.0


def test_scavenging_monotonic():
    c_dry = evaluate_gaussian_plume(0.0, 101.0, 0.5, 101.0, Q=2.0, wind_speed=5.0, wind_dir_deg=180, precip_rate=0)
    c_wet = evaluate_gaussian_plume(0.0, 101.0, 0.5, 101.0, Q=2.0, wind_speed=5.0, wind_dir_deg=180, precip_rate=5.0)
    assert c_wet < c_dry
    assert c_wet >= 0


def test_crosswind_decay():
    # Same downwind distance, larger crosswind should reduce concentration
    c_center = evaluate_gaussian_plume(0.0, 101.0, 1.0, 101.0, Q=2.0, wind_speed=5.0, wind_dir_deg=180, precip_rate=0)
    c_off = evaluate_gaussian_plume(0.0, 101.0, 1.0, 101.5, Q=2.0, wind_speed=5.0, wind_dir_deg=180, precip_rate=0)
    assert c_center > c_off


def test_distance_positive():
    # Kalimantan to Kuching ~ moderate distance
    c = evaluate_gaussian_plume(-2.415, 111.512, 1.5533, 110.3592, Q=6.21, wind_speed=4.17, wind_dir_deg=225, precip_rate=0.5)
    assert np.isfinite(c) and c >= 0
