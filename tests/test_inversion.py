import pandas as pd
from src.physics.inversion import run_bayesian_inversion


def test_posterior_sums_to_one():
    hotspots = pd.DataFrame({"latitude": [0.5, -1.0], "longitude": [101.0, 103.0], "frp": [100, 200], "confidence": [90, 90]})
    receptors = pd.DataFrame({"station_name": ["KL"], "latitude": [3.139], "longitude": [101.68], "pm25_obs": [80.0]})
    meteo = {"wind_speed": 5.0, "wind_dir": 225, "precipitation": 0.5}
    out = run_bayesian_inversion(hotspots, receptors, meteo)
    assert abs(out["posterior_prob"].sum() - 1.0) < 1e-6
    assert (out["posterior_prob"] >= 0).all()


def test_does_not_mutate_input():
    hotspots = pd.DataFrame({"latitude": [0.5], "longitude": [101.0], "frp": [100], "confidence": [90]})
    receptors = pd.DataFrame({"station_name": ["KL"], "latitude": [3.139], "longitude": [101.68], "pm25_obs": [80.0]})
    meteo = {"wind_speed": 5.0, "wind_dir": 225, "precipitation": 0.5}
    orig_cols = list(hotspots.columns)
    run_bayesian_inversion(hotspots, receptors, meteo)
    assert list(hotspots.columns) == orig_cols


def test_empty_hotspots():
    hotspots = pd.DataFrame({"latitude": [], "longitude": [], "frp": [], "confidence": []})
    receptors = pd.DataFrame({"station_name": ["KL"], "latitude": [3.139], "longitude": [101.68], "pm25_obs": [80.0]})
    out = run_bayesian_inversion(hotspots, receptors, {"wind_speed": 5, "wind_dir": 225, "precipitation": 0})
    assert out.empty
