"""Gridded haze forecast: sum Gaussian plume contributions over a lat/lon grid.

Provides pm25 + API values for heatmap layer and per-station forecasts.
"""

import numpy as np
import pandas as pd
from src.physics.dispersion import evaluate_gaussian_plume
from src.physics.api import pm25_to_api, api_category
from src.config import EARTH_RADIUS_M, SCAVENGE_A, SCAVENGE_B


def build_haze_grid(
    hotspots_df: pd.DataFrame,
    meteo: dict,
    lat_min: float = 0.5,
    lat_max: float = 6.5,
    lon_min: float = 99.5,
    lon_max: float = 119.0,
    n_lat: int = 18,
    n_lon: int = 22,
    background_pm25: float = 8.0,
    weight_by_posterior: bool = True,
    amplification: float = 80.0,
) -> pd.DataFrame:
    """Compute predicted PM2.5 on grid by summing plume contributions.

    Returns DataFrame with columns lat, lon, pm25, api, category, weight.
    """
    if hotspots_df.empty:
        return pd.DataFrame(columns=["lat", "lon", "pm25", "api", "category"])

    lats = np.linspace(lat_min, lat_max, n_lat)
    lons = np.linspace(lon_min, lon_max, n_lon)

    # Precompute Q and weights
    weights = hotspots_df.get("posterior_prob", pd.Series([1.0]*len(hotspots_df))).values
    if not weight_by_posterior:
        weights = np.ones(len(hotspots_df))
        weights /= weights.sum() if weights.sum() > 0 else 1.0
    # If weight_by_posterior, weights already sum to 1; use Q directly but scale contribution
    frps = hotspots_df["frp"].values
    Qs = frps * 0.02  # kg/s

    ws = float(meteo.get("wind_speed", 4.5))
    wd = float(meteo.get("wind_dir", 225))
    pr = float(meteo.get("precipitation", 0.5))

    def _long_range_heuristic(slat, slon, tlat, tlon, frp, ws, wd, pr, weight):
        R = EARTH_RADIUS_M
        dlat = np.radians(tlat - slat)
        dlon = np.radians(tlon - slon)
        lat_mid = np.radians((slat + tlat) / 2.0)
        x_dist = R * dlon * np.cos(lat_mid)
        y_dist = R * dlat
        phi = np.radians(270.0 - wd)
        x_down = x_dist * np.cos(phi) + y_dist * np.sin(phi)
        y_cross = -x_dist * np.sin(phi) + y_dist * np.cos(phi)
        if x_down <= 0:
            return 0.0
        x_down = max(x_down, 1000.0)
        L = 380_000.0
        along = np.exp(-x_down / L)
        sigma_cross = 0.42 * x_down + 18000.0
        cross = np.exp(-0.5 * (y_cross / sigma_cross) ** 2)
        Lambda = SCAVENGE_A * (pr ** SCAVENGE_B) if pr > 0 else 0.0
        scav = float(np.exp(-Lambda * (x_down / max(ws, 0.5))))
        wind_factor = 3.5 / max(ws, 1.0)
        base = float(frp) * 3.2 * along * cross * scav * wind_factor * float(weight)
        return float(np.clip(base, 0, 400))

    rows = []
    for la in lats:
        for lo in lons:
            total = background_pm25
            plume_sum = 0.0
            for idx, (_, spot) in enumerate(hotspots_df.iterrows()):
                w = float(weights[idx]) if weight_by_posterior else 1.0
                # Near-field (<60km): keep physical Gaussian, blended
                c_near = evaluate_gaussian_plume(float(spot["latitude"]), float(spot["longitude"]), float(la), float(lo), float(Qs[idx] * w), ws, wd, pr)
                c_far = _long_range_heuristic(float(spot["latitude"]), float(spot["longitude"]), float(la), float(lo), float(hotspots_df.iloc[idx]["frp"]), ws, wd, pr, w if weight_by_posterior else 1.0)
                # Choose max of near/far so city near fire gets Gaussian peak, far cities get heuristic
                c = max(c_near * float(amplification) * 0.08, c_far * float(amplification) / 80.0)
                # When amplification=80, far field is at calibrated level; scaling amplification scales it linearly
                plume_sum += c
            total += plume_sum
            total = float(np.clip(total, 0, 600))
            api = pm25_to_api(total)
            cat, _ = api_category(api)
            rows.append({"lat": la, "lon": lo, "pm25": total, "api": api, "category": cat})
    return pd.DataFrame(rows)


def predict_for_receptors(
    hotspots_df: pd.DataFrame,
    receptors_df: pd.DataFrame,
    meteo: dict,
    background_pm25: float = 8.0,
    weight_by_posterior: bool = True,
    amplification: float = 80.0,
) -> pd.DataFrame:
    """Sum plumes at receptor points (forecast API)."""
    if hotspots_df.empty:
        return pd.DataFrame(columns=["station_name", "lat", "lon", "pm25_pred", "api_pred", "category"])
    weights = hotspots_df.get("posterior_prob", pd.Series([1.0]*len(hotspots_df))).values
    frps = hotspots_df["frp"].values
    Qs = frps * 0.02
    ws = float(meteo.get("wind_speed", 4.5))
    wd = float(meteo.get("wind_dir", 225))
    pr = float(meteo.get("precipitation", 0.5))
    def _long_range_heuristic_r(slat, slon, tlat, tlon, frp, ws, wd, pr, weight):
        R = EARTH_RADIUS_M
        dlat = np.radians(tlat - slat)
        dlon = np.radians(tlon - slon)
        lat_mid = np.radians((slat + tlat) / 2.0)
        x_dist = R * dlon * np.cos(lat_mid)
        y_dist = R * dlat
        phi = np.radians(270.0 - wd)
        x_down = x_dist * np.cos(phi) + y_dist * np.sin(phi)
        y_cross = -x_dist * np.sin(phi) + y_dist * np.cos(phi)
        if x_down <= 0:
            return 0.0
        x_down = max(x_down, 1000.0)
        L = 380_000.0
        along = np.exp(-x_down / L)
        sigma_cross = 0.42 * x_down + 18000.0
        cross = np.exp(-0.5 * (y_cross / sigma_cross) ** 2)
        Lambda = SCAVENGE_A * (pr ** SCAVENGE_B) if pr > 0 else 0.0
        scav = float(np.exp(-Lambda * (x_down / max(ws, 0.5))))
        wind_factor = 3.5 / max(ws, 1.0)
        base = float(frp) * 3.2 * along * cross * scav * wind_factor * float(weight)
        return float(np.clip(base, 0, 400))

    rows = []
    for _, rec in receptors_df.iterrows():
        total = background_pm25
        plume_sum = 0.0
        for idx, (_, spot) in enumerate(hotspots_df.iterrows()):
            w = float(weights[idx]) if weight_by_posterior else 1.0
            c_near = evaluate_gaussian_plume(float(spot["latitude"]), float(spot["longitude"]), float(rec["latitude"]), float(rec["longitude"]), float(Qs[idx] * w), ws, wd, pr)
            c_far = _long_range_heuristic_r(float(spot["latitude"]), float(spot["longitude"]), float(rec["latitude"]), float(rec["longitude"]), float(hotspots_df.iloc[idx]["frp"]), ws, wd, pr, w if weight_by_posterior else 1.0)
            c = max(c_near * float(amplification) * 0.08, c_far * float(amplification) / 80.0)
            plume_sum += c
        total += plume_sum
        total = float(np.clip(total, 0, 600))
        api = pm25_to_api(total)
        cat, _ = api_category(api)
        rows.append({"station_name": rec["station_name"], "lat": rec["latitude"], "lon": rec["longitude"], "pm25_obs": float(rec["pm25_obs"]), "pm25_pred": total, "api_pred": api, "category": cat})
    return pd.DataFrame(rows)
