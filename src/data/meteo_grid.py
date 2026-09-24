"""Gridded meteo fetcher - samples Open-Meteo over a coarse grid to render wind/rain layers.

Falls back to uniform field (no API fan-out) when network fails. Caching via streamlit is handled in app.py.
"""

import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
import requests


GRID_BOUNDS = {
    "lat_min": -3.0,
    "lat_max": 7.0,
    "lon_min": 98.0,
    "lon_max": 119.0,
}


def _fetch_point(lat: float, lon: float, timeout: int = 8) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wind_speed_10m", "wind_direction_10m", "precipitation"],
        "forecast_days": 1,
        "forecast_hours": 24,
    }
    try:
        j = requests.get(url, params=params, timeout=timeout).json()
        h = j.get("hourly", {})
        ws = h.get("wind_speed_10m", [np.nan])
        wd = h.get("wind_direction_10m", [np.nan])
        pr = h.get("precipitation", [np.nan])
        ws_m = float(np.nanmean(ws)) / 3.6
        wd_m = float(np.nanmean(wd))
        pr_m = float(np.nanmean(pr))
        if not np.isfinite(ws_m):
            ws_m = 4.5
        if not np.isfinite(wd_m):
            wd_m = 225
        if not np.isfinite(pr_m):
            pr_m = 0.5
        return {"lat": lat, "lon": lon, "wind_speed": max(ws_m, 0.5), "wind_dir": wd_m % 360, "precip": max(pr_m, 0.0)}
    except Exception:
        return {"lat": lat, "lon": lon, "wind_speed": 4.5, "wind_dir": 225, "precip": 0.5}


def fetch_meteo_grid(n_lat: int = 4, n_lon: int = 6, timeout: int = 8) -> pd.DataFrame:
    """Returns DataFrame with columns lat, lon, wind_speed, wind_dir, precip, u, v."""
    lats = np.linspace(GRID_BOUNDS["lat_min"], GRID_BOUNDS["lat_max"], n_lat)
    lons = np.linspace(GRID_BOUNDS["lon_min"], GRID_BOUNDS["lon_max"], n_lon)
    points = [(float(la), float(lo)) for la in lats for lo in lons]

    results = []
    # Fan out but cap concurrency to avoid hammering free API
    with ThreadPoolExecutor(max_workers=4) as ex:
        fut = {ex.submit(_fetch_point, la, lo, timeout): (la, lo) for la, lo in points}
        for f in as_completed(fut):
            results.append(f.result())

    df = pd.DataFrame(results)
    # Derived u/v (to-direction) for arrow rendering
    th = np.radians(df["wind_dir"].values)
    U = df["wind_speed"].values
    df["u"] = -U * np.sin(th)
    df["v"] = -U * np.cos(th)
    return df
