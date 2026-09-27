"""ERA5 / Open-Meteo gridded meteorology with Planetary Boundary Layer height.

Fetches hourly gridded wind (u/v), precipitation from Open-Meteo.
PBL height is modeled from time of day and surface heating (since Open-Meteo
standard API doesn't provide PBL height directly).
"""

import json
import numpy as np
import pandas as pd
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from src.config import GRID_BOUNDS, DEFAULT_PBL_HEIGHT, MIN_PBL_HEIGHT, MAX_PBL_HEIGHT

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
METEO_GRID_CACHE_FILE = CACHE_DIR / "meteo_grid_cache.json"


def _load_cache() -> dict:
    if METEO_GRID_CACHE_FILE.exists():
        try:
            with open(METEO_GRID_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(METEO_GRID_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def estimate_pbl_height(hour_utc: int, lat: float) -> float:
    """Estimate PBL height from diurnal cycle and latitude.
    
    Simple model: nocturnal minimum ~200-300m, daytime max ~1000-1500m
    Peak around 14:00-15:00 local time.
    """
    # Convert UTC to local time (Malaysia is UTC+8)
    local_hour = (hour_utc + 8) % 24
    
    # Diurnal cycle: minimum at ~6 AM local, maximum at ~2 PM local
    # Using cosine curve
    phase = 2 * np.pi * (local_hour - 14) / 24  # Peak at 14:00
    amplitude = 0.5  # Fraction of day/night range
    
    # Base range depends on latitude (maritime tropical)
    pbl_min = 200.0   # Nocturnal stable boundary layer
    pbl_max = 1500.0  # Daytime convective boundary layer
    
    pbl = pbl_min + (pbl_max - pbl_min) * (0.5 + amplitude * np.cos(phase))
    return np.clip(pbl, MIN_PBL_HEIGHT, MAX_PBL_HEIGHT)


def _fetch_point_era5(lat: float, lon: float, forecast_hours: int = 24) -> dict | None:
    """Fetch hourly meteo from Open-Meteo for a single point."""
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": [
            "wind_speed_10m", "wind_direction_10m", 
            "precipitation"
        ],
        "forecast_days": 1,
        "timezone": "UTC",
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        if not times:
            return None
        
        n = len(times)
        ws = hourly.get("wind_speed_10m", [np.nan] * n)
        wd = hourly.get("wind_direction_10m", [np.nan] * n)
        pr = hourly.get("precipitation", [np.nan] * n)
        
        # Convert wind speed km/h -> m/s
        ws_ms = [float(w) / 3.6 if w is not None else np.nan for w in ws]
        wd_deg = [float(w) if w is not None else np.nan for w in wd]
        pr_mm = [float(w) if w is not None else np.nan for w in pr]
        
        # Estimate PBL height for each hour
        pbl_m = []
        for t_str in times:
            # Parse hour from ISO format "2026-09-27T00:00"
            hour_utc = int(t_str.split("T")[1].split(":")[0])
            pbl_m.append(estimate_pbl_height(hour_utc, lat))
        
        # Compute u/v components (to-direction for arrow rendering)
        u_vals = []
        v_vals = []
        for wd_d, ws_m in zip(wd_deg, ws_ms):
            if np.isfinite(wd_d) and np.isfinite(ws_m):
                th = np.radians(wd_d)
                u_vals.append(-ws_m * np.sin(th))
                v_vals.append(-ws_m * np.cos(th))
            else:
                u_vals.append(np.nan)
                v_vals.append(np.nan)
        
        return {
            "lat": lat,
            "lon": lon,
            "times": times,
            "wind_speed": ws_ms,
            "wind_dir": wd_deg,
            "precipitation": pr_mm,
            "pbl_height": pbl_m,
            "u": u_vals,
            "v": v_vals,
        }
    except Exception:
        return None


def fetch_meteo_grid_era5(
    n_lat: int = 8,
    n_lon: int = 12,
    forecast_hours: int = 24,
    use_cache: bool = True
) -> pd.DataFrame:
    """Fetch gridded hourly meteorology with modeled PBL height.
    
    Returns DataFrame with columns:
        lat, lon, time, wind_speed, wind_dir, precipitation, pbl_height, u, v
    One row per (lat, lon, hour).
    """
    cache_key = f"{n_lat}x{n_lon}x{forecast_hours}h"
    cache = _load_cache() if use_cache else {}
    
    if use_cache and cache_key in cache:
        return pd.DataFrame(cache[cache_key])
    
    lats = np.linspace(GRID_BOUNDS["lat_min"], GRID_BOUNDS["lat_max"], n_lat)
    lons = np.linspace(GRID_BOUNDS["lon_min"], GRID_BOUNDS["lon_max"], n_lon)
    points = [(float(la), float(lo)) for la in lats for lo in lons]
    
    all_rows = []
    
    with ThreadPoolExecutor(max_workers=6) as ex:
        futures = {ex.submit(_fetch_point_era5, la, lo, forecast_hours): (la, lo) 
                   for la, lo in points}
        for f in as_completed(futures):
            result = f.result()
            if result is None:
                continue
            la, lo = result["lat"], result["lon"]
            times = result["times"]
            n = len(times)
            for i in range(n):
                ws = result["wind_speed"][i]
                wd = result["wind_dir"][i]
                pr = result["precipitation"][i]
                pbl = result["pbl_height"][i]
                u = result["u"][i]
                v = result["v"][i]
                
                # Apply defaults/clamps
                if not np.isfinite(ws) or ws < 0.5:
                    ws = 4.5
                if not np.isfinite(wd):
                    wd = 225.0
                if not np.isfinite(pr) or pr < 0:
                    pr = 0.0
                if not np.isfinite(pbl) or pbl < MIN_PBL_HEIGHT:
                    pbl = DEFAULT_PBL_HEIGHT
                if pbl > MAX_PBL_HEIGHT:
                    pbl = MAX_PBL_HEIGHT
                if not np.isfinite(u):
                    th = np.radians(wd)
                    u = -ws * np.sin(th)
                if not np.isfinite(v):
                    th = np.radians(wd)
                    v = -ws * np.cos(th)
                
                all_rows.append({
                    "lat": la,
                    "lon": lo,
                    "time": times[i],
                    "wind_speed": ws,
                    "wind_dir": wd % 360,
                    "precipitation": pr,
                    "pbl_height": pbl,
                    "u": u,
                    "v": v,
                })
    
    df = pd.DataFrame(all_rows)
    
    if use_cache and not df.empty:
        cache[cache_key] = df.to_dict("records")
        _save_cache(cache)
    
    return df


def get_meteo_interpolators(df: pd.DataFrame) -> dict:
    """Create spatial-temporal interpolators for wind, precip, PBL.
    
    Returns dict with callable interpolators:
        u(lat, lon, hour), v(lat, lon, hour), precip(lat, lon, hour), pbl(lat, lon, hour)
    """
    from scipy.interpolate import RegularGridInterpolator
    
    if df.empty:
        return {}
    
    # Extract unique coordinates and times
    lats = np.sort(df["lat"].unique())
    lons = np.sort(df["lon"].unique())
    times = sorted(df["time"].unique())
    
    # Map time to hour index
    time_to_idx = {t: i for i, t in enumerate(times)}
    n_hours = len(times)
    
    # Build 3D arrays (n_lat, n_lon, n_hours)
    shape = (len(lats), len(lons), n_hours)
    u_arr = np.full(shape, np.nan)
    v_arr = np.full(shape, np.nan)
    pr_arr = np.full(shape, np.nan)
    pbl_arr = np.full(shape, np.nan)
    
    for _, row in df.iterrows():
        i = np.where(lats == row["lat"])[0][0]
        j = np.where(lons == row["lon"])[0][0]
        k = time_to_idx[row["time"]]
        u_arr[i, j, k] = row["u"]
        v_arr[i, j, k] = row["v"]
        pr_arr[i, j, k] = row["precipitation"]
        pbl_arr[i, j, k] = row["pbl_height"]
    
    # Fill NaN with nearest valid
    for arr in [u_arr, v_arr, pr_arr, pbl_arr]:
        mask = np.isnan(arr)
        if mask.any():
            from scipy.ndimage import gaussian_filter
            arr[mask] = gaussian_filter(arr, sigma=1)[mask]
    
    # Create interpolators
    points = (lats, lons, np.arange(n_hours))
    
    def make_interp(arr):
        return RegularGridInterpolator(points, arr, bounds_error=False, fill_value=None)
    
    return {
        "u": make_interp(u_arr),
        "v": make_interp(v_arr),
        "precip": make_interp(pr_arr),
        "pbl": make_interp(pbl_arr),
        "times": times,
        "lats": lats,
        "lons": lons,
    }


def get_current_meteo_at_point(
    interpolators: dict,
    lat: float,
    lon: float,
    hour_idx: int = 0
) -> dict:
    """Get interpolated meteo at a specific point and hour index."""
    if not interpolators:
        return {"wind_speed": 4.5, "wind_dir": 225.0, "precipitation": 0.0, "pbl_height": DEFAULT_PBL_HEIGHT}
    
    u = float(interpolators["u"]((lat, lon, hour_idx)))
    v = float(interpolators["v"]((lat, lon, hour_idx)))
    pr = float(interpolators["precip"]((lat, lon, hour_idx)))
    pbl = float(interpolators["pbl"]((lat, lon, hour_idx)))
    
    ws = np.hypot(u, v)
    wd = (270 - np.degrees(np.arctan2(-u, -v))) % 360 if ws > 0 else 225.0
    
    return {
        "wind_speed": max(ws, 0.5),
        "wind_dir": wd,
        "precipitation": max(pr, 0.0),
        "pbl_height": np.clip(pbl, MIN_PBL_HEIGHT, MAX_PBL_HEIGHT),
    }