"""CAMS EAC4 reanalysis data access via Open-Meteo Air Quality API.

Provides urban baseline extraction from non-haze months and
haze episode validation data.
"""

import json
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from src.config import URBAN_BASELINE_PARAMS, REGIONAL_BACKGROUND_UGM3

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
BASELINE_CACHE_FILE = CACHE_DIR / "urban_baseline_cache.json"

# Receptor locations for baseline extraction
RECEPTORS = {
    "Kuala Lumpur": {"lat": 3.1390, "lon": 101.6869},
    "Johor Bahru": {"lat": 1.4927, "lon": 103.7414},
    "Kuching": {"lat": 1.5533, "lon": 110.3592},
    "Ipoh": {"lat": 4.5975, "lon": 101.0901},
    "Kota Bharu": {"lat": 6.1254, "lon": 102.2381},
}


def _load_cache() -> dict:
    if BASELINE_CACHE_FILE.exists():
        try:
            with open(BASELINE_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(BASELINE_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def fetch_cams_pm25(lat: float, lon: float, start_date: str, end_date: str) -> pd.Series | None:
    """Fetch hourly PM2.5 from CAMS via Open-Meteo for a date range.
    
    Args:
        lat, lon: Coordinates
        start_date: YYYY-MM-DD
        end_date: YYYY-MM-DD
    
    Returns:
        Series indexed by hourly datetime with PM2.5 values (µg/m³)
    """
    url = (
        f"https://air-quality-api.open-meteo.com/v1/air-quality?"
        f"latitude={lat}&longitude={lon}"
        f"&hourly=pm2_5"
        f"&start_date={start_date}&end_date={end_date}"
        f"&timezone=UTC"
    )
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode())
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        values = hourly.get("pm2_5", [])
        if not times:
            return None
        idx = pd.to_datetime(times, utc=True)
        return pd.Series(values, index=idx, name="pm25")
    except Exception:
        return None


def extract_urban_baseline_from_cams(
    start_date: str = "2023-11-01",
    end_date: str = "2024-04-30"
) -> dict:
    """Extract diurnal urban baseline curves from CAMS non-haze months.
    
    Uses Northeast Monsoon months (Nov-Apr) when Indonesian fires are minimal
    and winds blow away from Malaysia.
    
    Returns:
        Dict[city] -> Dict[hour_utc] -> baseline_pm25 (µg/m³)
    """
    cache_key = f"{start_date}_{end_date}"
    cache = _load_cache()
    if cache_key in cache:
        return cache[cache_key]
    
    baselines = {}
    for city, coords in RECEPTORS.items():
        series = fetch_cams_pm25(coords["lat"], coords["lon"], start_date, end_date)
        if series is not None and len(series) > 100:
            # Group by hour of day (UTC) and compute median
            series.index = series.index.tz_convert("UTC")
            hourly_median = series.groupby(series.index.hour).median()
            # Ensure all 24 hours present
            hourly_median = hourly_median.reindex(range(24), method="nearest")
            baselines[city] = hourly_median.to_dict()
        else:
            # Fallback to parametric
            baselines[city] = generate_parametric_baseline(city)
    
    cache[cache_key] = baselines
    _save_cache(cache)
    return baselines


def generate_parametric_baseline(city: str) -> dict:
    """Generate parametric diurnal baseline from config parameters."""
    params = URBAN_BASELINE_PARAMS.get(city, URBAN_BASELINE_PARAMS["Kuala Lumpur"])
    hours = np.arange(24)
    morning = params["morning_amp"] * np.exp(-0.5 * ((hours - params["morning_peak"]) / params["morning_width"])**2)
    evening = params["evening_amp"] * np.exp(-0.5 * ((hours - params["evening_peak"]) / params["evening_width"])**2)
    baseline = params["base"] + morning + evening
    return dict(zip(hours, baseline))


def get_urban_baseline(city: str, hour_utc: int) -> float:
    """Get urban baseline PM2.5 for a city at a specific UTC hour.
    
    Tries cached CAMS baseline first, falls back to parametric.
    """
    cache = _load_cache()
    # Use most recent cached baseline
    for key in reversed(sorted(cache.keys())):
        if city in cache[key]:
            return cache[key][city].get(hour_utc, REGIONAL_BACKGROUND_UGM3)
    
    # Fallback
    return generate_parametric_baseline(city).get(hour_utc, REGIONAL_BACKGROUND_UGM3)


def get_total_baseline(city: str, hour_utc: int) -> float:
    """Total baseline = regional background + urban diurnal."""
    return REGIONAL_BACKGROUND_UGM3 + get_urban_baseline(city, hour_utc)


def fetch_haze_episode_cams(
    city: str,
    episode_start: str,
    episode_end: str
) -> pd.Series | None:
    """Fetch CAMS PM2.5 for a haze episode (for validation).
    
    Example: Sep 2019 haze episode
    """
    coords = RECEPTORS.get(city)
    if not coords:
        return None
    return fetch_cams_pm25(coords["lat"], coords["lon"], episode_start, episode_end)


# Pre-computed baseline for immediate use (from parametric)
PARAMETRIC_BASELINES = {city: generate_parametric_baseline(city) for city in RECEPTORS}