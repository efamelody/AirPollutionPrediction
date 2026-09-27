import pandas as pd
import urllib.request
import json
from src.data.cams import get_total_baseline, PARAMETRIC_BASELINES


def _fetch_pm25_open_meteo(lat: float, lon: float) -> float | None:
    """Fetch current PM2.5 from Open-Meteo Air Quality API (CAMS). Returns µg/m³ or None."""
    try:
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&timezone=auto&forecast_days=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=8).read().decode())
        hourly = data.get("hourly", {})
        vals = hourly.get("pm2_5", [])
        if not vals:
            return None
        # take latest non-null
        for v in reversed(vals):
            if v is not None:
                return float(v)
        return float(vals[-1]) if vals[-1] is not None else None
    except Exception:
        return None


def fetch_malaysia_receptors(live: bool = True) -> pd.DataFrame:
    stations = {
        "station_name": ["Kuala Lumpur", "Johor Bahru", "Kuching", "Ipoh", "Kota Bharu"],
        "latitude": [3.1390, 1.4927, 1.5533, 4.5975, 6.1254],
        "longitude": [101.6869, 103.7414, 110.3592, 101.0901, 102.2381],
        # Mock haze-event values — used only if live fetch fails.
        "pm25_obs": [85.4, 62.1, 112.0, 45.8, 38.2],
    }
    df = pd.DataFrame(stations)
    
    # Add urban baseline parameters for each city
    df["baseline_params"] = df["station_name"].map(PARAMETRIC_BASELINES)
    
    if not live:
        return df
    
    # Try to replace with live air-quality where available
    live_vals = []
    success = 0
    for _, r in df.iterrows():
        v = _fetch_pm25_open_meteo(float(r["latitude"]), float(r["longitude"]))
        if v is not None and 2 < v < 300:
            live_vals.append(v)
            success += 1
        else:
            live_vals.append(float(r["pm25_obs"]))
    
    if success >= 3:
        df["pm25_obs"] = live_vals
        df["pm25_source"] = "live Open-Meteo CAMS"
    else:
        df["pm25_source"] = "mock demo (live fetch failed)"
    
    return df


def get_urban_baseline_for_city(city: str, hour_utc: int) -> float:
    """Get urban baseline PM2.5 for a city at UTC hour."""
    return get_total_baseline(city, hour_utc)