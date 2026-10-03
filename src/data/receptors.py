import pandas as pd
import urllib.request
import json
from datetime import datetime, timezone
from src.data.cams import get_total_baseline, PARAMETRIC_BASELINES
from src.data.waqi import fetch_waqi_pm25, STATION_UIDS


def _fetch_pm25_open_meteo(lat: float, lon: float, target_hour_utc: int | None = None) -> float | None:
    """Fetch PM2.5 from Open-Meteo Air Quality API (CAMS) for a specific UTC hour.
    
    Args:
        lat, lon: Coordinates
        target_hour_utc: Specific UTC hour to fetch (0-23). If None, returns latest available.
    
    Returns:
        PM2.5 in µg/m³ or None if unavailable.
    """
    try:
        # Fetch today's data (forecast_days=1 gives current day + next)
        url = f"https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5&timezone=UTC&forecast_days=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        data = json.loads(urllib.request.urlopen(req, timeout=8).read().decode())
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        vals = hourly.get("pm2_5", [])
        if not times or not vals:
            return None
        
        if target_hour_utc is not None:
            # Find the index matching target hour
            for i, t_str in enumerate(times):
                # Format: "2026-09-30T14:00"
                hour = int(t_str.split("T")[1].split(":")[0])
                if hour == target_hour_utc:
                    v = vals[i]
                    return float(v) if v is not None else None
            return None
        else:
            # Latest non-null (original behavior)
            for v in reversed(vals):
                if v is not None:
                    return float(v)
            return float(vals[-1]) if vals[-1] is not None else None
    except Exception:
        return None


def fetch_malaysia_receptors(live: bool = True, target_hour_utc: int | None = None) -> pd.DataFrame:
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
        v = _fetch_pm25_open_meteo(float(r["latitude"]), float(r["longitude"]), target_hour_utc)
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


def fetch_waqi_receptors(token: str, target_hour_utc: int | None = None) -> pd.DataFrame:
    """Fetch PM2.5 from WAQI ground stations for Malaysia cities.
    
    Args:
        token: WAQI API token
        target_hour_utc: Not used for WAQI (returns latest measurement)
    
    Returns:
        DataFrame with pm25_obs from WAQI ground stations
    """
    stations = {
        "station_name": ["Kuala Lumpur", "Johor Bahru", "Kuching", "Ipoh", "Kota Bharu"],
        "latitude": [3.1390, 1.4927, 1.5533, 4.5975, 6.1254],
        "longitude": [101.6869, 103.7414, 110.3592, 101.0901, 102.2381],
        "pm25_obs": [85.4, 62.1, 112.0, 45.8, 38.2],  # fallback
    }
    df = pd.DataFrame(stations)
    df["baseline_params"] = df["station_name"].map(PARAMETRIC_BASELINES)
    
    live_vals = []
    success = 0
    for _, r in df.iterrows():
        uid = STATION_UIDS.get(r["station_name"])
        if uid:
            v = fetch_waqi_pm25(uid, token)
            if v is not None and 2 < v < 500:
                live_vals.append(v)
                success += 1
            else:
                live_vals.append(float(r["pm25_obs"]))
        else:
            live_vals.append(float(r["pm25_obs"]))
    
    if success >= 3:
        df["pm25_obs"] = live_vals
        df["pm25_source"] = "WAQI ground station"
    else:
        df["pm25_source"] = "mock demo (WAQI fetch failed)"
    
    return df


def fetch_combined_receptors(
    cams_live: bool = True,
    waqi_token: str | None = None,
    target_hour_utc: int | None = None
) -> pd.DataFrame:
    """Fetch receptors with both CAMS and WAQI observations."""
    # Start with CAMS
    df = fetch_malaysia_receptors(live=cams_live, target_hour_utc=target_hour_utc)
    df["pm25_cams"] = df["pm25_obs"]
    df["pm25_cams_source"] = df.get("pm25_source", "CAMS")
    
    # Add WAQI if token provided
    if waqi_token:
        waqi_df = fetch_waqi_receptors(waqi_token)
        # Merge WAQI values
        waqi_map = dict(zip(waqi_df["station_name"], waqi_df["pm25_obs"]))
        df["pm25_waqi"] = df["station_name"].map(waqi_map)
        df["pm25_waqi_source"] = "WAQI ground station"
    
    return df