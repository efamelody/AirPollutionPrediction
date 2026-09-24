import numpy as np
import requests


def fetch_open_meteo_weather(lat: float, lon: float) -> dict:
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": ["wind_speed_10m", "wind_direction_10m", "precipitation"],
        "forecast_days": 1,
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        hourly = res.get("hourly", {})
        ws = hourly.get("wind_speed_10m", [])
        wd = hourly.get("wind_direction_10m", [])
        pr = hourly.get("precipitation", [])
        wind_speed = float(np.nanmean(ws)) / 3.6 if ws else 4.17  # km/h -> m/s
        wind_dir = float(np.nanmean(wd)) if wd else 225.0
        precip = float(np.nanmean(pr)) if pr else 0.5
        if not np.isfinite(wind_speed):
            wind_speed = 4.17
        if not np.isfinite(wind_dir):
            wind_dir = 225.0
        if not np.isfinite(precip):
            precip = 0.5
        return {"wind_speed": max(wind_speed, 0.5), "wind_dir": float(wind_dir) % 360, "precipitation": max(precip, 0.0)}
    except Exception:
        return {"wind_speed": 4.17, "wind_dir": 225.0, "precipitation": 0.5}
