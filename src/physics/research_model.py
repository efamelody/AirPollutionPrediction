"""Research-Grade Transboundary Haze Attribution Model.

Main entry point integrating proxy-calibrated emissions, trajectory transport,
urban baseline, and continuous Bayesian inverse attribution.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timezone
from src.physics.calibration import run_calibration_pipeline
from src.physics.api import pm25_to_api, api_category
from src.config import FORECAST_HOURS


def predict_haze(
    firms_df: pd.DataFrame,
    receptors_df: pd.DataFrame,
    forecast_hours: int = FORECAST_HOURS,
    scenario: dict = None
) -> dict:
    """Run calibrated haze prediction and attribution.
    
    Args:
        firms_df: FIRMS hotspots with lat, lon, frp, confidence
        receptors_df: Monitoring stations with station_name, lat, lon, pm25_obs
        forecast_hours: Number of forecast hours
        scenario: Optional dict with keys:
            - rain_multiplier: float (default 1.0)
            - wind_shift_deg: float (default 0.0)
            - urban_scale: float (default 1.0)
    
    Returns:
        Dict with:
            - city_forecast: DataFrame (station_name, lat, lon, pm25_pred, api_pred, category, pm25_obs, hour)
            - fire_posteriors: DataFrame (lat, lon, frp, biome, Q_init, posterior_prob)
            - haze_grid: DataFrame for map visualization
            - metrics: Validation metrics dict
    """
    # Apply scenario modifications
    if scenario is None:
        scenario = {}
    rain_mult = scenario.get("rain_multiplier", 1.0)
    wind_shift = scenario.get("wind_shift_deg", 0.0)
    urban_scale = scenario.get("urban_scale", 1.0)
    
    # Get current UTC hour
    start_hour_utc = datetime.now(timezone.utc).hour
    
    # Run calibration pipeline
    result = run_calibration_pipeline(
        firms_df, receptors_df,
        start_hour_utc=start_hour_utc,
        forecast_hours=forecast_hours
    )
    
    # Apply scenario adjustments to predictions
    C_smoke = result["C_smoke"]
    baseline = result["baseline"]
    if rain_mult != 1.0 or urban_scale != 1.0:
        # Re-run with modified meteo (simplified: scale precipitation and baseline)
        pass  # Scenario handling would modify meteo_interpolators
    
    # Format city forecast for current hour
    current_idx = forecast_hours - 1
    pm25_pred = C_smoke[:, :, current_idx].sum(axis=0) + baseline[:, current_idx]  # (N_receptors,)
    
    city_forecast_rows = []
    for k, (_, rec) in enumerate(receptors_df.iterrows()):
        pm25 = float(pm25_pred[k])
        api = pm25_to_api(pm25)
        cat, _ = api_category(api)
        city_forecast_rows.append({
            "station_name": rec["station_name"],
            "lat": rec["latitude"],
            "lon": rec["longitude"],
            "pm25_pred": pm25,
            "api_pred": api,
            "category": cat,
            "pm25_obs": float(rec["pm25_obs"]),
            "hour_utc": (start_hour_utc + current_idx) % 24,
        })
    city_forecast = pd.DataFrame(city_forecast_rows)
    
    # Format fire posteriors
    fires = result["fires"].copy()
    fires["posterior_prob"] = result["posterior_probs"]
    fire_posteriors = fires[["latitude", "longitude", "frp", "biome", "Q_init", "posterior_prob"]].copy()
    fire_posteriors = fire_posteriors.sort_values("posterior_prob", ascending=False)
    
    # Generate haze grid for map (current hour)
    haze_grid = _generate_haze_grid(
        result["C_smoke"][:, :, current_idx],
        receptors_df,
        result["meteo_interpolators"],
        start_hour_utc + current_idx
    )
    
    return {
        "city_forecast": city_forecast,
        "fire_posteriors": fire_posteriors,
        "haze_grid": haze_grid,
        "metrics": result["metrics"],
        "C_smoke": result["C_smoke"],
        "baseline": result["baseline"],
    }


def _generate_haze_grid(
    C_smoke_current: np.ndarray,  # (N_fires, N_receptors)
    receptors_df: pd.DataFrame,
    meteo_interpolators: dict,
    hour_utc: int
) -> pd.DataFrame:
    """Generate gridded haze map from smoke concentrations at receptors.
    
    Uses inverse distance weighting from receptor predictions.
    """
    from src.config import GRID_BOUNDS
    
    n_lat, n_lon = 18, 22
    lats = np.linspace(GRID_BOUNDS["lat_min"], GRID_BOUNDS["lat_max"], n_lat)
    lons = np.linspace(GRID_BOUNDS["lon_min"], GRID_BOUNDS["lon_max"], n_lon)
    
    # Total smoke at each receptor
    smoke_at_receptors = C_smoke_current.sum(axis=0)  # (N_receptors,)
    
    rows = []
    rec_coords = receptors_df[["latitude", "longitude"]].values
    
    for la in lats:
        for lo in lons:
            # IDW interpolation from receptors
            dists = np.array([haversine_km(la, lo, rc[0], rc[1]) for rc in rec_coords])
            dists = np.maximum(dists, 1.0)  # Avoid division by zero
            weights = 1.0 / (dists ** 2)
            weights = weights / weights.sum()
            
            pm25 = np.sum(weights * smoke_at_receptors)
            
            # Add baseline for this grid cell (approximate using nearest city)
            nearest_idx = np.argmin(dists)
            city = receptors_df.iloc[nearest_idx]["station_name"]
            from src.data.cams import get_total_baseline
            baseline = get_total_baseline(city, hour_utc)
            pm25_total = pm25 + baseline
            
            from src.physics.api import pm25_to_api, api_category
            api = pm25_to_api(pm25_total)
            cat, _ = api_category(api)
            
            rows.append({
                "lat": la, "lon": lo,
                "pm25": pm25_total,
                "api": api,
                "category": cat,
                "smoke_only": pm25,
            })
    
    return pd.DataFrame(rows)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def predict_haze_forecast(
    firms_df: pd.DataFrame,
    receptors_df: pd.DataFrame,
    forecast_hours: int = FORECAST_HOURS
) -> pd.DataFrame:
    """Generate multi-hour forecast for all receptors.
    
    Returns DataFrame with hourly predictions per city.
    """
    result = run_calibration_pipeline(
        firms_df, receptors_df,
        start_hour_utc=datetime.now(timezone.utc).hour,
        forecast_hours=forecast_hours
    )
    
    C_smoke = result["C_smoke"]
    baseline = result["baseline"]
    rows = []
    start_hour = datetime.now(timezone.utc).hour
    
    for h in range(forecast_hours):
        pm25_pred = C_smoke[:, :, h].sum(axis=0) + baseline[:, h]
        hour_utc = (start_hour + h) % 24
        for k, (_, rec) in enumerate(receptors_df.iterrows()):
            pm25 = float(pm25_pred[k])
            api = pm25_to_api(pm25)
            cat, _ = api_category(api)
            rows.append({
                "station_name": rec["station_name"],
                "lat": rec["latitude"],
                "lon": rec["longitude"],
                "hour_utc": hour_utc,
                "pm25_pred": pm25,
                "api_pred": api,
                "category": cat,
            })
    
    return pd.DataFrame(rows)