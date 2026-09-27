"""Gridded meteo fetcher - uses ERA5/Open-Meteo for hourly wind, precip, and PBL height.

Provides gridded meteorology for trajectory integration and transport modeling.
Caching via streamlit is handled in app.py.
"""

import numpy as np
import pandas as pd
from src.data.era5 import fetch_meteo_grid_era5, get_meteo_interpolators, get_current_meteo_at_point
from src.config import GRID_BOUNDS


def fetch_meteo_grid(n_lat: int = 8, n_lon: int = 12, forecast_hours: int = 24) -> pd.DataFrame:
    """Returns DataFrame with hourly gridded meteorology including PBL height.
    
    Columns: lat, lon, time, wind_speed, wind_dir, precipitation, pbl_height, u, v
    One row per (lat, lon, hour).
    """
    return fetch_meteo_grid_era5(n_lat=n_lat, n_lon=n_lon, forecast_hours=forecast_hours)


def get_meteo_for_transport(n_lat: int = 8, n_lon: int = 12, forecast_hours: int = 24) -> dict:
    """Get meteorology interpolators for transport modeling."""
    df = fetch_meteo_grid_era5(n_lat=n_lat, n_lon=n_lon, forecast_hours=forecast_hours)
    return get_meteo_interpolators(df)


def get_point_meteo(
    interpolators: dict,
    lat: float,
    lon: float,
    hour_idx: int = 0
) -> dict:
    """Get interpolated meteo at a specific point for dashboard display."""
    return get_current_meteo_at_point(interpolators, lat, lon, hour_idx)


def get_centroid_meteo(interpolators: dict, hour_idx: int = 0) -> dict:
    """Get meteo at Strait of Malacca centroid for summary display."""
    from src.config import METEO_LAT, METEO_LON
    return get_current_meteo_at_point(interpolators, METEO_LAT, METEO_LON, hour_idx)