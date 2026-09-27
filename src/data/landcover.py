"""ESA WorldCover 10m land cover biome lookup for fire hotspots.

Uses ESA WorldCover WMS service for on-the-fly biome classification.
Caches results locally to avoid repeated queries.
"""

import json
import os
from pathlib import Path
import requests
import pandas as pd
from src.config import ESA_CLASS_TO_BIOME

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
BIOME_CACHE_FILE = CACHE_DIR / "biome_lookup_cache.json"

# ESA WorldCover WMS endpoint
WMS_URL = "https://services.terrascope.be/wms/v2"
WMS_LAYER = "WORLDCOVER_2021_MAP"

# Fallback: simple heuristic for Sumatra/Kalimantan peatlands
PEAT_BOUNDS = {
    "lat_min": -4.0,
    "lat_max": 1.0,
    "lon_min": 100.0,
    "lon_max": 112.0,
}


def _load_cache() -> dict:
    if BIOME_CACHE_FILE.exists():
        try:
            with open(BIOME_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(BIOME_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _query_wms_biome(lat: float, lon: float) -> str | None:
    """Query ESA WorldCover WMS for land cover class at coordinate."""
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetFeatureInfo",
        "LAYERS": WMS_LAYER,
        "QUERY_LAYERS": WMS_LAYER,
        "CRS": "EPSG:4326",
        "BBOX": f"{lon-0.0001},{lat-0.0001},{lon+0.0001},{lat+0.0001}",
        "WIDTH": 10,
        "HEIGHT": 10,
        "INFO_FORMAT": "application/json",
        "I": 5,
        "J": 5,
    }
    try:
        resp = requests.get(WMS_URL, params=params, timeout=10)
        if resp.status_code == 200:
            data = resp.json()
            features = data.get("features", [])
            if features:
                props = features[0].get("properties", {})
                # ESA WorldCover returns 'classification' or 'band1' property
                class_val = props.get("classification") or props.get("band1")
                if class_val is not None:
                    class_int = int(class_val)
                    return ESA_CLASS_TO_BIOME.get(class_int, "other")
    except Exception:
        pass
    return None


def _heuristic_biome(lat: float, lon: float) -> str:
    """Fallback heuristic: peatland in known peat regions, else tropical_forest."""
    if (PEAT_BOUNDS["lat_min"] <= lat <= PEAT_BOUNDS["lat_max"] and
        PEAT_BOUNDS["lon_min"] <= lon <= PEAT_BOUNDS["lon_max"]):
        return "peatland"
    return "tropical_forest"


def get_biome(lat: float, lon: float, use_cache: bool = True) -> str:
    """Get biome type for a coordinate (lat, lon in degrees)."""
    key = f"{lat:.4f},{lon:.4f}"
    cache = _load_cache() if use_cache else {}
    
    if key in cache:
        return cache[key]
    
    # Try WMS query
    biome = _query_wms_biome(lat, lon)
    
    if biome is None:
        biome = _heuristic_biome(lat, lon)
    
    if use_cache:
        cache[key] = biome
        _save_cache(cache)
    
    return biome


def add_biome_column(df: pd.DataFrame, lat_col: str = "latitude", lon_col: str = "longitude") -> pd.DataFrame:
    """Add 'biome' column to FIRMS hotspot DataFrame."""
    df = df.copy()
    df["biome"] = df.apply(lambda r: get_biome(r[lat_col], r[lon_col]), axis=1)
    return df


def get_emission_params(biome: str) -> dict:
    """Return emission parameters for a biome."""
    from src.config import BIOME_EMISSION_PARAMS
    return BIOME_EMISSION_PARAMS.get(biome, BIOME_EMISSION_PARAMS["other"])


def calculate_Q(frp_mw: float, biome: str, cloud_cover: float = None) -> float:
    """Calculate PM2.5 emission rate Q (kg/s) from FRP and biome."""
    from src.config import BIOME_EMISSION_PARAMS, CLOUD_COVER_FRACTION
    
    if cloud_cover is None:
        cloud_cover = CLOUD_COVER_FRACTION
    
    params = BIOME_EMISSION_PARAMS.get(biome, BIOME_EMISSION_PARAMS["other"])
    
    # Cloud attenuation correction
    frp_adjusted = frp_mw / max(1.0 - cloud_cover, 0.1)
    
    # Q = Cf * FRP_adj * EF_pm25 * 1e-3 (kg/s)
    Q_kg_per_sec = params["Cf"] * frp_adjusted * params["EF_pm25"] * 1e-3
    return Q_kg_per_sec