"""WAQI (World Air Quality Index) API client for ground-station PM2.5 data."""

import json
import time
import urllib.request
from pathlib import Path
from typing import Optional

CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CACHE_DIR.mkdir(exist_ok=True)
WAQI_CACHE_FILE = CACHE_DIR / "waqi_cache.json"

# Station UIDs for Malaysia cities (pre-mapped from search)
STATION_UIDS = {
    "Kuala Lumpur": 5780,
    "Johor Bahru": 2578,
    "Kuching": 2610,
    "Ipoh": 5777,
    "Kota Bharu": 9556,
}

# Cache TTL in seconds (5 minutes)
CACHE_TTL = 300


def _load_cache() -> dict:
    if WAQI_CACHE_FILE.exists():
        try:
            with open(WAQI_CACHE_FILE, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_cache(cache: dict):
    try:
        with open(WAQI_CACHE_FILE, "w") as f:
            json.dump(cache, f)
    except Exception:
        pass


def _fetch_waqi(station_uid: int, token: str) -> dict | None:
    """Fetch full WAQI feed for a station UID."""
    url = f"https://api.waqi.info/feed/@{station_uid}/?token={token}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data.get("status") == "ok":
            return data["data"]
    except Exception:
        pass
    return None


def fetch_waqi_pm25(station_uid: int, token: str, use_cache: bool = True) -> Optional[float]:
    """Fetch current PM2.5 (µg/m³) for a station UID."""
    cache_key = f"pm25_{station_uid}"
    cache = _load_cache() if use_cache else {}
    
    if use_cache and cache_key in cache:
        entry = cache[cache_key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            return entry["value"]
    
    data = _fetch_waqi(station_uid, token)
    if data and "iaqi" in data and "pm25" in data["iaqi"]:
        pm25 = data["iaqi"]["pm25"].get("v")
        if pm25 is not None:
            value = float(pm25)
            if use_cache:
                cache[cache_key] = {"value": value, "timestamp": time.time()}
                _save_cache(cache)
            return value
    return None


def fetch_waqi_full(station_uid: int, token: str, use_cache: bool = True) -> dict | None:
    """Fetch full WAQI data including all pollutants, forecast, metadata."""
    cache_key = f"full_{station_uid}"
    cache = _load_cache() if use_cache else {}
    
    if use_cache and cache_key in cache:
        entry = cache[cache_key]
        if time.time() - entry["timestamp"] < CACHE_TTL:
            return entry["value"]
    
    data = _fetch_waqi(station_uid, token)
    if data:
        if use_cache:
            cache[cache_key] = {"value": data, "timestamp": time.time()}
            _save_cache(cache)
        return data
    return None


def search_stations(keyword: str, token: str) -> list[dict]:
    """Search WAQI stations by keyword."""
    url = f"https://api.waqi.info/search/?token={token}&keyword={keyword.replace(' ', '%20')}"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.loads(resp.read().decode())
        if data.get("status") == "ok":
            return data["data"]
    except Exception:
        pass
    return []


def get_station_uid(city_name: str, token: str) -> int | None:
    """Get station UID for a city (uses pre-mapped or searches)."""
    if city_name in STATION_UIDS:
        return STATION_UIDS[city_name]
    
    results = search_stations(city_name, token)
    for r in results:
        if "malaysia" in r["station"]["url"].lower() or r["station"].get("country") == "MY":
            return r["uid"]
    return None


def fetch_all_cities_waqi(token: str) -> dict[str, float]:
    """Fetch PM2.5 for all 5 Malaysia cities."""
    results = {}
    for city, uid in STATION_UIDS.items():
        pm25 = fetch_waqi_pm25(uid, token)
        if pm25 is not None:
            results[city] = pm25
    return results


def fetch_waqi_forecast(station_uid: int, token: str) -> dict | None:
    """Fetch WAQI forecast data (daily PM2.5 averages)."""
    data = fetch_waqi_full(station_uid, token)
    if data and "forecast" in data and "daily" in data["forecast"]:
        return data["forecast"]["daily"].get("pm25", [])
    return None