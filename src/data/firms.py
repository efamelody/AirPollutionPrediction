import pandas as pd
import io
import requests
import streamlit as st
from src.data.landcover import add_biome_column, calculate_Q


def _synthetic_hotspots() -> pd.DataFrame:
    data = {
        "latitude": [0.512, -1.231, -2.415, 1.312, -0.854],
        "longitude": [101.442, 103.812, 111.512, 102.114, 104.215],
        "frp": [145.2, 89.1, 310.5, 62.3, 210.8],
        "confidence": [90, 85, 95, 78, 92],
        "acq_date": ["2023-10-01"] * 5,
    }
    df = pd.DataFrame(data)
    df = add_biome_column(df)
    df["Q_init"] = df.apply(lambda r: calculate_Q(r["frp"], r["biome"]), axis=1)
    return df


# Indonesia bounding box (Sumatra + Kalimantan fire regions)
INDONESIA_BBOX = "95,-5,120,7"  # min_lon,min_lat,max_lon,max_lat
DEFAULT_SOURCE = "VIIRS_SNPP_NRT"
REQUEST_TIMEOUT = 30  # seconds


def fetch_firms_hotspots(map_key: str, country_code: str = "IDN", days: int = 1) -> pd.DataFrame:
    """Fetch FIRMS hotspots for Indonesia using area endpoint.
    
    Country endpoint is deprecated; uses area bounding box instead.
    """
    # DEMO_KEY or empty always returns synthetic without recursion
    if not map_key or map_key.strip() == "" or map_key == "DEMO_KEY":
        if map_key == "DEMO_KEY":
            st.info("Using demo fire data. Enter your NASA FIRMS API key in the sidebar to see live fires.")
        return _synthetic_hotspots()

    # Validate key format (basic check)
    if len(map_key.strip()) < 20:
        st.warning("API key appears too short. Please check your NASA FIRMS MAP_KEY.")
        return _synthetic_hotspots()

    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{map_key}/{DEFAULT_SOURCE}/{INDONESIA_BBOX}/{days}"
    try:
        # Use requests for better timeout handling
        headers = {"User-Agent": "Mozilla/5.0 (compatible; AirPollutionPrediction/1.0)"}
        
        # Retry once on failure
        for attempt in range(2):
            try:
                resp = requests.get(url, timeout=REQUEST_TIMEOUT, headers=headers)
                resp.raise_for_status()
                break
            except requests.exceptions.RequestException as e:
                if attempt == 0:
                    continue  # retry once
                raise
        
        # Parse CSV from text
        df = pd.read_csv(io.StringIO(resp.text))
        
        # Normalize columns
        df.columns = [c.lower() for c in df.columns]
        
        # Filter confidence >= 70 (numeric) or 'h'/'n'
        if "confidence" in df.columns:
            conf = pd.to_numeric(df["confidence"], errors="coerce")
            # 'h'=high, 'n'=nominal, 'l'=low -> treat h/n as >=70
            mask = conf.fillna(80) >= 70
            df = df[mask]
        
        if df.empty:
            st.warning("FIRMS returned no hotspots for the window. Using synthetic fallback.")
            return _synthetic_hotspots()
        
        # Keep only needed columns
        keep_cols = ["latitude", "longitude", "frp", "confidence", "acq_date"]
        df = df[keep_cols].copy()
        
        # Limit to top N fires by FRP (model can't handle thousands)
        max_fires = 20
        if len(df) > max_fires:
            df = df.nlargest(max_fires, "frp").copy()
        
        # Add biome classification and calibrated Q
        # Use fast_mode (heuristic only) for large datasets to avoid WMS timeout
        use_fast = len(df) > 100
        df = add_biome_column(df, fast_mode=use_fast)
        df["Q_init"] = df.apply(lambda r: calculate_Q(r["frp"], r["biome"]), axis=1)
        return df
    except Exception as e:
        import traceback
        st.warning(f"Unable to reach NASA FIRMS API ({type(e).__name__}: {e}). Loading synthetic fallback data.")
        st.caption(f"Debug: URL was {url}")
        return _synthetic_hotspots()