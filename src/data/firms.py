import pandas as pd
import streamlit as st


def _synthetic_hotspots() -> pd.DataFrame:
    data = {
        "latitude": [0.512, -1.231, -2.415, 1.312, -0.854],
        "longitude": [101.442, 103.812, 111.512, 102.114, 104.215],
        "frp": [145.2, 89.1, 310.5, 62.3, 210.8],
        "confidence": [90, 85, 95, 78, 92],
        "acq_date": ["2023-10-01"] * 5,
    }
    return pd.DataFrame(data)


def fetch_firms_hotspots(map_key: str, country_code: str = "IDN", days: int = 1) -> pd.DataFrame:
    # DEMO_KEY or empty always returns synthetic without recursion
    if not map_key or map_key.strip() == "" or map_key == "DEMO_KEY":
        return _synthetic_hotspots()

    url = f"https://firms.modaps.eosdis.nasa.gov/api/country/csv/{map_key}/VIIRS_SNPP_NRT/{country_code}/{days}"
    try:
        df = pd.read_csv(url)
        # FIRMS CSV columns may vary in case
        df.columns = [c.lower() for c in df.columns]
        if "confidence" in df.columns:
            # confidence can be 'h','n','l' or numeric; filter best effort
            try:
                df = df[pd.to_numeric(df["confidence"], errors="coerce").fillna(80) >= 70]
            except Exception:
                pass
        if df.empty:
            st.warning("FIRMS returned no hotspots for the window. Using synthetic fallback.")
            return _synthetic_hotspots()
        return df
    except Exception as e:
        st.warning(f"Unable to reach NASA FIRMS API ({e}). Loading synthetic fallback data.")
        return _synthetic_hotspots()
