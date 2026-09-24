import numpy as np
import pandas as pd
import streamlit as st

from src.config import METEO_LAT, METEO_LON, SIGMA_OBS_UGM3
from src.data.firms import fetch_firms_hotspots
from src.data.meteo import fetch_open_meteo_weather
from src.data.receptors import fetch_malaysia_receptors
from src.physics.dispersion import evaluate_gaussian_plume
from src.physics.inversion import run_bayesian_inversion
from src.viz.maps import build_deck

# ------------------------------------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="Transboundary Haze Bayesian Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("Transboundary Haze Prediction & Bayesian Source Identification")
st.markdown(
    "Operational atmospheric dispersion and inverse Bayesian modeling for regional haze transport "
    "between Indonesia and Malaysia. Gaussian plume with wet-deposition scavenging `exp(-Λ·x/U)` coupled to discrete Bayesian inversion."
)
st.caption(
    "DISCLAIMER: Prototype demonstrator (v0.1). Gaussian plume is a surrogate valid ~1-20km; transboundary 300-800km transport ideally requires HYSPLIT/WRF-Chem. `Q=0.02·FRP` is uncalibrated."
)

# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------
st.sidebar.header("Control Panel & Parameters")

# Prefer Streamlit secrets for Cloud deploy; fallback to sidebar input
default_key = "DEMO_KEY"
try:
    if "FIRMS_MAP_KEY" in st.secrets:
        default_key = st.secrets["FIRMS_MAP_KEY"]
except Exception:
    pass

firms_api_key = st.sidebar.text_input("NASA FIRMS API Key", value=default_key, type="password", help="Obtain free MAP_KEY at firms.modaps.eosdis.nasa.gov. Leave DEMO_KEY for synthetic data.")
selected_days = st.sidebar.slider("Fire Acquisition Window (Days)", 1, 5, 1)
sigma_obs = st.sidebar.slider("Observation error σ_obs (µg/m³)", 5.0, 30.0, float(SIGMA_OBS_UGM3), help="Covariance R diagonal. Higher = more tolerant likelihood.")

st.sidebar.subheader("Atmospheric Manual Overrides")
override_meteo = st.sidebar.checkbox("Override Meteorological API")

if override_meteo:
    manual_wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.5, 20.0, 5.0)
    manual_wind_dir = st.sidebar.slider("Wind Direction (°)", 0, 360, 225, help="Meteorological FROM direction: 0=N,90=E,180=S,270=W")
    manual_precip = st.sidebar.slider("Precipitation Rate (mm/h)", 0.0, 20.0, 1.0)
    meteo_data = {"wind_speed": manual_wind_speed, "wind_dir": manual_wind_dir, "precipitation": manual_precip}
else:
    with st.spinner("Fetching Open-Meteo wind/precip at Strait of Malacca..."):
        meteo_data = fetch_open_meteo_weather(METEO_LAT, METEO_LON)

st.sidebar.markdown("---")
st.sidebar.metric("Wind Speed", f"{meteo_data['wind_speed']:.2f} m/s")
st.sidebar.metric("Wind Direction", f"{meteo_data['wind_dir']:.1f}°")
st.sidebar.metric("Rain Intensity", f"{meteo_data['precipitation']:.2f} mm/h")
st.sidebar.caption("Source: Open-Meteo ECMWF IFS/GFS, 10m wind. Scavenging Λ = 1e-4·P^0.8")

# ------------------------------------------------------------------------------
# DATA INGESTION (cached via underlying functions where possible)
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_hotspots(key: str, days: int) -> pd.DataFrame:
    return fetch_firms_hotspots(key, "IDN", days)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_meteo(lat: float, lon: float, override: bool, ws: float, wd: float, pr: float) -> dict:
    if override:
        return {"wind_speed": ws, "wind_dir": wd, "precipitation": pr}
    return fetch_open_meteo_weather(lat, lon)

# Recompute meteo if override path already handled, else use cached
if not override_meteo:
    meteo_data = cached_meteo(METEO_LAT, METEO_LON, False, 0, 0, 0)
else:
    meteo_data = cached_meteo(METEO_LAT, METEO_LON, True, meteo_data["wind_speed"], meteo_data["wind_dir"], meteo_data["precipitation"])

hotspots_raw = cached_hotspots(firms_api_key, selected_days)
receptors_df = fetch_malaysia_receptors()

# ------------------------------------------------------------------------------
# BAYESIAN INVERSION
# ------------------------------------------------------------------------------
with st.spinner("Running Bayesian source inversion..."):
    hotspots_analyzed = run_bayesian_inversion(hotspots_raw, receptors_df, meteo_data, sigma_obs=sigma_obs)

# Also compute predicted vs observed for diagnostics
def compute_rmse(df_hotspots: pd.DataFrame, df_receptors: pd.DataFrame, meteo: dict) -> float:
    # Use MAP hotspot
    if df_hotspots.empty or df_hotspots["posterior_prob"].sum() == 0:
        return float("nan")
    map_idx = df_hotspots["posterior_prob"].idxmax()
    spot = df_hotspots.loc[map_idx]
    Q_est = float(spot["frp"] * 0.02)
    preds = []
    for _, rec in df_receptors.iterrows():
        c = evaluate_gaussian_plume(float(spot["latitude"]), float(spot["longitude"]), float(rec["latitude"]), float(rec["longitude"]), Q_est, float(meteo["wind_speed"]), float(meteo["wind_dir"]), float(meteo["precipitation"]))
        preds.append(c)
    obs = df_receptors["pm25_obs"].values
    return float(np.sqrt(np.mean((obs - np.array(preds)) ** 2)))

rmse_val = compute_rmse(hotspots_analyzed, receptors_df, meteo_data)

# ------------------------------------------------------------------------------
# LAYOUT
# ------------------------------------------------------------------------------
col_map, col_stats = st.columns([2.2, 1])

with col_map:
    st.subheader("Regional Transboundary Haze Spatial Map")
    st.caption("Orange circles: FIRMS hotspots (radius ∝ FRP, greener = lower posterior). Blue columns: Malaysian receptors (height ∝ PM2.5).")
    try:
        deck = build_deck(hotspots_analyzed, receptors_df)
        st.pydeck_chart(deck, use_container_width=True)
    except Exception as e:
        st.error(f"Map rendering failed: {e}")
        st.map(pd.DataFrame({"lat": hotspots_analyzed["latitude"], "lon": hotspots_analyzed["longitude"]}))

    with st.expander("Physics diagnostics"):
        st.latex(r"C(x,y,z)=\frac{Q}{2\pi U \sigma_y\sigma_z}\exp\!\left(-\frac{y^2}{2\sigma_y^2}\right)\!\left[\exp\!\left(-\frac{(z-h)^2}{2\sigma_z^2}\right)+\exp\!\left(-\frac{(z+h)^2}{2\sigma_z^2}\right)\right]\!\exp\!\left(-\Lambda\frac{x}{U}\right)")
        st.latex(r"\Lambda = 10^{-4}\,P^{0.8},\quad t=x/U,\quad h=30\text{m},\quad Q=0.02\cdot\text{FRP}")
        st.latex(r"p(\theta|d_{obs})\propto \mathcal{N}(d_{obs}|f(\theta),R)\,p(\theta),\; p(\theta)\propto \text{FRP}")
        st.write(f"**RMSE (MAP hotspot vs all receptors):** `{rmse_val:.2f} µg/m³` | **σ_obs:** `{sigma_obs:.1f}` | **Scavenging:** `exp(-Λ·x/U)` with `Λ={1e-4*(meteo_data['precipitation']**0.8):.2e} s⁻¹`")

with col_stats:
    st.subheader("Bayesian Source Probabilities")
    display_df = hotspots_analyzed[["latitude", "longitude", "frp", "posterior_prob"]].sort_values(by="posterior_prob", ascending=False)
    # Highlight MAP
    try:
        st.dataframe(display_df.style.format({"frp": "{:.1f}", "posterior_prob": "{:.4f}", "latitude": "{:.3f}", "longitude": "{:.3f}"}).background_gradient(subset=["posterior_prob"], cmap="Oranges"), use_container_width=True, height=260)
    except Exception:
        st.dataframe(display_df, use_container_width=True)

    map_row = display_df.iloc[0] if not display_df.empty else None
    if map_row is not None:
        st.success(f"**MAP source:** `{map_row['latitude']:.3f}, {map_row['longitude']:.3f}` FRP `{map_row['frp']:.1f} MW` posterior `{map_row['posterior_prob']:.4f}`")

    st.subheader("Malaysian Receptor Stations")
    st.dataframe(receptors_df[["station_name", "pm25_obs"]].style.format({"pm25_obs": "{:.1f} µg/m³"}), use_container_width=True)

    # Predicted vs observed table for MAP
    if map_row is not None:
        st.markdown("**MAP prediction vs observation**")
        rows = []
        spot = hotspots_analyzed.loc[hotspots_analyzed["posterior_prob"].idxmax()]
        Q_est = float(spot["frp"] * 0.02)
        for _, rec in receptors_df.iterrows():
            c_pred = evaluate_gaussian_plume(float(spot["latitude"]), float(spot["longitude"]), float(rec["latitude"]), float(rec["longitude"]), Q_est, float(meteo_data["wind_speed"]), float(meteo_data["wind_dir"]), float(meteo_data["precipitation"]))
            rows.append({"station": rec["station_name"], "obs": float(rec["pm25_obs"]), "pred": c_pred})
        pred_df = pd.DataFrame(rows)
        st.dataframe(pred_df.style.format({"obs": "{:.1f}", "pred": "{:.1f}"}), use_container_width=True)

st.success("Bayesian atmospheric model inversion step successfully executed.")
st.info("Publish to Streamlit Cloud: push this repo to GitHub → share.streamlit.io → New app → set `FIRMS_MAP_KEY` in Secrets if you have one. See README.md.")
