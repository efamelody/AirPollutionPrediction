import numpy as np
import pandas as pd
import streamlit as st

from src.config import METEO_LAT, METEO_LON, SIGMA_OBS_UGM3
from src.data.firms import fetch_firms_hotspots
from src.data.meteo import fetch_open_meteo_weather
from src.data.meteo_grid import fetch_meteo_grid
from src.data.receptors import fetch_malaysia_receptors
from src.physics.dispersion import evaluate_gaussian_plume
from src.physics.inversion import run_bayesian_inversion
from src.physics.api import pm25_to_api, api_category, api_color_for_pm25
from src.physics.forecast_grid import build_haze_grid, predict_for_receptors
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
    "End-to-end haze forecast: **satellite fires (FRP)** → **wind + rain scavenging** → **plume dispersion** → **predicted PM2.5 / Malaysian API** at your cities. "
    "Wind arrows show where smoke is blowing; blue rain dots clean the air; heatmap = where haze will be."
)
st.caption(
    "Demonstrator v0.2 — Gaussian plume surrogate + Bayesian hotspot attribution. For 300-800 km transport this is illustrative; operational use needs HYSPLIT/WRF-Chem."
)

# ------------------------------------------------------------------------------
# SIDEBAR
# ------------------------------------------------------------------------------
st.sidebar.header("Controls")

default_key = "DEMO_KEY"
try:
    if "FIRMS_MAP_KEY" in st.secrets:
        default_key = st.secrets["FIRMS_MAP_KEY"]
except Exception:
    pass

firms_api_key = st.sidebar.text_input("NASA FIRMS MAP_KEY", value=default_key, type="password", help="firms.modaps.eosdis.nasa.gov — leave DEMO_KEY for synthetic fires (Sumatra/Kalimantan).")
selected_days = st.sidebar.slider("Hotspot window (days)", 1, 5, 1)
sigma_obs = st.sidebar.slider("Observation error σ_obs (µg/m³)", 5.0, 30.0, float(SIGMA_OBS_UGM3), help="Tighter = posteriors more peaked.")

st.sidebar.subheader("Weather")
override_meteo = st.sidebar.checkbox("Override weather manually")
if override_meteo:
    manual_wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.5, 20.0, 5.5)
    manual_wind_dir = st.sidebar.slider("Wind FROM direction (°) 0=N 90=E", 0, 360, 225)
    manual_precip = st.sidebar.slider("Rain (mm/h) — higher kills haze faster", 0.0, 20.0, 0.6)
    meteo_data = {"wind_speed": manual_wind_speed, "wind_dir": manual_wind_dir, "precipitation": manual_precip}
    meteo_grid = None
else:
    with st.spinner("Fetching regional wind + rain (Open-Meteo) ..."):
        meteo_data = fetch_open_meteo_weather(METEO_LAT, METEO_LON)

st.sidebar.markdown("---")
c1, c2, c3 = st.sidebar.columns(3)
c1.metric("Wind", f"{meteo_data['wind_speed']:.1f} m/s")
c2.metric("From", f"{meteo_data['wind_dir']:.0f}°")
c3.metric("Rain", f"{meteo_data['precipitation']:.1f} mm/h")
st.sidebar.caption("Wind FROM: 0=N 90=E 180=S 270=W (SW monsoon ≈225° blows toward Malaysia). Rain removes PM2.5 via Λ=1e-4·P^0.8.")

st.sidebar.subheader("Display")
show_haze = st.sidebar.checkbox("Show haze heatmap", value=True, help="Predicted PM2.5 grid summed over all fires (with rain scavenging).")
show_wind = st.sidebar.checkbox("Show wind vectors", value=True)
show_rain = st.sidebar.checkbox("Show rain overlay (blue dots)", value=False)
show_receptors = st.sidebar.checkbox("Show city towers", value=True)
haze_mode = st.sidebar.radio("City towers color by", ["Predicted API (forecast)", "Observed PM2.5 (now)"], index=0)
background_pm25 = st.sidebar.slider("Background PM2.5 (µg/m³) — clean-air baseline", 0.0, 25.0, 8.0)
grid_density = st.sidebar.select_slider("Map detail (grid size)", options=["Coarse", "Medium", "Fine"], value="Medium")

density_map = {"Coarse": (10, 14), "Medium": (18, 22), "Fine": (26, 30)}
n_lat_haze, n_lon_haze = density_map[grid_density]

weight_opt = st.sidebar.radio("Plume sum method", ["Bayesian avg (posterior-weighted)", "Sum all fires (worst-case)"], index=0)
weight_by_posterior = weight_opt.startswith("Bayesian")
amplification = st.sidebar.slider("Haze visual amplification (uncalibrated model)", 1, 200, 80, help="Gaussian plume under-predicts at 300-800km. This scales plume concentrations for pattern visibility. Higher = more haze shown. Keep constant when comparing wind/rain effects.")
st.sidebar.caption("⚠️ Amplification is a visual aid — physics pattern is real, absolute numbers are illustrative until calibrated with DOE data.")

# ------------------------------------------------------------------------------
# CACHED FETCHERS
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_hotspots(key: str, days: int) -> pd.DataFrame:
    return fetch_firms_hotspots(key, "IDN", days)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_meteo(lat: float, lon: float, override: bool, ws: float, wd: float, pr: float) -> dict:
    if override:
        return {"wind_speed": ws, "wind_dir": wd, "precipitation": pr}
    return fetch_open_meteo_weather(lat, lon)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_meteo_grid(n_lat: int, n_lon: int, override: bool, ws: float, wd: float, pr: float) -> pd.DataFrame:
    if override:
        # Build uniform grid from manual values for visual consistency
        import numpy as np
        lats = np.linspace(-3, 7, n_lat)
        lons = np.linspace(98, 119, n_lon)
        rows = []
        for la in lats:
            for lo in lons:
                th = np.radians(wd)
                u = -ws * np.sin(th); v = -ws * np.cos(th)
                rows.append({"lat": la, "lon": lo, "wind_speed": ws, "wind_dir": wd, "precip": pr, "u": u, "v": v})
        return pd.DataFrame(rows)
    return fetch_meteo_grid(n_lat=n_lat, n_lon=n_lon)

if not override_meteo:
    meteo_data = cached_meteo(METEO_LAT, METEO_LON, False, 0, 0, 0)
else:
    meteo_data = cached_meteo(METEO_LAT, METEO_LON, True, meteo_data["wind_speed"], meteo_data["wind_dir"], meteo_data["precipitation"])

hotspots_raw = cached_hotspots(firms_api_key, selected_days)
receptors_df = fetch_malaysia_receptors()

# ------------------------------------------------------------------------------
# BAYESIAN INVERSION
# ------------------------------------------------------------------------------
with st.spinner("Attributing smoke to hotspots (Bayes) ..."):
    hotspots_analyzed = run_bayesian_inversion(hotspots_raw, receptors_df, meteo_data, sigma_obs=sigma_obs)

# ------------------------------------------------------------------------------
# HAZE FORECAST GRID + CITY FORECASTS
# ------------------------------------------------------------------------------
with st.spinner("Forecasting haze over Malaysia (plume sum + API) ..."):
    haze_grid = build_haze_grid(
        hotspots_analyzed, meteo_data,
        n_lat=n_lat_haze, n_lon=n_lon_haze,
        background_pm25=background_pm25,
        weight_by_posterior=weight_by_posterior,
        amplification=float(amplification),
    )
    city_forecast = predict_for_receptors(
        hotspots_analyzed, receptors_df, meteo_data,
        background_pm25=background_pm25,
        weight_by_posterior=weight_by_posterior,
        amplification=float(amplification),
    )

# Meteo grid for arrows/rain (separate fetch - cached)
meteo_grid = cached_meteo_grid(4, 6, override_meteo, float(meteo_data["wind_speed"]), float(meteo_data["wind_dir"]), float(meteo_data["precipitation"]))

# ------------------------------------------------------------------------------
# ALERTS — Will my city get hazy?
# ------------------------------------------------------------------------------
st.subheader("Will it be hazy? — City forecast (next hours, same wind/rain assumption)")
alert_cols = st.columns(len(city_forecast))
for col, (_, row) in zip(alert_cols, city_forecast.iterrows()):
    api = int(row["api_pred"])
    cat, color = api_category(api)
    # Color border by category
    with col:
        st.markdown(
            f"<div style='padding:10px;border-radius:10px;border:2px solid {color};background:#0f1117'>"
            f"<b>{row['station_name']}</b><br>"
            f"<span style='font-size:22px;color:{color}'>{api} — {cat}</span><br>"
            f"<span style='font-size:12px'>PM2.5 pred {row['pm25_pred']:.1f} µg/m³<br>obs {row['pm25_obs']:.1f} µg/m³</span>"
            f"</div>",
            unsafe_allow_html=True,
        )
        if api >= 101:
            st.warning("Unhealthy — limit outdoor activity" if api < 201 else "Very unhealthy / hazardous — stay indoors")
        elif api >= 51:
            st.info("Moderate — sensitive groups take care")

# Quick haze summary stat
if not haze_grid.empty:
    worst = haze_grid.loc[haze_grid["pm25"].idxmax()]
    avg_my = haze_grid[(haze_grid["lat"].between(0.8, 6.8)) & (haze_grid["lon"].between(99.5, 119.0))]["pm25"].mean()
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Worst grid cell", f"{worst['pm25']:.0f} µg/m³", f"API {worst['api']} {worst['category']}")
    kpi2.metric("Avg over Malaysia grid", f"{avg_my:.1f} µg/m³", f"API {pm25_to_api(avg_my)}")
    kpi3.metric("Sum of FRP (fire strength)", f"{hotspots_analyzed['frp'].sum():.0f} MW")
    kpi4.metric("MAP hotspot (most likely source)", f"{hotspots_analyzed.sort_values('posterior_prob', ascending=False).iloc[0]['posterior_prob']:.2%}")

# ------------------------------------------------------------------------------
# MAIN MAP
# ------------------------------------------------------------------------------
col_map, col_stats = st.columns([2.3, 1])

with col_map:
    st.subheader("Map — Haze, Wind, Rain")
    st.caption("Heatmap = predicted PM2.5 (haze). Height = pollution amount. Orange circles = fires (bigger = stronger FRP, redder = more likely source). Arrows = wind blowing toward Malaysia. Blue dots = rain washing haze away.")
    try:
        deck = build_deck(
            hotspots_analyzed,
            city_forecast if haze_mode.startswith("Predicted") else receptors_df,
            haze_grid=haze_grid if show_haze else None,
            meteo_grid=meteo_grid,
            show_haze=show_haze,
            show_wind=show_wind,
            show_rain=show_rain,
            show_receptors=show_receptors,
        )
        st.pydeck_chart(deck, use_container_width=True)
        # Legend
        st.markdown(
            "<span style='color:#00b050'>■ Good (0-50)</span> &nbsp; <span style='color:#ffeb3b'>■ Moderate (51-100)</span> &nbsp; "
            "<span style='color:#ff9800'>■ Unhealthy (101-200)</span> &nbsp; <span style='color:#f44336'>■ Very Unhealthy (201-300)</span> &nbsp; "
            "<span style='color:#8b0000'>■ Hazardous (301+)</span>",
            unsafe_allow_html=True,
        )
    except Exception as e:
        st.error(f"Map render failed: {e}")
        st.map(pd.DataFrame({"lat": hotspots_analyzed["latitude"], "lon": hotspots_analyzed["longitude"]}))

    with st.expander("How this forecasts haze (physics you can audit)"):
        st.latex(r"C=\frac{Q}{\pi U\sigma_y\sigma_z}\exp(-\tfrac{y^2}{2\sigma_y^2})\exp(-\tfrac{h^2}{2\sigma_z^2})\exp(-\Lambda x/U)\cdot10^6")
        st.latex(r"\Lambda=10^{-4}P^{0.8},\; h=30\text{m},\; Q=0.02\cdot\text{FRP},\; p(\theta|d)\propto\mathcal{N}(d|f(\theta),R)\,p(\theta)")
        st.write(
            f"Forecast at each grid cell = `background ({background_pm25:.0f}) + Σ fires Q·plume·scavenging` with today wind `{meteo_data['wind_speed']:.1f} m/s from {meteo_data['wind_dir']:.0f}°` and rain `{meteo_data['precipitation']:.1f} mm/h`. "
            f"Method: `{weight_opt}`. RMSE (MAP vs observed, debug): `{np.sqrt(np.mean((city_forecast['pm25_obs']-city_forecast['pm25_pred'])**2)):.1f}` µg/m³."
        )

with col_stats:
    st.subheader("Which fire is causing it? (Bayes)")
    disp = hotspots_analyzed[["latitude", "longitude", "frp", "posterior_prob"]].sort_values("posterior_prob", ascending=False)
    try:
        st.dataframe(disp.style.format({"frp": "{:.1f}", "posterior_prob": "{:.4f}", "latitude": "{:.3f}", "longitude": "{:.3f}"}).background_gradient(subset=["posterior_prob"], cmap="Oranges"), use_container_width=True, height=240)
    except Exception:
        st.dataframe(disp, use_container_width=True)
    if not disp.empty:
        top = disp.iloc[0]
        st.success(f"**Most likely source:** `{top['latitude']:.3f}, {top['longitude']:.3f}` FRP `{top['frp']:.1f} MW` — `{top['posterior_prob']:.2%}` posterior")

    st.subheader("Cities — Predicted vs Observed")
    show_cols = ["station_name", "pm25_obs", "pm25_pred", "api_pred", "category"]
    try:
        st.dataframe(city_forecast[show_cols].style.format({"pm25_obs": "{:.1f}", "pm25_pred": "{:.1f}", "api_pred": "{:.0f}"}).background_gradient(subset=["api_pred"], cmap="Reds"), use_container_width=True)
    except Exception:
        st.dataframe(city_forecast[show_cols], use_container_width=True)

    st.markdown("**Download forecast**")
    st.download_button("Haze grid CSV", data=haze_grid.to_csv(index=False).encode(), file_name="haze_grid_forecast.csv", mime="text/csv")
    st.download_button("City forecast CSV", data=city_forecast.to_csv(index=False).encode(), file_name="city_forecast.csv", mime="text/csv")

st.success(f"Inversion + forecast done for {len(hotspots_analyzed)} fires × {len(city_forecast)} cities × {len(haze_grid)} grid cells.")
st.info("Publish update: `git push` then Streamlit Cloud auto-redeploys. Set FIRMS key in App Secrets for live fires.")
