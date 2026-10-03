import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime, timezone

from src.config import METEO_LAT, METEO_LON
from src.data.firms import fetch_firms_hotspots
from src.data.meteo_grid import fetch_meteo_grid, get_meteo_for_transport, get_centroid_meteo
from src.data.receptors import fetch_malaysia_receptors, fetch_combined_receptors
from src.physics.api import pm25_to_api, api_category
from src.physics.research_model import predict_haze, predict_haze_forecast
from src.physics.calibration import compute_validation_metrics

# ------------------------------------------------------------------------------
# PAGE CONFIG — plain language
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Will My City Have Haze? — Malaysia Haze Forecast", layout="wide", initial_sidebar_state="expanded")

# Helpers for plain wind description
def wind_plain(dir_deg: float) -> str:
    dirs = [(0,"North"),(45,"Northeast"),(90,"East"),(135,"Southeast"),(180,"South"),(225,"Southwest"),(270,"West"),(315,"Northwest"),(360,"North")]
    best = min(dirs, key=lambda x: abs(x[0]-dir_deg) if abs(x[0]-dir_deg)<180 else 360-abs(x[0]-dir_deg))
    to_deg = (dir_deg + 180) % 360
    to_label = min(dirs, key=lambda x: abs(x[0]-to_deg) if abs(x[0]-to_deg)<180 else 360-abs(x[0]-to_deg))[1]
    return f"from the {best[1]} → blowing to the {to_label}"

def rain_plain(p: float) -> str:
    if p < 0.3: return "No rain — haze can travel far"
    if p < 2: return "Light rain — some haze washed away"
    if p < 7: return "Moderate rain — haze reduced a lot"
    return "Heavy rain — air is being cleaned"

# ------------------------------------------------------------------------------
# HEADER — layman first
# ------------------------------------------------------------------------------
st.title("Will My City Have Haze?")
st.markdown("### A simple forecast for Malaysia — based on fires in Indonesia, wind, and rain")
st.info(
    "**How to use:** 1) Check the coloured cards below — green = good air, red = haze. "
    "2) Open **Prediction** to see where haze will be on the map. "
    "3) Open **Where is it coming from?** to see fires + wind + rain evidence. "
    "No jargon — just the answer: *do I need to worry about haze?*"
)

# ------------------------------------------------------------------------------
# SIDEBAR — plain language, advanced hidden
# ------------------------------------------------------------------------------
st.sidebar.markdown("## ⚙️ Settings")
st.sidebar.caption("You don't need to change anything — defaults use live data. Change only to ask *what if?*")

default_key = "DEMO_KEY"
try:
    if "FIRMS_MAP_KEY" in st.secrets:
        default_key = st.secrets["FIRMS_MAP_KEY"]
        # Debug: show first 8 chars of key
        st.sidebar.caption(f"Using FIRMS key from secrets: {default_key[:8]}...")
except Exception as e:
    st.sidebar.caption(f"Could not load FIRMS key from secrets: {e}")

st.sidebar.markdown("**Fires:** Using demo fires in Sumatra & Kalimantan. Add your NASA key to see live satellite fires.")
firms_api_key = st.sidebar.text_input("NASA FIRMS key (leave DEMO_KEY for demo)", value=default_key, type="password", help="Free at firms.modaps.eosdis.nasa.gov")
selected_days = st.sidebar.slider("Fire history (days)", 1, 5, 1, help="1 day = only today's fires. 5 days = more fires, but older.")

st.sidebar.markdown("### 🌤️ Weather right now")
override_meteo = st.sidebar.checkbox("Try a different wind/rain (what-if?)", value=False)

# Scenario controls for research model
st.sidebar.markdown("### 🧪 Research Model Scenarios")
rain_multiplier = st.sidebar.slider("Rainfall multiplier", 0.0, 3.0, 1.0, 0.1, help="Scale rainfall to test wet/dry scenarios")
wind_shift = st.sidebar.slider("Wind direction shift (°)", -45, 45, 0, 5, help="Shift wind direction to test transport sensitivity")
urban_scale = st.sidebar.slider("Urban baseline scale", 0.5, 2.0, 1.0, 0.1, help="Scale local urban pollution baseline")

if override_meteo:
    st.sidebar.caption("Move the sliders to see how wind and rain change the haze.")
    manual_wind_speed = st.sidebar.slider("Wind speed", 0.5, 20.0, 5.5, help="Faster wind spreads haze faster")
    manual_wind_dir = st.sidebar.slider("Wind direction — where it comes FROM", 0, 360, 225, help="225° = from Southwest (typical haze season) → blows toward Malaysia. 90° = from East → blows away from Malaysia.")
    st.sidebar.caption(f"→ {wind_plain(manual_wind_dir)}")
    manual_precip = st.sidebar.slider("Rain amount", 0.0, 20.0, 0.6, help="More rain = more haze washed out of the air")
    st.sidebar.caption(rain_plain(manual_precip))
    meteo_data = {"wind_speed": manual_wind_speed, "wind_dir": manual_wind_dir, "precipitation": manual_precip}
else:
    with st.spinner("Getting live wind and rain..."):
        # Get centroid meteo from gridded ERA5
        meteo_interpolators = get_meteo_for_transport()
        meteo_data = get_centroid_meteo(meteo_interpolators)

# Live summary in plain language, top of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Live weather (Strait of Malacca):**<br>
💨 Wind **{meteo_data['wind_speed']:.1f} m/s** {wind_plain(meteo_data['wind_dir'])}<br>
🌧️ Rain **{meteo_data['precipitation']:.1f} mm/h** — {rain_plain(meteo_data['precipitation'])}<br>
📏 PBL Height **{meteo_data.get('pbl_height', 1000):.0f} m**<br>
<span style="color:#888;font-size:12px;">Source: Open-Meteo ERA5. Rain cleans haze. PBL caps vertical mixing.</span>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🗺️ What to show on the map")
show_haze = st.sidebar.checkbox("Show predicted haze cloud", value=True, help="Coloured blobs where we predict haze will be")
show_wind = st.sidebar.checkbox("Show wind arrows", value=True, help="Arrows show where smoke is being blown")
show_rain = st.sidebar.checkbox("Show rain dots", value=False, help="Blue dots where it's raining")
show_receptors = st.sidebar.checkbox("Show city towers", value=True)

# Model info
st.sidebar.markdown("---")
st.sidebar.markdown("**Model:** Proxy-Calibrated Research Model")
st.sidebar.caption("• Biome-specific emissions (peat vs forest)\n• Trajectory transport with ERA5 PBL\n• Path-integrated wet scavenging\n• Continuous Bayesian inverse\n• Urban diurnal baseline from CAMS")

# ------------------------------------------------------------------------------
# CACHED FETCHERS
# ------------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def cached_hotspots(key: str, days: int) -> pd.DataFrame:
    return fetch_firms_hotspots(key, "IDN", days)

@st.cache_data(ttl=1800, show_spinner=False)
def cached_meteo_grid(n_lat: int = 8, n_lon: int = 12, forecast_hours: int = 24) -> pd.DataFrame:
    return fetch_meteo_grid(n_lat=n_lat, n_lon=n_lon, forecast_hours=forecast_hours)

# Fetch data
use_live_air = st.sidebar.checkbox("Use live air quality (Open-Meteo CAMS)", value=True, help="When ON: city 'Measured now' is real current PM2.5 from Open-Meteo. When OFF: shows mock haze-event to demo a heavy haze day.")

# WAQI token
try:
    waqi_token = st.secrets.get("WAQI_API_TOKEN", "")
except Exception:
    waqi_token = ""

# Validation source selector
validation_source = st.sidebar.radio(
    "Validation data",
    ["CAMS (model)", "WAQI (ground)", "Both"],
    index=2,
    help="CAMS = ECMWF model reanalysis. WAQI = Malaysia DOE ground stations via waqi.info"
)

# Target hour for validation: model predicts for (start_hour + forecast_hours - 1) % 24
start_hour_utc = datetime.now(timezone.utc).hour
forecast_hours = 24
target_hour_utc = (start_hour_utc + forecast_hours - 1) % 24

receptors_df = fetch_combined_receptors(
    cams_live=use_live_air,
    waqi_token=waqi_token if waqi_token else None,
    target_hour_utc=target_hour_utc
)
if not receptors_df.empty and "pm25_source" in receptors_df.columns:
    source_info = receptors_df['pm25_source'].iloc[0]
    if "pm25_waqi" in receptors_df.columns:
        waqi_count = receptors_df["pm25_waqi"].notna().sum()
        source_info += f" + WAQI ({waqi_count}/5 stations)"
    st.sidebar.caption(f"City air: {source_info} (hour {target_hour_utc}:00 UTC)")

hotspots_raw = cached_hotspots(firms_api_key, selected_days)

# Build scenario dict
scenario = {
    "rain_multiplier": rain_multiplier,
    "wind_shift_deg": wind_shift,
    "urban_scale": urban_scale,
}

# Run research model
with st.spinner("Running calibrated haze attribution model..."):
    result = predict_haze(hotspots_raw, receptors_df, forecast_hours=24, scenario=scenario)

city_forecast = result["city_forecast"]
fire_posteriors = result["fire_posteriors"]
haze_grid = result["haze_grid"]
metrics = result["metrics"]

# Get meteo grid for visualization
meteo_grid_df = cached_meteo_grid(4, 6, 24)
# Use current hour for viz
current_hour = datetime.now(timezone.utc).hour
meteo_grid_viz = meteo_grid_df[meteo_grid_df["time"].str.contains(f"T{current_hour:02d}:")].copy()
if meteo_grid_viz.empty:
    meteo_grid_viz = meteo_grid_df.iloc[:24].copy()

# ------------------------------------------------------------------------------
# PLAIN KPI — what matters to a normal person
# ------------------------------------------------------------------------------
k1, k2, k3 = st.columns(3)
with k1:
    n_fires = len(fire_posteriors)
    total_strength = fire_posteriors["frp"].sum() if n_fires else 0
    st.metric("🔥 Fires detected", f"{n_fires}", f"Total FRP {total_strength:.0f} MW")
    st.caption("From NASA satellites over Indonesia (last {} day(s))".format(selected_days))
with k2:
    st.metric("💨 Wind", f"{meteo_data['wind_speed']:.1f} m/s", wind_plain(meteo_data['wind_dir']))
    st.caption(rain_plain(meteo_data['precipitation']))
with k3:
    n_haze = (city_forecast["api_pred"] >= 101).sum() if not city_forecast.empty else 0
    st.metric("🏙️ Cities with haze forecast", f"{n_haze} of {len(city_forecast)}", f"API ≥100 = haze" if n_haze else "Air looks okay")
    st.caption("Calibrated: biome emissions + trajectory transport + urban baseline")

# Validation metrics badge
if metrics:
    st.sidebar.markdown("---")
    
    # CAMS validation (from model)
    st.sidebar.markdown("**Proxy Validation (vs CAMS model):**")
    st.sidebar.metric("RMSE", f"{metrics.get('RMSE', 0):.1f} µg/m³")
    st.sidebar.metric("NMB", f"{metrics.get('NMB', 0):.1f}%")
    st.sidebar.metric("R", f"{metrics.get('R', 0):.2f}")
    
    # WAQI validation (ground truth) - compute if WAQI data available
    if "pm25_waqi" in receptors_df.columns and receptors_df["pm25_waqi"].notna().any():
        pred_pm25 = city_forecast["pm25_pred"].values
        waqi_obs = []
        for _, r in city_forecast.iterrows():
            city = r["station_name"]
            waqi_val = receptors_df.loc[receptors_df["station_name"] == city, "pm25_waqi"].values
            if len(waqi_val) > 0 and not pd.isna(waqi_val[0]):
                waqi_obs.append(float(waqi_val[0]))
            else:
                waqi_obs.append(np.nan)
        waqi_obs = np.array(waqi_obs)
        valid = ~np.isnan(waqi_obs)
        if valid.sum() >= 3:
            waqi_metrics = compute_validation_metrics(pred_pm25[valid], waqi_obs[valid])
            st.sidebar.markdown("**Ground-Truth Validation (vs WAQI stations):**")
            st.sidebar.metric("RMSE", f"{waqi_metrics.get('RMSE', 0):.1f} µg/m³")
            st.sidebar.metric("NMB", f"{waqi_metrics.get('NMB', 0):.1f}%")
            st.sidebar.metric("R", f"{waqi_metrics.get('R', 0):.2f}")
            st.sidebar.caption(f"Based on {valid.sum()}/5 stations")

# ------------------------------------------------------------------------------
# CITY CARDS — prediction vs truth, plain language
# ------------------------------------------------------------------------------
st.markdown("## 🏙️ Will it be hazy? — City by city")
st.caption("**Top = calibrated prediction** (fire smoke only). **Middle = CAMS model** (ECMWF reanalysis). **Bottom = WAQI ground stations** (Malaysia DOE). Model predicts fire smoke; CAMS/WAQI include local pollution.")

mismatch = 0
if not city_forecast.empty:
    for _, r in city_forecast.iterrows():
        if abs(pm25_to_api(float(r["pm25_obs"])) - int(r["api_pred"])) >= 50:
            mismatch += 1
if mismatch >= 2:
    st.warning(f"⚠️ Predicted haze much lower than measured in {mismatch} cities → local pollution dominates, wind blowing smoke away.")
elif mismatch >= 1:
    st.info("ℹ️ Small mismatch: predicted fire smoke vs measured total air. Check wind direction in Evidence tab.")

cols = st.columns(len(city_forecast))
for col, (_, row) in zip(cols, city_forecast.iterrows()):
    api = int(row["api_pred"])
    cat, color = api_category(api)
    
    # CAMS observation
    cams_obs = float(row["pm25_obs"])
    cams_api = pm25_to_api(cams_obs)
    cams_cat, cams_color = api_category(cams_api)
    if cams_api >= 201: cams_advice = "🔴 Hazardous — stay indoors"
    elif cams_api >= 101: cams_advice = "🟠 Unhealthy — limit outdoor"
    elif cams_api >= 51: cams_advice = "🟡 Moderate — sensitive groups care"
    else: cams_advice = "🟢 Good air now"
    
    # WAQI observation (ground truth)
    city_name = row['station_name']
    waqi_obs = None
    if "pm25_waqi" in receptors_df.columns:
        waqi_val = receptors_df.loc[receptors_df["station_name"] == city_name, "pm25_waqi"].values
        if len(waqi_val) > 0 and not pd.isna(waqi_val[0]):
            waqi_obs = float(waqi_val[0])
    
    if waqi_obs is not None:
        waqi_api = pm25_to_api(waqi_obs)
        waqi_cat, waqi_color = api_category(waqi_api)
        if waqi_api >= 201: waqi_advice = "🔴 Hazardous — stay indoors"
        elif waqi_api >= 101: waqi_advice = "🟠 Unhealthy — limit outdoor"
        elif waqi_api >= 51: waqi_advice = "🟡 Moderate — sensitive groups care"
        else: waqi_advice = "🟢 Good air now"
    
    if api >= 201: pred_advice, pred_face = "🔴 Fire smoke: Stay indoors", "😷"
    elif api >= 101: pred_advice, pred_face = "🟠 Fire smoke: Limit outdoor", "😶‍🌫️"
    elif api >= 51: pred_advice, pred_face = "🟡 Some fire haze", "🙂"
    else: pred_advice, pred_face = "🟢 No fire haze coming", "😊"
    
    with col:
        # Build WAQI section if available
        waqi_section = ""
        if waqi_obs is not None:
            waqi_section = f"""
    <div style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2a3a;">
      <div style="font-size:11px;color:#888;">WAQI GROUND STATION (DOE)</div>
      <div style="font-size:14px;font-weight:700;color:{waqi_color};">API {waqi_api} — {waqi_cat} <span style="font-weight:400;color:#aaa;">({waqi_obs:.1f} µg/m³)</span></div>
      <div style="font-size:11px;color:{waqi_color};margin-top:2px;">{waqi_advice}</div>
      <div style="font-size:10px;color:#888;margin-top:4px;">Ground measurement from Malaysia DOE station.</div>
    </div>
"""
        
        st.markdown(f"""
<div style="padding:12px;border-radius:12px;border:2px solid {color};background:#0f1117;text-align:center;">
  <div style="font-weight:700;font-size:15px;">{city_name}</div>
  <div style="font-size:26px;margin:4px 0;">{pred_face}</div>
  <div style="font-size:11px;color:#888;letter-spacing:0.5px;">PREDICTED (fire smoke only)</div>
  <div style="font-size:20px;font-weight:800;color:{color};">API {api} — {cat}</div>
  <div style="font-size:11px;color:#ccc;">PM2.5 {row['pm25_pred']:.1f} µg/m³</div>
  <div style="font-size:11px;color:{color};margin-top:4px;font-weight:600;">{pred_advice}</div>
  <div style="margin-top:8px;padding-top:8px;border-top:1px solid #2a2a3a;">
    <div style="font-size:11px;color:#888;">CAMS MODEL (Open-Meteo ECMWF)</div>
    <div style="font-size:14px;font-weight:700;color:{cams_color};">API {cams_api} — {cams_cat} <span style="font-weight:400;color:#aaa;">({cams_obs:.1f} µg/m³)</span></div>
    <div style="font-size:11px;color:{cams_color};margin-top:2px;">{cams_advice}</div>
    <div style="font-size:10px;color:#888;margin-top:4px;">Model reanalysis ≠ ground sensors. Often higher.</div>
  </div>
{waqi_section}
</div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TABS — prediction vs evidence vs explanation
# ------------------------------------------------------------------------------
tab_pred, tab_source, tab_forecast, tab_how = st.tabs([
    "🌫️ Prediction — Where will haze be?", 
    "🔥 Evidence — Where is smoke coming from?", 
    "📈 Forecast — Next 24 hours",
    "❓ How it works (simple)"
])

with tab_pred:
    st.markdown("### Prediction: where the haze will be")
    st.caption("Calibrated forecast: biome-differentiated emissions, trajectory transport, PBL-capped mixing, path-integrated rain scavenging.")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        try:
            from src.viz.maps import build_deck
            deck_pred = build_deck(
                fire_posteriors.head(0),  # hide fires for clean prediction view
                city_forecast,
                haze_grid=haze_grid if show_haze else None,
                meteo_grid=None,
                show_haze=show_haze, show_wind=False, show_rain=False, show_receptors=show_receptors,
            )
            st.pydeck_chart(deck_pred, use_container_width=True)
        except Exception as e:
            st.error(f"Map failed: {e}")
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:10px;padding:10px 12px;">
<b>Air quality scale (Malaysian API):</b><br>
<span style="background:#00b050;color:white;padding:2px 7px;border-radius:4px;">Good 0–50</span>
<span style="background:#ffeb3b;color:#111;padding:2px 7px;border-radius:4px;">Moderate 51–100</span>
<span style="background:#ff9800;color:white;padding:2px 7px;border-radius:4px;">Unhealthy 101–200</span>
<span style="background:#f44336;color:white;padding:2px 7px;border-radius:4px;">Very Unhealthy 201–300</span>
<span style="background:#8b0000;color:white;padding:2px 7px;border-radius:4px;">Hazardous 301+</span>
<br><span style="color:#aaa;font-size:12px;">API ≥100 = haze. 3D columns: taller & redder = worse haze.</span>
</div>
        """, unsafe_allow_html=True)
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:10px;padding:10px 12px;margin-top:10px;">
<b>🌫️ Haze cloud</b> = coloured blobs where calibrated model predicts smoke. Colour = API level.<br>
<b>🏙️ City towers</b> = 3D columns over KL, JB, Kuching, Ipoh, Kota Bharu. Height & colour = predicted API.
</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("#### Prediction vs CAMS model — are we right?")
        st.caption("If the two numbers are close, the calibrated forecast matches the CAMS model (not ground truth). Ground sensors (IQAir/DOE) often read lower.")
        disp_pred = city_forecast[["station_name", "pm25_obs", "pm25_pred", "api_pred", "category"]].copy()
        disp_pred.columns = ["City", "CAMS Model", "Predicted", "API", "Level"]
        try:
            st.dataframe(disp_pred.style.format({"CAMS Model": "{:.1f}", "Predicted": "{:.1f}", "API": "{:.0f}"}).background_gradient(subset=["API"], cmap="Reds"), use_container_width=True)
        except Exception:
            st.dataframe(disp_pred, use_container_width=True)
        avg_err = float((city_forecast["pm25_obs"] - city_forecast["pm25_pred"]).abs().mean()) if not city_forecast.empty else 0
        st.metric("Avg diff vs CAMS", f"{avg_err:.1f} µg/m³", "lower = better")
        st.caption("Difference vs CAMS model. Proxy validation RMSE vs CAMS reanalysis shown in sidebar.")
        st.download_button("⬇️ Download city forecast CSV", data=city_forecast.to_csv(index=False).encode(), file_name="city_forecast.csv", mime="text/csv")
        st.download_button("⬇️ Download haze map grid CSV", data=haze_grid.to_csv(index=False).encode(), file_name="haze_grid.csv", mime="text/csv")

with tab_source:
    st.markdown("### Evidence: where is the smoke coming from?")
    st.caption("Calibrated attribution: fires colored by posterior probability from Bayesian inverse.")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        try:
            from src.viz.maps import build_deck
            deck_source = build_deck(fire_posteriors, receptors_df, haze_grid=haze_grid if show_haze else None, meteo_grid=meteo_grid_viz, show_haze=show_haze, show_wind=show_wind, show_rain=show_rain, show_receptors=False)
            st.pydeck_chart(deck_source, use_container_width=True)
        except Exception as e:
            st.error(f"Map failed: {e}")
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:10px;padding:12px 14px;font-size:13px;line-height:1.5;">
  <b>How to read this evidence map:</b><br>
  🔴 <b>Red/orange circles</b> = fires from NASA FIRMS. <b>Size = FRP (MW)</b>. <b>Color = posterior probability</b> (redder = more likely culprit).<br>
  💨 <b>Grey arrows</b> = ERA5 wind. Arrow points where smoke is blown.<br>
  🌧️ <b>Blue dots</b> = rain rate. Bigger = heavier rain washing out smoke.<br>
  🌫️ <b>Faint blobs</b> = calibrated haze forecast (matches downwind of high-probability fires).
</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("#### Which fire is most likely the culprit?")
        st.caption("Bayesian inverse attribution with spatial error covariance. Posterior = how well fire explains observed PM2.5 given wind/rain.")
        disp = fire_posteriors[["latitude", "longitude", "frp", "biome", "Q_init", "posterior_prob"]].sort_values("posterior_prob", ascending=False)
        disp.columns = ["Lat", "Lon", "FRP (MW)", "Biome", "Q (kg/s)", "Posterior Prob"]
        try:
            st.dataframe(disp.style.format({"FRP (MW)": "{:.1f}", "Q (kg/s)": "{:.4f}", "Posterior Prob": "{:.2%}", "Lat": "{:.3f}", "Lon": "{:.3f}"}).background_gradient(subset=["Posterior Prob"], cmap="Oranges"), use_container_width=True, height=260)
        except Exception:
            st.dataframe(disp, use_container_width=True)
        if not disp.empty:
            top = disp.iloc[0]
            st.success(f"**Most likely source:** {top['Biome']} fire at {top['Lat']:.3f}, {top['Lon']:.3f} — FRP {top['FRP (MW)']:.1f} MW — **{top['Posterior Prob']:.0%} posterior probability**")
            st.caption("Posterior = how well this fire's smoke, transported by ERA5 winds and scavenged by rain, matches observed PM2.5 at Malaysian cities.")
        st.markdown("---")
        st.markdown("**Data sources:**")
        st.markdown("- **Fires:** NASA FIRMS VIIRS (375m, 3h) + ESA WorldCover biome\n- **Wind/PBL/Rain:** ERA5 / Open-Meteo (hourly, 0.25°)\n- **Urban baseline:** CAMS EAC4 non-haze months (Nov–Apr)\n- **Validation:** CAMS EAC4 Sep 2019 haze reanalysis")

with tab_forecast:
    st.markdown("### 24-Hour Calibrated Forecast")
    st.caption("Hourly predictions showing how haze evolves with changing wind, rain, and PBL height.")
    
    with st.spinner("Generating 24-hour forecast..."):
        forecast_df = predict_haze_forecast(hotspots_raw, receptors_df, forecast_hours=24)
    
    # Fetch WAQI forecast if token available
    waqi_forecast_data = None
    if waqi_token:
        from src.data.waqi import fetch_waqi_forecast, STATION_UIDS
        selected_city = None  # will be set by selectbox
        # We'll fetch for all cities and use the selected one
        waqi_forecasts = {}
        for city, uid in STATION_UIDS.items():
            fc = fetch_waqi_forecast(uid, waqi_token)
            if fc:
                waqi_forecasts[city] = fc
    
    # City selector
    cities = forecast_df["station_name"].unique()
    selected_city = st.selectbox("Select city", cities)
    
    city_forecast = forecast_df[forecast_df["station_name"] == selected_city].copy()
    
    # Plot
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=city_forecast["hour_utc"],
        y=city_forecast["pm25_pred"],
        mode="lines+markers",
        name="Model Predicted PM2.5",
        line=dict(color="#ff9800", width=3),
        marker=dict(size=8)
    ))
    
    # Add WAQI forecast if available
    if waqi_token and selected_city in waqi_forecasts:
        waqi_fc = waqi_forecasts[selected_city]
        # WAQI forecast is daily averages - convert to hourly for plotting
        # For now, show as daily markers
        days = [d["day"] for d in waqi_fc]
        avg_vals = [d["avg"] for d in waqi_fc]
        # Convert day strings to hour positions (assume mid-day)
        day_hours = [i * 24 + 12 for i in range(len(days))]  # rough mapping
        fig.add_trace(go.Scatter(
            x=day_hours,
            y=avg_vals,
            mode="markers",
            name="WAQI Forecast (daily avg)",
            marker=dict(color="#00b050", size=12, symbol="diamond"),
        ))
    
    # Add API threshold lines
    for thresh, label, color in [(12, "Good→Moderate", "#00b050"), (35.4, "Moderate→Unhealthy", "#ff9800"), (55.4, "Unhealthy→Very Unhealthy", "#f44336")]:
        fig.add_hline(y=thresh, line_dash="dash", line_color=color, annotation_text=label, annotation_position="right")
    
    fig.update_layout(
        title=f"{selected_city} — 24h PM2.5 Forecast",
        xaxis_title="Hour (UTC)",
        yaxis_title="PM2.5 (µg/m³)",
        template="plotly_dark",
        height=400,
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Table
    st.dataframe(city_forecast[["hour_utc", "pm25_pred", "api_pred", "category"]].style.format({"pm25_pred": "{:.1f}", "api_pred": "{:.0f}"}), use_container_width=True)
    
    # WAQI forecast table
    if waqi_token and selected_city in waqi_forecasts:
        st.markdown("#### WAQI Forecast (Daily Average PM2.5)")
        waqi_fc = waqi_forecasts[selected_city]
        waqi_df = pd.DataFrame(waqi_fc)
        waqi_df["API"] = waqi_df["avg"].apply(pm25_to_api)
        waqi_df["Category"] = waqi_df["API"].apply(lambda x: api_category(x)[0])
        st.dataframe(waqi_df[["day", "avg", "min", "max", "API", "Category"]].style.format({"avg": "{:.0f}", "min": "{:.0f}", "max": "{:.0f}", "API": "{:.0f}"}), use_container_width=True)
    
    st.download_button("⬇️ Download 24h forecast CSV", data=forecast_df.to_csv(index=False).encode(), file_name="haze_forecast_24h.csv", mime="text/csv")

with tab_how:
    st.markdown("### How it works — calibrated research model")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:12px;padding:14px;text-align:center;">
  <div style="font-size:32px;">🔥</div>
  <div style="font-weight:700;margin:6px 0;">1. Fires → Emissions</div>
  <div style="color:#aaa;font-size:13px;">ESA WorldCover biome (peat/forest) → literature EF_PM2.5 → Q (kg/s)</div>
</div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:12px;padding:14px;text-align:center;">
  <div style="font-size:32px;">💨</div>
  <div style="font-weight:700;margin:6px 0;">2. Trajectory Transport</div>
  <div style="color:#aaa;font-size:13px;">ERA5 winds → 2D trajectory → PBL-capped box model</div>
</div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:12px;padding:14px;text-align:center;">
  <div style="font-size:32px;">🌧️</div>
  <div style="font-weight:700;margin:6px 0;">3. Rain Scavenging</div>
  <div style="color:#aaa;font-size:13px;">Path-integrated Λ=a·P^b along trajectory</div>
</div>
        """, unsafe_allow_html=True)
    with s4:
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:12px;padding:14px;text-align:center;">
  <div style="font-size:32px;">📊</div>
  <div style="font-weight:700;margin:6px 0;">4. Bayesian Attribution</div>
  <div style="color:#aaa;font-size:13px;">Importance sampling + spatial covariance → posterior fire probabilities</div>
</div>
        """, unsafe_allow_html=True)
    
    st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:10px;padding:12px 14px;margin-top:14px;">
<b>What is API?</b> Malaysian Air Pollutant Index — 0-500 scale from PM2.5:<br>
<span style="background:#00b050;color:white;padding:2px 7px;border-radius:4px;">0-50 Good</span> →
<span style="background:#ffeb3b;color:#111;padding:2px 7px;border-radius:4px;">51-100 Moderate</span> →
<span style="background:#ff9800;color:white;padding:2px 7px;border-radius:4px;">101-200 Unhealthy</span> →
<span style="background:#f44336;color:white;padding:2px 7px;border-radius:4px;">201-300 Very Unhealthy</span>. Haze = Unhealthy+.
</div>
    """, unsafe_allow_html=True)
    
    st.markdown("#### Data sources in this dashboard")
    st.markdown("""
- **Predicted**: Calibrated model (biome emissions + ERA5 trajectory + PBL + rain scavenging + urban baseline)
- **Reference (CAMS)**: Open-Meteo CAMS EAC4 reanalysis — ECMWF model output, **not ground sensors**
- **Ground truth (IQAir/DOE)**: Actual station measurements — often lower than CAMS model
- **CAMS model typically reads 30-100% higher** than ground sensors in SE Asia due to model resolution and lack of local deposition
    """)
    
    st.markdown("#### Why this model is different")
    st.markdown("""
- **Biome-specific emissions**: Peat fires (smoldering) emit 3× more PM2.5 per MW than forest fires (flaming)
- **Trajectory transport**: Follows actual wind curves, not straight-line Gaussian
- **PBL capping**: Daytime mixing to 1500m, nighttime to 300m — changes concentrations 5×
- **Path-integrated rain**: Rain along entire trajectory, not just at source
- **Urban baseline**: CAMS-derived diurnal curve separates local traffic from transboundary smoke
- **Continuous inverse**: 2000 importance samples with spatial error covariance, not discrete FRP weighting
- **Proxy validated**: Against CAMS EAC4 Sep 2019 haze reanalysis
    """)
    
    with st.expander("For experts — formulas", expanded=False):
        st.latex(r"Q_i = C_f \cdot \frac{\text{FRP}_i}{1-\alpha_{cloud}} e^{\tau_{canopy}} \cdot EF_{PM2.5,biome} \cdot 10^{-3}")
        st.latex(r"\frac{d\mathbf{x}}{dt} = \mathbf{u}(\mathbf{x},t), \quad \mathbf{x}(0) = \mathbf{x}_{fire}")
        st.latex(r"C = \frac{Q}{\sqrt{2\pi} U \sigma_y H_{pbl}} \exp\left(-\frac{y^2}{2\sigma_y^2}\right) \exp\left(-\int_0^T \Lambda(t) dt\right) \times 10^6")
        st.latex(r"\Lambda(t) = a P(t)^b, \quad a=10^{-4}, b=0.8")
        st.latex(r"p(\mathbf{Q}|\mathbf{d}) \propto \exp\left(-\frac{1}{2}(\mathbf{d}-\mathbf{H}\mathbf{Q})^T \mathbf{R}^{-1}(\mathbf{d}-\mathbf{H}\mathbf{Q})\right) \prod_i \text{LogNormal}(Q_i|Q_{init,i},\sigma_{prior})")
        st.caption("H = transport matrix (N_fires × N_receptors), R = spatial covariance matrix.")

# Footer
st.success(f"Done — {len(fire_posteriors)} fires → {len(city_forecast)} cities → {len(haze_grid)} map points. Proxy-calibrated model.")
st.caption("Tip: Use sidebar scenarios to test sensitivity — increase rain, shift wind, scale urban baseline. Data refreshes hourly.")

# WAQI Attribution (required by terms)
st.caption("""
**Air quality data sources:** 
- Fire hotspots: NASA FIRMS
- Meteorology: Open-Meteo ERA5
- Model validation (CAMS): Open-Meteo CAMS EAC4 reanalysis
- **Ground stations (WAQI):** World Air Quality Index Project (waqi.info) & Malaysia Department of Environment (DOE)
Attribution required per WAQI terms of use.
""")