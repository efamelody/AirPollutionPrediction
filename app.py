import numpy as np
import pandas as pd
import streamlit as st

from src.config import METEO_LAT, METEO_LON, SIGMA_OBS_UGM3
from src.data.firms import fetch_firms_hotspots
from src.data.meteo import fetch_open_meteo_weather
from src.data.meteo_grid import fetch_meteo_grid
from src.data.receptors import fetch_malaysia_receptors
from src.physics.inversion import run_bayesian_inversion
from src.physics.api import pm25_to_api, api_category
from src.physics.forecast_grid import build_haze_grid, predict_for_receptors
from src.viz.maps import build_deck

# ------------------------------------------------------------------------------
# PAGE CONFIG — plain language
# ------------------------------------------------------------------------------
st.set_page_config(page_title="Will My City Have Haze? — Malaysia Haze Forecast", layout="wide", initial_sidebar_state="expanded")

# Helpers for plain wind description
def wind_plain(dir_deg: float) -> str:
    # FROM direction to plain
    dirs = [(0,"North"),(45,"Northeast"),(90,"East"),(135,"Southeast"),(180,"South"),(225,"Southwest"),(270,"West"),(315,"Northwest"),(360,"North")]
    # find closest
    best = min(dirs, key=lambda x: abs(x[0]-dir_deg) if abs(x[0]-dir_deg)<180 else 360-abs(x[0]-dir_deg))
    # also where it blows TO
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
except Exception:
    pass

st.sidebar.markdown("**Fires:** Using demo fires in Sumatra & Kalimantan. Add your NASA key to see live satellite fires.")
firms_api_key = st.sidebar.text_input("NASA FIRMS key (leave DEMO_KEY for demo)", value=default_key, type="password", help="Free at firms.modaps.eosdis.nasa.gov")
selected_days = st.sidebar.slider("Fire history (days)", 1, 5, 1, help="1 day = only today's fires. 5 days = more fires, but older.")

st.sidebar.markdown("### 🌤️ Weather right now")
override_meteo = st.sidebar.checkbox("Try a different wind/rain (what-if?)", value=False)
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
        meteo_data = fetch_open_meteo_weather(METEO_LAT, METEO_LON)

# Live summary in plain language, top of sidebar
st.sidebar.markdown("---")
st.sidebar.markdown(f"""
**Live weather (Strait of Malacca):**<br>
💨 Wind **{meteo_data['wind_speed']:.1f} m/s** {wind_plain(meteo_data['wind_dir'])}<br>
🌧️ Rain **{meteo_data['precipitation']:.1f} mm/h** — {rain_plain(meteo_data['precipitation'])}<br>
<span style="color:#888;font-size:12px;">Source: Open-Meteo (ECMWF/GFS). Rain cleans haze.</span>
""", unsafe_allow_html=True)

st.sidebar.markdown("### 🗺️ What to show on the map")
show_haze = st.sidebar.checkbox("Show predicted haze cloud", value=True, help="Coloured blobs where we predict haze will be")
show_wind = st.sidebar.checkbox("Show wind arrows", value=True, help="Arrows show where smoke is being blown")
show_rain = st.sidebar.checkbox("Show rain dots", value=False, help="Blue dots where it's raining")
show_receptors = st.sidebar.checkbox("Show city towers", value=True)

haze_mode = "Forecast (what we predict)"
background_pm25 = 8.0
grid_density = "Medium"
weight_opt = "Smart average (uses evidence)"
amplification = 80
sigma_obs = float(SIGMA_OBS_UGM3)
with st.sidebar.expander("🔧 Advanced (for experts)", expanded=False):
    haze_mode = st.radio("City towers show", ["Forecast (what we predict)", "Observed now (what was measured)"], index=0, key="haze_mode_adv")
    background_pm25 = st.slider("Clean-air background (µg/m³)", 0.0, 25.0, 8.0, help="Air is never 0 — this is the normal clean value", key="bg_adv")
    grid_density = st.select_slider("Map detail", options=["Coarse", "Medium", "Fine"], value="Medium", key="grid_adv")
    weight_opt = st.radio("How to add fires together", ["Smart average (uses evidence)", "Worst case (add all fires)"], index=0, key="weight_adv")
    amplification = st.slider("Make haze more visible (visual aid)", 1, 200, 80, help="The raw physics gives tiny numbers at 300km. This scales it so you can see the pattern. Keep it at 80 to compare.", key="amp_adv")
    sigma_obs = st.slider("How strict is the source matching?", 5.0, 30.0, float(SIGMA_OBS_UGM3), help="Lower = picky, higher = tolerant", key="sigma_adv")

density_map = {"Coarse": (10, 14), "Medium": (18, 22), "Fine": (26, 30)}
n_lat_haze, n_lon_haze = density_map[grid_density]
weight_by_posterior = weight_opt.startswith("Smart")

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

with st.spinner("Checking which fires match the pollution in cities..."):
    hotspots_analyzed = run_bayesian_inversion(hotspots_raw, receptors_df, meteo_data, sigma_obs=sigma_obs)

with st.spinner("Predicting haze for each city..."):
    haze_grid = build_haze_grid(hotspots_analyzed, meteo_data, n_lat=n_lat_haze, n_lon=n_lon_haze, background_pm25=background_pm25, weight_by_posterior=weight_by_posterior, amplification=float(amplification))
    city_forecast = predict_for_receptors(hotspots_analyzed, receptors_df, meteo_data, background_pm25=background_pm25, weight_by_posterior=weight_by_posterior, amplification=float(amplification))

meteo_grid = cached_meteo_grid(4, 6, override_meteo, float(meteo_data["wind_speed"]), float(meteo_data["wind_dir"]), float(meteo_data["precipitation"]))

# ------------------------------------------------------------------------------
# PLAIN KPI — what matters to a normal person
# ------------------------------------------------------------------------------
k1, k2, k3 = st.columns(3)
with k1:
    n_fires = len(hotspots_analyzed)
    total_strength = hotspots_analyzed["frp"].sum() if n_fires else 0
    st.metric("🔥 Fires detected", f"{n_fires}", f"Total strength {total_strength:.0f} MW")
    st.caption("From NASA satellites over Indonesia (last {} day(s))".format(selected_days))
with k2:
    st.metric("💨 Wind", f"{meteo_data['wind_speed']:.1f} m/s", wind_plain(meteo_data['wind_dir']))
    st.caption(rain_plain(meteo_data['precipitation']))
with k3:
    n_haze = (city_forecast["api_pred"] >= 101).sum() if not city_forecast.empty else 0
    st.metric("🏙️ Cities with haze forecast", f"{n_haze} of {len(city_forecast)}", f"API ≥100 = haze" if n_haze else "Air looks okay")
    st.caption("Based on fires + wind + rain → API")

# ------------------------------------------------------------------------------
# CITY CARDS — prediction vs truth, plain language
# ------------------------------------------------------------------------------
st.markdown("## 🏙️ Will it be hazy? — City by city")
st.caption("Each card = one Malaysian city. **Left number = our prediction for the next hours.** Right small grey = what was actually measured now (ground truth). If they match, the model is doing well.")
cols = st.columns(len(city_forecast))
for col, (_, row) in zip(cols, city_forecast.iterrows()):
    api = int(row["api_pred"])
    cat, color = api_category(api)
    obs = float(row["pm25_obs"])
    obs_api = pm25_to_api(obs)
    obs_cat, _ = api_category(obs_api)
    # plain advice
    if api >= 201:
        advice = "🔴 Stay indoors, close windows"
        face = "😷"
    elif api >= 101:
        advice = "🟠 Limit outdoor activity"
        face = "😶‍🌫️"
    elif api >= 51:
        advice = "🟡 Sensitive people take care"
        face = "🙂"
    else:
        advice = "🟢 Air is good — no worry"
        face = "😊"
    with col:
        st.markdown(f"""
<div style="padding:12px;border-radius:12px;border:2px solid {color};background:#0f1117;text-align:center;">
  <div style="font-weight:700;font-size:15px;">{row['station_name']}</div>
  <div style="font-size:28px;margin:4px 0;">{face}</div>
  <div style="font-size:22px;font-weight:800;color:{color};">API {api} — {cat}</div>
  <div style="font-size:12px;color:#ccc;">Predicted PM2.5 {row['pm25_pred']:.1f} µg/m³</div>
  <div style="font-size:11px;color:#888;margin-top:6px;border-top:1px solid #2a2a3a;padding-top:6px;">Measured now: {obs:.1f} µg/m³ → API {obs_api} {obs_cat}</div>
  <div style="font-size:11px;color:{color};margin-top:4px;font-weight:600;">{advice}</div>
</div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TABS — separate prediction vs evidence vs explanation
# ------------------------------------------------------------------------------
tab_pred, tab_source, tab_how = st.tabs(["🌫️ Prediction — Where will haze be?", "🔥 Evidence — Where is smoke coming from?", "❓ How it works (simple)"])

with tab_pred:
    st.markdown("### Prediction: where the haze will be")
    st.caption("This is the **forecast** — what we think will happen based on fires + wind + rain. Compare with the **measured now** values on the cards above to see if it matches.")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        try:
            # Prediction map: haze + cities only (no fire clutter) so layman sees just the answer
            deck_pred = build_deck(
                hotspots_analyzed.head(0),  # hide fires here to keep it simple
                city_forecast if haze_mode.startswith("Forecast") else receptors_df,
                haze_grid=haze_grid if show_haze else None,
                meteo_grid=None,  # no wind/rain here — keep prediction clean
                show_haze=show_haze, show_wind=False, show_rain=False, show_receptors=show_receptors,
            )
            st.pydeck_chart(deck_pred, use_container_width=True)
        except Exception as e:
            st.error(f"Map failed: {e}")
        # Simple API scale
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:10px;padding:10px 12px;">
<b>Air quality scale (Malaysian API):</b><br>
<span style="background:#00b050;color:white;padding:2px 7px;border-radius:4px;">Good 0–50</span>
<span style="background:#ffeb3b;color:#111;padding:2px 7px;border-radius:4px;">Moderate 51–100</span>
<span style="background:#ff9800;color:white;padding:2px 7px;border-radius:4px;">Unhealthy 101–200</span>
<span style="background:#f44336;color:white;padding:2px 7px;border-radius:4px;">Very Unhealthy 201–300</span>
<span style="background:#8b0000;color:white;padding:2px 7px;border-radius:4px;">Hazardous 301+</span>
<br><span style="color:#aaa;font-size:12px;">API is calculated from PM2.5. <b>API ≥100 = haze.</b> 3D columns over cities: taller & redder = worse haze.</span>
</div>
        """, unsafe_allow_html=True)
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:10px;padding:10px 12px;margin-top:10px;">
<b>🌫️ Haze cloud</b> = coloured blobs where we predict smoke will be. Colour = how bad (same green→red as cities). We hide the normal 8 µg clean-air background so you only see the <b>extra</b> smoke.<br>
<b>🏙️ City towers</b> = 3D columns over KL, JB, Kuching, Ipoh, Kota Bharu. Height & colour = predicted haze at that city.
</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("#### Prediction vs measured — are we right?")
        st.caption("If the two numbers are close, the forecast matches reality.")
        disp_pred = city_forecast[["station_name", "pm25_obs", "pm25_pred", "api_pred", "category"]].copy()
        disp_pred.columns = ["City", "Measured now", "Predicted", "API", "Level"]
        try:
            st.dataframe(disp_pred.style.format({"Measured now": "{:.1f}", "Predicted": "{:.1f}", "API": "{:.0f}"}).background_gradient(subset=["API"], cmap="Reds"), use_container_width=True)
        except Exception:
            st.dataframe(disp_pred, use_container_width=True)
        # Quick verdict
        avg_err = float((city_forecast["pm25_obs"] - city_forecast["pm25_pred"]).abs().mean()) if not city_forecast.empty else 0
        st.metric("Average error", f"{avg_err:.1f} µg/m³", "lower = better")
        st.caption("Error = difference between predicted and measured. Large error means the model needs better tuning or more accurate fire/wind data.")
        st.download_button("⬇️ Download city forecast CSV", data=city_forecast.to_csv(index=False).encode(), file_name="city_forecast.csv", mime="text/csv")
        st.download_button("⬇️ Download haze map grid CSV", data=haze_grid.to_csv(index=False).encode(), file_name="haze_grid.csv", mime="text/csv")

with tab_source:
    st.markdown("### Evidence: where is the smoke coming from?")
    st.caption("This is **not** the prediction — this is the **proof** we use to make the prediction: satellite fires, wind direction, and rain.")
    c1, c2 = st.columns([2.2, 1])
    with c1:
        st.markdown("**Map: fires + wind + rain + haze together** — see how smoke is carried")
        try:
            deck_source = build_deck(hotspots_analyzed, receptors_df, haze_grid=haze_grid if show_haze else None, meteo_grid=meteo_grid, show_haze=show_haze, show_wind=show_wind, show_rain=show_rain, show_receptors=False)
            st.pydeck_chart(deck_source, use_container_width=True)
        except Exception as e:
            st.error(f"Map failed: {e}")
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:10px;padding:12px 14px;font-size:13px;line-height:1.5;">
  <b>How to read this evidence map:</b><br>
  🔴 <b>Red/orange circles</b> = fires seen from space (NASA FIRMS satellites). <b>Bigger circle = stronger fire = more smoke.</b> Redder = our system thinks this fire is the most likely cause of Malaysia's haze (it best explains the city pollution when wind is considered).<br>
  💨 <b>Grey arrows</b> = wind. Arrow points where smoke is <b>being blown to</b>. Example: arrow pointing northeast means smoke from Sumatra is heading toward KL/JB.<br>
  🌧️ <b>Blue dots</b> = rain. Bigger/blue = heavier rain → smoke is being washed out of the air before it reaches you. Toggle rain on in the sidebar to see.<br>
  🌫️ <b>Faint coloured blobs</b> = same haze forecast as in Prediction tab, shown here so you can see it lines up downwind of the fires.
</div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("#### Which fire is most likely the culprit?")
        st.caption("We compare every fire with the pollution measured in cities, considering wind. The one that best explains the data gets the highest score.")
        disp = hotspots_analyzed[["latitude", "longitude", "frp", "posterior_prob"]].sort_values("posterior_prob", ascending=False)
        disp.columns = ["Lat", "Lon", "Fire strength (MW)", "Likelihood"]
        try:
            st.dataframe(disp.style.format({"Fire strength (MW)": "{:.1f}", "Likelihood": "{:.2%}", "Lat": "{:.3f}", "Lon": "{:.3f}"}).background_gradient(subset=["Likelihood"], cmap="Oranges"), use_container_width=True, height=260)
        except Exception:
            st.dataframe(disp, use_container_width=True)
        if not disp.empty:
            top = disp.iloc[0]
            st.success(f"**Most likely source:** Fire at {top['Lat']:.3f}, {top['Lon']:.3f} — strength {top['Fire strength (MW)']:.1f} MW — **{top['Likelihood']:.0%} likely**")
            st.caption("Likelihood = how well this fire's smoke, carried by today's wind and cleaned by rain, matches the haze measured in Malaysian cities.")
        st.markdown("---")
        st.markdown("**Where data comes from:**")
        st.markdown("- **Fires:** NASA FIRMS (VIIRS satellite, 375m resolution, every 3h)\n- **Wind/Rain:** Open-Meteo (ECMWF/GFS weather models, hourly)\n- **City air:** DOE APIMS / OpenDOSM (ground sensors, PM2.5 µg/m³)")

with tab_how:
    st.markdown("### How it works — in 3 simple steps")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:12px;padding:14px;text-align:center;">
  <div style="font-size:32px;">🔥</div>
  <div style="font-weight:700;margin:6px 0;">1. Fires make smoke</div>
  <div style="color:#aaa;font-size:13px;">Satellites spot hot fires in Indonesia. Hotter fire (higher MW) = more smoke. We turn fire strength into smoke amount.</div>
  <div style="margin-top:8px;color:#888;font-size:12px;">Example: 300 MW fire → lots of smoke</div>
</div>
        """, unsafe_allow_html=True)
    with s2:
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:12px;padding:14px;text-align:center;">
  <div style="font-size:32px;">💨🌧️</div>
  <div style="font-weight:700;margin:6px 0;">2. Wind carries, rain cleans</div>
  <div style="color:#aaa;font-size:13px;">Wind blows smoke toward or away from Malaysia. Rain washes smoke out of the air on the way. No rain + right wind = haze arrives.</div>
  <div style="margin-top:8px;color:#888;font-size:12px;">225° from SW → blows NE toward KL/JB</div>
</div>
        """, unsafe_allow_html=True)
    with s3:
        st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:12px;padding:14px;text-align:center;">
  <div style="font-size:32px;">🏙️📊</div>
  <div style="font-weight:700;margin:6px 0;">3. We predict city air</div>
  <div style="color:#aaa;font-size:13px;">We add smoke from all fires at each Malaysian city, get PM2.5, then convert to Malaysian API. API ≥100 = haze.</div>
  <div style="margin-top:8px;color:#888;font-size:12px;">API 0-50 Good, 101+ Unhealthy</div>
</div>
        """, unsafe_allow_html=True)
    st.markdown("""
<div style="background:#0f1117;border:1px solid #2a2a3a;border-radius:10px;padding:12px 14px;margin-top:14px;">
<b>What is API?</b> Malaysian Air Pollutant Index — the number you hear on the news. It is <b>not</b> PM2.5 directly, but a 0-500 scale from PM2.5:<br>
<span style="background:#00b050;color:white;padding:2px 7px;border-radius:4px;">0-50 Good</span> →
<span style="background:#ffeb3b;color:#111;padding:2px 7px;border-radius:4px;">51-100 Moderate</span> →
<span style="background:#ff9800;color:white;padding:2px 7px;border-radius:4px;">101-200 Unhealthy</span> →
<span style="background:#f44336;color:white;padding:2px 7px;border-radius:4px;">201-300 Very Unhealthy</span>. Haze = Unhealthy and above.
</div>
    """, unsafe_allow_html=True)
    st.markdown("#### Why predictions can be wrong")
    st.caption("Science, simply:")
    st.markdown("- **Distance:** Smoke travels 300-800km over the sea — simple wind arrows are a simplification. Real air has twists and layers.\n- **Unmeasured fires:** Clouds can hide fires from satellites.\n- **Factory/traffic pollution:** Cities also make their own haze, not just Indonesian fires.\n- **Model is unscaled:** We scale haze to be visible — real tuning needs months of DO E history.")
    with st.expander("For experts — show the actual formulas", expanded=False):
        st.latex(r"Q = 0.02 \times \text{FRP},\quad \Lambda = 10^{-4} P^{0.8},\quad \text{scavenging}=e^{-\Lambda x/U}")
        st.latex(r"C = \frac{Q}{\pi U \sigma_y\sigma_z} e^{-y^2/2\sigma_y^2} e^{-h^2/2\sigma_z^2} e^{-\Lambda x/U}\times10^6")
        st.latex(r"p(\theta|d) \propto \mathcal{N}(d|f(\theta),R)\,p(\theta),\; p(\theta)\propto \text{FRP}")
        st.caption("Forecast per grid/city: background (8 µg) + Σ fires C, with long-range heuristic exp(-x/260km) to make plume reach Malaysia. See src/physics/forecast_grid.py:24 and src/physics/dispersion.py:5. RMSE and MAP diagnostics logged in code.")
        st.code(f"Today: wind {meteo_data['wind_speed']:.1f} m/s from {meteo_data['wind_dir']:.0f}°, rain {meteo_data['precipitation']:.1f} mm/h, amplification {amplification}, method {weight_opt}")

# Footer
st.success(f"Done — {len(hotspots_analyzed)} fires → {len(city_forecast)} cities → {len(haze_grid)} map points. Switch tabs above to see prediction vs evidence.")
st.caption("Tip: Use the sidebar to try *what if* — change wind to 90° (easterly) and see haze blow away from Malaysia, or add heavy rain and watch it fade. Data auto-refreshes every 30-60 min.")
