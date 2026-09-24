# Transboundary Haze Prediction — Indonesian Fires → Malaysian API

> **Research demonstrator, not an operational forecaster.**  
> Question: *Will my Malaysian city have haze (API ≥100) tomorrow given fires in Indonesia, wind, and rain?*  
> This repo builds the data pipeline + physics + Bayesian attribution as a Streamlit dashboard so you can see fires, wind, rain, and predicted API together — then refine the science.

Live demo: `https://<you>-airpollutionprediction-xxxx.streamlit.app` (auto-deploys on `git push` to `main`).

---

## 1) What we actually built — honest summary

| Component | Status | What it does | What it *doesn't* do (your research gap) |
|-----------|--------|--------------|------------------------------------------|
| **Fire ingestion** `src/data/firms.py:18` | Works | NASA FIRMS VIIRS 375m hotspots over Indonesia (lat, lon, FRP, confidence). Filters `confidence≥70`. Falls back to 5 synthetic Sumatra/Kalimantan fires when `DEMO_KEY`. | No cloud-cover correction, no peat vs forest FRP→Q conversion, no hotspot clustering. Needs emission inventory calibration. |
| **Weather (wind + rain)** `src/data/meteo.py:8`, `src/data/meteo_grid.py:28` | Works but crude | Open-Meteo ECMWF/GFS `wind_speed_10m, wind_direction_10m, precipitation` at Strait of Malacca centroid + 4×6 grid for arrows. Converts to `U, θ, Λ`. | **Single centroid wind** for whole domain — real transboundary needs gridded 3D NWP (ECMWF 0.1° + PBL height, stability class). No vertical wind shear. |
| **Air quality (ground truth)** `src/data/receptors.py:14` | Live via Open-Meteo CAMS, fallback mock | Tries `air-quality-api.open-meteo.com/v1/air-quality?hourly=pm2_5` for KL/JB/Kuching/Ipoh/KB. If fails (3+ stations must succeed), uses mock haze-event `85.4/62.1/112.0 µg` (API 160+). Toggle in sidebar. | **No direct DOE APIMS ingestion** — EQMS `api3/publicmapproxy` is proxied ArcGIS with no documented hourly JSON. DOE real-time requires scraping or JAS API token. OpenDOSM `monthly air_pollution` is daily aggregates, not hourly API. |
| **Physics — plume** `src/physics/dispersion.py:5`, `src/physics/forecast_grid.py:49` | Demonstrator — pattern correct, magnitude **uncalibrated** | Steady-state Gaussian + exponential wet scavenging: `C = Q/(πUσyσz)·exp(-y²/2σy²)·exp(-h²/2σz²)·exp(-Λx/U)·1e6`, `Λ=1e-4·P^0.8`, `Q=0.02·FRP`, plus empirical long-range heuristic `C_far = FRP·3.2·exp(-x/380km)·exp(-y²/2σc²)` with `σc=0.42x+18km` so plume reaches 300–600km. Rain kills plume (light rain 0.5 mm/h → 97% removal over 300km at 4.5 m/s). | **Gaussian valid ~1–20km**, not 300–800km maritime. Real system needs Lagrangian (HYSPLIT) or Eulerian (WRF-Chem/CMAQ) with 3D advection-diffusion, PBL entrainment, chemical aging. `Q=0.02·FRP` and heuristic `×3.2` are **tuned for visibility**, not fitted to DOE history. `L=380km` chosen so Sumatra→JB shows haze. Background fixed 8 µg. |
| **PM2.5 → API** `src/physics/api.py:10` | Correct | DOE breakpoint piecewise linear (US-EPA PM2.5 → API 0–500). API ≥100 = Unhealthy = haze. Used for city cards and map colours. | DOE API is max over 5 pollutants (PM2.5, PM10, O3, SO2, NO2, CO). We use PM2.5 only — true API needs all. |
| **Inverse — which fire is culprit?** `src/physics/inversion.py:14` | Works (discrete) | `prior ∝ FRP`, likelihood `Π Normal(d_k| C_k(θ), σ=15)`, posterior `∝ prior·likelihood`, log-sum-exp normalized. Exhaustive over N hotspots, `O(N·K)` <1s. | Discrete grid only (not continuous `x_s,y_s,Q`). No MCMC/Gelman-Rubin, diagonal `R`, no spatial correlation. High-dimensional `M×forward` intractable without PINN surrogate. |
| **Forecast grid + map** `src/physics/forecast_grid.py:13`, `src/viz/maps.py:28` | Works | Sums plumes over 18×22 grid across Malaysia (excess only `>0.8 µg` so background doesn't paint map). Wind LineLayer + rain Scatterplot + city Columns. | No time stepping — assumes steady wind over forecast horizon. No accumulation over hours/days. No trajectory. |
| **Dashboard** `app.py:180` | Layman-ready | Tabs: Prediction (haze cloud + city API) vs Evidence (fires+wind+rain) vs How-it-works (plain language). Plain cards separate *Predicted fire smoke* vs *Measured now (DOE-like)* and warn on mismatch. | Predictions suck by design — see below. |

**Bottom line:** The AI-built pipeline is *correct in structure* (fires → wind/rain → plume → API → Bayes) but **wrong in calibration**. That's expected — the science you need to refine is the calibration, not the code.

---

## 2) End-to-end flow (the formula you need to refine)

```
NASA FIRMS (lat,lon,FRP) ─┐
                           ├─→ Q = 0.02·FRP  (your first calibration: FRP→kg/s via emission factor, fuel load, peat vs forest)
Open-Meteo (U,θ,P) ────────┤     Λ = 1e-4·P^0.8 ,  t = x/U
                           ├─→ C(x,y) = plume(Q,U,θ,Λ, x,y)  (your main physics: replace Gaussian with HYSPLIT/WRF)
                           │     Total PM2.5 = background + Σ fires C
                           ├─→ API = breakpoint(PM2.5)  (DOE table, src/physics/api.py:7)
                           └─→ p(θ|d) ∝ Normal(d | f(θ),R)·(FRP/ΣFRP)  (your inference: which fire explains city PM2.5?)
                                          ↑ d = measured PM2.5 at KL/JB/Kuching/Ipoh/KB
```

**Concrete example (how to audit):**  
Fire `-1.23,103.81 FRP 89 MW` → `Q=1.78 kg/s`, wind `5 m/s FROM 225° (blows NE)`, rain `0.6 mm/h` → to JB `1.49,103.74` `x≈ 280km` → `C_far ≈ 89·3.2·exp(-280/380)=89·3.2·0.48=136·cross(~0.7)·scav(0.3)·wind(0.7)=20 µg` at amp 80 → `Total 28 µg → API 83 Moderate`. Make it dry `P=0` + `L=380` + amp 80 → `~70 µg → API 158 Unhealthy`. That's how wind/rain move the API over the 100 threshold.

---

## 3) Data sources — where we get each number

| Data | Endpoint | Fields | Frequency | File |
|------|----------|--------|-----------|------|
| Fires | `https://firms.modaps.eosdis.nasa.gov/api/country/csv/{MAP_KEY}/VIIRS_SNPP_NRT/IDN/{days}` | `latitude, longitude, frp, confidence` | 3h swath | `src/data/firms.py:18` |
| Weather | `https://api.open-meteo.com/v1/forecast?latitude=2.5&longitude=101.5&hourly=wind_speed_10m,wind_direction_10m,precipitation` | `wind_speed_10m (km/h→m/s), wind_direction_10m (deg FROM), precipitation (mm/h)` | Hourly | `src/data/meteo.py:8` |
| Weather grid | Same ×24 (4×6) | `u=-U sinθ, v=-U cosθ` for arrows | Hourly | `src/data/meteo_grid.py:28` |
| Air quality (live) | `https://air-quality-api.open-meteo.com/v1/air-quality?latitude={lat}&longitude={lon}&hourly=pm2_5` (CAMS) | `pm2_5 (µg/m³)` | Hourly | `src/data/receptors.py:14` |
| Air quality (DOE truth you check) | `https://eqms.doe.gov.my/APIMS/main` (APIMS) — **no documented hourly JSON**; proxied `api3/publicmapproxy` (ArcGIS). Our mock `85.4,62.1,112.0,45.8,38.2` is a synthetic haze-event when live fails. | `API` (DOE, 0–500) | Hourly | — |
| Constants | — | `a=1e-4, b=0.8, h=30m, R=6371000m, σy=0.11x(1+0.0001x)^-0.5` | — | `src/config.py:3` |

**Key mismatch you saw:** DOE APIMS live KL often `API 65–90 Moderate` (from Open-Meteo CAMS `71 µg → API 159` in our live fetch, or DOE's own `70`), while Streamlit predicted `API 34 Good`. That's **smoke-only vs total-air**: predicted = Indonesian smoke carried by wind, measured = smoke + KL traffic/factories + local dust. When wind is `138° SE (blows NW away from MY)`, predicted correctly goes to 11 but measured stays 71 — local sources dominate. The card now says this explicitly at `app.py:185`.

---

## 4) Why predictions suck right now — your research tasks

1. **Q is uncalibrated.** `0.02` and heuristic `3.2` were hand-tuned for visibility, not fitted. **Task:** regress `FRP × emission_factor × fuel_load × burn_area` against DOE PM2.5 history (dry season 2019, 2023) to learn `Q(FRP, vegetation, peat depth)`. Try `Q = α·FRP^β` per region.
2. **Gaussian physics wrong at 300–800km.** Steady-state, flat terrain, uniform wind. **Task:** replace with HYSPLIT back-trajectories or WRF-Chem Eulerian. Compare plume vs trajectory for same wind. Keep our `exp(-Λx/U)` but with path-integrated `P` along trajectory, not point `P`.
3. **Single wind.** **Task:** ingest ECMWF 0.1° gridded `u,v` at 10m/100m/850hPa, interpolate to plume path. Validate with `θ=225° SW monsoon (haze season) vs 138° SE (smoke misses MY)` — we saw this live.
4. **Background + local pollution ignored.** Predicted = background 8 + fire smoke, but KL baseline is 20–40 even without fires. **Task:** estimate `background(city, season)` from non-haze months, add traffic/industrial term, or predict *excess* only.
5. **API incomplete.** **Task:** compute true DOE API = max(PM2.5→API, PM10→API, O3→API...). You'll need PM10/CO/O3 from CAMS too.
6. **Inverse too simple.** **Task:** move from discrete `N` hotspots to continuous `θ=[lon,lat,Q]` with MCMC (NUTS) + spatial `R` (Matern), Gelman-Rubin `R̂<1.05`, OSSE validation. Sample thousands, needs PINN surrogate for speed.
7. **No temporal accumulation.** **Task:** forecast `PM2.5(t)` = advection with `τ` lifetime, not instantaneous `C(x)`. Add 24–72h forecast slider driven by Open-Meteo `forecast_days=3`.
8. **Validation absent.** **Task:** hold-out stations, compute `RMSE = sqrt(mean((d - f(θ_MAP))²))` at `app.py:200`, NMB, hit rate for `API≥100`. Compare mock 85/112 haze-event vs live 20/70 baseline. The current error `~60 µg` is your baseline to beat.

---

## 5) How to run and publish

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
# http://localhost:8501
# No key: defaults DEMO_KEY synthetic fires. Live fires: add FIRMS MAP_KEY at https://firms.modaps.eosdis.nasa.gov/api/area/ → sidebar or .streamlit/secrets.toml:
# FIRMS_MAP_KEY = "your_key"
python -m pytest tests -v
```

Publish: `git push` → `share.streamlit.io` → New app → `main` / `app.py` → Settings → Secrets → `FIRMS_MAP_KEY`.

Map is at `src/viz/maps.py:28` (scatter plume excess + wind LineLayer + rain dots). API at `src/physics/api.py:10`. Don't trust absolute numbers until you complete tasks 1–8 — pattern (wind swings plume, rain kills it) is real, magnitude is illustrative.

---

## 6) What to show your supervisor

This repo is **Phase 1 — pipeline demonstrator**: data flows, formulas are transparent, dashboard separates *Prediction (fire smoke)* vs *Evidence (fires+wind+rain)* vs *Measured truth (DOE-like)* so a layman sees `API 34 Predicted Good but Actually 159 Unhealthy → local pollution, wind away`. Your research is Phase 2 — calibration and physics replacement (tasks above). Point them to `app.py:240` expander (formulas) and `src/physics/forecast_grid.py:49` heuristic comment where the tuning lives.

