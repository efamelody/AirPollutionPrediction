# Transboundary Haze Attribution — Calibrated Research Model

> **Proxy-calibrated research model** for transboundary haze attribution from Indonesian fires to Malaysian cities.
> Uses biome-differentiated emissions, trajectory transport with ERA5 PBL, path-integrated scavenging, urban baseline separation, and continuous Bayesian inverse attribution.
> Validated against CAMS EAC4 reanalysis (proxy validation).

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                        TRANSBOUNDARY HAZE ATTRIBUTION PIPELINE                       │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│  NASA FIRMS VIIRS                    ESA WorldCover 10m                              │
│  ┌──────────────┐                    ┌──────────────┐                                │
│  │ Hotspots     │───────────────────▶│ Biome        │                                │
│  │ (lat,lon,FRP)│                    │ Classification│                                │
│  └──────────────┘                    └──────┬───────┘                                │
│        │                                     │                                        │
│        ▼                                     ▼                                        │
│  ┌──────────────────────────────────────────────────────────────┐                     │
│  │                    EMISSION ENGINE                            │                     │
│  │  Q = C_f × FRP_adj × EF_PM2.5 × 10⁻³   (kg/s)               │                     │
│  │  Peat:      C_f=0.52, EF=28.0  → Q ≈ 0.0146 × FRP            │                     │
│  │  Forest:    C_f=0.36, EF=9.1   → Q ≈ 0.0033 × FRP            │                     │
│  │  Cropland:  C_f=0.45, EF=7.8   → Q ≈ 0.0035 × FRP            │                     │
│  └──────────────────────────────┬────────────────────────────────┘                     │
│                                 │                                                      │
│                                 ▼                                                      │
│  ┌──────────────────────────────────────────────────────────────┐                     │
│  │                    TRAJECTORY TRANSPORT                       │                     │
│  │  dx/dt = u(x,t), dy/dt = v(x,t)    (ERA5 hourly winds)      │                     │
│  │  PBL height: H_pbl(t) = 200-1500m (diurnal cycle)            │                     │
│  │  Box model: C = Q/(√2π U σ_y H_pbl) × exp(-y²/2σ_y²)         │                     │
│  │  Scavenging: exp(-∫ Λ dt),  Λ = 10⁻⁴ P^0.8                   │                     │
│  └──────────────────────────────┬────────────────────────────────┘                     │
│                                 │                                                      │
│                    ┌────────────┴────────────┐                                          │
│                    ▼                         ▼                                          │
│  ┌─────────────────────────┐   ┌─────────────────────────┐                            │
│  │    URBAN BASELINE       │   │  BAYESIAN INVERSE       │                            │
│  │  CAMS non-haze months   │   │  2000 importance samples │                            │
│  │  C_urban(t) = base +    │   │  Log-normal prior        │                            │
│  │    amp·cos(2π(t-peak)/24)│   │  Spatial covariance R    │                            │
│  │  Regional: 12 µg/m³     │   │  Posterior P(fire|data)  │                            │
│  └───────────┬─────────────┘   └───────────┬──────────────┘                            │
│              │                             │                                            │
│              ▼                             ▼                                            │
│  ┌──────────────────────────────────────────────────────────────┐                     │
│  │                    TOTAL PM2.5 PREDICTION                     │                     │
│  │  C_total = C_smoke + C_urban + C_background                  │                     │
│  │  API = DOE breakpoints (0-500 scale)                         │                     │
│  └──────────────────────────────────────────────────────────────┘                     │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Module Details

### 1. Emission Engine (`src/data/landcover.py`, `src/physics/calibration.py`)

**Biome Classification** (ESA WorldCover 10m via WMS + heuristic fallback):
| ESA Class | Biome | C_f (kg/MJ) | EF_PM2.5 (g/kg) | Q_factor (kg/s per MW) |
|-----------|-------|-------------|-----------------|------------------------|
| 90 (Wetland) | Peatland | 0.52 | 28.0 | 0.0146 |
| 10 (Tree cover) | Tropical Forest | 0.36 | 9.1 | 0.0033 |
| 40 (Cropland) | Oil Palm/Cropland | 0.45 | 7.8 | 0.0035 |

**Emission Calculation**:
```
FRP_adj = FRP_obs / ((1 - α_cloud) × exp(-τ_canopy))
Q = C_f × FRP_adj × EF_PM2.5 × 10⁻³  [kg/s]
```
- α_cloud = 0.15 (cloud cover fraction)
- τ_canopy = 0.1 (peat) to 0.8 (forest)

### 2. Trajectory Transport (`src/physics/calibration.py`)

**2D Trajectory Integration**:
```
dx/dt = u(x,y,t),  dy/dt = v(x,y,t)     [ERA5 hourly u/v at 0.25°]
x(t+Δt) = x(t) + u × 3600 × Δt / (R × cos(lat))
y(t+Δt) = y(t) + v × 3600 × Δt / R
```

**PBL-Capped Box Model**:
```
σ_y(x) = 0.11 × x × (1 + 0.0001×x)⁻⁰·⁵     [crosswind spread]
C(x,y) = Q / (√(2π) × U × σ_y × H_pbl) × exp(-y²/(2σ_y²)) × Retention × 10⁶
```
- H_pbl: 200m (night) → 1500m (day) from diurnal model
- U: mean wind speed along trajectory
- y: crosswind distance from trajectory to receptor

**Path-Integrated Wet Scavenging**:
```
Λ(t) = a × P(t)^b     [a=10⁻⁴, b=0.8, P in mm/h]
Retention = exp(-∫₀ᵀ Λ(t) dt) ≈ exp(-Σ Λ_i × Δt)
```
Integrated along full trajectory, not just at source.

### 3. Urban Baseline (`src/data/cams.py`)

**CAMS Non-Haze Months** (Nov–Apr, NE Monsoon):
- Extract hourly PM2.5 from CAMS EAC4 via Open-Meteo API
- Compute diurnal median per city → 24-hour curve

**Parametric Fallback** (double-peak diurnal):
```
C_urban(t) = base + morning_amp × exp(-0.5×((t-morning_peak)/width)²) 
                  + evening_amp × exp(-0.5×((t-evening_peak)/width)²)
```
| City | Base | Morning Amp | Morning Peak | Evening Amp | Evening Peak |
|------|------|-------------|--------------|-------------|--------------|
| Kuala Lumpur | 12 | 12 | 08:00 | 10 | 20:00 |
| Johor Bahru | 10 | 10 | 08:00 | 8 | 20:00 |
| Kuching | 8 | 8 | 07:00 | 6 | 19:00 |
| Ipoh | 9 | 9 | 08:00 | 7 | 20:00 |
| Kota Bharu | 7 | 6 | 07:00 | 5 | 19:00 |

**Regional Background**: 12 µg/m³ (maritime clean air from CAMS)

### 4. Bayesian Inverse Attribution (`src/physics/calibration.py`)

**State Vector**: Q = [Q₁, Q₂, ..., Qₙ] (emission rates for N fires)

**Prior**: Log-normal constrained by FRP
```
ln(Q_i) ~ N(ln(Q_init_i), σ_prior²)    σ_prior = 0.7
```

**Likelihood** with Spatial Covariance:
```
d_obs = observed PM2.5 at K receptors (5 cities)
C_pred = H × Q    [H: transport matrix, K×N]
R_jk = σ_inst² × exp(-d_jk / L_corr) + δ_jk × σ_inst²
     [L_corr = 50 km, σ_inst = 5 µg/m³]

log p(d|Q) = -0.5 × (d - C_pred)ᵀ R⁻¹ (d - C_pred) - 0.5 log|R|
```

**Importance Sampling** (2000 samples, vectorized):
```
1. Draw Q_samples ~ Prior
2. C_pred = Q_samples @ H.T
3. weight ∝ exp(log_likelihood + log_prior)
4. Normalize weights
5. Posterior P(fire_i major) = mean(Q_i > 2 × median(Q_i))
```

### 5. Total Concentration & API

```
C_total(city, hour) = C_background + C_urban(city, hour) + Σ_fires C_smoke(fire, city, hour)

API = DOE piecewise linear from PM2.5:
  0-12 µg/m³    → API 0-50     (Good)
  12.1-35.4    → API 51-100   (Moderate)
  35.5-55.4    → API 101-150  (Unhealthy)
  55.5-150.4   → API 151-200  (Very Unhealthy)
  150.5-250.4  → API 201-300  (Hazardous)
  ...
```

---

## Data Flow Diagram

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│  NASA FIRMS │     │ ESA World   │     │ Open-Meteo  │
│  Hotspots   │     │ Cover       │     │ ERA5        │
│  (lat,lon,  │     │ Biome Map   │     │ (u,v,P,     │
│   FRP,conf) │     │ (WMS/GeoTIFF)│    │  PBL model) │
└──────┬──────┘     └──────┬──────┘     └──────┬──────┘
       │                   │                   │
       ▼                   ▼                   ▼
┌─────────────────────────────────────────────────────┐
│              EMISSION ENGINE                         │
│  Q_i = biome_emission_factor(biome_i) × FRP_adj_i   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│           TRAJECTORY TRANSPORT                       │
│  For each fire: integrate dx/dt = u(x,t)             │
│  → trajectory(t), wind(t), precip(t), PBL(t)        │
│  → C_smoke(fire, city, hour) via box model          │
└────────────────────────┬────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Urban      │  │  Inverse     │  │  Validation  │
│   Baseline   │  │  Attribution │  │  (CAMS proxy)│
│  C_urban(t)  │  │  P(Q|data)   │  │  RMSE, NMB   │
└──────┬───────┘  └──────┬───────┘  └──────┬───────┘
       │                 │                 │
       └─────────────────┼─────────────────┘
                         ▼
┌─────────────────────────────────────────────────────┐
│            PREDICTION OUTPUT                         │
│  • City forecast: PM2.5, API, category (24h)        │
│  • Fire posteriors: biome, Q, posterior probability │
│  • Haze grid: spatial map for visualization         │
│  • Metrics: proxy RMSE vs CAMS reanalysis           │
└─────────────────────────────────────────────────────┘
```

---

## Dashboard Tabs

| Tab | Purpose |
|-----|---------|
| **🌫️ Prediction** | Haze map + city cards (calibrated PM2.5/API vs CAMS model) |
| **🔥 Evidence** | Fire posterior probabilities + ERA5 wind/rain + transport |
| **📈 Forecast** | 24-hour hourly PM2.5 forecast per city with API thresholds |
| **❓ How it works** | Plain-language + formulas for each module |

---

## Key Differences from Original Model

| Component | Original (Phase 1) | Calibrated (Phase 2) |
|-----------|-------------------|---------------------|
| **Emissions** | Q = 0.02 × FRP (single) | Biome-specific: peat 0.0146, forest 0.0033 |
| **Transport** | Gaussian plume + heuristic | ERA5 trajectory + PBL box model |
| **PBL Height** | Fixed 1000m | Diurnal 200-1500m |
| **Scavenging** | Point precipitation | Path-integrated along trajectory |
| **Baseline** | Fixed 8 µg/m³ | CAMS diurnal (12-37 µg/m³) + regional 12 |
| **Inverse** | Discrete FRP-weighted | Continuous importance sampling + spatial R |
| **Validation** | None | Proxy RMSE/NMB vs CAMS Sep 2019 |

---

## Running the Model

```bash
# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run app.py
# http://localhost:8501

# Run tests
python -m pytest tests/ -v
```

**Sidebar Controls**:
- NASA FIRMS key (DEMO_KEY for synthetic fires)
- Fire history: 1-5 days
- Scenario sliders: rain multiplier, wind shift, urban baseline scale
- Live CAMS toggle

---

## Proxy Validation

Validated against **CAMS EAC4 Sep 2019 haze reanalysis** (satellite-assimilated):
- Metrics: RMSE, Normalized Mean Bias (NMB), Correlation (R)
- Shown in sidebar as "Proxy Validation"
- Note: CAMS model ≠ ground sensors (typically 30-100% higher in SE Asia)

---

## File Structure

```
src/
├── config.py                 # All calibrated parameters
├── data/
│   ├── firms.py              # FIRMS fetch + biome classification
│   ├── landcover.py          # ESA WorldCover biome lookup
│   ├── cams.py               # Urban baseline from CAMS non-haze months
│   ├── era5.py               # Gridded ERA5 wind/precip + PBL model
│   ├── meteo_grid.py         # Interface for gridded meteo
│   └── receptors.py          # City locations + live CAMS fetch
├── physics/
│   ├── api.py                # PM2.5 → Malaysian API conversion
│   ├── calibration.py        # Core pipeline: emissions→transport→baseline→inverse
│   ├── research_model.py     # Main entry: predict_haze(), predict_haze_forecast()
│   └── [legacy: dispersion.py, inversion.py, forecast_grid.py]
├── viz/
│   └── maps.py               # PyDeck visualization
└── app.py                    # Streamlit dashboard
```

---

## What to Show Your Supervisor

1. **Architecture diagram** (above) — complete physics-based pipeline
2. **Proxy calibration methodology** — CAMS non-haze baseline + ESA biome emissions + ERA5 transport
3. **Validation results** — RMSE/NMB/R vs CAMS Sep 2019 reanalysis (sidebar)
4. **Attribution output** — posterior probabilities per fire with biome classification
5. **Scenario testing** — real-time sliders for rain/wind/urban sensitivity
6. **Code transparency** — every formula in `calibration.py` and `research_model.py`

---

## Known Limitations

1. **CAMS reference ≠ ground truth** — model typically 30-100% higher than DOE/IQAir sensors
2. **PBL model** — simplified diurnal cycle, not full ERA5 PBL (API limitation)
3. **ESA WMS** — online queries; fallback heuristic for peat regions
4. **No chemical aging** — PM2.5 treated as inert tracer
5. **2D transport** — no vertical wind shear or layer-resolved advection
6. **Prior uncertainty** — σ_prior=0.7 chosen heuristically
7. **Scavenging time units** — Lambda = a·Pᵇ with a=1e-4 gives Lambda in h⁻¹ for P in mm/h. Code uses hourly timesteps (dt_hours) so exponent Lambda·dt is dimensionless. Verified against CAMS Sep 2019 proxy.
8. **Covariance nugget** — Added 1e-6 diagonal jitter to R before inversion to prevent numerical instability when stations are geographically close.

---

## Next Research Steps

1. **Calibrate σ_prior, L_corr** via cross-validation on historical episodes
2. **Integrate ERA5 PBL directly** via CDS API (requires account)
3. **Add HYSPLIT back-trajectories** for transport validation
4. **Incorporate DOE ground data** when available for true validation
5. **Chemical aging module** for secondary organic aerosol formation
6. **Ensemble forecasts** using ECMWF ensemble members

---

## License & Citation

Research code for academic use. If used in publications, please cite the methodology:
> "Proxy-calibrated transboundary haze attribution using biome-differentiated emissions, ERA5 trajectory transport, and continuous Bayesian inverse modeling validated against CAMS EAC4 reanalysis."