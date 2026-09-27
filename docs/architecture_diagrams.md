# Transboundary Haze Attribution — Architecture Diagrams

This document contains Mermaid diagrams for the calibrated haze attribution model.

---

## 1. Full Pipeline Architecture

```mermaid
flowchart TD
    subgraph Data["Data Sources"]
        FIRMS["NASA FIRMS VIIRS\nHotspots (lat,lon,FRP,conf)"]
        ESA["ESA WorldCover 10m\nBiome Classification"]
        ERA5["Open-Meteo ERA5\nHourly u,v,P + PBL Model"]
        CAMS["Open-Meteo CAMS\nNon-haze months (Nov-Apr)"]
    end

    subgraph Emission["Emission Engine"]
        BiomeLookup["Biome Lookup\n(WMS + Heuristic)"]
        QCalc["Q = C_f × FRP_adj × EF_PM2.5 × 10⁻³\nPeat: 0.0146 | Forest: 0.0033 | Crop: 0.0035"]
    end

    subgraph Transport["Trajectory Transport"]
        TrajInt["Trajectory Integration\ndx/dt = u(x,t), dy/dt = v(x,t)"]
        PBL["PBL Height Model\n200m (night) → 1500m (day)"]
        BoxModel["Box Model\nC = Q/(√2π U σ_y H_pbl) × exp(-y²/2σ_y²)"]
        Scavenging["Path Scavenging\nRetention = exp(-∫ Λ dt), Λ = 10⁻⁴ P^0.8"]
    end

    subgraph Baseline["Urban Baseline"]
        CAMSBaseline["CAMS Diurnal Curves\nPer city, 24h median"]
        Parametric["Parametric Fallback\nDouble-peak (morning/evening)"]
        Regional["Regional Background\n12 µg/m³"]
    end

    subgraph Inverse["Bayesian Inverse"]
        Prior["Log-Normal Prior\nln(Q) ~ N(ln(Q_init), 0.7²)"]
        Likelihood["Spatial Likelihood\nR_jk = σ²exp(-d/L) + δσ²"]
        ImportanceSampling["2000 Importance Samples\nVectorized: Q_samples @ H.T"]
        Posterior["Posterior P(fire|data)\nMean(Q > 2×median)"]
    end

    subgraph Output["Prediction Output"]
        CityForecast["City Forecast\nPM2.5, API, Category (24h)"]
        FirePosteriors["Fire Posteriors\nBiome, Q, Probability"]
        HazeGrid["Spatial Haze Grid\nFor Map Visualization"]
        Metrics["Proxy Validation\nRMSE, NMB, R vs CAMS"]
    end

    FIRMS --> BiomeLookup
    ESA --> BiomeLookup
    BiomeLookup --> QCalc
    QCalc --> TrajInt
    ERA5 --> TrajInt
    TrajInt --> PBL
    TrajInt --> BoxModel
    TrajInt --> Scavenging
    PBL --> BoxModel
    Scavenging --> BoxModel
    BoxModel --> CityForecast
    CAMS --> CAMSBaseline
    CAMSBaseline --> Parametric
    Parametric --> Regional
    Regional --> CityForecast
    Baseline --> CityForecast
    Transport --> Inverse
    Baseline --> Inverse
    Inverse --> FirePosteriors
    Inverse --> CityForecast
    Output --> Metrics
```

---

## 2. Emission Calculation Flow

```mermaid
flowchart LR
    FRP["FRP_obs (MW)"] --> CloudAdj["Cloud Attenuation\nFRP_adj = FRP/(1-α)e^{-τ}"]
    ESA["ESA Class → Biome"] --> EmissionParams["C_f, EF_PM2.5, τ"]
    CloudAdj --> QCalc["Q = C_f × FRP_adj × EF × 10⁻³"]
    EmissionParams --> QCalc
    QCalc --> QInit["Q_init (kg/s)"]
```

---

## 3. Trajectory Transport Detail

```mermaid
flowchart TD
    Start["Fire Source\n(lat₀, lon₀)"] --> Loop["For each hour t=0..T"]
    Loop --> Interp["Interpolate u,v,P,PBL\nat (lat,lon,t)"]
    Interp --> WindCheck{"|wind| > 0.5 m/s?"}
    WindCheck -- No --> Stop["Stop trajectory"]
    WindCheck -- Yes --> Step["Δlat, Δlon from u,v"]
    Step --> Update["lat += Δlat, lon += Δlon"]
    Update --> Record["Record: traj, ws, precip, PBL"]
    Update --> ScavAcc["Λ += a·P^b × dt"]
    Record --> Loop
    Stop --> Output["Trajectory + Integrated\nscavenging + PBL heights"]
    Output --> Concentration["Compute C at Receptors\nBox model + crosswind decay"]
```

---

## 4. Bayesian Inverse Attribution

```mermaid
flowchart TD
    QInit["Q_init from Emissions\n(N fires)"] --> Prior["Log-Normal Prior\nln(Q) ~ N(ln(Q_init), σ²)"]
    Prior --> Sample["Draw 2000 Samples\nQ_samples ~ Prior"]
    Transport["Transport Matrix H\n(N_fires × N_receptors)"] --> Forward["C_pred = Q_samples @ H.T\n(2000 × N_receptors)"]
    Obs["Observed PM2.5\n(N_receptors)"] --> Likelihood["Spatial Likelihood\nR = σ²exp(-d/L) + δσ²"]
    Forward --> Likelihood
    Likelihood --> Weights["Weights = exp(log_lik + log_prior)"]
    Weights --> Normalize["Normalize weights"]
    Normalize --> Posterior["Posterior Prob\nP(fire major) = mean(Q > 2×median)"]
```

---

## 5. Data Source Summary

```mermaid
graph LR
    subgraph Real["Real Observations"]
        FIRMS["NASA FIRMS\nVIIRS 375m, 3h"]
        CAMS_Obs["Open-Meteo CAMS\nPM2.5 (model)"]
        ERA5_Real["ERA5 / Open-Meteo\nu,v,P (forecast)"]
    end

    subgraph Static["Static / Model"]
        ESA_Static["ESA WorldCover\n10m Land Cover"]
        Lit_Params["Literature Emission\nFactors (Akagi 2011)"]
        PBL_Model["Diurnal PBL Model\n(not direct ERA5 PBL)"]
    end

    subgraph Proxy["Proxy Validation"]
        CAMS_Val["CAMS EAC4\nSep 2019 Reanalysis"]
    end

    FIRMS --> Emission["Emission Engine"]
    ESA_Static --> Emission
    Lit_Params --> Emission
    ERA5_Real --> Transport["Trajectory Transport"]
    PBL_Model --> Transport
    CAMS_Obs --> Baseline["Urban Baseline"]
    Emission --> Transport
    Transport --> Inverse["Bayesian Inverse"]
    Baseline --> Inverse
    Inverse --> Output["Predictions"]
    CAMS_Val --> Metrics["Validation Metrics"]
    Output --> Metrics
```

---

## 7. Implementation Refinements (Viva Defense)

### 7.1 Scavenging Time Units
The wet scavenging coefficient uses standard literature parameterization:
- **Lambda = a × Pᵇ** with **a = 1×10⁻⁴ h⁻¹** (for P in mm/h)
- Time step **dt = 1 hour** → exponent **Lambda × dt** is dimensionless
- No unit conversion needed — verified against CAMS Sep 2019 proxy validation

```mermaid
flowchart LR
    P["P (mm/h)"] --> Lambda["Lambda = a × Pᵇ\na = 1e-4 h⁻¹, b = 0.8"]
    Lambda --> Exponent["Lambda × dt_hours\n(dimensionless)"]
    Exponent --> Retention["Retention = exp(-Σ Lambda·dt)"]
```

### 7.2 Covariance Matrix Nugget Effect
When receptor stations are geographically close (< 10 km), the spatial covariance matrix R can become near-singular, causing numerical instability in `np.linalg.inv(R)`.

**Fix**: Add small diagonal nugget before inversion:
```python
R = build_spatial_covariance(receptor_coords)
R += np.eye(len(receptor_coords)) * 1e-6  # Nugget: 1 µg²/m⁶
R_inv = np.linalg.inv(R)
```

This ensures condition number remains bounded while negligibly affecting inference.

### 7.3 Verification
```bash
# Run tests including covariance invertibility
python -m pytest tests/ -v

# Integration test
python -c "
from src.physics.calibration import build_spatial_covariance
import numpy as np
coords = np.array([[3.14, 101.69], [3.15, 101.70]])  # KL + nearby
R = build_spatial_covariance(coords)
R += np.eye(2) * 1e-6
print('Condition number:', np.linalg.cond(R))
print('Invertible:', np.all(np.isfinite(np.linalg.inv(R))))
"
```

---

## 6. Rendering Diagrams

These Mermaid diagrams can be rendered in:
- **GitHub/GitLab** - native rendering in markdown files
- **VS Code** - with Mermaid Preview extension
- **Obsidian/Notion** - native support
- **Mermaid Live Editor** - https://mermaid.live
- **Documentation sites** - MkDocs, Sphinx, Docusaurus with Mermaid plugins

To embed in documentation:
```markdown
```mermaid
<paste diagram code here>
```
```