# Transboundary Haze Bayesian Predictor — IDN → MY

Gaussian plume with wet-deposition scavenging + discrete Bayesian inversion over NASA FIRMS hotspots.

## Run locally (Windows PowerShell 5.1)

```powershell
python -m pip install -r requirements.txt
streamlit run app.py
# http://localhost:8501
```

No API key required: defaults to `DEMO_KEY` synthetic fires (Sumatra/Kalimantan). For live fires, set NASA FIRMS MAP_KEY:

* Obtain at https://firms.modaps.eosdis.nasa.gov/api/area/
* Sidebar → paste key, or add to `.streamlit/secrets.toml`:
  ```toml
  FIRMS_MAP_KEY = "your_key"
  ```

## Publish to Streamlit Cloud

1. Push to GitHub (public repo)
2. https://share.streamlit.io → New app → select repo/branch `main`, file `app.py`
3. App Settings → Secrets → add `FIRMS_MAP_KEY` if available
4. Deploy → URL `https://<user>-airpollutionprediction-....streamlit.app`
5. Subsequent `git push` auto-redeploys

## Equations

* Forward: `C = Q/(π U σ_y σ_z) exp(-y²/2σ_y²) exp(-h²/2σ_z²) exp(-Λ x/U) *1e6`, `Λ=1e-4 P^0.8`, `h=30m`, Pasquill-Gifford D.
* Inverse: `p(θ|d) ∝ N(d|f(θ),R) p(θ)`, `p(θ)∝FRP`, `Q=0.02·FRP`, `R=σ²I`.

## Limitations (v0.1 demonstrator)

* Steady-state plume assumes 1–20km; 300–800km maritime transport needs HYSPLIT.
* Single centroid wind at (2.50,101.50); no 3D NWP or PBL.
* Receptors are mock (OpenDOSM daily aggregates, DOE APIMS has no public realtime API).
* No MCMC; exhaustive discrete posterior over hotspots only.

## Tests

```powershell
python -m pytest tests -v
```
