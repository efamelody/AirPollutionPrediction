"""Proxy Calibration Pipeline.

Orchestrates the full calibration workflow:
1. Urban baseline extraction from CAMS non-haze months
2. Biome-specific emission factor mapping via ESA WorldCover
3. Dynamic transport with ERA5 PBL and path-integrated scavenging
4. Validation against CAMS haze episode reanalysis
"""

import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp

from src.config import (
    EARTH_RADIUS_M, SCAVENGE_A, SCAVENGE_B,
    N_IMPORTANCE_SAMPLES, PRIOR_LOG_SIGMA, INSTRUMENT_ERROR, CORR_LENGTH,
    TRAJ_DT_HOURS, MAX_TRANSPORT_HOURS,
)
from src.data.cams import get_total_baseline, PARAMETRIC_BASELINES
from src.data.landcover import get_emission_params, calculate_Q
from src.data.era5 import fetch_meteo_grid_era5, get_meteo_interpolators


CACHE_DIR = Path(__file__).parent.parent.parent / "cache"
CALIBRATION_CACHE_FILE = CACHE_DIR / "calibration_results.json"


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    R = 6371.0
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def integrate_trajectory(
    source_lat: float, source_lon: float,
    u_interp, v_interp, precip_interp, pbl_interp,
    start_hour: int = 0,
    max_hours: int = MAX_TRANSPORT_HOURS,
    dt_hours: float = TRAJ_DT_HOURS
) -> dict:
    """Integrate 2D trajectory from source using gridded wind fields.
    
    Returns dict with trajectory arrays and path-integrated quantities.
    
    Note: start_hour is the absolute UTC hour for reference. The interpolators
    use 0-based forecast hour indices, so we query with relative indices.
    """
    lat, lon = source_lat, source_lon
    trajectory = [(lat, lon)]
    wind_speeds = []
    precip_rates = []
    pbl_heights = []
    scavenging_integral = 0.0
    
    for hour_idx in range(max_hours):
        # Query interpolators with relative forecast hour index (0-based)
        u = float(u_interp((lat, lon, hour_idx)))
        v = float(v_interp((lat, lon, hour_idx)))
        pr = float(precip_interp((lat, lon, hour_idx)))
        pbl = float(pbl_interp((lat, lon, hour_idx)))
        
        ws = np.hypot(u, v)
        if ws < 0.5:
            break
        
        # Convert u/v (m/s) to lat/lon displacement per hour
        # u = eastward, v = northward
        dlon = (u * 3600 * dt_hours) / (EARTH_RADIUS_M * np.cos(np.radians(lat)))
        dlat = (v * 3600 * dt_hours) / EARTH_RADIUS_M
        
        lat += np.degrees(dlat)
        lon += np.degrees(dlon)
        trajectory.append((lat, lon))
        
        wind_speeds.append(ws)
        precip_rates.append(pr)
        pbl_heights.append(pbl)
        
        # Path-integrated scavenging: Lambda = a * P^b
        # Lambda in h^-1 (a=1e-4 calibrated for h^-1, P in mm/h)
        # dt_hours in hours → Lambda * dt is dimensionless
        Lambda = SCAVENGE_A * (pr ** SCAVENGE_B) if pr > 0 else 0.0
        scavenging_integral += Lambda * dt_hours
        
        # Stop if left domain
        if lat < -5 or lat > 10 or lon < 95 or lon > 125:
            break
    
    return {
        "trajectory": np.array(trajectory),  # (T, 2)
        "wind_speeds": np.array(wind_speeds),
        "precip_rates": np.array(precip_rates),
        "pbl_heights": np.array(pbl_heights),
        "scavenging_integral": scavenging_integral,
    }


def compute_concentration_at_receptor(
    Q: float,
    traj_result: dict,
    receptor_lat: float, receptor_lon: float,
    start_hour: int = 0
) -> np.ndarray:
    """Compute smoke concentration at receptor for each hour along trajectory.
    
    Returns array of shape (T,) with concentration (µg/m³) at each hour.
    """
    trajectory = traj_result["trajectory"]
    wind_speeds = traj_result["wind_speeds"]
    pbl_heights = traj_result["pbl_heights"]
    scavenging_integral = traj_result["scavenging_integral"]
    
    if len(trajectory) < 2:
        return np.zeros(len(wind_speeds))
    
    concentrations = []
    
    for t in range(1, len(trajectory)):
        lat_f, lon_f = trajectory[t]
        
        # Downwind distance from source to current position
        dlat = np.radians(lat_f - trajectory[0, 0])
        dlon = np.radians(lon_f - trajectory[0, 1])
        lat_mid = np.radians((trajectory[0, 0] + lat_f) / 2)
        x_dist = EARTH_RADIUS_M * dlon * np.cos(lat_mid)
        y_dist = EARTH_RADIUS_M * dlat
        
        # Crosswind distance from trajectory to receptor
        dlat_r = np.radians(receptor_lat - lat_f)
        dlon_r = np.radians(receptor_lon - lon_f)
        lat_mid_r = np.radians((lat_f + receptor_lat) / 2)
        x_r = EARTH_RADIUS_M * dlon_r * np.cos(lat_mid_r)
        y_r = EARTH_RADIUS_M * dlat_r
        
        # Crosswind offset (perpendicular to trajectory direction)
        if t > 0:
            dx = trajectory[t, 1] - trajectory[t-1, 1]  # dlon
            dy = trajectory[t, 0] - trajectory[t-1, 0]  # dlat
            traj_angle = np.arctan2(dy, dx)
        else:
            traj_angle = 0
        
        # Project receptor offset onto crosswind axis
        along = x_r * np.cos(traj_angle) + y_r * np.sin(traj_angle)
        cross = -x_r * np.sin(traj_angle) + y_r * np.cos(traj_angle)
        
        if along < 0:  # Receptor upwind of current plume position
            concentrations.append(0.0)
            continue
        
        # Crosswind dispersion
        sigma_y = 0.11 * along * (1.0 + 0.0001 * along) ** (-0.5)
        sigma_y = max(sigma_y, 1000.0)  # Minimum 1km
        
        # PBL height at current position
        H_pbl = pbl_heights[t-1] if t-1 < len(pbl_heights) else pbl_heights[-1]
        
        # Mean wind speed up to this point
        U_mean = np.mean(wind_speeds[:t])
        
        # Scavenging retention up to this point
        precip_rates = traj_result["precip_rates"][:t]
        if len(precip_rates) > 0:
            # Handle NaN and negative values
            precip_rates = np.nan_to_num(precip_rates, nan=0.0, neginf=0.0, posinf=0.0)
            precip_rates = np.maximum(precip_rates, 0.0)
            Lambda_t = SCAVENGE_A * (precip_rates ** SCAVENGE_B)
            # Lambda in h^-1 (a=1e-4 calibrated for h^-1, P in mm/h)
            # TRAJ_DT_HOURS in hours → Lambda * dt is dimensionless
            retention = np.exp(-np.sum(Lambda_t) * TRAJ_DT_HOURS)
        else:
            retention = 1.0
        
        # Box model concentration
        centerline = Q / (np.sqrt(2 * np.pi) * U_mean * sigma_y * H_pbl) * 1e6
        cross_decay = np.exp(-0.5 * (cross / sigma_y) ** 2)
        
        C = centerline * cross_decay * retention
        concentrations.append(float(np.clip(C, 0, 1000)))
    
    # Pad to match wind_speeds length
    while len(concentrations) < len(wind_speeds):
        concentrations.append(0.0)
    
    return np.array(concentrations)


def build_transport_matrix(
    fires_df: pd.DataFrame,
    receptors_df: pd.DataFrame,
    meteo_interpolators: dict,
    start_hour: int = 0,
    forecast_hours: int = 24
) -> np.ndarray:
    """Build transport matrix C_smoke of shape (N_fires, N_receptors, N_hours)."""
    N_fires = len(fires_df)
    N_receptors = len(receptors_df)
    
    C_smoke = np.zeros((N_fires, N_receptors, forecast_hours))
    
    u_interp = meteo_interpolators["u"]
    v_interp = meteo_interpolators["v"]
    precip_interp = meteo_interpolators["precip"]
    pbl_interp = meteo_interpolators["pbl"]
    
    for i, (_, fire) in enumerate(fires_df.iterrows()):
        traj = integrate_trajectory(
            fire["latitude"], fire["longitude"],
            u_interp, v_interp, precip_interp, pbl_interp,
            start_hour=start_hour, max_hours=forecast_hours
        )
        
        for k, (_, rec) in enumerate(receptors_df.iterrows()):
            conc = compute_concentration_at_receptor(
                fire["Q_init"], traj,
                rec["latitude"], rec["longitude"],
                start_hour=start_hour
            )
            # Pad or truncate to forecast_hours
            if len(conc) > forecast_hours:
                conc = conc[:forecast_hours]
            elif len(conc) < forecast_hours:
                conc = np.pad(conc, (0, forecast_hours - len(conc)))
            C_smoke[i, k, :] = conc
    
    return C_smoke


def add_baseline(
    C_smoke: np.ndarray,
    receptors_df: pd.DataFrame,
    start_hour_utc: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """Add regional background + urban diurnal baseline to smoke concentrations.
    
    Args:
        C_smoke: (N_fires, N_receptors, N_hours)
        receptors_df: DataFrame with station_name column
        start_hour_utc: Starting hour in UTC
    
    Returns:
        C_smoke: unchanged smoke concentrations (N_fires, N_receptors, N_hours)
        baseline: (N_receptors, N_hours) baseline to add after summing fires
    """
    from src.data.cams import get_total_baseline
    
    N_fires, N_receptors, N_hours = C_smoke.shape
    baseline = np.zeros((N_receptors, N_hours))
    
    for k in range(N_receptors):
        city = receptors_df.iloc[k]["station_name"]
        for h in range(N_hours):
            hour_utc = (start_hour_utc + h) % 24
            baseline[k, h] = get_total_baseline(city, hour_utc)
    
    return C_smoke, baseline


def build_spatial_covariance(receptor_coords: np.ndarray) -> np.ndarray:
    """Build spatial error covariance matrix R.
    
    R_jk = sigma^2 * exp(-d_jk / L_corr) + delta_jk * sigma_inst^2
    """
    K = len(receptor_coords)
    R = np.zeros((K, K))
    for j in range(K):
        for k in range(K):
            d = haversine_km(
                receptor_coords[j, 0], receptor_coords[j, 1],
                receptor_coords[k, 0], receptor_coords[k, 1]
            )
            R[j, k] = INSTRUMENT_ERROR**2 * np.exp(-d / (CORR_LENGTH / 1000))
            if j == k:
                R[j, k] += INSTRUMENT_ERROR**2
    return R


def run_inverse_importance_sampling(
    fires_df: pd.DataFrame,
    receptors_df: pd.DataFrame,
    C_smoke: np.ndarray,
    obs_pm25: np.ndarray,
    n_samples: int = N_IMPORTANCE_SAMPLES
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized importance sampling for continuous Bayesian inverse.
    
    Args:
        fires_df: DataFrame with Q_init column
        receptors_df: DataFrame with station locations
        C_smoke: (N_fires, N_receptors, N_hours) - unit response per kg/s
        obs_pm25: (N_receptors,) observed PM2.5 at current hour
        n_samples: Number of importance samples
    
    Returns:
        Q_samples: (n_samples, N_fires)
        weights: (n_samples,) normalized importance weights
    """
    N_fires = len(fires_df)
    Q_init = fires_df["Q_init"].values
    
    # Use current hour (last hour in C_smoke)
    C_unit = C_smoke[:, :, -1]  # (N_fires, N_receptors) - concentration per unit Q
    
    # Draw samples from log-normal prior
    log_Q_samples = np.random.normal(
        np.log(Q_init + 1e-12), PRIOR_LOG_SIGMA, size=(n_samples, N_fires)
    )
    Q_samples = np.exp(log_Q_samples)
    
    # Forward model: C_pred = C_unit @ Q  (N_receptors,)
    # Vectorized: (n_samples, N_receptors)
    C_pred = Q_samples @ C_unit.T
    
    # Observed data
    d_obs = obs_pm25
    
    # Spatial covariance
    receptor_coords = receptors_df[["latitude", "longitude"]].values
    R = build_spatial_covariance(receptor_coords)
    # Add nugget effect for numerical stability when stations are close
    R += np.eye(len(receptor_coords)) * 1e-6
    R_inv = np.linalg.inv(R)
    
    # Vectorized log-likelihood
    diff = d_obs - C_pred  # (n_samples, N_receptors)
    log_lik = -0.5 * np.einsum('ni,ij,nj->n', diff, R_inv, diff)
    log_lik -= 0.5 * (len(d_obs) * np.log(2*np.pi) + np.log(np.linalg.det(R)))
    
    # Prior log-prob
    log_prior = -0.5 * np.sum(((np.log(Q_samples) - np.log(Q_init)) / PRIOR_LOG_SIGMA) ** 2, axis=1)
    log_prior -= N_fires * (np.log(PRIOR_LOG_SIGMA) + 0.5 * np.log(2*np.pi))
    
    # Normalize weights
    log_weights = log_lik + log_prior
    max_logw = np.max(log_weights)
    weights = np.exp(log_weights - max_logw)
    weights = weights / (weights.sum() + 1e-12)
    
    return Q_samples, weights


def marginal_posterior_prob(Q_samples: np.ndarray, weights: np.ndarray, threshold: float = 2.0) -> np.ndarray:
    """Marginal posterior probability that each fire is a major contributor."""
    median_Q = np.median(Q_samples, axis=0)
    return np.mean(Q_samples > threshold * median_Q, axis=0)


def compute_validation_metrics(
    pred_pm25: np.ndarray,
    obs_pm25: np.ndarray
) -> dict:
    """Compute proxy validation metrics."""
    rmse = np.sqrt(np.mean((pred_pm25 - obs_pm25)**2))
    nmb = np.sum(pred_pm25 - obs_pm25) / np.sum(obs_pm25) * 100
    r = np.corrcoef(pred_pm25, obs_pm25)[0, 1] if len(pred_pm25) > 1 else 0
    return {"RMSE": rmse, "NMB": nmb, "R": r}


def run_calibration_pipeline(
    fires_df: pd.DataFrame,
    receptors_df: pd.DataFrame,
    start_hour_utc: int = 0,
    forecast_hours: int = 24
) -> dict:
    """Run full proxy calibration pipeline.
    
    Returns dict with calibrated results ready for dashboard.
    """
    # 1. Ensure fires have biome and Q_init
    if "biome" not in fires_df.columns:
        from src.data.landcover import add_biome_column
        fires_df = add_biome_column(fires_df)
    if "Q_init" not in fires_df.columns:
        fires_df["Q_init"] = fires_df.apply(
            lambda r: calculate_Q(r["frp"], r["biome"]), axis=1
        )
    
    # 2. Fetch gridded meteo with PBL
    meteo_df = fetch_meteo_grid_era5(forecast_hours=forecast_hours)
    meteo_interpolators = get_meteo_interpolators(meteo_df)
    
    # 3. Build transport matrix
    C_smoke = build_transport_matrix(
        fires_df, receptors_df, meteo_interpolators,
        start_hour=start_hour_utc, forecast_hours=forecast_hours
    )
    
    # 4. Add baseline (returns C_smoke unchanged + baseline array)
    C_smoke, baseline = add_baseline(C_smoke, receptors_df, start_hour_utc)
    
    # 5. Current hour predictions (last hour) - sum fires THEN add baseline
    current_hour_idx = forecast_hours - 1
    pred_pm25_current = C_smoke[:, :, current_hour_idx].sum(axis=0) + baseline[:, current_hour_idx]
    
    # 6. Inverse attribution
    obs_pm25 = receptors_df["pm25_obs"].values
    Q_samples, weights = run_inverse_importance_sampling(
        fires_df, receptors_df, C_smoke, obs_pm25
    )
    post_probs = marginal_posterior_prob(Q_samples, weights)
    
    # 7. Validation metrics (against CAMS proxy if available)
    metrics = compute_validation_metrics(pred_pm25_current, obs_pm25)
    
    return {
        "fires": fires_df,
        "C_smoke": C_smoke,
        "baseline": baseline,
        "pred_pm25_current": pred_pm25_current,
        "posterior_probs": post_probs,
        "Q_samples": Q_samples,
        "weights": weights,
        "metrics": metrics,
        "meteo_interpolators": meteo_interpolators,
    }