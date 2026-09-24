import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.special import logsumexp
from src.config import FRP_TO_Q_KGS, SIGMA_OBS_UGM3
from src.physics.dispersion import evaluate_gaussian_plume


def run_bayesian_inversion(
    hotspots_df: pd.DataFrame,
    receptors_df: pd.DataFrame,
    meteo: dict,
    sigma_obs: float = SIGMA_OBS_UGM3,
) -> pd.DataFrame:
    """Discrete Bayesian posterior over hotspots. Does not mutate input."""
    df = hotspots_df.copy()
    if df.empty:
        df["posterior_prob"] = []
        return df

    frp_sum = df["frp"].sum()
    if frp_sum <= 0:
        frp_sum = 1.0

    log_posts = []
    for _, spot in df.iterrows():
        prior = float(spot["frp"] / frp_sum)
        # Guard log(0)
        log_prior = np.log(max(prior, 1e-12))
        Q_est = float(spot["frp"] * FRP_TO_Q_KGS)

        log_lik = 0.0
        for _, rec in receptors_df.iterrows():
            c_pred = evaluate_gaussian_plume(
                float(spot["latitude"]),
                float(spot["longitude"]),
                float(rec["latitude"]),
                float(rec["longitude"]),
                Q_est,
                float(meteo["wind_speed"]),
                float(meteo["wind_dir"]),
                float(meteo["precipitation"]),
            )
            log_lik += norm.logpdf(float(rec["pm25_obs"]), loc=c_pred, scale=sigma_obs)

        log_posts.append(log_prior + log_lik)

    # Normalize in log-space for stability
    lse = logsumexp(log_posts)
    posteriors = np.exp(np.array(log_posts) - lse)
    df["posterior_prob"] = posteriors
    return df
