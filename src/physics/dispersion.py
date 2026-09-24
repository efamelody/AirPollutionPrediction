import numpy as np
from src.config import EARTH_RADIUS_M, EFFECTIVE_HEIGHT_M, SCAVENGE_A, SCAVENGE_B


def evaluate_gaussian_plume(
    source_lat: float,
    source_lon: float,
    target_lat: float,
    target_lon: float,
    Q: float,
    wind_speed: float,
    wind_dir_deg: float,
    precip_rate: float,
) -> float:
    """Rain-depleted Gaussian plume at ground level (z=0).

    C = Q/(pi U sigma_y sigma_z) * exp(-0.5*(y/sigma_y)^2) * exp(-0.5*(h/sigma_z)^2) * exp(-Lambda*x/U) *1e6
    Lambda = a*P^b ; h=30m ; sigma_y/z from Pasquill-Gifford D.
    Returns ug/m3. Upwind returns 0.
    """
    R = EARTH_RADIUS_M
    dlat = np.radians(target_lat - source_lat)
    dlon = np.radians(target_lon - source_lon)
    lat_mid = np.radians((source_lat + target_lat) / 2.0)

    x_dist = R * dlon * np.cos(lat_mid)  # east
    y_dist = R * dlat  # north

    # Downwind axis phi = 270 - wind_dir (meteorological FROM convention)
    phi = np.radians(270.0 - wind_dir_deg)
    x_down = x_dist * np.cos(phi) + y_dist * np.sin(phi)
    y_cross = -x_dist * np.sin(phi) + y_dist * np.cos(phi)

    if x_down <= 0:
        return 0.0

    # Clamp to avoid sigma -> 0 near source
    x_down = max(x_down, 10.0)

    sigma_y = 0.11 * x_down * (1.0 + 0.0001 * x_down) ** (-0.5)
    sigma_z = 0.08 * x_down * (1.0 + 0.0001 * x_down) ** (-0.5)
    sigma_y = max(sigma_y, 1.0)
    sigma_z = max(sigma_z, 1.0)

    Lambda = SCAVENGE_A * (precip_rate ** SCAVENGE_B) if precip_rate > 0 else 0.0
    scavenging = float(np.exp(-Lambda * (x_down / max(wind_speed, 0.5))))

    h = EFFECTIVE_HEIGHT_M
    centerline = Q / (np.pi * max(wind_speed, 0.5) * sigma_y * sigma_z)
    crosswind_decay = np.exp(-0.5 * (y_cross / sigma_y) ** 2)
    vertical_decay = np.exp(-0.5 * (h / sigma_z) ** 2)

    C = centerline * crosswind_decay * vertical_decay * scavenging * 1e6
    return float(C)
