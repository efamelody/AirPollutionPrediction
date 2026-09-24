"""Central constants for dispersion + inversion."""

# Wet deposition scavenging: Lambda = a * P^b  (P in mm/h)
SCAVENGE_A = 1e-4
SCAVENGE_B = 0.8

# Plume geometry
EFFECTIVE_HEIGHT_M = 30.0
EARTH_RADIUS_M = 6_371_000.0

# Inversion
SIGMA_OBS_UGM3 = 15.0
FRP_TO_Q_KGS = 0.02  # Q [kg/s] = 0.02 * FRP [MW]  (empirical, uncalibrated)

# Map centroid (Strait of Malacca)
METEO_LAT = 2.50
METEO_LON = 101.50
