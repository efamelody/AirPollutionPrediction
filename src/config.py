"""Central constants for dispersion + inversion + proxy calibration."""

# Wet deposition scavenging: Lambda = a * P^b  (P in mm/h)
SCAVENGE_A = 1e-4
SCAVENGE_B = 0.8

# Plume geometry
EFFECTIVE_HEIGHT_M = 30.0
EARTH_RADIUS_M = 6_371_000.0

# Inversion
SIGMA_OBS_UGM3 = 15.0
FRP_TO_Q_KGS = 0.02  # Legacy uncalibrated fallback

# Map centroid (Strait of Malacca)
METEO_LAT = 2.50
METEO_LON = 101.50

# ==============================================================================
# PROXY CALIBRATION PARAMETERS
# ==============================================================================

# Biome emission factors (from literature: Akagi 2011, Andreae 2019, SE Asian campaigns)
BIOME_EMISSION_PARAMS = {
    "peatland": {
        "Cf": 0.52,           # kg/MJ - dry matter combustion factor
        "EF_pm25": 28.0,      # g/kg - PM2.5 emission factor (smoldering)
        "canopy_tau": 0.1,    # canopy thermal attenuation
        "description": "Smoldering Peat"
    },
    "tropical_forest": {
        "Cf": 0.36,
        "EF_pm25": 9.1,       # g/kg - flaming dominant
        "canopy_tau": 0.8,
        "description": "Flaming Forest Canopy"
    },
    "cropland_oilpalm": {
        "Cf": 0.45,
        "EF_pm25": 7.8,       # g/kg - agricultural clearing
        "canopy_tau": 0.3,
        "description": "Agricultural Scrub"
    },
    "other": {
        "Cf": 0.40,
        "EF_pm25": 12.0,
        "canopy_tau": 0.5,
        "description": "Mixed/Other Vegetation"
    }
}

# Cloud cover correction (fraction 0-1)
CLOUD_COVER_FRACTION = 0.15

# ESA WorldCover class mapping to biomes
ESA_CLASS_TO_BIOME = {
    10: "tropical_forest",      # Tree cover
    20: "tropical_forest",      # Shrubland
    30: "tropical_forest",      # Grassland
    40: "cropland_oilpalm",     # Cropland
    50: "tropical_forest",      # Built-up
    60: "tropical_forest",      # Bare/sparse vegetation
    70: "tropical_forest",      # Snow/ice
    80: "tropical_forest",      # Permanent water
    90: "peatland",             # Wetland / Peatland
    95: "tropical_forest",      # Mangroves
    100: "tropical_forest",     # Moss/lichen
}

# Planetary Boundary Layer
DEFAULT_PBL_HEIGHT = 1000.0  # m, fallback if ERA5/Open-Meteo unavailable
MIN_PBL_HEIGHT = 200.0       # m, nocturnal minimum
MAX_PBL_HEIGHT = 2500.0      # m, daytime convective max

# Urban Baseline (diurnal curve parameters per city)
# Derived from CAMS EAC4 non-haze months (Nov-Apr) proxy
URBAN_BASELINE_PARAMS = {
    "Kuala Lumpur": {"base": 12.0, "morning_amp": 12.0, "morning_peak": 8, "morning_width": 2.0,
                     "evening_amp": 10.0, "evening_peak": 20, "evening_width": 2.5},
    "Johor Bahru": {"base": 10.0, "morning_amp": 10.0, "morning_peak": 8, "morning_width": 2.0,
                    "evening_amp": 8.0, "evening_peak": 20, "evening_width": 2.5},
    "Kuching": {"base": 8.0, "morning_amp": 8.0, "morning_peak": 7, "morning_width": 2.0,
                "evening_amp": 6.0, "evening_peak": 19, "evening_width": 2.5},
    "Ipoh": {"base": 9.0, "morning_amp": 9.0, "morning_peak": 8, "morning_width": 2.0,
             "evening_amp": 7.0, "evening_peak": 20, "evening_width": 2.5},
    "Kota Bharu": {"base": 7.0, "morning_amp": 6.0, "morning_peak": 7, "morning_width": 2.0,
                   "evening_amp": 5.0, "evening_peak": 19, "evening_width": 2.5},
}

REGIONAL_BACKGROUND_UGM3 = 12.0  # Maritime background from CAMS clean months

# Inverse model (continuous importance sampling)
N_IMPORTANCE_SAMPLES = 2000
PRIOR_LOG_SIGMA = 0.7
INSTRUMENT_ERROR = 5.0
CORR_LENGTH = 50000.0  # 50 km spatial error correlation

# Trajectory integration
TRAJ_DT_HOURS = 1.0
MAX_TRANSPORT_HOURS = 72

# Forecast
FORECAST_HOURS = 24

# Grid bounds for meteo
GRID_BOUNDS = {
    "lat_min": -3.0,
    "lat_max": 7.0,
    "lon_min": 98.0,
    "lon_max": 119.0,
}