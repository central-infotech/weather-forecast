"""Configuration constants for the weather forecast pipeline."""

import os

# ---------------------------------------------------------------------------
# CDS API (ERA5)
# ---------------------------------------------------------------------------
CDS_API_URL: str = os.environ.get("CDS_API_URL", "https://cds.climate.copernicus.eu/api/v2")
CDS_API_KEY: str = os.environ.get("CDS_API_KEY", "")  # TODO: set via env

# ---------------------------------------------------------------------------
# GFS (NOAA NOMADS) – OPeNDAP endpoint for 0.25-degree GFS
# ---------------------------------------------------------------------------
GFS_DATA_URL: str = "https://nomads.ncep.noaa.gov/dods/gfs_0p25"

# ---------------------------------------------------------------------------
# ECMWF Open Data
# ---------------------------------------------------------------------------
ECMWF_OPEN_DATA_URL: str = "https://data.ecmwf.int/forecasts"

# ---------------------------------------------------------------------------
# Supabase
# ---------------------------------------------------------------------------
SUPABASE_URL: str = os.environ.get("SUPABASE_URL", "")
SUPABASE_KEY: str = os.environ.get("SUPABASE_KEY", "")

# ---------------------------------------------------------------------------
# Japan region bounds
# ---------------------------------------------------------------------------
JAPAN_LAT_MIN: float = 24.0
JAPAN_LAT_MAX: float = 46.0
JAPAN_LON_MIN: float = 122.0
JAPAN_LON_MAX: float = 150.0

JAPAN_REGION: dict = {
    "lat_min": JAPAN_LAT_MIN,
    "lat_max": JAPAN_LAT_MAX,
    "lon_min": JAPAN_LON_MIN,
    "lon_max": JAPAN_LON_MAX,
}

# ---------------------------------------------------------------------------
# ERA5 variables
# ---------------------------------------------------------------------------
ERA5_VARIABLES: list[str] = [
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
    "relative_humidity",
    "geopotential",
]

# ---------------------------------------------------------------------------
# Ensemble settings
# ---------------------------------------------------------------------------
ENSEMBLE_SIZE: int = 50
PERTURBATION_SCALE: float = 0.03  # 3% of climatological variance

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------
MODEL_NAMES: list[str] = [
    "graphcast",
    "fourcastnet",
    "metnet3",
]
