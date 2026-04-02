"""Data fetching module for ERA5, GFS, and ECMWF open data.

Provides functions to download, preprocess, and unify weather data from
multiple sources into xarray Datasets suitable for ensemble inference.

Note: Data fetched is the *latest available analysis/forecast* (i.e. near
today's date), which serves as the initial condition for AI model inference.
The TARGET_DATE in the pipeline is the date we want to *predict*, not the
date of the input data.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr

from . import config

logger = logging.getLogger(__name__)


def _latest_gfs_date() -> datetime:
    """Return the most recent GFS cycle date (usually ~6h behind real time)."""
    now = datetime.utcnow()
    # GFS runs at 00, 06, 12, 18 UTC; data appears ~4-5h after cycle time
    cycle_hour = (now.hour // 6) * 6
    if now.hour - cycle_hour < 5:
        # Data for this cycle may not be ready yet, use previous cycle
        cycle_hour -= 6
    if cycle_hour < 0:
        now -= timedelta(days=1)
        cycle_hour = 18
    return now.replace(hour=cycle_hour, minute=0, second=0, microsecond=0)


# ---------------------------------------------------------------------------
# ERA5 via CDS API
# ---------------------------------------------------------------------------

def fetch_era5_data(
    date: datetime | None = None,
    variables: list[str] | None = None,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download ERA5 reanalysis data for a given date and region.

    Parameters
    ----------
    date:
        Target date for the reanalysis data.  Defaults to 2 days ago
        (ERA5 has ~5-day latency, but ERA5T preliminary data is available
        within ~2 days).
    variables:
        List of ERA5 variable names.  Defaults to ``config.ERA5_VARIABLES``.
    region:
        Dict with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.

    Returns
    -------
    xr.Dataset
    """
    import cdsapi

    if date is None:
        date = datetime.utcnow() - timedelta(days=5)
    if variables is None:
        variables = config.ERA5_VARIABLES
    if region is None:
        region = config.JAPAN_REGION

    logger.info("Fetching ERA5 data for %s with %d variables", date.strftime("%Y-%m-%d"), len(variables))

    # CDS API area format: [north, west, south, east]
    area = [region["lat_max"], region["lon_min"], region["lat_min"], region["lon_max"]]

    client = cdsapi.Client(url=config.CDS_API_URL, key=config.CDS_API_KEY)

    request_params: dict[str, Any] = {
        "product_type": ["reanalysis"],
        "format": "netcdf",
        "variable": variables,
        "year": str(date.year),
        "month": f"{date.month:02d}",
        "day": f"{date.day:02d}",
        "time": [f"{h:02d}:00" for h in range(0, 24, 6)],
        "area": area,
    }

    output_path = f"/tmp/era5_{date:%Y%m%d}.nc"
    try:
        client.retrieve("reanalysis-era5-single-levels", request_params, output_path)
        ds = xr.open_dataset(output_path)
        logger.info("ERA5 data loaded: %s", list(ds.data_vars))
        return ds
    except Exception:
        logger.exception("Failed to fetch ERA5 data for %s", date.strftime("%Y-%m-%d"))
        raise


# ---------------------------------------------------------------------------
# GFS via OPeNDAP (NOMADS)
# ---------------------------------------------------------------------------

def fetch_gfs_data(
    date: datetime | None = None,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download latest GFS analysis from NOAA NOMADS via OPeNDAP.

    Parameters
    ----------
    date:
        Target analysis date/cycle. Defaults to the latest available cycle.
    region:
        Dict with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.

    Returns
    -------
    xr.Dataset
    """
    if region is None:
        region = config.JAPAN_REGION
    if date is None:
        date = _latest_gfs_date()

    # NOMADS URL format: gfs_0p25/gfsYYYYMMDD/gfs_0p25_HHz
    date_str = date.strftime("%Y%m%d")
    cycle_str = f"{date.hour:02d}z"
    url = f"https://nomads.ncep.noaa.gov/dods/gfs_0p25/gfs{date_str}/gfs_0p25_{cycle_str}"

    logger.info("Fetching GFS data from %s", url)

    try:
        ds = xr.open_dataset(url, engine="netcdf4")

        # Select analysis time (first time step = t=0, the initial condition)
        if "time" in ds.dims and len(ds.time) > 1:
            ds = ds.isel(time=0)

        # Crop to region
        # GFS lat: 90 to -90, lon: 0 to 359.75
        ds = ds.sel(
            lat=slice(region["lat_max"], region["lat_min"]),
            lon=slice(region["lon_min"], region["lon_max"]),
        )
        logger.info("GFS data loaded with %d variables", len(ds.data_vars))
        return ds
    except Exception:
        logger.exception("Failed to fetch GFS data for %s %s", date_str, cycle_str)
        raise


# ---------------------------------------------------------------------------
# ECMWF Open Data
# ---------------------------------------------------------------------------

def fetch_ecmwf_data(
    date: datetime | None = None,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download ECMWF open data forecast.

    Parameters
    ----------
    date:
        Target date.  Defaults to today (latest available forecast).
    region:
        Dict with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.

    Returns
    -------
    xr.Dataset
    """
    from ecmwf.opendata import Client

    if region is None:
        region = config.JAPAN_REGION
    if date is None:
        date = datetime.utcnow()

    # ECMWF open data is available for the last ~2 days; try today then yesterday
    for days_back in range(3):
        try_date = date - timedelta(days=days_back)
        logger.info("Trying ECMWF open data for %s", try_date.strftime("%Y-%m-%d"))

        try:
            client = Client(source="ecmwf")
            output_path = f"/tmp/ecmwf_{try_date:%Y%m%d}.grib2"

            client.retrieve(
                date=try_date.strftime("%Y-%m-%d"),
                time=0,
                step=[0, 6, 12, 18, 24],
                type="fc",
                param=["2t", "msl", "10u", "10v", "tp"],
                target=output_path,
            )

            ds = xr.open_dataset(output_path, engine="cfgrib")

            # Crop to region
            ds = ds.sel(
                latitude=slice(region["lat_max"], region["lat_min"]),
                longitude=slice(region["lon_min"], region["lon_max"]),
            )
            logger.info("ECMWF data loaded with %d variables", len(ds.data_vars))
            return ds
        except Exception as exc:
            logger.warning("ECMWF data not available for %s: %s", try_date.strftime("%Y-%m-%d"), exc)
            continue

    raise RuntimeError("ECMWF open data not available for the last 3 days")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(dataset: xr.Dataset, region: dict[str, float] | None = None) -> xr.Dataset:
    """Normalize variables, interpolate missing values, and crop to region.

    Parameters
    ----------
    dataset:
        Raw xarray Dataset.
    region:
        Crop bounds.  If ``None``, uses Japan region.

    Returns
    -------
    xr.Dataset
        Preprocessed dataset with z-score normalized variables and no NaNs.
    """
    if region is None:
        region = config.JAPAN_REGION

    ds = dataset.copy(deep=True)

    # Interpolate missing values along spatial dimensions
    for dim in ("lat", "latitude", "lon", "longitude"):
        if dim in ds.dims:
            ds = ds.interpolate_na(dim=dim, method="linear")
    # Fill any remaining NaNs (edges) with nearest
    dims_list = list(ds.dims.keys()) if hasattr(ds.dims, 'keys') else list(ds.dims)
    if dims_list:
        last_dim = dims_list[-1]
        ds = ds.ffill(dim=last_dim).bfill(dim=last_dim)

    # Z-score normalization per variable
    for var in ds.data_vars:
        mean = float(ds[var].mean())
        std = float(ds[var].std())
        if std > 0:
            ds[var] = (ds[var] - mean) / std
        else:
            ds[var] = ds[var] - mean

    logger.info("Preprocessing complete – %d variables normalized", len(ds.data_vars))
    return ds


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def fetch_all_data(
    target_date: datetime | str,
    region: str = "japan",
) -> dict[str, xr.Dataset]:
    """Fetch and preprocess data from all sources.

    Parameters
    ----------
    target_date:
        The date we want to *predict* (used for logging only).
        Actual data fetched is the latest available analysis.
    region:
        Region name (currently only ``"japan"`` is supported).

    Returns
    -------
    dict[str, xr.Dataset]
        Keys ``era5``, ``gfs``, ``ecmwf`` mapping to preprocessed Datasets.
    """
    if isinstance(target_date, str):
        target_date = datetime.fromisoformat(target_date)

    region_bounds = config.JAPAN_REGION

    logger.info("Fetching initial-condition data for forecast target: %s", target_date.strftime("%Y-%m-%d"))

    results: dict[str, xr.Dataset] = {}
    errors: list[str] = []

    # ERA5 (uses ~5 days ago as default)
    try:
        era5_raw = fetch_era5_data(region=region_bounds)
        results["era5"] = preprocess(era5_raw, region=region_bounds)
    except Exception as exc:
        logger.error("ERA5 fetch failed: %s", exc)
        errors.append("era5")

    # GFS (uses latest available cycle)
    try:
        gfs_raw = fetch_gfs_data(region=region_bounds)
        results["gfs"] = preprocess(gfs_raw, region=region_bounds)
    except Exception as exc:
        logger.error("GFS fetch failed: %s", exc)
        errors.append("gfs")

    # ECMWF (uses latest available forecast)
    try:
        ecmwf_raw = fetch_ecmwf_data(region=region_bounds)
        results["ecmwf"] = preprocess(ecmwf_raw, region=region_bounds)
    except Exception as exc:
        logger.error("ECMWF fetch failed: %s", exc)
        errors.append("ecmwf")

    if not results:
        raise RuntimeError(f"All data sources failed: {errors}")

    if errors:
        logger.warning("Some data sources failed: %s – continuing with available data", errors)

    return results
