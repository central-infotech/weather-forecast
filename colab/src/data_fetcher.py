"""Data fetching module for ERA5, GFS, and ECMWF open data.

Provides functions to download, preprocess, and unify weather data from
multiple sources into xarray Datasets suitable for ensemble inference.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import numpy as np
import xarray as xr

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# ERA5 via CDS API
# ---------------------------------------------------------------------------

def fetch_era5_data(
    date: datetime,
    variables: list[str] | None = None,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download ERA5 reanalysis data for a given date and region.

    Parameters
    ----------
    date:
        Target date for the reanalysis data.
    variables:
        List of ERA5 variable names.  Defaults to ``config.ERA5_VARIABLES``.
    region:
        Dict with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.
        Defaults to Japan region defined in config.

    Returns
    -------
    xr.Dataset
        ERA5 reanalysis data cropped to the specified region.
    """
    import cdsapi  # lazy import – only available when CDS API is configured

    if variables is None:
        variables = config.ERA5_VARIABLES
    if region is None:
        region = config.JAPAN_REGION

    logger.info("Fetching ERA5 data for %s with %d variables", date.isoformat(), len(variables))

    # CDS API area format: [north, west, south, east]
    area = [region["lat_max"], region["lon_min"], region["lat_min"], region["lon_max"]]

    client = cdsapi.Client(url=config.CDS_API_URL, key=config.CDS_API_KEY)

    request_params: dict[str, Any] = {
        "product_type": "reanalysis",
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
        logger.exception("Failed to fetch ERA5 data for %s", date.isoformat())
        raise


# ---------------------------------------------------------------------------
# GFS via OPeNDAP (NOMADS)
# ---------------------------------------------------------------------------

def fetch_gfs_data(
    date: datetime,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download latest GFS analysis from NOAA NOMADS via OPeNDAP.

    Parameters
    ----------
    date:
        Target analysis date/cycle.
    region:
        Dict with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.

    Returns
    -------
    xr.Dataset
        GFS analysis data cropped to the specified region.
    """
    if region is None:
        region = config.JAPAN_REGION

    # Build OPeNDAP URL for the 00z cycle of the given date
    cycle = "gfs_0p25_00z"
    url = f"{config.GFS_DATA_URL}/{cycle}"
    logger.info("Fetching GFS data from %s for %s", url, date.isoformat())

    try:
        ds = xr.open_dataset(url, engine="netcdf4")

        # Crop to region
        ds = ds.sel(
            lat=slice(region["lat_min"], region["lat_max"]),
            lon=slice(region["lon_min"], region["lon_max"]),
        )
        logger.info("GFS data loaded with %d variables", len(ds.data_vars))
        return ds
    except Exception:
        logger.exception("Failed to fetch GFS data for %s", date.isoformat())
        raise


# ---------------------------------------------------------------------------
# ECMWF Open Data
# ---------------------------------------------------------------------------

def fetch_ecmwf_data(
    date: datetime,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download ECMWF open data forecast.

    Parameters
    ----------
    date:
        Target date.
    region:
        Dict with keys ``lat_min``, ``lat_max``, ``lon_min``, ``lon_max``.

    Returns
    -------
    xr.Dataset
        ECMWF open data cropped to the specified region.
    """
    from ecmwf.opendata import Client  # lazy import

    if region is None:
        region = config.JAPAN_REGION

    logger.info("Fetching ECMWF open data for %s", date.isoformat())

    try:
        client = Client(source="ecmwf")
        output_path = f"/tmp/ecmwf_{date:%Y%m%d}.grib2"

        client.retrieve(
            date=date,
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
    except Exception:
        logger.exception("Failed to fetch ECMWF data for %s", date.isoformat())
        raise


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
    ds = ds.ffill(dim=list(ds.dims.keys())[-1]).bfill(dim=list(ds.dims.keys())[-1])

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
    target_date: datetime,
    region: str = "japan",
) -> dict[str, xr.Dataset]:
    """Fetch and preprocess data from all sources.

    Parameters
    ----------
    target_date:
        Date to fetch data for.
    region:
        Region name (currently only ``"japan"`` is supported).

    Returns
    -------
    dict[str, xr.Dataset]
        Keys ``era5``, ``gfs``, ``ecmwf`` mapping to preprocessed Datasets.
    """
    region_bounds = config.JAPAN_REGION  # extend with other regions as needed

    results: dict[str, xr.Dataset] = {}
    errors: list[str] = []

    # ERA5
    try:
        era5_raw = fetch_era5_data(target_date, region=region_bounds)
        results["era5"] = preprocess(era5_raw, region=region_bounds)
    except Exception as exc:
        logger.error("ERA5 fetch failed: %s", exc)
        errors.append("era5")

    # GFS
    try:
        gfs_raw = fetch_gfs_data(target_date, region=region_bounds)
        results["gfs"] = preprocess(gfs_raw, region=region_bounds)
    except Exception as exc:
        logger.error("GFS fetch failed: %s", exc)
        errors.append("gfs")

    # ECMWF
    try:
        ecmwf_raw = fetch_ecmwf_data(target_date, region=region_bounds)
        results["ecmwf"] = preprocess(ecmwf_raw, region=region_bounds)
    except Exception as exc:
        logger.error("ECMWF fetch failed: %s", exc)
        errors.append("ecmwf")

    if not results:
        raise RuntimeError(f"All data sources failed: {errors}")

    if errors:
        logger.warning("Some data sources failed: %s – continuing with available data", errors)

    return results
