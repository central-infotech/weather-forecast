"""Data fetching module for ERA5, GFS, and ECMWF open data.

Fetches the *latest available analysis/forecast* as initial conditions
for AI model inference. TARGET_DATE is the date we want to predict,
not the date of input data.
"""

from __future__ import annotations

import logging
import subprocess
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr

from . import config

logger = logging.getLogger(__name__)


def _latest_gfs_cycle() -> tuple[str, str]:
    """Return (YYYYMMDD, HH) for the most recent likely-available GFS cycle."""
    now = datetime.utcnow()
    # GFS data appears ~4-5h after cycle time
    cycle_hour = ((now.hour - 5) // 6) * 6
    if cycle_hour < 0:
        now -= timedelta(days=1)
        cycle_hour = 18
    return now.strftime("%Y%m%d"), f"{cycle_hour:02d}"


# ---------------------------------------------------------------------------
# ERA5 via CDS API
# ---------------------------------------------------------------------------

def fetch_era5_data(
    date: datetime | None = None,
    variables: list[str] | None = None,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download ERA5 reanalysis data via CDS API.

    Requires CDS_API_KEY and licence acceptance at:
    https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels
    """
    import cdsapi

    if date is None:
        date = datetime.utcnow() - timedelta(days=5)
    if variables is None:
        variables = config.ERA5_VARIABLES
    if region is None:
        region = config.JAPAN_REGION

    logger.info("Fetching ERA5 data for %s", date.strftime("%Y-%m-%d"))

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
    client.retrieve("reanalysis-era5-single-levels", request_params, output_path)
    ds = xr.open_dataset(output_path)
    logger.info("ERA5 data loaded: %s", list(ds.data_vars))
    return ds


# ---------------------------------------------------------------------------
# GFS via AWS Open Data (GRIB2 download)
# ---------------------------------------------------------------------------

def fetch_gfs_data(
    date: datetime | None = None,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download GFS analysis from AWS Open Data (NOAA GFS archive).

    Uses HTTPS download of GRIB2 files instead of OPeNDAP (more reliable).
    """
    if region is None:
        region = config.JAPAN_REGION

    date_str, cycle = _latest_gfs_cycle()

    # AWS Open Data GFS URL
    base_url = "https://noaa-gfs-bdp-pds.s3.amazonaws.com"
    grib_file = f"gfs.{date_str}/{cycle}/atmos/gfs.t{cycle}z.pgrb2.0p25.f000"
    url = f"{base_url}/{grib_file}"

    output_path = f"/tmp/gfs_{date_str}_{cycle}z.grib2"
    logger.info("Fetching GFS data from AWS: %s", url)

    # Download via curl (faster than Python requests for large files)
    result = subprocess.run(
        ["curl", "-s", "-f", "-o", output_path, url],
        capture_output=True, text=True, timeout=300,
    )
    if result.returncode != 0:
        raise RuntimeError(f"GFS download failed: {result.stderr}")

    # Open with cfgrib, selecting surface-level variables
    ds = xr.open_dataset(
        output_path,
        engine="cfgrib",
        backend_kwargs={
            "filter_by_keys": {"typeOfLevel": "heightAboveGround", "level": 2},
        },
    )

    # Ensure latitude is ascending for consistent slicing
    if ds.latitude.values[0] > ds.latitude.values[-1]:
        ds = ds.sortby("latitude")

    # Crop to region
    ds = ds.sel(
        latitude=slice(region["lat_min"], region["lat_max"]),
        longitude=slice(region["lon_min"], region["lon_max"]),
    )

    logger.info("GFS data loaded: %s (%d lat x %d lon)",
                list(ds.data_vars), len(ds.latitude), len(ds.longitude))
    return ds


# ---------------------------------------------------------------------------
# ECMWF Open Data
# ---------------------------------------------------------------------------

def fetch_ecmwf_data(
    date: datetime | None = None,
    region: dict[str, float] | None = None,
) -> xr.Dataset:
    """Download ECMWF open data forecast.

    Tries today, yesterday, and 2 days ago (ECMWF keeps ~2 days of data).
    """
    from ecmwf.opendata import Client

    if region is None:
        region = config.JAPAN_REGION
    if date is None:
        date = datetime.utcnow()

    for days_back in range(3):
        try_date = date - timedelta(days=days_back)
        date_str = try_date.strftime("%Y-%m-%d")
        logger.info("Trying ECMWF open data for %s", date_str)

        try:
            client = Client(source="ecmwf")
            output_path = f"/tmp/ecmwf_{try_date:%Y%m%d}.grib2"

            client.retrieve(
                date=date_str,
                time=0,
                step=[0, 6, 12, 18, 24],
                type="fc",
                param=["2t", "msl", "10u", "10v", "tp"],
                target=output_path,
            )

            # Use filter_by_keys to avoid heightAboveGround conflicts
            # (2m temp vs 10m wind have different levels)
            datasets = []
            for level_val in [2, 10]:
                try:
                    ds_part = xr.open_dataset(
                        output_path,
                        engine="cfgrib",
                        backend_kwargs={
                            "filter_by_keys": {
                                "typeOfLevel": "heightAboveGround",
                                "level": level_val,
                            },
                        },
                    )
                    datasets.append(ds_part)
                except Exception:
                    pass

            # Also get surface/mean-sea-level variables
            for level_type in ["surface", "meanSea"]:
                try:
                    ds_part = xr.open_dataset(
                        output_path,
                        engine="cfgrib",
                        backend_kwargs={
                            "filter_by_keys": {"typeOfLevel": level_type},
                        },
                    )
                    datasets.append(ds_part)
                except Exception:
                    pass

            if not datasets:
                raise RuntimeError("No variables could be decoded from GRIB2")

            # Merge all parts
            ds = xr.merge(datasets, compat="override")

            # Ensure latitude is ascending
            if ds.latitude.values[0] > ds.latitude.values[-1]:
                ds = ds.sortby("latitude")

            # Crop to region
            ds = ds.sel(
                latitude=slice(region["lat_min"], region["lat_max"]),
                longitude=slice(region["lon_min"], region["lon_max"]),
            )

            logger.info("ECMWF data loaded: %s", list(ds.data_vars))
            return ds

        except Exception as exc:
            logger.warning("ECMWF not available for %s: %s", date_str, exc)
            continue

    raise RuntimeError("ECMWF open data not available for the last 3 days")


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(dataset: xr.Dataset, region: dict[str, float] | None = None) -> xr.Dataset:
    """Normalize variables, interpolate missing values."""
    if region is None:
        region = config.JAPAN_REGION

    ds = dataset.copy(deep=True)

    # Interpolate missing values along spatial dimensions
    for dim in ("lat", "latitude", "lon", "longitude"):
        if dim in ds.dims:
            ds = ds.interpolate_na(dim=dim, method="linear")

    # Fill remaining NaNs
    dims_list = list(ds.dims)
    if dims_list:
        ds = ds.ffill(dim=dims_list[-1]).bfill(dim=dims_list[-1])

    # Z-score normalization per variable
    for var in ds.data_vars:
        vals = ds[var].values
        mean = float(np.nanmean(vals))
        std = float(np.nanstd(vals))
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

    target_date is the prediction target. Actual data fetched is the latest
    available analysis/forecast.
    """
    if isinstance(target_date, str):
        target_date = datetime.fromisoformat(target_date)

    region_bounds = config.JAPAN_REGION
    logger.info("Fetching initial-condition data for forecast target: %s", target_date.strftime("%Y-%m-%d"))

    results: dict[str, xr.Dataset] = {}
    errors: list[str] = []

    # ERA5
    try:
        era5_raw = fetch_era5_data(region=region_bounds)
        results["era5"] = preprocess(era5_raw, region=region_bounds)
    except Exception as exc:
        logger.error("ERA5 fetch failed: %s", exc)
        errors.append("era5")

    # GFS (AWS)
    try:
        gfs_raw = fetch_gfs_data(region=region_bounds)
        results["gfs"] = preprocess(gfs_raw, region=region_bounds)
    except Exception as exc:
        logger.error("GFS fetch failed: %s", exc)
        errors.append("gfs")

    # ECMWF
    try:
        ecmwf_raw = fetch_ecmwf_data(region=region_bounds)
        results["ecmwf"] = preprocess(ecmwf_raw, region=region_bounds)
    except Exception as exc:
        logger.error("ECMWF fetch failed: %s", exc)
        errors.append("ecmwf")

    if not results:
        raise RuntimeError(f"All data sources failed: {errors}")

    if errors:
        logger.warning("Some sources failed: %s – continuing with %s", errors, list(results.keys()))

    return results
