"""Upload module for pushing forecast results to Supabase.

Creates a forecast run record and inserts individual forecast entries
into the Supabase database.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from . import config

logger = logging.getLogger(__name__)


def upload_to_supabase(
    forecasts: list[dict[str, Any]],
    run_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Upload forecast results to Supabase.

    Creates a ``forecast_runs`` record first, then batch-inserts all
    individual forecast rows into the ``forecasts`` table with a foreign
    key reference to the run.

    Parameters
    ----------
    forecasts:
        List of forecast dicts as returned by
        ``meta_learner.run_meta_learning``.  Each dict should contain keys
        like ``date``, ``location``, ``latitude``, ``longitude``, ``weather``,
        ``temp_max``, ``temp_min``, ``precipitation_prob``, ``confidence``,
        ``model_agreement``, ``humidity``, ``wind_speed``, ``pressure``.
    run_metadata:
        Optional dict with metadata about the pipeline run, e.g.
        ``{"models": [...], "ensemble_size": 50, "region": "japan"}``.
        Stored alongside the run record.

    Returns
    -------
    dict[str, Any]
        Summary with keys ``run_id``, ``n_inserted``, ``status``.

    Raises
    ------
    RuntimeError
        If Supabase credentials are not configured or insertion fails.
    """
    from supabase import create_client, Client  # lazy import

    # --- Validate credentials ---
    supabase_url = config.SUPABASE_URL
    supabase_key = config.SUPABASE_KEY

    if not supabase_url or not supabase_key:
        raise RuntimeError(
            "Supabase credentials are not configured. "
            "Set SUPABASE_URL and SUPABASE_KEY environment variables."
        )

    logger.info("Connecting to Supabase at %s", supabase_url)
    client: Client = create_client(supabase_url, supabase_key)

    # --- Create forecast run record ---
    now = datetime.now(timezone.utc).isoformat()
    meta = run_metadata or {}
    run_record = {
        "executed_at": now,
        "initial_data_source": meta.get("initial_data_source", "GFS"),
        "ensemble_size": meta.get("ensemble_size", config.ENSEMBLE_SIZE),
        "models_used": meta.get("models", config.MODEL_NAMES),
        "status": "completed",
    }

    try:
        run_response = (
            client.table("forecast_runs")
            .insert(run_record)
            .execute()
        )
        run_id = run_response.data[0]["id"]
        logger.info("Created forecast_run record: id=%s", run_id)
    except Exception:
        logger.exception("Failed to create forecast_run record")
        raise RuntimeError("Failed to create forecast run record in Supabase")

    # --- Insert individual forecast rows ---
    import math

    def _clean(val: Any) -> Any:
        """Replace NaN/Inf with None for JSON serialization."""
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val

    def _to_int_percent(val: Any) -> int | None:
        """Convert 0-1 float probability to 0-100 int percent."""
        if val is None:
            return None
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return None
        # If already in 0-100 range, just round; if 0-1, multiply by 100
        if v <= 1.0:
            return int(round(v * 100))
        return int(round(v))

    rows = []
    for fc in forecasts:
        rows.append({
            "run_id": run_id,
            "target_date": fc.get("date"),
            "location": fc.get("location"),
            "latitude": _clean(fc.get("latitude")),
            "longitude": _clean(fc.get("longitude")),
            "weather": fc.get("weather"),
            "temp_max": _clean(fc.get("temp_max")),
            "temp_min": _clean(fc.get("temp_min")),
            "precipitation_prob": _to_int_percent(fc.get("precipitation_prob")),
            "confidence": _clean(fc.get("confidence")),
            "model_agreement": _clean(fc.get("model_agreement")),
            "humidity": _clean(fc.get("humidity")),
            "wind_speed": _clean(fc.get("wind_speed")),
            "pressure": _clean(fc.get("pressure")),
            "created_at": now,
        })

    # Batch insert in chunks to avoid payload limits
    batch_size = 500
    n_inserted = 0

    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        try:
            response = (
                client.table("forecasts")
                .insert(batch)
                .execute()
            )
            n_inserted += len(response.data)
            logger.info(
                "Inserted batch %d-%d (%d rows)",
                start, start + len(batch), len(response.data),
            )
        except Exception:
            logger.exception(
                "Failed to insert forecast batch starting at index %d", start,
            )
            # Update run status to partial failure
            try:
                client.table("forecast_runs").update(
                    {"status": "partial_failure"}
                ).eq("id", run_id).execute()
            except Exception:
                logger.exception("Failed to update run status")
            raise RuntimeError(
                f"Failed to insert forecasts at batch index {start}"
            )

    logger.info(
        "Upload complete: run_id=%s, %d/%d forecasts inserted",
        run_id, n_inserted, len(forecasts),
    )

    return {
        "run_id": run_id,
        "count": n_inserted,
        "status": "completed",
    }
