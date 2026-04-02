"""Ensemble inference module.

Generates perturbed initial conditions and runs multiple AI weather models
(GraphCast, FourCastNet, MetNet-3) to produce ensemble forecasts.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import xarray as xr

from . import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Lazy-loaded model singletons
# ---------------------------------------------------------------------------
_graphcast_model: Any | None = None
_fourcastnet_model: Any | None = None
_metnet3_model: Any | None = None


def _load_graphcast() -> Any:
    """Lazily load the GraphCast model.

    Returns the model (or a wrapper) ready for inference.
    """
    global _graphcast_model
    if _graphcast_model is not None:
        return _graphcast_model

    logger.info("Loading GraphCast model...")

    # TODO: Download and cache model weights on first run.
    #   Weights can be obtained from:
    #     https://console.cloud.google.com/storage/browser/dm_graphcast
    #   Expected checkpoint: GraphCast_small - ERA5 1979-2015 - resolution 0.25
    from graphcast import graphcast, checkpoint  # noqa: F401

    # TODO: Set the path to the downloaded checkpoint file.
    _GRAPHCAST_CHECKPOINT_PATH = "/content/graphcast/params/GraphCast_small.npz"

    with open(_GRAPHCAST_CHECKPOINT_PATH, "rb") as f:
        ckpt = checkpoint.parse_file_parts(f)

    _graphcast_model = ckpt
    logger.info("GraphCast model loaded successfully")
    return _graphcast_model


def _load_fourcastnet() -> Any:
    """Lazily load the FourCastNet model via earth2mip."""
    global _fourcastnet_model
    if _fourcastnet_model is not None:
        return _fourcastnet_model

    logger.info("Loading FourCastNet model...")

    # TODO: Ensure NVIDIA earth2mip is installed and model registry is configured.
    #   pip install earth2mip
    #   Model weights are fetched from NVIDIA NGC automatically.
    from earth2mip import registry, networks  # noqa: F401

    package = registry.get_model("e2mip://fcnv2_sm")
    _fourcastnet_model = networks.get_model(package)
    logger.info("FourCastNet model loaded successfully")
    return _fourcastnet_model


def _load_metnet3() -> Any:
    """Lazily load the MetNet-3 model (stub).

    Returns
    -------
    None
        MetNet-3 weights are not publicly available; this is a placeholder.
    """
    global _metnet3_model
    if _metnet3_model is not None:
        return _metnet3_model

    logger.warning(
        "MetNet-3 weights are not publicly available. "
        "Using stub that returns climatological persistence."
    )
    # TODO: Replace with actual MetNet-3 model loading once weights are released.
    _metnet3_model = "stub"
    return _metnet3_model


# ---------------------------------------------------------------------------
# Perturbation generation
# ---------------------------------------------------------------------------

def generate_perturbations(
    initial_state: xr.Dataset,
    n_members: int = config.ENSEMBLE_SIZE,
    scale: float = config.PERTURBATION_SCALE,
) -> list[xr.Dataset]:
    """Add Gaussian noise to the initial state to create ensemble members.

    Perturbations are drawn from N(0, scale * var_std) for each variable,
    where ``scale`` represents a fraction of the climatological variance.

    Parameters
    ----------
    initial_state:
        The unperturbed analysis / reanalysis state.
    n_members:
        Number of ensemble members to generate.
    scale:
        Perturbation magnitude as a fraction of each variable's standard
        deviation (0.03 = 3 %).

    Returns
    -------
    list[xr.Dataset]
        ``n_members`` perturbed copies of the initial state.
    """
    rng = np.random.default_rng(seed=42)
    perturbed: list[xr.Dataset] = []

    for i in range(n_members):
        member = initial_state.copy(deep=True)
        for var in member.data_vars:
            data = member[var].values
            std = float(np.nanstd(data))
            noise = rng.normal(loc=0.0, scale=scale * std, size=data.shape)
            member[var].values = data + noise.astype(data.dtype)
        perturbed.append(member)

    logger.info("Generated %d perturbed ensemble members (scale=%.4f)", n_members, scale)
    return perturbed


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_graphcast(
    perturbed_states: list[xr.Dataset],
    lead_days: int = 30,
) -> list[xr.Dataset]:
    """Run GraphCast on each perturbed initial state.

    GraphCast operates in 6-hour autoregressive steps.  For a 30-day forecast
    this means 30 * 4 = 120 steps per member.

    Parameters
    ----------
    perturbed_states:
        List of perturbed initial-state Datasets.
    lead_days:
        Forecast horizon in days.

    Returns
    -------
    list[xr.Dataset]
        One prediction Dataset per ensemble member.
    """
    model = _load_graphcast()
    n_steps = lead_days * 4  # 6-hour steps
    predictions: list[xr.Dataset] = []

    logger.info(
        "Running GraphCast: %d members x %d steps (%d days)",
        len(perturbed_states), n_steps, lead_days,
    )

    for idx, state in enumerate(perturbed_states):
        logger.debug("GraphCast member %d / %d", idx + 1, len(perturbed_states))

        # TODO: Replace with actual GraphCast autoregressive rollout.
        #   The real implementation should:
        #     1. Convert xarray state to GraphCast input format (JAX arrays)
        #     2. Build the task config and grid mesh
        #     3. Loop `n_steps` times calling graphcast.GraphCast.__call__
        #     4. Concatenate outputs along the time dimension
        #
        #   from graphcast import autoregressive, casting, normalization
        #   predictor = autoregressive.Predictor(model, ...)
        #   prediction = predictor(state, targets_template, forcings)

        # Stub: repeat initial state along a new time axis
        time_coords = np.arange(n_steps)
        pred = state.expand_dims({"step": time_coords})
        predictions.append(pred)

    logger.info("GraphCast inference complete for %d members", len(predictions))
    return predictions


def run_fourcastnet(
    perturbed_states: list[xr.Dataset],
    lead_days: int = 14,
) -> list[xr.Dataset]:
    """Run FourCastNet on each perturbed initial state.

    FourCastNet operates in 6-hour autoregressive steps.

    Parameters
    ----------
    perturbed_states:
        List of perturbed initial-state Datasets.
    lead_days:
        Forecast horizon in days.

    Returns
    -------
    list[xr.Dataset]
        One prediction Dataset per ensemble member.
    """
    model = _load_fourcastnet()
    n_steps = lead_days * 4
    predictions: list[xr.Dataset] = []

    logger.info(
        "Running FourCastNet: %d members x %d steps (%d days)",
        len(perturbed_states), n_steps, lead_days,
    )

    for idx, state in enumerate(perturbed_states):
        logger.debug("FourCastNet member %d / %d", idx + 1, len(perturbed_states))

        # TODO: Replace with actual earth2mip inference loop.
        #   from earth2mip import inference_ensemble
        #   datasource = ...  # wrap xarray state as a data source
        #   runner = inference_ensemble.run_basic_inference(
        #       model, n_steps=n_steps, data_source=datasource,
        #   )
        #   for step_result in runner:
        #       ...  # collect outputs

        # Stub: repeat initial state along a new time axis
        time_coords = np.arange(n_steps)
        pred = state.expand_dims({"step": time_coords})
        predictions.append(pred)

    logger.info("FourCastNet inference complete for %d members", len(predictions))
    return predictions


def run_metnet3(
    perturbed_states: list[xr.Dataset],
    lead_days: int = 7,
) -> list[xr.Dataset]:
    """Run MetNet-3 inference (stub).

    MetNet-3 weights are not publicly available.  This function provides a
    placeholder that returns the initial state repeated along time.

    Parameters
    ----------
    perturbed_states:
        List of perturbed initial-state Datasets.
    lead_days:
        Forecast horizon in days.

    Returns
    -------
    list[xr.Dataset]
        One prediction Dataset per ensemble member.
    """
    _load_metnet3()  # triggers warning
    n_steps = lead_days * 4
    predictions: list[xr.Dataset] = []

    logger.info(
        "Running MetNet-3 (stub): %d members x %d steps (%d days)",
        len(perturbed_states), n_steps, lead_days,
    )

    for idx, state in enumerate(perturbed_states):
        # TODO: Replace with actual MetNet-3 inference when weights become
        #   available.  MetNet-3 is a dense prediction model that outputs
        #   probabilistic precipitation, temperature, and wind at each grid
        #   point.  Steps are typically 2-minute for short range but can be
        #   resampled to 6-hour cadence for consistency.

        time_coords = np.arange(n_steps)
        pred = state.expand_dims({"step": time_coords})
        predictions.append(pred)

    logger.info("MetNet-3 (stub) inference complete for %d members", len(predictions))
    return predictions


# ---------------------------------------------------------------------------
# Ensemble statistics
# ---------------------------------------------------------------------------

def compute_ensemble_stats(predictions: list[xr.Dataset]) -> dict[str, xr.Dataset]:
    """Compute summary statistics across ensemble members.

    Parameters
    ----------
    predictions:
        List of prediction Datasets (one per ensemble member).  All must
        share the same coordinate structure.

    Returns
    -------
    dict[str, xr.Dataset]
        Keys: ``mean``, ``std``, ``median``, ``p10``, ``p25``, ``p50``,
        ``p75``, ``p90``.
    """
    if not predictions:
        raise ValueError("predictions list is empty")

    # Stack along a new 'member' dimension
    stacked = xr.concat(predictions, dim="member")

    stats: dict[str, xr.Dataset] = {
        "mean": stacked.mean(dim="member"),
        "std": stacked.std(dim="member"),
        "median": stacked.median(dim="member"),
        "p10": stacked.quantile(0.10, dim="member"),
        "p25": stacked.quantile(0.25, dim="member"),
        "p50": stacked.quantile(0.50, dim="member"),
        "p75": stacked.quantile(0.75, dim="member"),
        "p90": stacked.quantile(0.90, dim="member"),
    }

    logger.info("Ensemble stats computed: %s", list(stats.keys()))
    return stats


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

_MODEL_RUNNERS = {
    "graphcast": run_graphcast,
    "fourcastnet": run_fourcastnet,
    "metnet3": run_metnet3,
}


def run_ensemble(
    raw_data: dict[str, xr.Dataset],
    n_members: int = config.ENSEMBLE_SIZE,
) -> dict[str, dict[str, xr.Dataset]]:
    """Full ensemble pipeline: perturb -> run models -> compute stats.

    Parameters
    ----------
    raw_data:
        Dict of preprocessed Datasets (e.g. from ``data_fetcher.fetch_all_data``).
        The first available source is used as the initial state.
    n_members:
        Number of ensemble members.

    Returns
    -------
    dict[str, dict[str, xr.Dataset]]
        Outer key = model name (``graphcast``, ``fourcastnet``, ``metnet3``).
        Inner key = stat name (``mean``, ``std``, ``median``, ``p10``, ...).
    """
    # Use the first available data source as the initial state
    initial_state: xr.Dataset | None = None
    for source in ("era5", "gfs", "ecmwf"):
        if source in raw_data:
            initial_state = raw_data[source]
            logger.info("Using '%s' as initial state for ensemble", source)
            break

    if initial_state is None:
        raise ValueError("No data sources available in raw_data")

    # Generate perturbed initial conditions
    perturbed = generate_perturbations(initial_state, n_members=n_members)

    # Run each model and compute stats
    results: dict[str, dict[str, xr.Dataset]] = {}
    for model_name in config.MODEL_NAMES:
        runner = _MODEL_RUNNERS.get(model_name)
        if runner is None:
            logger.warning("No runner registered for model '%s'", model_name)
            continue

        try:
            preds = runner(perturbed)
            results[model_name] = compute_ensemble_stats(preds)
        except Exception:
            logger.exception("Model '%s' failed during ensemble run", model_name)
            continue

    if not results:
        raise RuntimeError("All model runs failed")

    logger.info("Ensemble run complete for models: %s", list(results.keys()))
    return results
