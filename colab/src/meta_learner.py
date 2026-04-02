"""Meta-learning module.

Combines ensemble predictions from multiple AI weather models using a
stacking approach (LightGBM + MLP) and converts numeric outputs to
human-readable weather categories.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any

import numpy as np
import xarray as xr

from . import config

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def build_feature_vector(
    ensemble_results: dict[str, dict[str, xr.Dataset]],
) -> np.ndarray:
    """Build a feature matrix from ensemble statistics of all models.

    The feature vector at each grid point / lead time contains:
      [gc_mean, gc_std, gc_median,
       fcn_mean, fcn_std, fcn_median,
       mn3_mean, mn3_std, mn3_median,
       agreement_score, lead_time, month, lat, lon]

    Parameters
    ----------
    ensemble_results:
        Nested dict ``{model_name: {stat_name: xr.Dataset}}``.

    Returns
    -------
    np.ndarray
        2-D feature matrix of shape ``(n_samples, 14)``.
    """
    model_keys = ["graphcast", "fourcastnet", "metnet3"]
    stat_keys = ["mean", "std", "median"]

    # Determine spatial/temporal coordinates from the first available model
    ref_model = next(iter(ensemble_results))
    ref_ds = ensemble_results[ref_model]["mean"]

    # Identify coordinate names
    lat_name = "lat" if "lat" in ref_ds.coords else "latitude"
    lon_name = "lon" if "lon" in ref_ds.coords else "longitude"
    time_name = "step" if "step" in ref_ds.dims else "time"

    lats = ref_ds[lat_name].values
    lons = ref_ds[lon_name].values
    steps = ref_ds[time_name].values if time_name in ref_ds.dims else np.array([0])

    n_lats = len(lats)
    n_lons = len(lons)
    n_steps = len(steps)
    n_samples = n_steps * n_lats * n_lons
    n_features = 14

    features = np.zeros((n_samples, n_features), dtype=np.float32)

    idx = 0
    for t_idx, step in enumerate(steps):
        lead_time = float(step)
        # Approximate month from lead time (assuming forecast starts now)
        approx_month = (datetime.utcnow() + timedelta(hours=lead_time * 6)).month

        for i, lat in enumerate(lats):
            for j, lon in enumerate(lons):
                feat = []

                # Per-model stats (mean, std, median) for the first data variable
                model_means: list[float] = []
                for model in model_keys:
                    if model in ensemble_results:
                        stats = ensemble_results[model]
                        first_var = list(stats["mean"].data_vars)[0]

                        for stat in stat_keys:
                            ds = stats[stat]
                            if time_name in ds.dims:
                                val = float(ds[first_var].isel(
                                    **{time_name: t_idx, lat_name: i, lon_name: j}
                                ))
                            else:
                                val = float(ds[first_var].isel(
                                    **{lat_name: i, lon_name: j}
                                ))
                            feat.append(val)

                        # Collect means for agreement score
                        model_means.append(feat[-3])  # the mean value
                    else:
                        feat.extend([np.nan, np.nan, np.nan])

                # Agreement score: 1 - normalised std of model means
                if len(model_means) >= 2:
                    mean_of_means = np.mean(model_means)
                    std_of_means = np.std(model_means)
                    agreement = 1.0 - min(std_of_means / (abs(mean_of_means) + 1e-8), 1.0)
                else:
                    agreement = 1.0
                feat.append(agreement)

                # Contextual features
                feat.append(lead_time)
                feat.append(float(approx_month))
                feat.append(float(lat))
                feat.append(float(lon))

                features[idx] = feat
                idx += 1

    logger.info("Built feature matrix: shape=%s", features.shape)
    return features


# ---------------------------------------------------------------------------
# LightGBM training
# ---------------------------------------------------------------------------

def train_lightgbm(
    features: np.ndarray,
    targets: np.ndarray,
) -> Any:
    """Train a LightGBM regressor with Optuna hyperparameter tuning.

    Parameters
    ----------
    features:
        Feature matrix of shape ``(n_samples, n_features)``.
    targets:
        Target array of shape ``(n_samples,)`` or ``(n_samples, n_targets)``.

    Returns
    -------
    lightgbm.LGBMRegressor
        Trained model.
    """
    import lightgbm as lgb
    import optuna

    logger.info("Training LightGBM with Optuna tuning...")

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "num_leaves": trial.suggest_int("num_leaves", 16, 256),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "verbosity": -1,
        }

        from sklearn.model_selection import cross_val_score

        model = lgb.LGBMRegressor(**params)
        scores = cross_val_score(
            model, features, targets, cv=3, scoring="neg_mean_squared_error",
        )
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50, show_progress_bar=False)

    best_params = study.best_params
    best_params["random_state"] = 42
    best_params["verbosity"] = -1
    logger.info("Best LightGBM params: %s", best_params)

    model = lgb.LGBMRegressor(**best_params)
    model.fit(features, targets)
    return model


# ---------------------------------------------------------------------------
# MLP training (PyTorch)
# ---------------------------------------------------------------------------

def train_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    epochs: int = 100,
    lr: float = 1e-3,
) -> Any:
    """Train a small 3-layer MLP for meta-learning.

    Architecture: input -> 128 -> 64 -> output.

    Parameters
    ----------
    features:
        Feature matrix ``(n_samples, n_features)``.
    targets:
        Target array ``(n_samples,)`` or ``(n_samples, n_targets)``.
    epochs:
        Number of training epochs.
    lr:
        Learning rate.

    Returns
    -------
    torch.nn.Module
        Trained MLP model (in eval mode).
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    logger.info("Training MLP meta-learner (epochs=%d, lr=%.4f)", epochs, lr)

    n_features = features.shape[1]
    n_targets = targets.shape[1] if targets.ndim > 1 else 1

    class MetaMLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(n_features, 128),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(64, n_targets),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X = torch.tensor(features, dtype=torch.float32)
    y = torch.tensor(
        targets if targets.ndim > 1 else targets.reshape(-1, 1),
        dtype=torch.float32,
    )

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = MetaMLP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_x)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if (epoch + 1) % 20 == 0:
            logger.debug("MLP epoch %d/%d  loss=%.6f", epoch + 1, epochs, total_loss / len(loader))

    model.eval()
    logger.info("MLP training complete")
    return model


# ---------------------------------------------------------------------------
# Meta prediction
# ---------------------------------------------------------------------------

def predict_meta(
    lgbm_model: Any,
    mlp_model: Any,
    features: np.ndarray,
    lgbm_weight: float = 0.6,
) -> np.ndarray:
    """Weighted average of LightGBM and MLP predictions.

    Parameters
    ----------
    lgbm_model:
        Trained LightGBM model.
    mlp_model:
        Trained PyTorch MLP model.
    features:
        Feature matrix ``(n_samples, n_features)``.
    lgbm_weight:
        Weight for the LightGBM prediction (MLP gets ``1 - lgbm_weight``).

    Returns
    -------
    np.ndarray
        Blended predictions ``(n_samples,)`` or ``(n_samples, n_targets)``.
    """
    import torch

    # LightGBM prediction
    lgbm_pred = lgbm_model.predict(features)

    # MLP prediction
    device = next(mlp_model.parameters()).device
    with torch.no_grad():
        X = torch.tensor(features, dtype=torch.float32).to(device)
        mlp_pred = mlp_model(X).cpu().numpy().squeeze()

    blended = lgbm_weight * lgbm_pred + (1.0 - lgbm_weight) * mlp_pred
    logger.info("Meta prediction complete (lgbm_weight=%.2f)", lgbm_weight)
    return blended


# ---------------------------------------------------------------------------
# Weather category determination
# ---------------------------------------------------------------------------

def determine_weather(
    temp_max: float,
    temp_min: float,
    precip_prob: float,
    humidity: float,
) -> str:
    """Convert numeric predictions to a Japanese weather category.

    Categories:
      - 晴れ       (clear)
      - 曇り       (cloudy)
      - 雨         (rain)
      - 雪         (snow)
      - 曇時々晴   (cloudy with occasional sun)
      - 曇時々雨   (cloudy with occasional rain)
      - 晴時々曇   (clear with occasional clouds)

    Parameters
    ----------
    temp_max:
        Maximum temperature (°C).
    temp_min:
        Minimum temperature (°C).
    precip_prob:
        Precipitation probability (0-1).
    humidity:
        Relative humidity (%).

    Returns
    -------
    str
        Japanese weather category string.
    """
    avg_temp = (temp_max + temp_min) / 2.0

    # Snow: precipitation likely and cold enough
    if precip_prob >= 0.5 and avg_temp <= 1.0:
        return "雪"

    # Rain: high precipitation probability
    if precip_prob >= 0.6:
        return "雨"

    # Cloudy with occasional rain
    if 0.3 <= precip_prob < 0.6 and humidity >= 70:
        return "曇時々雨"

    # Cloudy
    if humidity >= 75 and precip_prob < 0.3:
        return "曇り"

    # Cloudy with occasional sun
    if 60 <= humidity < 75 and precip_prob < 0.2:
        return "曇時々晴"

    # Clear with occasional clouds
    if 40 <= humidity < 60 and precip_prob < 0.15:
        return "晴時々曇"

    # Clear
    if humidity < 60 and precip_prob < 0.1:
        return "晴れ"

    # Default fallback
    if precip_prob >= 0.3:
        return "曇時々雨"
    if humidity >= 60:
        return "曇時々晴"
    return "晴時々曇"


# ---------------------------------------------------------------------------
# Full meta-learning pipeline
# ---------------------------------------------------------------------------

def run_meta_learning(
    ensemble_results: dict[str, dict[str, xr.Dataset]],
) -> list[dict[str, Any]]:
    """Run the full meta-learning pipeline.

    Steps:
      1. Build feature vectors from ensemble stats.
      2. Load or train LightGBM + MLP models.
      3. Generate blended predictions.
      4. Convert to weather forecasts with category labels.

    Parameters
    ----------
    ensemble_results:
        Output of ``ensemble_inference.run_ensemble``.

    Returns
    -------
    list[dict]
        List of forecast dicts, each containing:
          date, location, latitude, longitude, weather, temp_max, temp_min,
          precipitation_prob, confidence, model_agreement, humidity,
          wind_speed, pressure.
    """
    features = build_feature_vector(ensemble_results)

    # --- Load or train models ---
    # TODO: In production, load pre-trained models from storage.
    #   For now we use the ensemble mean directly as a simple baseline
    #   since we have no ERA5 ground-truth targets available at inference
    #   time.  During a training run, pass actual observations as targets.

    # Extract target-like values from the ensemble mean (self-supervised stub)
    ref_model = next(iter(ensemble_results))
    ref_mean = ensemble_results[ref_model]["mean"]
    first_var = list(ref_mean.data_vars)[0]
    targets = ref_mean[first_var].values.flatten()[:features.shape[0]]

    lgbm_model = train_lightgbm(features, targets)
    mlp_model = train_mlp(features, targets)

    # Blended predictions
    raw_predictions = predict_meta(lgbm_model, mlp_model, features)

    # --- Build forecast list ---
    lat_name = "lat" if "lat" in ref_mean.coords else "latitude"
    lon_name = "lon" if "lon" in ref_mean.coords else "longitude"
    time_name = "step" if "step" in ref_mean.dims else "time"

    lats = ref_mean[lat_name].values
    lons = ref_mean[lon_name].values
    steps = ref_mean[time_name].values if time_name in ref_mean.dims else np.array([0])

    forecasts: list[dict[str, Any]] = []
    idx = 0
    base_date = datetime.utcnow()

    for step in steps:
        forecast_date = base_date + timedelta(hours=float(step) * 6)

        for lat in lats:
            for lon in lons:
                if idx >= len(raw_predictions):
                    break

                # Extract per-variable stats for this grid point
                # TODO: Use separate target heads for each variable instead
                #   of a single prediction value.
                pred_val = float(raw_predictions[idx])

                # Heuristic decomposition into forecast variables
                temp_max = pred_val + 3.0   # rough offset
                temp_min = pred_val - 3.0
                precip_prob = float(np.clip(
                    ensemble_results.get("graphcast", ensemble_results[ref_model])["std"][first_var]
                    .values.flatten()[idx % features.shape[0]] * 2.0,
                    0.0, 1.0,
                ))
                humidity = float(np.clip(60.0 + pred_val * 5.0, 0.0, 100.0))
                wind_speed = abs(pred_val) * 2.0
                pressure = 1013.25 + pred_val * 0.5

                # Agreement score from features
                agreement = float(features[idx, 9])

                # Confidence based on agreement and ensemble spread
                confidence = float(np.clip(agreement * 0.8 + 0.2, 0.0, 1.0))

                weather = determine_weather(temp_max, temp_min, precip_prob, humidity)

                forecasts.append({
                    "date": forecast_date.strftime("%Y-%m-%d"),
                    "location": f"Grid({float(lat):.2f}, {float(lon):.2f})",
                    "latitude": float(lat),
                    "longitude": float(lon),
                    "weather": weather,
                    "temp_max": round(temp_max, 1),
                    "temp_min": round(temp_min, 1),
                    "precipitation_prob": round(precip_prob, 3),
                    "confidence": round(confidence, 3),
                    "model_agreement": round(agreement, 3),
                    "humidity": round(humidity, 1),
                    "wind_speed": round(wind_speed, 1),
                    "pressure": round(pressure, 1),
                })
                idx += 1

    logger.info("Meta-learning pipeline complete: %d forecasts generated", len(forecasts))
    return forecasts
