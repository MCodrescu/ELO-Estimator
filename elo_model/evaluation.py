"""Rating-band evaluation and train-only balancing weights."""

import numpy as np
import pandas as pd

from .config import RATING_BAND_EDGES as BAND_EDGES
from .config import RATING_BAND_NAMES as BAND_NAMES
from .feature_schema import FEATURE_GROUPS


def rating_bands(ratings) -> pd.Categorical:
    """Use left-closed rating intervals, including the requested edge values.

    Args:
        ratings: Positive numeric game-time rating labels.

    Returns:
        Categorical band labels in the original row order.
    """
    return pd.cut(ratings, BAND_EDGES, labels=BAND_NAMES, right=False)


def error_summary(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Report row-weighted error and equally weighted target-player MAE.

    Args:
        frame: Rows with rating and normalized target_username metadata.
        predictions: One prediction per frame row, in positional order.

    Returns:
        Counts, MAE, RMSE and macro-player MAE; empty errors are None.
    """
    if not len(frame):
        return {
            "rows": 0,
            "players": 0,
            "mae": None,
            "rmse": None,
            "player_macro_mae": None,
        }
    error = predictions - frame["rating"].to_numpy(dtype=float)
    player_errors = (
        pd.DataFrame(
            {"player": frame["target_username"].to_numpy(), "error": np.abs(error)}
        )
        .groupby("player")["error"]
        .mean()
    )
    return {
        "rows": len(frame),
        "players": len(player_errors),
        "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "player_macro_mae": float(player_errors.mean()),
    }


def evaluate(frame: pd.DataFrame, predictions: np.ndarray) -> dict:
    """Include empty rating bands rather than hiding missing coverage.

    Args:
        frame: Labeled rows with target identity and time category metadata.
        predictions: Finite predictions aligned by position, not index label.

    Returns:
        Overall, rating-band and category metric dictionaries.
    """
    predictions = np.asarray(predictions, dtype=float)
    if predictions.shape != (len(frame),) or not np.isfinite(predictions).all():
        raise ValueError("Predictions must be one finite value per row")
    bands = rating_bands(frame["rating"])
    return {
        "overall": error_summary(frame, predictions),
        "by_rating_band": {
            name: error_summary(frame.loc[bands == name], predictions[bands == name])
            for name in BAND_NAMES
        },
        "by_category": {
            name: error_summary(
                frame.loc[frame["category"] == name],
                predictions[(frame["category"] == name).to_numpy()],
            )
            for name in sorted(frame["category"].unique())
        },
    }


def sample_weights(
    train: pd.DataFrame, balance_bands: bool, balance_players: bool
) -> np.ndarray:
    """Compute clipped inverse frequencies using training labels only.

    Args:
        train: Training rows only, with rating and target identity columns.
        balance_bands: Weight rarer training rating bands more heavily.
        balance_players: Reduce weights of frequently observed targets.

    Returns:
        Finite row weights normalized to mean one.
    """
    weights = np.ones(len(train))
    if balance_bands:
        bands = pd.Series(rating_bands(train["rating"]), index=train.index)
        counts = bands.value_counts()
        weights *= np.asarray(
            bands.map(lambda band: len(train) / max(1, counts[band])), dtype=float
        )
    if balance_players:
        counts = train["target_username"].value_counts()
        weights *= (
            train["target_username"].map(lambda player: 1 / counts[player]).to_numpy()
        )
    weights /= weights.mean()
    weights = np.clip(weights, 0.2, 5)
    return weights / weights.mean()


def grouped_importance(booster, groups: dict | None = None) -> dict:
    """Sum tree gain and split counts by the versioned feature groups.

    Args:
        booster: Trained native LightGBM booster with saved feature names.
        groups: Feature groups for the selected schema; legacy groups by default.

    Returns:
        Per-group gain, gain fraction and split counts.
    """
    names = booster.feature_name()
    gains = dict(zip(names, booster.feature_importance(importance_type="gain")))
    splits = dict(zip(names, booster.feature_importance(importance_type="split")))
    total = sum(gains.values())
    return {
        group: {
            "gain": float(sum(gains[name] for name in features)),
            "gain_fraction": float(sum(gains[name] for name in features) / total)
            if total
            else 0.0,
            "splits": int(sum(splits[name] for name in features)),
        }
        for group, features in (groups if groups is not None else FEATURE_GROUPS).items()
    }
