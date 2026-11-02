"""Shared training results and leakage checks for rating feature tables."""

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict

from .config import FEATURE_SCHEMA_VERSION, TIME_CATEGORIES
from .feature_schema import feature_schema
from .neural_model import TorchRatingModel
from .split import PARTITIONS, assert_disjoint, normalize_player


class TrainingResult(BaseModel):
    """Trained networks, evaluation reports, and reproducibility metadata."""

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    models: dict[str, TorchRatingModel]
    report: dict
    metadata: dict


def validate_feature_table(
    frame: pd.DataFrame, schema_version: int = FEATURE_SCHEMA_VERSION
) -> pd.DataFrame:
    """Reject leakage, missing labels, invalid features, and duplicate target rows.

    Args:
        frame: Numeric features, rating labels, player identities, and partitions.
        schema_version: Feature contract used to construct the table.

    Returns:
        Validated copy with normalized target usernames.
    """
    feature_names = feature_schema(schema_version).names
    required_columns = {
        *feature_names,
        "rating",
        "category",
        "partition",
        "target_username",
        "game_id",
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Missing columns: {sorted(missing_columns)}")

    validated_frame = frame.copy()
    validated_frame["target_username"] = validated_frame["target_username"].map(
        normalize_player
    )
    assert_disjoint(validated_frame)
    if set(validated_frame["partition"]) != set(PARTITIONS):
        raise ValueError(
            "Training requires nonempty train, validation and test partitions"
        )

    feature_matrix = validated_frame.loc[:, list(feature_names)].to_numpy(dtype=float)
    if np.isinf(feature_matrix).any():
        raise ValueError("Feature values may be finite or NaN, never infinite")

    ratings = validated_frame["rating"].to_numpy(dtype=float)
    if not np.isfinite(ratings).all() or (ratings <= 0).any():
        raise ValueError("Labels must be finite positive game-time ratings")
    if not set(validated_frame["category"]).issubset(TIME_CATEGORIES):
        raise ValueError("Unsupported training time category")
    if validated_frame.duplicated(["game_id", "target_username"]).any():
        raise ValueError("Duplicate player/game rows")
    return validated_frame
