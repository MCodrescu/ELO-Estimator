"""Portable PyTorch rating networks with train-fitted preprocessing."""

from pathlib import Path
from typing import Self

import numpy as np
import pandas as pd
import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import nn

from .config import NeuralTrainingConfig


class NeuralPreprocessing(BaseModel):
    """Numeric transformations serialized alongside network weights."""

    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    medians: list[float]
    means: list[float]
    scales: list[float]
    target_mean: float
    target_scale: float = Field(gt=0)

    @model_validator(mode="after")
    def validate_dimensions(self) -> Self:
        """Check transform lengths and positive feature scales.

        Args:
            None.

        Returns:
            Validated transformations; invalid dimensions raise ValueError.
        """
        if not self.medians or not len(self.medians) == len(self.means) == len(
            self.scales
        ):
            raise ValueError(
                "Preprocessing arrays must have matching nonempty dimensions"
            )
        if any(scale <= 0 for scale in self.scales):
            raise ValueError("Feature scales must be positive")
        return self

    @classmethod
    def fit(cls, features: np.ndarray, ratings: np.ndarray) -> Self:
        """Learn imputation and scaling using training rows only.

        Args:
            features: Training feature matrix with optional missing values.
            ratings: Positive game-time training rating labels.

        Returns:
            Transformations with zero imputation for entirely missing columns.
        """
        medians = np.ma.median(np.ma.masked_invalid(features), axis=0).filled(0)
        imputed_features = np.where(np.isnan(features), medians, features)
        scales = imputed_features.std(axis=0)
        scales = np.where(scales > 0, scales, 1)

        return cls(
            medians=medians.tolist(),
            means=imputed_features.mean(axis=0).tolist(),
            scales=scales.tolist(),
            target_mean=float(ratings.mean()),
            target_scale=max(float(ratings.std()), 1),
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        """Impute, scale and append a fixed indicator for every input column.

        Args:
            features: Numeric matrix in the saved feature order.

        Returns:
            Finite float32 network inputs with twice the original feature count.
        """
        missing_values = np.isnan(features)
        imputed_features = np.where(missing_values, self.medians, features)
        scaled_features = (imputed_features - self.means) / self.scales
        network_inputs = np.concatenate(
            (scaled_features, missing_values.astype(float)), axis=1
        )
        return network_inputs.astype(np.float32)


class TorchRatingModel:
    """Small MLP and its exact preprocessing contract for training and inference."""

    def __init__(
        self,
        feature_names: tuple[str, ...],
        config: NeuralTrainingConfig,
        preprocessing: NeuralPreprocessing,
    ):
        """Construct the CPU network with a scalar normalized-rating output.

        Args:
            feature_names: Ordered original feature names.
            config: Hidden layer widths, dropout and training settings.
            preprocessing: Training-only numeric transformations.

        Returns:
            None.
        """
        if len(feature_names) != len(preprocessing.medians):
            raise ValueError("Feature names and preprocessing dimensions disagree")

        self.feature_names = tuple(feature_names)
        self.config = config
        self.preprocessing = preprocessing
        self.best_epoch = 0

        layers = []
        input_width = len(feature_names) * 2
        for hidden_width in config.hidden_sizes:
            layers.extend(
                (
                    nn.Linear(input_width, hidden_width),
                    nn.ReLU(),
                    nn.Dropout(config.dropout),
                )
            )
            input_width = hidden_width

        layers.append(nn.Linear(input_width, 1))
        self.network = nn.Sequential(*layers)

    def predict(self, features: np.ndarray | pd.DataFrame) -> np.ndarray:
        """Predict original rating units without refitting preprocessing.

        Args:
            features: Ordered numeric matrix or DataFrame; NaN is allowed.

        Returns:
            One finite, unconstrained raw rating prediction per input row.
        """
        if isinstance(features, pd.DataFrame) and list(features.columns) != list(
            self.feature_names
        ):
            raise ValueError("DataFrame columns must exactly match saved feature order")

        matrix = np.asarray(features, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(self.feature_names)
            or np.isinf(matrix).any()
        ):
            raise ValueError("Expected an ordered finite-or-NaN feature matrix")

        if not len(matrix):
            return np.empty(0)

        device = next(self.network.parameters()).device
        inputs = torch.from_numpy(self.preprocessing.transform(matrix)).to(device)
        self.network.eval()

        with torch.inference_mode():
            normalized_predictions = (
                self.network(inputs).squeeze(1).cpu().numpy().astype(float)
            )

        predictions = (
            normalized_predictions * self.preprocessing.target_scale
            + self.preprocessing.target_mean
        )
        if not np.isfinite(predictions).all():
            raise ValueError("Network produced nonfinite rating predictions")

        return predictions

    def save_model(self, path: str | Path) -> None:
        """Save weights and primitive preprocessing data without a pickled model object.

        Args:
            path: Completed-model checkpoint destination.

        Returns:
            None.
        """
        checkpoint = {
            "version": 1,
            "feature_names": list(self.feature_names),
            "config": self.config.model_dump(mode="json"),
            "preprocessing": self.preprocessing.model_dump(mode="json"),
            "best_epoch": self.best_epoch,
            "state_dict": {
                name: value.detach().cpu()
                for name, value in self.network.state_dict().items()
            },
        }
        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a trusted local weights-only checkpoint on CPU.

        Args:
            path: Exported PyTorch checkpoint.

        Returns:
            An evaluation-mode model with its saved feature and preprocessing contract.
        """
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if checkpoint.get("version") != 1:
            raise ValueError("Unsupported PyTorch checkpoint version")

        model = cls(
            tuple(checkpoint["feature_names"]),
            NeuralTrainingConfig.model_validate(checkpoint["config"]),
            NeuralPreprocessing.model_validate(checkpoint["preprocessing"]),
        )
        model.network.load_state_dict(checkpoint["state_dict"], strict=True)
        model.best_epoch = checkpoint["best_epoch"]
        model.network.eval()

        return model
