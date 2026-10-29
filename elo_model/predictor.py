"""Native LightGBM bundles for fast Python inference, without pickle or engines."""

import json
import os
import tempfile
import time
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, FiniteFloat

from .config import (
    MODEL_MANIFEST_FILENAME,
    TIME_CATEGORIES,
    TIME_CATEGORY_LIMITS,
    TIME_CONTROL_INCREMENT_MOVES,
    EngineCompatibility,
    FeatureConfig,
    validate_feature_engine,
)
from .feature_schema import feature_schema
from .features import GameFeatureExtractor
from .game import GameInput, PositionAnalysis
from .storage import write_json


class RatingPredictions(BaseModel):
    """Validated finite player ratings returned by single-game inference."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    white: FiniteFloat
    black: FiniteFloat


class RatingPredictor:
    """Load a trusted local artifact directory once and reuse across requests."""

    def __init__(self, directory: str | Path):
        """Load native trees and reject incompatible feature contracts.

        Args:
            directory: Trusted artifact folder with model_manifest and schema.

        Returns:
            None; the instance retains reusable boosters and an extractor.
        """
        directory = Path(directory)
        self.manifest = json.loads(
            (directory / MODEL_MANIFEST_FILENAME).read_text(encoding="utf-8")
        )
        names = json.loads(
            (directory / "feature_names.json").read_text(encoding="utf-8")
        )
        self.schema = feature_schema(self.manifest["schema_version"])
        if (
            names != list(self.schema.names)
            or self.manifest["feature_hash"] != self.schema.fingerprint
            or self.manifest.get("feature_config", {}).get("schema_version") != self.schema.version
        ):
            raise ValueError(
                "Artifact feature schema does not match this extraction code"
            )
        self.models = {}
        for name, filename in self.manifest["models"].items():
            path = directory / filename
            if path.resolve().parent != directory.resolve():
                raise ValueError("Model file must be inside the artifact directory")
            model = lgb.Booster(model_file=str(path))
            if model.feature_name() != names:
                raise ValueError("Booster feature names disagree with the saved schema")
            self.models[name] = model
        if not self.models:
            raise ValueError("Artifact contains no models")
        self.extractor = GameFeatureExtractor(
            FeatureConfig(**self.manifest["feature_config"])
        )
        validate_feature_engine(self.extractor.config, self.manifest.get("engine"))

    def predict(self, features, categories: list[str] | None = None) -> np.ndarray:
        """Predict ordered rows; separate models require a category per row.

        Args:
            features: Numeric (n, 93) array or exactly ordered DataFrame; NaN is allowed.
            categories: One trained time category per row for separate bundles.

        Returns:
            One floating-point rating per feature row.
        """
        if isinstance(features, pd.DataFrame) and list(features.columns) != list(
            self.schema.names
        ):
            raise ValueError("DataFrame columns must exactly match saved feature order")
        matrix = np.asarray(features, dtype=float)
        if (
            matrix.ndim != 2
            or matrix.shape[1] != len(self.schema.names)
            or np.isinf(matrix).any()
        ):
            raise ValueError(
                "Expected a two-dimensional ordered feature matrix, finite or NaN"
            )
        if not len(matrix):
            return np.empty(0)
        if "pooled" in self.models:
            return self.models["pooled"].predict(matrix, num_threads=1)
        if categories is None or len(categories) != len(matrix):
            raise ValueError("Supply one time category per feature row")
        result = np.empty(len(matrix))
        for category in set(categories):
            if category not in self.models:
                raise ValueError(f"No trained model for category {category}")
            mask = np.asarray(categories) == category
            result[mask] = self.models[category].predict(matrix[mask], num_threads=1)
        return result

    def validate_engine_metadata(
        self,
        engine_metadata: dict | None,
        *,
        engine_compatibility: EngineCompatibility = "exact",
    ) -> None:
        """Check analysis provenance, optionally allowing a different binary build.

        Args:
            engine_metadata: Actual runtime provenance; never relabeled as training data.
            engine_compatibility: Exact metadata or same-release metadata excluding binary hash.

        Returns:
            None; unknown policies and incompatible engine contracts raise ValueError.
        """
        if engine_compatibility not in ("exact", "same-release"):
            raise ValueError("Unknown engine compatibility policy")
        expected = self.manifest.get("engine")
        if expected is None or engine_metadata == expected:
            return
        if engine_compatibility == "same-release" and isinstance(expected, dict) and isinstance(engine_metadata, dict):
            expected_contract = {key: value for key, value in expected.items() if key != "binary_sha256"}
            actual_contract = {key: value for key, value in engine_metadata.items() if key != "binary_sha256"}
            identity = expected.get("id")
            complete = (
                isinstance(identity, dict) and bool(identity.get("name"))
                and "settings" in expected and "analysis_version" in expected
            )
            hashes = (expected.get("binary_sha256"), engine_metadata.get("binary_sha256"))
            valid_hashes = all(
                isinstance(value, str) and len(value) == 64
                and all(character in "0123456789abcdef" for character in value)
                for value in hashes
            )
            if complete and valid_hashes and expected_contract == actual_contract:
                return
        raise ValueError(
            f"Analysis engine metadata must match the training artifact ({engine_compatibility})"
        )

    def predict_game(
        self,
        game: GameInput,
        analysis: list[PositionAnalysis],
        engine_metadata: dict | None = None,
        *,
        engine_compatibility: EngineCompatibility = "exact",
    ) -> dict[str, float]:
        """Extract shared features and predict both sides using supplied analysis.

        Args:
            game: Parsed standard-chess game, without requiring rating headers.
            analysis: Precomputed White-perspective scores aligned with replay.
            engine_metadata: Engine provenance matching the training artifact.
            engine_compatibility: Allow a same-release build with a different binary hash.

        Returns:
            White and Black numeric predicted ratings.
        """
        self.validate_engine_metadata(
            engine_metadata, engine_compatibility=engine_compatibility
        )
        rows = self.extractor.extract(game, analysis)
        predictions = self.predict(rows, [game.time_control()[2]] * 2)
        response = RatingPredictions(white=predictions[0], black=predictions[1])

        return response.model_dump()


def _save_booster(model, path: Path) -> None:
    """Atomically save only the selected LightGBM iterations.

    Args:
        model: Native booster, optionally with a selected best iteration.
        path: Destination text model path in an existing directory.

    Returns:
        None; filesystem and serialization failures propagate.
    """
    with tempfile.NamedTemporaryFile(
        dir=path.parent, delete=False, suffix=".txt"
    ) as handle:
        temporary = Path(handle.name)
    try:
        model.save_model(str(temporary), num_iteration=model.best_iteration or -1)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def export_bundle(
    models: dict,
    directory: str | Path,
    sample: pd.DataFrame,
    metadata: dict,
    report: dict,
    feature_config: FeatureConfig | None = None,
    model_version: str | None = None,
) -> dict:
    """Save native trees and verify reloaded predictions on supplied sample rows.

    Args:
        models: Pooled or category-name to native booster mapping.
        directory: Output artifact directory.
        sample: Held-out feature rows, with category metadata for routing.
        metadata: Training, engine and target provenance for the manifest.
        report: JSON-compatible model evaluation and importance reports.
        feature_config: Exact transformations used to create training features.
        model_version: Optional release label, separate from the Python package version.

    Returns:
        Reload parity, native model bytes and warm single-row timing measurements.
    """
    if not len(sample):
        raise ValueError("Export verification needs at least one sample row")
    if model_version is not None and not model_version.strip():
        raise ValueError("model_version must not be empty")
    if not set(models).issubset({"pooled", *TIME_CATEGORIES}):
        raise ValueError("Invalid model names")
    feature_config = feature_config or FeatureConfig()
    schema = feature_schema(feature_config.schema_version)
    validate_feature_engine(feature_config, metadata.get("engine"))
    directory = Path(directory)
    existing = directory / MODEL_MANIFEST_FILENAME
    if existing.exists():
        previous = json.loads(existing.read_text(encoding="utf-8"))
        if previous.get("engine") != metadata.get("engine") or previous["schema_version"] != schema.version:
            raise ValueError("Export a different contract into a separate versioned directory")
    directory.mkdir(parents=True, exist_ok=True)
    files = {name: f"{name}.txt" for name in models}
    for name, model in models.items():
        _save_booster(model, directory / files[name])
    manifest = {
        **metadata,
        "artifact_version": 1,
        "schema_version": schema.version,
        "feature_hash": schema.fingerprint,
        "models": files,
        "feature_config": asdict(feature_config),
        "package_version": "0.2.0",
        "engine_compatibility_policy": "exact",
        "analysis_contract": {
            "engine": metadata.get("engine"),
            "compatibility_policy": "exact",
            "schema_version": schema.version,
            "settings": metadata["engine"]["settings"],
        },
        "time_category_rule": {
            "estimated_seconds": f"base + {TIME_CONTROL_INCREMENT_MOVES} * increment",
            "upper_bounds": dict(zip(TIME_CATEGORIES[:-1], TIME_CATEGORY_LIMITS[:-1])),
        },
        "libraries": {
            name: version(name)
            for name in ("lightgbm", "numpy", "scikit-learn", "chess")
        },
    }
    write_json(directory / "feature_names.json", schema.names)
    if model_version is not None:
        manifest["model_version"] = model_version
    write_json(directory / "feature_groups.json", schema.groups)
    write_json(directory / MODEL_MANIFEST_FILENAME, manifest)
    predictor = RatingPredictor(directory)
    matrix = sample.loc[:, list(schema.names)]
    categories = sample["category"].tolist()
    expected = np.empty(len(sample))
    for name, model in models.items():
        mask = (
            np.ones(len(sample), dtype=bool)
            if name == "pooled"
            else (sample["category"] == name).to_numpy()
        )
        expected[mask] = model.predict(matrix.loc[mask], num_threads=1)
    actual = predictor.predict(matrix, categories)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-9)
    predictor.predict(matrix.iloc[:1], categories[:1])
    started = time.perf_counter()
    for _ in range(30):
        predictor.predict(matrix.iloc[:1], categories[:1])
    verification = {
        "sample_rows": len(sample),
        "max_absolute_difference": float(np.max(np.abs(actual - expected))),
        "warm_single_row_ms": (time.perf_counter() - started) * 1000 / 30,
        "native_model_bytes": sum(
            (directory / filename).stat().st_size for filename in files.values()
        ),
    }
    write_json(
        directory / "evaluation.json",
        {"models": report, "export_verification": verification},
    )
    return verification
