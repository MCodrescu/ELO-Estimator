"""Small offline tests verify tuning, reporting and native model parity."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from elo_model.config import EngineConfig, FeatureConfig, NeuralTrainingConfig
from elo_model.evaluation import evaluate, rating_bands, sample_weights
from elo_model.feature_schema import FEATURE_NAMES
from elo_model.neural_training import TorchModelTrainer
from elo_model.predictor import RatingPredictor, export_bundle
from tests.test_features import sample_engine_metadata


def synthetic_frame():
    """Use distinct players/games per partition and one entirely absent feature.

    Args:
        None.

    Returns:
        Small deterministic labeled table for real training/export tests.
    """
    random = np.random.default_rng(42)
    frame = pd.DataFrame(
        random.normal(size=(180, len(FEATURE_NAMES))), columns=FEATURE_NAMES
    )
    frame["self_endgame_loss_mean"] = np.nan
    frame["rating"] = 1500 + frame["self_loss_mean"] * 250
    frame["category"] = ["blitz", "rapid"] * 90
    frame["partition"] = ["train"] * 100 + ["validation"] * 40 + ["test"] * 40
    frame["target_username"] = [f"player{index}" for index in range(len(frame))]
    frame["game_id"] = [f"game{index}" for index in range(len(frame))]
    return frame


class TrainingTests(unittest.TestCase):
    def test_neural_separate_models_and_deterministic_preprocessing(self):
        """Keep holdout statistics out of training and route separate checkpoints.

        Args:
            None.

        Returns:
            None; repeatability, preprocessing isolation and category routing are asserted.
        """
        frame = synthetic_frame()
        config = NeuralTrainingConfig(
            mode="separate",
            hidden_sizes=(8,),
            max_epochs=2,
            min_rows=10,
            batch_size=32,
            progress=False,
        )
        previous_threads = torch.get_num_threads()
        previous_random = torch.random.get_rng_state().clone()
        result = TorchModelTrainer(config).fit(frame)
        self.assertEqual(torch.get_num_threads(), previous_threads)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), previous_random))
        changed = frame.copy()
        changed.loc[changed["partition"] != "train", "self_loss_mean"] = 10000
        repeated = TorchModelTrainer(config).fit(frame)
        shifted = TorchModelTrainer(config).fit(changed)
        for category, model in result.models.items():
            self.assertEqual(
                model.preprocessing, shifted.models[category].preprocessing
            )
            rows = frame.loc[frame["category"] == category, list(FEATURE_NAMES)]
            np.testing.assert_array_equal(
                model.predict(rows), repeated.models[category].predict(rows)
            )
        result.metadata["engine"] = sample_engine_metadata()
        with tempfile.TemporaryDirectory() as directory:
            export_bundle(
                result.models,
                directory,
                frame.iloc[-10:],
                result.metadata,
                result.report,
            )
            predictor = RatingPredictor(directory)
            sample = frame.iloc[-10:]
            actual = predictor.predict(
                sample.loc[:, list(FEATURE_NAMES)], sample["category"].tolist()
            )
            self.assertTrue(np.isfinite(actual).all())
            with self.assertRaisesRegex(ValueError, "category"):
                predictor.predict(sample.loc[:, list(FEATURE_NAMES)])
            with self.assertRaisesRegex(ValueError, "category"):
                predictor.predict(
                    sample.loc[:, list(FEATURE_NAMES)], ["bullet"] * len(sample)
                )

    def test_neural_train_and_export_roundtrip(self):
        """Verify neural preprocessing, validation selection and portable predictions.

        Args:
            None.

        Returns:
            None; train-only statistics, reports and reload parity are asserted.
        """
        frame = synthetic_frame()
        config = NeuralTrainingConfig(
            hidden_sizes=(16, 8),
            max_epochs=3,
            patience=2,
            min_rows=10,
            batch_size=32,
            progress=False,
        )
        result = TorchModelTrainer(config).fit(frame)
        model = result.models["pooled"]
        train = frame.loc[frame["partition"] == "train"]
        self.assertAlmostEqual(model.preprocessing.target_mean, train["rating"].mean())
        missing_index = list(FEATURE_NAMES).index("self_endgame_loss_mean")
        self.assertEqual(model.preprocessing.medians[missing_index], 0)
        report = result.report["pooled"]
        self.assertEqual(
            report["best_validation_mae"],
            min(epoch["validation_mae"] for epoch in report["history"]),
        )
        self.assertEqual(report["partitions"]["test"]["pytorch"]["overall"]["rows"], 40)
        result.metadata["engine"] = sample_engine_metadata()
        with tempfile.TemporaryDirectory() as directory:
            verification = export_bundle(
                result.models,
                directory,
                frame.iloc[-10:],
                result.metadata,
                result.report,
            )
            self.assertLessEqual(verification["max_absolute_difference"], 1e-3)
            predictor = RatingPredictor(directory)
            self.assertEqual(predictor.manifest["model_backend"], "pytorch")
            self.assertEqual(predictor.manifest["models"], {"pooled": "pooled.pt"})
            actual = predictor.predict(frame.loc[:4, list(FEATURE_NAMES)])
            np.testing.assert_allclose(
                actual,
                model.predict(frame.loc[:4, list(FEATURE_NAMES)]),
                rtol=1e-6,
                atol=1e-4,
            )

    def test_expanded_report_boundaries_bias_and_empty_categories(self):
        """Keep exact edge ratings and sparse category groups visible.

        Args:
            None.

        Returns:
            None; expanded intervals, counts and signed bias are asserted.
        """
        frame = pd.DataFrame(
            {
                "rating": [99, 100, 349, 350, 2349, 2350, 2600, 2601],
                "target_username": [f"player{index}" for index in range(8)],
                "category": ["blitz"] * 8,
            }
        )
        self.assertEqual(
            list(rating_bands(frame["rating"], expanded=True)),
            [
                "<100",
                "100-350",
                "100-350",
                "350-600",
                "2100-2350",
                "2350-2600",
                "2350-2600",
                ">2600",
            ],
        )
        report = evaluate(frame, frame["rating"].to_numpy() + 25)
        self.assertEqual(report["overall"]["signed_bias"], 25)
        self.assertEqual(report["by_expanded_rating_band"]["2350-2600"]["rows"], 2)
        self.assertEqual(report["by_expanded_rating_band"][">2600"]["rows"], 1)
        empty = report["by_rating_band_category"]["100-350"]["rapid"]
        self.assertEqual(empty["rows"], 0)
        self.assertEqual(empty["players"], 0)
        self.assertIsNone(empty["signed_bias"])

    def test_fast_schema_export_and_incompatible_analysis_rejection(self):
        """Train reduced inputs and reject legacy provenance or overwrite attempts.

        Args:
            None.

        Returns:
            None; native reduced-schema parity and contract isolation are asserted.
        """
        features = FeatureConfig(schema_version=2)
        frame = synthetic_frame()
        result = TorchModelTrainer(
            NeuralTrainingConfig(
                hidden_sizes=(8,), max_epochs=2, min_rows=10, progress=False
            ),
            features,
        ).fit(frame)
        engine = asdict(
            EngineConfig(
                "mock",
                depth=None,
                multipv=1,
                time_seconds=0.1,
                played_move_scoring="next-position",
            )
        )
        engine.pop("executable")
        result.metadata["engine"] = {
            "settings": engine,
            "analysis_version": 2,
            "id": {"name": "mock"},
            "binary_sha256": "a" * 64,
        }
        with tempfile.TemporaryDirectory() as directory:
            export_bundle(
                result.models,
                directory,
                frame.iloc[-10:],
                result.metadata,
                result.report,
                features,
                model_version="test-fast-v2",
            )
            predictor = RatingPredictor(directory)
            self.assertEqual(len(predictor.schema.names), 93)
            self.assertTrue(
                np.isfinite(
                    predictor.predict(frame.loc[:2, list(predictor.schema.names)])
                ).all()
            )
            with self.assertRaisesRegex(ValueError, "metadata"):
                predictor.validate_engine_metadata({"settings": {"depth": 12}})
            changed = {
                **result.metadata,
                "engine": {**result.metadata["engine"], "binary_sha256": "b" * 64},
            }
            with self.assertRaisesRegex(ValueError, "separate"):
                export_bundle(
                    result.models, directory, frame.iloc[-10:], changed, {}, features
                )

    def test_export_rejects_reordered_feature_names(self):
        """Verify checkpoint reload parity and reject reordered feature schemas.

        Args:
            None.

        Returns:
            None; finite predictions, parity and schema rejection are asserted.
        """
        frame = synthetic_frame()
        result = TorchModelTrainer(
            NeuralTrainingConfig(
                hidden_sizes=(8,), max_epochs=2, min_rows=10, progress=False
            )
        ).fit(frame)
        result.metadata["engine"] = sample_engine_metadata()
        self.assertIn("train_median", result.report["pooled"]["partitions"]["test"])
        with tempfile.TemporaryDirectory() as directory:
            verification = export_bundle(
                result.models,
                directory,
                frame.iloc[100:120],
                result.metadata,
                result.report,
            )
            self.assertLessEqual(verification["max_absolute_difference"], 1e-3)
            predictor = RatingPredictor(directory)
            actual = predictor.predict(frame.loc[:4, list(FEATURE_NAMES)])
            self.assertTrue(np.isfinite(actual).all())
            names_path = Path(directory) / "feature_names.json"
            names_path.write_text(
                json.dumps(list(reversed(FEATURE_NAMES))), encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "schema"):
                RatingPredictor(directory)

    def test_separate_models_and_balancing(self):
        """Check category training, combined reports and required routing metadata.

        Args:
            None.

        Returns:
            None; model coverage, train weights and routing errors are asserted.
        """
        frame = synthetic_frame()
        train = frame.loc[frame["partition"] == "train"]
        weights = sample_weights(train, True, True)
        self.assertTrue(np.isfinite(weights).all())
        result = TorchModelTrainer(
            NeuralTrainingConfig(
                mode="separate",
                hidden_sizes=(8,),
                max_epochs=2,
                min_rows=10,
                balance_bands=True,
                balance_players=True,
                progress=False,
            )
        ).fit(frame)
        result.metadata["engine"] = sample_engine_metadata()
        self.assertEqual(set(result.models), {"blitz", "rapid"})
        self.assertEqual(
            sum(
                report["partitions"]["test"]["pytorch"]["overall"]["rows"]
                for report in result.report.values()
            ),
            40,
        )
        with tempfile.TemporaryDirectory() as directory:
            export_bundle(
                result.models,
                directory,
                frame.iloc[-20:],
                result.metadata,
                result.report,
            )
            predictor = RatingPredictor(directory)
            rows = frame.loc[:1, list(FEATURE_NAMES)]
            with self.assertRaisesRegex(ValueError, "category"):
                predictor.predict(rows)

    def test_empty_bands_are_reported(self):
        """Keep empty requested rating bands visible with null error values.

        Args:
            None.

        Returns:
            None; all band keys and the empty-band result are asserted.
        """
        frame = synthetic_frame().iloc[:1]
        metrics = evaluate(frame, frame["rating"].to_numpy())
        self.assertEqual(len(metrics["by_rating_band"]), 5)
        self.assertIsNone(metrics["by_rating_band"]["<1000"]["mae"])


if __name__ == "__main__":
    unittest.main()
