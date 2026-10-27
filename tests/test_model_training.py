"""Tiny synthetic experiments verify tuning, reporting and native model parity."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from elo_model.evaluation import evaluate, sample_weights
from elo_model.feature_schema import FEATURE_NAMES
from elo_model.predictor import RatingPredictor, export_bundle
from elo_model.training import ModelTrainer, TrainingConfig


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
    def test_train_tune_and_export_roundtrip(self):
        """Run one tuning trial and verify native reload/schema protection.

        Args:
            None.

        Returns:
            None; finite predictions, parity and schema rejection are asserted.
        """
        frame = synthetic_frame()
        result = ModelTrainer(
            TrainingConfig(trials=1, max_trees=30, early_stopping_rounds=5)
        ).fit(frame)
        self.assertIn("ridge", result.report["pooled"]["partitions"]["test"])
        self.assertEqual(len(result.report["pooled"]["trials"]), 1)
        with tempfile.TemporaryDirectory() as directory:
            verification = export_bundle(
                result.models,
                directory,
                frame.iloc[100:120],
                result.metadata,
                result.report,
            )
            self.assertLessEqual(verification["max_absolute_difference"], 1e-9)
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
        result = ModelTrainer(
            TrainingConfig(mode="separate", trials=0, max_trees=5)
        ).fit(frame)
        self.assertEqual(set(result.models), {"blitz", "rapid"})
        self.assertEqual(
            result.report["combined"]["partitions"]["test"]["lightgbm"]["overall"][
                "rows"
            ],
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
