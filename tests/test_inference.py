"""Public inference uses real exported trees and isolated engine adapters."""

import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

from elo_model.config import (
    ANALYSIS_VERSION,
    EngineConfig,
    InferenceConfig,
    TrainingConfig,
)
from elo_model.game import GameInput
from elo_model.inference import RatingEstimator
from elo_model.predictor import export_bundle
from elo_model.training import ModelTrainer
from tests.test_features import PGN, sample_analysis
from tests.test_model_training import synthetic_frame


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Export one tiny real model shared by inference contract tests.

        Args:
            None.

        Returns:
            None; temporary artifacts and synthetic engine provenance are retained.
        """
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        cls.directory = Path(temporary.name)
        cls.metadata = {
            "id": {"name": "Test engine"},
            "binary_sha256": "a" * 64,
            "settings": {key: value for key, value in asdict(EngineConfig("mock")).items() if key != "executable"},
            "analysis_version": ANALYSIS_VERSION,
        }
        frame = synthetic_frame()
        result = ModelTrainer(TrainingConfig(trials=0, max_trees=5)).fit(frame)
        result.metadata.update(engine=cls.metadata)
        export_bundle(
            result.models, cls.directory, frame.iloc[-10:], result.metadata, result.report,
            model_version="test-v1",
        )

    def test_precomputed_analysis_needs_no_engine_and_matches_predictor(self):
        """Use existing scores without opening engines or calling training code.

        Args:
            None.

        Returns:
            None; predictions, model identity and engine-free operation are asserted.
        """
        estimator = RatingEstimator(InferenceConfig(model_directory=self.directory))
        game = GameInput.from_pgn(PGN)
        analysis = sample_analysis(game)
        with patch("elo_model.inference.StockfishAnalyzer") as engine:
            result = estimator.predict_pgn(PGN, analysis=analysis, engine_metadata=self.metadata)
            engine.assert_not_called()
        expected = estimator.predictor.predict_game(game, analysis, self.metadata)
        self.assertEqual(result.white, expected["white"])
        self.assertEqual(result.black, expected["black"])
        self.assertEqual(result.model_version, "test-v1")
        self.assertEqual(len(result.model_id), 64)
        self.assertIsNone(result.cache_hit)
        self.assertEqual(
            result.model_id,
            RatingEstimator(InferenceConfig(model_directory=self.directory)).model_id,
        )

    def test_raw_pgn_uses_manifest_settings_and_reports_cache_hit(self):
        """Derive engine settings from the bundle and preserve cache provenance.

        Args:
            None.

        Returns:
            None; engine options, analysis and cache reporting are asserted.
        """
        config = InferenceConfig(
            model_directory=self.directory, engine_executable="stockfish", cache_directory="cache"
        )
        estimator = RatingEstimator(config)
        analyzer = MagicMock(metadata=self.metadata, cache_hits=1)
        analyzer.__enter__.return_value = analyzer
        analyzer.analyze.return_value = ("b" * 64, sample_analysis(GameInput.from_pgn(PGN)))
        with patch("elo_model.inference.StockfishAnalyzer", return_value=analyzer) as factory:
            result = estimator.predict_pgn(PGN)
        self.assertTrue(result.cache_hit)
        self.assertEqual(result.analysis_key, "b" * 64)
        self.assertEqual(asdict(factory.call_args.args[0]), {"executable": "stockfish", **self.metadata["settings"]})
        analyzer.__exit__.assert_called_once()

    def test_mismatched_engine_is_rejected_before_analysis(self):
        """Fail before expensive analysis when runtime engine provenance differs.

        Args:
            None.

        Returns:
            None; mismatch raises and analysis is not invoked.
        """
        estimator = RatingEstimator(InferenceConfig(
            model_directory=self.directory, engine_executable="stockfish", cache_directory="cache"
        ))
        analyzer = MagicMock(metadata={**self.metadata, "binary_sha256": "wrong"})
        analyzer.__enter__.return_value = analyzer
        with (
            patch("elo_model.inference.StockfishAnalyzer", return_value=analyzer),
            self.assertRaisesRegex(ValueError, "metadata"),
        ):
            estimator.predict_pgn(PGN)
        analyzer.analyze.assert_not_called()
        analyzer.__exit__.assert_called_once()

    def test_raw_pgn_requires_explicit_engine_paths(self):
        """Require portable deployment configuration instead of local defaults.

        Args:
            None.

        Returns:
            None; missing engine and cache paths raise ValueError.
        """
        estimator = RatingEstimator(InferenceConfig(model_directory=self.directory))
        with self.assertRaisesRegex(ValueError, "engine_executable"):
            estimator.predict_pgn(PGN)