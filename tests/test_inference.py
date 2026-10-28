"""Public inference uses real exported trees and isolated engine adapters."""

import contextlib
import io
import json
import tempfile
import unittest
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

from elo_model.__main__ import main
from elo_model.analysis import AnalysisCache
from elo_model.config import (
    ANALYSIS_VERSION,
    EngineConfig,
    InferenceConfig,
    TrainingConfig,
)
from elo_model.game import GameInput
from elo_model.inference import RatingEstimator
from elo_model.predictor import export_bundle
from elo_model.storage import write_json
from elo_model.training import ModelTrainer
from tests.test_features import PGN, sample_analysis
from tests.test_model_training import synthetic_frame


class InferenceTests(unittest.TestCase):
    def test_cli_same_release_accepts_cross_build_analysis(self):
        """Forward the explicit CLI policy without changing engine metadata.

        Args:
            None.

        Returns:
            None; CLI predictions match the public predictor for another build.
        """
        runtime = {**self.metadata, "binary_sha256": "b" * 64}
        analysis = sample_analysis(GameInput.from_pgn(PGN))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pgn_path, analysis_path, output = root / "game.pgn", root / "analysis.json", root / "prediction.json"
            pgn_path.write_text(PGN, encoding="utf-8")
            write_json(analysis_path, {
                "engine": runtime, "positions": [asdict(position) for position in analysis],
            })
            with (
                patch("sys.argv", [
                    "elo_model", "predict", "--model", str(self.directory),
                    "--pgn", str(pgn_path), "--analysis", str(analysis_path),
                    "--output", str(output), "--engine-compatibility", "same-release",
                ]),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                main()
            expected = RatingEstimator(InferenceConfig(model_directory=self.directory)).predictor.predict_game(
                GameInput.from_pgn(PGN), analysis, self.metadata
            )
            self.assertEqual(json.loads(output.read_text()), expected)

    def test_same_release_accepts_cross_build_without_relabeling_provenance(self):
        """Allow a different binary hash for otherwise identical engine contracts.

        Args:
            None.

        Returns:
            None; predictions work, runtime provenance survives and cache keys differ.
        """
        runtime = {**self.metadata, "binary_sha256": "b" * 64}
        estimator = RatingEstimator(InferenceConfig(
            model_directory=self.directory, engine_compatibility="same-release"
        ))
        game = GameInput.from_pgn(PGN)
        analysis = sample_analysis(game)
        result = estimator.predict_pgn(PGN, analysis=analysis, engine_metadata=runtime)
        expected = estimator.predictor.predict_game(game, analysis, self.metadata)
        self.assertEqual((result.white, result.black), (expected["white"], expected["black"]))
        self.assertEqual(result.engine_metadata, runtime)
        self.assertEqual(result.engine_compatibility, "same-release")
        self.assertEqual(estimator.predictor.manifest["engine"], self.metadata)
        self.assertNotEqual(AnalysisCache.key(game, runtime), AnalysisCache.key(game, self.metadata))
        with self.assertRaisesRegex(ValueError, "metadata"):
            estimator.predictor.predict_game(game, analysis, runtime)

    def test_same_release_rejects_changed_settings_release_and_protocol(self):
        """Retain engine release and analysis-contract checks across platforms.

        Args:
            None.

        Returns:
            None; changed settings, release, protocol and incomplete provenance fail.
        """
        predictor = RatingEstimator(InferenceConfig(model_directory=self.directory)).predictor
        invalid = [None, {}, {**self.metadata, "binary_sha256": "invalid"}]
        for field in ("depth", "multipv", "threads", "hash_mb"):
            metadata = deepcopy(self.metadata)
            metadata["settings"][field] += 1
            invalid.append(metadata)
        invalid.append({**self.metadata, "id": {"name": "Different release"}})
        invalid.append({**self.metadata, "analysis_version": ANALYSIS_VERSION + 1})
        for metadata in invalid:
            with self.subTest(metadata=metadata), self.assertRaisesRegex(ValueError, "metadata"):
                predictor.validate_engine_metadata(metadata, engine_compatibility="same-release")

    def test_raw_pgn_same_release_uses_actual_runtime_metadata(self):
        """Apply cross-build policy before engine analysis and at prediction time.

        Args:
            None.

        Returns:
            None; raw PGN analysis accepts a different hash and reports it unchanged.
        """
        runtime = {**self.metadata, "binary_sha256": "b" * 64}
        estimator = RatingEstimator(InferenceConfig(
            model_directory=self.directory, engine_executable="stockfish", cache_directory="cache",
            engine_compatibility="same-release",
        ))
        analyzer = MagicMock(metadata=runtime, cache_hits=0)
        analyzer.__enter__.return_value = analyzer
        analyzer.analyze.return_value = ("c" * 64, sample_analysis(GameInput.from_pgn(PGN)))
        with patch("elo_model.inference.StockfishAnalyzer", return_value=analyzer):
            result = estimator.predict_pgn(PGN)
        analyzer.analyze.assert_called_once()
        self.assertEqual(result.engine_metadata, runtime)
        self.assertFalse(result.cache_hit)

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