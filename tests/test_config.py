"""Centralized configuration retains component and artifact compatibility."""

import unittest
from dataclasses import asdict, replace
from pathlib import Path

from elo_model.analysis import EngineConfig as AnalysisEngineConfig
from elo_model.config import (
    DatasetConfig,
    EngineConfig,
    FeatureConfig,
    PipelineConfig,
    StratificationConfig,
    TrainingConfig,
)
from elo_model.features import FeatureConfig as ExtractionFeatureConfig
from elo_model.training import TrainingConfig as ModelTrainingConfig
from training_data.config import DatasetConfig as CollectionDatasetConfig


class ConfigTests(unittest.TestCase):
    """Check canonical definitions and independently editable workflow settings."""

    def test_defaults_match_fast_bundle(self):
        """Use the selected fast search and schema by default.

        Args:
            None.

        Returns:
            None; fast search settings and schema are asserted.
        """
        fast = PipelineConfig()
        self.assertIsNone(fast.engine.depth)
        self.assertEqual(fast.engine.time_seconds, 0.1)
        self.assertEqual(fast.engine.multipv, 1)
        self.assertEqual(fast.engine.hash_reset, "game")
        self.assertEqual(fast.engine.played_move_scoring, "next-position")
        self.assertEqual(fast.features.schema_version, 2)
        self.assertEqual(fast.artifact_directory, Path("artifacts/rating_model"))

    def test_stratification_validation_and_candidate_budget(self):
        """Validate rating/category quotas without changing engine fingerprints.

        Args:
            None.

        Returns:
            None; budgets, interval constraints and unchanged engine settings are asserted.
        """
        settings = PipelineConfig()
        self.assertEqual(settings.candidate_games, 3000)
        disabled = replace(settings, stratification=StratificationConfig())
        self.assertEqual(disabled.candidate_games, settings.target_games)
        self.assertEqual(disabled.engine, settings.engine)
        self.assertEqual(disabled.engine_cache_directory, settings.engine_cache_directory)
        invalid = [
            {"candidate_multiplier": 0},
            {"max_rows_per_player": True},
            {"quotas": [{"lower": 1400, "upper": 1000, "category": "blitz", "rows": 5}]},
            {"quotas": [{"lower": 0, "category": "daily", "rows": 5}]},
            {"quotas": [{"lower": 0, "category": "blitz", "rows": 0}]},
            {"quotas": [
                {"lower": 0, "upper": 1400, "category": "blitz", "rows": 5},
                {"lower": 1000, "category": "blitz", "rows": 5},
            ]},
        ]
        for payload in invalid:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                StratificationConfig.model_validate(payload)

    def test_components_import_the_same_configuration_classes(self):
        """Keep compatibility aliases without duplicate class definitions.

        Args:
            None.

        Returns:
            None; shared class identity and stable engine serialization are asserted.
        """
        self.assertIs(AnalysisEngineConfig, EngineConfig)
        self.assertIs(ExtractionFeatureConfig, FeatureConfig)
        self.assertIs(ModelTrainingConfig, TrainingConfig)
        self.assertIs(CollectionDatasetConfig, DatasetConfig)

        self.assertEqual(
            asdict(EngineConfig("stockfish")),
            {
                "executable": "stockfish",
                "depth": None,
                "multipv": 1,
                "threads": 1,
                "hash_mb": 64,
                "time_seconds": 0.1,
                "nodes": None,
                "hash_reset": "game",
                "played_move_scoring": "next-position",
            },
        )

    def test_workflow_target_does_not_invalidate_engine_cache(self):
        """Allow a larger dataset or new run directory to reuse engine results.

        Args:
            None.

        Returns:
            None; cache location, feature settings and engine settings stay fixed.
        """
        settings = PipelineConfig()
        larger_run = replace(
            settings, target_games=2000, data_directory=Path("data/larger")
        )

        self.assertEqual(settings.target_games, 1000)
        self.assertTrue(settings.training.progress)
        self.assertEqual(settings.collection.max_games_per_player, 10)
        self.assertEqual(
            larger_run.engine_cache_directory, settings.engine_cache_directory
        )
        self.assertEqual(larger_run.engine, settings.engine)
        self.assertEqual(larger_run.features, settings.features)
        self.assertEqual(larger_run.raw_games_path, Path("data/larger/games.parquet"))

    def test_invalid_targets_fail_before_external_work(self):
        """Reject nonpositive and noninteger collection targets.

        Args:
            None.

        Returns:
            None; ValueError is asserted for each invalid target.
        """
        for target in (0, -1, True, 1.5):
            with self.subTest(target=target), self.assertRaises(ValueError):
                PipelineConfig(target_games=target)


if __name__ == "__main__":
    unittest.main()
