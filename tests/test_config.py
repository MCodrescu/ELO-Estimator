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
    TrainingConfig,
)
from elo_model.features import FeatureConfig as ExtractionFeatureConfig
from elo_model.training import TrainingConfig as ModelTrainingConfig
from training_data.config import DatasetConfig as CollectionDatasetConfig


class ConfigTests(unittest.TestCase):
    """Check canonical definitions and independently editable workflow settings."""

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
                "depth": 12,
                "multipv": 3,
                "threads": 1,
                "hash_mb": 64,
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
