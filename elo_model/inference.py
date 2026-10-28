"""Reusable PGN-to-rating inference without collection or training dependencies."""

import hashlib
import json

from .analysis import AnalysisCache, StockfishAnalyzer
from .config import EngineCompatibility, EngineConfig, InferenceConfig
from .game import GameInput, PositionAnalysis
from .predictor import RatingPredictions, RatingPredictor


class GameRatingEstimate(RatingPredictions):
    """Player estimates with model identity and optional engine-cache provenance."""

    model_id: str
    model_version: str | None = None
    category: str | None
    analysis_key: str | None = None
    cache_hit: bool | None = None
    engine_compatibility: EngineCompatibility = "exact"
    engine_metadata: dict | None = None


class RatingEstimator:
    """Load a model once; analyze and predict individual PGNs on demand."""

    def __init__(self, config: InferenceConfig):
        """Load trusted model files without network calls or engine startup.

        Args:
            config: Model location and optional engine/cache locations.

        Returns:
            None; models and a content-derived identifier are retained for reuse.
        """
        self.config = config
        self.predictor = RatingPredictor(config.model_directory)
        digest = hashlib.sha256(
            json.dumps(self.predictor.manifest, sort_keys=True, allow_nan=False).encode()
        )
        for name, filename in sorted(self.predictor.manifest["models"].items()):
            digest.update(name.encode())
            with (config.model_directory / filename).open("rb") as handle:
                digest.update(hashlib.file_digest(handle, "sha256").digest())
        self.model_id = digest.hexdigest()

    def predict_pgn(
        self,
        pgn: str,
        *,
        analysis: list[PositionAnalysis] | None = None,
        engine_metadata: dict | None = None,
        progress: bool = False,
    ) -> GameRatingEstimate:
        """Predict both players from a PGN and compatible supplied or new analysis.

        Args:
            pgn: One legal standard-chess PGN, preferably completed and clocked.
            analysis: Optional precomputed scores aligned with every reached board.
            engine_metadata: Provenance for supplied scores, matching the model.
            progress: Show position progress when new engine analysis is required.

        Returns:
            Validated ratings, model identity, category and available cache details.
        """
        game = GameInput.from_pgn(pgn)
        key, cache_hit = None, None
        if analysis is None:
            if engine_metadata is not None:
                raise ValueError("engine_metadata requires precomputed analysis")
            if self.config.engine_executable is None or self.config.cache_directory is None:
                raise ValueError("Raw PGN inference requires engine_executable and cache_directory")
            expected = self.predictor.manifest.get("engine")
            if not isinstance(expected, dict) or not isinstance(expected.get("settings"), dict):
                raise ValueError("Model bundle has no engine settings for raw PGN analysis")
            config = EngineConfig(
                executable=str(self.config.engine_executable), **expected["settings"]
            )
            with StockfishAnalyzer(config, AnalysisCache(self.config.cache_directory)) as analyzer:
                self.predictor.validate_engine_metadata(
                    analyzer.metadata, engine_compatibility=self.config.engine_compatibility
                )
                key, analysis = analyzer.analyze(game, progress=progress)
                engine_metadata = analyzer.metadata
                cache_hit = analyzer.cache_hits > 0
        predictions = self.predictor.predict_game(
            game, analysis, engine_metadata,
            engine_compatibility=self.config.engine_compatibility,
        )
        return GameRatingEstimate(
            **predictions,
            model_id=self.model_id,
            model_version=self.predictor.manifest.get("model_version"),
            category=game.time_control()[2],
            analysis_key=key,
            cache_hit=cache_hit,
            engine_compatibility=self.config.engine_compatibility,
            engine_metadata=engine_metadata,
        )