"""One source for collection, engine, feature, training and workflow settings."""

import math
from dataclasses import asdict, dataclass, field
from datetime import date
from itertools import pairwise
from pathlib import Path
from typing import Literal, Self

import chess
from pydantic import BaseModel, ConfigDict, Field, model_validator

PARTITIONS = ("train", "validation", "test")
DEFAULT_SEED = 42
DEFAULT_ANALYSIS_WORKERS = 8
DEFAULT_SPLIT_FRACTIONS = (0.8, 0.1, 0.1)
TIME_CATEGORIES = ("bullet", "blitz", "rapid", "classical")
TIME_CATEGORY_LIMITS = (180, 600, 3600, math.inf)
TIME_CONTROL_INCREMENT_MOVES = 40
RATING_BAND_EDGES = (0, 1000, 1400, 1800, 2200, math.inf)
RATING_BAND_NAMES = ("<1000", "1000-1400", "1400-1800", "1800-2200", "2200+")
PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 300,
    chess.BISHOP: 300,
    chess.ROOK: 500,
    chess.QUEEN: 900,
}
ANALYSIS_VERSION = 1
FEATURE_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class DatasetConfig:
    """Collection, discovery and HTTP retry settings."""

    country_codes: tuple[str, ...] = ("CA",)
    months: tuple[tuple[int, int], ...] = ((2026, 10),)
    player_limit: int = 10
    max_games_per_player: int = 50

    seed_username: str = "MCodr"
    search_max_players: int = 100
    search_max_depth: int | None = None
    search_time_class: str = "blitz"
    search_max_expansions: int = 1000
    search_band_edges: tuple[int, ...] = tuple(range(0, 2501, 250))

    user_agent: str = "ChessTrainingResearch/0.1"
    proxy_url: str = "http://sub.proxy.att.com:8080"
    request_delay: float = 1.0
    retry_delay: float = 5.0
    retry_backoff: float = 2.0
    retries: int = 2
    max_workers: int = 4

    log_level: str = "INFO"
    log_format: str = "%(asctime)s %(levelname)s [%(threadName)s] %(message)s"

    def __post_init__(self) -> None:
        """Validate settings before any network work.

        Args:
            None.

        Returns:
            None; invalid settings raise ValueError.
        """
        self._validate_collection()
        self._validate_discovery()
        self._validate_transport()

    def _validate_collection(self) -> None:
        """Check country, month and contribution limits.

        Args:
            None.

        Returns:
            None; invalid collection settings raise ValueError.
        """
        if not self.country_codes or any(
            len(code) != 2 or not code.isascii() or not code.isalpha()
            for code in self.country_codes
        ):
            raise ValueError("Provide two-letter country codes")

        if not self.months or len(set(self.months)) != len(self.months):
            raise ValueError("Provide at least one month, without duplicates")

        for year, month in self.months:
            date(year, month, 1)

        if self.player_limit < 1:
            raise ValueError("player_limit must be positive")

        invalid_limit = isinstance(self.max_games_per_player, bool) or not isinstance(
            self.max_games_per_player, int
        )

        if invalid_limit or self.max_games_per_player < 1:
            raise ValueError("max_games_per_player must be a positive integer")

    def _validate_discovery(self) -> None:
        """Check optional player-discovery controls.

        Args:
            None.

        Returns:
            None; invalid discovery settings raise ValueError.
        """
        if not self.seed_username.strip():
            raise ValueError("seed_username must not be empty")

        if self.search_max_players < 1:
            raise ValueError("search_max_players must be positive")

        if self.search_max_depth is not None and self.search_max_depth < 0:
            raise ValueError("search_max_depth must be nonnegative or None")

        if self.search_time_class not in ("blitz", "rapid", "bullet"):
            raise ValueError("search_time_class must be blitz, rapid, or bullet")

        invalid_budget = isinstance(self.search_max_expansions, bool) or not isinstance(
            self.search_max_expansions, int
        )

        if invalid_budget or self.search_max_expansions < 1:
            raise ValueError("search_max_expansions must be a positive integer")

        if (
            len(self.search_band_edges) < 2
            or self.search_band_edges[0] != 0
            or any(
                not isinstance(edge, int) or isinstance(edge, bool)
                for edge in self.search_band_edges
            )
            or any(
                left >= right
                for left, right in zip(
                    self.search_band_edges, self.search_band_edges[1:]
                )
            )
        ):
            raise ValueError(
                "search_band_edges must be increasing integers starting at 0"
            )

    def _validate_transport(self) -> None:
        """Check retry, logging and worker settings.

        Args:
            None.

        Returns:
            None; invalid transport settings raise ValueError.
        """
        if not self.user_agent.strip():
            raise ValueError("user_agent must not be empty")

        if self.request_delay < 0 or self.retry_delay < 0:
            raise ValueError("Delays must be nonnegative")

        if self.retries < 0:
            raise ValueError("retries must be nonnegative")

        if self.retry_backoff < 1:
            raise ValueError("retry_backoff must be at least 1")

        if self.max_workers < 1:
            raise ValueError("max_workers must be positive")

        if self.log_level not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
            raise ValueError(
                "log_level must be DEBUG, INFO, WARNING, ERROR, or CRITICAL"
            )


@dataclass(frozen=True)
class FeatureConfig:
    """Versioned feature thresholds shared by training and inference."""

    probability_scale: float = 200
    mate_cp: float = 10000

    opening_fullmoves: int = 10
    endgame_material: int = 2600

    winning_probability: float = 0.75
    losing_probability: float = 0.25
    inaccuracy: float = 0.05
    mistake: float = 0.10
    blunder: float = 0.20

    complexity_low: float = 0.05
    only_move_gap: float = 0.15

    instant_seconds: float = 1
    trouble_fraction: float = 0.10
    clock_jitter: float = 0.05

    def __post_init__(self) -> None:
        """Check finite settings and threshold ordering.

        Args:
            None.

        Returns:
            None; invalid settings raise ValueError.
        """
        if not all(math.isfinite(value) for value in asdict(self).values()):
            raise ValueError("Feature settings must be finite")

        if self.probability_scale <= 0 or self.mate_cp <= 0:
            raise ValueError("Score scales must be positive")

        if not 0 < self.inaccuracy < self.mistake < self.blunder < 1:
            raise ValueError("Error thresholds must increase between zero and one")

        if not 0 < self.losing_probability < self.winning_probability < 1:
            raise ValueError("Position probability thresholds must increase")

        if not 0 < self.complexity_low < self.only_move_gap < 1:
            raise ValueError("Complexity thresholds must increase")

        if (
            min(
                self.instant_seconds,
                self.clock_jitter,
                self.opening_fullmoves,
                self.endgame_material,
            )
            < 0
        ):
            raise ValueError("Invalid phase or clock settings")

        if not 0 < self.trouble_fraction < 1:
            raise ValueError("Invalid phase or clock settings")


@dataclass(frozen=True)
class EngineConfig:
    """Fixed-depth analysis settings; the engine is installed separately."""

    executable: str
    depth: int = 12
    multipv: int = 3
    threads: int = 1
    hash_mb: int = 64

    def __post_init__(self) -> None:
        """Reject invalid engine limits before starting a process.

        Args:
            None.

        Returns:
            None; invalid settings raise ValueError.
        """
        limits = (self.depth, self.threads, self.hash_mb)

        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in limits
        ):
            raise ValueError("Depth, threads and hash_mb must be positive integers")

        if self.multipv != 3:
            raise ValueError("The feature contract requires MultiPV=3")


@dataclass(frozen=True)
class TrainingConfig:
    """Bounded reproducible tuning and optional progress display."""

    mode: str = "pooled"
    trials: int = 20
    seed: int = DEFAULT_SEED

    max_trees: int = 500
    early_stopping_rounds: int = 30
    min_rows: int = 10
    threads: int = 1

    balance_bands: bool = False
    balance_players: bool = False
    progress: bool = False

    def __post_init__(self) -> None:
        """Check model modes and bounded experiment sizes.

        Args:
            None.

        Returns:
            None; invalid limits raise ValueError.
        """
        if self.mode not in ("pooled", "separate") or self.trials < 0:
            raise ValueError("Use pooled/separate mode and nonnegative trials")

        if (
            min(self.max_trees, self.early_stopping_rounds, self.min_rows, self.threads)
            < 1
        ):
            raise ValueError("Training limits must be positive")


@dataclass(frozen=True)
class StorageConfig:
    """Bounded handling of transient Windows checkpoint replacement locks."""

    replace_attempts: int = 5
    retry_delay: float = .05

    def __post_init__(self) -> None:
        """Reject unbounded or invalid retry settings.

        Args:
            None.

        Returns:
            None; invalid retry limits raise ValueError.
        """
        if self.replace_attempts < 1:
            raise ValueError("replace_attempts must be positive")

        if not math.isfinite(self.retry_delay) or self.retry_delay < 0:
            raise ValueError("retry_delay must be finite and nonnegative")


class InferenceConfig(BaseModel):
    """Portable paths for loading weights and optionally analyzing new PGNs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model_directory: Path
    engine_executable: Path | None = None
    cache_directory: Path | None = None


class RatingQuota(BaseModel):
    """Requested player-side rows for one game-time rating/category interval."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lower: int = Field(ge=0, strict=True)
    upper: int | None = Field(default=None, gt=0, strict=True)
    category: Literal["bullet", "blitz", "rapid", "classical"]
    rows: int = Field(gt=0, strict=True)

    @model_validator(mode="after")
    def validate_interval(self) -> Self:
        """Require a nonempty half-open rating interval.

        Args:
            None.

        Returns:
            Validated quota; reversed intervals raise ValueError.
        """
        if self.upper is not None and self.upper <= self.lower:
            raise ValueError("upper must exceed lower")
        return self

    @property
    def label(self) -> str:
        """Describe the inclusive lower and exclusive upper rating bounds.

        Args:
            None.

        Returns:
            Human-readable rating-band label.
        """
        if self.upper is None:
            return f"{self.lower}+"
        if self.lower == 0:
            return f"<{self.upper}"
        return f"{self.lower}-{self.upper}"


class StratificationConfig(BaseModel):
    """Training-only quotas, processed in order under a shared player-row cap.

    An empty quota tuple disables selection and candidate-pool expansion.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    quotas: tuple[RatingQuota, ...] = ()
    candidate_multiplier: int = Field(default=3, ge=1, strict=True)
    max_rows_per_player: int = Field(default=10, ge=1, strict=True)

    @model_validator(mode="after")
    def validate_quotas(self) -> Self:
        """Reject overlapping intervals within the same time category.

        Args:
            None.

        Returns:
            Validated settings; ambiguous quota membership raises ValueError.
        """
        for category in TIME_CATEGORIES:
            quotas = sorted(
                (quota for quota in self.quotas if quota.category == category),
                key=lambda quota: quota.lower,
            )
            for left, right in pairwise(quotas):
                if left.upper is None or left.upper > right.lower:
                    raise ValueError(f"Overlapping rating quotas for {category}")
        return self


def default_stratification() -> StratificationConfig:
    """Provide editable coverage targets across ratings and time categories.

    Args:
        None.

    Returns:
        Training quotas with a larger blitz allocation and smaller other groups.
    """
    return StratificationConfig(
        quotas=tuple(
            RatingQuota(
                lower=int(lower),
                upper=None if math.isinf(upper) else int(upper),
                category=category,
                rows=500 if category == "blitz" else 100,
            )
            for lower, upper in pairwise(RATING_BAND_EDGES)
            for category in TIME_CATEGORIES
        )
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Editable run settings for the saved-usernames notebook workflow."""

    target_games: int = 1000
    preview_rows: int = 10
    usernames_path: Path = Path("player_usernames.json")

    data_directory: Path = Path("data/notebook")
    engine_cache_directory: Path = Path("data/analysis_cache")
    artifact_directory: Path = Path("artifacts/rating_model")
    training_directory: Path = Path("runs/latest")

    split_fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS
    min_players_per_partition: int = 5
    strict_participants: bool = False
    stratification: StratificationConfig = field(default_factory=default_stratification)

    collection: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(
            max_games_per_player=10,
            log_level="WARNING",
        )
    )
    engine: EngineConfig = field(
        default_factory=lambda: EngineConfig(
            executable=(
                "C:/Users/mc678p/Downloads/stockfish-windows-x86-64-universal/"
                "stockfish/stockfish-windows-x86-64-universal.exe"
            ),
        )
    )
    features: FeatureConfig = field(default_factory=FeatureConfig)
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            min_rows=30,
            balance_bands=True,
            progress=True,
        )
    )

    def __post_init__(self) -> None:
        """Validate workflow targets without filesystem or engine work.

        Args:
            None.

        Returns:
            None; invalid targets and splits raise ValueError.
        """
        if (
            isinstance(self.target_games, bool)
            or not isinstance(self.target_games, int)
            or self.target_games < 1
        ):
            raise ValueError("target_games must be a positive integer")

        if len(self.split_fractions) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in self.split_fractions
        ):
            raise ValueError("Provide three positive split fractions")

        if not math.isclose(sum(self.split_fractions), 1):
            raise ValueError("Split fractions must sum to one")

        if self.min_players_per_partition < 2:
            raise ValueError("min_players_per_partition must be at least two")

    @property
    def candidate_games(self) -> int:
        """Budget a larger raw pool when training quotas are enabled.

        Args:
            None.

        Returns:
            Candidate-game collection target, independent of engine cache settings.
        """
        multiplier = (
            self.stratification.candidate_multiplier if self.stratification.quotas else 1
        )
        return self.target_games * multiplier

    @property
    def raw_games_path(self) -> Path:
        """Locate the resumable raw game checkpoint.

        Args:
            None.

        Returns:
            Raw game Parquet path below data_directory.
        """
        return self.data_directory / "games.parquet"

    @property
    def collection_settings_path(self) -> Path:
        """Locate the collection provenance checkpoint.

        Args:
            None.

        Returns:
            Collection settings JSON path.
        """
        return self.data_directory / "collection_settings.json"

    @property
    def split_path(self) -> Path:
        """Locate the selected game/target split manifest.

        Args:
            None.

        Returns:
            Split manifest Parquet path.
        """
        return self.data_directory / "split.parquet"

    @property
    def analyzed_path(self) -> Path:
        """Locate analyzed-game cache references.

        Args:
            None.

        Returns:
            Analyzed game manifest Parquet path.
        """
        return self.data_directory / "analyzed.parquet"

    @property
    def features_path(self) -> Path:
        """Locate the extracted feature table.

        Args:
            None.

        Returns:
            Feature table Parquet path.
        """
        return self.data_directory / "features.parquet"


CLI_CONFIG = PipelineConfig(
    data_directory=Path("data"),
    collection=DatasetConfig(),
    training=TrainingConfig(),
)
