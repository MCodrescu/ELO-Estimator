"""Estimate remaining engine work independently of cache-hit throughput."""

import pandas as pd
from tqdm.auto import tqdm

from .analysis import AnalysisCache
from .game import GameInput


class AnalysisTiming:
    """Accumulate fresh worker durations and track uncached game tasks."""

    def __init__(
        self, games: pd.DataFrame, cache: AnalysisCache, metadata: dict, workers: int
    ):
        """Identify pending fresh work before any task writes new cache files.

        Args:
            games: Input game rows in scheduling order.
            cache: Analysis cache using the current engine contract.
            metadata: Current engine identity and settings.
            workers: Number of active engine workers.

        Returns:
            None.
        """
        keys = {
            index: cache.key(GameInput.from_pgn(record["pgn"]), metadata)
            for index, record in enumerate(games.to_dict("records"))
        }
        self.pending = {
            index
            for index, key in keys.items()
            if not (cache.directory / f"{key}.parquet").exists()
        }
        self.workers = workers
        self.fresh_seconds = 0.0
        self.fresh_games = 0
        self.cached_games = 0

    def record(self, index: int, cached: bool, seconds: float) -> dict:
        """Record one completed task and produce fresh-only timing statistics.

        Args:
            index: Completed task's position in the input manifest.
            cached: Whether this task reused a cached evaluation.
            seconds: Worker duration; cache-hit duration is ignored.

        Returns:
            Progress counters, fresh seconds per game and parallel-adjusted ETA.
        """
        self.pending.discard(index)
        self.cached_games += int(cached)
        if not cached:
            self.fresh_games += 1
            self.fresh_seconds += seconds

        average_seconds = (
            self.fresh_seconds / self.fresh_games if self.fresh_games else None
        )
        remaining_games = len(self.pending)
        estimated_time = "00:00" if not remaining_games else "unknown"

        if average_seconds is not None and remaining_games:
            active_workers = min(self.workers, remaining_games)
            remaining_seconds = average_seconds * remaining_games / active_workers
            estimated_time = tqdm.format_interval(remaining_seconds)

        return {
            "cached": self.cached_games,
            "new": self.fresh_games,
            "fresh_s_per_game": f"{average_seconds:.2f}"
            if average_seconds is not None
            else "unknown",
            "fresh_remaining": remaining_games,
            "fresh_ETA": estimated_time,
        }
