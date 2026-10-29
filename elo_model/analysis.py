"""Contract-bound Stockfish analysis and history-aware resumable Parquet cache."""

import asyncio
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import chess
import chess.engine
import pandas as pd
from tqdm.auto import tqdm

from .config import ANALYSIS_VERSION, CACHE_VERSION, CLEAR_HASH_OPTION, EngineConfig
from .game import Candidate, GameInput, PositionAnalysis, Score
from .storage import write_parquet


def fingerprint(payload) -> str:
    """Hash canonical structured input, independent of dictionary insertion order.

    Args:
        payload: JSON-compatible configuration or game history.

    Returns:
        Lowercase SHA-256 hexadecimal digest.
    """
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def analysis_from_payload(payload: list[dict]) -> list[PositionAnalysis]:
    """Decode the public White-perspective analysis contract.

    Args:
        payload: Ordered position dictionaries, as produced by dataclasses.asdict.

    Returns:
        Typed position records; malformed scores raise ValueError.
    """
    result = []
    for position in payload:
        candidates = tuple(
            Candidate(
                move=item["move"],
                score=Score(**item["score"]),
                depth=item.get("depth", 0),
                lowerbound=item.get("lowerbound", False),
                upperbound=item.get("upperbound", False),
            )
            for item in position["candidates"]
        )
        result.append(
            PositionAnalysis(
                fen=position["fen"],
                candidates=candidates,
                depth=position.get("depth", 0),
                evaluation=Score(**position["evaluation"])
                if position.get("evaluation")
                else None,
            )
        )
    return result


class AnalysisCache:
    """One atomic Parquet file per game history and engine fingerprint."""

    def __init__(self, directory: str | Path):
        """Select a local cache without creating files yet.

        Args:
            directory: Root directory for per-game Parquet files.

        Returns:
            None.
        """
        self.directory = Path(directory)

    @staticmethod
    def key(game: GameInput, engine_metadata: dict) -> str:
        """Identify analysis by history and engine, not by labels or clocks.

        Args:
            game: Initial position and complete chronological move history.
            engine_metadata: Binary/version/settings fingerprint.

        Returns:
            Stable cache key for these engine inputs.
        """
        return fingerprint(
            {
                "version": CACHE_VERSION,
                "initial_fen": game.initial_fen,
                "moves": game.moves,
                "engine": engine_metadata,
            }
        )

    def load(self, key: str) -> list[PositionAnalysis] | None:
        """Read a complete cached analysis without starting Stockfish.

        Args:
            key: Canonical SHA-256 analysis key.

        Returns:
            Ordered position records, or None when not cached.
        """
        if len(key) != 64 or any(
            character not in "0123456789abcdef" for character in key
        ):
            raise ValueError("Invalid analysis cache key")
        path = self.directory / f"{key}.parquet"
        if not path.exists():
            return None
        frame = pd.read_parquet(path)
        if len(frame) != 1 or frame.iloc[0]["key"] != key:
            raise ValueError(f"Invalid cache record: {path}")
        return analysis_from_payload(json.loads(frame.iloc[0]["analysis"]))

    def save(self, key: str, analysis: list[PositionAnalysis]) -> None:
        """Atomically persist raw scores so features can be recalculated.

        Args:
            key: Canonical game/engine cache key.
            analysis: Initial and reached-position results in replay order.

        Returns:
            None.
        """
        write_parquet(
            self.directory / f"{key}.parquet",
            pd.DataFrame(
                [
                    {
                        "key": key,
                        "analysis": json.dumps(
                            [asdict(position) for position in analysis], allow_nan=False
                        ),
                    }
                ]
            ),
        )


class StockfishAnalyzer:
    """Search each reached board once with game-local hash retention."""

    def __init__(self, config: EngineConfig, cache: AnalysisCache):
        """Prepare an adapter; enter its context to launch Stockfish.

        Args:
            config: Executable location and bounded analysis settings.
            cache: Local resumable result store.

        Returns:
            None.
        """
        self.config = config
        self.cache = cache
        self.engine = None
        self.metadata = None
        self.cache_hits = 0
        self.analyzed_games = 0

    def __enter__(self):
        """Start the supplied executable and fingerprint its configuration.

        Windows selector policies are replaced for subprocess-capable background
        loops; an already-running notebook event loop is not replaced.

        Args:
            None.

        Returns:
            This ready-to-use analyzer.
        """
        path = Path(self.config.executable).resolve(strict=True)
        with path.open("rb") as handle:
            binary_hash = hashlib.file_digest(handle, "sha256").hexdigest()
        if sys.platform == "win32" and isinstance(
            asyncio.get_event_loop_policy(), asyncio.WindowsSelectorEventLoopPolicy
        ):
            asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
        self.engine = chess.engine.SimpleEngine.popen_uci(str(path))
        try:
            self.engine.configure(
                {"Threads": self.config.threads, "Hash": self.config.hash_mb}
            )
        except (chess.engine.EngineError, ValueError):
            self.engine.quit()
            self.engine = None
            raise
        settings = asdict(self.config)
        settings.pop("executable")
        self.metadata = {
            "id": self.engine.id,
            "binary_sha256": binary_hash,
            "settings": settings,
            "analysis_version": ANALYSIS_VERSION,
        }
        self.metadata.update({
            "initialization": "clear-hash-per-game",
            "position_order": "initial-then-reached-chronological-full-history",
            "uci_defaults": {
                name: option.default for name, option in self.engine.options.items()
                if name not in (CLEAR_HASH_OPTION, "Threads", "Hash", "MultiPV")
            },
        })
        return self

    def __exit__(self, exception_type, exception, traceback):
        """Shut down the UCI process without suppressing caller exceptions.

        Args:
            exception_type: Exception class from the managed block, if any.
            exception: Exception instance from the block, if any.
            traceback: Original traceback, if any.

        Returns:
            None, allowing exceptions to propagate.
        """
        if self.engine is not None:
            self.engine.quit()
            self.engine = None

    @staticmethod
    def _score(info: dict) -> Score:
        """Normalize python-chess mate signs, including positive/negative zero.

        Args:
            info: Engine info dictionary containing a PovScore.

        Returns:
            Exact White-perspective centipawn or signed mate score.
        """
        score = info["score"].white()
        mate = score.mate()
        if mate is None:
            return Score(cp=score.score())
        if mate == 0:
            mate = 1 if score.score(mate_score=10000) > 0 else -1
        return Score(mate=mate)

    def _position(self, board: chess.Board) -> PositionAnalysis:
        """Analyze one root within its total search allowance.

        Args:
            board: Position with full history retained for repetition rules.

        Returns:
            Exact single-PV evaluation or terminal outcome.
        """
        outcome = board.outcome(claim_draw=False)
        if outcome is not None:
            evaluation = Score(cp=0)
            if outcome.winner is not None:
                evaluation = Score(mate=1 if outcome.winner else -1)
            return PositionAnalysis(
                board.fen(), (), depth=0, evaluation=evaluation
            )
        limit = chess.engine.Limit(
            depth=self.config.depth, time=self.config.time_seconds,
            nodes=self.config.nodes,
        )
        infos = self._bounded_infos(board, limit)
        candidates = tuple(
            Candidate(
                info["pv"][0].uci(),
                self._score(info),
                info.get("depth", 0),
                info.get("lowerbound", False),
                info.get("upperbound", False),
            )
            for info in infos
        )
        return PositionAnalysis(
            board.fen(), candidates, candidates[0].depth, candidates[0].score
        )

    def _bounded_infos(self, board: chess.Board, limit: chess.engine.Limit) -> list[dict]:
        """Keep the last complete exact iteration within a single bounded search.

        Args:
            board: Root with complete repetition history.
            limit: Total time/depth/node allowance for this reached position.

        Returns:
            Exact ranked scores at one completed depth; no synthesized candidates.
        """
        latest = None
        with self.engine.analysis(board, limit, multipv=self.config.multipv) as stream:
            for info in stream:
                if not info.get("pv") or "score" not in info or "depth" not in info:
                    continue
                if info.get("lowerbound") or info.get("upperbound"):
                    continue
                if info.get("multipv", 1) == 1 and (latest is None or info["depth"] >= latest["depth"]):
                    latest = info
        if latest is not None:
            return [latest]
        raise ValueError("Search budget produced no complete exact candidate iteration")

    def analyze(
        self, game: GameInput, progress: bool = False
    ) -> tuple[str, list[PositionAnalysis]]:
        """Reuse a complete cached game or analyze every reached position.

        Args:
            game: Parsed standard-chess game, including any starting FEN.
            progress: Show a per-position bar only when new analysis is required.

        Returns:
            Cache key and ordered position records.
        """
        if self.engine is None or self.metadata is None:
            raise RuntimeError("Use StockfishAnalyzer as a context manager")
        key = self.cache.key(game, self.metadata)
        cached = self.cache.load(key)
        if cached is not None:
            self.cache_hits += 1
            return key, cached
        if CLEAR_HASH_OPTION not in self.engine.options:
            raise ValueError("Game-local hash policy requires Clear Hash support")
        self.engine.configure({CLEAR_HASH_OPTION: None})
        boards = tqdm(
            game.boards(),
            desc="Stockfish positions",
            unit="position",
            disable=not progress,
            leave=False,
        )
        result = [self._position(board) for board in boards]
        self.cache.save(key, result)
        self.analyzed_games += 1
        return key, result
