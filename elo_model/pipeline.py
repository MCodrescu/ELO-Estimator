"""File-based orchestration; expensive stages run only when explicitly requested."""

import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .analysis import AnalysisCache, StockfishAnalyzer
from .config import (
    DEFAULT_ANALYSIS_WORKERS,
    EngineConfig,
    FeatureConfig,
    validate_feature_engine,
)
from .feature_schema import feature_schema
from .features import GameFeatureExtractor
from .game import GameInput
from .split import PARTITIONS, assert_disjoint
from .storage import write_json, write_parquet


def manifest_rows(games: pd.DataFrame) -> pd.DataFrame:
    """Expand eligible target rows for preflight without running Stockfish.

    Args:
        games: Game manifest with side masks and partition labels.

    Returns:
        Side metadata rows, with normalized target identities and labels.
    """
    records = []
    for game in games.to_dict("records"):
        for color in ("white", "black"):
            if game[f"keep_{color}"]:
                records.append(
                    {
                        "game_id": game["game_id"],
                        "partition": game["partition"],
                        "target_username": game[f"{color}_username"].strip().casefold(),
                        "rating": game[f"{color}_rating"],
                        "category": game["category"],
                    }
                )
    return pd.DataFrame(
        records,
        columns=("game_id", "partition", "target_username", "rating", "category"),
    )


def split_summary(games: pd.DataFrame) -> dict:
    """Summarize retained target rows after checking isolation.

    Args:
        games: Game manifest with partition assignments and side masks.

    Returns:
        Per-partition row, game and distinct-target counts.
    """
    rows = manifest_rows(games)
    assert_disjoint(rows)
    return {
        name: {
            "rows": int((rows["partition"] == name).sum()),
            "games": int(rows.loc[rows["partition"] == name, "game_id"].nunique()),
            "target_players": int(
                rows.loc[rows["partition"] == name, "target_username"].nunique()
            ),
        }
        for name in PARTITIONS
    }


def analyze_dataset(
    games: pd.DataFrame,
    config: EngineConfig,
    cache_directory: str | Path,
    output: str | Path,
    progress: bool = False,
    workers: int = DEFAULT_ANALYSIS_WORKERS,
) -> dict:
    """Analyze split-retained games and checkpoint the index after each game.

    Args:
        games: Split manifest containing raw PGNs and eligible side masks.
        config: Supplied executable and bounded single-PV engine settings.
        cache_directory: Local Parquet analysis store.
        output: Destination analyzed manifest; engine provenance is a sidecar.
        progress: Display game progress and cumulative cache-hit counts; show
            position progress only when using one worker.
        workers: Maximum simultaneous engines, each with config.threads threads.

    Returns:
        Engine binary/version/configuration metadata.
    """
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    cache = AnalysisCache(cache_directory)
    records = {}
    remaining = iter(enumerate(games.to_dict("records")))
    with ExitStack() as stack:
        analyzers = [
            stack.enter_context(StockfishAnalyzer(config, cache))
            for _ in range(min(workers, max(1, len(games))))
        ]
        metadata = analyzers[0].metadata
        if any(analyzer.metadata != metadata for analyzer in analyzers):
            raise ValueError("Analysis workers have inconsistent engine metadata")
        sidecar = Path(output).with_suffix(".engine.json")
        previous = json.loads(sidecar.read_text(encoding="utf-8")) if sidecar.exists() else metadata
        if Path(output).exists() and previous != metadata:
            raise ValueError("Use a separate analyzed manifest for each engine contract")
        write_json(Path(output).with_suffix(".engine.json"), metadata)
        executor = stack.enter_context(ThreadPoolExecutor(max_workers=len(analyzers)))
        pending = {}
        for analyzer in analyzers:
            item = next(remaining, None)
            if item is not None:
                index, record = item
                future = executor.submit(
                    analyzer.analyze,
                    GameInput.from_pgn(record["pgn"]),
                    progress=progress and workers == 1,
                )
                pending[future] = (index, record, analyzer)
        with tqdm(
            total=len(games),
            desc="Analyze games",
            unit="game",
            disable=not progress,
        ) as bar:
            while pending:
                completed, _ = wait(pending, return_when=FIRST_COMPLETED)
                for future in completed:
                    index, record, analyzer = pending.pop(future)
                    key, _ = future.result()
                    records[index] = {**record, "analysis_key": key}
                    write_parquet(
                        output, pd.DataFrame([records[index] for index in sorted(records)])
                    )
                    bar.update(1)
                    bar.set_postfix(
                        cached=sum(worker.cache_hits for worker in analyzers),
                        new=sum(worker.analyzed_games for worker in analyzers),
                        refresh=False,
                    )
                    item = next(remaining, None)
                    if item is not None:
                        index, record = item
                        following = executor.submit(
                            analyzer.analyze,
                            GameInput.from_pgn(record["pgn"]),
                            progress=progress and workers == 1,
                        )
                        pending[following] = (index, record, analyzer)
    return metadata


def extract_dataset(
    games: pd.DataFrame,
    cache_directory: str | Path,
    config: FeatureConfig | None = None,
    progress: bool = False,
    engine_metadata: dict | None = None,
) -> pd.DataFrame:
    """Recompute fixed features entirely from cached engine output.

    Args:
        games: Analyzed manifest with cache keys and row-selection metadata.
        cache_directory: Existing analysis cache; missing records are errors.
        config: Feature transformations; defaults to versioned thresholds.
        progress: Display a game-level extraction bar, without running Stockfish.
        engine_metadata: Expected engine contract; required for fast schemas.

    Returns:
        Eligible numeric side rows with labels/identities in separate columns.
    """
    extractor = GameFeatureExtractor(config)
    validate_feature_engine(extractor.config, engine_metadata)
    cache = AnalysisCache(cache_directory)
    records = []
    for game in tqdm(
        games.to_dict("records"),
        desc="Extract features",
        unit="game",
        disable=not progress,
    ):
        parsed = GameInput.from_pgn(game["pgn"])
        if engine_metadata is not None and cache.key(parsed, engine_metadata) != game["analysis_key"]:
            raise ValueError("Analysis cache key does not match the requested engine contract")
        analysis = cache.load(game["analysis_key"])
        if analysis is None:
            raise ValueError(
                f"Missing cached analysis for {game['game_id']}; run analyze first"
            )
        matrix = extractor.extract(parsed, analysis)
        for index, color in enumerate(("white", "black")):
            if not game[f"keep_{color}"]:
                continue
            records.append(
                {
                    **dict(zip(extractor.schema.names, matrix[index])),
                    "game_id": game["game_id"],
                    "partition": game["partition"],
                    "target_username": game[f"{color}_username"].strip().casefold(),
                    "rating": game[f"{color}_rating"],
                    "category": game["category"],
                }
            )
    frame = pd.DataFrame(
        records,
        columns=(
            *extractor.schema.names,
            "game_id",
            "partition",
            "target_username",
            "rating",
            "category",
        ),
    )
    assert_disjoint(frame)
    return frame


def save_feature_dataset(
    frame: pd.DataFrame,
    output: str | Path,
    engine_metadata: dict,
    config: FeatureConfig | None = None,
) -> None:
    """Persist features and their transformation/engine provenance together.

    Args:
        frame: Extracted side feature table and metadata.
        output: Destination Parquet filename.
        engine_metadata: Engine fingerprint used for cached analysis.
        config: Exact feature settings used during extraction.

    Returns:
        None; writes Parquet plus a strict JSON metadata sidecar.
    """
    config = config or FeatureConfig()
    schema = feature_schema(config.schema_version)
    validate_feature_engine(config, engine_metadata)
    if not set(schema.names).issubset(frame.columns):
        raise ValueError("Feature table does not match its schema")
    write_parquet(output, frame)
    write_json(
        Path(output).with_suffix(".metadata.json"),
        {
            "schema_version": schema.version,
            "feature_hash": schema.fingerprint,
            "feature_config": asdict(config),
            "engine": engine_metadata,
        },
    )


def load_feature_metadata(path: str | Path) -> dict:
    """Fail rather than silently train with unknown feature calculation settings.

    Args:
        path: Feature Parquet path with a matching metadata sidecar.

    Returns:
        Validated feature schema/configuration and engine provenance.
    """
    metadata = json.loads(
        Path(path).with_suffix(".metadata.json").read_text(encoding="utf-8")
    )
    schema = feature_schema(metadata["schema_version"])
    config = FeatureConfig(**metadata["feature_config"])
    if config.schema_version != schema.version or metadata["feature_hash"] != schema.fingerprint:
        raise ValueError("Feature metadata schema mismatch")
    validate_feature_engine(config, metadata.get("engine"))
    return metadata


def export_sample(frame: pd.DataFrame, models: dict, limit: int = 64) -> pd.DataFrame:
    """Choose non-training verification rows covering every exported category.

    Args:
        frame: Feature rows with unchanged partition/category metadata.
        models: Native booster mapping to verify.
        limit: Maximum sample rows per exported model.

    Returns:
        Held-out verification rows, without training observations.
    """
    heldout = frame.loc[frame["partition"] != "train"]
    if "pooled" in models:
        return heldout.head(limit)
    sample = pd.concat(
        [
            heldout.loc[heldout["category"] == category].head(limit)
            for category in models
        ]
    )
    if not len(sample):
        raise ValueError("No held-out export sample")
    return sample


def feature_missingness(frame: pd.DataFrame, schema_version: int = 2) -> dict:
    """Report absent features explicitly so sparse phases are visible.

    Args:
        frame: Side feature rows in the saved schema.
        schema_version: Feature contract used to construct the table.

    Returns:
        Feature-to-missing-fraction mapping; empty tables use None.
    """
    names = feature_schema(schema_version).names
    matrix = frame.loc[:, list(names)].to_numpy(dtype=float)
    if not len(matrix):
        return dict.fromkeys(names, None)
    return dict(zip(names, np.isnan(matrix).mean(axis=0).tolist()))
