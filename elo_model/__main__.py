"""Run stages with python -m elo_model; --help lists each stage's inputs."""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from training_data.chess_com_connection import ChessComConnection
from training_data.model_dataset import ModelDatasetBuilder

from .analysis import analysis_from_payload
from .config import (
    CLI_CONFIG,
    DEFAULT_ANALYSIS_WORKERS,
    DatasetConfig,
    EngineConfig,
    FeatureConfig,
    StratificationConfig,
    TrainingConfig,
)
from .evaluation import BAND_NAMES, rating_bands
from .feature_schema import FEATURE_NAMES
from .game import GameInput
from .pipeline import (
    analyze_dataset,
    export_sample,
    extract_dataset,
    feature_missingness,
    load_feature_metadata,
    manifest_rows,
    save_feature_dataset,
    split_summary,
)
from .predictor import RatingPredictor, export_bundle
from .selection import StratifiedGameSelector
from .split import PlayerSplitter
from .storage import write_json, write_parquet
from .training import ModelTrainer

DEFAULT_FEATURES = str(CLI_CONFIG.features_path)
logger = logging.getLogger(__name__)


def read_json(path):
    """Read a local UTF-8 JSON artifact.

    Args:
        path: Filename containing one JSON value.

    Returns:
        Deserialized JSON value.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def run_collect(args):
    """Download eligible raw-metadata archives through the existing client.

    Args:
        args: Parsed collection filenames, months and connection settings.

    Returns:
        None; writes the game table and rejection report.
    """
    months = tuple(tuple(map(int, month.split("-"))) for month in args.months)
    settings = DatasetConfig(
        months=months,
        max_games_per_player=args.max_games,
        user_agent=args.user_agent,
        proxy_url=args.proxy,
    )
    usernames = read_json(args.usernames)
    if not isinstance(usernames, list) or not all(
        isinstance(name, str) for name in usernames
    ):
        raise ValueError("Username file must contain a JSON list of strings")
    usernames = list(dict.fromkeys(name.strip().casefold() for name in usernames))
    builder = ModelDatasetBuilder(ChessComConnection(settings))
    frame = builder.build(usernames)
    if frame.empty:
        raise ValueError(
            f"No eligible games; rejection counts: {dict(builder.rejections)}"
        )
    write_parquet(args.output, frame)
    write_json(
        Path(args.output).with_suffix(".collection.json"),
        {
            "games": len(frame),
            "rejections": dict(builder.rejections),
            "months": months,
            "source_player_cap": args.max_games,
        },
    )
    print(f"Collected {len(frame)} games to {args.output}")


def run_split(args):
    """Create and preflight a deterministic player/game split manifest.

    Args:
        args: Input/output paths, fractions, seed and coverage limits.

    Returns:
        None; writes the split manifest and coverage report.
    """
    games = pd.read_parquet(args.input)
    splitter = PlayerSplitter(
        tuple(args.fractions), args.seed, args.strict_participants
    )
    selection_config = (
        StratificationConfig.model_validate(read_json(args.selection_config))
        if args.selection_config else CLI_CONFIG.stratification
    )
    selector = StratifiedGameSelector(selection_config, seed=args.seed)
    manifest = selector.select(splitter.assign(games))
    selection = {
        "config": selection_config.model_dump(mode="json"),
        "seed": args.seed,
        "candidate_games": len(games),
        "coverage": selector.coverage.to_dict("records"),
    }
    write_json(Path(args.output).with_suffix(".selection.json"), selection)
    if not selector.coverage.empty and selector.coverage["shortfall"].any():
        logger.warning("Training quotas are underfilled; inspect the selection report")
    rows = manifest_rows(manifest)
    counts = split_summary(manifest)
    if any(
        value["rows"] < args.min_rows or value["target_players"] < 2
        for value in counts.values()
    ):
        raise ValueError(f"Insufficient split coverage before analysis: {counts}")
    bands = rating_bands(rows["rating"])
    by_band = {name: int((bands == name).sum()) for name in BAND_NAMES}
    report = {
        "partitions": counts,
        "input_games": len(games),
        "retained_games": len(manifest),
        "retained_rows": len(rows),
        "possible_rows": 2 * len(games),
        "rating_bands": by_band,
        "categories": rows["category"].value_counts().to_dict(),
        "strict_participants": args.strict_participants,
        "seed": args.seed,
        "fractions": args.fractions,
        "selection": selection,
    }
    write_parquet(args.output, manifest)
    write_json(Path(args.output).with_suffix(".split.json"), report)
    print(json.dumps(report, indent=2))


def run_analyze(args):
    """Run the explicitly requested expensive engine stage.

    Args:
        args: Split manifest, executable, fixed-depth settings and cache paths.

    Returns:
        None; writes resumable analyzed-game and engine metadata artifacts.
    """
    config = EngineConfig(
        args.stockfish, args.depth, threads=args.threads, hash_mb=args.hash_mb
    )
    games = pd.read_parquet(args.input)
    analyze_dataset(games, config, args.cache, args.output, workers=args.workers)
    print(f"Analyzed/cached {len(games)} games")


def run_features(args):
    """Recompute features without invoking an engine.

    Args:
        args: Analyzed manifest, cache, optional feature config and output paths.

    Returns:
        None; writes features, provenance and missingness diagnostics.
    """
    games = pd.read_parquet(args.input)
    metadata = read_json(Path(args.input).with_suffix(".engine.json"))
    config = FeatureConfig(**read_json(args.config)) if args.config else FeatureConfig()
    frame = extract_dataset(games, args.cache, config)
    save_feature_dataset(frame, args.output, metadata, config)
    write_json(
        Path(args.output).with_suffix(".missingness.json"), feature_missingness(frame)
    )
    print(f"Extracted {len(frame)} rows with {len(FEATURE_NAMES)} features")


def run_train(args):
    """Train selected native models and print baseline comparisons.

    Args:
        args: Feature table, output directory and bounded experiment settings.

    Returns:
        None; writes a verified model bundle and evaluation report.
    """
    frame = pd.read_parquet(args.input)
    metadata = load_feature_metadata(args.input)
    config = TrainingConfig(
        mode=args.mode,
        trials=args.trials,
        seed=args.seed,
        max_trees=args.max_trees,
        min_rows=args.min_rows,
        balance_bands=args.balance_bands,
        balance_players=args.balance_players,
        threads=args.threads,
    )
    result = ModelTrainer(config).fit(frame)
    result.metadata.update(metadata)
    sample = export_sample(frame, result.models)
    verification = export_bundle(
        result.models,
        args.output,
        sample,
        result.metadata,
        result.report,
        FeatureConfig(**metadata["feature_config"]),
        model_version=args.model_version,
    )
    for name, report in result.report.items():
        for partition in ("validation", "test"):
            for estimator in ("lightgbm", "ridge"):
                metrics = report["partitions"][partition][estimator]["overall"]
                print(
                    f"{name} {partition} {estimator}: MAE={metrics['mae']:.1f}, RMSE={metrics['rmse']:.1f}"
                )
    print(json.dumps(verification, indent=2))


def run_export(args):
    """Create a deployment bundle and repeat held-out prediction parity checks.

    Args:
        args: Source/destination bundles and provenance-matching sample table.

    Returns:
        None; writes artifacts and prints size/timing/parity measurements.
    """
    source = RatingPredictor(args.source)
    frame = pd.read_parquet(args.sample_features)
    report = read_json(Path(args.source) / "evaluation.json")["models"]
    metadata = load_feature_metadata(args.sample_features)
    if metadata["feature_config"] != source.manifest["feature_config"] or metadata.get(
        "engine"
    ) != source.manifest.get("engine"):
        raise ValueError("Export sample provenance differs from the source model")
    verification = export_bundle(
        source.models,
        args.output,
        export_sample(frame, source.models),
        source.manifest,
        report,
        source.extractor.config,
        model_version=args.model_version,
    )
    print(json.dumps(verification, indent=2))


def run_predict(args):
    """Predict from ordered features or supplied PGN/analysis without an engine.

    Args:
        args: Model/input paths and optional JSON output filename.

    Returns:
        None; writes or prints numeric predictions.
    """
    predictor = RatingPredictor(args.model)
    if args.features:
        frame = pd.read_parquet(args.features)
        predictions = predictor.predict(
            frame.loc[:, list(FEATURE_NAMES)], frame["category"].tolist()
        )
        result = predictions.tolist()
    else:
        if not args.analysis:
            raise ValueError(
                "PGN prediction requires --analysis with precomputed engine output"
            )
        payload = read_json(args.analysis)
        game = GameInput.from_pgn(Path(args.pgn).read_text(encoding="utf-8"))
        result = predictor.predict_game(
            game, analysis_from_payload(payload["positions"]), payload.get("engine"),
            engine_compatibility=args.engine_compatibility,
        )
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, indent=2))


def parser() -> argparse.ArgumentParser:
    """Build the CLI without performing network, engine, or training work.

    Args:
        None.

    Returns:
        Argument parser with stage-specific handlers.
    """
    root = argparse.ArgumentParser(
        description="Single-game Chess.com rating model pipeline"
    )
    commands = root.add_subparsers(dest="command", required=True)

    collect = commands.add_parser(
        "collect", help="Download and filter recent Chess.com games"
    )
    collect.add_argument("--usernames", default=str(CLI_CONFIG.usernames_path))
    collect.add_argument("--months", nargs="+", required=True, metavar="YYYY-MM")
    collect.add_argument("--max-games", type=int, default=CLI_CONFIG.collection.max_games_per_player)
    collect.add_argument("--user-agent", required=True)
    collect.add_argument("--proxy", default=CLI_CONFIG.collection.proxy_url)
    collect.add_argument("--output", default=str(CLI_CONFIG.raw_games_path))
    collect.set_defaults(handler=run_collect)

    split = commands.add_parser(
        "split", help="Preflight target-player and game isolation"
    )
    split.add_argument("--input", default=str(CLI_CONFIG.raw_games_path))
    split.add_argument("--output", default=str(CLI_CONFIG.split_path))
    split.add_argument("--fractions", type=float, nargs=3, default=CLI_CONFIG.split_fractions)
    split.add_argument("--seed", type=int, default=CLI_CONFIG.training.seed)
    split.add_argument("--min-rows", type=int, default=CLI_CONFIG.training.min_rows)
    split.add_argument("--strict-participants", action="store_true")
    split.add_argument(
        "--selection-config", help="JSON training quotas; use an empty object to disable selection",
    )
    split.set_defaults(handler=run_split)

    analyze = commands.add_parser(
        "analyze", help="Run or resume fixed-depth Stockfish analysis"
    )
    analyze.add_argument("--input", default=str(CLI_CONFIG.split_path))
    analyze.add_argument("--output", default=str(CLI_CONFIG.analyzed_path))
    analyze.add_argument("--stockfish", required=True)
    analyze.add_argument("--depth", type=int, default=CLI_CONFIG.engine.depth)
    analyze.add_argument("--threads", type=int, default=CLI_CONFIG.engine.threads)
    analyze.add_argument("--hash-mb", type=int, default=CLI_CONFIG.engine.hash_mb)
    analyze.add_argument(
        "--workers", type=int, default=DEFAULT_ANALYSIS_WORKERS,
        help="Concurrent Stockfish processes; --threads and --hash-mb apply per process",
    )
    analyze.add_argument("--cache", default=str(CLI_CONFIG.engine_cache_directory))
    analyze.set_defaults(handler=run_analyze)

    features = commands.add_parser(
        "features", help="Extract features from cache without running an engine"
    )
    features.add_argument("--input", default=str(CLI_CONFIG.analyzed_path))
    features.add_argument("--output", default=DEFAULT_FEATURES)
    features.add_argument("--cache", default=str(CLI_CONFIG.engine_cache_directory))
    features.add_argument(
        "--config", help="Optional JSON FeatureConfig overrides (full or partial)"
    )
    features.set_defaults(handler=run_features)

    train = commands.add_parser(
        "train", help="Tune, evaluate, and save native Python models"
    )
    train.add_argument("--input", default=DEFAULT_FEATURES)
    train.add_argument("--output", default=str(CLI_CONFIG.training_directory))
    train.add_argument("--mode", choices=("pooled", "separate"), default=CLI_CONFIG.training.mode)
    train.add_argument("--trials", type=int, default=CLI_CONFIG.training.trials)
    train.add_argument("--max-trees", type=int, default=CLI_CONFIG.training.max_trees)
    train.add_argument("--min-rows", type=int, default=CLI_CONFIG.training.min_rows)
    train.add_argument("--threads", type=int, default=CLI_CONFIG.training.threads)
    train.add_argument("--seed", type=int, default=CLI_CONFIG.training.seed)
    train.add_argument("--balance-bands", action="store_true")
    train.add_argument("--balance-players", action="store_true")
    train.add_argument("--model-version", help="Release label for the exported weights")
    train.set_defaults(handler=run_train)

    export = commands.add_parser(
        "export", help="Export/reverify a trained native model bundle"
    )
    export.add_argument("--source", default=str(CLI_CONFIG.training_directory))
    export.add_argument("--output", default=str(CLI_CONFIG.artifact_directory))
    export.add_argument("--sample-features", default=DEFAULT_FEATURES)
    export.add_argument("--model-version", help="Release label; retains source version when omitted")
    export.set_defaults(handler=run_export)

    predict = commands.add_parser(
        "predict", help="Predict from ordered features or PGN plus analysis"
    )
    predict.add_argument("--model", default=str(CLI_CONFIG.artifact_directory))
    inputs = predict.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--features")
    inputs.add_argument("--pgn")
    predict.add_argument("--analysis")
    predict.add_argument(
        "--engine-compatibility", choices=("exact", "same-release"), default="exact",
        help="Allow a different binary build of the same engine release with matching settings",
    )
    predict.add_argument("--output")
    predict.set_defaults(handler=run_predict)
    return root


def main():
    """Dispatch one explicitly requested CLI stage.

    Args:
        None; arguments are parsed from sys.argv.

    Returns:
        None; stage failures propagate with their diagnostic context.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )
    arguments = parser().parse_args()
    arguments.handler(arguments)


if __name__ == "__main__":
    main()
