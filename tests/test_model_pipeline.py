"""End-to-end offline CLI smoke test using synthetic engine evaluations."""

import contextlib
import io
import json
import runpy
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from threading import Barrier, get_ident
from unittest.mock import MagicMock, Mock, patch

import pandas as pd

from elo_model.__main__ import main
from elo_model.analysis import AnalysisCache
from elo_model.config import (
    EngineConfig,
    PipelineConfig,
    RatingQuota,
    StratificationConfig,
    TrainingConfig,
)
from elo_model.game import GameInput
from elo_model.pipeline import analyze_dataset, extract_dataset, save_feature_dataset
from elo_model.split import PlayerSplitter
from elo_model.storage import replace_checkpoint, write_json, write_parquet
from tests.test_features import PGN, sample_analysis
from tests.test_model_dataset import source_game
from tests.test_model_training import synthetic_frame
from training_data.model_dataset import training_record


class PipelineTests(unittest.TestCase):
    def test_cli_split_applies_training_quotas_and_saves_provenance(self):
        """Apply CLI quotas before analysis without resampling hold-out games.

        Args:
            None.

        Returns:
            None; selected training counts, hold-outs and selection sidecars are asserted.
        """
        frame = pd.DataFrame([
            training_record(source)
            for _, archive in self.notebook_archives(90)
            for source in archive
        ])
        config = StratificationConfig(max_rows_per_player=1, quotas=(
            RatingQuota(lower=0, category="blitz", rows=12),
        ))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "games.parquet", root / "split.parquet"
            config_path = root / "quotas.json"
            write_parquet(source, frame)
            write_json(config_path, config.model_dump(mode="json"))
            self.run_cli(
                "split", "--input", str(source), "--output", str(output),
                "--selection-config", str(config_path), "--min-rows", "2",
            )
            selected = pd.read_parquet(output)
            train = selected.loc[selected["partition"] == "train"]
            self.assertEqual(int(train[["keep_white", "keep_black"]].sum().sum()), 12)
            expected = PlayerSplitter().assign(frame)
            pd.testing.assert_frame_equal(
                selected.loc[selected["partition"] != "train"].reset_index(drop=True),
                expected.loc[expected["partition"] != "train"].reset_index(drop=True),
            )
            report = json.loads(output.with_suffix(".selection.json").read_text())
            self.assertEqual(report["config"], config.model_dump(mode="json"))
            self.assertEqual(report["coverage"][0]["shortfall"], 0)

    def test_analysis_rejects_invalid_workers_before_startup(self):
        """Reject invalid pool sizes before accessing an engine or output file.

        Args:
            None.

        Returns:
            None; invalid counts raise ValueError without creating engines.
        """
        games = pd.DataFrame()
        config = EngineConfig("mock")
        with patch("elo_model.pipeline.StockfishAnalyzer") as factory:
            for workers in (0, -1, True, 1.5):
                with self.subTest(workers=workers), self.assertRaises(ValueError):
                    analyze_dataset(
                        games, config, "cache", "output",
                        workers=workers,
                    )
            factory.assert_not_called()

    def test_analysis_failure_preserves_checkpoint_and_closes_engine(self):
        """Retain completed references and propagate an engine failure.

        Args:
            None.

        Returns:
            None; the checkpoint survives and the engine context is closed.
        """
        analyzer = MagicMock(metadata={"engine": "failure-test"})
        analyzer.__enter__.return_value = analyzer
        analyzer.analyze.side_effect = [("cached-key", []), RuntimeError("engine failed")]
        analyzer.cache_hits = 1
        analyzer.analyzed_games = 0
        games = pd.DataFrame({"game_id": [0, 1], "pgn": [PGN, PGN]})
        config = EngineConfig("mock")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analyzed.parquet"
            with (
                patch("elo_model.pipeline.StockfishAnalyzer", return_value=analyzer),
                self.assertRaisesRegex(RuntimeError, "engine failed"),
            ):
                analyze_dataset(games, config, directory, output, workers=1)
            self.assertEqual(pd.read_parquet(output)["game_id"].tolist(), [0])
            analyzer.__exit__.assert_called_once()

    def test_cli_analysis_worker_count(self):
        """Forward default and overridden CLI worker counts to analysis.

        Args:
            None.

        Returns:
            None; eight workers are the default and explicit overrides are honored.
        """
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split.parquet"
            write_parquet(path, pd.DataFrame({"pgn": [PGN]}))
            for options, workers in (((), 8), (("--workers", "2"), 2)):
                with self.subTest(workers=workers), patch(
                    "elo_model.__main__.analyze_dataset"
                ) as analyze:
                    self.run_cli(
                        "analyze", "--stockfish", "mock", "--input", str(path), *options
                    )
                    self.assertEqual(analyze.call_args.kwargs["workers"], workers)

    def test_analysis_uses_eight_reusable_engines(self):
        """Run two concurrent waves while checkpointing only on the caller thread.

        Args:
            None.

        Returns:
            None; concurrency, reuse, output ordering and shutdown are asserted.
        """
        barrier = Barrier(8, timeout=10)
        metadata = {"engine": "parallel-test"}
        analyzers = []
        main_thread = get_ident()
        frames = []

        def analyze(game, progress=False):
            """Synchronize eight engine calls to prove concurrent execution.

            Args:
                game: Parsed input game.
                progress: Whether to display position progress.

            Returns:
                A synthetic cache key and unused position list.
            """
            self.assertFalse(progress)
            barrier.wait()
            return "synthetic-key", []

        def checkpoint(path, frame):
            """Record and persist checkpoints from the coordinating thread.

            Args:
                path: Temporary output filename.
                frame: Completed game references in input order.

            Returns:
                None; thread ownership is asserted.
            """
            self.assertEqual(get_ident(), main_thread)
            frames.append(frame.copy())
            write_parquet(path, frame)

        for _ in range(8):
            analyzer = Mock(metadata=metadata, cache_hits=0, analyzed_games=0)
            analyzer.__enter__ = Mock(return_value=analyzer)
            analyzer.__exit__ = Mock(return_value=False)
            analyzer.analyze.side_effect = analyze
            analyzers.append(analyzer)

        games = pd.DataFrame({"game_id": range(16), "pgn": [PGN] * 16})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "analyzed.parquet"
            with (
                patch("elo_model.pipeline.StockfishAnalyzer", side_effect=analyzers) as factory,
                patch("elo_model.pipeline.write_parquet", side_effect=checkpoint),
            ):
                result = analyze_dataset(games, EngineConfig("mock"), directory, output)
            self.assertEqual(factory.call_count, 8)
            self.assertEqual(result, metadata)
            self.assertEqual(pd.read_parquet(output)["game_id"].tolist(), list(range(16)))
            self.assertEqual(len(frames), 16)
            self.assertEqual([len(frame) for frame in frames], list(range(1, 17)))
            for analyzer in analyzers:
                self.assertEqual(analyzer.analyze.call_count, 2)
                analyzer.__exit__.assert_called_once()

    def test_transient_checkpoint_locks_are_retried(self):
        """Retry only permissions failures and propagate a permanent lock.

        Args:
            None.

        Returns:
            None; bounded retry calls, delay handling and final errors are asserted.
        """
        with (
            patch("elo_model.storage.os.replace", side_effect=[PermissionError("locked"), None]) as replace_file,
            patch("elo_model.storage.time.sleep") as backoff,
            self.assertLogs("elo_model.storage", level="WARNING"),
        ):
            replace_checkpoint(Path("source"), Path("destination"))

        self.assertEqual(replace_file.call_count, 2)
        backoff.assert_called_once()

        with (
            patch("elo_model.storage.os.replace", side_effect=PermissionError("permanent")),
            patch("elo_model.storage.time.sleep"),
            self.assertLogs("elo_model.storage", level="WARNING"),
            self.assertRaisesRegex(PermissionError, "permanent"),
        ):
            replace_checkpoint(Path("source"), Path("destination"))

    def test_notebook_setup_and_collection_use_current_pipeline(self):
        """Exercise all notebook stages with mocked archives and precomputed scores.

        Args:
            None.

        Returns:
            None; target sizing, checkpoint/cache reuse and native export are asserted.
        """
        path = Path(__file__).resolve().parent.parent / "build_training_data.ipynb"
        notebook = json.loads(path.read_text(encoding="utf-8"))
        cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            usernames_path = root / "usernames.json"
            write_json(usernames_path, ["Alice"])

            settings = PipelineConfig(
                target_games=90,
                usernames_path=usernames_path,
                data_directory=root / "data",
                engine_cache_directory=root / "cache",
                artifact_directory=root / "model",
                split_fractions=(0.6, 0.2, 0.2),
                min_players_per_partition=2,
                stratification=StratificationConfig(quotas=(
                    RatingQuota(lower=0, category="blitz", rows=1000),
                )),
                training=TrainingConfig(trials=0, max_trees=5, min_rows=2),
            )
            archives = self.notebook_archives(settings.candidate_games)
            metadata = {"engine": "synthetic-notebook-test"}
            namespace = {"test_settings": settings}

            def analyze_fixture(games, config, cache_directory, output, progress=False):
                """Write synthetic analysis using the real cache and artifact formats.

                Args:
                    games: Split-retained raw game rows.
                    config: Supplied engine settings, unused for synthetic scores.
                    cache_directory: Temporary persistent cache location.
                    output: Analyzed manifest destination.
                    progress: Requested display mode, unused in this fixture.

                Returns:
                    Stable synthetic engine metadata.
                """
                cached_rows = []
                cache = AnalysisCache(cache_directory)

                for record in games.to_dict("records"):
                    game = GameInput.from_pgn(record["pgn"])
                    key = cache.key(game, metadata)

                    if cache.load(key) is None:
                        cache.save(key, sample_analysis(game))

                    cached_rows.append({**record, "analysis_key": key})

                namespace["pd"].DataFrame(cached_rows).to_parquet(output, index=False)
                return metadata

            with (
                contextlib.redirect_stdout(io.StringIO()),
                patch("logging.basicConfig"),
            ):
                setup = cells[0].replace(
                    "settings = PipelineConfig()", "settings = test_settings"
                )
                self.run_notebook_stage(setup, namespace, root)
                self.run_notebook_stage(cells[1], namespace, root)

                with patch.object(
                    namespace["connection"],
                    "iter_raw_player_games",
                    return_value=archives,
                ):
                    self.run_notebook_stage(cells[2], namespace, root)

                with patch.object(
                    namespace["connection"], "iter_raw_player_games"
                ) as download:
                    self.run_notebook_stage(cells[2], namespace, root)
                    download.assert_not_called()

                self.run_notebook_stage(cells[3], namespace, root)
                self.run_notebook_stage(cells[4], namespace, root)
                namespace["analyze_dataset"] = analyze_fixture

                for source in cells[5:10]:
                    self.run_notebook_stage(source, namespace, root)

            self.assertEqual(len(namespace["games_df"]), settings.candidate_games)
            self.assertTrue(settings.features_path.exists())
            self.assertTrue(
                (settings.artifact_directory / "model_manifest.json").exists()
            )
            manifest = json.loads(
                (settings.artifact_directory / "model_manifest.json").read_text()
            )
            self.assertEqual(
                manifest["selection"]["config"], settings.stratification.model_dump(mode="json")
            )
            self.assertEqual(manifest["candidate_games"], settings.candidate_games)
            self.assertEqual(manifest["dataset_games"], len(namespace["split_games"]))
            self.assertEqual(
                json.loads(usernames_path.read_text(encoding="utf-8")), ["Alice"]
            )

    @staticmethod
    def run_notebook_stage(source: str, namespace: dict, directory: Path) -> None:
        """Run trusted repository notebook code as a temporary ordinary script.

        Args:
            source: One notebook stage's Python source.
            namespace: State carried between notebook stages.
            directory: Temporary fixture directory for the stage script.

        Returns:
            None; namespace receives updated stage globals.
        """
        script_path = directory / "notebook_stage.py"
        script_path.write_text(source, encoding="utf-8")
        namespace.update(runpy.run_path(str(script_path), init_globals=namespace))

    @staticmethod
    def notebook_archives(count: int) -> list:
        """Create distinct games and players for a meaningful split preflight.

        Args:
            count: Number of unique eligible games to synthesize.

        Returns:
            Mock owner/archive pairs in the connection's raw response format.
        """
        records = []

        for index in range(count):
            record = source_game(url=f"https://chess.com/game/{index}")
            record["pgn"] = record["pgn"].replace('"Alice"', f'"white{index}"')
            record["pgn"] = record["pgn"].replace('"Bob"', f'"black{index}"')
            records.append((f"owner{index}", [record]))

        return records

    def test_feature_recomputation_and_row_filtering(self):
        """Recompute both perspectives from cache and retain the eligible target.

        Args:
            None.

        Returns:
            None; selected row identity and game-time label are asserted.
        """
        game = GameInput.from_pgn(PGN)
        frame = synthetic_frame().iloc[:1].copy()
        frame["pgn"] = PGN
        frame["white_username"], frame["black_username"] = "Alice", "Bob"
        frame["white_rating"], frame["black_rating"] = 1200, 1300
        frame["keep_white"], frame["keep_black"] = True, False
        with tempfile.TemporaryDirectory() as directory:
            cache = AnalysisCache(directory)
            key = cache.key(game, {"engine": "mock"})
            cache.save(key, sample_analysis(game))
            frame["analysis_key"] = key
            rows = extract_dataset(frame, directory)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.iloc[0]["target_username"], "alice")
            self.assertEqual(rows.iloc[0]["rating"], 1200)

    def test_cli_train_export_predict(self):
        """Exercise the offline CLI with vector and PGN-analysis inference.

        Args:
            None.

        Returns:
            None; persisted predictions and both game sides are asserted.
        """
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            features = root / "features.parquet"
            trained, exported = root / "trained", root / "exported"
            metadata = {"engine": "synthetic-test-only"}
            save_feature_dataset(synthetic_frame(), features, metadata)
            self.run_cli(
                "train",
                "--input",
                str(features),
                "--output",
                str(trained),
                "--trials",
                "0",
                "--max-trees",
                "5",
                "--model-version",
                "test-v1",
            )
            self.assertEqual(
                json.loads((trained / "model_manifest.json").read_text())["model_version"],
                "test-v1",
            )
            self.run_cli(
                "export",
                "--source",
                str(trained),
                "--output",
                str(exported),
                "--sample-features",
                str(features),
                "--model-version",
                "test-v2",
            )
            self.assertEqual(
                json.loads((exported / "model_manifest.json").read_text())["model_version"],
                "test-v2",
            )
            output = root / "predictions.json"
            self.run_cli(
                "predict",
                "--model",
                str(exported),
                "--features",
                str(features),
                "--output",
                str(output),
            )
            self.assertEqual(len(json.loads(output.read_text())), 180)
            game = GameInput.from_pgn(PGN)
            pgn_path, analysis_path = root / "game.pgn", root / "analysis.json"
            pgn_path.write_text(PGN, encoding="utf-8")
            write_json(
                analysis_path,
                {
                    "engine": metadata,
                    "positions": [asdict(item) for item in sample_analysis(game)],
                },
            )
            self.run_cli(
                "predict",
                "--model",
                str(exported),
                "--pgn",
                str(pgn_path),
                "--analysis",
                str(analysis_path),
                "--output",
                str(output),
            )
            self.assertEqual(set(json.loads(output.read_text())), {"white", "black"})

    @staticmethod
    def run_cli(*arguments):
        """Invoke a CLI stage while capturing its human-readable output.

        Args:
            arguments: Command and option strings to place in sys.argv.

        Returns:
            None; handler failures propagate to the calling test.
        """
        with (
            patch("sys.argv", ["elo_model", *arguments]),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            main()


if __name__ == "__main__":
    unittest.main()
