"""Cache and engine protocol tests use a mock UCI engine."""

import asyncio
import sys
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import asdict, replace
from unittest.mock import Mock, patch

import chess.engine

from elo_model.analysis import AnalysisCache, EngineConfig, StockfishAnalyzer
from elo_model.game import GameInput
from tests.test_features import PGN, sample_analysis


class AnalysisTests(unittest.TestCase):
    def test_every_contract_setting_invalidates_cache(self):
        """Keep limits, resource settings and scoring policies in cache identity.

        Args:
            None.

        Returns:
            None; changing any behavior-affecting field produces a different key.
        """
        game = GameInput.from_pgn(PGN)
        config = EngineConfig("mock", depth=None, multipv=1, time_seconds=.1,
                              hash_reset="game", played_move_scoring="next-position")
        settings = asdict(config)
        metadata = {"settings": settings, "binary_sha256": "a" * 64, "analysis_version": 2}
        key = AnalysisCache.key(game, metadata)
        for field, value in {"depth": 4, "multipv": 3, "threads": 2, "hash_mb": 128,
                             "time_seconds": .2, "nodes": 1000, "hash_reset": "position",
                             "played_move_scoring": "root-restricted"}.items():
            with self.subTest(field=field):
                self.assertNotEqual(key, AnalysisCache.key(game, {
                    **metadata, "settings": {**settings, field: value},
                }))

    def test_repetition_history_and_terminal_outcomes_need_no_search(self):
        """Recognize automatic fivefold draws using replay history, not FEN alone.

        Args:
            None.

        Returns:
            None; the terminal repeated board and mate need no engine search.
        """
        pgn = '1. Nf3 Nf6 2. Ng1 Ng8 3. Nf3 Nf6 4. Ng1 Ng8 5. Nf3 Nf6 6. Ng1 Ng8 7. Nf3 Nf6 8. Ng1 Ng8 *'
        game = GameInput.from_pgn(pgn)
        board = game.boards()[-1]
        self.assertEqual(len(board.move_stack), 16)
        self.assertIsNotNone(board.outcome())
        with tempfile.TemporaryDirectory() as directory:
            analyzer = StockfishAnalyzer(EngineConfig("mock"), AnalysisCache(directory))
            analyzer.engine = Mock()
            result = analyzer._position(board)
            self.assertEqual(result.evaluation.cp, 0)
            self.assertEqual(result.candidates, ())
            mate = GameInput.from_pgn('1. f3 e5 2. g4 Qh4# 0-1').boards()[-1]
            self.assertEqual(analyzer._position(mate).evaluation.mate, -1)
            analyzer.engine.analyse.assert_not_called()
            analyzer.engine.analysis.assert_not_called()

    def test_timed_search_ignores_interrupted_bound_iteration(self):
        """Keep exact candidates at one complete depth, not the final bound.

        Args:
            None.

        Returns:
            None; exact complete iteration selection and missing-data errors are checked.
        """
        board = chess.Board()
        moves = list(board.legal_moves)
        exact = [{"depth": 4, "multipv": rank, "pv": [move],
                  "score": chess.engine.PovScore(chess.engine.Cp(rank), chess.WHITE)}
                 for rank, move in enumerate(moves[:1], 1)]
        bound = {**exact[0], "depth": 5, "lowerbound": True}
        with tempfile.TemporaryDirectory() as directory:
            analyzer = StockfishAnalyzer(EngineConfig(
                "mock", depth=None, time_seconds=.1, played_move_scoring="next-position",
            ), AnalysisCache(directory))
            analyzer.engine = Mock()
            analyzer.engine.analysis.return_value = nullcontext(iter([*exact, bound]))
            self.assertEqual(analyzer._bounded_infos(board, chess.engine.Limit(time=.1)), exact)
            analyzer.engine.analysis.return_value = nullcontext(iter([bound]))
            limit = chess.engine.Limit(time=.1)
            with self.assertRaisesRegex(ValueError, "no complete exact"):
                analyzer._bounded_infos(board, limit)

    def test_fast_search_has_one_budget_per_reached_position(self):
        """Check time caps, history and absence of fallback searches.

        Args:
            None.

        Returns:
            None; each reached board gets exactly one capped search.
        """
        game = GameInput.from_pgn(PGN)
        with tempfile.TemporaryDirectory() as directory:
            config = EngineConfig("mock")
            analyzer = StockfishAnalyzer(config, AnalysisCache(directory))
            engine = Mock()
            engine.options = {"Clear Hash": None}
            analyzer.engine = engine
            analyzer.metadata = {"settings": config.__dict__}

            def analyse(board, limit, multipv):
                """Return one exact score while checking history and the allowance.

                Args:
                    board: History-preserving reached board.
                    limit: Total position search limit.
                    multipv: Requested principal variation count.

                Returns:
                    A context-managed stream of exact single-PV results.
                """
                self.assertEqual(limit.time, 0.1)
                self.assertIsNone(limit.depth)
                self.assertEqual(multipv, 1)
                self.assertEqual(len(board.move_stack), engine.analysis.call_count - 1)
                return nullcontext(iter([{
                    "pv": [next(iter(board.legal_moves))], "depth": 4,
                    "score": chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE),
                }]))

            engine.analysis.side_effect = analyse
            key, positions = analyzer.analyze(game)
            calls = engine.analysis.call_count
            self.assertEqual(calls, len(game.moves) + 1)
            self.assertEqual(analyzer.analyze(game), (key, positions))
            self.assertEqual(engine.analysis.call_count, calls)
            self.assertEqual(analyzer.cache_hits, 1)
            engine.analyse.assert_not_called()
            engine.configure.assert_called_once_with({"Clear Hash": None})

    @unittest.skipUnless(sys.platform == "win32", "Windows event-loop compatibility")
    def test_engine_startup_with_selector_policy(self):
        """Launch a UCI subprocess under the policy used by Windows notebooks.

        Args:
            None.

        Returns:
            None; startup, readiness and shutdown must succeed.
        """
        previous_policy = asyncio.get_event_loop_policy()
        self.addCleanup(asyncio.set_event_loop_policy, previous_policy)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        launch = chess.engine.SimpleEngine.popen_uci
        uci_script = (
            "import sys\n"
            "for line in sys.stdin:\n"
            "    command = line.strip()\n"
            "    if command == 'uci':\n"
            "        print('id name Test UCI')\n"
            "        print('option name Threads type spin default 1 min 1 max 128')\n"
            "        print('option name Hash type spin default 16 min 1 max 4096')\n"
            "        print('uciok', flush=True)\n"
            "    elif command == 'isready':\n"
            "        print('readyok', flush=True)\n"
            "    elif command == 'quit':\n"
            "        break\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            analyzer = StockfishAnalyzer(
                EngineConfig(sys.executable), AnalysisCache(directory)
            )
            with patch(
                "elo_model.analysis.chess.engine.SimpleEngine.popen_uci",
                side_effect=lambda command: launch(
                    [sys.executable, "-u", "-c", uci_script]
                ),
            ):
                with analyzer:
                    analyzer.engine.ping()
                    self.assertEqual(analyzer.metadata["id"]["name"], "Test UCI")
                self.assertIsNone(analyzer.engine)

    def test_roundtrip_and_settings_invalidation(self):
        """Verify Parquet replay and history/configuration cache invalidation.

        Args:
            None.

        Returns:
            None; typed round-trip equality and distinct keys are asserted.
        """
        game = GameInput.from_pgn(PGN)
        with tempfile.TemporaryDirectory() as directory:
            cache = AnalysisCache(directory)
            key = cache.key(game, {"time_seconds": 0.1})
            self.assertIsNone(cache.load(key))
            cache.save(key, sample_analysis(game))
            self.assertEqual(cache.load(key), sample_analysis(game))
            self.assertNotEqual(key, cache.key(game, {"time_seconds": 0.2}))
            self.assertNotEqual(
                key, cache.key(replace(game, moves=game.moves[:-1]), {"time_seconds": 0.1})
            )


if __name__ == "__main__":
    unittest.main()
