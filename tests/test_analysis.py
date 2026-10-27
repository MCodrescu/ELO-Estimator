"""Cache and engine protocol tests use a mock UCI engine."""

import asyncio
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock, patch

import chess.engine

from elo_model.analysis import AnalysisCache, EngineConfig, StockfishAnalyzer
from elo_model.game import GameInput
from tests.test_features import PGN, sample_analysis


class AnalysisTests(unittest.TestCase):
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
            key = cache.key(game, {"depth": 12})
            self.assertIsNone(cache.load(key))
            cache.save(key, sample_analysis(game))
            self.assertEqual(cache.load(key), sample_analysis(game))
            self.assertNotEqual(key, cache.key(game, {"depth": 13}))
            self.assertNotEqual(
                key, cache.key(replace(game, moves=game.moves[:-1]), {"depth": 12})
            )

    def test_outside_top_three_is_analyzed_and_cached(self):
        """Check root-restricted analysis and complete-game cache reuse.

        Args:
            None.

        Returns:
            None; mock call counts prove required work and subsequent reuse.
        """
        game = GameInput.from_pgn(PGN)
        with tempfile.TemporaryDirectory() as directory:
            analyzer = StockfishAnalyzer(EngineConfig("mock"), AnalysisCache(directory))
            engine = Mock()
            engine.options = {}
            analyzer.engine = engine
            analyzer.metadata = {"engine": "mock", "depth": 12}

            def analyse(board, limit, multipv=None, root_moves=None):
                """Return minimal UCI info for ordinary or restricted searches.

                Args:
                    board: Replayed board supplied by the adapter.
                    limit: Fixed-depth limit from the adapter.
                    multipv: Requested ranked candidate count, if applicable.
                    root_moves: Optional played-move restriction.

                Returns:
                    Ranked info list or one root-restricted exact score.
                """
                score = chess.engine.PovScore(chess.engine.Cp(0), chess.WHITE)
                if root_moves:
                    return {"score": score}
                legal = list(board.legal_moves)
                return [
                    {"pv": [move], "score": score, "depth": 12} for move in legal[:3]
                ]

            engine.analyse.side_effect = analyse
            key, result = analyzer.analyze(game)
            calls = engine.analyse.call_count
            self.assertGreater(calls, len(game.moves) + 1)
            self.assertIsNotNone(result[0].played_score)
            self.assertEqual(analyzer.analyze(game), (key, result))
            self.assertEqual(engine.analyse.call_count, calls)
            self.assertEqual(analyzer.cache_hits, 1)
            self.assertEqual(analyzer.analyzed_games, 1)
            self.assertTrue((Path(directory) / f"{key}.parquet").exists())


if __name__ == "__main__":
    unittest.main()
