"""Training eligibility, rated metadata, and recent-game collection tests."""

import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from chessdotcom.endpoints.player_games_by_month import ResponseBuilder

from tests.test_features import PGN
from training_data.chess_com_connection import ChessComConnection
from training_data.config import DatasetConfig
from training_data.model_dataset import ModelDatasetBuilder, training_record


def source_game(**changes):
    """Build a small raw API game with explicit rated and result fields.

    Args:
        changes: Raw game fields to override in the fixture.

    Returns:
        API-style game dictionary with rated metadata and clocked PGN.
    """
    source = {
        "rules": "chess",
        "rated": True,
        "time_class": "blitz",
        "url": "https://chess.com/game/1",
        "pgn": PGN.replace(
            '[White "Alice"]', '[White "Alice"]\n[WhiteElo "1200"]'
        ).replace('[Black "Bob"]', '[Black "Bob"]\n[BlackElo "1300"]'),
        "white": SimpleNamespace(result="win"),
        "black": SimpleNamespace(result="resigned"),
    }
    source.update(changes)
    source["white"] = vars(source["white"])
    source["black"] = vars(source["black"])
    return source


class ModelDatasetTests(unittest.TestCase):
    def test_filters_and_deduplication(self):
        """Count rejection reasons while accepting each eligible game once.

        Args:
            None.

        Returns:
            None; accepted rows and rejection counters are asserted.
        """
        connection = Mock()
        connection.config = DatasetConfig()
        connection.iter_raw_player_games.return_value = [
            ("Alice", [source_game(rated=False), source_game(), source_game()])
        ]
        builder = ModelDatasetBuilder(connection)
        frame = builder.build(["Alice"])
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["category"], "blitz")
        self.assertEqual(builder.rejections, {"unrated": 1, "duplicate": 1})

    def test_rejects_abandonment_and_missing_clocks(self):
        """Exclude explicitly abandoned and partially clocked training games.

        Args:
            None.

        Returns:
            None; filter-specific ValueErrors are asserted.
        """
        abandoned = source_game(black=SimpleNamespace(result="abandoned"))
        with self.assertRaisesRegex(ValueError, "abandoned"):
            training_record(abandoned)
        missing = source_game()
        missing["pgn"] = missing["pgn"].replace("{[%clk 0:05:01]}", "")
        with self.assertRaisesRegex(ValueError, "missing_clocks"):
            training_record(missing)

    def test_raw_collection_preserves_rated_field(self):
        """Exercise the SDK boundary that drops rated from typed games.

        Args:
            None.

        Returns:
            None; raw-metadata collection must retain the eligible game.
        """
        source = source_game()
        source.update(start_time=1, end_time=2)
        response = ResponseBuilder().build(json.dumps({"games": [source]}))
        self.assertFalse(hasattr(response.games[0], "rated"))
        connection = ChessComConnection(DatasetConfig(max_workers=1))
        connection.client = Mock()
        connection.client.get_player_games_by_month.return_value = response
        frame = ModelDatasetBuilder(connection).build(["Alice"])
        self.assertEqual(len(frame), 1)
        self.assertEqual(frame.iloc[0]["white_rating"], 1200)

    def test_contribution_cap_chooses_recent_games(self):
        """Apply the source-player cap after ordering eligible games newest first.

        Args:
            None.

        Returns:
            None; only the newest game's durable ID is retained.
        """
        connection = Mock()
        connection.config = DatasetConfig(max_games_per_player=1)
        old = source_game(end_time=1, url="https://chess.com/game/old")
        recent = source_game(end_time=2, url="https://chess.com/game/new")
        connection.iter_raw_player_games.return_value = [("Alice", [old, recent])]
        frame = ModelDatasetBuilder(connection).build(["Alice"])
        self.assertEqual(frame.iloc[0]["game_id"], training_record(recent)["game_id"])

    def test_target_count_resume_and_checkpoints(self):
        """Stop at the exact unique-game target and avoid requests when cached.

        Args:
            None.

        Returns:
            None; cache extension, global cap and checkpoint contents are asserted.
        """
        connection = Mock()
        connection.config = DatasetConfig()
        first = source_game(url="https://chess.com/game/first")
        second = source_game(url="https://chess.com/game/second")
        third = source_game(url="https://chess.com/game/third")
        initial = pd.DataFrame([training_record(first)])
        connection.iter_raw_player_games.return_value = [
            ("Alice", [first, second, third])
        ]
        checkpoint = Mock()
        builder = ModelDatasetBuilder(connection)
        frame = builder.build(
            ["Alice"], target_games=2, initial_games=initial, checkpoint=checkpoint
        )
        self.assertEqual(len(frame), 2)
        self.assertEqual(frame["game_id"].nunique(), 2)
        self.assertEqual(len(checkpoint.call_args.args[0]), 2)
        connection.iter_raw_player_games.reset_mock()
        reused = builder.build(["Alice"], target_games=2, initial_games=frame)
        self.assertEqual(len(reused), 2)
        connection.iter_raw_player_games.assert_not_called()


if __name__ == "__main__":
    unittest.main()
