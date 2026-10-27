"""Deterministic feature tests require neither Stockfish nor network access."""

import unittest

import numpy as np

from elo_model import (
    FEATURE_NAMES,
    Candidate,
    GameFeatureExtractor,
    GameInput,
    PositionAnalysis,
    Score,
)
from elo_model.features import blunder_runs

PGN = """[White "Alice"]
[Black "Bob"]
[TimeControl "300+2"]
[Result "1-0"]
[Termination "Alice won by resignation"]

1. e4 {[%clk 0:05:01]} e5 {[%clk 0:05:00]} 2. Nf3 {[%clk 0:05:00]} Nc6 {[%clk 0:04:58]} 1-0
"""


def sample_analysis(game):
    """Provide legal synthetic root scores, including the final position.

    Args:
        game: Parsed fixture game to replay.

    Returns:
        Legal, exact White-perspective synthetic analysis records.
    """
    result = []
    for index, board in enumerate(game.boards()):
        legal = [move.uci() for move in board.legal_moves]
        if index < len(game.moves):
            legal.remove(game.moves[index])
            legal.insert(0, game.moves[index])
        result.append(
            PositionAnalysis(
                board.fen(),
                tuple(
                    Candidate(move, Score(cp=-rank * 10 if board.turn else rank * 10))
                    for rank, move in enumerate(legal[:3])
                ),
            )
        )
    return result


class FeatureTests(unittest.TestCase):
    def test_order_clock_and_mirroring(self):
        """Verify feature size, elapsed times, NaNs and side symmetry.

        Args:
            None.

        Returns:
            None; unittest assertions validate expected numeric features.
        """
        game = GameInput.from_pgn(PGN)
        rows = GameFeatureExtractor().extract(game, sample_analysis(game))
        self.assertEqual(len(set(FEATURE_NAMES)), 109)
        self.assertEqual(rows.shape, (2, 109))
        white, black = (dict(zip(FEATURE_NAMES, row)) for row in rows)
        self.assertEqual(white["self_think_mean"], 2)
        self.assertEqual(black["self_think_mean"], 3)
        self.assertEqual(white["self_clock_budget_used"], 4 / 304)
        self.assertEqual(white["self_loss_mean"], 0)
        self.assertEqual(white["opponent_think_mean"], black["self_think_mean"])
        self.assertEqual((white["result"], black["result"]), (1, 0))
        self.assertTrue(np.isnan(white["self_endgame_loss_mean"]))
        self.assertTrue(np.isnan(white["material_10"]))

    def test_score_perspective_and_missing_played(self):
        """Check move losses and reject an absent outside-top-three score.

        Args:
            None.

        Returns:
            None; assertions cover loss sign and mate endpoint probabilities.
        """
        game = GameInput.from_pgn(PGN)
        analysis = sample_analysis(game)
        first = analysis[0]
        analysis[0] = PositionAnalysis(first.fen, first.candidates[1:], Score(cp=-200))
        rows = GameFeatureExtractor().extract(game, analysis)
        self.assertGreater(rows[0, FEATURE_NAMES.index("self_loss_mean")], 0)
        analysis[0] = PositionAnalysis(first.fen, first.candidates[1:])
        extractor = GameFeatureExtractor()
        with self.assertRaisesRegex(ValueError, "played_score"):
            extractor.extract(game, analysis)
        self.assertEqual(Score(mate=2).probability(), 1)
        self.assertEqual(Score(mate=-2).probability(), 0)

        black_analysis = sample_analysis(game)
        second = black_analysis[1]
        black_analysis[1] = PositionAnalysis(
            second.fen, second.candidates[1:], Score(cp=200)
        )
        black_rows = extractor.extract(game, black_analysis)
        self.assertGreater(black_rows[1, FEATURE_NAMES.index("self_loss_mean")], 0)

    def test_custom_start_and_missing_clocks(self):
        """Check a Black-to-move sparse custom start without clocks.

        Args:
            None.

        Returns:
            None; missing times/phases/checkpoints must remain NaN.
        """
        text = """[SetUp "1"]
[FEN "7k/4P3/8/8/8/8/8/K7 b - - 0 25"]
[TimeControl "300+2"]

25... Kh7 26. e8=Q *
"""
        game = GameInput.from_pgn(text)
        rows = GameFeatureExtractor().extract(game, sample_analysis(game))
        values = dict(zip(FEATURE_NAMES, rows[1]))
        self.assertTrue(np.isnan(values["self_think_mean"]))
        self.assertTrue(np.isnan(values["self_opening_loss_mean"]))
        self.assertEqual(values["self_endgame_loss_mean"], 0)
        self.assertTrue(np.isnan(values["material_20"]))

    def test_alignment_is_required(self):
        """Reject an analysis sequence missing the final reached board.

        Args:
            None.

        Returns:
            None; the expected ValueError is asserted.
        """
        game = GameInput.from_pgn(PGN)
        truncated = sample_analysis(game)[:-1]
        extractor = GameFeatureExtractor()
        with self.assertRaisesRegex(ValueError, "every reached"):
            extractor.extract(game, truncated)

    def test_blunder_runs_and_ambiguous_mates(self):
        """Verify maximal blunder runs and rejection of unsigned mate zero.

        Args:
            None.

        Returns:
            None; run statistics and score validation are asserted.
        """
        self.assertEqual(blunder_runs([True, True, False, True, True, True]), (3, 2))
        with self.assertRaisesRegex(ValueError, "normalize mate zero"):
            Score(mate=0)

    def test_custom_start_first_think_time_is_unknown(self):
        """Do not substitute base time for an unknown custom-start clock.

        Args:
            None.

        Returns:
            None; only elapsed times with observed prior clocks are usable.
        """
        text = PGN.replace(
            '[White "Alice"]',
            '[SetUp "1"]\n'
            '[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 5"]',
        )
        game = GameInput.from_pgn(text)
        measurements = GameFeatureExtractor().measurements(game, sample_analysis(game))
        self.assertTrue(np.isnan(measurements[0]["think"]))
        self.assertTrue(np.isnan(measurements[1]["think"]))
        self.assertEqual(measurements[2]["think"], 3)


if __name__ == "__main__":
    unittest.main()
