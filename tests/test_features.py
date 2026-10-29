"""Deterministic feature tests require neither Stockfish nor network access."""

import unittest
from dataclasses import asdict, replace

import numpy as np

from elo_model import (
    FEATURE_NAMES,
    Candidate,
    GameFeatureExtractor,
    GameInput,
    PositionAnalysis,
    Score,
)
from elo_model.config import ANALYSIS_VERSION, EngineConfig, FeatureConfig
from elo_model.features import blunder_runs

PGN = """[White "Alice"]
[Black "Bob"]
[TimeControl "300+2"]
[Result "1-0"]
[Termination "Alice won by resignation"]

1. e4 {[%clk 0:05:01]} e5 {[%clk 0:05:00]} 2. Nf3 {[%clk 0:05:00]} Nc6 {[%clk 0:04:58]} 1-0
"""


def sample_engine_metadata():
    """Provide complete synthetic provenance for the supported analysis contract.

    Args:
        None.

    Returns:
        Synthetic engine identity, binary fingerprint and current settings.
    """
    settings = asdict(EngineConfig("mock"))
    settings.pop("executable")
    return {"id": {"name": "Test engine"}, "binary_sha256": "a" * 64,
            "settings": settings, "analysis_version": ANALYSIS_VERSION}


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
                tuple(Candidate(move, Score(cp=0)) for move in legal[:1]),
                evaluation=Score(cp=0),
            )
        )
    return result


class FeatureTests(unittest.TestCase):
    def test_next_position_perspective_and_no_candidate_substitution(self):
        """Use reached scores for both colors even for a top-one played move.

        Args:
            None.

        Returns:
            None; chronological White-score losses and reduced schema are checked.
        """
        game = GameInput.from_pgn(PGN)
        analysis = [replace(position, candidates=position.candidates[:1],
                            evaluation=Score(cp=score))
                    for position, score in zip(sample_analysis(game), (0, -200, 200, 0, 0))]
        extractor = GameFeatureExtractor(FeatureConfig(schema_version=2))
        records = extractor.measurements(game, analysis)
        self.assertGreater(records[0]["loss"], 0)
        self.assertGreater(records[1]["loss"], 0)
        self.assertEqual(records[0]["cp_loss"], 200)
        self.assertEqual(records[1]["cp_loss"], 200)
        self.assertEqual(extractor.extract(game, analysis).shape, (2, 93))
        self.assertFalse(any("top3" in name or "complexity" in name
                             or "only_move" in name for name in extractor.schema.names))

    def test_next_position_terminal_mate_and_draw(self):
        """Preserve terminal outcomes instead of fabricating candidate moves.

        Args:
            None.

        Returns:
            None; a terminal mate scores the preceding move in White perspective.
        """
        game = GameInput.from_pgn('1. f3 e5 2. g4 Qh4# 0-1')
        analysis = sample_analysis(game)
        analysis = [replace(position, candidates=position.candidates[:1],
                            evaluation=Score(cp=0)) for position in analysis]
        analysis[-1] = replace(analysis[-1], candidates=(), evaluation=Score(mate=-1))
        records = GameFeatureExtractor(FeatureConfig(schema_version=2)).measurements(game, analysis)
        self.assertEqual(records[-1]["after"], 1)
        analysis[-1] = replace(analysis[-1], evaluation=Score(cp=0))
        self.assertEqual(GameFeatureExtractor(FeatureConfig(schema_version=2)).measurements(
            game, analysis)[-1]["after"], 0.5)

    def test_order_clock_and_mirroring(self):
        """Verify feature size, elapsed times, NaNs and side symmetry.

        Args:
            None.

        Returns:
            None; unittest assertions validate expected numeric features.
        """
        game = GameInput.from_pgn(PGN)
        rows = GameFeatureExtractor().extract(game, sample_analysis(game))
        self.assertEqual(len(set(FEATURE_NAMES)), 93)
        self.assertEqual(rows.shape, (2, 93))
        white, black = (dict(zip(FEATURE_NAMES, row)) for row in rows)
        self.assertEqual(white["self_think_mean"], 2)
        self.assertEqual(black["self_think_mean"], 3)
        self.assertEqual(white["self_clock_budget_used"], 4 / 304)
        self.assertEqual(white["self_loss_mean"], 0)
        self.assertEqual(white["opponent_think_mean"], black["self_think_mean"])
        self.assertEqual((white["result"], black["result"]), (1, 0))
        self.assertTrue(np.isnan(white["self_endgame_loss_mean"]))
        self.assertTrue(np.isnan(white["material_10"]))

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
