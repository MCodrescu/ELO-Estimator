"""Pure per-move measurements and fixed-length per-player aggregations."""

import math
from itertools import pairwise

import chess
import numpy as np

from .config import PIECE_VALUES, FeatureConfig
from .feature_schema import SIDE_GROUPS, feature_schema
from .game import GameInput, PositionAnalysis


def material_balance(board: chess.Board) -> float:
    """Return White minus Black material, excluding kings.

    Args:
        board: Replayed standard-chess position.

    Returns:
        Signed centipawn material balance.
    """
    return float(
        sum(
            value
            * (
                len(board.pieces(piece, chess.WHITE))
                - len(board.pieces(piece, chess.BLACK))
            )
            for piece, value in PIECE_VALUES.items()
        )
    )


def phase_name(board: chess.Board, config: FeatureConfig) -> str:
    """Classify material first, so sparse custom starting positions work.

    Args:
        board: Pre-move position with its actual fullmove counter.
        config: Material and opening thresholds.

    Returns:
        Opening, middlegame, or endgame name.
    """
    material = sum(
        value * len(board.pieces(piece, color))
        for piece, value in PIECE_VALUES.items()
        if piece != chess.PAWN
        for color in chess.COLORS
    )
    if material <= config.endgame_material:
        return "endgame"
    return (
        "opening" if board.fullmove_number <= config.opening_fullmoves else "middlegame"
    )


def mean(values) -> float:
    """Average finite observations without inventing missing values.

    Args:
        values: Iterable of numeric observations.

    Returns:
        Mean, or NaN when no finite observation exists.
    """
    values = np.asarray(list(values), dtype=float)
    values = values[np.isfinite(values)]
    return float(values.mean()) if len(values) else math.nan


def blunder_runs(flags: list[bool]) -> tuple[int, int]:
    """Count consecutive own-move blunder runs.

    Args:
        flags: One blunder indicator per own move, in game order.

    Returns:
        Longest run length and count of maximal runs of length at least two.
    """
    longest, clusters, current = 0, 0, 0
    for flag in [*flags, False]:
        if flag:
            current += 1
            longest = max(longest, current)
        else:
            clusters += current >= 2
            current = 0
    return longest, clusters


def validate_analysis(
    boards: list[chess.Board], analysis: list[PositionAnalysis]
) -> None:
    """Reject truncated, misaligned, bounded or illegal root analysis.

    Args:
        boards: Initial board and all reached boards in replay order.
        analysis: One ranked analysis record per board, in the same order.

    Returns:
        None; malformed or non-exact candidate analysis raises ValueError.
    """
    if len(analysis) != len(boards):
        raise ValueError("Supply analysis for initial and every reached position")
    for board, position in zip(boards, analysis):
        if board.fen() != position.fen:
            raise ValueError("Analysis FEN does not align with replay")
        moves = [candidate.move for candidate in position.candidates]
        if len(set(moves)) != len(moves) or len(moves) > 3:
            raise ValueError("Expected up to three distinct ranked moves")
        if any(chess.Move.from_uci(move) not in board.legal_moves for move in moves):
            raise ValueError("Analysis contains an illegal candidate")
        if any(
            candidate.lowerbound or candidate.upperbound
            for candidate in position.candidates
        ):
            raise ValueError("Features require exact candidate scores, not bounds")


def clock_measurement(
    start: float,
    clock: float | None,
    base: float,
    increment: float,
    config: FeatureConfig,
) -> dict:
    """Derive elapsed time without inventing a starting clock.

    Args:
        start: Previous own remaining clock, or NaN.
        clock: Current post-move clock, or None.
        base: Starting base seconds, or NaN.
        increment: Increment seconds, or NaN.
        config: Jitter and time-trouble thresholds.

    Returns:
        Think time, observed clock budget and trouble indicator for one move.
    """
    think = start + increment - clock if clock is not None else math.nan
    if math.isfinite(think):
        think = max(0, think) if think >= -config.clock_jitter else math.nan
    trouble = math.nan
    if math.isfinite(start) and math.isfinite(base):
        trouble = float(start < base * config.trouble_fraction)
    return {
        "think": think,
        "clock_start": start,
        "budget": start + increment,
        "increment": increment,
        "trouble": trouble,
    }


def clock_features(records: list[dict], config: FeatureConfig) -> dict:
    """Aggregate observed clock segments.

    Args:
        records: One side's nonempty chronological move measurements.
        config: Clock/error thresholds.

    Returns:
        Numeric clock summaries; unobserved values remain NaN.
    """
    timed = [record for record in records if math.isfinite(record["think"])]
    times = [record["think"] for record in timed]
    budget = 0
    continuous = False
    for record in records:
        usable = math.isfinite(record["think"])
        if usable:
            budget += record["increment"] if continuous else record["budget"]
        continuous = usable
    result = {
        "think_mean": mean(times),
        "think_median": float(np.median(times)) if times else math.nan,
        "think_total": float(sum(times)) if times else math.nan,
        "clock_budget_used": sum(times) / budget if budget > 0 else math.nan,
        "time_trouble_share": mean(record["trouble"] for record in records),
        "time_trouble_blunder_rate": mean(
            record["loss"] >= config.blunder
            for record in records
            if record["trouble"] == 1
        ),
        "instant_rate": mean(value <= config.instant_seconds for value in times),
        "clock_coverage": len(timed) / len(records),
    }
    return result


def context_features(game: GameInput, color: bool) -> dict:
    """Encode known game metadata with explicit unknown categories.

    Args:
        game: PGN-derived inputs, with optional result/ECO/termination headers.
        color: True for White, False for Black.

    Returns:
        Shared context and time-control features in the requested perspective.
    """
    base, increment, category = game.time_control()
    termination = game.termination()
    outcome = {"1-0": 1.0, "0-1": 0.0, "1/2-1/2": 0.5}.get(
        game.headers.get("Result"), math.nan
    )
    values = {
        "base_seconds": base,
        "increment_seconds": increment,
        "game_plies": len(game.moves),
        "is_white": float(color),
        "result": outcome if color else 1 - outcome,
    }
    for name in ("bullet", "blitz", "rapid", "classical"):
        values[name] = float(category == name) if category else math.nan
    for name in ("checkmate", "resignation", "timeout", "draw", "other", "unknown"):
        values[f"termination_{name}"] = float(termination == name)
    eco = game.headers.get("ECO", "")[:1]
    eco = eco if eco in ("A", "B", "C", "D", "E") else "unknown"
    for name in ("A", "B", "C", "D", "E", "unknown"):
        values[f"eco_{name}"] = float(eco == name)
    return values


def material_features(boards: list[chess.Board], color: bool) -> dict:
    """Measure only checkpoints actually reached in the supplied history.

    Args:
        boards: Initial and reached positions, preserving fullmove counters.
        color: Perspective for signed balances.

    Returns:
        Material at fullmoves 10/20/30 and at end; absent checkpoints are NaN.
    """
    sign = 1 if color else -1
    values = {"material_end": sign * material_balance(boards[-1])}
    for fullmove in (10, 20, 30):
        values[f"material_{fullmove}"] = math.nan
        for before, after in pairwise(boards):
            if before.fullmove_number == fullmove and before.turn == chess.BLACK:
                values[f"material_{fullmove}"] = sign * material_balance(after)
    return values


def flag_value(game: GameInput, color: bool) -> float:
    """Use explicit timeout plus decisive result, never a zero-clock heuristic.

    Args:
        game: Game with optional termination and result headers.
        color: Side whose flag indicator is requested.

    Returns:
        One if this side flagged, zero if known not to, otherwise NaN.
    """
    termination = game.termination()
    if termination == "unknown":
        return math.nan
    if termination != "timeout":
        return 0.0
    winner = {"1-0": chess.WHITE, "0-1": chess.BLACK}.get(game.headers.get("Result"))
    return float(color != winner) if winner is not None else math.nan


class GameFeatureExtractor:
    """Build two ordered numeric rows from a game and precomputed analysis."""

    def __init__(self, config: FeatureConfig | None = None):
        """Configure shared transformations without starting external work.

        Args:
            config: Feature thresholds; defaults to the versioned standard set.

        Returns:
            None.
        """
        self.config = config or FeatureConfig()
        self.schema = feature_schema(self.config.schema_version)

    def measurements(
        self, game: GameInput, analysis: list[PositionAnalysis]
    ) -> list[dict]:
        """Measure each move without engine or network access.

        Args:
            game: Replay-complete moves, clocks and metadata.
            analysis: Aligned White-perspective initial/reached-board scores.

        Returns:
            One chronological measurement dictionary per played move.
        """
        boards = game.boards()
        validate_analysis(boards, analysis)
        base, increment, _ = game.time_control()
        previous = {
            color: base if game.initial_fen == chess.STARTING_FEN else math.nan
            for color in chess.COLORS
        }
        records = []
        for index, move in enumerate(game.moves):
            board = boards[index]
            clock = game.clocks[index]
            following = analysis[index + 1]
            reached_score = following.evaluation
            if reached_score is None and following.candidates:
                reached_score = following.candidates[0].score
            record = self._move(board, move, analysis[index], reached_score)
            record.update(
                clock_measurement(
                    previous[board.turn], clock, base, increment, self.config
                )
            )
            previous[board.turn] = clock if clock is not None else math.nan
            records.append(record)
        return records

    def _move(
        self, board: chess.Board, move: str, position: PositionAnalysis,
        reached_score=None,
    ) -> dict:
        """Measure mover-relative accuracy using the next reached evaluation.

        Args:
            board: Position immediately before the move.
            move: Played legal UCI move.
            position: Exact single-PV root score.
            reached_score: Next reached position's White-perspective evaluation.

        Returns:
            Per-move losses, best-move match and phase.
        """
        config = self.config
        if not position.candidates:
            raise ValueError("A played position needs ranked engine candidates")
        best = position.candidates[0].score
        if len(position.candidates) != 1:
            raise ValueError("Single-PV schema requires one candidate per played position")
        played = reached_score
        if played is None:
            raise ValueError("Next-position scoring requires a reached-position evaluation")
        sign = 1 if board.turn else -1
        best_probability = best.probability(config.probability_scale)
        played_probability = played.probability(config.probability_scale)
        before = best_probability if board.turn else 1 - best_probability
        after = played_probability if board.turn else 1 - played_probability
        return {
            "color": board.turn,
            "phase": phase_name(board, config),
            "before": before,
            "after": after,
            "loss": max(0, before - after),
            "cp_loss": max(
                0,
                sign
                * (best.centipawns(config.mate_cp) - played.centipawns(config.mate_cp)),
            ),
            "top1": move == position.candidates[0].move,
        }

    def _side(self, records: list[dict], flagged: float) -> dict[str, float]:
        """Aggregate one side without padding missing phases or observations.

        Args:
            records: Chronological own-move measurements.
            flagged: Explicit timeout indicator or NaN.

        Returns:
            Observed numeric features without a self/opponent prefix.
        """
        config = self.config
        names = [name for group in SIDE_GROUPS.values() for name in group]
        result = dict.fromkeys(names, math.nan)
        if not records:
            return result
        losses = np.array([record["loss"] for record in records])
        result.update(
            loss_mean=float(losses.mean()),
            loss_median=float(np.median(losses)),
            loss_std=float(losses.std()),
            loss_p90=float(np.quantile(losses, 0.9)),
            cp_loss_mean=mean(record["cp_loss"] for record in records),
            inaccuracy_rate=mean(
                (losses >= config.inaccuracy) & (losses < config.mistake)
            ),
            mistake_rate=mean((losses >= config.mistake) & (losses < config.blunder)),
            blunder_rate=mean(losses >= config.blunder),
            top1_rate=mean(record["top1"] for record in records),
        )
        for phase in ("opening", "middlegame", "endgame"):
            subset = [record for record in records if record["phase"] == phase]
            result[f"{phase}_loss_mean"] = mean(record["loss"] for record in subset)
            result[f"{phase}_blunder_rate"] = mean(
                record["loss"] >= config.blunder for record in subset
            )
            result[f"{phase}_top1_rate"] = mean(record["top1"] for record in subset)
        winning = [
            record
            for record in records
            if record["before"] >= config.winning_probability
        ]
        losing = [
            record
            for record in records
            if record["before"] <= config.losing_probability
        ]
        equal = [
            record
            for record in records
            if config.losing_probability < record["before"] < config.winning_probability
        ]
        for name, subset in (
            ("winning", winning),
            ("equal", equal),
            ("losing", losing),
        ):
            result[f"{name}_loss_mean"] = mean(record["loss"] for record in subset)
        result["winning_retention"] = mean(
            record["after"] >= config.winning_probability for record in winning
        )
        result["losing_resilience"] = mean(
            record["loss"] < config.inaccuracy for record in losing
        )
        result["max_blunder_streak"], result["blunder_clusters"] = blunder_runs(
            [record["loss"] >= config.blunder for record in records]
        )
        result.update(clock_features(records, config))
        result["flagged"] = flagged
        return result

    def extract(self, game: GameInput, analysis: list[PositionAnalysis]) -> np.ndarray:
        """Build the fixed feature matrix with NaN for unobserved values.

        Args:
            game: Replay-complete game; ratings are not required.
            analysis: One aligned record for the initial and every reached board.

        Returns:
            Float64 array of shape (2, 93), White first, in schema order.
        """
        records = self.measurements(game, analysis)
        boards = game.boards()
        sides = {}
        for color in chess.COLORS:
            sides[color] = self._side(
                [record for record in records if record["color"] == color],
                flag_value(game, color),
            )
        rows = []
        for color in (chess.WHITE, chess.BLACK):
            values = dict.fromkeys(self.schema.names, math.nan)
            values.update(context_features(game, color))
            for prefix, side in (("self", color), ("opponent", not color)):
                values.update(
                    {f"{prefix}_{name}": value for name, value in sides[side].items()}
                )
            values.update(material_features(boards, color))
            rows.append([values[name] for name in self.schema.names])
        return np.asarray(rows, dtype=np.float64)
