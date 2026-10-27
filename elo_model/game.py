"""Rating-independent PGN and White-perspective engine input contracts."""

import io
import math
import re

import chess
import chess.pgn
from pydantic import StrictInt
from pydantic.dataclasses import dataclass

from .config import (
    TIME_CATEGORIES,
    TIME_CATEGORY_LIMITS,
    TIME_CONTROL_INCREMENT_MOVES,
    EngineConfig,
    FeatureConfig,
)


@dataclass(frozen=True)
class Score:
    """One White-perspective score: centipawns OR signed mate distance."""

    cp: float | None = None
    mate: StrictInt | None = None

    def __post_init__(self):
        """Reject ambiguous or nonfinite engine scores.

        Args:
            None.

        Returns:
            None; invalid score representations raise ValueError.
        """
        if (self.cp is None) == (self.mate is None):
            raise ValueError("Provide exactly one of cp or mate")
        if self.cp is not None and not math.isfinite(self.cp):
            raise ValueError("Centipawns must be finite")
        if self.mate is not None and (
            isinstance(self.mate, bool)
            or not isinstance(self.mate, int)
            or self.mate == 0
        ):
            raise ValueError(
                "Mate must be a nonzero signed integer; normalize mate zero first"
            )

    def centipawns(self, mate_cap: float = FeatureConfig.mate_cp) -> float:
        """Return centipawns or a signed mate surrogate.

        Args:
            mate_cap: Absolute centipawn value substituted for mate scores.

        Returns:
            White-perspective numeric evaluation.
        """
        if self.cp is not None:
            return float(self.cp)
        return mate_cap if self.mate > 0 else -mate_cap

    def probability(self, scale: float = FeatureConfig.probability_scale) -> float:
        """Convert to fixed logistic win probability, not calibrated WDL.

        Args:
            scale: Positive centipawn scale for the logistic transform.

        Returns:
            White win probability in [0, 1], with mates mapped to endpoints.
        """
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("Probability scale must be finite and positive")
        if self.mate is not None:
            return float(self.mate > 0)
        return 1 / (1 + math.exp(-max(-700, min(700, self.cp / scale))))


@dataclass(frozen=True)
class Candidate:
    """Ranked legal UCI root move and its White-perspective score."""

    move: str
    score: Score
    depth: int = EngineConfig.depth
    lowerbound: bool = False
    upperbound: bool = False


@dataclass(frozen=True)
class PositionAnalysis:
    """Analysis aligned with one reached board; played score is root-restricted."""

    fen: str
    candidates: tuple[Candidate, ...]
    played_score: Score | None = None
    depth: int = EngineConfig.depth
    evaluation: Score | None = None


@dataclass(frozen=True)
class GameInput:
    """Replay-complete PGN inputs; no rating is required for extraction."""

    initial_fen: str
    moves: tuple[str, ...]
    clocks: tuple[float | None, ...]
    headers: dict[str, str]

    @classmethod
    def from_pgn(cls, text: str) -> "GameInput":
        """Parse one legal standard-chess game, including custom starting FENs.

        Args:
            text: One PGN with optional clock comments and rating headers.

        Returns:
            Replay-complete inputs; missing clocks remain None.
        """
        stream = io.StringIO(text)
        game = chess.pgn.read_game(stream)
        if game is None or game.errors:
            raise ValueError("Missing or invalid PGN")
        if chess.pgn.read_game(stream) is not None:
            raise ValueError("Expected exactly one PGN game")
        board = game.board()
        if board.uci_variant != "chess" or board.chess960 or not board.is_valid():
            raise ValueError("Expected a legal standard-chess starting position")
        moves, clocks = [], []
        for node in game.mainline():
            if node.move not in board.legal_moves:
                raise ValueError("Illegal PGN move")
            moves.append(node.move.uci())
            clock = node.clock()
            if clock is not None and (not math.isfinite(clock) or clock < 0):
                raise ValueError("Invalid clock annotation")
            clocks.append(clock)
            board.push(node.move)
        return cls(game.board().fen(), tuple(moves), tuple(clocks), dict(game.headers))

    def boards(self) -> list[chess.Board]:
        """Replay moves from the original position, preserving history.

        Args:
            None.

        Returns:
            Initial board followed by a copy after each played move.
        """
        if len(self.clocks) != len(self.moves):
            raise ValueError("Clocks and moves must align")
        board = chess.Board(self.initial_fen)
        result = [board.copy()]
        for text in self.moves:
            board.push_uci(text)
            result.append(board.copy())
        return result

    def time_control(self) -> tuple[float, float, str | None]:
        """Parse simple live controls; classify using base + 40 increments.

        Args:
            None.

        Returns:
            Base seconds, increment seconds, category; unknown is NaN/NaN/None.
        """
        match = re.fullmatch(r"(\d+)(?:\+(\d+))?", self.headers.get("TimeControl", ""))
        if not match or int(match[1]) <= 0:
            return math.nan, math.nan, None
        base, increment = float(match[1]), float(match[2] or 0)
        duration = base + TIME_CONTROL_INCREMENT_MOVES * increment
        categories = zip(TIME_CATEGORY_LIMITS, TIME_CATEGORIES)
        category = next(name for limit, name in categories if duration < limit)
        return base, increment, category

    def termination(self) -> str:
        """Resolve terminal board outcomes and explicit termination metadata.

        Args:
            None.

        Returns:
            Canonical termination category, including unknown when unavailable.
        """
        board = self.boards()[-1]
        text = self.headers.get("Termination", "").lower()
        if board.is_checkmate():
            return "checkmate"
        if "resign" in text:
            return "resignation"
        if any(word in text for word in ("time", "flag")):
            return "timeout"
        if self.headers.get("Result") == "1/2-1/2":
            return "draw"
        return "other" if text and text != "?" else "unknown"
