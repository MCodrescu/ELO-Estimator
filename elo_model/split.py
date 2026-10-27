"""Deterministic target-player and underlying-game isolation."""

import hashlib
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEFAULT_SEED, DEFAULT_SPLIT_FRACTIONS, PARTITIONS


def normalize_player(username: str) -> str:
    """Use platform case-insensitive identities, including external opponents.

    Args:
        username: Chess.com identity, possibly with whitespace or mixed case.

    Returns:
        Stripped, case-folded identity; missing identities raise ValueError.
    """
    result = str(username).strip().casefold()
    if not result or result in ("?", "none", "nan"):
        raise ValueError("Missing player identity")
    return result


def hash_fraction(text: str, seed: int) -> float:
    """Return a stable seeded fraction independent of Python's process hash.

    Args:
        text: Player or game identity.
        seed: Reproducible partition seed.

    Returns:
        Numeric fraction in [0, 1).
    """
    digest = hashlib.sha256(f"{seed}:{text}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


@dataclass(frozen=True)
class PlayerSplitter:
    """Keep games disjoint; optionally require both participants to be disjoint."""

    fractions: tuple[float, float, float] = DEFAULT_SPLIT_FRACTIONS
    seed: int = DEFAULT_SEED
    strict_participants: bool = False

    def __post_init__(self):
        """Check that all three fractions are positive and sum to one.

        Args:
            None.

        Returns:
            None; invalid fractions raise ValueError.
        """
        if len(self.fractions) != 3 or any(
            not math.isfinite(value) or value <= 0 for value in self.fractions
        ):
            raise ValueError("Provide three positive split fractions")
        if not math.isclose(sum(self.fractions), 1):
            raise ValueError("Split fractions must sum to one")

    def player_partition(self, username: str) -> str:
        """Assign a player independently of their rating or game results.

        Args:
            username: Identity to normalize and hash.

        Returns:
            Train, validation, or test label.
        """
        fraction = hash_fraction(normalize_player(username), self.seed)
        index = int(np.searchsorted(np.cumsum(self.fractions), fraction, side="right"))
        return PARTITIONS[min(index, 2)]

    def assign(self, games: pd.DataFrame, require_all: bool = True) -> pd.DataFrame:
        """Create eligible row masks before engine work, without consulting labels.

        Args:
            games: Unique game records containing both participant identities.
            require_all: Fail unless every partition retains a game.

        Returns:
            Game manifest with partition assignments and eligible side masks.
        """
        if games["game_id"].duplicated().any():
            raise ValueError(
                "Duplicate game identities must be removed before splitting"
            )
        result = games.copy()
        white = result["white_username"].map(self.player_partition)
        black = result["black_username"].map(self.player_partition)
        selected = np.where(
            result["game_id"].map(lambda value: hash_fraction(str(value), self.seed))
            < 0.5,
            white,
            black,
        )
        result["partition"] = selected
        result["white_partition"] = white
        result["black_partition"] = black
        result["keep_white"] = white == selected
        result["keep_black"] = black == selected
        if self.strict_participants:
            result = result[white == black].copy()
        if require_all and set(result["partition"]) != set(PARTITIONS):
            raise ValueError(
                "Insufficient player coverage: collect more games or adjust split fractions"
            )
        return result.reset_index(drop=True)


def assert_disjoint(frame: pd.DataFrame) -> None:
    """Reject target or underlying-game overlap and invalid split labels.

    Args:
        frame: Side rows with game_id, target_username and partition columns.

    Returns:
        None; leakage or unknown labels raise ValueError.
    """
    if not set(frame["partition"]).issubset(PARTITIONS):
        raise ValueError("Unknown partition")
    for column in ("game_id", "target_username"):
        values = frame[column]
        if column == "target_username":
            values = values.map(normalize_player)
        assignments = pd.DataFrame({"value": values, "partition": frame["partition"]})
        if (assignments.groupby("value")["partition"].nunique() > 1).any():
            raise ValueError(f"Cross-partition leakage in {column}")
