"""Collect clocked PGNs and game-time labels for the feature-based rating model."""

import hashlib
import logging
from collections import Counter
from collections.abc import Callable
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
from tqdm.auto import tqdm

from elo_model.game import GameInput

from .chess_com_connection import ChessComConnection

logger = logging.getLogger(__name__)
COLUMNS = (
    "game_id",
    "pgn",
    "white_username",
    "black_username",
    "white_rating",
    "black_rating",
    "category",
    "termination",
    "time_control",
)


def validate_source(source: dict) -> None:
    """Check eligibility using raw API metadata, not the SDK's lossy Game type.

    Args:
        source: One raw Chess.com archive game object.

    Returns:
        None; raises ValueError with a rejection reason for ineligible games.
    """
    if source.get("rules") not in (None, "chess"):
        raise ValueError("nonstandard_variant")
    if source.get("rated") is not True:
        raise ValueError("unrated")
    if source.get("time_class") == "daily":
        raise ValueError("correspondence")
    for color in ("white", "black"):
        result = (source.get(color) or {}).get("result", "")
        if result in ("abandoned", "aborted"):
            raise ValueError(result)


def training_record(source: dict) -> dict:
    """Validate a labeled game and retain its replay-complete PGN.

    Args:
        source: Raw Chess.com game including rated and per-side result fields.

    Returns:
        Rich training metadata with labels separate from numeric features.
    """
    validate_source(source)
    game = GameInput.from_pgn(source.get("pgn") or "")
    if not game.moves or game.headers.get("Result") not in ("1-0", "0-1", "1/2-1/2"):
        raise ValueError("unfinished_or_empty")
    if any(clock is None for clock in game.clocks):
        raise ValueError("missing_clocks")
    _, _, category = game.time_control()
    if category is None:
        raise ValueError("unsupported_time_control")
    record = {
        "pgn": source["pgn"],
        "category": category,
        "termination": game.termination(),
        "time_control": game.headers["TimeControl"],
    }
    for color in ("white", "black"):
        name = game.headers.get(color.title(), "").strip()
        if not name or name == "?":
            raise ValueError("missing_username")
        rating = int(game.headers.get(f"{color.title()}Elo", "0"))
        if rating <= 0:
            raise ValueError("invalid_rating")
        record[f"{color}_username"] = name
        record[f"{color}_rating"] = rating
    termination = game.headers.get("Termination", "").lower()
    if "abandon" in termination or "abort" in termination:
        raise ValueError("abandoned_or_aborted")
    url = source.get("url", "")
    if url:
        parts = urlsplit(url)
        identity = urlunsplit(
            (parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", "")
        )
    else:
        identity = source["pgn"].strip()
    record["game_id"] = hashlib.sha256(identity.encode()).hexdigest()
    return record


class ModelDatasetBuilder:
    """Collect unique eligible games, reporting source-owner caps and rejections."""

    def __init__(self, connection: ChessComConnection):
        """Reuse a configured API connection without issuing requests yet.

        Args:
            connection: Existing bounded, retry-aware Chess.com connection.

        Returns:
            None; rejection counts start empty.
        """
        self.connection = connection
        self.rejections = Counter()

    def build(
        self,
        usernames: list[str],
        target_games: int | None = None,
        initial_games: pd.DataFrame | None = None,
        progress: bool = False,
        checkpoint: Callable[[pd.DataFrame], None] | None = None,
    ) -> pd.DataFrame:
        """Read configured recent archives; keep labels separate from predictors.

        Args:
            usernames: Supplied archive owners; no player discovery is performed.
            target_games: Optional cap on total unique eligible games, including cached games.
            initial_games: Previously saved eligible games to reuse and extend.
            progress: Display an accepted-game progress bar, including cached rows.
            checkpoint: Optional callback saving current rows after every archive.

        Returns:
            Unique eligible game metadata and raw PGNs as a DataFrame.
        """
        if target_games is not None and (
            isinstance(target_games, bool)
            or not isinstance(target_games, int)
            or target_games < 1
        ):
            raise ValueError("target_games must be a positive integer")
        initial = (
            pd.DataFrame(columns=COLUMNS)
            if initial_games is None
            else initial_games.copy()
        )
        if not set(COLUMNS).issubset(initial.columns):
            raise ValueError("Cached games do not match the model collection schema")
        initial = initial.loc[:, list(COLUMNS)].drop_duplicates("game_id")
        if target_games is not None:
            initial = initial.head(target_games)
        records = initial.to_dict("records")
        seen = set(initial["game_id"])
        counts = Counter()
        self.rejections.clear()
        limit = self.connection.config.max_games_per_player
        if target_games is not None and len(records) >= target_games:
            return initial.reset_index(drop=True)
        archives = self.connection.iter_raw_player_games(usernames)
        try:
            with tqdm(
                total=target_games,
                initial=len(records),
                desc="Collect eligible games",
                unit="game",
                disable=not progress,
            ) as bar:
                for owner, games in archives:
                    owner = owner.strip().casefold()
                    for source in sorted(
                        games, key=lambda item: item.get("end_time") or 0, reverse=True
                    ):
                        if counts[owner] >= limit or (
                            target_games is not None and len(records) >= target_games
                        ):
                            break
                        record = self._eligible_record(source, seen)
                        if record is None:
                            continue
                        seen.add(record["game_id"])
                        records.append(record)
                        counts[owner] += 1
                        bar.update(1)
                    bar.set_postfix(
                        owner=owner,
                        rejected=sum(self.rejections.values()),
                        refresh=False,
                    )
                    if checkpoint is not None:
                        checkpoint(pd.DataFrame(records, columns=COLUMNS))
                    if target_games is not None and len(records) >= target_games:
                        break
        finally:
            close = getattr(archives, "close", None)
            if close is not None:
                close()
        logger.info(
            "Collected %d games; source-owner contributions=%s; rejected=%s",
            len(records),
            dict(counts),
            dict(self.rejections),
        )
        return pd.DataFrame(records, columns=COLUMNS)

    def _eligible_record(self, source: dict, seen: set[str]) -> dict | None:
        """Validate one source game and record reasons for rejected duplicates.

        Args:
            source: Raw API game with rated and result metadata.
            seen: Durable identities already retained, including cached games.

        Returns:
            Eligible labeled game record, or None with a recorded rejection reason.
        """
        try:
            record = training_record(source)
        except ValueError as error:
            self.rejections[str(error)] += 1
            logger.info("Rejected %s: %s", source.get("url", "unknown"), error)
            return None
        if record["game_id"] in seen:
            self.rejections["duplicate"] += 1
            return None
        return record
