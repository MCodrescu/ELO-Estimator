"""Share raw monthly archives between discovery and training collection."""

import hashlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from elo_model.storage import write_json

logger = logging.getLogger(__name__)


class ArchiveSnapshot(BaseModel):
    """Versioned raw response with an identity and refresh timestamp."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal[1] = 1
    username: str
    year: int
    month: int = Field(ge=1, le=12)
    fetched_at: float = Field(ge=0, allow_inf_nan=False)
    games: list[dict]


class MonthlyArchiveCache:
    """Refresh active months while retaining immutable historical snapshots."""

    _guard: ClassVar[Lock] = Lock()
    _locks: ClassVar[dict[Path, Lock]] = {}

    def __init__(self, directory: Path, ttl_seconds: float) -> None:
        """Keep the shared cache location and active-month refresh interval.

        Args:
            directory: Root directory for raw archive snapshots.
            ttl_seconds: Maximum age of current or future month snapshots.

        Returns:
            None.
        """
        self.directory = directory.resolve()
        self.ttl_seconds = ttl_seconds

    def get(
        self, username: str, year: int, month: int, fetch: Callable[[], list[dict]]
    ) -> list[dict]:
        """Load a snapshot or download it once under a per-archive lock.

        Args:
            username: Archive owner, normalized case-insensitively for caching.
            year: Archive year.
            month: Archive month.
            fetch: Existing retry-aware downloader called only on a cache miss.

        Returns:
            Raw game dictionaries with all API fields preserved.
        """
        owner = username.strip().casefold()
        identity = hashlib.sha256(owner.encode("utf-8")).hexdigest()
        path = self.directory / identity / f"{year:04d}-{month:02d}.json"
        with self._guard:
            lock = self._locks.setdefault(path, Lock())
        with lock:
            now = datetime.now(UTC)
            cached = self._load(path, owner, year, month, now)
            if cached is not None:
                logger.debug("Reusing raw archive %s/%04d/%02d", owner, year, month)
                return cached
            snapshot = ArchiveSnapshot(
                username=owner, year=year, month=month,
                fetched_at=now.timestamp(), games=fetch(),
            )
            write_json(path, snapshot.model_dump(mode="json"))
            return snapshot.games

    def _load(
        self, path: Path, owner: str, year: int, month: int, now: datetime
    ) -> list[dict] | None:
        """Read a matching snapshot when its freshness policy permits reuse.

        Args:
            path: Snapshot file to inspect.
            owner: Expected normalized username.
            year: Expected archive year.
            month: Expected archive month.
            now: Current UTC time for active-month expiration.

        Returns:
            Raw games, or None for a missing, invalid or expired snapshot.
        """
        if not path.exists():
            return None
        try:
            snapshot = ArchiveSnapshot.model_validate_json(path.read_text(encoding="utf-8"))
        except ValidationError as error:
            logger.warning("Invalid archive cache %s; downloading again: %s", path, error)
            return None
        if (snapshot.username, snapshot.year, snapshot.month) != (owner, year, month):
            logger.warning("Archive cache identity mismatch at %s; downloading again", path)
            return None
        age = now.timestamp() - snapshot.fetched_at
        if (year, month) >= (now.year, now.month) and not 0 <= age < self.ttl_seconds:
            return None
        return snapshot.games