import logging
from bisect import bisect_right
from collections import deque

from chessdotcom.endpoints.player_games_by_month import Game

from .chess_com_connection import ChessComConnection
from .discovery_progress import DiscoveryProgress

logger = logging.getLogger(__name__)


class RatingStratifiedPlayerSearch:
    """Prioritize underfilled rating bands while retaining over-quota bridges.

    Ratings are latest observed standard-game ratings in one time class within
    the configured months, not necessarily a player's latest site-wide rating.
    Quotas apply to returned usernames, not expansions or game appearances.
    """

    def __init__(self, connection: ChessComConnection) -> None:
        """Prepare quotas and discovery state without sending requests.

        Args:
            connection: Archive client with rating-band and search limits.

        Returns:
            None.
        """
        self.connection = connection
        self._edges = connection.config.search_band_edges
        base, remainder = divmod(connection.config.search_max_players, len(self._edges))
        self._quotas = list(connection.config.search_band_quotas) if (
            connection.config.search_band_quotas is not None
        ) else [base + (index < remainder) for index in range(len(self._edges))]
        self._reset()

    def _reset(self) -> None:
        """Clear mutable discovery state for an independent search.

        Args:
            None.

        Returns:
            None; configured band quotas are retained.
        """
        self._names: dict[str, str] = {}
        self._depths: dict[str, int] = {}
        self._ratings: dict[str, tuple[int, int]] = {}
        self._buckets: list[dict[str, None]] = [{} for _ in self._edges]
        self._queues: list[deque[str]] = [deque() for _ in range(len(self._edges) + 1)]
        self._queued_band: dict[str, int] = {}
        self._expanded: set[str] = set()
        self._best_retained = 0
        self._last_improvement = 0
        self.stop_reason = "not started"

    def _label(self, band: int) -> str:
        """Describe one configured rating interval.

        Args:
            band: Zero-based interval index.

        Returns:
            Inclusive display range, or an open-ended top-band label.
        """
        if band == len(self._edges) - 1:
            return f"{self._edges[band]}+"
        return f"{self._edges[band]}-{self._edges[band + 1] - 1}"

    @property
    def band_quotas(self) -> dict[str, int]:
        """Expose the requested number of usernames in each rating band.

        Args:
            None.

        Returns:
            Display-label to quota mapping.
        """
        return {self._label(band): quota for band, quota in enumerate(self._quotas)}

    @property
    def band_counts(self) -> dict[str, int]:
        """Expose discovered counts capped at each return quota.

        Args:
            None.

        Returns:
            Display-label to retained-count mapping.
        """
        return {
            self._label(band): min(len(bucket), self._quotas[band])
            for band, bucket in enumerate(self._buckets)
        }

    @property
    def expanded_count(self) -> int:
        """Count players whose archives have been scheduled for expansion.

        Args:
            None.

        Returns:
            Number of expanded identities.
        """
        return len(self._expanded)

    def _band(self, rating: int) -> int:
        """Locate a validated rating in the configured intervals.

        Args:
            rating: Nonnegative observed rating.

        Returns:
            Zero-based rating-band index.
        """
        return bisect_right(self._edges, rating) - 1

    def _enqueue(self, key: str) -> None:
        """Schedule an unexpanded player in its latest known band.

        Args:
            key: Normalized identity already registered in discovery state.

        Returns:
            None; expanded or depth-limited identities are not enqueued.
        """
        limit = self.connection.config.search_max_depth
        if key in self._expanded or (limit is not None and self._depths[key] >= limit):
            return
        band = (
            self._band(self._ratings[key][1])
            if key in self._ratings
            else len(self._edges)
        )
        if self._queued_band.get(key) != band:
            self._queued_band[key] = band
            self._queues[band].append(key)

    def _register(self, name: str, depth: int) -> str:
        """Record a newly encountered identity and its first discovery depth.

        Args:
            name: Display username from an archive.
            depth: Opponent graph distance from the seed.

        Returns:
            Case-folded identity used for state lookups.
        """
        key = name.casefold()
        if key not in self._names:
            self._names[key] = name
            self._depths[key] = depth
            self._enqueue(key)
        return key

    def _record_rating(
        self, key: str, rating: int | None, timestamp: int, source: str
    ) -> None:
        """Update a player's band using its latest valid observed rating.

        Args:
            key: Registered normalized player identity.
            rating: Game-time rating, or None when absent.
            timestamp: Game completion timestamp.
            source: Archive/game identity for diagnostics.

        Returns:
            None; invalid or stale observations are logged or ignored.
        """
        if not isinstance(rating, int) or isinstance(rating, bool) or rating < 0:
            logger.warning(
                "Missing/invalid %s rating for %s in %s",
                self.connection.config.search_time_class,
                self._names[key],
                source,
            )
            return
        previous = self._ratings.get(key)
        if previous is not None:
            if timestamp < previous[0]:
                return
            if timestamp == previous[0]:
                if rating != previous[1]:
                    logger.warning(
                        "Conflicting same-time ratings for %s in %s; keeping first",
                        self._names[key],
                        source,
                    )
                return
            if self._band(previous[1]) != self._band(rating):
                self._buckets[self._band(previous[1])].pop(key, None)
        self._ratings[key] = (timestamp, rating)
        self._buckets[self._band(rating)][key] = None
        self._enqueue(key)

    def _observe(self, owner: str, game: Game) -> None:
        """Register valid participant links and ratings from one archive game.

        Args:
            owner: Player whose archive supplied the game.
            game: Typed SDK game, filtered by configured variant/time class.

        Returns:
            None; malformed links are logged without inventing identities.
        """
        config = self.connection.config
        if game.rules != "chess" or game.time_class != config.search_time_class:
            logger.debug(
                "Ignoring %s: not standard %s", game.url, config.search_time_class
            )
            return
        players = (game.white, game.black)
        if any(
            player is None or not player.username or not player.username.strip()
            for player in players
        ):
            logger.warning(
                "Skipping player links in %s: missing participant username", game.url
            )
            return
        names = [
            player.username.strip()
            for player in players
            if player is not None and player.username is not None
        ]
        if owner.casefold() not in {name.casefold() for name in names}:
            logger.warning(
                "Skipping %s: archive owner %s is not a participant", game.url, owner
            )
            return
        depth = self._depths[owner.casefold()] + 1
        keys = [self._register(name, depth) for name in names]
        timestamp = game.end_time
        if (
            not isinstance(timestamp, int)
            or isinstance(timestamp, bool)
            or timestamp < 0
        ):
            logger.warning(
                "Cannot assign ratings from %s: missing/invalid end_time", game.url
            )
            return
        for key, player in zip(keys, players):
            if player is not None:
                self._record_rating(key, player.rating, timestamp, game.url or owner)

    def _next_batch(self) -> list[str]:
        """Choose a bounded expansion batch favoring underfilled rating bands.

        Args:
            None.

        Returns:
            Display usernames to expand next, or an empty list when exhausted.
        """
        deficits = [
            max(0, quota - len(bucket))
            for quota, bucket in zip(self._quotas, self._buckets)
        ]
        missing = [band for band, deficit in enumerate(deficits) if deficit]

        def priority(band: int) -> tuple[int, int, int]:
            """Rank bands by deficit while retaining useful bridge players.

            Args:
                band: Queue index, including the unknown-rating queue.

            Returns:
                Stable sort key for expansion priority.
            """
            if band == len(self._edges):
                return (1, 0, band)
            if deficits[band]:
                return (0, -deficits[band], band)
            distance = min((abs(band - target) for target in missing), default=0)
            return (2, distance, band)

        batch: list[str] = []
        remaining = self.connection.config.search_max_expansions - self.expanded_count
        for band in sorted(range(len(self._queues)), key=priority):
            while self._queues[band] and len(batch) < min(
                remaining, self.connection.config.max_workers
            ):
                key = self._queues[band].popleft()
                if self._queued_band.get(key) != band or key in self._expanded:
                    continue
                del self._queued_band[key]
                self._expanded.add(key)
                batch.append(self._names[key])
        return batch

    def _initialize_seeds(
        self, seed_username: str | None, seed_usernames: list[str] | None
    ) -> str:
        """Reset the graph and register validated starting identities.

        Args:
            seed_username: Primary seed, or None to use the configured seed.
            seed_usernames: Additional seeds spanning otherwise disconnected bands.

        Returns:
            Trimmed primary seed for progress logging.
        """
        seed = (
            self.connection.config.seed_username if seed_username is None else seed_username
        ).strip()
        if not seed:
            raise ValueError("seed_username must not be empty")
        self._reset()
        self._register(seed, 0)
        for additional_seed in seed_usernames or []:
            if not additional_seed.strip():
                raise ValueError("seed_usernames must not contain empty usernames")
            self._register(additional_seed.strip(), 0)
        return seed

    def _stalled(self) -> bool:
        """Track retained coverage and bound expansions without improvement.

        Args:
            None.

        Returns:
            Whether the configured no-improvement budget has been exhausted.
        """
        retained = sum(self.band_counts.values())
        if retained > self._best_retained:
            self._best_retained = retained
            self._last_improvement = self.expanded_count
        limit = self.connection.config.search_stagnation_expansions
        return limit is not None and self.expanded_count - self._last_improvement >= limit

    def _accepted_players(self) -> list[str]:
        """Collect within-quota usernames and report remaining band shortages.

        Args:
            None.

        Returns:
            Accepted display usernames in stable rating-band order.
        """
        result = [
            self._names[key]
            for bucket, quota in zip(self._buckets, self._quotas)
            for key in list(bucket)[:quota]
        ]
        for band, (bucket, quota) in enumerate(zip(self._buckets, self._quotas)):
            if len(bucket) < quota:
                logger.warning(
                    "Unfilled rating band %s: %d/%d usernames",
                    self._label(band), len(bucket), quota,
                )
        return result

    def search(
        self, seed_username: str | None = None, *, seed_usernames: list[str] | None = None,
        progress: bool = False,
    ) -> list[str]:
        """Return a partial within-quota sample if the expansion budget is exhausted.

        Equal quotas differ by at most one; remainder slots go to lower bands.
        The seed is a discovery bridge and is returned only if rated and within
        quota. Optional depth bounds use each player's first discovery path.

        Args:
            seed_username: Starting username, or the configured seed.
            seed_usernames: Optional additional starting players across rating bands.
            progress: Display per-band quotas, remaining usernames and estimated time.

        Returns:
            Unique usernames within rating quotas, possibly a partial sample.
        """
        config = self.connection.config
        seed = self._initialize_seeds(seed_username, seed_usernames)
        logger.info(
            "Starting rating-stratified search from %s: %s, %d expansion budget, quotas %s",
            seed,
            config.search_time_class,
            config.search_max_expansions,
            self.band_quotas,
        )
        reason = "band quotas filled"
        with DiscoveryProgress(
            config.search_time_class, self.band_quotas,
            config.search_max_expansions * len(config.months), progress,
        ) as display:
            while any(
                len(bucket) < quota for bucket, quota in zip(self._buckets, self._quotas)
            ):
                batch = self._next_batch()
                if not batch:
                    reason = (
                        "expansion budget reached"
                        if self.expanded_count >= config.search_max_expansions
                        else "frontier exhausted or depth limit reached"
                    )
                    break
                logger.info(
                    "Expanding %d players (%d/%d budget): %s",
                    len(batch),
                    self.expanded_count,
                    config.search_max_expansions,
                    batch,
                )
                for owner, games in self.connection.iter_player_games(batch):
                    for game in games:
                        self._observe(owner, game)
                    display.update(owner, self.band_counts)
                logger.info("Rating-band progress: %s", self.band_counts)
                if self._stalled():
                    reason = "no rating-quota improvement within stagnation budget"
                    break
            display.finish(reason)

        self.stop_reason = reason
        result = self._accepted_players()
        logger.info(
            "Stratified search complete: %d accepted, %d discovered, %d expanded; %s",
            len(result),
            len(self._names),
            self.expanded_count,
            reason,
        )
        return result
