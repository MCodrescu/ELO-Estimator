import logging
import time
from collections import deque
from collections.abc import Callable, Generator, Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from itertools import islice
from typing import ParamSpec, TypeVar

from chessdotcom import ChessDotComClient, RateLimitHandler
from chessdotcom.endpoints.player_games_by_month import Game
from chessdotcom.errors import ChessDotComClientError
from requests.exceptions import ConnectionError, SSLError, Timeout
from retryable import retry

from elo_model.config import DatasetConfig

logger = logging.getLogger(__name__)
P = ParamSpec("P")
T = TypeVar("T")
R = TypeVar("R")


class ChessComConnection:
    """Bounded parallel access with the wrapper's fixed 30-second timeout."""

    def __init__(self, config: DatasetConfig) -> None:
        """Prepare the SDK client without sending requests.

        Args:
            config: Shared proxy, retry, archive and concurrency settings.

        Returns:
            None.
        """
        self.config = config

        self.client = ChessDotComClient(
            user_agent=config.user_agent,
            request_config={
                "headers": {"Accept": "application/json"},
                "proxies": {
                    "http": config.proxy_url,
                    "https": config.proxy_url,
                },
            },
            rate_limit_handler=RateLimitHandler(retries=0),
        )

    def _should_retry(self, error: Exception) -> bool:
        """Recognize retryable errors and honor server-requested backoff.

        Args:
            error: Transport or SDK exception from a request attempt.

        Returns:
            Whether the configured retry wrapper should make another attempt.
        """
        if isinstance(error, SSLError):
            return False

        if isinstance(error, (ConnectionError, Timeout)):
            return True

        if not isinstance(error, ChessDotComClientError):
            return False

        if error.status_code not in (429, 500, 502, 503, 504):
            return False

        retry_after = next(
            (
                value
                for key, value in error.headers.items()
                if key.lower() == "retry-after"
            ),
            None,
        )
        if retry_after is not None:
            # Retry-After may contain seconds or an absolute HTTP date.
            try:
                delay = float(retry_after)
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
                except (ValueError, TypeError, OverflowError):
                    logger.warning("Invalid Retry-After header: %r", retry_after)
                    return True
            if delay > 0:
                time.sleep(delay)
        return True

    def _request(
        self, resource: str, endpoint: Callable[P, T], *args: P.args, **kwargs: P.kwargs
    ) -> T:
        """Call one endpoint with shared retry policy and resource-aware logging.

        Args:
            resource: Human-readable endpoint identity for diagnostics.
            endpoint: Bound SDK endpoint to invoke.
            args: Positional endpoint arguments.
            kwargs: Keyword endpoint arguments.

        Returns:
            The endpoint response; exhausted or permanent errors propagate.
        """

        @retry(
            count=self.config.retries,
            delay=self.config.retry_delay,
            backoff=self.config.retry_backoff,
            callback=self._should_retry,
        )
        def request(resource: str) -> T:
            """Execute a single attempt without suppressing failures.

            Args:
                resource: Endpoint identity included in progress logs.

            Returns:
                SDK endpoint response.
            """
            logger.info("Request started: %s", resource)

            result = endpoint(*args, **kwargs)

            logger.info("Request completed: %s", resource)
            return result

        return request(resource)

    def _parallel_map(
        self, function: Callable[[T], R], items: Iterable[T]
    ) -> Iterator[R]:
        """Bound in-flight requests while yielding results in input order.

        Args:
            function: Request operation to run for each item.
            items: Lazily consumed request jobs.

        Returns:
            Ordered iterator of completed results; failures propagate.
        """
        items = iter(items)
        with ThreadPoolExecutor(
            max_workers=self.config.max_workers, thread_name_prefix="chess-api"
        ) as pool:
            pending = deque(
                pool.submit(function, item)
                for item in islice(items, self.config.max_workers)
            )
            try:
                while pending:
                    yield pending.popleft().result()
                    for item in islice(items, 1):
                        pending.append(pool.submit(function, item))
            finally:
                for future in pending:
                    future.cancel()

    def _get_country_players(self, country: str) -> list[str]:
        """Download one country's public player list.

        Args:
            country: Two-letter country code.

        Returns:
            Usernames returned by the endpoint.
        """
        response = self._request(
            f"country/{country}/players",
            self.client.get_country_players,
            country,
            tts=self.config.request_delay,
        )
        logger.info("Downloaded %d player names for %s", len(response.players), country)
        return response.players

    def get_players(self) -> list[str]:
        """Collect unique country usernames up to the configured discovery cap.

        Args:
            None.

        Returns:
            Case-insensitively deduplicated usernames in discovery order.
        """
        players: dict[str, str] = {}
        countries = dict.fromkeys(code.upper() for code in self.config.country_codes)
        for country_players in self._parallel_map(self._get_country_players, countries):
            for username in country_players:
                players.setdefault(username.casefold(), username)
        selected = list(players.values())[: self.config.player_limit]
        logger.info("Selected %d of %d discovered players", len(selected), len(players))
        return selected

    def _get_monthly_games(self, job: tuple[str, int, int]) -> tuple[str, list[Game]]:
        """Download a typed monthly archive for player-discovery callers.

        Args:
            job: Archive owner, year and month.

        Returns:
            Archive owner and typed games; raw metadata has a separate API.
        """
        username, year, month = job
        response = self._request(
            f"player/{username}/games/{year}/{month:02d}",
            self.client.get_player_games_by_month,
            username,
            year,
            month,
            tts=self.config.request_delay,
        )
        logger.info(
            "Downloaded %d games for %s/%04d/%02d",
            len(response.games),
            username,
            year,
            month,
        )
        return username, response.games

    def iter_games(self, username: str) -> Iterator[Game]:
        """Flatten the configured monthly archives for one player.

        Args:
            username: Archive owner.

        Returns:
            Iterator of typed games in configured archive order.
        """
        for _, games in self.iter_player_games([username]):
            yield from games

    def _get_raw_monthly_games(
        self, job: tuple[str, int, int]
    ) -> tuple[str, list[dict]]:
        """Retain API fields omitted by the SDK's typed Game, including rated.

        Args:
            job: Username, year, and month identifying one archive.

        Returns:
            Archive owner and raw JSON game objects.
        """
        username, year, month = job
        response = self._request(
            f"player/{username}/games/{year}/{month:02d}",
            self.client.get_player_games_by_month,
            username,
            year,
            month,
            tts=self.config.request_delay,
        )
        games = response.json.get("games", [])
        if not isinstance(games, list) or not all(
            isinstance(game, dict) for game in games
        ):
            raise ValueError(f"Invalid raw archive for {username}/{year}/{month}")
        logger.info(
            "Downloaded %d raw games for %s/%04d/%02d",
            len(games),
            username,
            year,
            month,
        )
        return username, games

    def iter_raw_player_games(
        self, usernames: Iterable[str]
    ) -> Generator[tuple[str, list[dict]], None, None]:
        """Download raw monthly archives with the existing retry/proxy limits.

        Args:
            usernames: Archive owners to query for configured months.

        Returns:
            Generator yielding owner and raw games in bounded request order.
        """
        jobs = (
            (username, year, month)
            for username in usernames
            for year, month in sorted(self.config.months, reverse=True)
        )
        yield from self._parallel_map(self._get_raw_monthly_games, jobs)

    def iter_player_games(
        self, usernames: Iterable[str]
    ) -> Generator[tuple[str, list[Game]], None, None]:
        """Download typed archives with bounded concurrency and stable ordering.

        Args:
            usernames: Player identities to query for configured months.

        Returns:
            Generator yielding owner and typed games for each archive.
        """
        jobs = (
            (username, year, month)
            for username in usernames
            for year, month in self.config.months
        )
        yield from self._parallel_map(self._get_monthly_games, jobs)
