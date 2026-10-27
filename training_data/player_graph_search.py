import logging
from contextlib import closing

from .chess_com_connection import ChessComConnection

logger = logging.getLogger(__name__)


class PlayerGraphSearch:
    """Breadth-first opponent discovery across all countries in configured months.

    The seed has depth zero and counts toward search_max_players. None means no
    depth limit, not an unlimited crawl: the username cap always applies. Results
    include discovered players even if their archives have not been expanded.
    """

    def __init__(self, connection: ChessComConnection) -> None:
        """Use a configured connection for bounded opponent discovery.

        Args:
            connection: Shared archive client and search settings.

        Returns:
            None; no requests are sent during initialization.
        """
        self.connection = connection

    def search(self, seed_username: str | None = None) -> list[str]:
        """Discover opponents breadth-first within configured limits.

        Args:
            seed_username: Starting player, or the configured seed when omitted.

        Returns:
            Deduplicated usernames in discovery order, including the seed.
        """
        config = self.connection.config
        seed = (
            config.seed_username if seed_username is None else seed_username
        ).strip()

        if not seed:
            raise ValueError("seed_username must not be empty")

        discovered = {seed.casefold(): seed}
        expanded: set[str] = set()
        frontier = [seed]
        depth = 0
        logger.info(
            "Starting BFS from %s: limit %d players, max depth %s, months %s",
            seed,
            config.search_max_players,
            config.search_max_depth,
            config.months,
        )

        while (
            frontier
            and len(discovered) < config.search_max_players
            and (config.search_max_depth is None or depth < config.search_max_depth)
        ):
            logger.info(
                "BFS depth %d: expanding %d players; %d discovered",
                depth,
                len(frontier),
                len(discovered),
            )
            next_frontier: list[str] = []
            with closing(self.connection.iter_player_games(frontier)) as batches:
                for username, games in batches:
                    expanded.add(username.casefold())
                    added = 0
                    for game in games:
                        white = game.white.username if game.white is not None else None
                        black = game.black.username if game.black is not None else None
                        if (
                            not white
                            or not black
                            or not white.strip()
                            or not black.strip()
                        ):
                            logger.warning(
                                "Skipping player link in %s: missing participant username",
                                game.url,
                            )
                            continue
                        participants = (white.strip(), black.strip())
                        if username.casefold() not in {
                            name.casefold() for name in participants
                        }:
                            logger.warning(
                                "Skipping player link in %s: archive owner %s is not a participant",
                                game.url,
                                username,
                            )
                            continue
                        for name in participants:
                            key = name.casefold()
                            if key not in discovered:
                                discovered[key] = name
                                next_frontier.append(name)
                                added += 1
                                if len(discovered) == config.search_max_players:
                                    break
                        if len(discovered) == config.search_max_players:
                            break
                    logger.info(
                        "BFS processed %s at depth %d: %d new players; %d/%d discovered",
                        username,
                        depth,
                        added,
                        len(discovered),
                        config.search_max_players,
                    )
                    if len(discovered) == config.search_max_players:
                        break
            frontier = next_frontier
            depth += 1

        if len(discovered) == config.search_max_players:
            reason = "username limit reached"
        elif frontier:
            reason = "depth limit reached"
        else:
            reason = "reachable opponent links exhausted"
        logger.info(
            "BFS complete: %d usernames, %d players expanded; %s",
            len(discovered),
            len(expanded),
            reason,
        )
        return list(discovered.values())
