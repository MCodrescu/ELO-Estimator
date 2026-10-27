import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from requests.exceptions import ConnectionError

from training_data import ChessComConnection, DatasetConfig, PlayerGraphSearch


def player_game(white: str | None, black: str | None) -> SimpleNamespace:
    """Build a lightweight game containing only discovery metadata.

    Args:
        white: White username, or None to exercise missing metadata.
        black: Black username, or None to exercise missing metadata.

    Returns:
        Mock-compatible game with participant links and no PGN.
    """
    return SimpleNamespace(
        url="test-game",
        white=SimpleNamespace(username=white) if white is not None else None,
        black=SimpleNamespace(username=black) if black is not None else None,
        pgn=None,
        rules="chess960",
    )


class PlayerGraphSearchTests(unittest.TestCase):
    def make_connection(self, graph: dict[str, list[str]], **settings):
        """Build a connection whose archive endpoint traverses a fixture graph.

        Args:
            graph: Username-to-opponent adjacency mapping.
            settings: DatasetConfig overrides for this fixture.

        Returns:
            Configured connection and patched archive endpoint mock.
        """
        config = DatasetConfig(seed_username="Seed", request_delay=0, **settings)
        connection = ChessComConnection(config)
        graph = {key.casefold(): opponents for key, opponents in graph.items()}

        def download(username, year, month, tts):
            """Return fixture opponents while checking the archive request.

            Args:
                username: Requested archive owner.
                year: Configured archive year.
                month: Configured archive month.
                tts: SDK request delay, expected to be zero in tests.

            Returns:
                Typed-response-shaped fixture containing opponent games.
            """
            self.assertIn((year, month), config.months)
            self.assertEqual(tts, 0)
            return SimpleNamespace(
                games=[
                    player_game(username, opponent)
                    for opponent in graph.get(username.casefold(), [])
                ]
            )

        patcher = patch.object(
            connection.client, "get_player_games_by_month", side_effect=download
        )
        endpoint = patcher.start()
        self.addCleanup(patcher.stop)
        return connection, endpoint

    def test_bfs_cycles_multiple_months_and_case_insensitive_duplicates(self):
        """Preserve breadth-first order across cycles and duplicate identities.

        Args:
            None.

        Returns:
            None; ordering, request counts and completion logs are asserted.
        """
        connection, endpoint = self.make_connection(
            {
                "Seed": ["B", "C", "b", "SEED"],
                "B": ["Seed", "D"],
                "C": ["seed", "E", "d"],
                "D": ["B"],
                "E": ["C"],
            },
            months=((2026, 9), (2026, 10)),
            max_workers=2,
        )
        with (
            patch.object(
                connection, "iter_player_games", wraps=connection.iter_player_games
            ) as batches,
            patch.object(connection, "get_players") as country_discovery,
            self.assertLogs("training_data.player_graph_search", level="INFO") as logs,
        ):
            result = PlayerGraphSearch(connection).search()
        self.assertEqual(result, ["Seed", "B", "C", "D", "E"])
        self.assertEqual(
            [call.args[0] for call in batches.call_args_list],
            [["Seed"], ["B", "C"], ["D", "E"]],
        )
        self.assertEqual(endpoint.call_count, 10)
        self.assertEqual(
            len({tuple(call.args) for call in endpoint.call_args_list}), 10
        )
        country_discovery.assert_not_called()
        self.assertIn("reachable opponent links exhausted", logs.output[-1])

    def test_search_cap_includes_seed_and_closes_pending_iterator(self):
        """Stop and close the archive iterator as soon as the cap is met.

        Args:
            None.

        Returns:
            None; retained usernames and iterator cleanup are asserted.
        """
        connection = Mock(spec=ChessComConnection)
        connection.config = DatasetConfig(seed_username="Seed", search_max_players=3)
        closed = []

        def batches(usernames):
            """Expose excess links and detect consumption after the cap.

            Args:
                usernames: Expected seed frontier.

            Returns:
                Generator of one mocked archive batch.
            """
            self.assertEqual(usernames, ["Seed"])
            try:
                yield (
                    "Seed",
                    [player_game("Seed", name) for name in ("B", "C", "D", "E")],
                )
                self.fail("Search consumed a batch after reaching the cap")
            finally:
                closed.append(True)

        connection.iter_player_games.side_effect = batches
        self.assertEqual(PlayerGraphSearch(connection).search(), ["Seed", "B", "C"])
        self.assertEqual(closed, [True])
        connection.iter_player_games.assert_called_once_with(["Seed"])

    def test_depth_limits_and_single_player_cap_avoid_unnecessary_requests(self):
        """Avoid archive work when caps prevent further expansion.

        Args:
            None.

        Returns:
            None; seed-only and one-hop request behavior are asserted.
        """
        for settings in ({"search_max_players": 1}, {"search_max_depth": 0}):
            with self.subTest(settings=settings):
                connection, endpoint = self.make_connection({"Seed": ["B"]}, **settings)
                self.assertEqual(PlayerGraphSearch(connection).search(), ["Seed"])
                endpoint.assert_not_called()

        connection, endpoint = self.make_connection(
            {"Seed": ["B", "C"], "B": ["D"], "C": ["E"]},
            search_max_depth=1,
        )
        self.assertEqual(PlayerGraphSearch(connection).search(), ["Seed", "B", "C"])
        endpoint.assert_called_once_with("Seed", 2026, 10, tts=0)

    def test_empty_archives_return_seed_and_repeated_searches_reset_state(self):
        """Keep empty searches independent and retain their seed.

        Args:
            None.

        Returns:
            None; repeat results and request counts are asserted.
        """
        connection, endpoint = self.make_connection({})
        search = PlayerGraphSearch(connection)
        self.assertEqual(search.search(), ["Seed"])
        self.assertEqual(search.search(), ["Seed"])
        self.assertEqual(endpoint.call_count, 2)

    def test_seed_override_and_bad_seed_validation(self):
        """Use a supplied seed and reject an empty normalized identity.

        Args:
            None.

        Returns:
            None; override requests and validation are asserted.
        """
        connection, endpoint = self.make_connection(
            {"Other": ["B"]}, search_max_depth=1
        )
        self.assertEqual(
            PlayerGraphSearch(connection).search(" Other "), ["Other", "B"]
        )
        endpoint.assert_called_once_with("Other", 2026, 10, tts=0)
        with self.assertRaises(ValueError):
            PlayerGraphSearch(connection).search(" ")

    def test_missing_or_unrelated_metadata_is_logged_and_skipped(self):
        """Ignore malformed links without losing valid opponents.

        Args:
            None.

        Returns:
            None; retained identities and warning context are asserted.
        """
        connection = Mock(spec=ChessComConnection)
        connection.config = DatasetConfig(seed_username="Seed", search_max_depth=1)

        def batches(usernames):
            """Yield mixed malformed and valid participant metadata.

            Args:
                usernames: Requested seed frontier.

            Returns:
                Generator of fixture archive results.
            """
            yield (
                usernames[0],
                [
                    player_game(None, "B"),
                    player_game("Seed", " "),
                    player_game("Other", "B"),
                    player_game("SEED", "Valid"),
                ],
            )

        connection.iter_player_games.side_effect = batches
        with self.assertLogs(
            "training_data.player_graph_search", level="WARNING"
        ) as logs:
            self.assertEqual(PlayerGraphSearch(connection).search(), ["Seed", "Valid"])
        self.assertEqual(len(logs.output), 3)
        self.assertTrue(all("test-game" in line for line in logs.output))

    def test_api_failures_propagate_and_close_search(self):
        """Never turn API failures into successful partial graph searches.

        Args:
            None.

        Returns:
            None; original failure and generator cleanup are asserted.
        """
        connection = Mock(spec=ChessComConnection)
        connection.config = DatasetConfig()
        closed = []

        def batches(usernames):
            """Raise a transport error while recording generator cleanup.

            Args:
                usernames: Requested archive owner list.

            Returns:
                Generator that raises ConnectionError before yielding data.
            """
            try:
                yield from ()
                raise ConnectionError(f"Unable to fetch {usernames[0]}")
            finally:
                closed.append(True)

        connection.iter_player_games.side_effect = batches
        with self.assertRaisesRegex(ConnectionError, "Unable to fetch MCodr"):
            PlayerGraphSearch(connection).search()
        self.assertEqual(closed, [True])

    def test_invalid_search_configuration(self):
        """Reject invalid graph-search settings before requests.

        Args:
            None.

        Returns:
            None; each invalid configuration raises ValueError.
        """
        for settings in (
            {"seed_username": " "},
            {"search_max_players": 0},
            {"search_max_depth": -1},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                DatasetConfig(**settings)


if __name__ == "__main__":
    unittest.main()
