import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from requests.exceptions import ConnectionError

from training_data import (
    ChessComConnection,
    DatasetConfig,
    RatingStratifiedPlayerSearch,
)


def game(
    white: str,
    white_rating: int | None,
    black: str,
    black_rating: int | None,
    timestamp: int | None = 1,
    time_class: str = "blitz",
    rules: str = "chess",
) -> SimpleNamespace:
    """Build archive metadata without requiring PGN parsing.

    Args:
        white: White username.
        white_rating: White game-time rating, or None.
        black: Black username.
        black_rating: Black game-time rating, or None.
        timestamp: Completion time, optionally missing for validation tests.
        time_class: API time-control category.
        rules: API chess variant.

    Returns:
        SDK-shaped game fixture for discovery tests.
    """
    return SimpleNamespace(
        white=SimpleNamespace(username=white, rating=white_rating),
        black=SimpleNamespace(username=black, rating=black_rating),
        end_time=timestamp,
        time_class=time_class,
        rules=rules,
        url=f"{white}-{black}-{timestamp}",
        pgn=None,
    )


class RatingStratifiedPlayerSearchTests(unittest.TestCase):
    def make_search(self, graph, **settings):
        """Build a bounded search over deterministic archive fixtures.

        Args:
            graph: Username-to-game-list mapping.
            settings: DatasetConfig overrides for the fixture.

        Returns:
            Rating-stratified search and its mocked connection.
        """
        connection = Mock(spec=ChessComConnection)
        connection.config = DatasetConfig(
            **{
                "seed_username": "Seed",
                "search_max_players": 11,
                "max_workers": 1,
                **settings,
            }
        )
        graph = {name.casefold(): games for name, games in graph.items()}

        def batches(usernames):
            """Yield configured archive games in requested player order.

            Args:
                usernames: Archive owners scheduled by the search.

            Returns:
                Generator of owner and game-list pairs.
            """
            for username in usernames:
                yield username, graph.get(username.casefold(), [])

        connection.iter_player_games.side_effect = batches
        return RatingStratifiedPlayerSearch(connection), connection

    def test_exact_250_point_boundaries_and_2500_plus(self):
        """Use exact lower-inclusive rating-band boundaries.

        Args:
            None.

        Returns:
            None; band indexes and open-ended top quota are asserted.
        """
        search, _ = self.make_search({})
        for rating, band in (
            (0, 0),
            (249, 0),
            (250, 1),
            (499, 1),
            (500, 2),
            (2249, 8),
            (2250, 9),
            (2499, 9),
            (2500, 10),
            (3500, 10),
        ):
            with self.subTest(rating=rating):
                self.assertEqual(search._band(rating), band)
        self.assertEqual(len(search.band_quotas), 11)
        self.assertIn("2500+", search.band_quotas)
        self.assertEqual(sum(search.band_quotas.values()), 11)

    def test_quotas_divide_username_cap_without_overflow(self):
        """Distribute every requested slot without exceeding the player cap.

        Args:
            None.

        Returns:
            None; quota totals and at-most-one differences are asserted.
        """
        for cap in (1, 5, 11, 100, 100000):
            search, _ = self.make_search({}, search_max_players=cap)
            quotas = list(search.band_quotas.values())
            self.assertEqual(sum(quotas), cap)
            self.assertLessEqual(max(quotas) - min(quotas), 1)
        search, _ = self.make_search({}, search_max_players=100)
        self.assertEqual(list(search.band_quotas.values()), [10] + [9] * 10)

    def test_all_bands_fill_from_seed_metadata_without_pgns(self):
        """Fill quotas from archive metadata without fetching PGN data.

        Args:
            None.

        Returns:
            None; coverage, discovery counts and requests are asserted.
        """
        search, connection = self.make_search(
            {
                "Seed": [
                    game("Seed", 1700, f"Player{band}", band * 250)
                    for band in range(11)
                ],
            }
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ) as logs:
            result = search.search()
        self.assertEqual(len(result), 11)
        self.assertEqual(len({name.casefold() for name in result}), 11)
        self.assertEqual(search.band_counts, search.band_quotas)
        self.assertEqual(search.expanded_count, 1)
        self.assertIn("Seed", result)
        self.assertIn("Player10", result)
        connection.iter_player_games.assert_called_once_with(["Seed"])
        connection.get_players.assert_not_called()
        self.assertIn("band quotas filled", logs.output[-1])

    def test_full_band_bridge_is_expanded_but_not_overaccepted(self):
        """Retain over-quota bridge players for expansion but not output.

        Args:
            None.

        Returns:
            None; bridge expansion and final quota compliance are asserted.
        """
        search, connection = self.make_search(
            {
                "Seed": [game("Seed", 1700, "Bridge", 1710)],
                "Bridge": [
                    game("Bridge", 1710, f"Player{band}", band * 250)
                    for band in range(11)
                ],
            }
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ):
            result = search.search()
        self.assertEqual(len(result), 11)
        self.assertNotIn("Bridge", result)
        self.assertEqual(
            [call.args[0] for call in connection.iter_player_games.call_args_list],
            [["Seed"], ["Bridge"]],
        )
        self.assertEqual(search.band_counts, search.band_quotas)

    def test_underfilled_band_has_priority_over_full_band_candidates(self):
        """Prioritize missing coverage over already-full rating bands.

        Args:
            None.

        Returns:
            None; archive expansion order is asserted.
        """
        search, connection = self.make_search(
            {
                "Seed": [
                    game("Seed", 1700, "Bridge", 1710),
                    game("Seed", 1700, "Low", 300),
                ]
            },
            search_max_players=22,
            search_max_expansions=2,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ):
            search.search()
        self.assertEqual(
            [call.args[0] for call in connection.iter_player_games.call_args_list],
            [["Seed"], ["Low"]],
        )

    def test_budget_is_player_count_and_returns_logged_partial_sample(self):
        """Respect expansion budgets without disguising partial coverage.

        Args:
            None.

        Returns:
            None; expansion totals, partial results and warnings are asserted.
        """
        search, connection = self.make_search(
            {
                "Seed": [
                    game("Seed", 1700, f"Bridge{index}", 1701 + index)
                    for index in range(10)
                ]
            },
            search_max_expansions=2,
            max_workers=8,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ) as logs:
            result = search.search()
        self.assertEqual(search.expanded_count, 2)
        self.assertEqual(len(result), 1)
        self.assertTrue(
            any("Unfilled rating band 0-249" in line for line in logs.output)
        )
        self.assertIn("expansion budget reached", logs.output[-1])
        self.assertEqual(
            [len(call.args[0]) for call in connection.iter_player_games.call_args_list],
            [1, 1],
        )

    def test_latest_end_time_wins_independently_of_archive_game_order(self):
        """Use the newest observed rating even when archives arrive out of order.

        Args:
            None.

        Returns:
            None; the final rating-band assignment is asserted.
        """
        for observations in (
            [game("Seed", 1700, "A", 800, 20), game("Seed", 1700, "A", 100, 10)],
            [game("Seed", 1700, "A", 100, 10), game("Seed", 1700, "A", 800, 20)],
        ):
            search, _ = self.make_search(
                {"Seed": observations}, search_max_expansions=1
            )
            with self.assertLogs(
                "training_data.rating_stratified_player_search", level="INFO"
            ):
                result = search.search()
            self.assertIn("A", result)
            self.assertEqual(search.band_counts["750-999"], 1)
            self.assertEqual(search.band_counts["0-249"], 0)

    def test_newer_observation_reclassifies_player_without_double_counting(self):
        """Move a player between bands instead of counting both observations.

        Args:
            None.

        Returns:
            None; identity uniqueness and corrected band counts are asserted.
        """
        search, _ = self.make_search(
            {
                "Seed": [game("Seed", 1700, "A", 100, 10)],
                "A": [game("A", 800, "Seed", 1700, 20)],
            },
            search_max_expansions=2,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ):
            result = search.search()
        self.assertEqual(result.count("A"), 1)
        self.assertEqual(search.band_counts["0-249"], 0)
        self.assertEqual(search.band_counts["750-999"], 1)

    def test_same_timestamp_conflict_keeps_first_and_logs(self):
        """Keep deterministic results when equal-time ratings disagree.

        Args:
            None.

        Returns:
            None; retained band and conflict warnings are asserted.
        """
        search, _ = self.make_search(
            {
                "Seed": [game("Seed", 1700, "A", 100), game("Seed", 1700, "A", 800)],
            },
            search_max_expansions=1,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ) as logs:
            search.search()
        self.assertEqual(search.band_counts["0-249"], 1)
        self.assertTrue(
            any("Conflicting same-time ratings for A" in line for line in logs.output)
        )

    def test_time_class_variants_and_bad_metadata_are_not_classified(self):
        """Exclude unsupported games and invalid rating/link observations.

        Args:
            None.

        Returns:
            None; valid coverage and diagnostic warnings are asserted.
        """
        unrelated = game("Other", 100, "Foreign", 200)
        no_name = game("Seed", 1700, " ", 100)
        search, _ = self.make_search(
            {
                "Seed": [
                    game("Seed", 1700, "Rapid", 100, time_class="rapid"),
                    game("Seed", 1700, "Bullet", 100, time_class="bullet"),
                    game("Seed", 1700, "Variant", 100, rules="chess960"),
                    no_name,
                    unrelated,
                    game("Seed", 1700, "NoTime", 100, timestamp=None),
                    game("Seed", 1700, "NoRating", None),
                    game("Seed", 1700, "Negative", -1),
                    game("Seed", 1700, "Valid", 600),
                ],
            },
            search_max_expansions=1,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ) as logs:
            result = search.search()
        self.assertEqual(set(result), {"Seed", "Valid"})
        self.assertTrue(any("missing/invalid end_time" in line for line in logs.output))
        self.assertTrue(
            any("missing participant username" in line for line in logs.output)
        )

    def test_unknown_rating_player_can_be_bridge_then_classified(self):
        """Allow an unrated bridge to reveal useful later rating observations.

        Args:
            None.

        Returns:
            None; bridge expansion and eventual classification are asserted.
        """
        search, connection = self.make_search(
            {
                "Seed": [game("Seed", 1700, "Unknown", None)],
                "Unknown": [game("Unknown", 100, "Low", 600, 2)],
            },
            search_max_expansions=2,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ):
            result = search.search()
        self.assertEqual(set(result), {"Seed", "Unknown", "Low"})
        self.assertEqual(
            connection.iter_player_games.call_args_list[1].args[0], ["Unknown"]
        )

    def test_cycles_case_dedup_and_repeated_calls_are_deterministic(self):
        """Reset mutable state and normalize cyclic participant identities.

        Args:
            None.

        Returns:
            None; repeated results and band counts are identical.
        """
        search, _ = self.make_search(
            {
                "Seed": [game("Seed", 1700, "B", 1750), game("SEED", 1700, "b", 1750)],
                "B": [game("b", 1750, "seed", 1700)],
            },
            search_max_players=100,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ):
            first = search.search()
            first_counts = search.band_counts
            second = search.search()
        self.assertEqual(first, second)
        self.assertEqual(first_counts, search.band_counts)
        self.assertEqual(first, ["Seed", "B"])
        self.assertEqual(search.expanded_count, 2)

    def test_depth_zero_cannot_classify_seed_and_makes_no_requests(self):
        """Avoid expansion and classification when the depth limit is zero.

        Args:
            None.

        Returns:
            None; empty output and absence of requests are asserted.
        """
        search, connection = self.make_search({}, search_max_depth=0)
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ):
            self.assertEqual(search.search(), [])
        connection.iter_player_games.assert_not_called()

    def test_api_failure_is_not_disguised_as_partial_success(self):
        """Propagate archive failures rather than returning misleading coverage.

        Args:
            None.

        Returns:
            None; the original API exception is asserted.
        """
        search, connection = self.make_search({})
        connection.iter_player_games.side_effect = ConnectionError("Unavailable")
        with self.assertRaisesRegex(ConnectionError, "Unavailable"):
            search.search()

    def test_invalid_stratified_configuration(self):
        """Reject invalid quotas, rating edges and discovery limits.

        Args:
            None.

        Returns:
            None; each invalid configuration raises ValueError.
        """
        for settings in (
            {"search_max_expansions": 0},
            {"search_max_expansions": True},
            {"search_time_class": "daily"},
            {"search_band_edges": ()},
            {"search_band_edges": (250, 500)},
            {"search_band_edges": (0, 250, 250)},
            {"search_band_edges": (0, -250, 500)},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                DatasetConfig(**settings)


if __name__ == "__main__":
    unittest.main()
