import contextlib
import io
import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from requests.exceptions import ConnectionError
from tqdm.std import tqdm

from elo_model.config import PipelineConfig
from elo_model.storage import write_json
from training_data import (
    ChessComConnection,
    DatasetConfig,
    RatingStratifiedPlayerSearch,
)
from training_data.discovery_progress import DiscoveryProgress
from training_data.preparation import PlayerPoolBuilder


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
    def test_notebook_progress_reuses_one_output_and_throttles_updates(self):
        """Keep one persistent notebook display without rapid stream redraws.

        Args:
            None.

        Returns:
            None; stable display identity, throttling and final content are checked.
        """
        with (
            patch(
                "training_data.discovery_progress.get_ipython",
                return_value=SimpleNamespace(kernel=object()),
            ),
            patch("training_data.discovery_progress.display") as render,
            patch(
                "training_data.discovery_progress.monotonic", return_value=0
            ) as clock,
        ):
            with DiscoveryProgress("blitz", {"100-349": 3}, 10, True) as progress:
                render.assert_called_once()
                self.assertTrue(render.call_args.kwargs["display_id"])
                handle = render.return_value
                clock.return_value = 0.1
                progress.update("Alice", {"100-349": 1})
                handle.update.assert_not_called()
                clock.return_value = 1.1
                progress.update("<Bob>", {"100-349": 2})
                handle.update.assert_called_once()
                content = handle.update.call_args.args[0].data
                self.assertIn("Remaining: 1", content)
                self.assertIn("&lt;Bob&gt;", content)
                self.assertNotIn("<Bob>", content)
                progress.finish("expansion budget reached")
            render.assert_called_once()
            content = handle.update.call_args.args[0].data
            self.assertIn("expansion budget reached", content)
            self.assertIn("Remaining: 1", content)
            self.assertIn("Archives processed: 2/10", content)

    def test_quota_progress_remaining_counts_and_uncertain_eta(self):
        """Keep rare-band uncertainty and partial-stop shortages visible.

        Args:
            None.

        Returns:
            None; remaining quotas, completion counts and estimated times are checked.
        """
        with (
            patch(
                "training_data.discovery_progress.monotonic", return_value=0
            ) as clock,
            DiscoveryProgress(
                "blitz", {"bridge": 0, "100-349": 3, "350-599": 3}, 10, False
            ) as display,
        ):
            self.assertEqual(len(display.bands), 2)
            self.assertIn("remaining=6", display.summary.postfix)
            self.assertIn("unknown (unseen bands)", display.summary.postfix)
            clock.return_value = 10
            display.update("Alice", {"100-349": 1, "350-599": 1})
            self.assertIn("remaining=4", display.summary.postfix)
            self.assertIn("bands_filled=0/2", display.summary.postfix)
            self.assertIn("quota_ETA=~00:20", display.summary.postfix)
            clock.return_value = 20
            display.update("Bob", {"100-349": 10, "350-599": 2})
            self.assertIn("remaining=1", display.summary.postfix)
            self.assertIn("bands_filled=1/2", display.summary.postfix)
            self.assertIn("quota_ETA=~00:10", display.summary.postfix)
            clock.return_value = 81
            display.update("Carol", {"100-349": 3, "350-599": 2})
            self.assertIn("unknown (quota gains stalled)", display.summary.postfix)
            display.finish("expansion budget reached")
            self.assertIn("remaining=1", display.summary.postfix)
            self.assertIn("quota_ETA=not reached", display.summary.postfix)
            self.assertIn("stopped=expansion budget reached", display.summary.postfix)

    def test_stagnation_budget_stops_unproductive_bridge_expansions(self):
        """Keep rare quota gaps visible without exhausting the full search budget.

        Args:
            None.

        Returns:
            None; productive discovery resets the bounded no-improvement budget.
        """
        search, _ = self.make_search(
            {
                "Seed": [game("Seed", 1700, "Bridge1", 1710)],
                "Bridge1": [game("Bridge1", 1710, "Bridge2", 1720)],
                "Bridge2": [game("Bridge2", 1720, "Bridge3", 1730)],
            },
            search_stagnation_expansions=2,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ):
            result = search.search()
        self.assertEqual(len(result), 1)
        self.assertEqual(search.expanded_count, 3)
        self.assertIn("stagnation budget", search.stop_reason)

    def test_progress_counts_archives_and_capped_rating_quotas(self):
        """Count monthly archives separately without overcounting accepted players.

        Args:
            None.

        Returns:
            None; progress totals, final counts and bar closure are asserted.
        """
        search, connection = self.make_search(
            {},
            search_max_expansions=1,
            months=((2026, 9), (2026, 10)),
        )
        connection.iter_player_games.side_effect = None
        connection.iter_player_games.return_value = [
            (
                "Seed",
                [game("Seed", 1700, f"Player{band}", band * 250) for band in range(6)],
            ),
            (
                "Seed",
                [
                    game("Seed", 1700, f"Player{band}", band * 250)
                    for band in range(6, 11)
                ],
            ),
        ]
        archive_bar = tqdm(total=2, file=io.StringIO())
        player_bar = tqdm(total=11, file=io.StringIO())
        band_bars = [tqdm(total=1, file=io.StringIO()) for _ in range(11)]
        with (
            patch(
                "training_data.discovery_progress.tqdm",
                side_effect=[player_bar, archive_bar, *band_bars],
            ) as bars,
            self.assertLogs(
                "training_data.rating_stratified_player_search", level="INFO"
            ),
        ):
            result = search.search(progress=True)
        self.assertEqual(len(result), 11)
        self.assertEqual(archive_bar.n, 2)
        self.assertEqual(player_bar.n, 11)
        self.assertEqual(
            [call.kwargs["total"] for call in bars.call_args_list], [11, 2, *([1] * 11)]
        )
        self.assertTrue(all(not call.kwargs["disable"] for call in bars.call_args_list))
        self.assertTrue(archive_bar.disable)
        self.assertTrue(player_bar.disable)
        self.assertTrue(all(bar.n == 1 and bar.disable for bar in band_bars))
        self.assertIn("remaining=0", player_bar.postfix)
        self.assertIn("bands_filled=11/11", player_bar.postfix)

    def test_player_pool_preserves_source_and_reuses_category_checkpoints(self):
        """Reuse searches while allowing new explicit seeds to extend discovery.

        Args:
            None.

        Returns:
            None; source preservation, category isolation and cache reuse are checked.
        """
        search = Mock()
        search.search.return_value = ["NewPlayer"]
        search.band_counts = {"100-349": 1}
        search.band_quotas = {"100-349": 300}
        search.expanded_count = 1
        search.stop_reason = "frontier exhausted"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.json"
            write_json(source, ["Seed", "Existing"])
            settings = replace(
                PipelineConfig.expanded(),
                seed_usernames_path=source,
                usernames_path=root / "pool.json",
                data_directory=root,
            )
            with (
                patch("training_data.preparation.ChessComConnection") as connection,
                patch(
                    "training_data.preparation.RatingStratifiedPlayerSearch",
                    return_value=search,
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                first = PlayerPoolBuilder(settings).build()
                self.assertEqual(set(first), {"seed", "existing", "newplayer"})
                self.assertEqual(PlayerPoolBuilder(settings).build(), first)
                self.assertEqual(search.search.call_count, 3)
                self.assertTrue(search.search.call_args.kwargs["progress"])
                self.assertEqual(
                    {
                        call.args[0].search_time_class
                        for call in connection.call_args_list
                    },
                    {"blitz", "rapid", "bullet"},
                )
                self.assertTrue(
                    all(
                        call.args[0].months == settings.discovery_months
                        for call in connection.call_args_list
                    )
                )
                self.assertEqual(len(settings.discovery_months), 1)
                self.assertEqual(len(settings.collection.months), 6)
                changed = replace(settings, discovery_seed_usernames=("Elite",))
                PlayerPoolBuilder(changed).build(progress=False)
                self.assertEqual(search.search.call_count, 6)
                self.assertFalse(search.search.call_args.kwargs["progress"])
                self.assertIn("Elite", search.search.call_args.kwargs["seed_usernames"])
            self.assertEqual(json.loads(source.read_text()), ["Seed", "Existing"])

    def test_multiple_seeds_and_zero_quota_bridge_bands(self):
        """Reach disconnected bands without retaining out-of-range bridges.

        Args:
            None.

        Returns:
            None; multiple seeds fill exact lower-inclusive target intervals.
        """
        search, _ = self.make_search(
            {
                "Seed": [game("Seed", 99, "Low", 100)],
                "Elite": [game("Elite", 3501, "High", 3500)],
            },
            search_band_edges=(0, 100, 350, 3350, 3501),
            search_band_quotas=(0, 1, 0, 1, 0),
            search_max_players=2,
        )
        with self.assertLogs(
            "training_data.rating_stratified_player_search", level="INFO"
        ):
            result = search.search(seed_usernames=["Elite", "seed"])
        self.assertEqual(set(result), {"Low", "High"})
        self.assertEqual(search.band_counts, search.band_quotas)

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
