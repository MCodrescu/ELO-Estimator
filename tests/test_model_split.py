"""Partition invariants include every target identity and underlying game."""

import unittest

import pandas as pd

from elo_model.config import RatingQuota, StratificationConfig
from elo_model.selection import StratifiedGameSelector
from elo_model.split import PlayerSplitter, assert_disjoint


class SplitTests(unittest.TestCase):
    def test_stratified_selection_keeps_one_game_for_two_requested_sides(self):
        """Keep both requested rating bands without duplicating their shared game.

        Args:
            None.

        Returns:
            None; both masks and a single underlying game are retained.
        """
        games = pd.DataFrame([{
            "game_id": "shared", "partition": "train", "category": "rapid",
            "white_username": "Alice", "black_username": "Bob",
            "white_rating": 1200, "black_rating": 1600,
            "keep_white": True, "keep_black": True,
        }])
        config = StratificationConfig(quotas=(
            RatingQuota(lower=1000, upper=1400, category="rapid", rows=1),
            RatingQuota(lower=1400, upper=1800, category="rapid", rows=1),
        ))
        selector = StratifiedGameSelector(config)
        selected = selector.select(games)
        pd.testing.assert_frame_equal(selected, games)
        self.assertEqual(selector.coverage["selected_rows"].tolist(), [1, 1])
        disabled = StratifiedGameSelector(StratificationConfig())
        pd.testing.assert_frame_equal(disabled.select(games), games)

    def test_stratified_selection_preserves_holdouts_and_side_eligibility(self):
        """Select only requested training sides and retain untouched hold-outs.

        Args:
            None.

        Returns:
            None; bounds, masks, category filtering and shortages are asserted.
        """
        games = pd.DataFrame([
            {
                "game_id": str(index), "partition": partition,
                "white_username": f"white{index}", "black_username": f"black{index}",
                "white_rating": white, "black_rating": black,
                "category": category, "keep_white": True, "keep_black": keep_black,
            }
            for index, (partition, white, black, category, keep_black) in enumerate([
                ("train", 1000, 1400, "blitz", True),
                ("train", 1399, 1100, "blitz", False),
                ("train", 1400, 1500, "blitz", True),
                ("train", 1200, 1200, "bullet", True),
                ("validation", 900, 1100, "rapid", True),
                ("test", 2200, 2300, "blitz", False),
            ])
        ])
        selector = StratifiedGameSelector(StratificationConfig(quotas=(
            RatingQuota(lower=1000, upper=1400, category="blitz", rows=3),
            RatingQuota(lower=2200, category="rapid", rows=2),
        )))
        selected = selector.select(games)
        self.assertEqual(selected["game_id"].tolist(), ["0", "1", "4", "5"])
        self.assertFalse(selected.iloc[0]["keep_black"])
        self.assertFalse(selected.iloc[1]["keep_black"])
        pd.testing.assert_frame_equal(
            selected.loc[selected["partition"] != "train"].reset_index(drop=True),
            games.loc[games["partition"] != "train"].reset_index(drop=True),
        )
        self.assertEqual(selector.coverage["shortfall"].tolist(), [1, 2])
        self.assertEqual(selector.coverage["selected_players"].tolist(), [2, 0])

    def test_stratified_selection_is_seeded_and_player_diverse(self):
        """Prefer distinct targets and enforce a cap across quota groups.

        Args:
            None.

        Returns:
            None; seeded ordering, player diversity and caps are asserted.
        """
        games = pd.DataFrame([
            {
                "game_id": str(index), "partition": "train",
                "white_username": f"player{index % 4}", "black_username": f"other{index}",
                "white_rating": 1200, "black_rating": 1200,
                "category": "blitz" if index < 12 else "rapid",
                "keep_white": True, "keep_black": False,
            }
            for index in range(24)
        ])
        config = StratificationConfig(max_rows_per_player=1, quotas=(
            RatingQuota(lower=1000, upper=1400, category="blitz", rows=4),
            RatingQuota(lower=1000, upper=1400, category="rapid", rows=4),
        ))
        selector = StratifiedGameSelector(config)
        selected = selector.select(games)
        self.assertEqual(len(selected), 4)
        self.assertEqual(selected["white_username"].nunique(), 4)
        self.assertEqual(selector.coverage["shortfall"].tolist(), [0, 4])
        repeated = selector.select(games.sample(frac=1, random_state=7))
        self.assertEqual(set(selected["game_id"]), set(repeated["game_id"]))
        changed = StratifiedGameSelector(config, seed=99).select(games)
        self.assertNotEqual(set(selected["game_id"]), set(changed["game_id"]))

    def test_target_and_game_disjointness(self):
        """Verify deterministic side masks and optional participant isolation.

        Args:
            None.

        Returns:
            None; game/target isolation and strict masks are asserted.
        """
        games = pd.DataFrame(
            [
                {
                    "game_id": str(index),
                    "white_username": f"p{index % 80}",
                    "black_username": f"p{(index * 7 + 3) % 80}",
                }
                for index in range(300)
            ]
        )
        splitter = PlayerSplitter((0.6, 0.2, 0.2))
        manifest = splitter.assign(games)
        rows = []
        for game in manifest.to_dict("records"):
            for color in ("white", "black"):
                if game[f"keep_{color}"]:
                    rows.append(
                        {
                            "game_id": game["game_id"],
                            "partition": game["partition"],
                            "target_username": game[f"{color}_username"],
                        }
                    )
        frame = pd.DataFrame(rows)
        assert_disjoint(frame)
        self.assertEqual(set(frame["partition"]), {"train", "validation", "test"})
        self.assertEqual(
            splitter.player_partition(" P1 "), splitter.player_partition("p1")
        )
        pd.testing.assert_frame_equal(manifest, splitter.assign(games))
        strict = PlayerSplitter((0.6, 0.2, 0.2), strict_participants=True).assign(games)
        self.assertTrue(strict["keep_white"].all() and strict["keep_black"].all())

    def test_detects_mirrored_game_leakage(self):
        """Reject opposite rows from one underlying game across partitions.

        Args:
            None.

        Returns:
            None; the leakage-specific ValueError is asserted.
        """
        frame = pd.DataFrame(
            {
                "game_id": ["same", "same"],
                "target_username": ["a", "b"],
                "partition": ["train", "test"],
            }
        )
        with self.assertRaisesRegex(ValueError, "game_id"):
            assert_disjoint(frame)


if __name__ == "__main__":
    unittest.main()
