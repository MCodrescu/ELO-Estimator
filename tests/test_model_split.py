"""Partition invariants include every target identity and underlying game."""

import unittest

import pandas as pd

from elo_model.split import PlayerSplitter, assert_disjoint


class SplitTests(unittest.TestCase):
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
