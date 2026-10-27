"""Seeded player-side training quotas applied after player-aware splitting."""

from collections import Counter

import pandas as pd

from .config import DEFAULT_SEED, StratificationConfig
from .split import assert_disjoint, hash_fraction, normalize_player


class StratifiedGameSelector:
    """Narrow training-side masks while preserving natural hold-out coverage."""

    def __init__(self, config: StratificationConfig, seed: int = DEFAULT_SEED):
        """Prepare reproducible selection without inspecting or analyzing games.

        Args:
            config: Game-time rating/category quotas and per-target-player cap.
            seed: Seed for stable game-side sampling priorities.

        Returns:
            None; coverage is populated by select.
        """
        self.config = config
        self.seed = seed
        self.coverage = pd.DataFrame()

    def _side_rows(self, games: pd.DataFrame) -> pd.DataFrame:
        """Expand eligible sides with stable sampling priorities.

        Args:
            games: Split manifest containing both players and eligibility masks.

        Returns:
            Player-side metadata with game identities and seeded priorities.
        """
        rows = []
        for record in games.to_dict("records"):
            for color in ("white", "black"):
                if record[f"keep_{color}"]:
                    rows.append(
                        {
                            "game_id": record["game_id"],
                            "partition": record["partition"],
                            "target_username": normalize_player(record[f"{color}_username"]),
                            "rating": record[f"{color}_rating"],
                            "category": record["category"],
                            "color": color,
                            "priority": hash_fraction(
                                f"selection:{record['game_id']}:{color}", self.seed
                            ),
                        }
                    )
        return pd.DataFrame(
            rows, columns=(
                "game_id", "partition", "target_username", "rating", "category",
                "color", "priority",
            ),
        )

    def select(self, games: pd.DataFrame) -> pd.DataFrame:
        """Choose diverse training sides; never enable a previously excluded side.

        Args:
            games: Unique split games with partition labels and eligible side masks.

        Returns:
            Retained manifest in input order; validation/test rows are unchanged.
        """
        if games["game_id"].duplicated().any():
            raise ValueError("Duplicate games must be removed before selection")
        sides = self._side_rows(games)
        assert_disjoint(sides)
        self.coverage = pd.DataFrame()
        if not self.config.quotas:
            return games.copy()

        train = sides.loc[sides["partition"] == "train"]
        selected = {"white": set(), "black": set()}
        player_counts = Counter()
        reports = []
        for quota in self.config.quotas:
            eligible = train.loc[
                (train["category"] == quota.category)
                & (train["rating"] >= quota.lower)
            ]
            if quota.upper is not None:
                eligible = eligible.loc[eligible["rating"] < quota.upper]
            ranked = eligible.sort_values(["priority", "game_id", "color"]).copy()
            ranked["player_round"] = ranked.groupby("target_username").cumcount()
            ranked = ranked.sort_values(["player_round", "priority", "game_id", "color"])
            selected_players = set()
            count = 0
            for side in ranked.to_dict("records"):
                player = side["target_username"]
                if player_counts[player] >= self.config.max_rows_per_player:
                    continue
                selected[side["color"]].add(side["game_id"])
                player_counts[player] += 1
                selected_players.add(player)
                count += 1
                if count == quota.rows:
                    break
            reports.append(
                {
                    "rating_band": quota.label,
                    "category": quota.category,
                    "requested_rows": quota.rows,
                    "available_rows": len(eligible),
                    "available_players": eligible["target_username"].nunique(),
                    "selected_rows": count,
                    "selected_players": len(selected_players),
                    "shortfall": quota.rows - count,
                }
            )
        result = games.copy()
        training = result["partition"] == "train"
        for color in ("white", "black"):
            result.loc[training, f"keep_{color}"] = (
                result.loc[training, f"keep_{color}"]
                & result.loc[training, "game_id"].isin(selected[color])
            )
        self.coverage = pd.DataFrame(reports)
        return result.loc[result["keep_white"] | result["keep_black"]].reset_index(drop=True)