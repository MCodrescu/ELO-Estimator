"""Prepare a persistent, multi-category username pool before game collection."""

import json
import random
from dataclasses import asdict, replace
from pathlib import Path

from elo_model.analysis import fingerprint
from elo_model.config import PipelineConfig
from elo_model.storage import write_json

from .chess_com_connection import ChessComConnection
from .rating_stratified_player_search import RatingStratifiedPlayerSearch


class PlayerPoolBuilder:
    """Keep existing usernames and checkpoint completed category searches."""

    def __init__(self, settings: PipelineConfig) -> None:
        """Store discovery settings without issuing requests.

        Args:
            settings: Collection, discovery and persistent username paths.

        Returns:
            None.
        """
        self.settings = settings

    @staticmethod
    def load(path: Path) -> list[str]:
        """Read and normalize an optional saved username list.

        Args:
            path: JSON list location; a missing file is treated as empty.

        Returns:
            Unique nonempty case-folded usernames in file order.
        """
        if not path.exists():
            return []

        names = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(names, list) or not all(
            isinstance(name, str) for name in names
        ):
            raise ValueError(f"{path} must contain a JSON list of strings")

        return list(
            dict.fromkeys(name.strip().casefold() for name in names if name.strip())
        )

    def _discover_category(
        self, category: str, seeds: list[str], progress: bool
    ) -> list[str]:
        """Reuse a matching discovery checkpoint or run one category search.

        Args:
            category: Time category whose rating quotas should be filled.
            seeds: Starting usernames shared across category searches.
            progress: Whether to display discovery progress.

        Returns:
            Discovered usernames in the search's stable rating-band order.
        """
        config = replace(
            self.settings.collection,
            search_time_class=category,
            months=self.settings.discovery_months,
        )
        search_settings = asdict(config)

        # Cache location and freshness do not change discovery's sampling contract.
        search_settings.pop("archive_cache_directory")
        search_settings.pop("archive_cache_ttl_seconds")
        signature = fingerprint({"config": search_settings, "seeds": seeds})
        checkpoint = self.settings.data_directory / f"discovery.{category}.json"
        saved = (
            json.loads(checkpoint.read_text(encoding="utf-8"))
            if checkpoint.exists()
            else {}
        )

        if saved.get("signature") == signature:
            discovered = saved["usernames"]
            print(f"Reusing {category} discovery: {len(discovered):,} usernames")
            return discovered

        print(
            f"Discovering {category} players: up to {config.search_max_expansions:,} archive owners"
        )
        search = RatingStratifiedPlayerSearch(ChessComConnection(config))
        discovered = search.search(seed_usernames=seeds, progress=progress)
        write_json(
            checkpoint,
            {
                "signature": signature,
                "usernames": discovered,
                "band_counts": search.band_counts,
                "band_quotas": search.band_quotas,
                "expanded_players": search.expanded_count,
                "months": [list(month) for month in config.months],
                "stop_reason": search.stop_reason,
            },
        )

        return discovered

    def build(self, progress: bool | None = None) -> list[str]:
        """Merge saved players with separately stratified time-category searches.

        Args:
            progress: Show discovery bars; None uses the training progress setting.

        Returns:
            Seeded-shuffled usernames; completed identical searches reuse checkpoints.
        """
        settings = self.settings
        show_progress = (
            settings.neural_training.progress if progress is None else progress
        )
        original = self.load(settings.seed_usernames_path)
        names = list(dict.fromkeys([*original, *self.load(settings.usernames_path)]))

        if settings.discovery_enabled:
            seeds = original.copy()
            random.Random(settings.neural_training.seed).shuffle(seeds)
            seeds = list(
                dict.fromkeys(
                    [
                        settings.collection.seed_username,
                        *settings.discovery_seed_usernames,
                        *seeds[: settings.discovery_seed_limit],
                    ]
                )
            )

            for category in dict.fromkeys(
                quota.category for quota in settings.stratification.quotas
            ):
                if category == "classical":
                    continue

                discovered = self._discover_category(category, seeds, show_progress)
                names = list(
                    dict.fromkeys([*names, *(name.casefold() for name in discovered)])
                )
                write_json(settings.usernames_path, names)

        if not names:
            raise ValueError("Provide saved usernames or enable player discovery")

        random.Random(settings.neural_training.seed).shuffle(names)
        return names
