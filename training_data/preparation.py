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
        if not isinstance(names, list) or not all(isinstance(name, str) for name in names):
            raise ValueError(f"{path} must contain a JSON list of strings")
        return list(dict.fromkeys(name.strip().casefold() for name in names if name.strip()))

    def build(self) -> list[str]:
        """Merge saved players with separately stratified time-category searches.

        Args:
            None.

        Returns:
            Seeded-shuffled usernames; completed identical searches reuse checkpoints.
        """
        settings = self.settings
        original = self.load(settings.seed_usernames_path)
        names = list(dict.fromkeys([*original, *self.load(settings.usernames_path)]))
        if settings.discovery_enabled:
            seeds = original.copy()
            random.Random(settings.training.seed).shuffle(seeds)
            seeds = list(dict.fromkeys([
                settings.collection.seed_username,
                *settings.discovery_seed_usernames,
                *seeds[:settings.discovery_seed_limit],
            ]))
            for category in dict.fromkeys(quota.category for quota in settings.stratification.quotas):
                if category == "classical":
                    continue
                config = replace(settings.collection, search_time_class=category)
                signature = fingerprint({"config": asdict(config), "seeds": seeds})
                checkpoint = settings.data_directory / f"discovery.{category}.json"
                saved = json.loads(checkpoint.read_text(encoding="utf-8")) if checkpoint.exists() else {}
                if saved.get("signature") == signature:
                    discovered = saved["usernames"]
                    print(f"Reusing {category} discovery: {len(discovered):,} usernames")
                else:
                    print(f"Discovering {category} players: up to {config.search_max_expansions:,} archive owners")
                    search = RatingStratifiedPlayerSearch(ChessComConnection(config))
                    discovered = search.search(seed_usernames=seeds)
                    write_json(checkpoint, {
                        "signature": signature,
                        "usernames": discovered,
                        "band_counts": search.band_counts,
                        "band_quotas": search.band_quotas,
                        "expanded_players": search.expanded_count,
                    })
                names = list(dict.fromkeys([*names, *(name.casefold() for name in discovered)]))
                write_json(settings.usernames_path, names)
        if not names:
            raise ValueError("Provide saved usernames or enable player discovery")
        random.Random(settings.training.seed).shuffle(names)
        return names