"""Versioned feature ordering; identity and rating metadata never enter X."""

import hashlib
import json

from .config import FEATURE_SCHEMA_VERSION

SCHEMA_VERSION = FEATURE_SCHEMA_VERSION

ACCURACY = (
    "loss_mean",
    "loss_median",
    "loss_std",
    "loss_p90",
    "cp_loss_mean",
    "inaccuracy_rate",
    "mistake_rate",
    "blunder_rate",
    "top1_rate",
    "top3_rate",
)
PHASE = tuple(
    f"{phase}_{metric}"
    for phase in ("opening", "middlegame", "endgame")
    for metric in ("loss_mean", "blunder_rate", "top1_rate")
)
POSITION = (
    "winning_loss_mean",
    "equal_loss_mean",
    "losing_loss_mean",
    "winning_retention",
    "losing_resilience",
)
COMPLEXITY = (
    "only_move_rate",
    "only_move_success",
    "complexity_low_loss",
    "complexity_medium_loss",
    "complexity_high_loss",
    "missed_tactic_rate",
)
PATTERNS = ("max_blunder_streak", "blunder_clusters")
CLOCK = (
    "think_mean",
    "think_median",
    "think_total",
    "clock_budget_used",
    "time_trouble_share",
    "time_trouble_blunder_rate",
    "instant_rate",
    "think_complexity_correlation",
    "flagged",
    "clock_coverage",
)
SIDE_GROUPS = {
    "accuracy": ACCURACY,
    "phase": PHASE,
    "position": POSITION,
    "complexity": COMPLEXITY,
    "patterns": PATTERNS,
    "clock": CLOCK,
}
FEATURE_GROUPS = {
    "time_control": (
        "base_seconds",
        "increment_seconds",
        "bullet",
        "blitz",
        "rapid",
        "classical",
    ),
    **{
        f"self_{group}": tuple(f"self_{name}" for name in names)
        for group, names in SIDE_GROUPS.items()
    },
    "context": ("game_plies", "is_white", "result")
    + tuple(
        f"termination_{name}"
        for name in ("checkmate", "resignation", "timeout", "draw", "other", "unknown")
    )
    + tuple(f"eco_{name}" for name in ("A", "B", "C", "D", "E", "unknown")),
    "material": ("material_10", "material_20", "material_30", "material_end"),
    **{
        f"opponent_{group}": tuple(f"opponent_{name}" for name in names)
        for group, names in SIDE_GROUPS.items()
    },
}
FEATURE_NAMES = tuple(name for names in FEATURE_GROUPS.values() for name in names)
FEATURE_HASH = hashlib.sha256(json.dumps(FEATURE_NAMES).encode()).hexdigest()
