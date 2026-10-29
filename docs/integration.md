# Integration

## Package and Weights

Use package **0.2.0** from a pinned repository commit or a wheel built from that
commit. It is not published on PyPI. Install `.[engine]` from a checkout for cached
PGN analysis, or the base package for supplied-analysis prediction. The `train`
extra is only needed for offline collection and training.

Deploy all five files from `artifacts/rating_model` separately from the package:
`pooled.txt`, `model_manifest.json`, `feature_names.json`, `feature_groups.json`
and `evaluation.json`. Verify checksums and load only trusted local files.
The selected model version is `rating-fast-v2`, schema 2 with 93 inputs.

The current weights used 45 training, 30 validation and 23 test player rows.
Test MAE is 606.6 and RMSE 732.8. The speed/accuracy tradeoff was explicitly
accepted; full-data retraining remains pending. The manifest and evaluation
describe the actual weights, not a production accuracy guarantee.

## Analyze Once

```python
from dataclasses import asdict

from elo_model.analysis import AnalysisCache, StockfishAnalyzer
from elo_model.config import EngineConfig, InferenceConfig
from elo_model.game import GameInput
from elo_model.inference import RatingEstimator

estimator = RatingEstimator(InferenceConfig(
    model_directory="artifacts/rating_model",
))
game = GameInput.from_pgn(pgn_text)
engine_config = EngineConfig(
    executable="/path/to/compatible/stockfish",
    **estimator.predictor.manifest["engine"]["settings"],
)
with StockfishAnalyzer(engine_config, AnalysisCache("data/review-cache")) as analyzer:
    estimator.predictor.validate_engine_metadata(analyzer.metadata)
    analysis_key, positions = analyzer.analyze(game)
    engine_metadata = analyzer.metadata

graph = [asdict(position.evaluation) for position in positions]
measurements = estimator.predictor.extractor.measurements(game, positions)
estimate = estimator.predict_pgn(
    pgn_text, analysis=positions, engine_metadata=engine_metadata,
)
payload = {
    "engine": engine_metadata,
    "positions": [asdict(position) for position in positions],
}
```

The supplied-analysis prediction launches no engine. Reload serialized positions
with `elo_model.analysis.analysis_from_payload(payload["positions"])` and retain
the original `payload["engine"]` as provenance.

Measurements include mover-relative probability loss, centipawn loss, best-move
match and phase. A simple per-player accuracy proxy is 100 times one minus the
mean probability loss over that player's moves. It is not Chess.com's accuracy
formula, calibrated WDL, or a separately validated accuracy model. Ratings are
post-game playing-strength estimates, not official Elo updates.

For raw PGNs without review output, set `engine_executable` and `cache_directory`
in `InferenceConfig`, then call `predict_pgn(pgn_text)`. Every search setting is
derived from the loaded bundle.

## Exact Contract

The current bundle requires Stockfish 19 with `depth=None`, `nodes=None`,
`time_seconds=0.1`, `multipv=1`, `threads=1`, `hash_mb=64`,
`hash_reset="game"`, and `played_move_scoring="next-position"`.

Each nonterminal reached position gets one UCI search. Streaming retains the last
completed exact iteration; interrupted bounded scores are ignored. If no exact
iteration completes, analysis raises an error rather than synthesizing scores or
starting a fallback search. Startup, scheduling, communication and cache I/O are
outside the allowance. It is not a hard end-to-end wall-clock guarantee.

For move index `i`, compare the best score at position `i` with the evaluation
of reached position `i+1`, even when the played move matches the root PV. Scores
remain White-relative before conversion to the mover's perspective. Adjacent
root searches differ in horizon and achieved depth, so this is not equivalent to
root-restricted analysis. Negative losses are clamped to zero.

Replay preserves history from the PGN's initial board. Checkmate uses the signed
mate endpoint; automatic draws use zero centipawns. Claimable threefold and
50-move draws are not automatically claimed. Resignation/timeout/result headers
do not replace a nonterminal engine evaluation. History before a custom initial
FEN cannot be reconstructed.

Clear hash before every uncached game and retain it only across that game's
chronological positions. Do not warm the hash with another game or parallelize
positions within a game. Training and inference use the same initialization.

Exact compatibility is the default: actual engine identity, binary SHA-256,
UCI defaults, analysis version and all search settings must match. Cache keys
include the actual binary and complete contract. The explicit `same-release`
policy ignores only the binary hash; it does not certify Linux parity or retained
accuracy. The shipped bundle records a Windows engine. Timed search can reach
different depths on different hardware or under load despite identical metadata.

## Deployment and Retraining

Load the estimator once per worker. Run engine work in background tasks and bound
task concurrency by CPU and memory. Keep the model directory read-only and use a
persistent writable cache. An analyzer is sequential and must not be shared by
simultaneous jobs. Invalid PGNs, missing scores and incompatible metadata are
explicit errors; never silently substitute other weights or search settings.

Train only in the notebook or offline CLI, never inside the consuming application.
Restart the notebook kernel after upgrading, then run cells in order. New runs
reuse raw games while keeping fast feature files separate. Export into a new
immutable directory for each release, verify held-out MAE/RMSE by rating band and
category, and restart application workers when changing bundles. A changed
analysis contract requires compatible features, retraining and evaluation.