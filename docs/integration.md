# Integration

## Install in Another Repository

Python 3.11 or newer is required. The distribution is named
`chess-rating-estimator`; the Python import package is `elo_model`. Package
**0.2.0** is not published on PyPI. Pin a repository commit, not just the package
version, because different commits can report the same version.

Replace `COMMIT_SHA` with the full commit selected for your deployment:

```text
python -m pip install "chess-rating-estimator[engine] @ git+https://github.com/MCodrescu/ELO-Estimator.git@COMMIT_SHA"
```

For a checked-out copy, install it into the consuming application's environment:

```text
python -m pip install "/path/to/ELO-Estimator[engine]"
```

Alternatively, build a wheel from the pinned checkout and distribute that wheel
through your normal dependency-management process. Do not copy individual Python
modules or add this repository's working directory to `sys.path`.

| Workload | Installation |
| --- | --- |
| Predict from supplied analysis or ordered numeric features | Base package |
| Analyze PGNs or read/write the Parquet analysis cache | `[engine]` |
| Run the stage CLI, collect training games, or train offline | `[train]` |
| Run the training notebook | `[train]` plus a Jupyter/IPython kernel |

PyTorch is a base dependency. Neither Stockfish nor model weights are installed
by pip. The CLI currently imports collection helpers at startup, so even its
`predict` command needs `[train]`; the Python inference API does not.

## Package and Weights

Train a neural bundle before deployment; weights are not included in the repository.
Deploy the complete `artifacts/rating_model_neural` directory separately from the
package. The directory can have any name or absolute location in the consuming
application. Keep these files together:

```text
rating-model/
    model_manifest.json
    feature_names.json
    feature_groups.json
    evaluation.json
    pooled.pt
```

For separate models, replace `pooled.pt` with the category checkpoints listed in
the manifest's `models` mapping. Do not rename those checkpoints independently.
Verify checksums and load only trusted local files. Checkpoints include
train-fitted imputation/scaling,
target normalization, architecture and weights, and load on CPU with
`torch.load(..., weights_only=True)`, even if training used CUDA. A CUDA runtime
is not required for inference.

`RatingEstimator` and `RatingPredictor` accept only PyTorch bundles. Point
`InferenceConfig.model_directory` at the exported directory. Neural models use
schema 2 with 93 inputs; no Stockfish rerun is required when training from
compatible saved features. Measure accuracy after training using the exported
evaluation report rather than assuming a production accuracy guarantee.

## Predict a PGN

Use the public `RatingEstimator` API for application integration. These paths
belong to the consuming application's deployment, not to this repository:

```python
from pathlib import Path

from elo_model.config import InferenceConfig
from elo_model.inference import RatingEstimator

estimator = RatingEstimator(
    InferenceConfig(
        model_directory=Path("/srv/models/rating-model"),
        engine_executable=Path("/usr/local/bin/stockfish"),
        cache_directory=Path("/srv/cache/chess-analysis"),
    )
)

pgn_text = Path("game.pgn").read_text(encoding="utf-8")
estimate = estimator.predict_pgn(pgn_text)
response = estimate.model_dump(mode="json")
```

Replace the example paths with Windows or Unix paths appropriate to your host.
Initialization loads weights but performs no network requests or engine searches.
Raw-PGN prediction starts Stockfish and derives its search settings from the
bundle; do not substitute unrelated engine defaults. Prefer completed,
standard-chess PGNs with clock annotations and a simple `TimeControl` header.
Rating headers are not required. Unobserved features remain missing rather than
being synthesized.

`predict_pgn` returns a validated `GameRatingEstimate`:

| Field | Meaning |
| --- | --- |
| `white`, `black` | Finite floating-point playing-strength estimates in rating units; not clipped or rounded |
| `model_id` | SHA-256 identity derived from the manifest and checkpoint contents |
| `model_version` | Optional export release label, or `None` |
| `category` | PGN time category, or `None` when it cannot be parsed |
| `analysis_key` | Cache key for analysis obtained internally, otherwise `None` |
| `cache_hit` | Whether internal analysis used the cache; `None` for supplied analysis |
| `engine_compatibility` | Applied compatibility policy |
| `engine_metadata` | Actual analysis provenance, never rewritten to match training |

These are post-game playing-strength estimates, not official Elo updates or
confidence intervals. Separate-category bundles require a parsed time category
with a matching checkpoint; a pooled bundle does not require category routing.

## Analyze Once

When a review pipeline already needs Stockfish, analyze once and share the
results with graphs, move measurements, and rating prediction. Continue with
`estimator` and `pgn_text` from the preceding example:

```python
import json
from dataclasses import asdict

from elo_model.analysis import AnalysisCache, StockfishAnalyzer
from elo_model.config import EngineConfig
from elo_model.game import GameInput

game = GameInput.from_pgn(pgn_text)
engine_config = EngineConfig(
    executable=str(estimator.config.engine_executable),
    **estimator.predictor.manifest["engine"]["settings"],
)
with StockfishAnalyzer(
    engine_config, AnalysisCache(estimator.config.cache_directory)
) as analyzer:
    estimator.predictor.validate_engine_metadata(
        analyzer.metadata,
        engine_compatibility=estimator.config.engine_compatibility,
    )
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
Path("analysis.json").write_text(json.dumps(payload), encoding="utf-8")
```

The payload contains the initial position and every reached position in
chronological order, so a game with N plies needs N+1 aligned records. Scores are
White-relative. Use this package's analyzer and serializer rather than
substituting a bare evaluation array.

To consume that payload in another process without Stockfish installed:

```python
import json
from pathlib import Path

from elo_model.analysis import analysis_from_payload
from elo_model.config import InferenceConfig
from elo_model.inference import RatingEstimator

estimator = RatingEstimator(
    InferenceConfig(model_directory=Path("/srv/models/rating-model"))
)
pgn_text = Path("game.pgn").read_text(encoding="utf-8")
payload = json.loads(Path("analysis.json").read_text(encoding="utf-8"))

estimate = estimator.predict_pgn(
    pgn_text,
    analysis=analysis_from_payload(payload["positions"]),
    engine_metadata=payload["engine"],
)
```

This prediction launches no engine and needs no Parquet cache. Pass both the
positions and their original engine metadata. Do not relabel old or incompatible
analysis as the model's training provenance.

Measurements include mover-relative probability loss, centipawn loss, best-move
match and phase. A simple per-player accuracy proxy is 100 times one minus the
mean probability loss over that player's moves. It is not Chess.com's accuracy
formula, calibrated WDL, or a separately validated accuracy model.

## Predict Feature Rows

Use `RatingPredictor` when your application already has features created by this
package's `GameFeatureExtractor` using the same feature and engine contract:

```python
import pandas as pd

from elo_model.predictor import RatingPredictor

predictor = RatingPredictor("/srv/models/rating-model")
rows = pd.read_parquet("features.parquet")
feature_rows = rows.loc[:, list(predictor.schema.names)]
predictions = predictor.predict(feature_rows, categories=rows["category"].tolist())
```

Reading this Parquet example needs `[engine]`; prediction itself needs only the
base package. Inputs must have exactly the saved feature order and width, with
finite numbers or NaN, never infinity. A DataFrame with extra or reordered
columns is rejected. Ratings, usernames, and split metadata must not be included
in the input matrix. For separate bundles, pass one supported category per row.
For pooled bundles, `categories` can be omitted. Do not refit preprocessing or
derive a replacement feature order in the consuming application.

## Exact Contract

The default training configuration uses Stockfish with `depth=None`, `nodes=None`,
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
accuracy. Opt in explicitly when deploying another build of the same release:

```python
config = InferenceConfig(
    model_directory="/srv/models/rating-model",
    engine_executable="/usr/local/bin/stockfish",
    cache_directory="/srv/cache/chess-analysis",
    engine_compatibility="same-release",
)
estimator = RatingEstimator(config)
```

All other recorded metadata, including engine identity, UCI defaults,
initialization, analysis version, and search settings, must still match. Consult
the exported manifest for the training engine build. Timed search can reach
different depths on different hardware or under load despite identical metadata.

## Deployment and Retraining

Load the estimator once per worker. Run engine work in background tasks and bound
task concurrency by CPU and memory. Keep the model directory read-only and use a
persistent writable cache. An analyzer is sequential and must not be shared by
simultaneous jobs. Invalid PGNs, missing scores and incompatible metadata are
explicit errors; never silently substitute other weights or search settings.

Expect `ValueError` for incompatible schemas, unsupported categories, malformed
analysis, or engine-contract mismatches. Missing files and engine process failures
also propagate. Translate these into your application's errors with context;
do not return a fabricated default rating. A failed constructor must not leave
the application claiming that its rating service is ready.

Keep the package commit, bundle checksum, optional `model_version`, and engine
build together in deployment records. Do not mix JSON files or checkpoints from
different training runs. Python wheels exclude model artifacts and collected
data. The consuming application should not import offline trainers or invoke
the training CLI during request handling.

Train only in the notebook or offline CLI, never inside the consuming application.
Restart the notebook kernel after upgrading, then run cells in order. New runs
reuse raw games while keeping fast feature files separate. Export into a new
immutable directory for each release, verify held-out MAE/RMSE by rating band and
category, and restart application workers when changing bundles. A changed
analysis contract requires compatible features, retraining and evaluation.