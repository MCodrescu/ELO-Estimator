# Chess Rating Estimator

Estimate both players' Chess.com playing strength from a completed game using
clock usage, game context and Stockfish evaluations. Package **0.2.0** supports
timed single-PV analysis with next-position move scoring and 93 schema-2 features.

## Analysis

- 0.1-second total engine-search allowance per reached position.
- One thread, 64 MiB hash, MultiPV 1; no played-move fallback searches.
- Clear hash before each game, retain it across chronological positions.
- White-perspective scores and full replay history, including the final position.

The allowance is not an end-to-end wall-clock guarantee. Analysis runs once and
can be reused for the evaluation graph, move insights, accuracy summaries and
rating predictions. Candidate-gap and top-three features are not used.

Train a neural bundle before prediction. Model weights are not included in the
repository. Only PyTorch checkpoints with the current analysis contract are supported.

## Use in Another Repository

Use Python 3.11+ and install a pinned commit or a wheel built from it. The pip
distribution is `chess-rating-estimator`; imports use `elo_model`.

```text
python -m pip install "chess-rating-estimator[engine] @ git+https://github.com/MCodrescu/ELO-Estimator.git@COMMIT_SHA"
```

Replace `COMMIT_SHA` with your selected full commit. Deploy a separately trained
PyTorch bundle and a compatible Stockfish executable. For supplied-analysis
prediction, omit `[engine]`; no Stockfish installation is needed. See the
[integration guide](docs/integration.md) for deployment files, dependency choices,
API examples, response fields, and engine compatibility requirements.

## Train

```text
python -m pip install ".[train]" ipykernel ipython
```

Install Stockfish separately and set its executable in
[elo_model/config.py](elo_model/config.py). Restart the notebook kernel, then run
[build_training_data.ipynb](build_training_data.ipynb) from Cell 1 onward. It reuses
saved raw games and collects more only when needed. The workflow splits by target
player and game before analysis, applies training quotas, extracts fast features,
trains a PyTorch multilayer perceptron (MLP) and reports held-out errors before
export.

### Neural Training

The network uses two hidden layers of 128 and 64 units with ReLU and 0.1 dropout.
Medians, feature scaling and rating normalization are fitted on training rows
only; each feature also has a missing-value indicator. AdamW optimizes weighted
Huber loss, with early stopping on validation MAE and a train-median baseline.
Test rows are evaluated only after the best epoch is restored.

Construct a validated `NeuralTrainingConfig` and replace `settings.neural_training`
to adjust architecture, batch size, learning rate, epochs and patience. CPU is
the default; `device="auto"` selects CUDA when available. CUDA requires a
compatible PyTorch installation.

If `data/expanded/features.parquet` and its metadata sidecar already exist,
restart the kernel and run **Cell 1, then Cells 9-11**. Collection, splitting,
Stockfish and feature extraction do not need to run again. Otherwise run the
notebook in order. Cell 2 currently overrides `target_games=2000`, reducing the
candidate budget to 6,000 games without reducing the 20,000-row training quotas.
Skip or edit that cell to use the expanded configuration's 60,000-game budget;
the smaller override is likely to leave quotas underfilled.

The CLI can also train directly from those features:

```text
python -m elo_model train --input data/expanded/features.parquet --output artifacts/rating_model_neural --hidden-sizes 128 64 --max-epochs 100 --patience 10 --device cpu --progress
```

Both paths export `pooled.pt` (or category-specific `.pt` files), preprocessing,
feature metadata, history and evaluation reports. Reload parity uses float32
tolerances (`rtol=1e-6`, `atol=1e-4`). The CLI's staged defaults use `runs/neural`
for training and `artifacts/rating_model_neural` for export/prediction.
Measure held-out MAE/RMSE by rating band and category before choosing deployment weights.

### Expanded Training Run

The notebook now selects `PipelineConfig.expanded()`. This leaves the original
collection configuration unchanged and uses these targets:

- 10 training bands: `[100,350)`, `[350,600)`, through `[2100,2350)`,
    then `[2350,2601)` to include rating 2600. Higher ratings have no discovery
    or training quota; out-of-range search bridges and holdout diagnostics remain.
- Per band: 1,000 blitz, 500 rapid and 500 bullet player-side training rows,
    totaling 20,000 rows. Classical training is deferred.
- At most 10 selected training rows per player across all quota groups.
- 60,000 candidate games from the current month and five preceding months.
- Discovery targets 300 usernames per band per time class, with at most
    3,000 archive-owner expansions per class (3,000 accepted usernames).
    These are bounded targets, not guarantees.

Restart the kernel, then run **Cells 1-6 first**. Cell 3 seeds separate blitz,
rapid and bullet searches using up to 200 shuffled existing usernames plus the
configured seed. It preserves the original username file, saves the expanded
pool under `data/expanded/usernames.json`, and checkpoints each completed
category search. An interrupted category restarts; completed unchanged searches
are reused. Discovery itself can require many archive requests and respects
the existing proxy, retry and request-delay settings.

Discovery progress puts quota completion first: accepted/target usernames,
remaining usernames, filled/total bands, and a separate bar for each rating band.
Quota ETA is a rough extrapolation from the slowest unfinished band's observed
acceptance rate, not a guaranteed finish time. It is unknown for unseen bands or
after a minute without net quota gains. Archive-budget ETA is labeled separately;
budget or stagnation stops leave unmet quotas visible rather than showing 100%.

Expanded discovery now uses only the newest configured collection month, while
training collection retains all six months. Raw monthly archives are shared
between discovery categories and training through `data/archive_cache`.
Completed historical months are reused; current-month snapshots refresh after
one hour. Duplicate requests within the same process share a per-archive lock.
Each category also stops after 500 player expansions without improved retained
quota coverage, saving its partial results and stop reason. Training quotas and
request concurrency are unchanged.

If coverage needs older discovery data, set `discovery_month_limit=2` (or `None`
for every collection month) in the Cell 1 settings. To disable the stall limit,
replace the collection settings with `search_stagnation_expansions=None`.
Changing discovery settings reruns the affected searches but reuses raw archives.
Restart the kernel, apply your settings overrides, and rerun Cell 3; the archive
budget is 3,000 per category rather than 18,000 before cache reuse or early stops.

To improve coverage or add known active high/low-rated players, edit Cell 1:

```python
settings = replace(
    settings,
    discovery_seed_usernames=("additional_username",),
    target_games=30000,
)
```

Only enter real usernames. New seeds invalidate discovery checkpoints and extend
the saved pool. Increasing the candidate budget resumes raw collection; it does
not rerun completed searches. To skip discovery, set `discovery_enabled=False`;
existing saved usernames remain available. Review shortfalls after each
collection pass and expand the budget or seeds where needed.

Cell 6 saves training and partition coverage before Stockfish. It checks for
unfilled training quotas or fewer than 100 distinct training players in a
rating/category group, and warns about holdout groups with fewer than 20 players.
Rare upper bands may remain underfilled. After reviewing those reports, either
collect more data or explicitly set `allow_underfilled_bands = True` and rerun
Cell 6 to accept the recorded shortfalls. The current notebook explicitly accepts
underfilled bands in that cell; remove that override to require approval.
Then run Cells 7-11 for analysis, training, evaluation and export.
Cell 12 is an optional live-game demonstration.

Expanded raw games, splits, analysis references and features live under
`data/expanded`; neural exports go to `artifacts/rating_model_neural`. The shared
`data/analysis_cache` is keyed by game history and the complete engine contract,
so unchanged evaluations are reused. Existing collection months and source-player
caps must remain fixed when resuming; if the calendar advances, restore the saved
months or choose new run directories. The notebook rejects a mismatch with
`Collection months or source-player cap changed`; do not delete the provenance
sidecar to bypass that check. Before Cell 3, restore the recorded collection
settings when intentionally resuming the same dataset:

```python
saved_collection = json.loads(
    settings.collection_settings_path.read_text(encoding="utf-8")
)
settings = replace(
    settings,
    collection=replace(
        settings.collection,
        months=tuple(tuple(month) for month in saved_collection["months"]),
        max_games_per_player=saved_collection["source_player_cap"],
    ),
)
```

This example uses the imports from Cell 1. Retraining from existing compatible
features can instead use the Cell 1, then Cells 9-11 shortcut above.

Evaluation includes broad-band, fine-band and
rating/category reports, including empty groups, row/player counts, MAE, RMSE,
macro-player MAE and signed bias (prediction minus actual rating). Training quotas
do not rebalance validation/test rows, and sparse holdout groups should not be
interpreted as reliable evidence of accuracy. Compare candidate models on the
same held-out data when assessing improvement over a previous release.

Individual stages are available through `python -m elo_model --help`.

## Predict

```python
from pathlib import Path

from elo_model.config import InferenceConfig
from elo_model.inference import RatingEstimator

estimator = RatingEstimator(InferenceConfig(
    model_directory="artifacts/rating_model_neural",
    engine_executable="/path/to/stockfish",
    cache_directory="data/analysis_cache",
))
pgn_text = Path("game.pgn").read_text(encoding="utf-8")
estimate = estimator.predict_pgn(pgn_text)
print(estimate.model_dump())
```

Reuse the estimator per worker and run engine analysis in background jobs.
The package does not include Stockfish or install weights automatically. See
[docs/integration.md](docs/integration.md) for deployment, exact compatibility
requirements and reusing supplied analysis without another engine pass.

## Validate

```text
python -m unittest discover -s tests -q
ruff check elo_model training_data tests
```