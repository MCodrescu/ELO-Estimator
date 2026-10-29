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

The selected bundle is [artifacts/rating_model](artifacts/rating_model/model_manifest.json),
version `rating-fast-v2`. It used 45 training, 30 validation and 23 test player
rows. Test MAE is **606.6** and RMSE **732.8** rating points. The speed/accuracy
tradeoff was explicitly accepted; full-data retraining remains pending. Older
analysis contracts and bundles are no longer supported.

## Train

```text
python -m pip install ".[train]"
```

Install Stockfish separately and set its executable in
[elo_model/config.py](elo_model/config.py). Restart the notebook kernel, then run
[build_training_data.ipynb](build_training_data.ipynb) from Cell 1 onward. It reuses
saved raw games and collects more only when needed. The workflow splits by target
player and game before analysis, applies training quotas, extracts fast features,
tunes LightGBM on validation rows and reports held-out errors before export.

### Expanded Training Run

The notebook now selects `PipelineConfig.expanded()`. This leaves the original
configuration and released model unchanged and uses these targets:

- 14 training bands: `[100,350)`, `[350,600)`, through `[3100,3350)`,
    then `[3350,3501)` to include rating 3500.
- Per band: 1,000 blitz, 500 rapid and 500 bullet player-side training rows,
    totaling 28,000 rows. Classical training is deferred.
- At most 10 selected training rows per player across all quota groups.
- 60,000 candidate games from the current month and five preceding months.
- Discovery targets 300 usernames per band per time class, with at most
    3,000 archive-owner expansions per class. These are bounded targets, not
    guarantees, particularly above 3100.

Restart the kernel, then run **Cells 1-5 first**. Cell 2 seeds separate blitz,
rapid and bullet searches using up to 200 shuffled existing usernames plus the
configured seed. It preserves the original username file, saves the expanded
pool under `data/expanded/usernames.json`, and checkpoints each completed
category search. An interrupted category restarts; completed unchanged searches
are reused. Discovery itself can require many archive requests and respects
the existing proxy, retry and request-delay settings.

To improve coverage or add known active high/low-rated players, edit Cell 1:

```python
settings = replace(
        PipelineConfig.expanded(),
        discovery_seed_usernames=("additional_username",),
        target_games=30000,  # Collect up to 90,000 candidates rather than 60,000.
)
```

Only enter real usernames. New seeds invalidate discovery checkpoints and extend
the saved pool. Increasing the candidate budget resumes raw collection; it does
not rerun completed searches. To skip discovery, set `discovery_enabled=False`;
existing saved usernames remain available. Review shortfalls after each
collection pass and expand the budget or seeds where needed.

Cell 5 saves training and partition coverage before Stockfish. It stops for
unfilled training quotas or fewer than 100 distinct training players in a
rating/category group, and warns about holdout groups with fewer than 20 players.
Rare upper bands may remain underfilled. After reviewing those reports, either
collect more data or explicitly set `allow_underfilled_bands = True` and rerun
Cell 5 to accept the recorded shortfalls. Then run Cells 6-10 for analysis,
training, evaluation and export. Cell 11 is an optional live-game demonstration.

Expanded raw games, splits, analysis references and features live under
`data/expanded`; exports go to `artifacts/rating_model_expanded`. The original
`artifacts/rating_model` bundle is not overwritten. The shared
`data/analysis_cache` is keyed by game history and the complete engine contract,
so unchanged evaluations are reused. Existing collection months and source-player
caps must remain fixed when resuming; if the calendar advances, restore the saved
months or choose new run directories.

Evaluation retains the legacy broad-band reports and adds fine-band and
rating/category reports, including empty groups, row/player counts, MAE, RMSE,
macro-player MAE and signed bias (prediction minus actual rating). Training quotas
do not rebalance validation/test rows, and sparse holdout groups should not be
interpreted as reliable evidence of accuracy. Compare candidate models on the
same held-out data when assessing improvement over a previous release.

Individual stages are available through `python -m elo_model --help`.

## Predict

```python
from elo_model.config import InferenceConfig
from elo_model.inference import RatingEstimator

estimator = RatingEstimator(InferenceConfig(
    model_directory="artifacts/rating_model",
    engine_executable="/path/to/stockfish",
    cache_directory="data/analysis_cache",
))
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