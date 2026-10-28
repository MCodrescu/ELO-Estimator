# Single-Game Chess Rating Estimator

Estimate each player's **Chess.com rating from a single completed game**. The
pipeline turns moves, clock usage, time control, and Stockfish evaluations into
109 numeric features per player, then trains a LightGBM regressor against their
recorded game-time rating.

The goal is to learn how playing behavior relates to rating, rather than identify
known players. Usernames and rating labels are excluded from model inputs.

## How It Works

1. **Collect games.** Fetch Chess.com archives for saved usernames, deduplicate
   games, and retain rated, completed standard games with usable clock annotations.
2. **Split before training.** Assign games and prediction-target players to
   train, validation, and test partitions to reduce leakage from repeat players.
   Select training sides by game-time rating/category quotas; leave hold-outs unchanged.
3. **Analyze decisions.** Replay each game with Stockfish at depth 12, comparing
   played moves with the top three alternatives. Moves outside the top three get
   a separate evaluation so their quality is measured rather than guessed.
4. **Summarize behavior.** Build a fixed-size feature vector for each side.
   Aggregates describe accuracy, decision difficulty, and time management across
   games of different lengths.
5. **Train and evaluate.** Tune LightGBM with Optuna using validation error, then
   evaluate on the untouched test partition. Compare with Ridge and a train-median
   baseline, reporting MAE and RMSE overall, by rating band, and by time category.
6. **Export and predict.** Save native LightGBM models with their feature schema
   and analysis settings. Verify that reloaded models reproduce predictions.

By default, opponents may appear across partitions even though prediction-target
players do not. A stricter split can require both participants to stay within the
same partition, at the cost of discarding more games.

## Features

| Signal | What It Captures |
| --- | --- |
| Move quality | Centipawn and score-derived probability loss; inaccuracies, mistakes, and blunders |
| Game phases | Accuracy during the opening, middlegame, and endgame |
| Position and difficulty | Play in winning or losing positions, gaps between candidate moves, and missed tactics |
| Error patterns | Consecutive blunders and clusters of mistakes |
| Time management | Think times, instant moves, time trouble, and time spent on difficult decisions |
| Game context | Time control, opening family, result, termination, and material checkpoints |
| Opponent behavior | Mirrored accuracy, difficulty, and clock summaries |

Missing observations remain missing rather than being filled with artificial
zeros. Result and termination features make this a **post-game estimator**;
predictions from unfinished games are outside the training distribution.

## Reuse a Model

Install the package with `pip install ".[engine]"` from this checkout and deploy
an exported model bundle plus a compatible Stockfish executable:

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

Reuse the estimator per worker; run engine analysis in background jobs for web
apps. Models and Stockfish are not bundled with the package. See
[docs/integration.md](docs/integration.md) for Django integration, installation in
another repository, engine compatibility, and publishing versioned weights.

## Running It

Install the Python dependencies and a Stockfish executable:

```powershell
python -m pip install -r requirements.txt
```

Set engine paths, collection settings, and training options in
[elo_model/config.py](elo_model/config.py), then run
[build_training_data.ipynb](build_training_data.ipynb) in order. It collects games
from [player_usernames.json](player_usernames.json), builds the dataset, trains the
model, and exports a bundle. Individual stages are also available through
`python -m elo_model --help`.

Training selection is configured through `PipelineConfig.stratification`. Defaults
collect up to 3,000 candidate games, then request 500 blitz rows and 100 rows for
each other time category per rating band, with at most 10 rows per target player.
Cell 5 reports available/selected players and rows, including unfilled quotas,
before Stockfish runs. Quotas are caps, not guaranteed collection yields.

For example, restrict training to a particular rating range and category:

```python
from dataclasses import replace
from elo_model.config import RatingQuota, StratificationConfig

settings = replace(settings, stratification=StratificationConfig(quotas=(
   RatingQuota(lower=1000, upper=1400, category="blitz", rows=500),
)))
```

Bounds are lower-inclusive and upper-exclusive. Apply overrides after notebook
setup and before collection. Use `StratificationConfig()` to disable selection.
CLI `split --selection-config quotas.json` accepts the same settings as JSON.
Selection settings and coverage are saved beside the split manifest and included
in the notebook's exported model provenance. Existing raw games remain reusable.

Analysis uses **eight concurrent Stockfish processes** by default. Per-game
evaluations are cached by move history, engine identity, and analysis settings, so
additional datasets can reuse completed work. Changing worker count does not
invalidate the cache; features can also be recalculated without rerunning Stockfish.

## Code Map

- [training_data/model_dataset.py](training_data/model_dataset.py): collection, filtering, and deduplication.
- [elo_model/analysis.py](elo_model/analysis.py): engine evaluation and persistent cache.
- [elo_model/features.py](elo_model/features.py): player-level feature extraction.
- [elo_model/split.py](elo_model/split.py): player-aware dataset partitioning.
- [elo_model/selection.py](elo_model/selection.py): reproducible training quotas and coverage reports.
- [elo_model/training.py](elo_model/training.py): model fitting, tuning, and evaluation.
- [elo_model/predictor.py](elo_model/predictor.py): model export, loading, and inference.
- [elo_model/pipeline.py](elo_model/pipeline.py): stage orchestration and parallel analysis.

Run the offline test suite with `python -m unittest discover -s tests`.