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

Raw games and splits live in `data/notebook`; fast analysis references and features
live in `data/notebook_fast_v2`. The shared engine cache is keyed by game history
and complete engine contract. Exports go to `artifacts/rating_model`; choose a new
artifact directory when preserving a release. Individual stages are available
through `python -m elo_model --help`.

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