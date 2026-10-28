# Reusing a Model

Install the Python package and deploy a compatible model bundle separately. The
package contains inference code, not trained weights or a Stockfish executable.
Code and model versions are independent; pin both in the consuming application.

The current bundle is tracked in `artifacts/rating_model`. Copy all five files
from a pinned checkout, including the native model, manifest, feature names/groups,
and evaluation report. Installing the wheel or Git-based Python dependency does
not install these weights; they must be deployed separately. The current manifest
records a Windows Stockfish fingerprint, so Linux-generated analysis is still
rejected until cross-build compatibility is explicitly validated and supported.

## Installation

From a checkout, install `pip install ".[engine]"` for PGN analysis with the
persistent Parquet cache, or `pip install .` for supplied-analysis/feature-only
prediction. Other repositories can install a built wheel or pin a Git commit:

```text
chess-rating-estimator[engine] @ git+https://github.com/MCodrescu/ELO-Estimator.git@<commit-sha>
```

Replace the placeholder with a real commit or published tag. This project is not
yet published on PyPI. The base install excludes Optuna, scikit-learn, and the
Chess.com SDK. Training/collection CLI stages require the `train` extra;
`chesscom` adds just the existing collection client. These examples were tested
on Python 3.13; package metadata permits Python 3.11 and newer.

## Predict a PGN

```python
from elo_model.config import InferenceConfig
from elo_model.inference import RatingEstimator

estimator = RatingEstimator(InferenceConfig(
    model_directory="/srv/models/rating-v1",
    engine_executable="/opt/stockfish/stockfish",
    cache_directory="/srv/cache/chess-analysis",
))

estimate = estimator.predict_pgn(pgn_text)
result = estimate.model_dump(mode="json")
```

Load the estimator once per worker process and reuse it. Construction loads models
but starts no engine and makes no network requests. Raw PGN prediction starts an
engine for that call and reuses complete cached evaluations. The result contains
`white`, `black`, a content-derived `model_id`, optional `model_version`, time
`category`, `analysis_key`, and `cache_hit`. Ratings are estimates, not calibrated
confidence intervals or official rating updates.

For a Chess.com API game object, pass `game_record["pgn"]` to the same method.
Retrieval stays outside inference so applications can supply their own HTTP,
retry, proxy, authentication, and game-selection policies. The package also
retains `training_data.ChessComConnection` for optional collection workflows.

Raw analysis uses the depth, MultiPV, thread count, and hash size recorded in the
bundle. The executable's binary/version fingerprint must match training provenance.
A Windows-trained model is not automatically compatible with Linux-generated
analysis: deploy matching engine environments or train a bundle for the target
environment. Never relabel incompatible engine metadata to bypass this check.

## Reuse Existing Analysis

```python
from elo_model.analysis import analysis_from_payload

estimate = estimator.predict_pgn(
    pgn_text,
    analysis=analysis_from_payload(payload["positions"]),
    engine_metadata=payload["engine"],
)
```

When supplying analysis, only `model_directory` is required in `InferenceConfig`.
The engine is not launched. Payload positions must match the initial board and
every reached board, including exact played-move scores outside the top three.
Cache fields are `None` when the service did not perform the cache lookup.

For already extracted features, use `elo_model.predictor.RatingPredictor.predict`.
DataFrame columns must exactly match the exported feature order; separate-category
models also require a category per row. Reuse this package's extractor rather than
reimplementing approximately equivalent features.

## Django Integration

No separate HTTP model server is required. Configure explicit deployment paths in
your Django settings:

```python
RATING_MODEL_DIRECTORY = "/srv/models/rating-v1"
RATING_STOCKFISH_EXECUTABLE = "/opt/stockfish/stockfish"
RATING_ANALYSIS_CACHE = "/srv/cache/chess-analysis"
```

Create the estimator lazily in a worker-local application service. Run game
retrieval and `predict_pgn` in a background task, then persist its JSON result.
Return a job identifier from the web request instead of holding a request open
during Stockfish analysis. Limit task concurrency to available engine CPU/memory;
eight web tasks do not share the notebook's eight-engine pool automatically.
Use a persistent writable cache and a read-only, trusted model directory.

Treat invalid PGNs, unsupported model categories, and incompatible engine/schema
errors as validation failures rather than silently returning fallback ratings.
Do not retrain in the Django process. Deploy a new immutable model directory for
each release and restart workers when switching versions.

## Releasing Weights

1. Train and evaluate in the intended serving engine environment. Record held-out
   metrics, rating/time-category coverage, and single-game limitations.
2. Export a verified bundle with a release label:

   ```text
   python -m elo_model export --source runs/latest --output artifacts/rating-v1 --sample-features data/notebook/features.parquet --model-version rating-v1
   ```

   Use the actual training run path. Notebook-trained models already live in the
   configured artifact directory and can also be exported with `export_bundle`.
   The export CLI needs the `train` extra, not just the inference dependencies.
3. Package the complete directory: native model text files, model manifest,
   feature names/groups, and evaluation report. Exclude raw games and engine cache.
4. Publish the archive and SHA-256 checksum as GitHub Release assets, together with
   the compatible package commit/version and engine build requirements. Consumers
   must verify the archive checksum before extracting trusted files.
5. Choose and publish appropriate code/model licenses after confirming ownership
   and redistribution rights. No open-source license is granted merely by package
   metadata or making the repository public.

The library does not automatically download models, extract archives, or update
weights. Release hosting, licensing, and deployment remain explicit steps.