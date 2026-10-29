"""LightGBM training with a train-only Ridge preprocessing baseline."""

from dataclasses import asdict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from pydantic.dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from .config import TIME_CATEGORIES, FeatureConfig, TrainingConfig
from .evaluation import evaluate, grouped_importance, sample_weights
from .feature_schema import feature_schema
from .split import PARTITIONS, assert_disjoint, normalize_player


@dataclass
class TrainingResult:
    """Selected native boosters, evaluation, and reproducibility information."""

    models: dict
    report: dict
    metadata: dict


def validate_feature_table(frame: pd.DataFrame, schema_version: int = 2) -> pd.DataFrame:
    """Reject leakage, missing labels, malformed numeric X and duplicate side rows.

    Args:
        frame: Ordered numeric features plus labels and split metadata.
        schema_version: Explicit feature contract used to construct the table.

    Returns:
        Validated copy with normalized target usernames.
    """
    names = feature_schema(schema_version).names
    required = {
        *names,
        "rating",
        "category",
        "partition",
        "target_username",
        "game_id",
    }
    if not required.issubset(frame.columns):
        raise ValueError(f"Missing columns: {sorted(required - set(frame.columns))}")
    result = frame.copy()
    result["target_username"] = result["target_username"].map(normalize_player)
    assert_disjoint(result)
    if set(result["partition"]) != set(PARTITIONS):
        raise ValueError(
            "Training requires nonempty train, validation and test partitions"
        )
    matrix = result.loc[:, list(names)].to_numpy(dtype=float)
    if np.isinf(matrix).any():
        raise ValueError("Feature values may be finite or NaN, never infinite")
    ratings = result["rating"].to_numpy(dtype=float)
    if not np.isfinite(ratings).all() or (ratings <= 0).any():
        raise ValueError("Labels must be finite positive game-time ratings")
    if not set(result["category"]).issubset(TIME_CATEGORIES):
        raise ValueError("Unsupported training time category")
    if result.duplicated(["game_id", "target_username"]).any():
        raise ValueError("Duplicate player/game rows")
    return result


class ModelTrainer:
    """Fit pooled or category-specific boosters on explicitly isolated rows."""

    def __init__(self, config: TrainingConfig | None = None, feature_config: FeatureConfig | None = None):
        """Prepare training without fitting or touching any data.

        Args:
            config: Reproducible tuning, model mode and weighting settings.
            feature_config: Versioned transformations used to construct the table.

        Returns:
            None.
        """
        self.config = config or TrainingConfig()
        self.schema = feature_schema((feature_config or FeatureConfig()).schema_version)

    def _fit_booster(
        self,
        train: pd.DataFrame,
        validation: pd.DataFrame,
        weights: np.ndarray,
        parameters: dict,
        progress: bool = False,
    ):
        """Fit one candidate, using validation only for early stopping.

        Args:
            train: Training feature/label rows.
            validation: Held-out model-selection rows.
            weights: Train-only sample weights.
            parameters: Bounded candidate tree settings.
            progress: Show a boosting-round bar for the selected final fit.

        Returns:
            Fitted LGBMRegressor with its best iteration selected.
        """
        model = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=self.config.max_trees,
            learning_rate=0.05,
            random_state=self.config.seed,
            n_jobs=self.config.threads,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            **parameters,
        )
        with tqdm(
            total=self.config.max_trees,
            desc="Selected LightGBM fit",
            unit="tree",
            disable=not progress,
            leave=False,
        ) as bar:

            def update_progress(environment):
                """Advance the final-fit bar after each completed boosting round.

                Args:
                    environment: LightGBM callback environment for the current round.

                Returns:
                    None; the progress display is updated in place.
                """
                bar.update(1)

            model.fit(
                train.loc[:, list(self.schema.names)],
                train["rating"],
                sample_weight=weights,
                eval_set=[
                    (validation.loc[:, list(self.schema.names)], validation["rating"])
                ],
                eval_metric="l1",
                callbacks=[
                    update_progress,
                    lgb.early_stopping(
                        self.config.early_stopping_rounds, verbose=False
                    ),
                ],
            )
            bar.set_postfix(best_iteration=model.best_iteration_, refresh=False)
        return model

    def _tune(
        self, train: pd.DataFrame, validation: pd.DataFrame, weights: np.ndarray
    ) -> tuple[dict, list[dict]]:
        """Choose parameters without consulting test rows.

        Args:
            train: Training feature/label rows.
            validation: Separate validation feature/label rows.
            weights: Weights computed from training rows alone.

        Returns:
            Best parameter dictionary and a JSON-compatible trial history.
        """
        if not self.config.trials:
            return {"num_leaves": 15, "min_child_samples": 20}, []

        def objective(trial):
            """Measure validation MAE for one bounded search candidate.

            Args:
                trial: Optuna trial requesting the candidate parameters.

            Returns:
                Validation MAE in rating points.
            """
            parameters = {
                "num_leaves": trial.suggest_int("num_leaves", 7, 31),
                "min_child_samples": trial.suggest_int("min_child_samples", 10, 60),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10, log=True),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
            }
            model = self._fit_booster(train, validation, weights, parameters)
            predictions = model.predict(validation.loc[:, list(self.schema.names)])
            return float(np.mean(np.abs(predictions - validation["rating"].to_numpy())))

        study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=self.config.seed),
        )
        study.optimize(
            objective,
            n_trials=self.config.trials,
            show_progress_bar=self.config.progress,
        )
        history = [
            {"trial": trial.number, "mae": trial.value, "parameters": trial.params}
            for trial in study.trials
        ]
        return study.best_params, history

    def _fit_group(self, frame: pd.DataFrame) -> tuple:
        """Fit a baseline and selected booster for one pooled/category group.

        Args:
            frame: Validated group rows spanning all three partitions.

        Returns:
            Native booster and metrics, tuning history and grouped importance.
        """
        partitions = {
            name: frame.loc[frame["partition"] == name] for name in PARTITIONS
        }
        for name, subset in partitions.items():
            if (
                len(subset) < self.config.min_rows
                or subset["target_username"].nunique() < 2
            ):
                raise ValueError(
                    f"Insufficient {name} rows/players; collect more data or adjust splits"
                )
        train, validation = partitions["train"], partitions["validation"]
        weights = sample_weights(
            train, self.config.balance_bands, self.config.balance_players
        )
        baseline = make_pipeline(
            SimpleImputer(strategy="median", keep_empty_features=True),
            StandardScaler(),
            Ridge(alpha=10),
            memory=None,
        )
        baseline.fit(
            train.loc[:, list(self.schema.names)],
            train["rating"],
            ridge__sample_weight=weights,
        )
        parameters, trials = self._tune(train, validation, weights)
        model = self._fit_booster(
            train, validation, weights, parameters, progress=self.config.progress
        )
        report = {
            "parameters": parameters,
            "trials": trials,
            "best_iteration": model.best_iteration_,
            "feature_importance": grouped_importance(model.booster_, self.schema.groups),
            "partitions": {},
        }
        for name, subset in partitions.items():
            matrix = subset.loc[:, list(self.schema.names)]
            report["partitions"][name] = {
                "lightgbm": evaluate(subset, model.predict(matrix)),
                "ridge": evaluate(subset, baseline.predict(matrix)),
                "train_median": evaluate(
                    subset, np.full(len(subset), train["rating"].median())
                ),
            }
        return model.booster_, report, baseline

    def fit(self, frame: pd.DataFrame) -> TrainingResult:
        """Select using validation, then evaluate the untouched test partition once.

        Args:
            frame: Features and game-time labels with preassigned split metadata.

        Returns:
            Selected boosters, evaluation reports and reproducibility metadata.
        """
        frame = validate_feature_table(frame, self.schema.version)
        groups = {"pooled": frame}
        if self.config.mode == "separate":
            groups = {
                name: frame.loc[frame["category"] == name]
                for name in sorted(frame["category"].unique())
            }
        models, reports, baselines = {}, {}, {}
        for name, group in tqdm(
            groups.items(),
            desc="Train models",
            unit="model",
            disable=not self.config.progress,
        ):
            models[name], reports[name], baselines[name] = self._fit_group(group)
        if self.config.mode == "separate":
            reports["combined"] = self._combined_report(
                frame, groups, models, baselines
            )
        return TrainingResult(
            models,
            reports,
            {
                "training": asdict(self.config),
                "target": "Chess.com game-time player rating",
                "split_policy": "target-player and game disjoint; opponents may overlap",
                "categories": sorted(frame["category"].unique()),
            },
        )

    def _combined_report(
        self, frame: pd.DataFrame, groups: dict, models: dict, baselines: dict
    ) -> dict:
        """Evaluate category routers as one predictor across rating bands.

        Args:
            frame: All category rows with their unchanged split assignments.
            groups: Category-to-row mappings used for the selected models.
            models: Selected category boosters.
            baselines: Train-only fitted category Ridge pipelines.

        Returns:
            Combined overall, category and rating-band metrics per partition.
        """
        report = {"partitions": {}}
        medians = {
            name: rows.loc[rows["partition"] == "train", "rating"].median()
            for name, rows in groups.items()
        }
        for partition in PARTITIONS:
            rows = frame.loc[frame["partition"] == partition]
            predictions = {
                name: np.empty(len(rows))
                for name in ("lightgbm", "ridge", "train_median")
            }
            for category, model in models.items():
                mask = (rows["category"] == category).to_numpy()
                matrix = rows.loc[mask, list(self.schema.names)]
                predictions["lightgbm"][mask] = model.predict(matrix, num_threads=1)
                predictions["ridge"][mask] = baselines[category].predict(matrix)
                predictions["train_median"][mask] = medians[category]
            report["partitions"][partition] = {
                name: evaluate(rows, values) for name, values in predictions.items()
            }
        return report
