"""Train PyTorch rating networks on the existing isolated feature partitions."""

from threading import Lock

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from .config import FeatureConfig, NeuralTrainingConfig
from .evaluation import evaluate, sample_weights
from .feature_schema import feature_schema
from .neural_model import NeuralPreprocessing, TorchRatingModel
from .split import PARTITIONS
from .training_contract import TrainingResult, validate_feature_table

TRAINING_LOCK = Lock()


class TorchModelTrainer:
    """Fit independent pooled/category MLPs and select epochs on validation MAE."""

    def __init__(
        self,
        config: NeuralTrainingConfig | None = None,
        feature_config: FeatureConfig | None = None,
    ):
        """Prepare settings without training or modifying feature data.

        Args:
            config: Neural architecture, weighting, optimizer and stopping settings.
            feature_config: Feature schema used by the saved dataset.

        Returns:
            None.
        """
        self.config = config or NeuralTrainingConfig()
        self.schema = feature_schema((feature_config or FeatureConfig()).schema_version)

    def _epoch(
        self, model: TorchRatingModel, loader: DataLoader, optimizer, device
    ) -> float:
        """Train one shuffled epoch using sample-weighted normalized Huber loss.

        Args:
            model: Current candidate network with fitted preprocessing.
            loader: Training-only feature, label and weight batches.
            optimizer: AdamW optimizer retained between epochs.
            device: Selected CPU or CUDA execution device.

        Returns:
            Mean normalized training loss for progress and training history.
        """
        model.network.train()
        criterion = nn.HuberLoss(reduction="none")
        weighted_loss = 0.0
        total_weight = 0.0

        for inputs, labels, weights in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            weights = weights.to(device)

            optimizer.zero_grad(set_to_none=True)
            predictions = model.network(inputs).squeeze(1)
            batch_losses = criterion(predictions, labels)
            loss = (batch_losses * weights).sum() / weights.sum()

            if not torch.isfinite(loss):
                raise ValueError(
                    "Nonfinite neural training loss; inspect features and hyperparameters"
                )

            loss.backward()
            optimizer.step()

            weighted_loss += float((batch_losses.detach() * weights).sum().cpu())
            total_weight += float(weights.sum().cpu())

        return weighted_loss / total_weight

    def _fit_group(self, frame: pd.DataFrame, device) -> tuple[TorchRatingModel, dict]:
        """Select one network using validation rows without inspecting test labels.

        Args:
            frame: One pooled/category group with fixed train/validation/test partitions.
            device: Selected execution device.

        Returns:
            Best-epoch network and evaluation/history report in original rating units.
        """
        partitions = {
            name: frame.loc[frame["partition"] == name] for name in PARTITIONS
        }
        for name, rows in partitions.items():
            if (
                len(rows) < self.config.min_rows
                or rows["target_username"].nunique() < 2
            ):
                raise ValueError(
                    f"Insufficient {name} rows/players for neural training"
                )

        train = partitions["train"]
        validation = partitions["validation"]
        features = train.loc[:, list(self.schema.names)].to_numpy(dtype=float)
        labels = train["rating"].to_numpy(dtype=float)

        preprocessing = NeuralPreprocessing.fit(features, labels)
        model = TorchRatingModel(self.schema.names, self.config, preprocessing)
        model.network.to(device)

        normalized_labels = (
            (labels - preprocessing.target_mean) / preprocessing.target_scale
        ).astype(np.float32)
        training_weights = sample_weights(
            train, self.config.balance_bands, self.config.balance_players
        ).astype(np.float32)
        dataset = TensorDataset(
            torch.from_numpy(preprocessing.transform(features)),
            torch.from_numpy(normalized_labels),
            torch.from_numpy(training_weights),
        )
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(self.config.seed),
            num_workers=0,
        )
        optimizer = torch.optim.AdamW(
            model.network.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        best_validation_mae = float("inf")
        best_weights = None
        epochs_without_improvement = 0
        history = []

        with tqdm(
            total=self.config.max_epochs,
            desc="PyTorch training",
            unit="epoch",
            disable=not self.config.progress,
        ) as bar:
            for epoch in range(1, self.config.max_epochs + 1):
                loss = self._epoch(model, loader, optimizer, device)
                predictions = model.predict(validation.loc[:, list(self.schema.names)])
                validation_mae = float(
                    np.abs(predictions - validation["rating"].to_numpy()).mean()
                )
                history.append(
                    {"epoch": epoch, "loss": loss, "validation_mae": validation_mae}
                )

                if validation_mae < best_validation_mae:
                    best_validation_mae = validation_mae
                    model.best_epoch = epoch
                    # Clone weights so later optimizer steps cannot change the checkpoint.
                    best_weights = {
                        name: value.detach().cpu().clone()
                        for name, value in model.network.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                bar.set_postfix(
                    validation_mae=f"{validation_mae:.1f}",
                    best_mae=f"{best_validation_mae:.1f}",
                    refresh=False,
                )
                bar.update(1)

                if epochs_without_improvement >= self.config.patience:
                    break

        if best_weights is None:
            raise ValueError("Neural training produced no usable validation checkpoint")

        model.network.load_state_dict(best_weights)
        model.network.to("cpu")
        model.network.eval()

        training_median = float(train["rating"].median())
        report = {
            "parameters": self.config.model_dump(mode="json"),
            "best_epoch": model.best_epoch,
            "best_validation_mae": best_validation_mae,
            "history": history,
            "partitions": {
                name: {
                    "pytorch": evaluate(
                        rows, model.predict(rows.loc[:, list(self.schema.names)])
                    ),
                    "train_median": evaluate(rows, np.full(len(rows), training_median)),
                }
                for name, rows in partitions.items()
            },
        }
        return model, report

    def fit(self, frame: pd.DataFrame) -> TrainingResult:
        """Train with preserved splits and return models ready for portable export.

        Args:
            frame: Existing feature dataset containing labels, identities and partitions.

        Returns:
            Models, original-scale evaluations and backend/reproducibility metadata.
        """
        frame = validate_feature_table(frame, self.schema.version)

        device_name = self.config.device
        if device_name == "auto":
            device_name = "cuda" if torch.cuda.is_available() else "cpu"
        if device_name == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but is unavailable; use device='cpu'")
        device = torch.device(device_name)

        groups = {"pooled": frame}
        if self.config.mode == "separate":
            groups = {
                name: frame.loc[frame["category"] == name]
                for name in sorted(frame["category"].unique())
            }
        models = {}
        reports = {}
        devices = [torch.cuda.current_device()] if device.type == "cuda" else []

        # Training must not change the caller's random state or CPU thread count.
        with TRAINING_LOCK, torch.random.fork_rng(devices=devices):
            previous_threads = torch.get_num_threads()
            torch.set_num_threads(self.config.threads)
            torch.manual_seed(self.config.seed)
            try:
                for name, group in groups.items():
                    models[name], reports[name] = self._fit_group(group, device)
            finally:
                torch.set_num_threads(previous_threads)

        return TrainingResult(
            models=models,
            report=reports,
            metadata={
                "model_backend": "pytorch",
                "training": self.config.model_dump(mode="json"),
                "target": "Chess.com game-time player rating",
                "split_policy": "target-player and game disjoint; opponents may overlap",
                "categories": sorted(frame["category"].unique()),
            },
        )
