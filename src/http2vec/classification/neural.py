"""Trainable MLP classifier head on top of frozen RoBERTa embeddings.

This is an extension *beyond the paper*: instead of only classifying the frozen
embeddings with classic scikit-learn models, we train a small multi-layer
perceptron directly on the embedding vectors. It is **not** end-to-end
fine-tuning of RoBERTa - the embeddings stay frozen - but a lightweight,
"fine-tune-like" supervised head that plugs into the same
:class:`~http2vec.interfaces.ScoringModel` contract and, because it is cheap,
exposes a per-epoch training/validation learning curve for inspection.

``torch`` is imported at module load (the head subclasses nothing but uses tensors
throughout); this module is only imported on demand from the pipeline so light
usage of the package never pays the import cost.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import nn

from ..config import MlpHeadConfig
from ..utils import logger, set_seed

ANOMALY_LABEL = 1


class _Mlp(nn.Module):
    """Two-layer perceptron: ``Linear -> ReLU -> Dropout -> Linear`` (2 logits)."""

    def __init__(self, in_features: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 2),
        )

    def forward(self, inputs: "torch.Tensor") -> "torch.Tensor":
        return self.net(inputs)


class MlpClassifierHead:
    """A small PyTorch MLP over embeddings, exposed as a :class:`ScoringModel`.

    Standardizes inputs, trains with Adam on cross-entropy, and records a
    per-epoch learning curve (``train_loss``, ``val_loss``, ``val_f1``) in
    :attr:`history`. ``anomaly_score`` returns the softmax probability of the
    anomaly class (higher = more anomalous), matching the other models. A seeded
    internal validation slice (held out of training) drives the curve; the model
    is never trained on its own validation data.
    """

    def __init__(self, config: MlpHeadConfig, *, device: str = "cpu") -> None:
        self.name = "mlp_head"
        self._config = config
        self._device = device
        self._scaler: Any | None = None
        self._model: _Mlp | None = None
        self.history: list[dict] = []

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "MlpClassifierHead":
        """Train the head on embeddings ``X`` with integer labels ``y``."""
        from sklearn.preprocessing import StandardScaler

        if y is None:
            raise ValueError("MlpClassifierHead.fit requires labels y.")

        set_seed(self._config.random_state)
        features = np.asarray(X, dtype=np.float32)
        labels = np.asarray(y).astype(np.int64)

        self._scaler = StandardScaler().fit(features)
        features = self._scaler.transform(features).astype(np.float32)

        x_train, x_val, y_train, y_val = _stratified_validation_split(
            features, labels, self._config.val_fraction, self._config.random_state
        )

        model = _Mlp(
            features.shape[1], self._config.hidden_size, self._config.dropout
        ).to(self._device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self._config.learning_rate,
            weight_decay=self._config.weight_decay,
        )
        criterion = nn.CrossEntropyLoss()

        train_x = torch.from_numpy(x_train).to(self._device)
        train_y = torch.from_numpy(y_train).to(self._device)
        val_x = torch.from_numpy(x_val).to(self._device)

        n_train = train_x.shape[0]
        batch_size = max(1, int(self._config.batch_size))
        rng = np.random.default_rng(self._config.random_state)

        patience = max(1, int(self._config.patience))
        best_val_loss = float("inf")
        best_state: dict | None = None
        epochs_without_improvement = 0

        history: list[dict] = []
        for epoch in range(1, int(self._config.epochs) + 1):
            model.train()
            order = rng.permutation(n_train)
            running_loss = 0.0
            for start in range(0, n_train, batch_size):
                batch_index = order[start : start + batch_size]
                batch_x = train_x[batch_index]
                batch_y = train_y[batch_index]
                optimizer.zero_grad()
                loss = criterion(model(batch_x), batch_y)
                loss.backward()
                optimizer.step()
                running_loss += float(loss.item()) * len(batch_index)

            record: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": running_loss / max(n_train, 1),
            }
            if val_x.shape[0] > 0:
                model.eval()
                with torch.no_grad():
                    val_logits = model(val_x)
                    val_loss = float(
                        criterion(
                            val_logits,
                            torch.from_numpy(y_val).to(self._device),
                        ).item()
                    )
                    val_pred = torch.argmax(val_logits, dim=1).cpu().numpy()
                record["val_loss"] = val_loss
                record["val_f1"] = _binary_f1(y_val, val_pred)

                # Early stopping: snapshot the best-validation weights and stop if
                # the validation loss has not improved for ``patience`` epochs, so
                # the returned model is the best epoch rather than a noisy late one.
                if val_loss < best_val_loss - 1e-4:
                    best_val_loss = val_loss
                    best_state = {
                        key: value.detach().cpu().clone()
                        for key, value in model.state_dict().items()
                    }
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
            history.append(record)

            if val_x.shape[0] > 0 and epochs_without_improvement >= patience:
                break

        # Restore the best-validation weights (if any) before scoring.
        if best_state is not None:
            model.load_state_dict(best_state)

        self._model = model
        self.history = history
        if history:
            if best_state is not None:
                logger.info(
                    "Trained MLP head: %d epoch(s) run, best val_loss=%.4f (restored).",
                    len(history),
                    best_val_loss,
                )
            else:
                logger.info(
                    "Trained MLP head: %d epoch(s), final train_loss=%.4f.",
                    len(history),
                    history[-1]["train_loss"],
                )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return hard 0/1 predictions."""
        return np.argmax(self._logits(X), axis=1).astype(int)

    def anomaly_score(self, X: ArrayLike) -> np.ndarray:
        """Return the softmax probability of the anomaly class (higher = worse)."""
        logits = self._logits(X)
        shifted = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(shifted)
        probabilities = exp / exp.sum(axis=1, keepdims=True)
        index = min(ANOMALY_LABEL, probabilities.shape[1] - 1)
        return probabilities[:, index]

    def _logits(self, X: ArrayLike) -> np.ndarray:
        if self._model is None or self._scaler is None:
            raise RuntimeError("MlpClassifierHead must be fitted before use.")
        features = self._scaler.transform(np.asarray(X, dtype=np.float32)).astype(
            np.float32
        )
        self._model.eval()
        with torch.no_grad():
            logits = self._model(torch.from_numpy(features).to(self._device))
        return logits.cpu().numpy()


def _stratified_validation_split(
    features: np.ndarray, labels: np.ndarray, fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/val split; returns an empty val set when it is not feasible."""
    from sklearn.model_selection import train_test_split

    empty_x = features[:0]
    empty_y = labels[:0]
    if fraction <= 0.0:
        return features, empty_x, labels, empty_y
    _, counts = np.unique(labels, return_counts=True)
    if counts.size < 2 or counts.min() < 2:
        return features, empty_x, labels, empty_y
    try:
        x_train, x_val, y_train, y_val = train_test_split(
            features,
            labels,
            test_size=fraction,
            random_state=seed,
            stratify=labels,
        )
        return x_train, x_val, y_train, y_val
    except ValueError:
        return features, empty_x, labels, empty_y


def _binary_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """F1 for the anomaly class, robust to empty predictions."""
    from sklearn.metrics import f1_score

    return float(f1_score(y_true, y_pred, pos_label=ANOMALY_LABEL, zero_division=0))
