"""Abstract contracts shared across the package.

Keeping the contracts in one place lets each layer depend on *abstractions*
rather than concrete implementations (Dependency Inversion), so the data,
modelling and classification layers can evolve independently.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .data.schemas import DatasetBundle, HttpRequest


class AbstractDatasetLoader(ABC):
    """Loads raw data from disk into a :class:`DatasetBundle`.

    Subclasses implement a single dataset format; adding a new dataset means
    adding a new subclass, never editing existing ones (Open/Closed).
    """

    @abstractmethod
    def load(self) -> DatasetBundle:
        """Read the dataset and return its three views."""
        raise NotImplementedError


@runtime_checkable
class Embedder(Protocol):
    """Turns HTTP requests into fixed-length numeric vectors."""

    @property
    def embedding_dim(self) -> int:
        """Dimensionality of the produced vectors."""

    def embed(self, requests: Sequence[HttpRequest]) -> np.ndarray:
        """Return an ``(len(requests), embedding_dim)`` float array."""


@runtime_checkable
class ScoringModel(Protocol):
    """Uniform interface for supervised classifiers and anomaly detectors.

    Conventions (shared by every implementation):

    - The positive class is the anomaly, encoded as ``1`` (normal is ``0``).
    - :meth:`predict` returns hard 0/1 labels.
    - :meth:`anomaly_score` returns a continuous score where *higher means more
      anomalous*. This is what ROC-AUC / FPR-at-TPR are computed from, so both
      supervised and unsupervised models can be evaluated the same way.
    """

    name: str

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "ScoringModel":
        """Fit the model. ``y`` is required for supervised models, ignored otherwise."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard 0/1 predictions."""

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return continuous anomaly scores (higher = more anomalous)."""
