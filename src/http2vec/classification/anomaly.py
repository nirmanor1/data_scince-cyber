"""Unsupervised anomaly detection on normal-only embeddings.

Following the paper's philosophy, every detector here is trained on *normal-only*
embeddings: anomalies are the points that look unlike that training distribution.
Each model exposes both outputs the project needs per request - a hard class
assignment and a continuous anomaly score where higher means more anomalous -
through the shared :class:`ScoringModel` contract.

:class:`IsolationForestDetector` (the paper's spirit) and
:class:`LocalOutlierFactorDetector` (an added comparison point) share the same
scaling/scoring plumbing via :class:`_NormalOnlyDetector` to avoid duplication.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from ..config import ClassifierConfig


class _NormalOnlyDetector:
    """Shared plumbing for scikit-learn novelty detectors fit on normal data.

    Subclasses pass an estimator that exposes ``fit``, ``predict`` (+1 inlier /
    -1 outlier) and ``score_samples`` (higher = more normal). When
    ``scale`` is true a :class:`~sklearn.preprocessing.StandardScaler` is fitted
    on the normal training data and applied before scoring. ``anomaly_score``
    negates ``score_samples`` so that, consistently with the supervised models,
    higher means more anomalous.
    """

    def __init__(self, name: str, estimator: Any, *, scale: bool) -> None:
        self.name = name
        self._estimator = estimator
        self._scale = scale
        self._scaler: Any | None = None

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "_NormalOnlyDetector":
        """Fit on normal-only embeddings ``X``; ``y`` is ignored by design."""
        from sklearn.preprocessing import StandardScaler

        features = np.asarray(X, dtype=float)
        if self._scale:
            self._scaler = StandardScaler().fit(features)
            features = self._scaler.transform(features)
        self._estimator.fit(features)
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Map detector output (+1 inlier / -1 outlier) to 0/1 labels."""
        raw = self._estimator.predict(self._transform(X))
        return (np.asarray(raw) == -1).astype(int)

    def anomaly_score(self, X: ArrayLike) -> np.ndarray:
        """Return continuous anomaly scores (higher = more anomalous)."""
        return -np.asarray(self._estimator.score_samples(self._transform(X)))

    def _transform(self, X: ArrayLike) -> np.ndarray:
        features = np.asarray(X, dtype=float)
        if self._scaler is not None:
            features = self._scaler.transform(features)
        return features


class IsolationForestDetector(_NormalOnlyDetector):
    """Isolation Forest adapted to the :class:`ScoringModel` contract.

    Anomalies are the points the forest isolates with abnormally short paths.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        from sklearn.ensemble import IsolationForest

        super().__init__(
            "isolation_forest",
            IsolationForest(
                n_estimators=config.iforest_n_estimators,
                contamination=config.iforest_contamination,
                max_samples=config.iforest_max_samples,
                random_state=config.random_state,
            ),
            scale=config.scale_features,
        )


class LocalOutlierFactorDetector(_NormalOnlyDetector):
    """Local Outlier Factor (novelty mode) adapted to :class:`ScoringModel`.

    Fit on normal-only embeddings with ``novelty=True`` so it can score unseen
    requests by comparing each point's local density to that of its neighbours;
    points in low-density regions relative to the normal manifold score as
    anomalies. Added as an unsupervised comparison point alongside the paper's
    Isolation Forest.
    """

    def __init__(self, config: ClassifierConfig) -> None:
        from sklearn.neighbors import LocalOutlierFactor

        super().__init__(
            "local_outlier_factor",
            LocalOutlierFactor(n_neighbors=config.lof_n_neighbors, novelty=True),
            scale=config.scale_features,
        )
