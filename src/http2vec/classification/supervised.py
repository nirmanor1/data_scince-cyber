"""Supervised classifiers adapted to the :class:`ScoringModel` contract.

Each estimator is wrapped so that ``anomaly_score`` returns the model's
confidence for the anomaly class (label ``1``); higher always means more
anomalous, matching the unsupervised detectors and the project's evaluation
code. :func:`build_classifiers` returns *factories* (zero-arg callables) so that
cross-validation can build a fresh, unfitted model for every fold.
"""

from __future__ import annotations

import collections.abc
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from ..config import ClassifierConfig
from ..evaluation.metrics import compute_classification_metrics, fpr_at_tpr

if TYPE_CHECKING:
    from ..interfaces import ScoringModel

ANOMALY_LABEL = 1


class SupervisedClassifier:
    """Wrap an sklearn-style estimator as a :class:`ScoringModel`.

    When ``scale`` is true the estimator is placed behind a
    :class:`~sklearn.preprocessing.StandardScaler` inside a
    :class:`~sklearn.pipeline.Pipeline`; tree ensembles can opt out via
    ``scale=False``. ``anomaly_score`` prefers ``predict_proba`` for the anomaly
    class and falls back to ``decision_function`` (whose positive direction is
    the anomaly class for binary problems).
    """

    def __init__(self, name: str, estimator: Any, *, scale: bool = True) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.name = name
        self.scale = scale
        if scale:
            self._model: Any = Pipeline(
                [("scaler", StandardScaler()), ("estimator", estimator)]
            )
        else:
            self._model = estimator
        self._positive_index = ANOMALY_LABEL

    def fit(self, X: ArrayLike, y: ArrayLike | None = None) -> "SupervisedClassifier":
        """Fit the wrapped estimator and locate the anomaly-class column."""
        self._model.fit(np.asarray(X), np.asarray(y))
        classes = list(getattr(self._model, "classes_", [0, 1]))
        self._positive_index = (
            classes.index(ANOMALY_LABEL)
            if ANOMALY_LABEL in classes
            else len(classes) - 1
        )
        return self

    def predict(self, X: ArrayLike) -> np.ndarray:
        """Return hard 0/1 predictions."""
        return np.asarray(self._model.predict(np.asarray(X))).astype(int)

    def anomaly_score(self, X: ArrayLike) -> np.ndarray:
        """Return continuous anomaly scores (higher = more anomalous)."""
        X = np.asarray(X)
        if hasattr(self._model, "predict_proba"):
            proba = np.asarray(self._model.predict_proba(X))
            index = min(self._positive_index, proba.shape[1] - 1)
            return proba[:, index]
        return np.asarray(self._model.decision_function(X)).ravel()


def build_classifiers(
    config: ClassifierConfig,
) -> dict[str, collections.abc.Callable[[], SupervisedClassifier]]:
    """Return named factories that each build a fresh :class:`SupervisedClassifier`.

    Returning callables (rather than instances) lets cross-validation
    re-instantiate an unfitted model per fold without leaking state between
    folds. The paper's "SVC with linear kernel" is realized here as
    :class:`~sklearn.svm.LinearSVC`, which has no ``predict_proba``, so its ROC
    scores come from ``decision_function``.
    """

    def logistic_regression() -> SupervisedClassifier:
        from sklearn.linear_model import LogisticRegression

        return SupervisedClassifier(
            "logistic_regression",
            LogisticRegression(
                max_iter=config.lr_max_iter, random_state=config.random_state
            ),
            scale=config.scale_features,
        )

    def random_forest() -> SupervisedClassifier:
        from sklearn.ensemble import RandomForestClassifier

        return SupervisedClassifier(
            "random_forest",
            RandomForestClassifier(
                n_estimators=config.rf_n_estimators,
                random_state=config.random_state,
                n_jobs=-1,
            ),
            scale=False,
        )

    def linear_svc() -> SupervisedClassifier:
        from sklearn.svm import LinearSVC

        return SupervisedClassifier(
            "linear_svc",
            LinearSVC(
                C=config.svc_c,
                random_state=config.random_state,
                max_iter=config.svc_max_iter,
            ),
            scale=config.scale_features,
        )

    def gradient_boosting() -> SupervisedClassifier:
        # HistGradientBoostingClassifier is the histogram-based, vectorised
        # gradient-boosting implementation; it is dramatically faster than
        # GradientBoostingClassifier on the high-dimensional (3072-d) embeddings
        # while giving comparable accuracy. ``gb_n_estimators`` maps to its
        # ``max_iter`` (number of boosting iterations).
        from sklearn.ensemble import HistGradientBoostingClassifier

        return SupervisedClassifier(
            "gradient_boosting",
            HistGradientBoostingClassifier(
                max_iter=config.gb_n_estimators, random_state=config.random_state
            ),
            scale=False,
        )

    def knn() -> SupervisedClassifier:
        from sklearn.neighbors import KNeighborsClassifier

        return SupervisedClassifier(
            "knn",
            KNeighborsClassifier(n_neighbors=config.knn_n_neighbors, n_jobs=-1),
            scale=config.scale_features,
        )

    return {
        "logistic_regression": logistic_regression,
        "random_forest": random_forest,
        "linear_svc": linear_svc,
        "gradient_boosting": gradient_boosting,
        "knn": knn,
    }


def cross_validate(
    factory: collections.abc.Callable[[], "ScoringModel"],
    X: ArrayLike,
    y: ArrayLike,
    *,
    cv_folds: int,
    seed: int,
    beta: float = 2.0,
    positive_label: int = 1,
    tpr_targets: collections.abc.Sequence[float] = (0.90, 0.99),
) -> list[dict]:
    """Stratified k-fold cross-validation returning one metric dict per fold.

    A fresh model is built from ``factory`` for every fold, fitted on the train
    split and scored on the test split. Each dict adds ``fpr_at_90`` and
    ``fpr_at_99`` to the metrics from
    :func:`http2vec.evaluation.metrics.compute_classification_metrics`.
    """
    from sklearn.model_selection import StratifiedKFold

    X = np.asarray(X)
    y = np.asarray(y)
    splitter = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)

    fold_metrics: list[dict] = []
    for train_index, test_index in splitter.split(X, y):
        model = factory()
        model.fit(X[train_index], y[train_index])

        y_test = y[test_index]
        y_pred = model.predict(X[test_index])
        y_score = model.anomaly_score(X[test_index])

        metrics = compute_classification_metrics(
            y_test, y_pred, y_score, beta=beta, positive_label=positive_label
        )
        for target in tpr_targets:
            metrics[f"fpr_at_{int(round(target * 100))}"] = fpr_at_tpr(
                y_test, y_score, target, positive_label=positive_label
            )
        fold_metrics.append(metrics)

    return fold_metrics
