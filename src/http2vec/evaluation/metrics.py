"""Pure, stateless metric functions for HTTP request classification.

These functions operate only on NumPy arrays and scikit-learn metrics so that
the same code evaluates supervised classifiers and unsupervised anomaly
detectors. They never plot, log payloads, or hold state.

Conventions (shared with :class:`http2vec.interfaces.ScoringModel`): the
positive class is the anomaly (label ``1``); ``y_score`` is a continuous score
where *higher means more anomalous*, which is what ROC-AUC and FPR-at-TPR use.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    fbeta_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def compute_classification_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_score: ArrayLike | None = None,
    *,
    beta: float = 2.0,
    positive_label: int = 1,
) -> dict[str, float]:
    """Compute the standard classification metrics for a single split.

    ``roc_auc`` is included only when ``y_score`` is provided and both classes
    are present (it is undefined otherwise). All threshold metrics use
    ``zero_division=0`` so empty predictions yield ``0`` rather than raising.

    Returns a dict with ``accuracy``, ``precision``, ``recall``, ``f1``,
    ``fbeta``, ``mcc``, the confusion-matrix counts ``tn``/``fp``/``fn``/``tp``
    and, when available, ``roc_auc``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    negative_label = 0 if positive_label != 0 else 1
    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[negative_label, positive_label]
    ).ravel()

    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, pos_label=positive_label, zero_division=0)
        ),
        "f1": float(f1_score(y_true, y_pred, pos_label=positive_label, zero_division=0)),
        "fbeta": float(
            fbeta_score(
                y_true, y_pred, beta=beta, pos_label=positive_label, zero_division=0
            )
        ),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    if y_score is not None:
        y_true_positive = y_true == positive_label
        if y_true_positive.min() != y_true_positive.max():
            metrics["roc_auc"] = float(
                roc_auc_score(y_true_positive, np.asarray(y_score))
            )
        else:
            metrics["roc_auc"] = float("nan")

    return metrics


def fpr_at_tpr(
    y_true: ArrayLike,
    y_score: ArrayLike,
    tpr_target: float,
    *,
    positive_label: int = 1,
) -> float:
    """Return the smallest FPR among ROC points whose TPR reaches ``tpr_target``.

    Reproduces the paper's FPR90 / FPR99 columns. Returns ``1.0`` when the target
    TPR is never attained.
    """
    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=positive_label)
    reached = tpr >= tpr_target
    if not reached.any():
        return 1.0
    return float(fpr[reached].min())


def aggregate_cv(fold_metrics: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate per-fold metric dicts into ``{metric: {"mean", "std"}}``.

    Only finite numeric values contribute; missing keys, booleans and
    non-numeric/NaN values are ignored so partial folds aggregate gracefully.
    The standard deviation is the population std (``ddof=0``) over folds.
    """
    keys: set[str] = set()
    for fold in fold_metrics:
        keys.update(fold.keys())

    aggregated: dict[str, dict[str, float]] = {}
    for key in sorted(keys):
        values = [
            float(fold[key])
            for fold in fold_metrics
            if isinstance(fold.get(key), (int, float))
            and not isinstance(fold.get(key), bool)
            and math.isfinite(fold[key])
        ]
        if values:
            array = np.asarray(values, dtype=float)
            aggregated[key] = {"mean": float(array.mean()), "std": float(array.std())}
    return aggregated
