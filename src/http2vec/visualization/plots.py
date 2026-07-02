"""Plotting helpers for the HTTP2vec analysis notebook.

Every function builds and *returns* a :class:`matplotlib.figure.Figure` and
never calls ``plt.show()`` or sets a backend, so the caller (notebook or script)
controls rendering. Heavy plotting/ML imports are deferred to call time to keep
importing the package cheap.

The anomaly class is shown in red throughout (label ``1``); normal is ``0``.
"""

from __future__ import annotations

import collections.abc
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from pathlib import Path

    import pandas
    from matplotlib.figure import Figure

_CORRELATION_METHODS = ("pearson", "spearman", "kendall")


def plot_training_curve(
    history: collections.abc.Sequence[dict],
    *,
    title: str = "RoBERTa MLM training progress",
    show_perplexity: bool = True,
) -> "Figure":
    """Plot a training loss curve and (when present) a validation loss curve.

    ``history`` may come from a HuggingFace ``Trainer`` (entries carry ``loss``
    and, per epoch, ``eval_loss``) or from the MLP head (each per-epoch entry
    carries ``train_loss`` and ``val_loss``); both layouts are handled. The
    x-axis uses ``step`` when available, otherwise ``epoch``. When
    ``show_perplexity`` is set a secondary y-axis shows perplexity (``exp(loss)``,
    meaningful for the MLM). An informative placeholder is returned when no loss
    was captured, instead of raising.
    """
    import matplotlib.pyplot as plt

    train_steps: list[float] = []
    train_loss: list[float] = []
    eval_steps: list[float] = []
    eval_loss: list[float] = []
    used_step = False
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("step") is not None:
            position = float(entry["step"])
            used_step = True
        elif entry.get("epoch") is not None:
            position = float(entry["epoch"])
        else:
            continue
        if "loss" in entry:
            train_steps.append(position)
            train_loss.append(float(entry["loss"]))
        elif "train_loss" in entry:
            train_steps.append(position)
            train_loss.append(float(entry["train_loss"]))
        if "eval_loss" in entry:
            eval_steps.append(position)
            eval_loss.append(float(entry["eval_loss"]))
        elif "val_loss" in entry:
            eval_steps.append(position)
            eval_loss.append(float(entry["val_loss"]))

    fig, ax = plt.subplots(figsize=(8, 5))
    if not train_loss and not eval_loss:
        ax.text(
            0.5,
            0.5,
            "No training history captured",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return fig

    if train_loss:
        ax.plot(
            train_steps,
            train_loss,
            color="tab:blue",
            lw=1.5,
            alpha=0.85,
            label="train loss",
        )
    if eval_loss:
        ax.plot(
            eval_steps,
            eval_loss,
            color="red",
            marker="o",
            lw=1.5,
            label="validation loss",
        )
    ax.set_xlabel("training step" if used_step else "epoch")
    ax.set_ylabel("loss")
    ax.set_title(title)
    ax.legend(loc="upper right")

    if show_perplexity:
        # Secondary axis: perplexity = exp(loss), aligned to the loss axis. The
        # transforms are clamped so they stay finite and monotonic for matplotlib.
        def _to_perplexity(loss: ArrayLike) -> np.ndarray:
            return np.exp(np.clip(loss, 0.0, 50.0))

        def _to_loss(perplexity: ArrayLike) -> np.ndarray:
            return np.log(np.clip(perplexity, 1e-9, None))

        secondary = ax.secondary_yaxis("right", functions=(_to_perplexity, _to_loss))
        secondary.set_ylabel("perplexity = exp(loss)")

    fig.tight_layout()
    return fig


def plot_tsne(
    embeddings: ArrayLike,
    labels: ArrayLike,
    *,
    seed: int = 42,
    perplexity: float = 30.0,
    title: str = "t-SNE of HTTP request embeddings",
) -> "Figure":
    """Project embeddings to 2-D with t-SNE and scatter normal vs anomaly."""
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    embeddings = np.asarray(embeddings, dtype=float)
    labels = np.asarray(labels)
    n = len(labels)
    if n < 3:
        fig, ax = plt.subplots(figsize=(7, 6))
        ax.text(
            0.5,
            0.5,
            "t-SNE needs >= 3 samples",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return fig
    # TSNE requires perplexity < n_samples.
    safe_perplexity = float(min(perplexity, n - 1))
    coords = TSNE(
        n_components=2, random_state=seed, perplexity=safe_perplexity
    ).fit_transform(embeddings)

    is_normal = labels == 0
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(
        coords[is_normal, 0],
        coords[is_normal, 1],
        s=14,
        c="tab:blue",
        alpha=0.6,
        label="normal",
    )
    ax.scatter(
        coords[~is_normal, 0],
        coords[~is_normal, 1],
        s=14,
        c="red",
        alpha=0.6,
        label="anomaly",
    )
    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()
    fig.tight_layout()
    return fig


def plot_roc(
    y_true: ArrayLike,
    y_score: ArrayLike,
    *,
    pos_label: int = 1,
    title: str = "ROC curve",
) -> "Figure":
    """Plot the ROC curve with its AUC and a chance-level diagonal baseline."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fpr, tpr, _ = roc_curve(y_true, y_score, pos_label=pos_label)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="tab:blue", lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    *,
    class_names: collections.abc.Sequence[str] = ("normal", "anomaly"),
    title: str = "Confusion matrix",
) -> "Figure":
    """Plot an annotated confusion-matrix heatmap with fixed 0/1 label order."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=list(class_names),
        yticklabels=list(class_names),
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_score_distribution(
    scores: ArrayLike,
    labels: ArrayLike,
    *,
    title: str = "Anomaly score distribution",
) -> "Figure":
    """Overlay per-class histograms (with KDE) of the continuous anomaly score."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)

    fig, ax = plt.subplots(figsize=(7, 5))
    for value, name, color in ((0, "normal", "tab:blue"), (1, "anomaly", "red")):
        subset = scores[labels == value]
        if subset.size == 0:
            continue
        use_kde = subset.size > 2 and float(np.ptp(subset)) > 0.0
        sns.histplot(
            subset,
            bins=30,
            stat="density",
            kde=use_kde,
            color=color,
            alpha=0.5,
            label=name,
            ax=ax,
        )
    ax.set_xlabel("Anomaly score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_feature_distributions(
    frame: "pandas.DataFrame",
    columns: collections.abc.Sequence[str],
    *,
    hue: str = "label",
) -> "Figure":
    """Small-multiples histograms of selected columns, split by ``hue`` if present."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    columns = list(columns)
    count = max(len(columns), 1)
    ncols = min(3, count)
    nrows = int(np.ceil(count / ncols))
    hue_column = hue if hue in frame.columns else None

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False
    )
    flat_axes = axes.ravel()
    for ax, column in zip(flat_axes, columns):
        sns.histplot(data=frame, x=column, hue=hue_column, bins=30, ax=ax)
        ax.set_title(str(column))
    for ax in flat_axes[len(columns):]:
        ax.set_visible(False)
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(
    frame: "pandas.DataFrame",
    *,
    method: str = "spearman",
    title: str | None = None,
) -> "Figure":
    """Plot an annotated correlation heatmap over numeric columns only."""
    import matplotlib.pyplot as plt
    import seaborn as sns

    if method not in _CORRELATION_METHODS:
        raise ValueError(
            f"method must be one of {_CORRELATION_METHODS}; got {method!r}."
        )

    correlation = frame.select_dtypes(include="number").corr(method=method)
    side = max(len(correlation.columns), 1)

    fig, ax = plt.subplots(figsize=(1.1 * side + 2, 1.0 * side + 2))
    sns.heatmap(
        correlation,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
        square=True,
        ax=ax,
    )
    ax.set_title(title or f"{method.capitalize()} correlation")
    fig.tight_layout()
    return fig


def plot_model_comparison(
    frame: "pandas.DataFrame",
    *,
    metrics: collections.abc.Sequence[str] = ("f1", "mcc", "roc_auc"),
    title: str = "Model comparison (shared holdout)",
) -> "Figure":
    """Grouped bar chart comparing selected ``metrics`` across models (frame rows).

    ``frame`` is a model-by-metric table (e.g. from
    :func:`http2vec.pipeline.comparison_frame`); the index holds model names.
    """
    import matplotlib.pyplot as plt

    available = [metric for metric in metrics if metric in frame.columns]
    fig, ax = plt.subplots(figsize=(max(7.0, 1.5 * max(len(frame.index), 1)), 5))
    if frame.empty or not available:
        ax.text(
            0.5,
            0.5,
            "No metrics to compare",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return fig

    models = list(frame.index)
    positions = np.arange(len(models))
    width = 0.8 / len(available)
    for index, metric in enumerate(available):
        offset = (index - (len(available) - 1) / 2) * width
        values = frame[metric].to_numpy(dtype=float)
        ax.bar(positions + offset, values, width, label=metric)
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_pca_explained_variance(
    embeddings: ArrayLike,
    *,
    n_components: int = 50,
    seed: int = 42,
    title: str = "PCA explained variance of the embedding space",
) -> "Figure":
    """Plot cumulative explained variance of the top principal components.

    Quantifies the intrinsic dimensionality of the high-dimensional embeddings -
    a feature-engineering / dimensionality-reduction diagnostic. Returns a
    placeholder when there is not enough data to fit a PCA.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    data = np.asarray(embeddings, dtype=float)
    fig, ax = plt.subplots(figsize=(7, 4))
    if data.ndim != 2 or min(data.shape) < 2:
        ax.text(
            0.5,
            0.5,
            "Not enough data for PCA",
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        ax.set_axis_off()
        ax.set_title(title)
        fig.tight_layout()
        return fig

    k = int(min(n_components, data.shape[0], data.shape[1]))
    pca = PCA(n_components=k, random_state=seed).fit(data)
    cumulative = np.cumsum(pca.explained_variance_ratio_)
    ax.plot(range(1, k + 1), cumulative, marker="o", ms=3, color="tab:blue")
    ax.axhline(0.9, color="grey", ls="--", lw=1, label="90% variance")
    ax.set_xlabel("number of principal components")
    ax.set_ylabel("cumulative explained variance")
    ax.set_ylim(0.0, 1.02)
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_roc_comparison(
    curves: "collections.abc.Mapping[str, tuple[ArrayLike, ArrayLike]]",
    *,
    pos_label: int = 1,
    title: str = "ROC comparison (shared holdout)",
) -> "Figure":
    """Overlay ROC curves for several models on one axes.

    ``curves`` maps a model name to a ``(y_true, y_score)`` pair, where
    ``y_score`` is the continuous anomaly score (higher = more anomalous). Models
    whose test split has a single class present are skipped (ROC is undefined).
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import auc, roc_curve

    fig, ax = plt.subplots(figsize=(7, 6))
    plotted = False
    for name, (y_true, y_score) in curves.items():
        y_true_array = np.asarray(y_true)
        if np.unique(y_true_array).size < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true_array, np.asarray(y_score), pos_label=pos_label)
        ax.plot(fpr, tpr, lw=1.6, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
        plotted = True
    ax.plot([0, 1], [0, 1], color="grey", lw=1, linestyle="--", label="chance")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(title)
    if plotted:
        ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


def save_figure(
    fig: "Figure",
    name: str,
    directory: "str | Path",
    *,
    fmt: str = "png",
    dpi: int = 150,
) -> "Path":
    """Save ``fig`` as ``directory/<sanitized name>.<fmt>`` and return the path.

    The directory is created if needed. ``name`` is sanitized to a filesystem-safe
    slug, so descriptive section labels can be passed verbatim from the notebook.
    """
    import re
    from pathlib import Path

    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_") or "figure"
    path = target_dir / f"{slug}.{fmt}"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path
