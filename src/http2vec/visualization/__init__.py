"""Visualization layer: t-SNE, ROC, confusion-matrix and distribution plots."""

from .plots import (
    plot_confusion_matrix,
    plot_correlation_heatmap,
    plot_feature_distributions,
    plot_model_comparison,
    plot_pca_explained_variance,
    plot_roc,
    plot_roc_comparison,
    plot_score_distribution,
    plot_training_curve,
    plot_tsne,
    save_figure,
)

__all__ = [
    "plot_confusion_matrix",
    "plot_correlation_heatmap",
    "plot_feature_distributions",
    "plot_model_comparison",
    "plot_pca_explained_variance",
    "plot_roc",
    "plot_roc_comparison",
    "plot_score_distribution",
    "plot_training_curve",
    "plot_tsne",
    "save_figure",
]
