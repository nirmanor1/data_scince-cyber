"""End-to-end HTTP2vec orchestration.

:class:`Http2VecPipeline` wires the independent layers together - data loading,
tokenizer training, RoBERTa MLM training, embedding, supervised classification
and unsupervised anomaly detection - behind a single ``run()`` call. Each step
is a small public method so callers (or tests) can override or reuse individual
stages, and the loader is injectable so a different dataset can be plugged in
without touching this module (Dependency Inversion).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

from .classification.anomaly import (
    IsolationForestDetector,
    LocalOutlierFactorDetector,
)
from .classification.supervised import build_classifiers, cross_validate
from .config import ExperimentConfig
from .data.features import feature_frame as build_feature_frame
from .data.loaders import Csic2010Loader
from .data.schemas import DatasetBundle
from .evaluation.metrics import (
    aggregate_cv,
    compute_classification_metrics,
    fpr_at_tpr,
)
from .interfaces import AbstractDatasetLoader, Embedder, ScoringModel
from .models.embedder import RobertaRequestEmbedder
from .models.language_model import train_mlm
from .tokenization.bbpe import train_bbpe_tokenizer
from .utils import configure_logging, logger, resolve_device, set_seed

if TYPE_CHECKING:  # pragma: no cover
    import pandas as pd


@dataclass
class EmbeddingSet:
    """Embeddings needed downstream."""

    inference_x: np.ndarray  # both classes, for supervised classification
    inference_y: np.ndarray  # labels for the inference set
    normal_train_x: np.ndarray  # normal-only, for fitting the anomaly detector


@dataclass
class SupervisedReport:
    """Cross-validated metrics for one supervised classifier."""

    name: str
    cv: dict[str, dict[str, float]]  # {metric: {"mean", "std"}}


@dataclass
class ClassificationReport:
    """Per-sample outputs and metrics for a single fitted model on a labelled set."""

    name: str
    metrics: dict[str, float]
    y_true: np.ndarray
    y_pred: np.ndarray
    y_score: np.ndarray


@dataclass
class PipelineResult:
    """Everything a caller (or the notebook) needs to report and visualize."""

    config: ExperimentConfig
    embeddings: EmbeddingSet
    feature_frame: "pd.DataFrame"
    training_history: list[dict]
    supervised_cv: dict[str, SupervisedReport]
    supervised_holdout: ClassificationReport
    supervised_holdouts: dict[str, ClassificationReport]
    anomaly: ClassificationReport
    anomaly_detectors: dict[str, ClassificationReport]
    mlp_head: ClassificationReport
    mlp_history: list[dict]
    holdout_test_indices: np.ndarray | None = None

    def summary(self) -> str:
        """Human-readable one-screen summary of the headline metrics."""
        lines = ["HTTP2vec pipeline results", "=" * 40]
        lines.append("Supervised classifiers (stratified k-fold CV on inference set):")
        for report in self.supervised_cv.values():
            lines.append(f"  {report.name}: {_format_cv(report.cv)}")
        holdout = self.supervised_holdout
        lines.append(
            f"Best supervised on holdout [{holdout.name}]: {_format_point(holdout.metrics)}"
        )
        lines.append(
            f"Isolation Forest (fit on normal only): {_format_point(self.anomaly.metrics)}"
        )
        lines.append(
            f"MLP head (trainable, frozen embeddings): {_format_point(self.mlp_head.metrics)}"
        )
        return "\n".join(lines)


def _format_cv(cv: dict[str, dict[str, float]]) -> str:
    parts = []
    for key in ("f1", "mcc", "roc_auc"):
        if key in cv:
            parts.append(f"{key}={cv[key]['mean']:.3f}+/-{cv[key]['std']:.3f}")
    return ", ".join(parts) if parts else "(no metrics)"


def _format_point(metrics: dict[str, float]) -> str:
    parts = []
    for key in ("f1", "mcc", "roc_auc"):
        if key in metrics and metrics[key] == metrics[key]:  # skip NaN
            parts.append(f"{key}={metrics[key]:.3f}")
    return ", ".join(parts) if parts else "(no metrics)"


_COMPARISON_METRICS = (
    "f1",
    "fbeta",
    "mcc",
    "roc_auc",
    "precision",
    "recall",
    "accuracy",
    "fpr_at_90",
    "fpr_at_99",
)


def comparison_frame(
    reports: dict[str, "ClassificationReport"],
    *,
    metrics: Sequence[str] = _COMPARISON_METRICS,
) -> "pd.DataFrame":
    """Assemble a model-by-metric table from a registry of classification reports.

    Rows are model names (the registry keys); columns are ``metrics`` pulled from
    each report's ``metrics`` dict (missing values become NaN). This is the single
    source the unified comparison table, bar chart and ROC overlay are built from,
    so any model added to the registry appears everywhere automatically.
    """
    import pandas as pd

    rows = []
    for name, report in reports.items():
        row: dict[str, object] = {"model": name}
        row.update(
            {metric: report.metrics.get(metric, float("nan")) for metric in metrics}
        )
        rows.append(row)
    frame = pd.DataFrame(rows)
    if "model" in frame.columns:
        frame = frame.set_index("model")
    return frame


def error_examples(
    report: "ClassificationReport",
    test_indices: Sequence[int],
    requests: Sequence,
    *,
    k: int = 5,
) -> dict:
    """Map a holdout report's false positives / negatives back to request lines.

    ``report`` must have been scored on the rows given by ``test_indices`` (the
    shared holdout), so row ``j`` of the report corresponds to
    ``requests[test_indices[j]]``. Returns up to ``k`` example request lines for
    each error type plus the total counts. Request text is returned verbatim for
    inspection and is never executed or rendered.
    """
    y_true = np.asarray(report.y_true)
    y_pred = np.asarray(report.y_pred)
    indices = np.asarray(test_indices)
    fp_positions = np.where((y_pred == 1) & (y_true == 0))[0]
    fn_positions = np.where((y_pred == 0) & (y_true == 1))[0]

    def _lines(positions: np.ndarray) -> list[str]:
        return [requests[int(indices[p])].request_line for p in positions[:k]]

    return {
        "model": report.name,
        "n_false_positives": int(fp_positions.size),
        "n_false_negatives": int(fn_positions.size),
        "false_positives": _lines(fp_positions),
        "false_negatives": _lines(fn_positions),
    }


class Http2VecPipeline:
    """Run the full HTTP2vec method from raw data to evaluation."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        loader: AbstractDatasetLoader | None = None,
    ) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self._loader = loader if loader is not None else Csic2010Loader(config.data)
        # Populated by ``train_language_model``: the RoBERTa MLM ``Trainer`` log
        # history (per-step training loss + per-epoch eval loss) for plotting.
        self.training_history: list[dict] = []

    # -- individual stages (each independently reusable) --------------------

    def load(self) -> DatasetBundle:
        return self._loader.load()

    def build_tokenizer(self, bundle: DatasetBundle):
        """Train the byte-level BPE tokenizer on all traffic."""
        corpus = bundle.tokenizer_corpus.texts(
            first_line_only=self.config.data.first_line_only
        )
        return train_bbpe_tokenizer(
            corpus, self.config.tokenizer, self.config.tokenizer_dir
        )

    def train_language_model(self, bundle: DatasetBundle, tokenizer):
        """Train the RoBERTa MLM on normal traffic only.

        A small seeded slice of the normal traffic is held out as a validation
        set so the captured history includes an eval-loss-per-epoch curve. The
        history is stored on ``self.training_history`` and the trained model is
        returned (unchanged contract for existing callers).
        """
        texts = bundle.lm_train.texts(first_line_only=self.config.data.first_line_only)
        train_texts, eval_texts = _split_validation(
            texts, self.config.model.eval_fraction, self.config.seed
        )
        model, history = train_mlm(
            texts=train_texts,
            eval_texts=eval_texts,
            tokenizer=tokenizer,
            model_config=self.config.model,
            max_length=self.config.tokenizer.max_length,
            output_dir=self.config.model_dir,
            device=self.device,
            seed=self.config.seed,
        )
        self.training_history = history
        return model

    def build_embedder(self, model, tokenizer) -> Embedder:
        return RobertaRequestEmbedder(
            model,
            tokenizer,
            self.config.embedding,
            max_length=self.config.tokenizer.max_length,
            device=self.device,
            first_line_only=self.config.data.first_line_only,
        )

    def compute_embeddings(
        self, embedder: Embedder, bundle: DatasetBundle
    ) -> EmbeddingSet:
        logger.info("Embedding inference set (%d requests).", len(bundle.inference))
        inference_x = embedder.embed(bundle.inference.requests)
        logger.info("Embedding normal-train set (%d requests).", len(bundle.lm_train))
        normal_train_x = embedder.embed(bundle.lm_train.requests)
        return EmbeddingSet(
            inference_x=inference_x,
            inference_y=bundle.inference.labels,
            normal_train_x=normal_train_x,
        )

    def evaluate_supervised(
        self, embeddings: EmbeddingSet
    ) -> tuple[dict[str, SupervisedReport], ClassificationReport]:
        """Cross-validate every classifier and produce a holdout report for the best."""
        config = self.config
        x, y = embeddings.inference_x, embeddings.inference_y
        folds = _safe_folds(y, config.classifier.cv_folds)

        factories = build_classifiers(config.classifier)
        reports: dict[str, SupervisedReport] = {}
        for name, factory in factories.items():
            fold_metrics = cross_validate(
                factory,
                x,
                y,
                cv_folds=folds,
                seed=config.seed,
                beta=config.evaluation.f_beta,
                positive_label=config.evaluation.positive_label,
                tpr_targets=config.evaluation.tpr_targets,
            )
            reports[name] = SupervisedReport(name, aggregate_cv(fold_metrics))

        best_name = max(
            reports,
            key=lambda n: reports[n].cv.get("f1", {}).get("mean", float("-inf")),
        )
        holdout = self._holdout_report(best_name, factories[best_name], embeddings)
        return reports, holdout

    def evaluate_anomaly(self, embeddings: EmbeddingSet) -> ClassificationReport:
        """Fit Isolation Forest on normal embeddings, evaluate on the inference set.

        The Isolation Forest is fit on the same normal traffic used to train the
        language model (in-sample normals), a known, documented simplification.
        """
        detector = IsolationForestDetector(self.config.classifier)
        detector.fit(embeddings.normal_train_x)
        return self._score_report(
            detector, embeddings.inference_x, embeddings.inference_y
        )

    def _shared_holdout_indices(self, y) -> tuple[np.ndarray, np.ndarray]:
        """Seeded stratified train/test *row indices* for the inference set.

        These indices are the single source of the shared holdout: every model
        scored on a holdout (the embedding classifiers, the non-embedding
        baselines and the MLP head) and the error-analysis mapping all consume
        them, so all models are compared on identical test rows and each
        prediction can be traced back to its original request by position.
        """
        y = np.asarray(y)
        indices = np.arange(len(y))
        _, counts = np.unique(y, return_counts=True)
        if counts.size < 2 or counts.min() < 2:
            logger.warning(
                "Shared holdout fell back to using all rows for both train and "
                "test (input too small to stratify with both classes present); "
                "this only happens on toy inputs, never on real CSIC data."
            )
            return indices, indices
        from sklearn.model_selection import train_test_split

        train_idx, test_idx = train_test_split(
            indices,
            test_size=self.config.classifier.test_size,
            random_state=self.config.seed,
            stratify=y,
        )
        return train_idx, test_idx

    def shared_holdout_indices(
        self, embeddings: EmbeddingSet
    ) -> tuple[np.ndarray, np.ndarray]:
        """Public accessor for the shared holdout ``(train, test)`` row indices."""
        return self._shared_holdout_indices(embeddings.inference_y)

    def _shared_holdout(self, embeddings: EmbeddingSet):
        """One seeded stratified train/test split of the inference embeddings.

        Deterministic in the config seed, so every model evaluated through
        :meth:`evaluate_supervised_holdouts`, :meth:`evaluate_anomaly_detectors`
        and the MLP head is compared on exactly the *same* held-out test set.
        """
        train_idx, test_idx = self._shared_holdout_indices(embeddings.inference_y)
        x, y = embeddings.inference_x, embeddings.inference_y
        return x[train_idx], y[train_idx], x[test_idx], y[test_idx]

    def evaluate_supervised_holdouts(
        self, embeddings: EmbeddingSet
    ) -> dict[str, ClassificationReport]:
        """Fit every supervised classifier on the shared split; score the holdout.

        Unlike :meth:`evaluate_supervised` (which cross-validates and reports only
        the single best model on a holdout), this returns a report for *each*
        classifier on one common test set, enabling a fair head-to-head table and
        ROC overlay.
        """
        x_train, y_train, x_test, y_test = self._shared_holdout(embeddings)
        reports: dict[str, ClassificationReport] = {}
        for name, factory in build_classifiers(self.config.classifier).items():
            model = factory()
            model.fit(x_train, y_train)
            reports[name] = self._score_report(model, x_test, y_test)
        return reports

    def evaluate_anomaly_detectors(
        self, embeddings: EmbeddingSet
    ) -> dict[str, ClassificationReport]:
        """Fit each unsupervised detector on normal-only data; score the shared holdout.

        Detectors are fit on the normal traffic used for the language model and
        scored on the *same* held-out test set as the supervised classifiers, so
        every model in the unified comparison shares one evaluation set.
        """
        _, _, x_test, y_test = self._shared_holdout(embeddings)
        detectors = (
            IsolationForestDetector(self.config.classifier),
            LocalOutlierFactorDetector(self.config.classifier),
        )
        reports: dict[str, ClassificationReport] = {}
        for detector in detectors:
            detector.fit(embeddings.normal_train_x)
            reports[detector.name] = self._score_report(detector, x_test, y_test)
        return reports

    def evaluate_mlp_head(
        self, embeddings: EmbeddingSet
    ) -> tuple[ClassificationReport, list[dict]]:
        """Train the trainable MLP head and score it on the shared holdout.

        The head is trained on the same train split used by the supervised
        classifiers (carving its own internal validation slice for the learning
        curve) and scored on the same held-out test set, so it sits in the unified
        comparison on equal footing. Returns ``(report, learning_history)``.
        """
        from .classification.neural import MlpClassifierHead

        x_train, y_train, x_test, y_test = self._shared_holdout(embeddings)
        head = MlpClassifierHead(self.config.mlp_head, device=self.device)
        head.fit(x_train, y_train)
        report = self._score_report(head, x_test, y_test)
        return report, head.history

    def evaluate_descriptive_baseline(
        self, bundle: DatasetBundle, embeddings: EmbeddingSet
    ) -> dict[str, ClassificationReport]:
        """Handcrafted-feature baseline: classic models on the descriptive features.

        Fits the handcrafted classifiers on the human-readable descriptive
        features (report Section 3) and scores them on the shared holdout, so the
        result is directly comparable with the embedding models. This measures how
        much the RoBERTa embedding adds over cheap, hand-crafted features.
        """
        from .classification.baselines import build_descriptive_baselines

        frame = build_feature_frame(bundle.inference)
        x = frame.drop(columns=["label"])
        y = np.asarray(embeddings.inference_y)
        train_idx, test_idx = self._shared_holdout_indices(y)
        reports: dict[str, ClassificationReport] = {}
        for name, factory in build_descriptive_baselines(self.config.classifier).items():
            model = factory()
            model.fit(x.iloc[train_idx], y[train_idx])
            reports[name] = self._score_report(model, x.iloc[test_idx], y[test_idx])
        return reports

    def evaluate_lexical_baseline(
        self, bundle: DatasetBundle, embeddings: EmbeddingSet
    ) -> dict[str, ClassificationReport]:
        """Lexical baseline: char n-gram TF-IDF + linear models on raw request text.

        A classic bag-of-character-n-grams representation (the simple alternative
        the paper's introduction contrasts itself with), scored on the same shared
        holdout as every other model.
        """
        from .classification.baselines import build_lexical_baselines

        texts = bundle.inference.texts()
        y = np.asarray(embeddings.inference_y)
        train_idx, test_idx = self._shared_holdout_indices(y)
        texts_train = [texts[i] for i in train_idx]
        texts_test = [texts[i] for i in test_idx]
        reports: dict[str, ClassificationReport] = {}
        for name, factory in build_lexical_baselines(self.config.classifier).items():
            model = factory()
            model.fit(texts_train, y[train_idx])
            reports[name] = self._score_report(model, texts_test, y[test_idx])
        return reports

    def evaluate_dedup_best(
        self,
        bundle: DatasetBundle,
        embeddings: EmbeddingSet,
        *,
        model_name: str = "logistic_regression",
        keys: Sequence | None = None,
    ) -> ClassificationReport:
        """Refit one embedding classifier on a de-duplicated split.

        CSIC 2010 is heavily templated; comparing this report against the same
        model's full-data holdout report shows how much duplicate requests inflate
        the headline metrics. ``keys`` is the per-request sequence de-duplicated on
        (keep-first): pass e.g. the request first lines to remove URL-template
        repetition. Defaults to the full request text (exact-duplicate removal).
        """
        from .data.duplicates import unique_first_indices

        dedup_keys = keys if keys is not None else bundle.inference.texts()
        keep = unique_first_indices(dedup_keys)
        x = np.asarray(embeddings.inference_x)[keep]
        y = np.asarray(embeddings.inference_y)[keep]
        x_train, y_train, x_test, y_test = _safe_split(
            x, y, self.config.classifier.test_size, self.config.seed
        )
        factory = build_classifiers(self.config.classifier)[model_name]
        model = factory()
        model.fit(x_train, y_train)
        return self._score_report(model, x_test, y_test)

    def run(self) -> PipelineResult:
        """Execute every stage and return the assembled result."""
        configure_logging()
        set_seed(self.config.seed)

        bundle = self.load()
        tokenizer = self.build_tokenizer(bundle)
        model = self.train_language_model(bundle, tokenizer)
        embedder = self.build_embedder(model, tokenizer)
        embeddings = self.compute_embeddings(embedder, bundle)

        feature_frame = build_feature_frame(bundle.inference)
        supervised_cv, supervised_holdout = self.evaluate_supervised(embeddings)
        supervised_holdouts = self.evaluate_supervised_holdouts(embeddings)
        anomaly = self.evaluate_anomaly(embeddings)
        anomaly_detectors = self.evaluate_anomaly_detectors(embeddings)
        mlp_head, mlp_history = self.evaluate_mlp_head(embeddings)
        _, holdout_test_indices = self.shared_holdout_indices(embeddings)

        return PipelineResult(
            config=self.config,
            embeddings=embeddings,
            feature_frame=feature_frame,
            training_history=self.training_history,
            supervised_cv=supervised_cv,
            supervised_holdout=supervised_holdout,
            supervised_holdouts=supervised_holdouts,
            anomaly=anomaly,
            anomaly_detectors=anomaly_detectors,
            mlp_head=mlp_head,
            mlp_history=mlp_history,
            holdout_test_indices=holdout_test_indices,
        )

    # -- helpers -----------------------------------------------------------

    def _holdout_report(
        self,
        name: str,
        factory: Callable[[], ScoringModel],
        embeddings: EmbeddingSet,
    ) -> ClassificationReport:
        x_train, y_train, x_test, y_test = self._shared_holdout(embeddings)
        model = factory()
        model.fit(x_train, y_train)
        report = self._score_report(model, x_test, y_test)
        return ClassificationReport(
            name=name,
            metrics=report.metrics,
            y_true=report.y_true,
            y_pred=report.y_pred,
            y_score=report.y_score,
        )

    def _score_report(
        self, model: ScoringModel, x: np.ndarray, y: np.ndarray
    ) -> ClassificationReport:
        y_pred = model.predict(x)
        y_score = model.anomaly_score(x)
        metrics = compute_classification_metrics(
            y,
            y_pred,
            y_score,
            beta=self.config.evaluation.f_beta,
            positive_label=self.config.evaluation.positive_label,
        )
        for target in self.config.evaluation.tpr_targets:
            key = f"fpr_at_{int(round(target * 100))}"
            metrics[key] = fpr_at_tpr(
                y, y_score, target, positive_label=self.config.evaluation.positive_label
            )
        return ClassificationReport(
            name=getattr(model, "name", "model"),
            metrics=metrics,
            y_true=np.asarray(y),
            y_pred=np.asarray(y_pred),
            y_score=np.asarray(y_score),
        )


def _split_validation(
    texts: list[str], fraction: float, seed: int
) -> tuple[list[str], list[str]]:
    """Split ``texts`` into (train, eval) holding out a seeded random ``fraction``.

    Returns an empty eval list when ``fraction`` is non-positive or the corpus is
    too small to spare a validation slice, so MLM training still proceeds.
    """
    items = list(texts)
    n = len(items)
    if fraction <= 0.0 or n < 20:
        return items, []
    n_eval = max(1, int(round(n * fraction)))
    perm = np.random.default_rng(seed).permutation(n)
    eval_index = perm[:n_eval]
    eval_mask = np.zeros(n, dtype=bool)
    eval_mask[eval_index] = True
    train_texts = [items[i] for i in range(n) if not eval_mask[i]]
    eval_texts = [items[i] for i in eval_index]
    return train_texts, eval_texts


def _safe_folds(y: np.ndarray, requested: int) -> int:
    """Clamp the fold count to the smallest class size (>= 2)."""
    _, counts = np.unique(np.asarray(y), return_counts=True)
    smallest = int(counts.min()) if counts.size else 0
    if smallest < 2:
        raise ValueError(
            "Cross-validation needs at least two samples per class; "
            f"smallest class has {smallest}."
        )
    return max(2, min(requested, smallest))


def _safe_split(
    x: np.ndarray, y: np.ndarray, test_size: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified train/test split that degrades gracefully on tiny inputs.

    On data too small to split with both classes present (a tiny toy input), the
    same set is returned for train and test so the pipeline still runs; this is
    never hit on real CSIC data.
    """
    from sklearn.model_selection import train_test_split

    x = np.asarray(x)
    y = np.asarray(y)
    fallback_message = (
        "Holdout split fell back to using all data for both train and test "
        "because the input was too small to split with both classes present; "
        "this only happens on toy inputs, never on real CSIC data."
    )
    _, counts = np.unique(y, return_counts=True)
    if counts.size < 2 or counts.min() < 2:
        logger.warning(fallback_message)
        return x, y, x, y
    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, test_size=test_size, random_state=seed, stratify=y
        )
        return x_train, y_train, x_test, y_test
    except ValueError:
        logger.warning(fallback_message)
        return x, y, x, y
