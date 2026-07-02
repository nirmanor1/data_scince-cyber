"""Build the final analysis notebook programmatically with ``nbformat``.

Keeping the notebook in code (instead of hand-editing JSON) makes it easy to
regenerate after the package changes and keeps the cells thin: every cell calls
into the ``http2vec`` package rather than re-implementing logic. Run with:

    python scripts/build_notebook.py            # writes notebooks/http2vec_analysis.ipynb
    python scripts/build_notebook.py --output other.ipynb
"""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

DEFAULT_OUTPUT = Path("notebooks/http2vec_analysis.ipynb")


def _cells() -> list:
    """Return the ordered list of notebook cells."""
    cells: list = []

    def md(text: str) -> None:
        cells.append(new_markdown_cell(text.strip("\n")))

    def code(text: str) -> None:
        cells.append(new_code_cell(text.strip("\n")))

    md(
        """
# HTTP2vec - Reproduction and Critical Study

Reproduction of **"HTTP2vec: Embedding of HTTP Requests for Detection of Anomalous
Traffic"** (Gniewkowski et al., 2021, arXiv:2108.01763) on the **CSIC 2010** dataset.

**Problem.** Most web attacks (SQL injection, XSS, CRLF, ...) travel over HTTP.
We want to flag anomalous HTTP requests. **Idea of the paper.** Treat an HTTP
request as text, learn a language model (RoBERTa) on *normal* traffic only, use
its hidden states to embed each request, then classify the embeddings.

This notebook walks through: data loading and inspection, EDA, the embedding
"feature engineering", model training (supervised classifiers **and** an
unsupervised Isolation Forest), evaluation, and error analysis. All logic lives
in the `http2vec` package; cells just orchestrate it.

> The token-removal "feature importance" analysis from the paper is intentionally
> out of scope here.
"""
    )

    md(
        """
## Colab bootstrap

When this notebook is opened directly from GitHub in Google Colab, only the
notebook file is present - the `http2vec` package and the CSIC2010 dataset are
not. This cell clones the repo, installs the package, and downloads the data. It
is a no-op when run locally (where the repo is already present), so the same
notebook works in both places.
"""
    )

    code(
        """
import sys, os, pathlib

IN_COLAB = "google.colab" in sys.modules
if IN_COLAB:
    os.chdir("/content")
    !rm -rf /content/cyberAno /cyberAno
    !git clone https://github.com/nirmanor1/cyberAno.git /content/cyberAno
    os.chdir("/content/cyberAno")
    !pip install -q -e . sentencepiece
    if not pathlib.Path("/content/cyberAno/data/raw/normalTrafficTraining.txt").exists():
        !python scripts/download_data.py --dest data/raw
    print("cwd is now:", os.getcwd())
else:
    print("Local run: skipping Colab bootstrap.")
"""
    )

    md("## 0. Setup")

    code(
        """
# Resolve the project root (so relative paths like data/raw work no matter where
# the notebook is launched from) and make the package importable (e.g. on Colab).
import os, sys, pathlib
root = pathlib.Path.cwd()
for _ in range(6):
    if (root / "pyproject.toml").exists() or (root / "data" / "raw").is_dir():
        break
    root = root.parent
os.chdir(root)
try:
    import http2vec  # noqa: F401
except ModuleNotFoundError:
    if (root / "src" / "http2vec").is_dir():
        sys.path.insert(0, str(root / "src"))
    import http2vec  # noqa: F401

%matplotlib inline
import numpy as np
import pandas as pd

from http2vec.config import ExperimentConfig
from http2vec.utils import configure_logging, set_seed, resolve_device
from http2vec.visualization.plots import save_figure

configure_logging()
print("http2vec", http2vec.__version__, "| device:", resolve_device("auto"))

# Every figure is saved here (in addition to being shown inline) so the plots can
# be reused directly in the written report. Use show(fig, name) throughout.
REPORTS_DIR = pathlib.Path("reports")
FIG_DIR = REPORTS_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def show(fig, name):
    # Save the figure into reports/figures and return it for inline display.
    save_figure(fig, name, FIG_DIR)
    return fig
"""
    )

    md(
        """
### Choose a configuration profile

The default here is the **`small`** profile: the full paper-size RoBERTa on a
seeded **60% subset** for **5 epochs** - a lighter stand-in for the full run.
Both profiles are heavy enough that a GPU (CUDA) is recommended.

- `small` - full paper-size model on a seeded 60% subset, 5 epochs (default here).
- `paper` - full dataset, paper-size RoBERTa, 10 epochs.
"""
    )

    code(
        """
PROFILE = "small"       # "small" (60% / 5ep, full model) | "paper" (full / 10ep)
EPOCHS_OVERRIDE = None  # None -> use the profile's default epochs; set an int to override

RAW_DIR = pathlib.Path("data/raw")
REQUIRED = ["normalTrafficTraining.txt", "normalTrafficTest.txt", "anomalousTrafficTest.txt"]
missing = [name for name in REQUIRED if not (RAW_DIR / name).exists()]
if missing:
    raise FileNotFoundError(
        "Missing CSIC2010 files in data/raw: " + ", ".join(missing)
        + ". Download them first: python scripts/download_data.py (see data/README.md)."
    )

kwargs = dict(raw_dir=str(RAW_DIR), device="auto")
if EPOCHS_OVERRIDE is not None:
    kwargs["num_train_epochs"] = EPOCHS_OVERRIDE
config = getattr(ExperimentConfig, PROFILE)(**kwargs)
set_seed(config.seed)
config
"""
    )

    md(
        """
## 1. Data loading and inspection

The loader returns three views (matching the paper): `lm_train` (normal-only, used
to train the language model), `inference` (labelled normal + anomalous, used for
classification) and `tokenizer_corpus` (all traffic, used to train the tokenizer).
"""
    )

    code(
        """
from http2vec.data.loaders import Csic2010Loader

bundle = Csic2010Loader(config.data).load()
print(f"lm_train (normal only):      {len(bundle.lm_train)}")
print(f"inference (normal+anomaly):  {len(bundle.inference)}")
print(f"tokenizer_corpus (all):      {len(bundle.tokenizer_corpus)}")

# Peek at one normal and one anomalous request (first line only, as text).
inf = bundle.inference.requests
example_normal = next(r for r in inf if int(r.label) == 0)
example_anomaly = next(r for r in inf if int(r.label) == 1)
print("\\nNormal  :", example_normal.request_line[:160])
print("Anomaly :", example_anomaly.request_line[:160])
"""
    )

    md(
        """
We turn each request into a small table of **descriptive features** (length,
number of parameters, %-encoding ratio, entropy, attack-signature flags, ...).
These are *not* the model features - they exist so we can do classical EDA.
"""
    )

    code(
        """
from http2vec.data.features import feature_frame

inference_df = feature_frame(bundle.inference)
print("shape:", inference_df.shape)
print("\\ndtypes:\\n", inference_df.dtypes)
inference_df.head()
"""
    )

    code(
        """
# Data-quality checks: missing values, duplicate rows, constant (single-value)
# columns, and duplicate (exactly redundant) feature columns.
print("Missing values per column:\\n", inference_df.isna().sum())
print("\\nFully duplicated rows:", int(inference_df.duplicated().sum()))
constant_cols = [c for c in inference_df.columns if inference_df[c].nunique(dropna=False) <= 1]
print("Constant columns:", constant_cols)
# Transpose so each former column becomes a row; duplicated() then flags columns
# whose values are identical to an earlier one (a vectorised redundancy check).
duplicate_feature_cols = inference_df.columns[inference_df.T.duplicated()].tolist()
print("Duplicate feature columns:", duplicate_feature_cols)
inference_df.describe(include="all").T
"""
    )

    md(
        """
**Columns and index - does this make sense?** The columns are descriptive,
model-agnostic attributes (lengths, counts, ratios, entropy, attack-signature
flags) plus the `label`; the names are self-explanatory and appropriate for EDA.
The frame uses a default positional integer index (one row per request): CSIC has
no natural key (no request id, no timestamp), so a positional index is the
sensible choice and conveniently aligns row *i* with
`bundle.inference.requests[i]` for error analysis later. The checks above confirm
no missing values and no exactly-duplicated feature columns; correlation-based
redundancy is examined in the EDA section.
"""
    )

    md(
        """
## 2. Exploratory Data Analysis (EDA)

### 2.1 Class balance (prevalence)

In the CSIC inference set the classes are roughly 59% normal / 41% anomalous
(the `small` subset profile keeps a seeded fraction of each file, so this ratio is
preserved). Accuracy is therefore sensitive to the imbalance - prefer F1, MCC and
ROC-AUC.
"""
    )

    code(
        """
counts = inference_df["label"].value_counts().rename({0: "normal", 1: "anomaly"})
shares = inference_df["label"].value_counts(normalize=True).rename({0: "normal", 1: "anomaly"})
pd.DataFrame({"count": counts, "share": shares.round(3)})
"""
    )

    md("### 2.2 Feature distributions by class")

    code(
        """
from http2vec.visualization.plots import plot_feature_distributions

dist_cols = ["target_length", "body_length", "n_query_params",
             "pct_encoding_ratio", "non_alnum_ratio", "shannon_entropy"]
fig = plot_feature_distributions(inference_df, dist_cols, hue="label")
show(fig, "2_2_feature_distributions")
"""
    )

    md(
        """
### 2.3 Correlation analysis

We report **Spearman** correlation as the primary measure: the descriptive
features are skewed, non-normal and contain outliers (attack payloads can be very
long), and we mostly care about *monotonic* association rather than strictly
linear association. For reference:

- **Pearson** - linear association between continuous variables; sensitive to
  outliers and assumes roughly linear, normal-ish data.
- **Spearman** - rank-based; captures monotonic relationships and is robust to
  outliers and non-normality (our case).
- **Kendall** - also rank-based; preferred for small samples / many ties, with a
  more conservative interpretation.
"""
    )

    code(
        """
from http2vec.visualization.plots import plot_correlation_heatmap

fig = plot_correlation_heatmap(inference_df, method="spearman")
show(fig, "2_3_correlation_spearman")
"""
    )

    code(
        """
# Cross-tabulate HTTP method against the label (group-by style view).
pd.crosstab(inference_df["method"], inference_df["label"].map({0: "normal", 1: "anomaly"}))
"""
    )

    md(
        """
### 2.4 Outliers and temporal note

`target_length` / `body_length` are heavy-tailed - long values are often attack
payloads, so "outliers" here are signal, not noise, and we keep them. The CSIC
dataset has **no timestamps**, so a temporal analysis is not applicable; this is a
limitation to note (concept drift cannot be studied with this data alone).
"""
    )

    code(
        """
inference_df[["target_length", "body_length", "shannon_entropy"]].describe(
    percentiles=[0.5, 0.9, 0.99]
).T
"""
    )

    md(
        """
## 3. Feature engineering: the HTTP2vec representation

The real features are RoBERTa embeddings. We:

1. train a byte-level BPE tokenizer on **all** traffic,
2. train a RoBERTa masked-language-model on **normal traffic only**,
3. embed each request as the mean (over its lines) of the concatenated last
   hidden layers.

These steps reuse the pipeline's stage methods so the notebook and the
programmatic `Http2VecPipeline.run()` share exactly one implementation.

> This trains the RoBERTa on the selected profile's normal traffic and embeds its
> requests - heavy, so a GPU is recommended. The default `small` profile uses the
> full-size model on a 60% subset for 5 epochs; `paper` is the full run.
"""
    )

    code(
        """
from http2vec.pipeline import Http2VecPipeline

pipeline = Http2VecPipeline(config, loader=Csic2010Loader(config.data))

tokenizer = pipeline.build_tokenizer(bundle)
model = pipeline.train_language_model(bundle, tokenizer)
embedder = pipeline.build_embedder(model, tokenizer)
print("tokenizer vocab:", tokenizer.vocab_size, "| embedding dim:", embedder.embedding_dim)
"""
    )

    md(
        """
### 3.1 Training progress (loss curve)

Because we train for only a few epochs, we capture the learning in detail: the
per-step **training loss** plus the **validation loss** evaluated once per epoch
on a seeded held-out slice of normal traffic (`ModelConfig.eval_fraction`). The
right-hand axis shows **perplexity** (`exp(loss)`). A loss that falls steadily
while the validation loss tracks it (rather than turning back up) indicates the
masked-language model is learning the structure of normal HTTP traffic without
obvious overfitting. The raw history is also saved to
`artifacts/roberta-mlm/training_log.json`.
"""
    )

    code(
        """
from http2vec.visualization.plots import plot_training_curve

fig = plot_training_curve(pipeline.training_history)
show(fig, "3_1_training_curve")
"""
    )

    code(
        """
embeddings = pipeline.compute_embeddings(embedder, bundle)
X = embeddings.inference_x
y = embeddings.inference_y
print("inference embeddings:", X.shape, "| normal-train embeddings:", embeddings.normal_train_x.shape)
"""
    )

    md(
        """
### 3.2 Visualising the embedding space (t-SNE)

If the representation is good, normal and anomalous requests should separate even
in 2-D. Scaling of embeddings for the classifiers is handled inside the
classifier wrappers (a `StandardScaler` for the linear models).
"""
    )

    code(
        """
from http2vec.visualization.plots import plot_tsne

# t-SNE scales poorly and gets cluttered with many points; a smaller random sample
# keeps the cluster structure while giving a cleaner plot. Lower for an even
# cleaner view, raise for more detail.
rng = np.random.default_rng(config.seed)
max_points = 3000
if len(y) > max_points:
    sel = rng.choice(len(y), size=max_points, replace=False)
    tsne_x, tsne_y = X[sel], y[sel]
    print(f"t-SNE on a random sample of {max_points} of {len(y)} requests.")
else:
    tsne_x, tsne_y = X, y
fig = plot_tsne(tsne_x, tsne_y, seed=config.seed)
show(fig, "3_2_tsne_embeddings")
"""
    )

    md(
        """
### 3.3 Feature engineering summary

Mapping our pipeline onto the standard feature-engineering checklist:

- **Encoding (categorical -> numeric).** Raw requests are text; the byte-level BPE
  tokenizer *is* the encoding step, turning bytes into token ids with no
  out-of-vocabulary problem. The descriptive `method` column is categorical and
  used only for EDA (the crosstab above); the model never one-hot encodes it
  because the model features are the dense embeddings.
- **Feature creation.** Two sets: human-readable descriptive features (for EDA)
  and the real model features - RoBERTa embeddings, the concatenation of the last
  4 hidden layers, mean-pooled over tokens and averaged over a request's lines
  (3072-dim for the full model).
- **Scaling.** Embeddings are standardized (`StandardScaler`) inside the linear
  classifiers, the anomaly detectors and the MLP head; tree ensembles skip it
  (scale-invariant).
- **Feature selection / dimensionality reduction.** Individual embedding
  dimensions are not human-interpretable, so we do not prune them by hand; instead
  the PCA curve below quantifies how few components already capture most variance
  (intrinsic dimensionality), and t-SNE gives a 2-D view. This is the paper's
  noted limitation (a ~3000-dim representation) made concrete.
"""
    )

    code(
        """
from http2vec.visualization.plots import plot_pca_explained_variance

fig = plot_pca_explained_variance(X, seed=config.seed)
show(fig, "3_3_pca_explained_variance")
"""
    )

    md(
        """
## 4. Model training

We train the paper's supervised classifiers with **stratified k-fold
cross-validation** - Logistic Regression, Random Forest and linear SVC (the
paper's three) plus **Gradient Boosting** and **KNN** - and two unsupervised
detectors fitted on normal embeddings only: **Isolation Forest** (the paper's
spirit) and **Local Outlier Factor**. Section 4.1 adds a small trainable **MLP
head** (an extension beyond the paper). Every model exposes the same interface:
per request a continuous anomaly score and a hard class.
"""
    )

    code(
        """
supervised_cv, supervised_holdout = pipeline.evaluate_supervised(embeddings)
anomaly = pipeline.evaluate_anomaly(embeddings)

rows = []
metric_keys = ["f1", "fbeta", "mcc", "roc_auc", "precision", "recall", "accuracy", "fpr_at_90", "fpr_at_99"]
for name, report in supervised_cv.items():
    row = {"model": name}
    row.update({k: report.cv[k]["mean"] for k in metric_keys if k in report.cv})
    rows.append(row)
cv_comparison = pd.DataFrame(rows).set_index("model")
print("Supervised classifiers - stratified 5-fold cross-validated means (paper protocol):")
cv_comparison.round(3)
"""
    )

    md(
        """
### 4.1 Trainable MLP head (extension beyond the paper)

Beyond the paper's frozen-embedding + classic-classifier recipe, we train a small
**MLP head** directly on the (still frozen) RoBERTa embeddings. Unlike the
language model - which sees only normal traffic - this head is *supervised*: it
uses labels, like the other classifiers. It is cheap, so we run it for several
epochs and capture a detailed train/validation learning curve. This is a
"fine-tune-like" addition: the RoBERTa weights themselves are not updated, only a
classifier on top of the embeddings. It is scored on the same shared holdout as
every other model, so it joins the unified comparison below on equal footing.
"""
    )

    code(
        """
mlp_report, mlp_history = pipeline.evaluate_mlp_head(embeddings)
print("MLP head (shared holdout):",
      {k: round(v, 3) for k, v in mlp_report.metrics.items() if isinstance(v, float)})

fig = plot_training_curve(mlp_history, title="MLP head learning curve", show_perplexity=False)
show(fig, "4_1_mlp_learning_curve")
"""
    )

    md(
        """
### 4.2 Unified model comparison (shared holdout)

The cross-validation above follows the paper for the supervised models, but to
compare *every* model - supervised, unsupervised and the MLP head - on an equal
footing we use **one shared, seeded stratified holdout split**: each supervised
model and the MLP head are trained on the same train split and scored on the same
test split, while the unsupervised detectors are fit on normal-only embeddings
and scored on that same test set. All the views below are built from a single
registry of per-model reports, so adding a model makes it appear everywhere.
"""
    )

    code(
        """
from http2vec.pipeline import comparison_frame
from http2vec.visualization.plots import plot_model_comparison

# One registry: name -> ClassificationReport, all on the SAME holdout test set.
model_reports = {}
model_reports.update(pipeline.evaluate_supervised_holdouts(embeddings))
model_reports.update(pipeline.evaluate_anomaly_detectors(embeddings))
model_reports["mlp_head (fine-tune)"] = mlp_report  # trained in Section 4.1

comparison_all = comparison_frame(model_reports)
print("Unified comparison on the shared holdout test set:")
comparison_all.round(3)
"""
    )

    code(
        """
fig = plot_model_comparison(comparison_all, metrics=["f1", "fbeta", "mcc", "roc_auc"])
show(fig, "4_2_model_comparison_bars")
"""
    )

    md(
        """
## 5. Evaluation

### Metrics and why they matter in this domain

- **Precision** - of the requests we flag as attacks, how many really are. Low
  precision means many **false positives** (benign traffic blocked - operator
  fatigue).
- **Recall (TPR)** - of the real attacks, how many we catch. Low recall means
  **false negatives** (attacks slip through - the dangerous failure).
- **F1 / Fβ** - harmonic mean of precision and recall; we use **β=2** to weight
  recall higher, because missing an attack is usually costlier than a false alarm.
- **MCC** - balanced even under class imbalance; robust single-number summary.
- **ROC-AUC** - threshold-independent ranking quality of the anomaly score.
- **FPR@TPR (FPR90/FPR99)** - the paper's headline: how many false positives we
  must accept to catch 90% / 99% of attacks.
"""
    )

    md(
        """
### 5.1 ROC comparison (all models, shared holdout)

Every model from the registry overlaid on one ROC plot, so their threshold-free
ranking quality is directly comparable on the same test set.
"""
    )

    code(
        """
from http2vec.visualization.plots import plot_roc_comparison

roc_curves = {name: (report.y_true, report.y_score) for name, report in model_reports.items()}
fig = plot_roc_comparison(roc_curves)
show(fig, "5_1_roc_comparison")
"""
    )

    md(
        """
### 5.2 Best supervised model (holdout)

A closer look at the single best supervised model: its confusion matrix and ROC.
"""
    )

    code(
        """
from http2vec.visualization.plots import plot_roc, plot_confusion_matrix, plot_score_distribution

print(f"Best supervised model on holdout: {supervised_holdout.name}")
print({k: round(v, 3) for k, v in supervised_holdout.metrics.items() if isinstance(v, float)})
fig = plot_confusion_matrix(supervised_holdout.y_true, supervised_holdout.y_pred,
                            title=f"Confusion - {supervised_holdout.name} (holdout)")
show(fig, "5_2_confusion_best_supervised")
"""
    )

    code(
        """
fig = plot_roc(supervised_holdout.y_true, supervised_holdout.y_score,
               title=f"ROC - {supervised_holdout.name} (holdout)")
show(fig, "5_2_roc_best_supervised")
"""
    )

    md(
        """
### 5.3 Unsupervised detector (Isolation Forest)

Fitted on normal embeddings only, then asked to score the (mixed) inference set.
The two outputs per request are shown: the score distribution (degree of anomaly)
and the assigned class (confusion matrix).
"""
    )

    code(
        """
print("Isolation Forest metrics (full inference set):")
print({k: round(v, 3) for k, v in anomaly.metrics.items() if isinstance(v, float)})
fig = plot_score_distribution(anomaly.y_score, anomaly.y_true)
show(fig, "5_3_iforest_score_distribution")
"""
    )

    code(
        """
fig = plot_confusion_matrix(anomaly.y_true, anomaly.y_pred, title="Confusion - Isolation Forest")
show(fig, "5_3_confusion_iforest")
"""
    )

    code(
        """
# Persist the unified comparison so the written report can cite exact numbers.
comparison_all.round(4).to_csv(REPORTS_DIR / "model_comparison.csv")
print("Saved reports/model_comparison.csv")
print("Figures saved under", FIG_DIR)
"""
    )

    md(
        """
## 6. Error analysis

We inspect false positives (benign flagged as attack) and false negatives
(attacks missed) from the Isolation Forest, which scores the full inference set in
order, so we can map errors straight back to the original requests.
"""
    )

    code(
        """
inf_requests = bundle.inference.requests
y_true = anomaly.y_true
y_pred = anomaly.y_pred
fp_idx = np.where((y_pred == 1) & (y_true == 0))[0][:3]
fn_idx = np.where((y_pred == 0) & (y_true == 1))[0][:3]

print("False positives (benign flagged as attack):")
for i in fp_idx:
    print("  ", inf_requests[i].request_line[:160])
print("\\nFalse negatives (attacks missed):")
for i in fn_idx:
    print("  ", inf_requests[i].request_line[:160])

print("\\nFP count:", int(((y_pred == 1) & (y_true == 0)).sum()),
      "| FN count:", int(((y_pred == 0) & (y_true == 1)).sum()))
"""
    )

    md(
        """
**False positive vs false negative trade-off.** In intrusion detection a false
negative (a missed attack) is typically far more costly than a false positive (a
false alarm). That is why we emphasise recall (Fβ with β=2) and report FPR at high
TPR: we want to catch almost all attacks while keeping the false-alarm rate
operationally acceptable. The decision threshold on the anomaly score is the knob
that trades one for the other.
"""
    )

    md(
        """
## 7. Executive summary

The cell below assembles this run's headline numbers so the written report can
quote exact figures. It prints a compact recap and writes
`reports/results_summary.json`; the full per-model table is in
`reports/model_comparison.csv` and every figure is under `reports/figures/`.
"""
    )

    code(
        """
import json

best_overall = comparison_all["f1"].idxmax()
summary = {
    "profile": PROFILE,
    "config": {
        "epochs": config.model.num_train_epochs,
        "subset_fraction": config.data.subset_fraction,
        "embedding_dim": int(embedder.embedding_dim),
    },
    "class_balance": {"normal": int((y == 0).sum()), "anomaly": int((y == 1).sum())},
    "best_supervised_holdout": supervised_holdout.name,
    "best_model_overall_by_f1": best_overall,
    "metrics_by_model": {
        name: {k: round(float(v), 4)
               for k, v in report.metrics.items()
               if isinstance(v, float) and v == v}
        for name, report in model_reports.items()
    },
}
(REPORTS_DIR / "results_summary.json").write_text(json.dumps(summary, indent=2))
print("Best model overall (by F1):", best_overall)
print("Saved reports/results_summary.json and reports/model_comparison.csv")
comparison_all.round(3)
"""
    )

    md(
        """
**Recap.**

- **Problem:** detect anomalous/attack HTTP requests on CSIC 2010.
- **Method:** RoBERTa embeddings of requests (LM trained on normal traffic only)
  plus supervised classifiers, two unsupervised detectors (Isolation Forest, LOF),
  and a trainable MLP head over the frozen embeddings.
- **Findings:** the embedding space separates classes (t-SNE), supervised models
  reach strong F1/MCC, the MLP head adds a learnable comparison point, and the
  unsupervised detectors give a per-request anomaly score without training labels.
- **Caveats:** the numbers reflect the selected profile (default `small`: the
  full-size model on a 60% subset, 5 epochs); use `paper` for the full run.

A full critical evaluation (claims vs evidence, reproducibility, limitations) is
left for the written report; the saved JSON/CSV/figures back it with exact numbers.
"""
    )

    return cells


def build_notebook(output_path: Path) -> Path:
    """Assemble and write the notebook, returning the output path."""
    notebook = new_notebook()
    notebook.cells = _cells()
    notebook.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    nbformat.validate(notebook)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        nbformat.write(notebook, handle)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the HTTP2vec analysis notebook.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    path = build_notebook(args.output)
    print(f"Wrote notebook to {path}")


if __name__ == "__main__":
    main()
