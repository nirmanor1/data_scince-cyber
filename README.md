# HTTP2vec - Reproduction

Reproduction and critical study of:

> **HTTP2vec: Embedding of HTTP Requests for Detection of Anomalous Traffic**
> Gniewkowski, Maciejewski, Surmacz, Walentynowicz (2021). arXiv:2108.01763

The method embeds raw HTTP requests with a RoBERTa language model (trained only on
legitimate traffic) and classifies the resulting vectors as normal or anomalous.

## What this repository contains

- A small, SOLID Python package (`src/http2vec`) implementing the full pipeline:
  byte-level BPE tokenizer -> RoBERTa masked-language-model -> request embeddings
  (mean of the last hidden layers, averaged over request lines) -> classifiers.
- Both the paper's **supervised** classifiers (Logistic Regression, Random Forest,
  linear SVC) and an added **unsupervised** anomaly detector (Isolation Forest)
  that outputs, per request, an *anomaly score* and an *assigned class*.
- A generated Jupyter notebook (`notebooks/http2vec_analysis.ipynb`) that walks
  through data loading, EDA, feature engineering, model training and evaluation.

## Project layout

```
src/http2vec/
  config.py            # experiment configuration + small/paper profiles
  interfaces.py        # abstract contracts shared across the package
  utils.py             # seeding, device resolution, logging
  data/                # parsing, loading, descriptive features
  tokenization/        # byte-level BPE tokenizer
  models/              # RoBERTa MLM training + request embedder
  classification/      # supervised classifiers + Isolation Forest detector
  evaluation/          # metrics
  visualization/       # plots (t-SNE, ROC, confusion matrix, ...)
  pipeline.py          # end-to-end orchestrator
scripts/
  download_data.py     # CSIC2010 download helper
  build_notebook.py    # builds notebooks/http2vec_analysis.ipynb
notebooks/             # generated notebook
data/                  # see data/README.md (raw files are not committed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate     # optional
pip install -e .            # or: pip install -r requirements.txt
```

GPU is optional; the code auto-detects CUDA, then Apple MPS, then falls back to CPU.

## Dataset

See [`data/README.md`](data/README.md). Place the three CSIC2010 text files in
`data/raw/` (or use `python scripts/download_data.py`).

## Usage

Configuration profiles (in `src/http2vec/config.py`):

- `ExperimentConfig.small()` - full paper-size model on a seeded 60% subset, 5 epochs (the default).
- `ExperimentConfig.paper()` - the hyper-parameters reported in the paper (full data, 10 epochs; GPU + hours).

Programmatic end-to-end run:

```python
from http2vec.config import ExperimentConfig
from http2vec.pipeline import Http2VecPipeline

config = ExperimentConfig.small()      # auto-detects CPU/GPU
pipeline = Http2VecPipeline(config)
results = pipeline.run()
print(results.summary())
```

`results` exposes, per request, both a continuous anomaly score and an assigned
class for the supervised classifiers and for the Isolation Forest detector.

### Notebook

Build (or rebuild) the analysis notebook after changing the package, then open it:

```bash
python scripts/build_notebook.py        # writes notebooks/http2vec_analysis.ipynb
```

The notebook reads the CSIC2010 files from `data/raw/`, so download the dataset
first (see the Dataset section). It defaults to the `small` profile (full-size
model, seeded 60% subset, 5 epochs); switch the `PROFILE` cell to `"paper"` for
the full run.

To execute it headlessly and save the outputs:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/http2vec_analysis.ipynb
```

Running the notebook also writes report-ready artifacts under `reports/`:

- `reports/figures/` - every figure as a PNG (MLM training curve, EDA plots,
  t-SNE, PCA variance, the model-comparison bar chart, the ROC overlay, confusion
  matrices, ...).
- `reports/model_comparison.csv` - the unified per-model metric table (all
  classifiers, the two anomaly detectors and the MLP head on one shared holdout).
- `reports/results_summary.json` - the run's headline numbers for the written report.

## Notes

- This is a local project (no git is required to use it).
- Attack payloads are always treated as opaque text and never executed.
