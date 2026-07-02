# HTTP2vec - Reproduction

Reproduction and critical study of:

> **HTTP2vec: Embedding of HTTP Requests for Detection of Anomalous Traffic**
> Gniewkowski, Maciejewski, Surmacz, Walentynowicz (2021). arXiv:2108.01763

The method embeds raw HTTP requests with a RoBERTa language model trained only
on legitimate traffic, then classifies the embeddings as normal or anomalous.
Dataset: **CSIC 2010 HTTP**.

## Contents

```text
README.md                  this file
http2vec_analysis.ipynb    analysis notebook (data -> EDA -> embeddings -> models)
Report.pdf                 final report
src/http2vec/              pipeline package (tokenizer, RoBERTa MLM, embedder, classifiers)
scripts/download_data.py   CSIC 2010 download helper
data/                      see data/README.md (raw files not committed)
artifacts/                 trained tokenizer + RoBERTa MLM checkpoint
```

## Setup

```bash
pip install torch transformers tokenizers accelerate safetensors \
            scikit-learn numpy pandas matplotlib seaborn nbformat tqdm
```

The notebook adds `src/` to the path automatically. GPU (CUDA) is recommended;
the code otherwise falls back to Apple MPS or CPU.

## Data

Not committed. Place the three CSIC 2010 text files in `data/raw/`:

```text
data/raw/normalTrafficTraining.txt
data/raw/normalTrafficTest.txt
data/raw/anomalousTrafficTest.txt
```

Get them from the [official CSIC source](https://www.tic.itefi.csic.es/dataset/)
or run `python scripts/download_data.py --dest data/raw`. See
[`data/README.md`](data/README.md) for details.

## Run

```bash
jupyter lab http2vec_analysis.ipynb
```

The notebook defaults to the `small` profile (full paper-size RoBERTa, seeded
40% subset, 5 epochs); switch the `PROFILE` cell to `"paper"` for the full run.

## Report

Final report: `Report.pdf`.
