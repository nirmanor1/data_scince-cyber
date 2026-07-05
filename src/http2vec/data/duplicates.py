"""Exact-duplicate and train/test leakage analysis on raw request text.

Section 3.1 of the report measures duplicate *descriptive fingerprints* (~64% of
rows collapse to the same feature vector). This module measures a stricter,
complementary quantity: exact duplicates of the **raw request text**, and how
many held-out test requests are byte-for-byte copies of a training request
(train/test leakage), which makes reported performance optimistic. All functions
are pure and operate on plain strings; request text is never executed.
"""

from __future__ import annotations

import collections.abc

import numpy as np


def duplicate_report(texts: collections.abc.Sequence[str]) -> dict[str, float]:
    """Count exact-duplicate request texts.

    Returns totals plus the duplicate rate (fraction of rows that are a repeat of
    an earlier row). A rate of 0 means every request text is unique.
    """
    items = list(texts)
    n_total = len(items)
    n_unique = len(set(items))
    n_duplicate = n_total - n_unique
    return {
        "n_total": n_total,
        "n_unique": n_unique,
        "n_duplicate": n_duplicate,
        "duplicate_rate": (n_duplicate / n_total) if n_total else 0.0,
    }


def unique_first_indices(texts: collections.abc.Sequence[str]) -> np.ndarray:
    """Indices of the first occurrence of each distinct request text (order kept).

    Selecting these rows yields an exact-duplicate-free view of the data, used to
    re-evaluate a model on de-duplicated traffic.
    """
    seen: set[str] = set()
    keep: list[int] = []
    for index, text in enumerate(texts):
        if text not in seen:
            seen.add(text)
            keep.append(index)
    return np.asarray(keep, dtype=int)


def train_test_leakage(
    texts: collections.abc.Sequence[str],
    train_indices: collections.abc.Sequence[int],
    test_indices: collections.abc.Sequence[int],
) -> dict[str, float]:
    """Fraction of test requests whose exact text also appears in the train split.

    Quantifies optimistic bias from CSIC's heavy templating: a test request that
    is byte-for-byte identical to a training request is trivially "predicted".
    """
    items = list(texts)
    train_set = {items[int(i)] for i in train_indices}
    test_texts = [items[int(i)] for i in test_indices]
    n_test = len(test_texts)
    n_leaked = sum(1 for text in test_texts if text in train_set)
    return {
        "n_test": n_test,
        "n_leaked": n_leaked,
        "leakage_rate": (n_leaked / n_test) if n_test else 0.0,
    }
