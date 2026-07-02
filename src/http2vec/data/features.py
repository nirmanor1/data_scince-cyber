"""Descriptive, model-agnostic features for exploratory data analysis.

These features are intentionally simple and human-readable; they drive the
notebook's distribution plots, correlation analysis and class-imbalance view.
They are *not* the RoBERTa embeddings used for classification.

Security
--------
All payloads are treated as opaque text. Attack signatures (SQL keywords, script
tags) are detected with plain, fixed substring checks - never with dynamic or
backtracking-prone regexes - so the feature extraction itself cannot be turned
into a denial-of-service vector.
"""

from __future__ import annotations

import collections.abc
import math
from urllib.parse import urlsplit

import pandas as pd

from .schemas import DatasetSplit, HttpRequest

_FEATURE_COLUMNS = (
    "method",
    "target_length",
    "body_length",
    "n_headers",
    "n_query_params",
    "path_depth",
    "has_body",
    "pct_encoding_count",
    "pct_encoding_ratio",
    "digit_ratio",
    "upper_ratio",
    "non_alnum_ratio",
    "shannon_entropy",
    "contains_sql_keyword",
    "contains_script_tag",
    "label",
)

# Lower-cased, fixed substrings. Kept tiny and literal to avoid ReDoS.
_SQL_KEYWORDS = ("select", "union", "drop", "insert", "--", ";", "or ", "' or")
_SCRIPT_PATTERNS = ("<script", "%3cscript")


def _shannon_entropy(text: str) -> float:
    """Shannon entropy (base 2) of the character distribution of ``text``."""
    if not text:
        return 0.0
    total = len(text)
    counts = collections.Counter(text)
    return -sum(
        (count / total) * math.log2(count / total) for count in counts.values()
    )


def _char_ratios(text: str) -> tuple[float, float, float]:
    """Return ``(digit_ratio, upper_ratio, non_alnum_ratio)`` for ``text``."""
    length = len(text)
    if length == 0:
        return 0.0, 0.0, 0.0
    digits = sum(char.isdigit() for char in text)
    uppers = sum(char.isupper() for char in text)
    non_alnum = sum(not char.isalnum() for char in text)
    return digits / length, uppers / length, non_alnum / length


def _count_query_params(target: str) -> int:
    """Count ``&``-separated parameters in the query string of ``target``."""
    if "?" not in target:
        return 0
    query = target.split("?", 1)[1]
    return sum(1 for part in query.split("&") if part)


def _path_depth(target: str) -> int:
    """Number of non-empty path segments in ``target``."""
    try:
        path = urlsplit(target).path
    except ValueError:
        path = target
    return sum(1 for segment in path.split("/") if segment)


def _row(request: HttpRequest) -> dict[str, object]:
    """Compute the feature row for a single request."""
    target = request.target
    body = request.body
    blob = target + body
    lower_blob = blob.lower()
    digit_ratio, upper_ratio, non_alnum_ratio = _char_ratios(blob)
    pct_count = blob.count("%")
    return {
        "method": request.method,
        "target_length": len(target),
        "body_length": len(body),
        "n_headers": len(request.headers),
        "n_query_params": _count_query_params(target),
        "path_depth": _path_depth(target),
        "has_body": int(bool(body)),
        "pct_encoding_count": pct_count,
        "pct_encoding_ratio": pct_count / max(len(blob), 1),
        "digit_ratio": digit_ratio,
        "upper_ratio": upper_ratio,
        "non_alnum_ratio": non_alnum_ratio,
        "shannon_entropy": _shannon_entropy(blob),
        "contains_sql_keyword": int(any(kw in lower_blob for kw in _SQL_KEYWORDS)),
        "contains_script_tag": int(any(pat in lower_blob for pat in _SCRIPT_PATTERNS)),
        "label": int(request.label),
    }


def extract_features(
    requests: collections.abc.Sequence[HttpRequest],
) -> pd.DataFrame:
    """Build a one-row-per-request descriptive feature frame.

    The column set is fixed (see module docstring), so an empty input yields an
    empty frame with the expected columns rather than a frame with no schema.
    """
    rows = [_row(request) for request in requests]
    return pd.DataFrame(rows, columns=list(_FEATURE_COLUMNS))


def feature_frame(split: DatasetSplit) -> pd.DataFrame:
    """Thin wrapper: :func:`extract_features` over a :class:`DatasetSplit`."""
    return extract_features(split.requests)
