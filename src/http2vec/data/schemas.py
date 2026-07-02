"""Immutable data structures shared across the pipeline.

These types are deliberately free of heavy dependencies (only NumPy) so every
layer - parsing, tokenization, modelling, evaluation - can agree on the same
representation of an HTTP request without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable

import numpy as np


class Label(IntEnum):
    """Binary label. The positive class is the anomaly, matching the task framing."""

    NORMAL = 0
    ANOMALY = 1


@dataclass(frozen=True)
class HttpRequest:
    """A single parsed HTTP request.

    Payloads (target, headers, body) may contain adversarial content (SQLi, XSS,
    CRLF, ...). They are stored and handled strictly as opaque text; nothing in
    this codebase executes, evaluates or renders them.
    """

    method: str
    target: str
    version: str
    headers: tuple[tuple[str, str], ...]
    body: str
    label: Label
    raw: str

    @property
    def request_line(self) -> str:
        return " ".join(part for part in (self.method, self.target, self.version) if part)

    @property
    def header_map(self) -> dict[str, str]:
        """Headers as a dict. On duplicate names the last occurrence wins."""
        return {name: value for name, value in self.headers}

    def to_lines(self, first_line_only: bool = False) -> list[str]:
        """Return the textual lines used for tokenization / embedding.

        ``first_line_only`` keeps just the request line (used by the paper for
        datasets whose anomalies live almost entirely in the first line).
        """
        if first_line_only:
            return [self.request_line] if self.request_line else []

        lines: list[str] = []
        if self.request_line:
            lines.append(self.request_line)
        lines.extend(f"{name}: {value}" for name, value in self.headers)
        if self.body:
            lines.append(self.body)
        return lines

    def to_text(self, first_line_only: bool = False) -> str:
        return "\n".join(self.to_lines(first_line_only))


@dataclass(frozen=True)
class DatasetSplit:
    """An ordered collection of labelled requests."""

    requests: tuple[HttpRequest, ...]
    name: str = ""

    def __len__(self) -> int:
        return len(self.requests)

    @property
    def labels(self) -> np.ndarray:
        return np.fromiter(
            (int(request.label) for request in self.requests),
            dtype=np.int64,
            count=len(self.requests),
        )

    def texts(self, first_line_only: bool = False) -> list[str]:
        """One text blob per request (its lines joined by ``\\n``)."""
        return [request.to_text(first_line_only) for request in self.requests]

    def iter_lines(self, first_line_only: bool = False) -> Iterable[str]:
        """Iterate over every individual line across all requests (tokenizer corpus)."""
        for request in self.requests:
            yield from request.to_lines(first_line_only)


@dataclass(frozen=True)
class DatasetBundle:
    """The three views of a dataset required by the HTTP2vec pipeline.

    - ``lm_train``: normal-only traffic, used to train the RoBERTa MLM.
    - ``inference``: labelled normal + anomalous traffic, embedded and classified.
    - ``tokenizer_corpus``: all traffic (normal + anomalous), used to *train the
      BBPE tokenizer* as described in the paper.
    """

    lm_train: DatasetSplit
    inference: DatasetSplit
    tokenizer_corpus: DatasetSplit
