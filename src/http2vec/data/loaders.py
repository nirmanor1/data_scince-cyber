"""Concrete dataset loader for the CSIC 2010 corpus."""

from __future__ import annotations

import random

from ..config import DataConfig
from ..interfaces import AbstractDatasetLoader
from ..utils import logger
from .parser import parse_csic_file
from .schemas import DatasetBundle, DatasetSplit, HttpRequest, Label


def _random_subset(
    requests: list[HttpRequest], fraction: float, seed: int
) -> list[HttpRequest]:
    """Return a seeded random ``fraction`` of ``requests`` (original order kept)."""
    count = len(requests)
    keep = max(1, round(count * fraction))
    if keep >= count:
        return requests
    indices = sorted(random.Random(seed).sample(range(count), keep))
    return [requests[index] for index in indices]


class Csic2010Loader(AbstractDatasetLoader):
    """Load the three CSIC 2010 text files into a :class:`DatasetBundle`.

    The three views follow the HTTP2vec paper:

    * ``lm_train`` - normal training traffic only (RoBERTa MLM training).
    * ``inference`` - labelled normal + anomalous test traffic, concatenated.
    * ``tokenizer_corpus`` - every request used above (normal *and* anomalous),
      because the byte-level BPE tokenizer is trained on all traffic.

    Subset runs are deterministic: ``subset_fraction`` keeps a seeded random
    fraction of each file, otherwise the absolute caps take the first ``N``
    requests in file order.
    """

    def __init__(self, config: DataConfig) -> None:
        self._config = config

    def load(self) -> DatasetBundle:
        config = self._config

        lm_train = self._select(
            self._read(config.normal_train_file, Label.NORMAL),
            config.max_lm_train_samples,
        )
        normal = self._select(
            self._read(config.normal_test_file, Label.NORMAL),
            config.max_inference_per_class,
        )
        anomalous = self._select(
            self._read(config.anomalous_test_file, Label.ANOMALY),
            config.max_inference_per_class,
        )
        inference = normal + anomalous

        corpus = lm_train + inference

        logger.info(
            "Loaded CSIC2010: lm_train=%d, inference=%d (normal=%d, anomaly=%d), "
            "tokenizer_corpus=%d",
            len(lm_train),
            len(inference),
            len(normal),
            len(anomalous),
            len(corpus),
        )

        return DatasetBundle(
            lm_train=DatasetSplit(requests=tuple(lm_train), name="lm_train"),
            inference=DatasetSplit(requests=tuple(inference), name="inference"),
            tokenizer_corpus=DatasetSplit(requests=tuple(corpus), name="tokenizer_corpus"),
        )

    def _select(
        self, requests: list[HttpRequest], cap: int | None
    ) -> list[HttpRequest]:
        """Apply ``subset_fraction`` (preferred) or an absolute head ``cap``."""
        if self._config.subset_fraction is not None:
            return _random_subset(
                requests, self._config.subset_fraction, self._config.subset_seed
            )
        if cap is not None:
            return requests[:cap]
        return requests

    def _read(self, filename: str, label: Label) -> list[HttpRequest]:
        """Parse one raw file, raising a clear error if it is missing."""
        path = self._config.raw_dir / filename
        if not path.is_file():
            raise FileNotFoundError(
                f"CSIC2010 file not found: {path}. Place the three raw .txt files "
                "in the data directory as described in data/README.md."
            )
        return parse_csic_file(
            path, label, encode_crlf_literally=self._config.encode_crlf_literally
        )
