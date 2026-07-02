"""Cross-cutting helpers: deterministic seeding, device resolution, logging.

Heavy dependencies (numpy, torch) are imported lazily so this module stays
importable in minimal environments.
"""

from __future__ import annotations

import logging
import os
import random

logger = logging.getLogger("http2vec")


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a single, concise handler for the package logger.

    Idempotent: repeated calls do not stack handlers.
    """
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False


def set_seed(seed: int) -> None:
    """Seed every relevant RNG for reproducible runs.

    Covers Python ``random``, ``PYTHONHASHSEED``, NumPy and PyTorch (CPU + CUDA)
    when those libraries are available.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:  # pragma: no cover - numpy is a hard dependency in practice
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:  # pragma: no cover - torch is optional for light usage
        pass


def resolve_device(device: str = "auto") -> str:
    """Resolve a device string.

    ``"auto"`` prefers CUDA, then Apple MPS (Metal), then CPU.
    """
    if device != "auto":
        return device
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"
