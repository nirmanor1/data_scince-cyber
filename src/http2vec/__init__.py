"""HTTP2vec - RoBERTa embeddings of HTTP requests for anomaly detection.

The public submodules are imported lazily (just ``import http2vec.config`` etc.)
so that importing the top-level package never forces heavy optional dependencies
(torch, transformers) to load before they are needed.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
