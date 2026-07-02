"""Request embedder built on a trained RoBERTa model.

A request is embedded by concatenating the last ``N`` hidden layers (clamped to
what the model actually exposes), pooling each layer over a line's tokens, then
averaging the resulting per-line vectors into one fixed-length request vector.
``torch``/``numpy`` are imported lazily inside the methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import only for static type checking
    import numpy as np

    from ..config import EmbeddingConfig
    from ..data.schemas import HttpRequest


class RobertaRequestEmbedder:
    """Embed HTTP requests using hidden states of a trained RoBERTa model.

    Implements the :class:`~http2vec.interfaces.Embedder` protocol.
    """

    def __init__(
        self,
        model,
        tokenizer,
        config: "EmbeddingConfig",
        *,
        max_length: int,
        device: str = "cpu",
        first_line_only: bool = False,
    ) -> None:
        self._model = model.to(device).eval()
        self._tokenizer = tokenizer
        self._config = config
        self._max_length = max_length
        self._device = device
        self._first_line_only = first_line_only

        # hidden_states exposes num_hidden_layers + 1 tensors (embeddings + layers);
        # clamp so tiny models with few layers still produce a valid vector.
        self._last_n_used = min(config.last_n_layers, model.config.num_hidden_layers + 1)
        self._embedding_dim = self._last_n_used * model.config.hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def embed(self, requests: Sequence["HttpRequest"]) -> "np.ndarray":
        """Return an ``(len(requests), embedding_dim)`` float32 array.

        Requests with no lines contribute an all-zero vector.
        """
        import numpy as np

        flat_lines: list[str] = []
        line_counts: list[int] = []
        for request in requests:
            lines = request.to_lines(first_line_only=self._first_line_only)
            line_counts.append(len(lines))
            flat_lines.extend(lines)

        result = np.zeros((len(requests), self._embedding_dim), dtype=np.float32)
        if not flat_lines:
            return result

        line_vectors = self._encode_lines(flat_lines)

        offset = 0
        for index, count in enumerate(line_counts):
            if count:
                result[index] = line_vectors[offset:offset + count].mean(axis=0)
            offset += count
        return result

    def _encode_lines(self, lines: list[str]) -> "np.ndarray":
        """Encode each line into a ``last_n_used * hidden_size`` vector."""
        import numpy as np
        import torch

        batches: list[np.ndarray] = []
        for start in range(0, len(lines), self._config.batch_size):
            batch = lines[start:start + self._config.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs, output_hidden_states=True)

            selected = outputs.hidden_states[-self._last_n_used:]
            attention_mask = inputs["attention_mask"]
            pooled = [self._pool_tokens(layer, attention_mask) for layer in selected]
            line_vectors = torch.cat(pooled, dim=-1)
            batches.append(line_vectors.to(torch.float32).cpu().numpy())

        return np.concatenate(batches, axis=0)

    def _pool_tokens(self, layer, attention_mask):
        """Pool a ``[B, T, H]`` layer over tokens into ``[B, H]``."""
        if self._config.token_pooling == "cls":
            return layer[:, 0, :]

        weights = attention_mask.unsqueeze(-1).to(layer.dtype)
        summed = (layer * weights).sum(dim=1)
        counts = weights.sum(dim=1).clamp(min=1.0)
        return summed / counts
