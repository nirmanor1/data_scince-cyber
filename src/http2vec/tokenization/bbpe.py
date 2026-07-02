"""Byte-level BPE (BBPE) tokenizer training and loading.

The paper trains a byte-level BPE tokenizer on *all* traffic (normal and
anomalous) and wraps it in a RoBERTa-compatible fast tokenizer. Heavy
dependencies (``tokenizers``, ``transformers``) are imported lazily inside the
functions so importing this module stays cheap and side-effect free.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from ..utils import logger

if TYPE_CHECKING:  # pragma: no cover - import only for static type checking
    from transformers import RobertaTokenizerFast

    from ..config import TokenizerConfig

# RoBERTa's fixed special-token order. ``<pad>`` lands on id 1, matching
# RoBERTa's padding_idx, which the position-embedding offset relies on.
SPECIAL_TOKENS = ["<s>", "<pad>", "</s>", "<unk>", "<mask>"]


def train_bbpe_tokenizer(
    corpus: Iterable[str],
    config: "TokenizerConfig",
    output_dir: Path,
) -> "RobertaTokenizerFast":
    """Train a byte-level BPE tokenizer and persist it as a RoBERTa fast tokenizer.

    Args:
        corpus: Iterable of text lines drawn from all traffic (normal + anomalous).
        config: Tokenizer hyper-parameters (vocab size, min frequency, max length).
        output_dir: Directory to write ``vocab.json``/``merges.txt`` and the
            serialized fast tokenizer. Created if missing.

    Returns:
        The trained :class:`transformers.RobertaTokenizerFast`.
    """
    from tokenizers import ByteLevelBPETokenizer
    from tokenizers.processors import RobertaProcessing
    from transformers import RobertaTokenizerFast

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    bbpe = ByteLevelBPETokenizer()
    bbpe.train_from_iterator(
        corpus,
        vocab_size=config.vocab_size,
        min_frequency=config.min_frequency,
        special_tokens=SPECIAL_TOKENS,
    )
    # Add RoBERTa's <s> ... </s> post-processing so encodings carry BOS/EOS exactly
    # as RobertaTokenizerFast expects.
    bbpe._tokenizer.post_processor = RobertaProcessing(
        ("</s>", bbpe.token_to_id("</s>")),
        ("<s>", bbpe.token_to_id("<s>")),
    )
    # Persist vocab/merges (for reference) and the full fast-tokenizer
    # serialization, which we load below.
    bbpe.save_model(str(output_dir))
    tokenizer_file = output_dir / "tokenizer.json"
    bbpe.save(str(tokenizer_file))

    # Build the fast tokenizer directly from tokenizer.json. This avoids
    # transformers' slow->fast conversion path, which needs ``sentencepiece`` and
    # is broken for byte-level BPE on some transformers versions.
    tokenizer = RobertaTokenizerFast(
        tokenizer_file=str(tokenizer_file),
        model_max_length=config.max_length,
        bos_token="<s>",
        eos_token="</s>",
        sep_token="</s>",
        cls_token="<s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Trained BBPE tokenizer (vocab=%d) -> %s", tokenizer.vocab_size, output_dir)
    return tokenizer


def load_tokenizer(output_dir: Path, max_length: int) -> "RobertaTokenizerFast":
    """Load a previously trained RoBERTa fast tokenizer.

    Args:
        output_dir: Directory produced by :func:`train_bbpe_tokenizer`.
        max_length: Maximum sequence length to enforce on the tokenizer.

    Returns:
        The loaded :class:`transformers.RobertaTokenizerFast`.
    """
    from transformers import RobertaTokenizerFast

    tokenizer = RobertaTokenizerFast.from_pretrained(str(output_dir))
    tokenizer.model_max_length = max_length
    return tokenizer
