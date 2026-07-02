"""RoBERTa masked-language-model: architecture, dataset, training and loading.

The model is trained only on *normal* traffic with dynamic masking, exactly as
described in the paper. ``transformers`` is imported lazily inside the functions
to keep import cost and version coupling localized; ``torch`` is imported at
module load because the dataset class derives from ``torch.utils.data.Dataset``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

import torch

from ..utils import logger, set_seed

if TYPE_CHECKING:  # pragma: no cover - import only for static type checking
    from transformers import RobertaForMaskedLM

    from ..config import ModelConfig


class TextLineDataset(torch.utils.data.Dataset):
    """A lazily-tokenized dataset of text examples for masked-language modeling.

    Each item is tokenized with truncation to ``max_length``; padding and the
    dynamic MLM masking are deferred to the data collator, so ``__getitem__``
    returns a single example's encoding (no tensors, no padding).
    """

    def __init__(self, texts: Sequence[str], tokenizer, max_length: int) -> None:
        self._texts = list(texts)
        self._tokenizer = tokenizer
        self._max_length = max_length

    def __len__(self) -> int:
        return len(self._texts)

    def __getitem__(self, index: int) -> dict:
        encoding = self._tokenizer(
            self._texts[index],
            truncation=True,
            max_length=self._max_length,
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }


def build_mlm_model(
    model_config: "ModelConfig",
    *,
    vocab_size: int,
    max_position_embeddings: int,
    pad_token_id: int,
) -> "RobertaForMaskedLM":
    """Construct a fresh RoBERTa masked-language model from the given config.

    Args:
        model_config: Architecture/training hyper-parameters.
        vocab_size: Size of the tokenizer vocabulary (the embedding rows).
        max_position_embeddings: Maximum positions; must equal ``max_length + 2``
            to leave room for RoBERTa's padding offset.
        pad_token_id: Padding token id (RoBERTa expects this as ``padding_idx``).

    Returns:
        An untrained :class:`transformers.RobertaForMaskedLM`.
    """
    from transformers import RobertaConfig, RobertaForMaskedLM

    config = RobertaConfig(
        vocab_size=vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=max_position_embeddings,
        pad_token_id=pad_token_id,
        type_vocab_size=1,
    )
    return RobertaForMaskedLM(config)


def _supported_eval_strategy_key(training_arguments_cls) -> str:
    """Return the eval-strategy kwarg name supported by this transformers version.

    ``TrainingArguments`` renamed ``evaluation_strategy`` to ``eval_strategy`` in
    transformers 4.41; older releases only accept the former. We introspect the
    constructor so the same code works across supported versions.
    """
    import inspect

    params = inspect.signature(training_arguments_cls.__init__).parameters
    return "eval_strategy" if "eval_strategy" in params else "evaluation_strategy"


def train_mlm(
    *,
    texts: Sequence[str],
    tokenizer,
    model_config: "ModelConfig",
    max_length: int,
    output_dir: Path,
    device: str,
    seed: int,
    eval_texts: Sequence[str] | None = None,
    logging_steps: int = 10,
) -> tuple["RobertaForMaskedLM", list[dict]]:
    """Train the RoBERTa MLM on normal traffic with dynamic masking.

    Args:
        texts: Normal-traffic text examples (one per training row).
        tokenizer: A trained RoBERTa fast tokenizer.
        model_config: Architecture/training hyper-parameters.
        max_length: Maximum sequence length (defines ``max_position_embeddings``).
        output_dir: Directory to save the trained model and tokenizer.
        device: Device string to place the returned model on.
        seed: Seed for deterministic training.
        eval_texts: Optional held-out normal examples; when given, the trainer
            evaluates once per epoch so the returned history contains an
            ``eval_loss`` curve in addition to the per-step training loss.
        logging_steps: How often (in steps) to log the training loss. Kept small
            so a few-epoch run still yields a detailed curve.

    Returns:
        A ``(model, history)`` tuple: the trained
        :class:`transformers.RobertaForMaskedLM` moved to ``device`` and the
        ``Trainer`` log history (list of dicts), which is also written to
        ``output_dir / "training_log.json"``.
    """
    from transformers import (
        DataCollatorForLanguageModeling,
        Trainer,
        TrainingArguments,
    )

    set_seed(seed)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = build_mlm_model(
        model_config,
        vocab_size=len(tokenizer),
        max_position_embeddings=max_length + 2,
        pad_token_id=tokenizer.pad_token_id,
    )

    train_dataset = TextLineDataset(texts, tokenizer, max_length)
    eval_dataset = (
        TextLineDataset(list(eval_texts), tokenizer, max_length)
        if eval_texts
        else None
    )
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=model_config.mlm_probability,
    )

    args_kwargs = dict(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=model_config.num_train_epochs,
        per_device_train_batch_size=model_config.per_device_batch_size,
        per_device_eval_batch_size=model_config.per_device_batch_size,
        learning_rate=model_config.learning_rate,
        weight_decay=model_config.weight_decay,
        warmup_ratio=model_config.warmup_ratio,
        seed=seed,
        logging_strategy="steps",
        logging_steps=max(1, int(logging_steps)),
        save_strategy="no",
        report_to="none",
        fp16=(device == "cuda"),
    )
    if eval_dataset is not None:
        args_kwargs[_supported_eval_strategy_key(TrainingArguments)] = "epoch"
    training_args = TrainingArguments(**args_kwargs)

    logger.info(
        "Training RoBERTa MLM on %d examples (%d held out for eval) for %s epoch(s).",
        len(train_dataset),
        len(eval_dataset) if eval_dataset is not None else 0,
        model_config.num_train_epochs,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )
    trainer.train()

    history = [dict(entry) for entry in trainer.state.log_history]
    try:
        (output_dir / "training_log.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
    except (TypeError, ValueError, OSError) as error:  # pragma: no cover - non-fatal
        logger.warning("Could not write training_log.json: %s", error)

    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    return model.to(device), history


def load_mlm_model(model_dir: Path, device: str) -> "RobertaForMaskedLM":
    """Load a trained RoBERTa MLM for inference.

    Args:
        model_dir: Directory produced by :func:`train_mlm`.
        device: Device string to place the model on.

    Returns:
        The loaded model in eval mode on ``device``.
    """
    from transformers import RobertaForMaskedLM

    model = RobertaForMaskedLM.from_pretrained(str(model_dir))
    return model.to(device).eval()
