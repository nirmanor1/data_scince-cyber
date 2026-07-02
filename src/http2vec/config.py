"""Typed, immutable experiment configuration with ready-made profiles.

Profiles
--------
- :meth:`ExperimentConfig.small` - the full paper-size model on a seeded 60% subset,
  5 epochs; a lighter stand-in for the full run (the default).
- :meth:`ExperimentConfig.paper`  - the hyper-parameters reported in the paper
  (full data, 10 epochs; heavy, intended for a real GPU run).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_OUTPUT_DIR = Path("artifacts")

CSIC_NORMAL_TRAIN = "normalTrafficTraining.txt"
CSIC_NORMAL_TEST = "normalTrafficTest.txt"
CSIC_ANOMALOUS_TEST = "anomalousTrafficTest.txt"


@dataclass(frozen=True)
class DataConfig:
    """Where the raw CSIC2010 files live and how much of them to use."""

    raw_dir: Path = DEFAULT_RAW_DIR
    normal_train_file: str = CSIC_NORMAL_TRAIN
    normal_test_file: str = CSIC_NORMAL_TEST
    anomalous_test_file: str = CSIC_ANOMALOUS_TEST
    # ``None`` means "use everything"; integers cap the number of samples (subset runs).
    max_lm_train_samples: int | None = None
    max_inference_per_class: int | None = None
    # When set, keep a seeded random fraction (0, 1] of each file (takes precedence
    # over the absolute caps above). Used for reduced "run on X% of the data" runs.
    subset_fraction: float | None = None
    subset_seed: int = 42
    first_line_only: bool = False
    encode_crlf_literally: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_dir", Path(self.raw_dir))
        if self.subset_fraction is not None and not 0.0 < self.subset_fraction <= 1.0:
            raise ValueError("subset_fraction must be in the interval (0, 1].")


@dataclass(frozen=True)
class TokenizerConfig:
    """Byte-level BPE tokenizer settings."""

    vocab_size: int = 30000
    min_frequency: int = 2
    max_length: int = 512


@dataclass(frozen=True)
class ModelConfig:
    """RoBERTa masked-language-model architecture and training schedule.

    ``max_position_embeddings`` is intentionally absent: it is derived from the
    tokenizer's ``max_length`` (+2 for RoBERTa's padding offset) by the model
    builder, keeping the two in sync by construction.
    """

    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    num_train_epochs: int = 10
    per_device_batch_size: int = 32
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.06
    mlm_probability: float = 0.15
    # Seeded fraction of normal traffic held out as an MLM validation set so the
    # training history includes an eval-loss-per-epoch curve. 0 disables eval.
    eval_fraction: float = 0.05

    def __post_init__(self) -> None:
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({self.hidden_size}) must be divisible by "
                f"num_attention_heads ({self.num_attention_heads})."
            )
        if not 0.0 < self.mlm_probability < 1.0:
            raise ValueError("mlm_probability must be in the open interval (0, 1).")
        if not 0.0 <= self.eval_fraction < 1.0:
            raise ValueError("eval_fraction must be in the half-open interval [0, 1).")


@dataclass(frozen=True)
class EmbeddingConfig:
    """How request embeddings are pooled from RoBERTa hidden states."""

    last_n_layers: int = 4
    token_pooling: str = "mean"  # 'mean' or 'cls'
    line_aggregation: str = "mean"  # how per-line vectors are combined
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.last_n_layers < 1:
            raise ValueError("last_n_layers must be >= 1.")
        if self.token_pooling not in {"mean", "cls"}:
            raise ValueError("token_pooling must be 'mean' or 'cls'.")
        if self.line_aggregation not in {"mean"}:
            raise ValueError("line_aggregation must be 'mean'.")


@dataclass(frozen=True)
class ClassifierConfig:
    """Supervised classifiers + Isolation Forest hyper-parameters."""

    cv_folds: int = 5
    test_size: float = 0.3
    random_state: int = 42
    scale_features: bool = True
    lr_max_iter: int = 2000
    rf_n_estimators: int = 100
    svc_c: float = 1.0
    svc_max_iter: int = 5000
    gb_n_estimators: int = 100
    knn_n_neighbors: int = 15
    iforest_n_estimators: int = 100
    iforest_contamination: float | str = "auto"
    iforest_max_samples: str | int | float = "auto"
    lof_n_neighbors: int = 20


@dataclass(frozen=True)
class MlpHeadConfig:
    """Trainable MLP classifier head on top of the frozen embeddings.

    An extension beyond the paper: a small supervised neural head over the
    (frozen) RoBERTa vectors. It is cheap enough to run for several epochs, which
    yields a detailed train/validation learning curve.
    """

    hidden_size: int = 256
    dropout: float = 0.2
    epochs: int = 40
    patience: int = 8
    learning_rate: float = 5e-4
    batch_size: int = 64
    weight_decay: float = 1e-4
    val_fraction: float = 0.15
    random_state: int = 42

    def __post_init__(self) -> None:
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be >= 1.")
        if self.patience < 1:
            raise ValueError("patience must be >= 1.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the half-open interval [0, 1).")
        if not 0.0 <= self.val_fraction < 1.0:
            raise ValueError("val_fraction must be in the half-open interval [0, 1).")


@dataclass(frozen=True)
class EvaluationConfig:
    """Evaluation knobs. ``f_beta`` > 1 weights recall higher (catching attacks)."""

    positive_label: int = 1
    f_beta: float = 2.0
    tpr_targets: tuple[float, ...] = (0.90, 0.99)


@dataclass(frozen=True)
class ExperimentConfig:
    """Aggregate configuration for one end-to-end run."""

    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    mlp_head: MlpHeadConfig = field(default_factory=MlpHeadConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    seed: int = 42
    device: str = "auto"
    output_dir: Path = DEFAULT_OUTPUT_DIR

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))

    @property
    def tokenizer_dir(self) -> Path:
        return self.output_dir / "tokenizer"

    @property
    def model_dir(self) -> Path:
        return self.output_dir / "roberta-mlm"

    @property
    def max_position_embeddings(self) -> int:
        """RoBERTa needs room for the padding offset (padding_idx=1)."""
        return self.tokenizer.max_length + 2

    @classmethod
    def paper(
        cls,
        *,
        raw_dir: Path | str = DEFAULT_RAW_DIR,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        device: str = "auto",
        seed: int = 42,
        num_train_epochs: int = 10,
    ) -> "ExperimentConfig":
        return cls(
            data=DataConfig(raw_dir=Path(raw_dir)),
            tokenizer=TokenizerConfig(vocab_size=30000, max_length=512),
            model=ModelConfig(
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                intermediate_size=3072,
                num_train_epochs=num_train_epochs,
                per_device_batch_size=32,
            ),
            embedding=EmbeddingConfig(last_n_layers=4, batch_size=32),
            seed=seed,
            device=device,
            output_dir=Path(output_dir),
        )

    @classmethod
    def small(
        cls,
        *,
        raw_dir: Path | str = DEFAULT_RAW_DIR,
        output_dir: Path | str = DEFAULT_OUTPUT_DIR,
        device: str = "auto",
        seed: int = 42,
        num_train_epochs: int = 5,
    ) -> "ExperimentConfig":
        """Full paper-size model on a seeded 60% subset, fewer epochs.

        A lighter stand-in for :meth:`paper`: same architecture, ~60% of the data
        and 5 epochs by default.
        """
        return cls(
            data=DataConfig(raw_dir=Path(raw_dir), subset_fraction=0.6),
            tokenizer=TokenizerConfig(vocab_size=30000, max_length=512),
            model=ModelConfig(
                hidden_size=768,
                num_hidden_layers=6,
                num_attention_heads=12,
                intermediate_size=3072,
                num_train_epochs=num_train_epochs,
                per_device_batch_size=32,
            ),
            embedding=EmbeddingConfig(last_n_layers=4, batch_size=32),
            seed=seed,
            device=device,
            output_dir=Path(output_dir),
        )
