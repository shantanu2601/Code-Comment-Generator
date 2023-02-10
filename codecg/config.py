import os
from dataclasses import asdict, dataclass, field


# Paths
PATH_RUNS = os.environ.get("PATH_RUNS", "./runs")
PATH_CACHE = os.environ.get("PATH_CACHE", "./cache")

# Data
DATASET_NAME = "google/code_x_glue_ct_code_to_text"
TOKENIZER_MODEL = "Salesforce/codet5-base"
LANGUAGES = ("python", "java", "javascript", "go", "php", "ruby")

GLOBAL_SEED = 42


@dataclass
class ModelConfig:
    """
        Shared by the summarizer, the text autoencoder and the code encoder,
        so that aligned encoders and decoders can be swapped between them
    """
    vocab_size: int = 32100
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    d_model: int = 512
    n_heads: int = 8
    d_ff: int = 2048
    n_layers: int = 6
    dropout: float = 0.2

    # "absolute", "relative" or "both" (encoder only, decoder is always absolute)
    encoder_positions: str = "relative"
    # Clipping distance k for relative positions
    max_relative_position: int = 16
    max_len: int = 512

    copy_attention: bool = True

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelConfig":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainConfig:
    language: str = "java"
    max_src_len: int = 256
    max_tgt_len: int = 64

    epochs: int = 50
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_clip: float = 5.0
    warmup_steps: int = 2000

    # Adversarial alignment
    g_lr: float = 1e-4
    d_lr: float = 1e-5
    mse_weight: float = 0.6

    # Decoding
    beam_size: int = 4
    max_decode_len: int = 64

    num_workers: int = field(default_factory=lambda: min(4, (os.cpu_count() or 2) // 2))
    seed: int = GLOBAL_SEED
    limit: int = 0  # Use only the first N examples per split (0 = all), for smoke tests

    def to_dict(self) -> dict:
        return asdict(self)
