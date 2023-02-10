from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

from codecg.config import DATASET_NAME, PATH_CACHE, TOKENIZER_MODEL, ModelConfig, TrainConfig


def load_tokenizer(name: str = TOKENIZER_MODEL):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, use_fast=True, cache_dir=PATH_CACHE)


def model_config_for(tokenizer, **overrides) -> ModelConfig:
    # Special token ids come from the tokenizer, so they always match the data
    return ModelConfig(
        vocab_size=len(tokenizer),
        pad_id=tokenizer.pad_token_id,
        bos_id=tokenizer.bos_token_id,
        eos_id=tokenizer.eos_token_id,
        **overrides,
    )


class CodeCommentDataset(Dataset):
    """
        CodeXGLUE code-to-text pairs, tokenized with the CodeT5 subword tokenizer.
        Code and comments are the dataset's whitespace-joined token lists
    """
    def __init__(self, split: str, tokenizer, cfg: TrainConfig) -> None:
        import datasets as ds

        # The Hub copy stores one parquet folder per language. datasets 2.x does not
        # map those folders to configs, so select the language with data_dir. The
        # repo's recorded split sizes describe a single language, so skip that check
        data = ds.load_dataset(
            DATASET_NAME,
            data_dir=cfg.language,
            split=split,
            cache_dir=PATH_CACHE,
            ignore_verifications=True,
        )
        if cfg.limit:
            data = data.select(range(min(cfg.limit, len(data))))

        codes = [" ".join(t) for t in data["code_tokens"]]
        self.comments = [" ".join(t) for t in data["docstring_tokens"]]

        self.src = tokenizer(codes, max_length=cfg.max_src_len, truncation=True)["input_ids"]
        self.tgt = tokenizer(self.comments, max_length=cfg.max_tgt_len, truncation=True)["input_ids"]

    def __len__(self) -> int:
        return len(self.src)

    def __getitem__(self, idx: int) -> Dict[str, List[int]]:
        return {"src": self.src[idx], "tgt": self.tgt[idx], "idx": idx}


def pad(sequences: List[List[int]], pad_id: int) -> torch.Tensor:
    width = max(len(s) for s in sequences)
    out = torch.full((len(sequences), width), pad_id, dtype=torch.long)
    for i, s in enumerate(sequences):
        out[i, : len(s)] = torch.tensor(s, dtype=torch.long)
    return out


class Collator:
    def __init__(self, pad_id: int) -> None:
        self.pad_id = pad_id

    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        return {
            "src": pad([b["src"] for b in batch], self.pad_id),
            "tgt": pad([b["tgt"] for b in batch], self.pad_id),
            "idx": torch.tensor([b["idx"] for b in batch]),
        }


def make_loader(dataset: Dataset, cfg: TrainConfig, pad_id: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        collate_fn=Collator(pad_id),
    )
