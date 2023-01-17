from typing import Optional
import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS, EVAL_DATALOADERS

from transformers import AutoTokenizer
import datasets as ds 

from config import (
    BATCH_SIZE,
    DATASET_NAME,
    MAX_SEQUENCE_LENGTH,
    NUM_WORKERS,
    PADDING,
    PATH_BASE_MODELS,
    PATH_CACHE_DATASETS,
    PL,
    TOKENIZER_MODEL,
)

class NL2NLDM(pl.LightningDataModule):

    loader_cols = [
        "input_ids", 
        "attention_mask", 
        "target_input_ids", 
        "target_attention_mask",
        "labels"
    ]

    def __init__(
        self,
        tokenizer_model: str = TOKENIZER_MODEL,
        pl: str = PL,
        path_base_models: str = PATH_BASE_MODELS,
        path_cache_dataset: str = PATH_CACHE_DATASETS,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ) -> None:

        super().__init__() 

        self.save_hyperparameters(logger=False)            # Can access __init__ params using self.hparams

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_model, 
            use_fast=True,
            cache_dir=self.hparams.path_base_models,
        )


    def prepare_data(self) -> None:
        ds.load_dataset(DATASET_NAME, self.hparams.pl, cache_dir=self.hparams.path_cache_dataset)
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_dataset(DATASET_NAME, self.hparams.pl, cache_dir=self.hparams.path_cache_dataset)
        
        
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._to_features,
                batched=True,
                batch_size=self.hparams.batch_size,
                num_proc=self.hparams.num_workers,
            )

            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_cols
            ]

        self.dataset.set_format(type='torch', columns=self.columns)


    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.dataset['train'], 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True,
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset['validation'], 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset['test'], 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
        )

    def _to_features(self, batch, indices=None):
        
        features = self.tokenizer(
            text=batch['docstring'],
            max_length=self.hparams.max_seq_len,
            padding=self.hparams.padding,
            truncation=True,
        ) 

        targets = self.tokenizer(
            text=batch['docstring'], 
            max_length=self.hparams.max_seq_len,
            padding=self.hparams.padding,
            truncation=True,
        )

        # Goal: D(E(x)) = x => our targets are nothing but our inputs
        
        features['target_input_ids'] = targets['input_ids']
        features['target_attention_mask'] = targets['attention_mask']

        features['labels'] = features['target_input_ids']
        return features 



class PL2NLDM(pl.LightningDataModule):
    loader_cols = [
        "input_ids", 
        "attention_mask", 
        "target_input_ids", 
        "target_attention_mask",
    ]
    
    def __init__(
        self,
        tokenizer_model: str = TOKENIZER_MODEL,
        pl: str = PL,
        path_base_models: str = PATH_BASE_MODELS,
        path_cache_dataset: str = PATH_CACHE_DATASETS,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        padding: str = PADDING,
        batch_size: int = BATCH_SIZE,
        num_workers: int = NUM_WORKERS,
    ) -> None: 
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer_model,
            useFast=True,
            cache_dir=path_base_models,
        )
    
    def prepare_data(self) -> None:
        ds.load_dataset(DATASET_NAME, self.hparams.pl, cache_dir=self.hparams.path_cache_dataset)
    
    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = ds.load_dataset(DATASET_NAME, self.hparams.pl, cache_dir=self.hparams.path_cache_dataset)
        
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self._to_features,
                batched=True,
                batch_size=self.hparams.batch_size,
                num_proc=self.hparams.num_workers,
            )

            self.columns = [
                c for c in self.dataset[split].column_names if c in self.loader_cols
            ]

        self.dataset.set_format(type='torch', columns=self.columns)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset=self.dataset['train'], 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset['validation'], 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(
            dataset=self.dataset['test'], 
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            drop_last=True
        )

    def _to_features(self, batch, indices=None):

        #TODO: Need to add support for GraphCodeBERT / Transformer efficiently encoding AST or DataFlow Graphs

        lang_name_code_pair = list(zip(batch['language'], batch['code']))

        features = self.tokenizer(
            text=lang_name_code_pair,
            max_length=self.hparams.max_seq_len,
            padding=self.hparams.padding,
            truncation=True
        )

        targets = self.tokenizer(
            text=batch['docstring'],
            max_length=self.hparams.max_seq_len,
            padding=self.hparams.padding,
            truncation=True
        )

        features['target_input_ids'] = targets['input_ids']
        features['target_attention_mask'] = targets['attention_mask']

        return features 
    
