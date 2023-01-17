from typing import Optional
import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import pytorch_lightning as pl 
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torchmetrics.functional import accuracy

from transformers import (
    AutoConfig,
    AutoModel,
    GPT2Config,
    RobertaConfig,
)

from config import (
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH,
    NL_DECODER_BASE_MODEL,
    NL_ENCODER_BASE_MODEL,
    PATH_BASE_MODELS,
    PATH_SAVE_NL_DECODER,
    PATH_SAVE_NL_ENCODER,
    PATH_SAVE_NL_LM,
    PATH_SAVE_PL_ENCODER,
    PL_ENCODER_BASE_MODEL, 
    VOCAB_SIZE,
    WEIGHT_DECAY
)

class LMHead(nn.Module):
    def __init__(
        self, 
        config, 
    ) -> None:
        super().__init__()

        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))   
        self.decoder.bias = self.bias 

    def forward(self, features):
        x = self.dense(features)
        x = self.layer_norm(x)
        x = F.gelu(x)

        # project back to size of vocab with bias 
        x = self.decoder(x)
        return x 

    def save(self, path: str = PATH_SAVE_NL_LM):
        torch.save(self.state_dict(), path)
        return path 


class NLEncoder(nn.Module):
    """
        Natural Language Encoder
    """

    def __init__(
        self, 
        model_name_or_path: str = NL_ENCODER_BASE_MODEL, 
        vocab_size: int = VOCAB_SIZE,
        usePretrained: bool = False,
    ) -> None:

        super().__init__()

        if usePretrained:
            self.config = RobertaConfig()
            self.config.vocab_size = vocab_size

            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path,
            )

        else:
            if model_name_or_path.lower() == "roberta-base":
                self.config = RobertaConfig()
                self.config.vocab_size = vocab_size                # Making vocab size to that of CodeT5 since we are using CodeT5 tokenizer
            
                self.model = AutoModel.from_config(self.config)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            output_hidden_states=True
        )

    def freeze(self):
        """
            Freeze Encoder Params for inference
        """

        for p in self.parameters():
            p.requires_grad = False 
    
    def save(self, path:str = PATH_SAVE_NL_ENCODER):
        self.model.save_pretrained(path)
        return path 

class NLDecoder(nn.Module):
    """
        Natural Language Decoder
    """
    def __init__(
        self,
        model_name_or_path: str = NL_DECODER_BASE_MODEL,
        vocab_size: int = VOCAB_SIZE,
        usePretrained: bool = False,
    ) -> None:

        super().__init__()

        if usePretrained:
            self.model = AutoModel.from_pretrained(
                pretrained_model_name_or_path=model_name_or_path
            )

        else:
            if model_name_or_path.lower() == "gpt2":
                self.config = GPT2Config()
                self.config.add_cross_attention = True           # Setting this as decoder
                self.config.vocab_size = vocab_size              # Making vocab size to that of CodeT5 since we are using CodeT5 tokenizer

            self.model = AutoModel.from_config(self.config)

    
    def forward(self, input_ids, attention_mask, encoder_hidden_states, encoder_attention_mask): 
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask
        )
    
    def freeze(self):
        """ 
            Freeze all weights for inference
        """

        for p in self.parameters():
            p.requires_grad = False 
    
    def save(self, path: str = PATH_SAVE_NL_DECODER):
        self.model.save_pretrained(path)
        return path 

        
class NL2NL(pl.LightningModule):
    def __init__(
        self, 
        encoder_model: str = NL_ENCODER_BASE_MODEL,
        decoder_model: str = NL_DECODER_BASE_MODEL,

        vocab_size: int = VOCAB_SIZE,
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Initializing encoder-decoder 
        self.encoder = NLEncoder(encoder_model, vocab_size)
        self.decoder = NLDecoder(decoder_model, vocab_size)

        self.lm_head = LMHead(self.decoder.config)

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, target_input_ids, target_attention_mask):
        encoder_outs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        ) 
        encoder_hidden_states = encoder_outs.last_hidden_state

        decoder_outs = self.decoder(
            input_ids=target_input_ids,
            attention_mask=target_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=attention_mask,
        )

        lm_outs = self.lm_head(decoder_outs.last_hidden_state)
        return lm_outs 

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outs = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['target_input_ids'],
            batch['target_attention_mask']
        )

        loss = self.loss_fn(
            outs.view(-1, self.decoder.config.vocab_size),
            batch['labels'].view(-1)
        )

        probs = F.softmax(outs, dim=-1)
        preds = probs.view(-1, self.decoder.config.vocab_size).argmax(dim=-1)
        acc = accuracy(preds=preds.view(-1), target=batch['labels'].view(-1))

        self.log('loss/train', loss)
        self.log('acc/train', acc)

        return loss 

    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outs = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['target_input_ids'],
            batch['target_attention_mask']
        )

        loss = self.loss_fn(
            outs.view(-1, self.decoder.config.vocab_size),
            batch['labels'].view(-1)
        )

        # Calculating token distribution to compute accuracy
        probs = F.softmax(outs, dim=-1)
        preds = probs.view(-1, self.decoder.config.vocab_size).argmax(dim=-1)
        acc = accuracy(preds=preds.view(-1), target=batch['labels'].view(-1))

        self.log('loss/val', loss)
        self.log('acc/val', acc)

    
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outs = self(
            batch['input_ids'],
            batch['attention_mask'],
            batch['target_input_ids'],
            batch['target_attention_mask']
        )

        loss = self.loss_fn(
            outs.view(-1, self.decoder.config.vocab_size),
            batch['labels'].view(-1)
        )


        probs = F.softmax(outs, dim=-1)
        preds = probs.view(-1, self.decoder.config.vocab_size).argmax(dim=-1)
        acc = accuracy(preds=preds.view(-1), target=batch['labels'].view(-1))

        self.log('loss/test', loss)
        self.log('acc/test', acc) 


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        return optimizer 

    def save(self, encoder_path: str = PATH_SAVE_NL_ENCODER, decoder_path: str = PATH_SAVE_NL_DECODER, lm_path: str = PATH_SAVE_NL_LM):
        self.encoder.save(encoder_path)
        self.decoder.save(decoder_path)
        self.lm_head.save(lm_path)

        return encoder_path, decoder_path, lm_path


class PLEncoder(nn.Module):
    def __init__(
        self, 
        model_name_or_path: str = PL_ENCODER_BASE_MODEL,
        vocab_size: int = VOCAB_SIZE,
        usePretrained: bool = False,
        path_base_models: str = PATH_BASE_MODELS,
    ) -> None:
        super().__init__()

        if usePretrained:
            self.model = AutoModel.from_pretrained(
                model_name_or_path,
            )

            self.config = self.model.config

        else:
            if model_name_or_path.lower() == "microsoft/codebert-base":
                self.config = AutoConfig.from_pretrained(
                    pretrained_model_name_or_path=model_name_or_path,
                    cache_dir=path_base_models
                ) 

                self.config.vocab_size = vocab_size

            elif model_name_or_path.lower() == "roberta-base":
                self.config = RobertaConfig()
                self.config.vocab_size = vocab_size

            self.model = AutoModel.from_config(self.config)


    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        return self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
        )
    
    def freeze(self):
        """
            Freeze params
        """

        for p in self.parameters():
            p.requires_grad = False 

    def save(self, path=PATH_SAVE_PL_ENCODER):
        self.model.save_pretrained(path)

        return path 

class Discriminator(nn.Module):
    def __init__(
        self, 
        input_dims: int,
    ) -> None:
        super().__init__()

        self.bi_lstm = nn.LSTM(
            input_size=input_dims,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
        )

        self.net = nn.Sequential(
            nn.Linear(512, 128), 
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.LayerNorm(32), 
            nn.GELU(),
            nn.Linear(32, 8), 
            nn.LayerNorm(8),
            nn.GELU(),
            nn.Linear(8, 1), 
        )

    def forward(self, h):
        y, _ = self.bi_lstm(h)
        y = self.net(y[:, 0])
        return y

class PLNLSpace(pl.LightningModule):
    def __init__(
        self, 
        nl_encoder_base_model: str = NL_ENCODER_BASE_MODEL,
        pl_encoder_base_model: str = PL_ENCODER_BASE_MODEL,
        g_learning_rate: float = LEARNING_RATE,
        d_learning_rate: float = LEARNING_RATE,
        g_weight_decay: float = WEIGHT_DECAY,
        d_weight_decay: float = WEIGHT_DECAY,
        max_seq_len: int = MAX_SEQUENCE_LENGTH,
        vocab_size: int = VOCAB_SIZE,
        path_base_models: str = PATH_BASE_MODELS,
        clip_value: int = 0.01
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        # Our PL Encoder acts as a generator which will give us some intermediate representations
        self.generator = PLEncoder(
            model_name_or_path=pl_encoder_base_model,
            vocab_size=vocab_size,
            path_base_models=path_base_models,
        )

        # NL Encoder representations will be source of truth
        self.sot = NLEncoder(
            model_name_or_path=nl_encoder_base_model,
            vocab_size=vocab_size,
            usePretrained=True,
        )
        self.sot.freeze()        # NL Encoder will be freezed as it is our source of truth 

        self.discriminator = Discriminator(
            input_dims=self.sot.config.hidden_size
        )

        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask):
        outs = self.generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        h = outs.last_hidden_state

        # Add gaussian noise to the latent representations
        u = self.sample_noise(input_ids.size(0))          
        u = u.type_as(h)

        return h + u
    
    def sample_noise(self, batch_size):
        u = torch.randn((
            batch_size,
            self.hparams.max_seq_len,
            self.generator.config.hidden_size,
        ))

        return u  
    
    def generator_step(self, input_ids, attention_mask):
        gen_embs = self(
            input_ids, 
            attention_mask
        )

        # Target -> All fake 
        valid = torch.ones(input_ids.size(0), 1)
        valid = valid.type_as(input_ids).float()

        d_outs = self.discriminator(gen_embs)

        g_loss = self.loss_fn(
            d_outs, 
            valid
        )

        # g_loss = -torch.mean(d_outs)

        g_acc = accuracy(d_outs, valid.int())

        self.log("acc/gen/train", g_acc)
        self.log("gen-loss/train", g_loss)
        
        return g_loss 
    
    def discriminator_step(self, input_ids, attention_mask, target_input_ids, target_attention_mask):
        valid = torch.ones(input_ids.size(0), 1)
        valid = valid.type_as(input_ids).float()
        
        h = self.sot(
            target_input_ids, 
            target_attention_mask,
        ).last_hidden_state

        d_outs = self.discriminator(h)

        loss_real = self.loss_fn(
            d_outs,
            valid
        )

        # loss_real = -torch.mean(d_outs)

        real_acc = accuracy(d_outs, valid.int())

        fake = torch.zeros(input_ids.size(0), 1)
        fake = fake.type_as(input_ids).float()

        gen_embs = self(
            input_ids, 
            attention_mask
        )

        d_outs = self.discriminator(gen_embs)
        loss_fake = self.loss_fn(
            d_outs, 
            fake
        )

        # loss_fake = torch.mean(d_outs)

        fake_acc = accuracy(d_outs, fake.int())

        self.log("acc/dis-fake/train", fake_acc)
        self.log("acc/dis-real/train", real_acc)

        d_loss = loss_real + loss_fake


        self.log("dis/real-loss/train", loss_real)
        self.log("dis/fake-loss/train", loss_fake)
        self.log("dis-loss/train", d_loss) 

        return d_loss 

    def training_step(self, batch, batch_idx, optimizer_idx) -> STEP_OUTPUT:
        
        # train generator 
        if optimizer_idx == 0:
            # Generated samples 
            return self.generator_step(
                batch['input_ids'], 
                batch['attention_mask']
            ) 
        
        # train discriminator 
        if optimizer_idx == 1:
            return self.discriminator_step(
                batch['input_ids'], 
                batch['attention_mask'], 
                batch['target_input_ids'], 
                batch['target_attention_mask']
            ) 
    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass
    
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        pass
    
    def configure_optimizers(self):

        # n_critic = 5
        # b1 = 0.5 
        # b2 = 0.999

        g_opt = torch.optim.AdamW(
            params=self.generator.parameters(),
            lr=self.hparams.g_learning_rate,
            weight_decay=self.hparams.g_weight_decay, 
            # betas=(b1, b2)
        )

        d_opt = torch.optim.AdamW(
            params=self.discriminator.parameters(),
            lr=self.hparams.d_learning_rate,
            weight_decay=self.hparams.d_weight_decay, 
            # betas=(b1, b2)
        )

        # g_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     g_opt, 
        #     T_0=100,
        #     verbose=True
        # )

        # d_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     d_opt,
        #     T_0=200,
        #     verbose=True
        # )

        # return [{'optimizer': g_opt, 'frequency': 1}, {'optimizer': d_opt, 'frequency': n_critic}] # [g_lr_scheduler, d_lr_scheduler]

        return [g_opt, d_opt], []

    def save(self, generator_path=PATH_SAVE_PL_ENCODER):
        self.generator.save(path=generator_path)
        
        return generator_path


class PL2NL(pl.LightningModule):
    def __init__(
        self,
        encoder_path: str = PATH_SAVE_NL_ENCODER,
        decoder_path: str = PATH_SAVE_NL_DECODER,
        vocab_size: str = VOCAB_SIZE,
        path_base_models: str = PATH_BASE_MODELS,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = PLEncoder(
            model_name_or_path=encoder_path,
            vocab_size=vocab_size,
            usePretrained=True,
            path_base_models=path_base_models,
        )

        self.decoder = NLDecoder(
            model_name_or_path=decoder_path,
            vocab_size=vocab_size,
            usePretrained=True,
            path_base_models=path_base_models
        )
    
    def forward(self, input_ids, attention_mask, target_input_ids, target_attention_mask):
        h = self.encoder(
            input_ids, 
            attention_mask
        ).last_hidden_state



    def training_step(self, *args, **kwargs) -> STEP_OUTPUT:
        return super().training_step(*args, **kwargs)
    
    def validation_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().validation_step(*args, **kwargs)
    
    def test_step(self, *args, **kwargs) -> Optional[STEP_OUTPUT]:
        return super().test_step(*args, **kwargs)