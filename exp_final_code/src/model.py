from typing import Optional

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT

from torchmetrics.functional import bleu_score

from transformers import (
    AutoConfig, 
    AutoModel,
    AutoTokenizer,
    EncoderDecoderConfig,
    EncoderDecoderModel, 
    GPT2Config, 
    RobertaConfig
)

from config import (
    LEARNING_RATE,
    NL_DECODER_BASE_MODEL,
    PATH_BASE_MODELS,
    PATH_SAVE_NL_DECODER,
    PATH_SAVE_NL_ENCODER,
    PATH_SAVE_PL_ENCODER,
    PL_ENCODER_BASE_MODEL,
    TOKENIZER_MODEL,
    VOCAB_SIZE,
    WEIGHT_DECAY    
)


class NL2NL(pl.LightningModule):
    """
        NL2NL is an autoencoder which learns an embedding space for the objective D(E(x)) = x 
        Essentially this can be used for continuous latent representations for text 
    """
    def __init__(
        self, 
        learning_rate: float = LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        vocab_size: int = VOCAB_SIZE,
        decoder_start_token_id: int = 1, 
        pad_token_id: int = 0,
    ) -> None: 

        super(NL2NL, self).__init__()
        self.save_hyperparameters()

        # Encoder config 
        self.config_encoder = RobertaConfig()

        # Set config according to CodeT5 tokenizer
        self.config_encoder.vocab_size = vocab_size
        self.config_encoder.pad_token_id = pad_token_id
        
        # Decoder config
        self.config_decoder = GPT2Config()

        # Set config according to CodeT5 Tokenizer
        self.config_decoder.vocab_size = vocab_size
        self.config_decoder.pad_token_id = pad_token_id
        self.config_decoder.decoder_start_token_id = decoder_start_token_id

        # Seq2Seq config 
        self.config = EncoderDecoderConfig.from_encoder_decoder_configs(
            encoder_config=self.config_encoder, 
            decoder_config=self.config_decoder
        )

        # Set config according to CodeT5 tokenizer
        self.config.decoder_start_token_id = decoder_start_token_id
        self.config.pad_token_id = pad_token_id
        self.config.vocab_size = vocab_size

        # Initialize Seq2Seq Model
        self.model = EncoderDecoderModel(config=self.config)
    

    def forward(self, **kwargs) -> torch.TensorType:
        return self.model(**kwargs)
    
    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['target_input_ids']
        )

        self.log('loss/train', outputs.loss)

        return outputs.loss

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        outputs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['target_input_ids']
        )

        self.log('loss/val', outputs.loss)

        return outputs.loss 
    

    def configure_optimizers(self):
        # AdamW Optimizer 
        optimizer = torch.optim.AdamW(
            params=self.parameters(), 
            lr=self.hparams.learning_rate, 
            weight_decay=self.hparams.weight_decay
        )

        return optimizer

    def save(self, encoder_path: str = PATH_SAVE_NL_ENCODER, decoder_path: str = PATH_SAVE_NL_DECODER):
        # Save encoder
        self.model.encoder.save_pretrained(encoder_path)
        
        # Save decoder
        self.model.decoder.save_pretrained(decoder_path)



class Discriminator(nn.Module):
    """
        Discriminator for GAN
    """
    def __init__(
        self, 
        input_dims: int,
    ) -> None:

        super().__init__()

        # BiLSTM 
        self.bi_lstm = nn.LSTM(
            input_size=input_dims, 
            hidden_size=256, 
            batch_first=True, 
            bidirectional=True 
        )

        # MLP Layer
        self.net = nn.Sequential(
            nn.Linear(512, 128), 
            nn.GELU(), 
            nn.Linear(128, 32), 
            nn.GELU(), 
            nn.Linear(32, 8), 
            nn.GELU(), 
            nn.Linear(8, 1)
        )
    
    def forward(self, h):
        """
            Returns logits for latent representations 
        """
        
        # BiLSTM outputs
        outs, _ = self.bi_lstm(h)

        # [CLS] token representation
        y = self.net(outs[:, 0])     
        return y 


class NLPLSpace(pl.LightningModule):
    """
        NLPLSpace makes our PL Encoder learn the NL Encoders embedding space
        Goal: p(PL Encoder) == p(NL Encoder)    [Distribution will be same Approximately]
    """
    def __init__(
        self, 
        path_sot_model: str = PATH_SAVE_NL_ENCODER,
        encoder_name_or_path: str = PL_ENCODER_BASE_MODEL,
        g_lr: float = LEARNING_RATE, 
        d_lr: float = LEARNING_RATE,
        g_wd: float = WEIGHT_DECAY,
        d_wd: float = WEIGHT_DECAY,
        vocab_size: int = VOCAB_SIZE,
        decoder_start_token_id: int = 1,
        pad_token_id: int = 0,
        path_base_models: str = PATH_BASE_MODELS
    ) -> None: 

        super(NLPLSpace, self).__init__()
        self.save_hyperparameters()

        
        # Source of Truth 
        self.sot = AutoModel.from_pretrained(path_sot_model)

        # Freeze our source of truth for inference
        for p in self.sot.parameters():
            p.requires_grad = False

        # Initialize Generator Config
        self.config_generator = AutoConfig.from_pretrained(
            encoder_name_or_path,
            cache_dir=path_base_models
        )

        # Set Config according to CodeT5 Tokenizer
        self.config_generator.vocab_size = vocab_size
        self.config_generator.pad_token_id = pad_token_id
        
        # Initialize generator
        self.generator = AutoModel.from_config(self.config_generator)

        # Initialize Discriminator 
        self.discriminator = Discriminator(
            self.config_generator.hidden_size
        )

        # Binary Cross Entropy Loss 
        self.BCELoss = nn.BCEWithLogitsLoss()
        self.MSELoss = nn.MSELoss()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outs =  self.generator(
            input_ids=input_ids, 
            attention_mask=attention_mask, 
            output_hidden_states=True
        )

        return outs.last_hidden_state

    def generator_step(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, target_input_ids, target_attention_mask, mode:str):

        # SOT = Target embeddings for MSE Loss
        # h = self.sot(
            # target_input_ids, 
            # target_attention_mask
        # ).last_hidden_state 

        # y = h[:, 0]      # CLS Token embedding

        # Generated Embeddings
        gen_embs = self(input_ids, attention_mask)
        
        # y_cap = gen_embs[:, 0]  # CLS token embedding

        # Targets : All Fake  
        # [ones will take care of negative sign, else by taking 0s as target you need to add minus sign]
        valid = torch.ones(input_ids.size(0), 1)
        
        # Bring on same device
        valid = valid.type_as(input_ids)

        # Discriminator logis 
        d_outs = self.discriminator(gen_embs)

        # Binary Cross Entropy Loss + MSE combined
        g_loss = self.BCELoss(d_outs, valid.float())

        self.log(f'loss/generator-{mode}', g_loss)

        return g_loss 

    def discriminator_step(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        target_input_ids: torch.Tensor, 
        target_attention_mask: torch.Tensor, 
        mode: str,
    ):

        # Feed Real 
      
        h = self.sot(target_input_ids, target_attention_mask).last_hidden_state
        d_outs = self.discriminator(h)

        # Target CLS token representation
        y = h[:, 0]

        valid = torch.ones(input_ids.size(0), 1)
        valid = valid.type_as(input_ids)

        loss_real = self.BCELoss(d_outs, valid.float())
        

        # Feed Fake 

        gen_embs = self(input_ids, attention_mask)
        d_outs = self.discriminator(gen_embs)

        # Predicted CLS token representation
        y_cap = gen_embs[:, 0]

        fake = torch.zeros(input_ids.size(0), 1)
        fake = fake.type_as(input_ids)

        loss_fake = self.BCELoss(d_outs, fake.float())


        # Discriminator Loss 
        d_loss = (loss_real + loss_fake) / 2 + 0.6 * self.MSELoss(y_cap, y)

        self.log(f'loss/discriminator-{mode}', d_loss)
        self.log(f'loss/real-dis-{mode}', loss_real)
        self.log(f'loss/fake-dis-{mode}', loss_fake)

        return d_loss 

    def training_step(self, batch, batch_idx, optimizer_idx) -> STEP_OUTPUT:
        # Train generator 
        if optimizer_idx == 0:
            return self.generator_step(
                batch['input_ids'], 
                batch['attention_mask'], 
                batch['target_input_ids'], 
                batch['target_attention_mask'],
                'train'
            )
        
        # Train discriminator
        else:
            return self.discriminator_step(
                batch['input_ids'], 
                batch['attention_mask'], 
                batch['target_input_ids'], 
                batch['target_attention_mask'], 
                'train'
            )

    
    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        # Generator Validity
        self.generator_step(
            batch['input_ids'], 
            batch['attention_mask'], 
            batch['target_input_ids'], 
            batch['target_attention_mask'],
            'train'
        )
        
        # Discriminator Validity
        self.discriminator_step(
            batch['input_ids'], 
            batch['attention_mask'], 
            batch['target_input_ids'], 
            batch['target_attention_mask'], 
            'val'
        )
    
    def configure_optimizers(self):
        """
            Configure optimizers and schedulers for generator and discriminator
        """

        # Generator optimizer 
        g_opt = torch.optim.AdamW(
            params=self.generator.parameters(), 
            lr=self.hparams.g_lr, 
            weight_decay=self.hparams.g_wd
        )

        # Discriminator optimizer 
        d_opt = torch.optim.AdamW(
            params=self.discriminator.parameters(), 
            lr=self.hparams.d_lr, 
            weight_decay=self.hparams.d_wd
        )

        return [g_opt, d_opt], []
    

    def save(self, encoder_path: str = PATH_SAVE_PL_ENCODER):

        # Save PL Encoder 
        self.generator.save_pretrained(encoder_path)

    


class PL2NL(pl.LightningModule):
    def __init__(
        self,
        encoder_name_or_path: str = PL_ENCODER_BASE_MODEL, 
        decoder_name_or_path: str = NL_DECODER_BASE_MODEL,
        tokenizer: str = TOKENIZER_MODEL,
        vocab_size: int = VOCAB_SIZE,
        decoder_start_token_id: int = 1, 
        pad_token_id: int = 0, 
        path_base_models: str = PATH_BASE_MODELS,
    ) -> None: 
        
        super(PL2NL, self).__init__()
        self.save_hyperparameters()

        # Initialize Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path=tokenizer,
            useFast=True,
            cache_dir=path_base_models
        )

        # Initialize our Novel Model
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(
            encoder_pretrained_model_name_or_path=encoder_name_or_path,
            decoder_pretrained_model_name_or_path=decoder_name_or_path,
        )

        # Configure model according to CodeT5 
        self.model.config.vocab_size = vocab_size
        self.model.config.decoder_start_token_id = decoder_start_token_id
        self.model.config.pad_token_id = pad_token_id

    def forward(self, input_ids, attention_mask, labels):
        outputs = self.model(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            labels=labels
        ) 
        
        # Prediction scores before softmax
        logits = outputs.logits 
        return F.softmax(logits, dim=-1)

    

    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        probs = self(
            input_ids=batch['input_ids'], 
            attention_mask=batch['attention_mask'], 
            labels=batch['target_input_ids']
        )

        # Token with max probability
        preds = probs.argmax(dim=-1)

        # Get text
        reference_comments = []
        target_comments = self.tokenizer.batch_decode(batch['target_input_ids'], skip_special_tokens=True)
        for comment in target_comments: 
            reference_comments.append([comment])

        gen_comments = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # BLEU Score 
        score = bleu_score(gen_comments, reference_comments)
        smoothed_score = bleu_score(gen_comments, reference_comments, smooth=True)

        self.log('BLEU Score/test', score)
        self.log('Smoothed BLUE Score/test', smoothed_score)

