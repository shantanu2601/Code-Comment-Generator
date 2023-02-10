from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from codecg.config import ModelConfig
from codecg.modules import Decoder, Encoder


def masked_mean(states: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
    keep = (~pad_mask).unsqueeze(-1).float()
    return (states * keep).sum(dim=1) / keep.sum(dim=1).clamp_min(1.0)


class TextAutoencoder(nn.Module):
    """
        Learns a latent space for comments with the objective D(E(x)) = x.
        Its encoder later serves as the source of truth for alignment, and its
        decoder can initialise the summarizer's decoder
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.generator = nn.Linear(cfg.d_model, cfg.vocab_size)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        memory = self.encoder(text)
        h = self.decoder(text[:, :-1], memory, text.eq(self.cfg.pad_id))
        logits = self.generator(h)
        return F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            text[:, 1:].reshape(-1),
            ignore_index=self.cfg.pad_id,
        )


class Discriminator(nn.Module):
    """
        Tells comment-encoder states (real) apart from code-encoder states (fake)
    """
    def __init__(self, input_dims: int) -> None:
        super().__init__()

        # BiLSTM
        self.bi_lstm = nn.LSTM(
            input_size=input_dims,
            hidden_size=256,
            batch_first=True,
            bidirectional=True,
        )

        # MLP Layer
        self.net = nn.Sequential(
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Linear(128, 32),
            nn.GELU(),
            nn.Linear(32, 8),
            nn.GELU(),
            nn.Linear(8, 1),
        )

    def forward(self, states: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        """
            Returns one logit per sequence: [B, 1]
        """
        lengths = (~pad_mask).sum(dim=1).clamp_min(1).cpu()
        packed = pack_padded_sequence(states, lengths, batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.bi_lstm(packed)

        # Final forward and backward hidden states
        h = torch.cat([h_n[-2], h_n[-1]], dim=-1)
        return self.net(h)


class AdversarialAligner(nn.Module):
    """
        Trains a code encoder so its states fall in the comment encoder's latent
        space. The comment encoder is frozen. The code encoder acts as the generator
        and is pushed to fool the discriminator, plus an MSE term that pulls the
        pooled code representation towards the pooled representation of its comment
    """
    def __init__(self, text_encoder: Encoder, code_encoder: Encoder, cfg: ModelConfig, mse_weight: float = 0.6) -> None:
        super().__init__()
        self.cfg = cfg
        self.mse_weight = mse_weight

        # Source of truth, frozen
        self.text_encoder = text_encoder
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        self.code_encoder = code_encoder
        self.discriminator = Discriminator(cfg.d_model)

    def train(self, mode: bool = True) -> "AdversarialAligner":
        super().train(mode)
        # Keep the frozen encoder deterministic
        self.text_encoder.eval()
        return self

    @torch.no_grad()
    def _real(self, text: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        pad_mask = text.eq(self.cfg.pad_id)
        return self.text_encoder(text), pad_mask

    def generator_loss(self, code: torch.Tensor, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        code_mask = code.eq(self.cfg.pad_id)
        fake = self.code_encoder(code)
        real, text_mask = self._real(text)

        logits = self.discriminator(fake, code_mask)
        adv = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits))
        mse = F.mse_loss(masked_mean(fake, code_mask), masked_mean(real, text_mask))

        return {"loss": adv + self.mse_weight * mse, "adv": adv, "mse": mse}

    def discriminator_loss(self, code: torch.Tensor, text: torch.Tensor) -> Dict[str, torch.Tensor]:
        code_mask = code.eq(self.cfg.pad_id)
        with torch.no_grad():
            fake = self.code_encoder(code)
        real, text_mask = self._real(text)

        real_logits = self.discriminator(real, text_mask)
        fake_logits = self.discriminator(fake, code_mask)

        loss_real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        loss_fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))

        return {"loss": (loss_real + loss_fake) / 2, "real": loss_real, "fake": loss_fake}
