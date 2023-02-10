import math
from typing import Optional

import torch
import torch.nn as nn

from codecg.config import ModelConfig
from codecg.modules.attention import MultiHeadAttention


def sinusoidal_positions(length: int, d_model: int, device: torch.device) -> torch.Tensor:
    position = torch.arange(length, device=device, dtype=torch.float)[:, None]
    div = torch.exp(torch.arange(0, d_model, 2, device=device, dtype=torch.float) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(length, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div)
    pe[:, 1::2] = torch.cos(position * div)
    return pe


def causal_mask(length: int, device: torch.device) -> torch.Tensor:
    # True above the diagonal, i.e. at future positions
    return torch.triu(torch.ones(length, length, dtype=torch.bool, device=device), diagonal=1)


class Embeddings(nn.Module):
    def __init__(self, cfg: ModelConfig, absolute_positions: bool) -> None:
        super().__init__()
        self.d_model = cfg.d_model
        self.absolute_positions = absolute_positions
        self.tokens = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.tokens(ids) * math.sqrt(self.d_model)
        if self.absolute_positions:
            x = x + sinusoidal_positions(ids.size(1), self.d_model, ids.device)
        return self.dropout(x)


class FeedForward(nn.Sequential):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_ff, cfg.d_model),
        )


class EncoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, max_relative_position: int) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout, max_relative_position)
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        a, _ = self.self_attn(h, h, h, key_padding_mask=pad_mask)
        x = x + self.dropout(a)
        return x + self.dropout(self.ff(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.cross_attn = MultiHeadAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.ff = FeedForward(cfg)
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.norm3 = nn.LayerNorm(cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(
        self,
        y: torch.Tensor,
        memory: torch.Tensor,
        tgt_pad_mask: Optional[torch.Tensor],
        src_pad_mask: torch.Tensor,
        future_mask: torch.Tensor,
    ) -> torch.Tensor:
        h = self.norm1(y)
        a, _ = self.self_attn(h, h, h, key_padding_mask=tgt_pad_mask, attn_mask=future_mask)
        y = y + self.dropout(a)

        h = self.norm2(y)
        a, _ = self.cross_attn(h, memory, memory, key_padding_mask=src_pad_mask)
        y = y + self.dropout(a)

        return y + self.dropout(self.ff(self.norm3(y)))


class Encoder(nn.Module):
    """
        Pre-norm Transformer encoder. Positions are absolute (sinusoidal), relative
        (clipped, in every self-attention layer) or both, set by cfg.encoder_positions
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        assert cfg.encoder_positions in ("absolute", "relative", "both")

        self.pad_id = cfg.pad_id
        use_absolute = cfg.encoder_positions in ("absolute", "both")
        max_rel = cfg.max_relative_position if cfg.encoder_positions in ("relative", "both") else 0

        self.embed = Embeddings(cfg, absolute_positions=use_absolute)
        self.layers = nn.ModuleList([EncoderLayer(cfg, max_rel) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        pad_mask = ids.eq(self.pad_id)
        x = self.embed(ids)
        for layer in self.layers:
            x = layer(x, pad_mask)
        return self.norm(x)


class Decoder(nn.Module):
    """
        Pre-norm Transformer decoder with absolute positions. Returns hidden states,
        projection to the vocabulary is left to the model using it
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.pad_id = cfg.pad_id
        self.embed = Embeddings(cfg, absolute_positions=True)
        self.layers = nn.ModuleList([DecoderLayer(cfg) for _ in range(cfg.n_layers)])
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, ids: torch.Tensor, memory: torch.Tensor, src_pad_mask: torch.Tensor) -> torch.Tensor:
        tgt_pad_mask = ids.eq(self.pad_id)
        future_mask = causal_mask(ids.size(1), ids.device)
        y = self.embed(ids)
        for layer in self.layers:
            y = layer(y, memory, tgt_pad_mask, src_pad_mask, future_mask)
        return self.norm(y)
