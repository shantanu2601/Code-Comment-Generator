import math
from typing import Optional, Tuple

import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    """
        Multi-head attention with optional relative position representations
        (Shaw et al., 2018). Pairwise distances are clipped to [-k, k], and a learned
        embedding per distance is added to both keys and values.
    """
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        dropout: float = 0.0,
        max_relative_position: int = 0,
    ) -> None:

        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.max_relative_position = max_relative_position

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

        if max_relative_position > 0:
            n_distances = 2 * max_relative_position + 1
            self.rel_k = nn.Embedding(n_distances, self.d_head)
            self.rel_v = nn.Embedding(n_distances, self.d_head)

    def relative_indices(self, q_len: int, k_len: int, device: torch.device) -> torch.Tensor:
        # Distance j - i, clipped, then shifted to be a valid embedding index
        pos_q = torch.arange(q_len, device=device)[:, None]
        pos_k = torch.arange(k_len, device=device)[None, :]
        k = self.max_relative_position
        return (pos_k - pos_q).clamp(-k, k) + k

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, L, _ = x.shape
        return x.view(B, L, self.n_heads, self.d_head).transpose(1, 2)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            key_padding_mask: [B, Lk], True at padding positions
            attn_mask: [Lq, Lk], True at positions that may not be attended to
            Returns the output [B, Lq, D] and the attention weights [B, H, Lq, Lk]
        """
        B, Lq, D = query.shape
        Lk = key.size(1)

        q = self._split(self.q_proj(query))
        k = self._split(self.k_proj(key))
        v = self._split(self.v_proj(value))

        scores = q @ k.transpose(-2, -1)

        if self.max_relative_position > 0:
            rel = self.relative_indices(Lq, Lk, query.device)
            scores = scores + torch.einsum("bhqd,qkd->bhqk", q, self.rel_k(rel))

        scores = scores / math.sqrt(self.d_head)

        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask[None, None], float("-inf"))
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))

        weights = scores.softmax(dim=-1)
        dropped = self.dropout(weights)

        out = dropped @ v
        if self.max_relative_position > 0:
            out = out + torch.einsum("bhqk,qkd->bhqd", dropped, self.rel_v(rel))

        out = out.transpose(1, 2).reshape(B, Lq, D)
        return self.out_proj(out), weights
