from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from codecg.config import ModelConfig
from codecg.modules import Decoder, Encoder, MultiHeadAttention


class Summarizer(nn.Module):
    """
        Encoder-decoder Transformer that generates a comment for a code snippet.

        With copy attention, an extra single-head attention layer sits on top of the
        decoder stack. Its weights over the source tokens form a copy distribution,
        and a learned gate mixes it with the vocabulary distribution:

            p(w) = p_gen * p_vocab(w) + (1 - p_gen) * sum_{i: src_i = w} a_i

        Code and comments share one subword vocabulary, so copied tokens map straight
        onto vocabulary ids and no extended vocabulary is needed
    """
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.encoder = Encoder(cfg)
        self.decoder = Decoder(cfg)
        self.generator = nn.Linear(cfg.d_model, cfg.vocab_size)

        if cfg.copy_attention:
            # No dropout, so the copy weights stay a proper distribution
            self.copy_attn = MultiHeadAttention(cfg.d_model, n_heads=1, dropout=0.0)
            self.copy_gate = nn.Linear(2 * cfg.d_model, 1)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        return self.encoder(src)

    def log_probs(self, tgt_in: torch.Tensor, memory: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
        """
            Returns log p(next token) for every target position: [B, T, V]
        """
        src_pad_mask = src.eq(self.cfg.pad_id)
        h = self.decoder(tgt_in, memory, src_pad_mask)
        vocab_log_probs = F.log_softmax(self.generator(h), dim=-1)

        if not self.cfg.copy_attention:
            return vocab_log_probs

        context, weights = self.copy_attn(h, memory, memory, key_padding_mask=src_pad_mask)
        copy_weights = weights.squeeze(1)  # [B, T, S]

        p_gen = torch.sigmoid(self.copy_gate(torch.cat([h, context], dim=-1)))  # [B, T, 1]

        B, T, S = copy_weights.shape
        copy_probs = torch.zeros_like(vocab_log_probs).scatter_add_(
            2, src[:, None, :].expand(B, T, S), copy_weights
        )

        probs = p_gen * vocab_log_probs.exp() + (1 - p_gen) * copy_probs
        return probs.clamp_min(1e-12).log()

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        """
            tgt starts with BOS and ends with EOS. Returns the mean token NLL
        """
        tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]
        log_probs = self.log_probs(tgt_in, self.encode(src), src)
        return F.nll_loss(
            log_probs.reshape(-1, log_probs.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=self.cfg.pad_id,
        )

    @torch.no_grad()
    def greedy(self, src: torch.Tensor, max_len: int) -> List[List[int]]:
        memory = self.encode(src)
        B = src.size(0)
        ys = torch.full((B, 1), self.cfg.bos_id, dtype=torch.long, device=src.device)
        done = torch.zeros(B, dtype=torch.bool, device=src.device)

        for _ in range(max_len):
            next_token = self.log_probs(ys, memory, src)[:, -1].argmax(dim=-1)
            next_token = next_token.masked_fill(done, self.cfg.pad_id)
            ys = torch.cat([ys, next_token[:, None]], dim=1)
            done |= next_token.eq(self.cfg.eos_id)
            if done.all():
                break

        return [self._strip(seq) for seq in ys[:, 1:].tolist()]

    @torch.no_grad()
    def beam_search(self, src: torch.Tensor, max_len: int, beam_size: int) -> List[List[int]]:
        if beam_size <= 1:
            return self.greedy(src, max_len)
        return [self._beam_search_one(src[i : i + 1], max_len, beam_size) for i in range(src.size(0))]

    def _beam_search_one(self, src: torch.Tensor, max_len: int, beam_size: int) -> List[int]:
        # Drop padding so the expanded source is as short as possible
        src = src[:, src[0].ne(self.cfg.pad_id)]
        memory = self.encode(src)

        beams = torch.full((1, 1), self.cfg.bos_id, dtype=torch.long, device=src.device)
        scores = torch.zeros(1, device=src.device)
        finished: List[Tuple[float, List[int]]] = []

        for _ in range(max_len):
            n = beams.size(0)
            step_log_probs = self.log_probs(beams, memory.expand(n, -1, -1), src.expand(n, -1))[:, -1]
            candidates = (scores[:, None] + step_log_probs).view(-1)
            top_scores, top_idx = candidates.topk(beam_size)

            vocab_size = step_log_probs.size(-1)
            # torch.div with rounding_mode, since `//` on tensors is deprecated in torch 1.12
            beam_idx = torch.div(top_idx, vocab_size, rounding_mode="floor")
            token_idx = top_idx % vocab_size
            new_beams = torch.cat([beams[beam_idx], token_idx[:, None]], dim=1)

            keep = token_idx.ne(self.cfg.eos_id)
            for seq, score in zip(new_beams[~keep].tolist(), top_scores[~keep].tolist()):
                # Length-normalised so that short outputs are not favoured
                finished.append((score / (len(seq) - 1), seq))

            if not keep.any():
                break
            beams, scores = new_beams[keep], top_scores[keep]

            # Stop once enough hypotheses have finished and none of the live beams
            # currently scores better than the worst of them
            if len(finished) >= beam_size:
                finished = sorted(finished, key=lambda x: x[0], reverse=True)[:beam_size]
                best_live = (scores / (beams.size(1) - 1)).max().item()
                if finished[-1][0] >= best_live:
                    break

        if not finished:
            finished = [(s / (len(seq) - 1), seq) for seq, s in zip(beams.tolist(), scores.tolist())]

        best = max(finished, key=lambda x: x[0])[1]
        return self._strip(best[1:])

    def _strip(self, seq: List[int]) -> List[int]:
        out = []
        for token in seq:
            if token in (self.cfg.eos_id, self.cfg.pad_id):
                break
            out.append(token)
        return out
