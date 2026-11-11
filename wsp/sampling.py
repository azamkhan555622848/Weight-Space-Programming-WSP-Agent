"""Sampling with common random numbers, for any Hugging Face generate() call.

Sampling follows vLLM's order (presence penalty on generated tokens -> temperature -> top-k -> top-p) and draws each
token with the Gumbel-max trick, where the Gumbel noise of token t at step s of item i is a hash of (seed_i, s, t).
Two models answering the same item therefore see the same noise for every candidate token: their answers coincide
until their distributions differ enough to change a choice, and every answer is reproducible whatever batch it is in,
up to floating-point noise."""
from __future__ import annotations

import torch
from transformers import LogitsProcessor

_M1, _M2, _M3 = 0x9E3779B97F4A7C15, 0xBF58476D1CE4E5B9, 0x94D049BB133111EB


def _signed(x: int) -> int:
    x &= (1 << 64) - 1
    return x - (1 << 64) if x >= 1 << 63 else x


def hash_uniform(seed: torch.Tensor, step: int, ids: torch.Tensor) -> torch.Tensor:
    """Counter-based uniform(0,1) noise: seed (B,) int64, ids (B, k) int64 -> (B, k) float64. SplitMix64-style mixing
    in wrapping int64 arithmetic (identical on CPU and GPU)."""
    x = seed[:, None] * _signed(_M1) + _signed((step + 1) * _M2) + ids * _signed(_M3)
    for m in (_signed(_M2), _signed(_M3)):
        x = x ^ ((x >> 31) & ((1 << 33) - 1))
        x = x * m
    x = x ^ ((x >> 29) & ((1 << 35) - 1))
    u = ((x >> 11) & ((1 << 53) - 1)).double() / float(1 << 53)
    return u.clamp(1e-12, 1 - 1e-12)


class CRNSampler(LogitsProcessor):
    """Samples one token per row and returns logits that make greedy decoding pick exactly that token."""

    def __init__(self, seeds, prompt_len, temperature=0.6, top_p=0.95, top_k=20, presence_penalty=0.0,
                 vocab_size=None, device="cuda"):
        self.seeds = torch.as_tensor(seeds, dtype=torch.int64, device=device)
        self.prompt_len, self.T, self.top_p, self.top_k = prompt_len, temperature, top_p, top_k
        self.pp = presence_penalty
        self.seen = None
        self.step = 0

    def __call__(self, input_ids, scores):
        scores = scores.float()
        B, V = scores.shape
        if self.pp:
            if self.seen is None:
                self.seen = torch.zeros(B, V, dtype=torch.bool, device=scores.device)
            if input_ids.shape[1] > self.prompt_len:
                self.seen[torch.arange(B, device=scores.device), input_ids[:, -1]] = True
            scores = scores - self.pp * self.seen.float()
        if self.T <= 0:                                     # greedy
            choice = scores.argmax(-1)
        else:
            k = min(self.top_k if self.top_k and self.top_k > 0 else V, V)
            vals, ids = torch.topk(scores / self.T, k, dim=-1)          # sorted, descending
            if self.top_p < 1.0:
                probs = torch.softmax(vals, -1)
                keep = (probs.cumsum(-1) - probs) < self.top_p          # always keeps the first
                vals = vals.masked_fill(~keep, float("-inf"))
            g = -torch.log(-torch.log(hash_uniform(self.seeds, self.step, ids)))
            choice = ids.gather(1, (vals.double() + g).argmax(-1, keepdim=True)).squeeze(1)
        self.step += 1
        out = torch.full_like(scores, float("-inf"))
        out[torch.arange(B, device=scores.device), choice] = 0.0
        return out
