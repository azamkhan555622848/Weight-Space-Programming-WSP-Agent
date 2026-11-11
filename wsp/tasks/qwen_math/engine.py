"""GPU engine for the language-model task: the model with swappable weight edits, sampled generation with common
random numbers, and teacher-forced reads (log-probs, top-k, hidden states, MLP activations).

Sampling follows vLLM's order (presence penalty on generated tokens -> temperature -> top-k -> top-p) and draws each
token with the Gumbel-max trick, where the Gumbel noise of token t at step s of item i is a hash of (seed_i, s, t).
Two models answering the same item therefore see the same noise for every candidate token: their answers coincide
until their distributions differ enough to change a choice (a coupling at least as tight as the paper's shared vLLM
seeds), and every answer is reproducible regardless of batch composition, up to bf16 kernel noise.
"""
from __future__ import annotations

import gc
import time

import numpy as np
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, LogitsProcessor, LogitsProcessorList

STOP_IDS = (248044, 248046)        # <|endoftext|>, <|im_end|>
_SDPA_ORIGINAL = None


def grouped_decode_attention(on: bool = True):
    """Decoding steps of the full-attention layers without materializing the repeated keys/values.

    With a padding mask, transformers' SDPA path expands the 2 key/value heads to the 8 query heads (repeat_kv) at
    every decoding step: at 12 answers x 27k tokens that is 2 x 1.2 GB per layer per step, which fragmented GPU memory
    until a 32,768-token batch ran out of it, and made each step ~10x slower than needed. A decoding step has one query
    position, so the 4 query heads of a group can instead be passed as 4 query rows against their shared key/value head
    (same mask for all of them, no causal pattern needed). Measured: bitwise-identical outputs, 1.7 vs 18 ms per call
    and <1 MB vs 2.8 GB extra memory at 30k tokens. Prefill and teacher-forced reads (more than one query position) keep
    the original path."""
    global _SDPA_ORIGINAL
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers.integrations.sdpa_attention import sdpa_attention_forward
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if _SDPA_ORIGINAL is None:
        _SDPA_ORIGINAL = sdpa_attention_forward
    if not on:
        ALL_ATTENTION_FUNCTIONS["sdpa"] = _SDPA_ORIGINAL
        return

    def attention(module, query, key, value, attention_mask, dropout=0.0, scaling=None, is_causal=None, **kwargs):
        B, H, Lq, D = query.shape
        KVH = key.shape[1]
        if (Lq != 1 or KVH == H or H % KVH or kwargs.get("output_attentions", False)
                or getattr(module, "sliding_window", None)):
            return _SDPA_ORIGINAL(module, query, key, value, attention_mask, dropout=dropout, scaling=scaling,
                                  is_causal=is_causal, **kwargs)
        mask = attention_mask[:, :, :, :key.shape[-2]] if attention_mask is not None else None
        with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):      # the kernel the original path ran (bitwise equal)
            out = torch.nn.functional.scaled_dot_product_attention(
                query.reshape(B, KVH, H // KVH, D), key, value, attn_mask=mask, dropout_p=dropout, scale=scaling,
                is_causal=False)
        return out.reshape(B, H, 1, D).transpose(1, 2).contiguous(), None

    ALL_ATTENTION_FUNCTIONS["sdpa"] = attention
from ...sampling import CRNSampler, hash_uniform  # noqa: E402,F401  (moved; still importable from here)


class Engine:
    def __init__(self, model_dir: str, device: str = "cuda", log=print, gpu_headroom_gb: float = 1.0):
        self.log, self.device = log, device
        t0 = time.time()
        if str(device).startswith("cuda"):
            # Cap PyTorch below the physical GPU memory. Without a cap, the Windows driver's "sysmem fallback" silently
            # spills allocations past 12 GB into system RAM (measured: 3 GB spilled at batch 32 x 8,192 tokens), which
            # slows decoding many times over and starves the 16 GB machine. With the cap, PyTorch frees its cache and
            # retries, and only then raises out-of-memory, which _gen_batch handles by splitting the batch.
            total = torch.cuda.get_device_properties(0).total_memory
            frac = max(0.5, (total - gpu_headroom_gb * 2**30) / total)
            torch.cuda.set_per_process_memory_fraction(frac)
            log(f"engine: GPU memory capped at {frac * total / 2**30:.1f} of {total / 2**30:.1f} GB")
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        # straight onto the GPU: loading on the CPU first needs ~2 GB more system RAM (16 GB machine)
        self.model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16,
                                                                 device_map=device).eval()
        try:
            import fla  # noqa: F401  (flash-linear-attention: the Gated DeltaNet kernels, ~2.4x faster decoding)
        except ImportError:
            log("engine: WARNING flash-linear-attention is not installed; decoding uses the slow torch fallback")
        if getattr(self.model.config, "_attn_implementation", None) == "sdpa":
            grouped_decode_attention(True)
        else:
            log(f"engine: attention is {getattr(self.model.config, '_attn_implementation', None)!r}, not sdpa; "
                "decoding keeps the default path")
        self.params = dict(self.model.named_parameters())
        self.lm = self.model.model.language_model
        self.n_layers = len(self.lm.layers)
        self.originals: dict[str, torch.Tensor] = {}       # name -> original bf16 copy (CPU) of edited tensors
        self.applied: dict[str, str] = {}                  # name -> fingerprint of the edit applied now
        log(f"engine: loaded {model_dir} in {time.time() - t0:.0f} s, {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ------------------------------------------------------------------------------------------------ edits
    def set_weights(self, edits: dict[str, np.ndarray] | None):
        """Make the model compute with `edits` ({name: float32 array}) on top of the original weights; None or {} =
        the original model. Tensors not in `edits` are restored."""
        edits = edits or {}
        with torch.no_grad():
            for n in list(self.applied):
                if n not in edits:
                    self.params[n].copy_(self.originals[n].to(self.device))
                    del self.applied[n]
            for n, a in edits.items():
                if n not in self.params:
                    raise KeyError(f"unknown tensor {n}")
                p = self.params[n]
                if tuple(a.shape) != tuple(p.shape):
                    raise ValueError(f"{n}: shape {a.shape} != {tuple(p.shape)}")
                fp = f"{a.shape}:{float(np.asarray(a, np.float64).sum())}:{float(np.abs(a).max())}"
                if self.applied.get(n) == fp:
                    continue
                if n not in self.originals:
                    self.originals[n] = p.detach().to("cpu").clone()
                p.copy_(torch.as_tensor(np.asarray(a, np.float32)).to(self.device, torch.bfloat16))
                self.applied[n] = fp

    # ------------------------------------------------------------------------------------------------ prompts
    def chat_ids(self, user: str, thinking: bool) -> list[int]:
        txt = self.tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=thinking)
        return self.tok(txt, add_special_tokens=False)["input_ids"]

    # ------------------------------------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate(self, prompts: list[list[int]], seeds: list[int], sampling: dict, max_new_tokens: int,
                 batch_size: int = 32, progress=None, on_done=None) -> list[dict]:
        """-> one {"ids": [...], "finish": "stop"|"length", "n_tokens"} per prompt (same order). `on_done(i, result)` is
        called for every prompt as soon as its batch finishes (so callers can save results incrementally)."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))      # similar lengths share a batch
        out: list = [None] * len(prompts)
        for b0 in range(0, len(order), batch_size):
            self._gen_batch(order[b0:b0 + batch_size], prompts, seeds, sampling, max_new_tokens, out)
            if on_done:
                for i in order[b0:b0 + batch_size]:
                    on_done(i, out[i])
            if progress:
                progress(min(b0 + batch_size, len(order)), len(order))
        return out

    def _gen_batch(self, idx, prompts, seeds, sampling, max_new_tokens, out):
        """Generate one batch; on running out of GPU memory, free it and do the two halves separately (the sampled
        answers do not depend on the batch, so the split changes nothing but the time)."""
        try:
            self._gen_batch_once(idx, prompts, seeds, sampling, max_new_tokens, out)
            return
        except (torch.OutOfMemoryError, torch.AcceleratorError) as e:
            if "out of memory" not in str(e).lower() or len(idx) == 1:
                raise
        # Retry only after leaving the except block: inside it, the exception's traceback keeps the failed attempt's
        # tensors (its whole KV cache) alive, so every smaller retry ran out of memory as well (seen at 32,768 tokens).
        gc.collect()
        torch.cuda.empty_cache()
        h = len(idx) // 2
        self.log(f"engine: out of GPU memory at batch {len(idx)}; splitting into {h} + {len(idx) - h}")
        self._gen_batch(idx[:h], prompts, seeds, sampling, max_new_tokens, out)
        self._gen_batch(idx[h:], prompts, seeds, sampling, max_new_tokens, out)

    def _gen_batch_once(self, idx, prompts, seeds, sampling, max_new_tokens, out):
        L = max(len(prompts[i]) for i in idx)
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else STOP_IDS[0]
        ids = torch.full((len(idx), L), pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, L - len(prompts[i]):] = torch.tensor(prompts[i])
            att[r, L - len(prompts[i]):] = 1
        proc = CRNSampler([seeds[i] for i in idx], L, sampling.get("temperature", 0.6), sampling.get("top_p", 1.0),
                          sampling.get("top_k", 0), sampling.get("presence_penalty", 0.0), device=self.device)
        res = self.model.generate(input_ids=ids.to(self.device), attention_mask=att.to(self.device),
                                  do_sample=False, max_new_tokens=max_new_tokens,
                                  logits_processor=LogitsProcessorList([proc]), eos_token_id=list(STOP_IDS),
                                  pad_token_id=pad, temperature=None, top_p=None, top_k=None)
        gen = res[:, L:].cpu()
        del res
        for r, i in enumerate(idx):
            g = gen[r].tolist()
            stop = next((j for j, t in enumerate(g) if t in STOP_IDS), None)
            if stop is not None:
                out[i] = {"ids": g[:stop], "finish": "stop", "n_tokens": stop + 1}
            else:
                out[i] = {"ids": g, "finish": "length", "n_tokens": len(g)}

    # ------------------------------------------------------------------------------------------------ teacher forcing
    @torch.no_grad()
    def read(self, seqs: list[list[int]], hidden_layers=(), mlp_layers=(), topk: int = 0, logprobs: bool = True,
             max_tokens_per_batch: int = 16384) -> dict:
        """Teacher-forced read of token sequences. -> {"lengths", "token_logprobs" (N, T-1), "topk_ids"/"topk_logprobs"
        (N, T, k), "hidden_<l>" (N, T, 1024), "mlp_act_<l>" (N, T, 3584)}, right-padded (float16 for activations)."""
        N, T = len(seqs), max(len(s) for s in seqs)
        out: dict = {"lengths": np.array([len(s) for s in seqs])}
        if logprobs:
            out["token_logprobs"] = np.zeros((N, T - 1), np.float32)
        if topk:
            out["topk_ids"] = np.zeros((N, T, topk), np.int32)
            out["topk_logprobs"] = np.zeros((N, T, topk), np.float32)
        for l in hidden_layers:
            out[f"hidden_{l}"] = np.zeros((N, T, self.lm.config.hidden_size), np.float16)
        for l in mlp_layers:
            out[f"mlp_act_{l}"] = np.zeros((N, T, self.lm.config.intermediate_size), np.float16)
        cap: dict = {}
        hooks = [self.lm.layers[l].mlp.down_proj.register_forward_pre_hook(
            lambda m, a, l=l: cap.__setitem__(l, a[0].detach())) for l in mlp_layers]
        try:
            order = sorted(range(N), key=lambda i: -len(seqs[i]))
            i = 0
            while i < N:
                L = len(seqs[order[i]])
                bs = max(1, max_tokens_per_batch // L)
                idx = order[i:i + bs]
                i += bs
                ids = torch.full((len(idx), L), STOP_IDS[0], dtype=torch.long)
                att = torch.zeros((len(idx), L), dtype=torch.long)
                for r, j in enumerate(idx):
                    ids[r, :len(seqs[j])] = torch.tensor(seqs[j])
                    att[r, :len(seqs[j])] = 1
                ids, att = ids.to(self.device), att.to(self.device)
                cap.clear()
                # the text backbone only: full-vocabulary logits for every position would not fit (T x 248,320), and
                # the multimodal wrapper's 3-D position ids keep state from the last generate() call
                o = self.lm(input_ids=ids, attention_mask=att, output_hidden_states=bool(hidden_layers), use_cache=False)
                h = o.last_hidden_state
                for r, j in enumerate(idx):
                    n = len(seqs[j])
                    # LM head in position chunks; skipped when only activations are wanted (a 2,048-position chunk is
                    # a 2 GB float32 log-softmax)
                    for a in (range(0, n, 2048) if (logprobs or topk) else ()):
                        b = min(n, a + 2048)
                        lp = torch.log_softmax(self.model.lm_head(h[r, a:b]).float(), -1)
                        if logprobs:
                            nxt = ids[r, a + 1:min(b + 1, n)]
                            out["token_logprobs"][j, a:a + len(nxt)] = lp[:len(nxt)].gather(-1, nxt[:, None]).squeeze(-1).cpu().numpy()
                        if topk:
                            v, t = lp.topk(topk, -1)
                            out["topk_ids"][j, a:b] = t.cpu().numpy()
                            out["topk_logprobs"][j, a:b] = v.cpu().numpy()
                        del lp
                    for l in hidden_layers:
                        out[f"hidden_{l}"][j, :n] = o.hidden_states[l][r, :n].float().cpu().numpy()
                    for l in mlp_layers:
                        out[f"mlp_act_{l}"][j, :n] = cap[l][r, :n].float().cpu().numpy()
                del o, h
        finally:
            for h in hooks:
                h.remove()
        return out
