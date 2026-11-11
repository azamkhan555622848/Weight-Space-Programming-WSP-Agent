"""Forward passes for any Hugging Face causal language model: sampled generation with common random numbers, and teacher-forced
reads (token log-probabilities, top-k, hidden states, activations at any module), with weight edits swapped in and out and
optional interventions on activations. The model-specific engine of the Qwen benchmark (wsp/tasks/qwen_math/engine.py) is the
tuned version of this one."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import re
import time

import numpy as np

from ...grading import loop_signals
from ..base import ForwardError
from .hooks import Hooks, SiteError, check_interventions, file_name, find_module, parse_site

CHAT_END_TOKENS = ("<|im_end|>", "<|eot_id|>", "<end_of_turn>", "<|end|>", "<|endoftext|>")
NAME = re.compile(r"^[\w.-]+$")


def item_seed(text: str, j: int) -> int:
    """The sampling seed of sample j of a prompt: fixed by the prompt, shared by every model and pass."""
    return int.from_bytes(hashlib.sha256(f"{text}#{j}".encode()).digest()[:7], "little")


def pick_class(model_dir: str, names: set):
    """The transformers class to load a model with, and an empty instance of it (no weights, no memory) for listing modules.

    Some checkpoints can be loaded as more than one class with different tensor names: a text model shipped inside a multimodal
    wrapper loads as a causal LM with the wrapper's prefix dropped. Edits are saved under the checkpoint's names, so the class
    chosen is the one whose tensor names match the checkpoint best."""
    from transformers import AutoModelForCausalLM
    candidates = [AutoModelForCausalLM]
    try:
        from transformers import AutoModelForImageTextToText
        candidates.append(AutoModelForImageTextToText)
    except ImportError:
        pass
    try:
        from accelerate import init_empty_weights
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_dir)
    except Exception:  # noqa: BLE001 - without a listing the first class is tried and the names are checked at the first edit
        return candidates[0], None
    best = (-1.0, candidates[0], None)
    for cls in candidates:
        try:
            with init_empty_weights():
                model = cls.from_config(config)
            score = len(set(model.state_dict()) & names) / max(1, len(names))
        except Exception:  # noqa: BLE001 - this class doesn't fit this model
            continue
        if score > best[0]:
            best = (score, cls, model)
    return best[1], best[2]


class LMEngine:
    def __init__(self, model_dir: str, device: str | None = None, log=print, gpu_headroom_gb: float = 1.0, names: set | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.log = torch, log
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        t0 = time.time()
        cuda = str(self.device).startswith("cuda")
        if cuda:
            # Below the physical GPU memory, so PyTorch frees its cache and raises out-of-memory (handled by splitting the
            # batch) instead of the Windows driver silently spilling into system RAM, which is many times slower.
            total = torch.cuda.get_device_properties(0).total_memory
            torch.cuda.set_per_process_memory_fraction(max(0.5, (total - gpu_headroom_gb * 2 ** 30) / total))
        self.tok = AutoTokenizer.from_pretrained(model_dir)
        self.tok.padding_side = "left"
        kw = dict(dtype="auto" if cuda else torch.float32, device_map=self.device)
        cls = pick_class(model_dir, names)[0] if names else AutoModelForCausalLM
        self.model = cls.from_pretrained(model_dir, **kw).eval()
        self.params = dict(self.model.state_dict())                   # name -> tensor sharing the model's storage
        if names and len(set(self.params) & names) < 0.5 * len(names):
            log(f"engine: WARNING only {len(set(self.params) & names)} of {len(names)} tensors of the checkpoint are in the loaded "
                f"{cls.__name__}; edits to the others can't be applied")
        self.originals: dict = {}
        self.applied: dict[str, str] = {}
        text = getattr(self.model.config, "text_config", None) or self.model.config       # a wrapper keeps the text model's here
        self.n_layers = int(getattr(text, "num_hidden_layers", None) or getattr(text, "n_layer", None) or 0)
        self.vocab = int(getattr(text, "vocab_size", None) or len(self.tok))
        gc_ = self.model.generation_config
        stop = set()
        for src in (getattr(gc_, "eos_token_id", None), self.tok.eos_token_id):
            if isinstance(src, int):
                stop.add(src)
            elif isinstance(src, (list, tuple)):
                stop.update(int(x) for x in src)
        if getattr(self.tok, "chat_template", None):
            for t in CHAT_END_TOKENS:
                i = self.tok.convert_tokens_to_ids(t)
                if isinstance(i, int) and i >= 0 and i != self.tok.unk_token_id:
                    stop.add(i)
        self.stop_ids = sorted(stop)
        self.pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else (self.stop_ids[0] if self.stop_ids else 0)
        self.has_chat = bool(getattr(self.tok, "chat_template", None))
        self.default_sampling = ({"temperature": float(getattr(gc_, "temperature", None) or 1.0),
                                  "top_p": float(getattr(gc_, "top_p", None) or 1.0), "top_k": int(getattr(gc_, "top_k", None) or 0)}
                                 if getattr(gc_, "do_sample", False) else {"temperature": 0.0})
        self.split = self._find_split()
        log(f"engine: loaded {model_dir} on {self.device} in {time.time() - t0:.0f} s")

    def _find_split(self):
        """(backbone, output layer) if the model's logits are exactly the output layer applied to the backbone's last hidden
        state. Reads then apply the output layer a few positions at a time: full-vocabulary logits for every position of every
        text would not fit in memory. Checked on a tiny input, so a model with an extra step falls back to the plain path."""
        torch = self.torch
        head = self.model.get_output_embeddings()
        base = getattr(self.model, "base_model", None)
        if head is None or base is None or base is self.model:
            return None
        base = getattr(base, "language_model", base)
        try:
            with torch.no_grad():
                ids = torch.tensor([[self.pad, self.pad, self.pad]], device=self.device)
                att = torch.ones_like(ids)
                full = self.model(input_ids=ids, attention_mask=att).logits.float()
                part = head(base(input_ids=ids, attention_mask=att)[0]).float()
            ok = bool((full - part).abs().max() <= 1e-3 * (1 + full.abs().max()))
        except Exception:  # noqa: BLE001 - any oddity means: use the plain path
            ok = False
        return (base, head) if ok else None

    # ------------------------------------------------------------------------------------------------ edits
    def set_weights(self, edits: dict | None):
        """Make the model compute with `edits` ({name: float32 array}) on top of the original weights; None or {} = the
        original model. Tensors not in `edits` are restored."""
        torch = self.torch
        edits = edits or {}
        with torch.no_grad():
            for n in list(self.applied):
                if n not in edits:
                    self.params[n].copy_(self.originals[n].to(self.device))
                    del self.applied[n]
            for n, a in edits.items():
                if n not in self.params:
                    raise KeyError(f"{n} is not a tensor of the loaded model")
                p = self.params[n]
                if tuple(a.shape) != tuple(p.shape):
                    raise ValueError(f"{n}: shape {tuple(a.shape)} != {tuple(p.shape)}")
                fp = f"{a.shape}:{float(np.asarray(a, np.float64).sum())}:{float(np.abs(a).max()) if a.size else 0}"
                if self.applied.get(n) == fp:
                    continue
                if n not in self.originals:
                    self.originals[n] = p.detach().to("cpu").clone()
                p.copy_(torch.as_tensor(np.asarray(a, np.float32)).to(self.device, p.dtype))
                self.applied[n] = fp

    # ------------------------------------------------------------------------------------------------ prompts
    def encode(self, item, generation_prompt: bool = True, template_kwargs: dict | None = None) -> list[int]:
        """Token ids of a raw text (a string) or of a chat (a list of {"role", "content"} messages, through the chat template)."""
        if isinstance(item, str):
            return self.tok(item)["input_ids"]
        if not self.has_chat:
            raise ValueError("this model has no chat template: give raw texts")
        text = self.tok.apply_chat_template(item, tokenize=False, add_generation_prompt=generation_prompt, **(template_kwargs or {}))
        return self.tok(text, add_special_tokens=False)["input_ids"]

    def decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    # ------------------------------------------------------------------------------------------------ generation
    def generate(self, prompts: list[list[int]], seeds: list[int], sampling: dict, max_new_tokens: int, batch_size: int = 16,
                 progress=None, interventions=()) -> list[dict]:
        """-> one {"ids", "finish": "stop" | "length", "n_tokens"} per prompt, in order."""
        order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))      # similar lengths share a batch
        out: list = [None] * len(prompts)
        for b0 in range(0, len(order), batch_size):
            self._gen_batch(order[b0:b0 + batch_size], prompts, seeds, sampling, max_new_tokens, out, interventions)
            if progress:
                progress(min(b0 + batch_size, len(order)), len(order))
        return out

    def _gen_batch(self, idx, prompts, seeds, sampling, max_new_tokens, out, interventions):
        """One batch; on running out of GPU memory, free it and do the two halves separately (the sampled answers don't depend
        on the batch they are in, up to floating-point noise)."""
        torch = self.torch
        try:
            self._gen_batch_once(idx, prompts, seeds, sampling, max_new_tokens, out, interventions)
            return
        except (torch.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" not in str(e).lower() or len(idx) == 1:
                raise
        gc.collect()                       # outside the except block: its traceback would keep the failed batch alive
        if str(self.device).startswith("cuda"):
            torch.cuda.empty_cache()
        h = len(idx) // 2
        self.log(f"engine: out of memory at batch {len(idx)}; splitting into {h} + {len(idx) - h}")
        self._gen_batch(idx[:h], prompts, seeds, sampling, max_new_tokens, out, interventions)
        self._gen_batch(idx[h:], prompts, seeds, sampling, max_new_tokens, out, interventions)

    def _gen_batch_once(self, idx, prompts, seeds, sampling, max_new_tokens, out, interventions):
        torch = self.torch
        from transformers import LogitsProcessorList

        from ...sampling import CRNSampler
        L = max(len(prompts[i]) for i in idx)
        ids = torch.full((len(idx), L), self.pad, dtype=torch.long)
        att = torch.zeros((len(idx), L), dtype=torch.long)
        for r, i in enumerate(idx):
            ids[r, L - len(prompts[i]):] = torch.tensor(prompts[i])
            att[r, L - len(prompts[i]):] = 1
        proc = CRNSampler([seeds[i] for i in idx], L, sampling.get("temperature", 0.0), sampling.get("top_p", 1.0),
                          sampling.get("top_k", 0), sampling.get("presence_penalty", 0.0), device=self.device)
        with torch.no_grad(), Hooks(self.model, (), interventions):
            res = self.model.generate(input_ids=ids.to(self.device), attention_mask=att.to(self.device), do_sample=False,
                                      max_new_tokens=max_new_tokens, logits_processor=LogitsProcessorList([proc]),
                                      eos_token_id=self.stop_ids or None, pad_token_id=self.pad, temperature=None, top_p=None,
                                      top_k=None)
        gen = res[:, L:].cpu()
        stops = set(self.stop_ids)
        for r, i in enumerate(idx):
            g = gen[r].tolist()
            stop = next((j for j, t in enumerate(g) if t in stops), None)
            if stop is not None:
                out[i] = {"ids": g[:stop], "finish": "stop", "n_tokens": stop + 1}
            else:
                out[i] = {"ids": g, "finish": "length", "n_tokens": len(g)}

    # ------------------------------------------------------------------------------------------------ teacher forcing
    def read(self, seqs: list[list[int]], capture=(), hidden_layers=(), topk: int = 0, logprobs: bool = True, interventions=(),
             max_tokens_per_batch: int = 16384, max_act_values: int = 1_000_000_000, progress=None) -> dict:
        """Teacher-forced read of token sequences. -> {"lengths", "tokens" (N, T; -1 = padding), "token_logprobs" (N, T-1),
        "topk_ids" / "topk_logprobs" (N, T, k), "hidden_<l>" (N, T, D), "act::<site>" (N, T, ...)}: right-padded, activations
        in float16."""
        torch = self.torch
        N, T = len(seqs), max(len(s) for s in seqs)
        out: dict = {"lengths": np.array([len(s) for s in seqs]), "tokens": np.full((N, T), -1, np.int32)}
        for j, s in enumerate(seqs):
            out["tokens"][j, :len(s)] = s
        if logprobs:
            out["token_logprobs"] = np.zeros((N, max(T - 1, 0)), np.float32)
        if topk:
            out["topk_ids"] = np.zeros((N, T, topk), np.int32)
            out["topk_logprobs"] = np.zeros((N, T, topk), np.float32)
        used = 0

        def room(key, shape):
            nonlocal used
            if key not in out:
                used += int(np.prod(shape))
                if used > max_act_values:
                    raise ValueError(f"the activations asked for hold more than {max_act_values:,} values: ask for fewer texts, "
                                     "layers or sites")
                out[key] = np.zeros(shape, np.float16)
            return out[key]
        order = sorted(range(N), key=lambda i: -len(seqs[i]))
        i = 0
        while i < N:
            L = len(seqs[order[i]])
            bs = max(1, max_tokens_per_batch // L) if self.split else max(1, int(2e8 // (L * self.vocab)))
            idx = order[i:i + bs]
            i += bs
            ids = torch.full((len(idx), L), self.pad, dtype=torch.long)
            att = torch.zeros((len(idx), L), dtype=torch.long)
            for r, j in enumerate(idx):
                ids[r, :len(seqs[j])] = torch.tensor(seqs[j])
                att[r, :len(seqs[j])] = 1
            ids, att = ids.to(self.device), att.to(self.device)
            with torch.no_grad(), Hooks(self.model, capture, interventions) as hk:
                kw = dict(input_ids=ids, attention_mask=att, output_hidden_states=bool(hidden_layers), use_cache=False)
                if self.split:
                    o = self.split[0](**kw)
                    h, logits = o[0], None
                else:
                    o = self.model(**kw)
                    h, logits = None, o.logits
                hidden = o.hidden_states if hidden_layers else None
                for r, j in enumerate(idx):
                    n = len(seqs[j])
                    for a in (range(0, n, 2048) if (logprobs or topk) else ()):
                        b = min(n, a + 2048)
                        lg = self.split[1](h[r, a:b]) if self.split else logits[r, a:b]
                        lp = torch.log_softmax(lg.float(), -1)
                        if logprobs:
                            nxt = ids[r, a + 1:min(b + 1, n)]
                            if len(nxt):
                                out["token_logprobs"][j, a:a + len(nxt)] = lp[:len(nxt)].gather(-1, nxt[:, None]).squeeze(-1).cpu().numpy()
                        if topk:
                            v, t = lp.topk(topk, -1)
                            out["topk_ids"][j, a:b] = t.cpu().numpy()
                            out["topk_logprobs"][j, a:b] = v.cpu().numpy()
                        del lp
                    for l in hidden_layers:
                        x = hidden[l]
                        room(f"hidden_{l}", (N, T, x.shape[-1]))[j, :n] = x[r, :n].float().cpu().numpy()
                    for site in capture:
                        if site not in hk.seen:
                            raise ValueError(f"the module at {site} was not called in this pass")
                        x = hk.seen[site]
                        if x.dim() >= 2 and x.shape[0] == len(idx) and x.shape[1] == L:
                            room(f"act::{site}", (N, T) + tuple(x.shape[2:]))[j, :n] = x[r, :n].float().cpu().numpy()
                        elif x.shape[0] == len(idx):
                            arr = room(f"act::{site}", (N,) + tuple(x.shape[1:]))
                            if arr.shape[1:] != tuple(x.shape[1:]):
                                raise ValueError(f"{site} gives arrays of different shapes for texts of different lengths; send "
                                                 "texts of one length to capture it")
                            arr[j] = x[r].float().cpu().numpy()
                        else:
                            raise ValueError(f"{site} gives a tensor of shape {tuple(x.shape)}, which has no row per text")
            del o
            if progress:
                progress(min(i, N), N)
        return out


# ---------------------------------------------------------------------------------------------------- the forward tool
class LMTool:
    """The forward-pass tool of a language-model target: validates a request, runs it and writes its files."""

    TEXT = """ A forward pass is one of two things, and each counts as one pass:
 (a) GENERATION: the model continues up to {gen_max} prompts x samples x models. Reply with exactly one block that starts with ```forward and contains JSON:
 {{"type": "generate", "model": "original", "edited", "both", or a list such as ["original", "edited", "cand_a"], "prompts": ["raw text", ...] or "chats": [[{{"role": "user", "content": "..."}}], ...]{chat_note}, "n": samples per prompt (default 1), "max_tokens": at most {max_tokens:,} (default {default_tokens}), "sampling": {{"temperature": 0 for greedy, "top_p", "top_k", "presence_penalty"}} (optional; default: {default_sampling}), "capture": [sites] (optional), "interventions": [...] (optional)}}
 "original" runs the unedited weights and "edited" your current edits. Any other name is a folder of .npy edits in your work folder (for example a copy of edited/ you saved as cand_a/), so one pass can compare several candidate edits or strengths on the same prompts. The random numbers of every sample are fixed by the prompt and the sample index and shared by all models: two models with the same weights give the same text, and an edit changes a text only where it changes the model's choices. You get, per model, the mean length, the share cut at max_tokens and how many of those end in a repetition loop, and how many texts are identical to the first model's; the texts are in gen_<k>.jsonl (with several models: gen_<k>_<model>.jsonl), one JSON object per line with prompt (its index), sample, n_tokens, finish and text. With "capture", each text's activations at those sites are saved twice, at the last prompt token and averaged over the generated tokens: gen_<k>_acts/ (index.json lists the files).
 (b) TEACHER-FORCED: the model reads up to {tf_max_texts} texts of up to {tf_max_tokens:,} tokens each, {tf_total_tokens:,} tokens in total (no generation). Reply with a ```forward block containing JSON:
 {{"type": "teacher_forced", "model": as above, "texts": ["raw text", ...] or "chats": [[...], ...], "return": {{"token_logprobs": true, "topk": k (at most 20), "hidden_states": [layer indices; 0 = embeddings], "capture": [sites]}}, "interventions": [...] (optional)}}
 The results are .npy files in fwd_<k>/ (with several models: fwd_<k>/<model>/), right-padded to the longest text, with meta.json and tokens.npy (the token ids, -1 = padding); activations are float16.
 SITES AND INTERVENTIONS. A site is a module path from model/modules.txt, optionally followed by ":input" for what enters the module instead of what leaves it, for example "{example_site}". An intervention changes an activation while the pass runs, without touching the weights, which is a cheap way to test a hypothesis before you write the edit: {{"site": "...", "op": "zero" | "scale" | "add" | "project_out" | "replace", "value": number, "vector": "file.npy" (a direction or values you saved in your work folder), "units": [indices] (optional), "dim": the axis of the units (default -1)}}.
 A request that fails does not count against the budget, and an identical request on unchanged edits is not run."""

    def __init__(self, task, model_dir: str):
        self.task, self.model_dir = task, model_dir
        self.engine: LMEngine | None = None

    def eng(self) -> LMEngine:
        if self.engine is None:
            self.engine = LMEngine(self.model_dir, device=self.task.cfg.get("device"), log=self.task.log,
                                   names=set(self.task.store.names()))
        return self.engine

    def close(self):
        self.engine = None

    def tools_text(self, p: dict, has_chat: bool, example_site: str, default_sampling: str) -> str:
        return self.TEXT.format(gen_max=p["gen_max"], max_tokens=p["max_tokens"], default_tokens=min(256, p["max_tokens"]),
                                tf_max_texts=p["tf_max_texts"], tf_max_tokens=p["tf_max_tokens"], tf_total_tokens=p["tf_total_tokens"],
                                chat_note=" (through the model's chat template)" if has_chat else " (this model has no chat template: use prompts)",
                                example_site=example_site, default_sampling=default_sampling)

    # ---------------------------------------------------------------------------------------------- shared pieces
    def _models(self, req) -> list[str]:
        m = req.get("model", "edited")
        models = ["original", "edited"] if m == "both" else m if isinstance(m, list) else [m]
        if not models or not all(isinstance(x, str) and NAME.match(x) for x in models) or len(set(models)) != len(models):
            raise ForwardError('"model" must be "original", "edited", "both", or a list of distinct names')
        return models

    def _items(self, req, E: LMEngine, generation_prompt: bool):
        if ("prompts" in req or "texts" in req) == ("chats" in req):
            raise ForwardError('give either raw texts ("prompts" / "texts") or "chats", not both and not neither')
        raw = req.get("chats") or req.get("prompts") or req.get("texts")
        if not isinstance(raw, list) or not raw:
            raise ForwardError("the texts must be a non-empty list")
        ids = []
        for it in raw:
            if "chats" in req:
                if not (isinstance(it, list) and it and all(isinstance(m, dict) and "role" in m and "content" in m for m in it)):
                    raise ForwardError('each chat is a list of {"role", "content"} messages')
            elif not isinstance(it, str) or not it:
                raise ForwardError("each text must be a non-empty string")
            try:
                ids.append(E.encode(it, generation_prompt, req.get("template_kwargs")))
            except Exception as e:  # noqa: BLE001
                raise ForwardError(f"can't tokenize an item: {e}") from None
        return raw, ids

    def _hooks(self, req, E: LMEngine, capture):
        try:
            for s in capture:
                find_module(E.model, parse_site(s)[0])
            vectors = {}
            for iv in req.get("interventions") or []:
                if isinstance(iv, dict) and isinstance(iv.get("vector"), str):
                    vectors[iv["vector"]] = E.torch.as_tensor(self.task.read_array(iv["vector"]))
            return check_interventions(E.model, req.get("interventions"), vectors)
        except SiteError as e:
            raise ForwardError(str(e)) from None

    def _set(self, E: LMEngine, model: str):
        try:
            E.set_weights(self.task.edits_of(model))
        except (KeyError, ValueError) as e:
            raise ForwardError(f"{model}: {e}") from None

    # ---------------------------------------------------------------------------------------------- the two passes
    def run(self, req: dict, k: int):
        t = req.get("type")
        if t not in ("generate", "teacher_forced"):
            raise ForwardError('"type" must be "generate" or "teacher_forced"')
        return self._generate(req, k) if t == "generate" else self._teacher_forced(req, k)

    def _generate(self, req, k):
        p, out_dir = self.task.p, self.task._out
        models = self._models(req)
        E = self.eng()
        raw, prompts = self._items(req, E, True)
        n = req.get("n", 1)
        mx = req.get("max_tokens", min(256, p["max_tokens"]))
        if not isinstance(n, int) or n < 1 or not isinstance(mx, int) or not 1 <= mx <= p["max_tokens"]:
            raise ForwardError(f'"n" must be a positive integer and "max_tokens" between 1 and {p["max_tokens"]}')
        if len(prompts) * n * len(models) > p["gen_max"]:
            raise ForwardError(f"prompts x n x models = {len(prompts) * n * len(models)}: at most {p['gen_max']} per pass")
        sampling = {**E.default_sampling, **(req.get("sampling") or {})}
        capture = req.get("capture") or []
        ivs = self._hooks(req, E, capture)
        rows = [(i, j) for i in range(len(prompts)) for j in range(n)]
        seeds = [item_seed(json.dumps(raw[i], sort_keys=True), j) for i, j in rows]
        summary = {"type": "generate", "models": models, "generations": len(rows) * len(models), "sampling": sampling}
        texts = {}
        for m in models:
            self._set(E, m)
            gens = E.generate([prompts[i] for i, _ in rows], seeds, sampling, mx, p["batch_size"], interventions=ivs,
                              progress=lambda d, t, m=m: self.task.progress(f"pass {k}: {d}/{t} texts ({m})"))
            sfx = f"_{m}" if len(models) > 1 else ""
            recs = []
            with open(os.path.join(out_dir, f"gen_{k}{sfx}.jsonl"), "w", encoding="utf-8") as f:
                for (i, j), g in zip(rows, gens):
                    text = E.decode(g["ids"])
                    recs.append({"prompt": i, "sample": j, "n_tokens": g["n_tokens"], "finish": g["finish"], "text": text})
                    f.write(json.dumps(recs[-1], ensure_ascii=False) + "\n")
            texts[m] = [r["text"] for r in recs]
            cut = [r for r in recs if r["finish"] == "length"]
            summary[m] = {"mean_tokens": round(float(np.mean([r["n_tokens"] for r in recs])), 1),
                          "cut_at_max_tokens": round(len(cut) / len(recs), 3),
                          "cut_texts_ending_in_a_loop": sum(any(loop_signals(r["text"]).values()) for r in cut)}
            if m != models[0]:
                summary[m][f"identical_to_{models[0]}"] = sum(a == b for a, b in zip(texts[m], texts[models[0]]))
            if capture:
                d = os.path.join(out_dir, f"gen_{k}_acts" + sfx)
                os.makedirs(d)
                seqs = [prompts[i] + g["ids"] for (i, _), g in zip(rows, gens)]
                acts = E.read(seqs, capture=capture, logprobs=False, interventions=ivs, max_act_values=p["act_values"])
                files = {}
                for site in capture:
                    a = acts[f"act::{site}"]
                    if a.ndim < 3:
                        raise ForwardError(f"{site} has no position axis, so it can't be captured during generation")
                    last = np.stack([a[r, len(prompts[i]) - 1] for r, (i, _) in enumerate(rows)])
                    mean = np.stack([a[r, len(prompts[i]):len(s)].astype(np.float32).mean(0) if len(s) > len(prompts[i])
                                     else np.zeros(a.shape[2:], np.float32) for r, ((i, _), s) in enumerate(zip(rows, seqs))])
                    for name, arr in (("last_prompt", last), ("mean_generated", mean)):
                        fn = f"{file_name(site)}__{name}.npy"
                        np.save(os.path.join(d, fn), arr.astype(np.float16))
                        files[fn] = list(arr.shape)
                json.dump({"rows": [{"prompt": i, "sample": j} for i, j in rows], "files": files},
                          open(os.path.join(d, "index.json"), "w"), indent=1)
        where = f"gen_{k}.jsonl" if len(models) == 1 else ", ".join(f"gen_{k}_{m}.jsonl" for m in models)
        return summary, where + (f" and gen_{k}_acts*/" if capture else "")

    def _teacher_forced(self, req, k):
        p, out_dir = self.task.p, self.task._out
        models = self._models(req)
        E = self.eng()
        _, seqs = self._items(req, E, False)
        lens = [len(s) for s in seqs]
        if len(seqs) > p["tf_max_texts"] or max(lens) > p["tf_max_tokens"] or sum(lens) * len(models) > p["tf_total_tokens"]:
            raise ForwardError(f"{len(seqs)} texts, longest {max(lens)} tokens, {sum(lens) * len(models):,} tokens in total: at "
                               f"most {p['tf_max_texts']} texts, {p['tf_max_tokens']:,} tokens each, {p['tf_total_tokens']:,} in total")
        if min(lens) < 1:
            raise ForwardError("a text has no tokens")
        ret = req.get("return") or {"token_logprobs": True}
        topk = int(ret.get("topk") or 0)
        layers = ret.get("hidden_states") or []
        capture = ret.get("capture") or req.get("capture") or []
        if not 0 <= topk <= 20 or not all(isinstance(l, int) and 0 <= l <= E.n_layers for l in layers):
            raise ForwardError(f'"topk" is at most 20 and "hidden_states" are layer indices 0..{E.n_layers}')
        ivs = self._hooks(req, E, capture)
        summary = {"type": "teacher_forced", "models": models, "texts": len(seqs), "lengths": lens[:64]}
        for m in models:
            self._set(E, m)
            try:
                res = E.read(seqs, capture=capture, hidden_layers=layers, topk=topk, logprobs=bool(ret.get("token_logprobs", True)),
                             interventions=ivs, max_act_values=p["act_values"],
                             progress=lambda d, t, m=m: self.task.progress(f"pass {k}: {d}/{t} texts ({m})"))
            except ValueError as e:
                raise ForwardError(str(e)) from None
            d = os.path.join(out_dir, f"fwd_{k}", m) if len(models) > 1 else os.path.join(out_dir, f"fwd_{k}")
            os.makedirs(d, exist_ok=True)
            files = {}
            for key, arr in res.items():
                fn = (f"act_{file_name(key[5:])}" if key.startswith("act::") else key) + ".npy"
                np.save(os.path.join(d, fn), arr)
                files[fn] = list(arr.shape)
            json.dump({"request": req, "model": m, "files": files, "padding": "right; tokens.npy holds -1 where a text has ended"},
                      open(os.path.join(d, "meta.json"), "w"), indent=1)
            if res.get("token_logprobs") is not None and "token_logprobs" in res:
                mask = np.arange(res["token_logprobs"].shape[1])[None] < (res["lengths"][:, None] - 1)
                summary[m] = {"mean_token_logprob": round(float(res["token_logprobs"][mask].mean()), 4) if mask.any() else None,
                              "files": files}
            else:
                summary[m] = {"files": files}
        return summary, f"fwd_{k}/"
