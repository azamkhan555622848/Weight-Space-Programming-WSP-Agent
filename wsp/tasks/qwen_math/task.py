"""LLM task: make Qwen3.5-0.8B better at math by editing its weights (the paper's Experiment 8), on one consumer GPU.

The task prompt, the kernel helpers (load / load_original / save_edit / drop_edit / edited / tok, refusals for the frozen
tensors and for whole-tensor rescaling) and the forward-pass tool (generation with the evaluation's sampling, paired
"both" comparisons, teacher-forced reads) follow the paper's turn-1 prompt (exp8_math/runs/p50c/turns/t001/prompt.txt).
What changes is the scale the hardware allows: the token cap, answers per pass and teacher-forced limits are parameters
(`cfg`), stated in the prompt, and the hidden evaluation uses the same protocol (scripts/eval_math.py).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil

import numpy as np

from ...grading import build_prompt, grade, loop_signals
from ..base import ForwardError, Task, TaskNotReady

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.5-0.8B")
PRACTICE = os.path.join(ROOT, "data", "math", "practice")
PREFIX = "model.language_model."
FROZEN = (PREFIX + "norm.weight", PREFIX + "embed_tokens.weight", "lm_head.weight")

# The protocol this GPU allows (RTX 4070 SUPER, HF generate at ~400-800 tok/s; the paper used vLLM and 32,768 tokens):
# thinking on (the base model's failure mode, looping in its thinking, is kept), 8,192 new tokens, 256 answers per pass;
# needs the fla kernels (flash-linear-attention + triton-windows): ~1,000 tok/s at batch 32 (batch 64 is slower).
DEFAULTS = {"thinking": True, "max_tokens": 8192, "gen_max": 256, "tf_max_texts": 512, "tf_max_tokens": 8192,
            "tf_total_tokens": 262144, "act_max_tokens": 4096, "batch_size": 16,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}}

PRELUDE = r'''
import json, os
import numpy as np
import torch
from safetensors import safe_open
from tokenizers import Tokenizer
np.set_printoptions(precision=4, suppress=True, linewidth=160)
tok = Tokenizer.from_file(__TOKENIZER__)

def _wsp_helpers():
    # the helpers keep their own state (absolute paths, the open weight file), so user code that reassigns globals
    # or changes the working directory cannot break them
    f = safe_open(__MODEL_FILE__, framework="pt")
    names = sorted(n for n in f.keys() if n.startswith("model.language_model."))
    frozen = __FROZEN__
    edir = os.path.join(os.getcwd(), "edited")

    def load_original(name):
        assert name in names, f"unknown tensor {name!r}"
        return f.get_tensor(name).float()

    def load(name):
        p = os.path.join(edir, name + ".npy")
        return torch.from_numpy(np.load(p)) if os.path.exists(p) else load_original(name)

    def save_edit(name, tensor):
        if name in frozen:
            raise ValueError(f"{name} is out of scope (the final norm and the embedding / output layer cannot be edited)")
        assert name in names, f"unknown tensor {name!r}"
        a = np.asarray(tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else tensor, np.float32)
        o = load_original(name).numpy()
        assert a.shape == o.shape, (name, a.shape, o.shape)
        assert np.isfinite(a).all(), f"{name}: non-finite values"
        oo = float((o.astype(np.float64) ** 2).sum())
        if oo > 0:
            c = float((a.astype(np.float64) * o).sum() / oo)
            if abs(c - 1) > 1e-4 and np.linalg.norm(a - c * o) <= 1e-3 * max(np.linalg.norm(a), 1e-30):
                raise ValueError(f"{name}: this edit only multiplies the whole tensor by {c:.4f}; whole-tensor "
                                 "rescaling is out of scope (edit specific rows, columns, directions or units)")
        os.makedirs(edir, exist_ok=True)
        p = os.path.join(edir, name + ".npy")
        if np.array_equal(a, o):
            if os.path.exists(p):
                os.remove(p)
            print(f"{name}: identical to the original; no edit kept")
            return
        np.save(p, a)
        ch = (a != o)
        print(f"saved edit of {name}: {int(ch.sum()):,} of {a.size:,} entries changed"
              + (f" ({int(ch.any(1).sum())} rows, {int(ch.any(0).sum())} columns)" if a.ndim == 2 else "")
              + f", relative change {np.linalg.norm(a - o) / max(np.linalg.norm(o), 1e-30):.3%}")

    def drop_edit(name):
        p = os.path.join(edir, name + ".npy")
        if os.path.exists(p):
            os.remove(p)
            print(f"{name}: edit dropped (original restored)")
        else:
            print(f"{name}: no edit to drop")

    def edited():
        return sorted(x[:-4] for x in os.listdir(edir) if x.endswith(".npy")) if os.path.isdir(edir) else []

    return names, load_original, load, save_edit, drop_edit, edited

NAMES, load_original, load, save_edit, drop_edit, edited = _wsp_helpers()
del _wsp_helpers
'''

TASK_PROMPT = """TASK. You are given the weights of a language model, Qwen3.5-0.8B (text part: 24 layers of width 1,024; layers 3, 7, 11, 15, 19 and 23 are full-attention layers with 8 query heads and 2 key/value heads of size 256 and a gated output, the other 18 are Gated DeltaNet linear-attention layers with 16 heads of size 128; each layer has a SwiGLU MLP of width 3,584; RMSNorm; vocabulary 248,320; input embedding and output layer tied). Your job is to make it better at mathematics by EDITING ITS WEIGHTS DIRECTLY: find out what goes wrong inside the network when it fails at math problems and repair that with targeted changes.

HOW IT WILL BE MEASURED. After you finish, your edited model and the unedited model are evaluated on MATH-500 and GSM8K with this fixed protocol: the chat template with thinking {thinking_text}; the user message is the problem, a newline and "Please reason step by step, and put your final answer within \\boxed{{}}."; sampling with temperature {T}, top_p {top_p}, top_k {top_k}, presence penalty {pp} and at most {max_tokens:,} new tokens{for_what}; the last \\boxed{{}} answer is graded against the reference answer. The config, tokenizer, chat template and decoding settings are fixed: only weight values can change (same tensor names and shapes). You never see these benchmarks.

WHAT COUNTS. We want edits that come from understanding the network: find out what goes wrong when it fails at math problems, identify the components responsible (specific neurons, heads, directions or circuits) and change them in a way you can justify with evidence. Repairing a specific failure mechanism is in scope: for example, if the model often fails because it repeats itself and never finishes, finding and fixing what drives that repetition counts. Out of scope, and refused by save_edit: the final norm (model.language_model.norm.weight), the input embedding / output layer (model.language_model.embed_tokens.weight, lm_head.weight), and any edit that only multiplies a whole tensor by a constant. Global knobs (making the whole output distribution sharper or flatter) are not what we are looking for, and neither is making the model skip its thinking or close it early regardless of the problem; repairing a specific failure inside the thinking (such as repetition) is in scope.

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU), run in your work folder (the current directory): variables survive between your code blocks, like a notebook. The model's weights are read through the helpers below (do not open the model files otherwise); practice problems are in practice/. Available: numpy, scipy, PyTorch (CPU), pandas, sympy, networkx, safetensors, tokenizers. Defined in advance: NAMES (all tensor names), load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor), drop_edit(name), edited() and tok (the tokenizer, a tokenizers.Tokenizer). Gradients are disabled. To run code, reply with exactly one block that starts with ```python and nothing else; if its last line is an expression, its value is shown. One execution may take up to {timeout} seconds; this machine has {ram_gb} GB of memory in total, so load tensors as you need them rather than all at once; output is cut at {max_out:,} characters. You may run code up to {max_runs} times. If the kernel is restarted (time or memory limit), its variables are lost and you are told so.
2. Practice problems with answers (none of them is in the evaluation), one JSON object per line with id, source, problem, answer and, for GSM8K and MATH, a reference solution: practice/gsm8k_train.jsonl (7,469 grade-school problems), practice/math_train.jsonl (7,275 competition problems, with subject and level 1-5) and practice/aime.jsonl (714 AIME problems, 1983-2023, answers only). Choose whichever serve your experiments.
3. A forward-pass tool with a budget of {budget} forward passes in total. A forward pass is one of two things, and each counts as one pass:
 (a) GENERATION: the model answers up to {gen_max} generations (problems x samples) with the evaluation's protocol (the same template and sampling). Reply with exactly one block that starts with ```forward and contains JSON:
 {{"type": "generate", "model": "original", "edited" or "both", "problems": ["<practice id>", ...], "prompts": ["<your own user message>", ...] (optional), "n": samples per item (items x n <= {gen_max}; with "both" <= {gen_half} per model), "max_tokens": at most {max_tokens}, "activations": {{"layers": [layer indices 0..24], "mlp": true or false}} (optional)}}
 "original" runs the unedited weights, "edited" your current edits, and "both" runs both on the same items and returns a paired comparison (the accuracy difference edited - original over items with its standard error, and how many samples went from wrong to right and back). The random numbers of every sample are fixed by the item and the sample index and shared by both models: two models with the same weights give the same answer, and an edit changes an answer only where it changes the model's choices. Every answer to a practice problem is graded. You get the accuracy (overall and by source), the mean length, the share cut at max_tokens, the failure counts and, for cut answers, the share that end in a loop; texts, extracted answers and grades are in gen_<k>.jsonl (with "both": gen_<k>_original.jsonl and gen_<k>_edited.jsonl). With "activations", for every generated answer the model's residual stream at the requested layers (0 = embeddings, l = after layer l) is saved, taken at the last token of the prompt and averaged over the answer's first {act_max_tokens:,} tokens (and the MLPs' down_proj inputs with "mlp": true), with each answer's grade: gen_<k>_acts/ (index.json lists the files and samples; with "both" one folder per model).
 (b) TEACHER-FORCED: the model reads up to {tf_max_texts} texts of up to {tf_max_tokens:,} tokens each, at most {tf_total_tokens:,} tokens in total (no generation). Reply with a ```forward block containing JSON:
 {{"type": "teacher_forced", "model": "original" or "edited", "texts": ["raw text", ...] or "chats": [[{{"role": "user", "content": "..."}}, {{"role": "assistant", "content": "..."}}], ...] (add "enable_thinking": true for the thinking template), "return": {{"token_logprobs": true, "topk": k (at most 20), "hidden_states": [layers 0..24], "mlp_act": [layers]}}}}
 The results of (b) are .npy files (with meta.json) in fwd_<k>/, right-padded to the longest text; hidden states and MLP activations are float16 and at most {tf_act_gb} GB per pass. The exact evaluation prompt format is in eval_prompt_format.txt: to score a generated answer with a teacher-forced pass, send prompt + answer as one raw text in "texts". A request that fails does not count against the budget, and an identical request on unchanged edits is not run. A generation pass takes from a few minutes to about half an hour.

HOW TO WORK.
1. Diagnose first. Look at how the model fails on practice problems: a wrong final answer, finishing without an answer, or being cut off at the token limit (often while repeating itself in its thinking). The generation summary counts these; read some failing texts. Go after the largest category you can explain.
2. Localize causally. That a component is more active on correct answers is not enough (correct answers come from easier and shorter problems). Before relying on a component, show that changing it changes the behaviour you predicted, and try several strengths to see a dose-response.
3. Measure with enough power. Accuracy over n generations has a standard error of about 50/sqrt(n) percentage points: about 4.4 points at 128 generations, 3.1 at 256. Compare original and edited weights on the SAME problems: a generation pass with "model": "both" does this with shared random numbers and reports the paired difference with its standard error, which is usually much smaller than that of two separate measurements; do not compare with an accuracy measured on other problems. Keep an edit only if its gain is clearly larger than its standard error on problems that were not used to choose it. A lower share of cut-off answers does not by itself mean more correct answers: judge edits by accuracy. Dense measurements help too and are cheap: one teacher-forced pass can read the reference solutions of hundreds of practice problems (or the model's own good versus failing continuations) and gives a signal on every token. Edits that change a tensor by a fraction of a percent usually have effects too small to see; test strengths large enough to matter and back off if they hurt.
4. Plan your forward passes (see the budget above): for example one diagnostic generation with activations, teacher-forced passes for dense paired measurements, and a final paired generation ("model": "both") on problems you have not used to confirm.
5. Work efficiently: state the hypothesis of each code block in its first comment line; work through the model in parts; load things once into kernel variables; print compact summaries, never whole arrays (and never raw lists of vocabulary tokens: the vocabulary contains offensive and adult strings and printing them can get the session refused by the provider's content filter; print digits, math symbols, common math words or token ids instead; likewise print short excerpts of generated texts, at most about 300 characters each, not whole transcripts); keep notes.md up to date (findings with numbers, edits tried and their measured effect, the current best edit; it is shown to you again if the conversation is restarted); keep copies of good edits (for example copy edited/ to best_1/). Before every forward pass, write to notes.md the exact current edit (tensors, what was changed, strength) and the problem set, so that you can rebuild them if the conversation is restarted.

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss; (b) running the model other than through the forward-pass tool: do not implement or simulate the network's computation on any input in the kernel; (c) reading files outside your work folder other than through the helpers, and internet access. Allowed: any analysis of the weights and of forward-pass results, any use of the practice problems, and any direct edit of the weights that you compute and that is in scope.

HOW THE SESSION RUNS. It continues until all {budget} forward passes are used (replying without a code block does not end it earlier); after that, make your final choice (the tensors in edited/ when you finish are submitted) and reply without a code block, saying what you changed, the evidence for it and why you expect it to help. At the end, submit the best edit you found, the candidate with the best measured paired gain, and report its measured gain with its standard error honestly."""


def _ram_gb() -> int:
    import psutil
    return round(psutil.virtual_memory().total / 2 ** 30)


def item_seed(pid: str, j: int) -> int:
    """The sampling seed of sample j of an item: fixed by the item, shared by every model and pass."""
    return int.from_bytes(hashlib.sha256(f"{pid}#{j}".encode()).digest()[:7], "little")


class QwenMathTask(Task):
    name = "qwen_math"

    def __init__(self, cfg, run_dir, work_dir):
        super().__init__(cfg, run_dir, work_dir)
        self.p = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
        self.model_file = os.path.join(MODEL_DIR, "model.safetensors-00001-of-00001.safetensors")
        self.problems = {}
        for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl"):
            for line in open(os.path.join(PRACTICE, f), encoding="utf-8"):
                r = json.loads(line)
                self.problems[r["id"]] = r
        self.engine = None

    # ------------------------------------------------------------------------------------------------ editor's view
    def task_prompt(self):
        p, s, c = self.p, self.p["sampling"], self.cfg
        return TASK_PROMPT.format(
            thinking_text=("enabled (enable_thinking=True: the model reasons between <think> and </think>, then answers)"
                           if p["thinking"] else "disabled (enable_thinking=False: the model answers directly)"),
            T=s["temperature"], top_p=s["top_p"], top_k=s["top_k"], pp=s["presence_penalty"], max_tokens=p["max_tokens"],
            for_what=" for reasoning and answer together" if p["thinking"] else "",
            timeout=c.get("kernel_timeout", 900), max_out=c.get("kernel_max_out", 10000), max_runs=c.get("max_runs", 300),
            budget=c.get("budget", 10), gen_max=p["gen_max"], gen_half=p["gen_max"] // 2, act_max_tokens=p["act_max_tokens"],
            tf_max_texts=p["tf_max_texts"], tf_max_tokens=p["tf_max_tokens"], tf_total_tokens=p["tf_total_tokens"],
            tf_act_gb=2, ram_gb=_ram_gb())

    def preflight(self):
        tokenizer = os.path.join(MODEL_DIR, "tokenizer.json")
        for p in (self.model_file, tokenizer):
            if not os.path.exists(p):
                raise TaskNotReady(f"the model is not downloaded ({p} is missing). Run `wsp setup --task {self.name}`.")
        import torch
        if not torch.cuda.is_available():
            raise TaskNotReady("it generates with the model on an NVIDIA GPU, and PyTorch here can't see one. `wsp doctor` "
                               "shows how to fix that. The edge_backdoor task runs without a GPU.")

    def kernel_ro_paths(self):
        return [self.model_file, os.path.join(MODEL_DIR, "tokenizer.json")]

    def kernel_setup(self):
        return (PRELUDE.replace("__MODEL_FILE__", repr(self.model_file)).replace("__FROZEN__", repr(FROZEN))
                .replace("__TOKENIZER__", repr(os.path.join(MODEL_DIR, "tokenizer.json"))))

    def prepare_work(self):
        super().prepare_work()
        os.makedirs(os.path.join(self.work_dir, "practice"), exist_ok=True)
        for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl", "README.txt"):
            shutil.copyfile(os.path.join(PRACTICE, f), os.path.join(self.work_dir, "practice", f))
        eng = self._engine_tok()
        example = eng.apply_chat_template([{"role": "user", "content": build_prompt("<problem>")}], tokenize=False,
                                          add_generation_prompt=True, enable_thinking=self.p["thinking"])
        open(os.path.join(self.work_dir, "eval_prompt_format.txt"), "w", encoding="utf-8").write(example)

    def _engine_tok(self):
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(MODEL_DIR)

    # ------------------------------------------------------------------------------------------------ weights
    def current_edits(self) -> dict:
        return {name: a.astype(np.float32) for name, a in self.edit_arrays().items()
                if name.startswith(PREFIX) and name not in FROZEN}

    # ------------------------------------------------------------------------------------------------ answer cache
    @staticmethod
    def _weights_key(edits: dict) -> str:
        """Content hash of the weights an answer was generated with ("original" = no edit)."""
        if not edits:
            return "original"
        h = hashlib.sha256()
        for n in sorted(edits):
            h.update(n.encode())
            h.update(np.ascontiguousarray(edits[n], np.float32).tobytes())
        return h.hexdigest()[:20]

    def _gen_key(self, user: str, seed: int, max_tokens: int, wkey: str) -> str:
        spec = [user, int(seed), int(max_tokens), self.p["sampling"], bool(self.p["thinking"]), wkey]
        return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest()[:32]

    def _gen_cache(self) -> dict:
        if getattr(self, "_cache", None) is None:
            self._cache = {}
            p = os.path.join(self.run_dir, "gen_cache.jsonl")
            if os.path.exists(p):
                for line in open(p, encoding="utf-8"):
                    try:
                        r = json.loads(line)
                    except ValueError:              # a line cut by a crash
                        continue
                    self._cache[r["key"]] = {"ids": r["ids"], "finish": r["finish"], "n_tokens": r["n_tokens"]}
        return self._cache

    def _gen_cache_add(self, key: str, g: dict):
        self._gen_cache()[key] = g
        with open(os.path.join(self.run_dir, "gen_cache.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"key": key, **g}) + "\n")

    def eng(self):
        if self.engine is None:
            from .engine import Engine
            self.engine = Engine(MODEL_DIR)
        return self.engine

    # ------------------------------------------------------------------------------------------------ forward tool
    def run_pass(self, req, k):
        t = req.get("type")
        if t not in ("generate", "teacher_forced"):
            raise ForwardError('"type" must be "generate" or "teacher_forced"')
        return self._generate(req, k) if t == "generate" else self._teacher_forced(req, k)

    def _items(self, req):
        items = []
        for pid in req.get("problems") or []:
            if pid not in self.problems:
                raise ForwardError(f"unknown practice id {pid!r}")
            items.append({"id": pid, "user": build_prompt(self.problems[pid]["problem"]),
                          "answer": self.problems[pid]["answer"], "source": self.problems[pid]["source"]})
        for i, pr in enumerate(req.get("prompts") or []):
            if not isinstance(pr, str) or not pr.strip():
                raise ForwardError('"prompts" must be non-empty strings')
            items.append({"id": f"prompt_{i}", "user": pr, "answer": None, "source": "prompt"})
        if not items:
            raise ForwardError('give "problems" and/or "prompts"')
        return items

    def _generate(self, req, k):
        model = req.get("model", "edited")
        if model not in ("original", "edited", "both"):
            raise ForwardError('"model" must be "original", "edited" or "both"')
        items = self._items(req)
        n = int(req.get("n", 1))
        cap = self.p["gen_max"] // 2 if model == "both" else self.p["gen_max"]
        if n < 1 or len(items) * n > cap:
            raise ForwardError(f"items x n = {len(items) * n}; at most {cap} with model {model!r}")
        mx = int(req.get("max_tokens", self.p["max_tokens"]))
        if not 1 <= mx <= self.p["max_tokens"]:
            raise ForwardError(f'"max_tokens" must be 1..{self.p["max_tokens"]}')
        acts = req.get("activations")
        layers, mlp = [], False
        if acts:
            layers = [int(l) for l in acts.get("layers", [])]
            mlp = bool(acts.get("mlp", False))
            if any(not 0 <= l <= 24 for l in layers):
                raise ForwardError("activation layers must be in 0..24")
        E = self.eng()
        prompts, seeds, rows = [], [], []
        for it in items:
            ids = E.chat_ids(it["user"], self.p["thinking"])
            for j in range(n):
                prompts.append(ids)
                seeds.append(item_seed(it["id"], j))
                rows.append((it, j))
        models = ["original", "edited"] if model == "both" else [model]
        summary = {"type": "generate", "model": model, "generations": len(prompts) * len(models)}
        results = {}
        for m in models:
            edits = self.current_edits() if m == "edited" else {}
            E.set_weights(edits or None)
            # per-answer cache (run_dir/gen_cache.jsonl): an answer is fixed by (prompt, seed, token cap, sampling,
            # weights), since sampling uses per-answer common random numbers. It keeps finished answers when a pass is
            # interrupted (the pass is not counted and is usually requested again) and reuses identical answers
            # across passes, e.g. the original model's side of paired passes. Passes are counted exactly as before.
            wkey = self._weights_key(edits)
            keys = [self._gen_key(it["user"], s, mx, wkey) for (it, _), s in zip(rows, seeds)]
            cache = self._gen_cache()
            gens = [cache.get(key) for key in keys]
            todo = [i for i, g in enumerate(gens) if g is None]
            if todo:
                def keep(t, g, todo=todo, keys=keys):
                    self._gen_cache_add(keys[todo[t]], g)
                def progress(done, total, m=m):
                    if self.on_progress:
                        self.on_progress(f"pass {k}: {done}/{total} answers ({m} model)")
                new = E.generate([prompts[i] for i in todo], [seeds[i] for i in todo], self.p["sampling"], mx,
                                 self.p["batch_size"], progress=progress, on_done=keep)
                for i, g in zip(todo, new):
                    gens[i] = g
            print(f"pass {k} ({m}): {len(prompts) - len(todo)} of {len(prompts)} answers reused from the cache",
                  flush=True)
            recs = []
            for (it, j), g in zip(rows, gens):
                text = E.tok.decode(g["ids"], skip_special_tokens=False)
                rec = {"id": it["id"], "sample": j, "source": it["source"], "n_tokens": g["n_tokens"],
                       "finish": g["finish"], "text": text}
                if it["answer"] is not None:
                    rec.update(grade(text, it["answer"], g["finish"]))
                    rec["answer"] = it["answer"]
                if g["finish"] == "length":
                    rec.update(loop_signals(text))
                recs.append(rec)
            results[m] = recs
            sfx = f"_{m}" if model == "both" else ""
            with open(os.path.join(self._out, f"gen_{k}{sfx}.jsonl"), "w", encoding="utf-8") as f:
                for r in recs:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
            summary[m if model == "both" else "result"] = self._gen_summary(recs)
            if layers or mlp:
                self._gen_acts(E, prompts, gens, recs, layers, mlp,
                               os.path.join(self._out, f"gen_{k}_acts" + (f"_{m}" if model == "both" else "")))
        if model == "both":
            summary["paired"] = self._paired(results["original"], results["edited"])
        E.set_weights(None)
        where = f"gen_{k}_original.jsonl and gen_{k}_edited.jsonl" if model == "both" else f"gen_{k}.jsonl"
        return summary, where

    @staticmethod
    def _gen_summary(recs):
        graded = [r for r in recs if "correct" in r]
        out = {"generations": len(recs), "graded": len(graded),
               "mean_tokens": round(float(np.mean([r["n_tokens"] for r in recs])), 1),
               "cut_share": round(float(np.mean([r["finish"] == "length" for r in recs])), 4)}
        if graded:
            out["accuracy"] = round(float(np.mean([r["correct"] for r in graded])), 4)
            src = sorted({r["source"] for r in graded})
            out["accuracy_by_source"] = {s: round(float(np.mean([r["correct"] for r in graded if r["source"] == s])), 4)
                                         for s in src}
            out["failures"] = {f: sum(r.get("failure") == f for r in graded)
                               for f in ("wrong_answer", "no_answer_finished", "cut_at_max_tokens")}
        cut = [r for r in recs if r["finish"] == "length"]
        if cut:
            out["cut_ending_in_loop"] = round(float(np.mean([r["verbatim_loop_end"] or r["repetitive_end"] for r in cut])), 4)
        return out

    @staticmethod
    def _paired(orig, edit):
        d = np.array([int(e["correct"]) - int(o["correct"]) for o, e in zip(orig, edit) if "correct" in o])
        if not len(d):
            return {"items": 0}
        return {"items": int(len(d)), "original_accuracy": round(float(np.mean([o["correct"] for o in orig if "correct" in o])), 4),
                "edited_accuracy": round(float(np.mean([e["correct"] for e in edit if "correct" in e])), 4),
                "difference": round(float(d.mean()), 4), "se": round(float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0, 4),
                "wrong_to_right": int((d == 1).sum()), "right_to_wrong": int((d == -1).sum()),
                "identical_answers": int(sum(o["text"] == e["text"] for o, e in zip(orig, edit)))}

    def _gen_acts(self, E, prompts, gens, recs, layers, mlp, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        mx = self.p["act_max_tokens"]
        seqs = [p + g["ids"][:mx] for p, g in zip(prompts, gens)]
        mlp_layers = [l - 1 for l in layers if l >= 1] if mlp else []
        res = {"hidden_last_prompt": [], "hidden_mean_answer": [], "mlp_last_prompt": [], "mlp_mean_answer": []}
        step = 4                                               # small chunks: system RAM is tight on this machine
        for i in range(0, len(seqs), step):
            r = E.read(seqs[i:i + step], hidden_layers=layers, mlp_layers=mlp_layers, topk=0, logprobs=False)
            for j in range(len(seqs[i:i + step])):
                pl, n = len(prompts[i + j]), len(seqs[i + j])
                # Small per-answer summaries computed straight from the float16 arrays. (Appending a slice such as
                # h[:, pl - 1] of a full (L, T, D) float32 copy kept that whole copy alive: ~0.3 GB per answer, which
                # exhausted system RAM on a 128-answer pass.)
                for key, names in (("hidden", [f"hidden_{l}" for l in layers]),
                                   ("mlp", [f"mlp_act_{l}" for l in mlp_layers])):
                    if not names:
                        continue
                    res[f"{key}_last_prompt"].append(np.stack([r[nm][j, pl - 1].astype(np.float32) for nm in names]))
                    res[f"{key}_mean_answer"].append(np.stack(
                        [r[nm][j, pl:n].mean(0, dtype=np.float32) if n > pl
                         else np.zeros(r[nm].shape[-1], np.float32) for nm in names]))
            del r
        files = {}
        for key, v in res.items():
            if v:
                arr = np.stack(v).astype(np.float32)
                np.save(os.path.join(out_dir, key + ".npy"), arr)
                files[key + ".npy"] = list(arr.shape)
        json.dump({"files": files, "layers": layers, "mlp_layers_note": "mlp arrays: down_proj inputs of layers "
                   + str([l + 1 for l in mlp_layers]) + " (the MLP inside layer l writes into residual l)",
                   "samples": [{"id": r["id"], "sample": r["sample"], "correct": r.get("correct"), "finish": r["finish"],
                                "n_tokens": r["n_tokens"]} for r in recs]},
                  open(os.path.join(out_dir, "index.json"), "w"), indent=1)

    def _teacher_forced(self, req, k):
        model = req.get("model", "edited")
        if model not in ("original", "edited"):
            raise ForwardError('"model" must be "original" or "edited" for teacher-forced passes')
        E = self.eng()
        if req.get("texts") is not None:
            seqs = [E.tok(t, add_special_tokens=False)["input_ids"] for t in req["texts"]]
        elif req.get("chats") is not None:
            seqs = []
            for c in req["chats"]:
                txt = E.tok.apply_chat_template(c, tokenize=False, enable_thinking=bool(req.get("enable_thinking")))
                seqs.append(E.tok(txt, add_special_tokens=False)["input_ids"])
        else:
            raise ForwardError('give "texts" or "chats"')
        p = self.p
        if not seqs or len(seqs) > p["tf_max_texts"]:
            raise ForwardError(f"1..{p['tf_max_texts']} texts per pass")
        lens = [len(s) for s in seqs]
        if max(lens) > p["tf_max_tokens"] or sum(lens) > p["tf_total_tokens"]:
            raise ForwardError(f"texts must have at most {p['tf_max_tokens']:,} tokens each and "
                               f"{p['tf_total_tokens']:,} in total (got max {max(lens):,}, total {sum(lens):,})")
        ret = req.get("return") or {"token_logprobs": True}
        hl = [int(l) for l in ret.get("hidden_states", [])]
        ml = [int(l) for l in ret.get("mlp_act", [])]
        topk = int(ret.get("topk", 0))
        if any(not 0 <= l <= 24 for l in hl) or any(not 0 <= l <= 23 for l in ml) or not 0 <= topk <= 20:
            raise ForwardError("hidden_states layers 0..24, mlp_act layers 0..23, topk 0..20")
        T = max(lens)
        vol = len(seqs) * T * (1024 * len(hl) + 3584 * len(ml)) * 2
        if vol > 2 * 1024 ** 3:
            raise ForwardError(f"the requested activations take {vol / 1e9:.1f} GB (padded); at most 2 GB per pass")
        if ret.get("attn_layers"):
            raise ForwardError("attention probabilities are not available in this harness; use hidden states or "
                               "MLP activations")
        E.set_weights(self.current_edits() if model == "edited" else None)
        out = E.read(seqs, hidden_layers=hl, mlp_layers=ml, topk=topk, logprobs=bool(ret.get("token_logprobs", True)))
        E.set_weights(None)
        d = os.path.join(self._out, f"fwd_{k}")
        os.makedirs(d, exist_ok=True)
        ids = np.zeros((len(seqs), T), np.int32)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = s
        out["input_ids"] = ids
        files = {}
        for key, v in out.items():
            np.save(os.path.join(d, key + ".npy"), v)
            files[key] = list(v.shape)
        meta = {"files": files, "lengths": lens, "model": model, "edited_tensors": len(self.current_edits()) if model == "edited" else 0}
        json.dump(meta, open(os.path.join(d, "meta.json"), "w"), indent=1)
        mean_lp = [float(out["token_logprobs"][i, :l - 1].mean()) if l > 1 and "token_logprobs" in out else None
                   for i, l in enumerate(lens)]
        summary = {"type": "teacher_forced", "model": model, "texts": len(seqs), "lengths": lens[:64],
                   "files": files, "mean_token_logprob": [round(x, 4) if x is not None else None for x in mean_lp[:64]]}
        return summary, f"fwd_{k}/"

    # ------------------------------------------------------------------------------------------------ submission
    def submit(self, out_dir):
        from ...weights import ModelWeights, export_edit
        ed = self.current_edits()
        if not ed:
            os.makedirs(out_dir, exist_ok=True)
            json.dump({"tensors": [], "changed_tensors": 0, "changed_entries_total": 0, "note": "no edit"},
                      open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
            return {"edited_tensors": [], "changed_entries_total": 0}
        man = export_edit(ed, ModelWeights(MODEL_DIR), out_dir, note="WSP agent submission (bfloat16 diff against "
                                                                      "Qwen/Qwen3.5-0.8B @ 2fc0636)")
        return {"edited_tensors": [t["name"] for t in man["tensors"]], "changed_entries_total": man["changed_entries_total"]}

    def close(self):
        super().close()
        self.engine = None
