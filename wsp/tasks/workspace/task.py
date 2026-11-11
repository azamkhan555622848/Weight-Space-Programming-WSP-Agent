"""The general task: program any network the user points at, towards a goal they state in their own words.

Where the two benchmark tasks have a fixed prompt, a fixed network and a hidden test, this one builds its prompt from the
target (wsp/tasks/workspace/targets.py): a description of the network read from its files, the kernel helpers, and the
forward-pass tool that fits it (lm.py for Hugging Face language models, torchmod.py for any PyTorch module, none for bare
weights). There is no hidden test: the session is a conversation, and the user judges the result by the agent's evidence.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np

from ...sandbox import UnsafePath, open_in
from ..base import ForwardError, Task, TaskNotReady
from .lm import NAME
from .store import Store, as_store, find_weight_files, module_listing, save_state_dict, summarize

DEFAULTS = {"gen_max": 256, "max_tokens": 4096, "tf_max_texts": 512, "tf_max_tokens": 8192, "tf_total_tokens": 262144,
            "act_values": 1_000_000_000, "batch_size": 16, "max_items": 20_000, "torch_batch": 256}
TORCH_DTYPES = {"BF16": "bfloat16", "F16": "float16", "F32": "float32", "F64": "float64", "I64": "int64", "I32": "int32",
                "I16": "int16", "I8": "int8", "U8": "uint8", "BOOL": "bool"}

PRELUDE = r'''
import json, os
import numpy as np
import torch
from safetensors import safe_open
np.set_printoptions(precision=4, suppress=True, linewidth=160)
__TOKENIZER__

def _wsp_helpers():
    # the helpers keep their own state (absolute paths, the open weight files), so user code that reassigns globals
    # or changes the working directory cannot break them
    where = {}
    for path in __FILES__:
        f = safe_open(path, framework="pt")
        for n in f.keys():
            where.setdefault(n, f)
    names = sorted(where)
    shapes = {n: tuple(where[n].get_slice(n).get_shape()) for n in names}
    frozen = set(__FROZEN__)
    edir = os.path.join(os.getcwd(), "edited")
    plan = os.path.join(os.getcwd(), ".wsp_plan")

    def load_original(name):
        assert name in where, f"unknown tensor {name!r}"
        # a copy: the file is memory-mapped, and without one an in-place change to a loaded tensor would show up in every
        # later read of the "original"
        return where[name].get_tensor(name).to(torch.float32, copy=True)

    def load(name):
        p = os.path.join(edir, name + ".npy")
        return torch.from_numpy(np.load(p)).float() if os.path.exists(p) else load_original(name)

    def save_edit(name, tensor):
        if os.path.exists(plan):
            raise RuntimeError("plan mode is on: edits are not saved until the user switches to build mode")
        if name in frozen:
            raise ValueError(f"{name} is out of scope: the user excluded it from editing")
        assert name in where, f"unknown tensor {name!r}"
        a = np.asarray(tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else tensor, np.float32)
        o = load_original(name).numpy()
        assert a.shape == o.shape, (name, a.shape, o.shape)
        assert np.isfinite(a).all(), f"{name}: non-finite values"
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

    def project_out(W, d, strength=1.0, side="out"):
        """W with direction d removed: side="out" from what W writes (its first axis), side="in" from what it reads (its
        second axis). strength=1 removes it fully; W may have more than two axes (a convolution)."""
        W = torch.as_tensor(W).float()
        d = torch.as_tensor(d).float().flatten()
        d = d / d.norm().clamp_min(1e-12)
        axis = 0 if side == "out" else 1
        assert side in ("out", "in") and W.shape[axis] == d.numel(), (side, tuple(W.shape), d.numel())
        M = W.movedim(axis, -1)
        return (M - strength * (M @ d).unsqueeze(-1) * d).movedim(-1, axis).contiguous()

    def rank_one_update(W, u, v, scale=1.0):
        """W + scale * outer(u, v), for a matrix W of shape (len(u), len(v))."""
        W = torch.as_tensor(W).float()
        u, v = torch.as_tensor(u).float().flatten(), torch.as_tensor(v).float().flatten()
        assert W.shape == (u.numel(), v.numel()), (tuple(W.shape), u.numel(), v.numel())
        return W + scale * torch.outer(u, v)

    return names, shapes, load_original, load, save_edit, drop_edit, edited, project_out, rank_one_update

NAMES, SHAPES, load_original, load, save_edit, drop_edit, edited, project_out, rank_one_update = _wsp_helpers()
del _wsp_helpers
'''

PROMPT = """TASK. You are working with a user on a neural network whose weights you can read and edit directly. This is weight-space programming: you find out how the network implements the behaviour in question (which tensors, units, directions or circuits), and you change those weights in a way you can justify with evidence. There is no training here: no gradients, no optimizers, no fitting.

THE USER'S REQUEST.
{goal}

THE NETWORK. {network}
The complete lists are files in your work folder: model/tensors.txt (every tensor: name, shape, type) and model/modules.txt (every module path with its class){config_note}.{project_notes}

WHAT YOU HAVE.
1. A persistent Python kernel (no internet, no GPU), run in your work folder (the current directory): variables survive between your code blocks, like a notebook. Available: numpy, scipy, PyTorch (CPU), pandas, sympy, networkx, safetensors{tok_lib}. Defined in advance: NAMES (all tensor names), SHAPES (name -> shape), load(name) (a float32 torch tensor: your current edit if there is one, else the original), load_original(name), save_edit(name, tensor) (stores it in edited/), drop_edit(name), edited(), project_out(W, d, strength=1.0, side="out") (W with direction d removed from what it writes; side="in": from what it reads) and rank_one_update(W, u, v, scale=1.0){tok_note}. Gradients are disabled. To run code, reply with exactly one block that starts with ```python and nothing else; if its last line is an expression, its value is shown. One execution may take up to {timeout} seconds; this machine has {ram_gb} GB of memory in total, so load tensors as you need them rather than all at once; output is cut at {max_out:,} characters. You may run code up to {max_runs} times. If the kernel is restarted (time or memory limit), its variables are lost and you are told so.
2. {forward}
3. A todo list the user watches. For work of more than a few steps, put a ```todo block before your code block whenever the list changes, one item per line: "- [ ] not started", "- [~] in progress", "- [x] done". It replaces the previous list, so always give the whole list; keep it to 3-8 items.

HOW TO WORK.
1. Start from the request. If something essential is missing and you can't find it out yourself (which behaviour, on what inputs, what must not change), ask the user before spending forward passes. Otherwise say how you will measure success, then begin.
2. Diagnose first. Look at how the network behaves on inputs you choose, and at its weights; form hypotheses about where the behaviour lives and test them.
3. Localize causally. That a component looks unusual or correlates with the behaviour is not enough. Before relying on it, show that changing it changes the behaviour you predicted (an intervention in a forward pass is the cheapest test), and try several strengths to see a dose-response.
4. Measure with enough power, on inputs you did not use to choose the edit. Compare the original and the edited network on the same inputs in one pass, and report differences with their uncertainty. Check what must not change (the network's ordinary behaviour) as well as what must.
5. Work efficiently: state the hypothesis of each code block in its first comment line; load things once into kernel variables; print compact summaries, never whole arrays; keep notes.md in your work folder up to date (findings with numbers, edits tried and their measured effect, the current best edit): it is shown to you again if the conversation is restarted. Keep copies of good edits (for example copy edited/ to best_1/). Plan your forward passes: test many variants in one pass.

RULES. Not allowed: (a) gradient descent or any other training: no backward passes, autograd or optimizers, and no iterative fitting of weights to a loss (closed-form computations from the weights and from measured activations, such as means, covariances, projections and least squares, are allowed); (b) running the network other than through the forward-pass tool: do not implement or simulate its computation on any input in the kernel; (c) reading files outside your work folder other than through the helpers, and internet access.

HOW THE SESSION RUNS. This is a conversation with the user. Each reply of yours holds at most one ```python or ```forward block; it is run and you get the result, and you carry on by yourself until the request is done or you need the user. To talk to the user, reply WITHOUT a code block: that reply is shown to them as your message, and the session waits for their answer. Use it to report (what you found, what you changed and by how much, the measured effect with its uncertainty, what you did not manage, and what you would do next) or to ask a question. Messages the user sends while you work reach you with your next result. The tensors in edited/ are the current state of your edit, which the user can export at any time. {budget_note}"""

NO_FORWARD = ("No forward passes: the user gave the weights but no code that runs the network, so everything has to come from "
              "reading the weights. Say so in your report, and say what a forward pass would let you check; the user can start a "
              "session with the code that builds the network (file.py:function) to make them available.")


class WorkspaceTask(Task):
    name = "workspace"

    def __init__(self, cfg, run_dir, work_dir, log=print):
        super().__init__(cfg, run_dir, work_dir)
        self.p = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
        self.spec = dict(cfg["target"])
        self.kind = self.spec["kind"]
        self.project_dir = cfg.get("project_dir") or os.getcwd()
        self.log = log
        self.store_dir = os.path.join(run_dir, "model_store")
        self.frozen = list(self.spec.get("frozen") or [])
        self.tool = None
        self.model_dir = None
        self.notes = ""
        self.modules: dict[str, str] | None = None
        if self.kind == "hf-lm":
            from .lm import LMTool
            self.model_dir = self._hf_dir()
            self.store = as_store(find_weight_files(self.model_dir), self.store_dir)
            self.tool = LMTool(self, self.model_dir)
        elif self.kind == "torch":
            from .torchmod import TorchTool, build_model
            model, self.notes = build_model(self.spec)
            path = os.path.join(self.store_dir, "weights.safetensors")
            if not os.path.exists(path):
                save_state_dict(model.state_dict(), path)
            self.store = Store([path])
            self.modules = module_listing(model)
            self.tool = TorchTool(self, model, self.spec)
        elif self.kind == "weights":
            path = self.spec["path"]
            files = find_weight_files(path) if os.path.isdir(path) else [path]
            if not files:
                raise FileNotFoundError(f"no weights file in {path}")
            self.store = as_store(files, self.store_dir)
        else:
            raise ValueError(f"unknown kind of target: {self.kind!r}")
        if not self.store.names():
            raise ValueError("the target holds no tensors")

    # ------------------------------------------------------------------------------------------------ the target
    def _hf_dir(self) -> str:
        path = self.spec["path"]
        if os.path.isdir(path):
            return path
        marker = os.path.join(self.run_dir, "target_dir.txt")
        if os.path.exists(marker):
            return open(marker, encoding="utf-8").read().strip()
        from huggingface_hub import snapshot_download
        self.log(f"downloading {path} from Hugging Face ...")
        d = snapshot_download(path)
        os.makedirs(self.run_dir, exist_ok=True)
        open(marker, "w", encoding="utf-8").write(d)
        return d

    def _config(self) -> dict:
        try:
            with open(os.path.join(self.model_dir, "config.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError):
            return {}

    def _hf_modules(self) -> dict[str, str]:
        """Module paths of the model without loading its weights; if that fails, the paths the tensor names imply."""
        try:
            from .lm import pick_class
            model = pick_class(self.model_dir, set(self.store.names()))[1]
            if model is not None:
                return module_listing(model)
        except Exception:  # noqa: BLE001 - a listing is a convenience; the tensor names always work
            pass
        return {n.rsplit(".", 1)[0]: "module" for n in self.store.names() if "." in n}

    def info(self) -> dict:
        return {"name": self.spec.get("name"), "kind": self.kind, "tensors": len(self.store.names()), "parameters": self.store.n_params()}

    def describe(self) -> str:
        n, params = len(self.store.names()), self.store.n_params()
        size = f"{params / 1e9:.2f} billion" if params >= 1e9 else f"{params / 1e6:.1f} million" if params >= 1e6 else f"{params:,}"
        name = self.spec.get("name") or "the network"
        if self.kind == "hf-lm":
            c = self._config()
            t = c.get("text_config") if isinstance(c.get("text_config"), dict) else c
            facts = [f"{k} {t[a]}" for k, names in (("layers", ("num_hidden_layers", "n_layer")), ("width", ("hidden_size", "n_embd", "d_model")),
                                                    ("attention heads", ("num_attention_heads", "n_head")),
                                                    ("MLP width", ("intermediate_size", "n_inner")), ("vocabulary", ("vocab_size",)))
                     for a in names[:1 + len(names)] if t.get(a) is not None][:5]
            head = (f"A language model, {name} (type {c.get('model_type', 'unknown')}; " + ", ".join(dict.fromkeys(facts)) + f"): {size} parameters in "
                    f"{n} tensors. It is run through Hugging Face transformers; module paths start at the model object, and for "
                    "most models the tensor names are the module paths plus .weight or .bias.")
        elif self.kind == "torch":
            head = (f"A PyTorch network, {name}, of class {type(self.tool.model).__name__}, built by the user's {os.path.basename(self.spec['builder'])}: "
                    f"{size} parameters in {n} tensors (the names are those of its state_dict).{self.notes}")
        else:
            head = f"The weights of a network, {name}: {size} parameters in {n} tensors."
        text = head + "\nTENSORS (names that differ only in a number share a line; xN = how many):\n" + summarize(
            {k: str(tuple(v)) for k, v in self.store.shapes().items()}, 60)
        if self.modules:
            text += "\nMODULES (sites for capturing or changing activations):\n" + summarize(self.modules, 40)
        return text

    def _example_site(self) -> str:
        mods = list(self.modules or {})
        for want in ("mlp", "fc", "linear", "conv"):
            hit = next((m for m in mods if want in m.lower().rsplit(".", 1)[-1]), None)
            if hit:
                return hit + ":input"
        return (mods[len(mods) // 2] if mods else "layers.0") + ":input"

    # ------------------------------------------------------------------------------------------------ the editor's view
    def task_prompt(self):
        c = self.cfg
        if self.modules is None:
            self.modules = self._hf_modules() if self.kind == "hf-lm" else {}
        tok = self._tokenizer()
        if self.kind == "hf-lm":
            has_chat = os.path.exists(os.path.join(self.model_dir, "chat_template.jinja")) or "chat_template" in self._tok_config()
            gen = self._generation_config()
            sampling = (f"temperature {gen.get('temperature', 1.0)}, top_p {gen.get('top_p', 1.0)}, top_k {gen.get('top_k', 0)}"
                        if gen.get("do_sample") else "greedy")
            forward = f"A forward-pass tool with a budget of {c.get('budget', 20)} forward passes." + self.tool.tools_text(
                self.p, has_chat, self._example_site(), sampling)
        elif self.kind == "torch":
            forward = f"A forward-pass tool with a budget of {c.get('budget', 20)} forward passes." + self.tool.tools_text(
                self.p, self._example_site())
        else:
            forward = NO_FORWARD
        notes = ""
        wsp_md = os.path.join(self.project_dir, "WSP.md")
        if os.path.isfile(wsp_md):
            notes = "\nNOTES FROM THE USER ABOUT THIS PROJECT (WSP.md):\n" + open(wsp_md, encoding="utf-8", errors="replace").read()[:4000]
        import psutil
        return PROMPT.format(
            goal=str(self.goal or "(no request yet)").strip(), network=self.describe(),
            config_note=", and model/config.json" if self.kind == "hf-lm" else "", project_notes=notes,
            tok_lib=", tokenizers" if tok else "",
            tok_note=", and tok (the model's tokenizer, a tokenizers.Tokenizer)" if tok else "",
            timeout=c.get("kernel_timeout", 900), ram_gb=round(psutil.virtual_memory().total / 2 ** 30),
            max_out=c.get("kernel_max_out", 10000), max_runs=c.get("max_runs", 300), forward=forward,
            budget_note=(f"Forward passes are limited to {c.get('budget', 20)} for the session unless the user raises that."
                         if self.tool else ""))

    def _tokenizer(self) -> str | None:
        p = os.path.join(self.model_dir, "tokenizer.json") if self.model_dir else None
        return os.path.realpath(p) if p and os.path.isfile(p) else None

    def _tok_config(self) -> dict:
        try:
            with open(os.path.join(self.model_dir, "tokenizer_config.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError):
            return {}

    def _generation_config(self) -> dict:
        try:
            with open(os.path.join(self.model_dir, "generation_config.json"), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError, TypeError):
            return {}

    def kernel_ro_paths(self):
        return self.store.files + ([self._tokenizer()] if self._tokenizer() else [])

    def kernel_setup(self):
        tok = self._tokenizer()
        return (PRELUDE.replace("__FILES__", repr(self.store.files)).replace("__FROZEN__", repr(self.frozen))
                .replace("__TOKENIZER__", f"from tokenizers import Tokenizer\ntok = Tokenizer.from_file({tok!r})" if tok else ""))

    def prepare_work(self):
        super().prepare_work()
        d = os.path.join(self.work_dir, "model")
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, "tensors.txt"), "w", encoding="utf-8") as f:
            for n in sorted(self.store.names()):
                f.write(f"{n}\t{tuple(self.store.shape(n))}\t{self.store.dtype(n)}\n")
        if self.modules is None:
            self.modules = self._hf_modules() if self.kind == "hf-lm" else {}
        with open(os.path.join(d, "modules.txt"), "w", encoding="utf-8") as f:
            for n, what in self.modules.items():
                f.write(f"{n}\t{what}\n")
        if self.kind == "hf-lm" and os.path.isfile(os.path.join(self.model_dir, "config.json")):
            shutil.copyfile(os.path.join(self.model_dir, "config.json"), os.path.join(d, "config.json"))

    def preflight(self):
        if self.kind == "hf-lm":
            try:
                import transformers  # noqa: F401
            except ImportError:
                raise TaskNotReady('running a Hugging Face model needs transformers: pip install -e ".[lm]"') from None

    def set_mode(self, mode: str):
        flag = os.path.join(self.work_dir, ".wsp_plan")
        os.makedirs(self.work_dir, exist_ok=True)
        if mode == "plan":
            open(flag, "w").close()
        elif os.path.exists(flag):
            os.remove(flag)

    # ------------------------------------------------------------------------------------------------ forward tool
    def progress(self, detail: str):
        if self.on_progress:
            self.on_progress(detail)

    def _candidates(self, req) -> list[str]:
        m = (req or {}).get("model")
        return [n for n in (m if isinstance(m, list) else []) if isinstance(n, str) and n not in ("original", "edited") and NAME.match(n)]

    def snapshot_edits(self, req=None):
        """The current edit, plus any other folders of edits the request compares, copied out of the editor's reach."""
        snap = super().snapshot_edits(req)
        for name in self._candidates(req):
            d = os.path.join(self.work_dir, name)
            if os.path.islink(d) or not os.path.isdir(d):
                raise UnsafePath(f"{name!r} is not a folder of edits in your work folder")
            dst = os.path.join(snap, "candidates", name)
            os.makedirs(dst)
            for f in sorted(os.listdir(d)):
                if f.endswith(".npy"):
                    with open_in(d, f) as src, open(os.path.join(dst, f), "wb") as out:
                        shutil.copyfileobj(src, out, 1 << 22)
        return snap

    def edits_of(self, model: str) -> dict:
        """{tensor name: array} of a model named in a request: "original", "edited" or a candidate folder."""
        if model == "original":
            return {}
        d = self._pass_edits if model == "edited" else os.path.join(self._pass_edits, "candidates", model)
        if not os.path.isdir(d):
            raise ForwardError(f"no folder of edits named {model!r} in your work folder")
        return {f[:-4]: np.load(os.path.join(d, f), allow_pickle=False) for f in sorted(os.listdir(d))
                if f.endswith(".npy") and f[:-4] not in self.frozen}

    def read_array(self, rel: str, allow_npz: bool = False):
        """An array the editor saved in its work folder, read without following links."""
        try:
            with open_in(self.work_dir, str(rel)) as f:
                z = np.load(f, allow_pickle=False)
                if isinstance(z, np.ndarray):
                    return z
                if not allow_npz:
                    raise ForwardError(f"{rel!r} must be a .npy file holding one array")
                return {k: z[k] for k in z.files}
        except FileNotFoundError:
            raise ForwardError(f"{rel!r} not found in your work folder") from None
        except UnsafePath as e:
            raise ForwardError(str(e)) from None
        except ValueError as e:
            raise ForwardError(f"cannot read {rel!r}: {e}") from None

    def run_pass(self, req, k):
        if self.tool is None:
            raise ForwardError("this target has weights only: there is no code to run the network, so no forward passes")
        return self.tool.run(req, k)

    # ------------------------------------------------------------------------------------------------ results
    def diff(self) -> list[dict]:
        """What the current edit changes, tensor by tensor: {name, shape, changed_entries, entries, relative_change}."""
        out = []
        for n, a in self.edit_arrays().items():
            if n not in self.store.where:
                continue
            o = self.store.f32(n)
            if a.shape != o.shape:
                continue
            a32 = np.asarray(a, np.float32)
            out.append({"name": n, "shape": list(a.shape), "entries": int(a32.size), "changed_entries": int((a32 != o).sum()),
                        "relative_change": float(np.linalg.norm(a32 - o) / max(float(np.linalg.norm(o)), 1e-30))})
        return out

    def export(self, out_dir: str, full: bool = False) -> dict:
        """Write the current edit: edit.safetensors (the edited tensors, whole, in their original types) and manifest.json.
        With `full`, also a complete copy of the model with the edit applied, ready to load."""
        import torch
        from safetensors.torch import load_file, save_file
        edits = {n: a for n, a in self.edit_arrays().items() if n in self.store.where and n not in self.frozen}
        os.makedirs(out_dir, exist_ok=True)
        tensors, info = {}, []
        for n, a in edits.items():
            o = self.store.f32(n)
            if a.shape != o.shape:
                continue
            a32 = np.asarray(a, np.float32)
            tensors[n] = torch.as_tensor(a32).to(getattr(torch, TORCH_DTYPES.get(self.store.dtype(n), "float32")))
            info.append({"name": n, "shape": list(a.shape), "changed_entries": int((a32 != o).sum()),
                         "relative_change": float(np.linalg.norm(a32 - o) / max(float(np.linalg.norm(o)), 1e-30))})
        if tensors:
            save_file(tensors, os.path.join(out_dir, "edit.safetensors"))
        manifest = {"target": {k: self.spec.get(k) for k in ("name", "kind", "path", "builder", "weights")},
                    "tensors": info, "changed_entries_total": sum(t["changed_entries"] for t in info),
                    "format": "edit.safetensors holds each edited tensor whole, in its original type; all other tensors are unchanged"}
        if full:
            d = os.path.join(out_dir, "model")
            os.makedirs(d, exist_ok=True)
            for path in self.store.files:
                sd = load_file(path)
                for n, t in tensors.items():
                    if n in sd:
                        sd[n] = t
                save_file(sd, os.path.join(d, os.path.basename(path)))
            if self.kind == "hf-lm":
                for f in os.listdir(self.model_dir):
                    src = os.path.join(self.model_dir, f)
                    if os.path.isfile(src) and not f.endswith((".safetensors", ".bin", ".pt", ".pth", ".h5", ".msgpack")):
                        shutil.copyfile(src, os.path.join(d, f))
            manifest["model"] = "model/ is the complete network with the edit applied"
        json.dump(manifest, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
        return {"edited_tensors": [t["name"] for t in info], "changed_entries_total": manifest["changed_entries_total"],
                "folder": out_dir}

    def submit(self, out_dir):
        return self.export(out_dir)

    def close(self):
        super().close()
        if self.tool:
            self.tool.close()
