"""Forward passes for any PyTorch module: the user names a function that builds the network (and optionally a checkpoint and
datasets), and the agent runs it on arrays, with weight edits swapped in, activations captured at any module, and optional
interventions.

The builder is the user's own code and runs outside the sandbox, with their permissions, like any script they run themselves.
"""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys

import numpy as np

from ..base import ForwardError
from .hooks import Hooks, SiteError, check_interventions, file_name, find_module, parse_site
from .lm import NAME
from .store import load_checkpoint


def build_model(spec: dict):
    """Call the target's builder and load its checkpoint. -> (model in eval mode, notes about keys that didn't match)."""
    import torch
    head, _, fn = spec["builder"].rpartition(":")
    if head.endswith(".py"):
        folder = os.path.dirname(head)
        if folder not in sys.path:
            sys.path.insert(0, folder)              # so the file can import its neighbours
        name = "wsp_target_" + os.path.splitext(os.path.basename(head))[0]
        mod_spec = importlib.util.spec_from_file_location(name, head)
        mod = importlib.util.module_from_spec(mod_spec)
        sys.modules[name] = mod
        mod_spec.loader.exec_module(mod)
    else:
        mod = importlib.import_module(head)
    obj = mod
    for part in fn.split("."):
        obj = getattr(obj, part)
    model = obj(**(spec.get("kwargs") or {}))
    if not isinstance(model, torch.nn.Module):
        raise TypeError(f"{spec['builder']} returned {type(model).__name__}, not a torch.nn.Module")
    notes = ""
    if spec.get("weights"):
        res = model.load_state_dict(load_checkpoint(spec["weights"]), strict=False)
        if res.missing_keys:
            notes += f" {len(res.missing_keys)} tensors of the network are not in the checkpoint (first: {res.missing_keys[:3]})."
        if res.unexpected_keys:
            notes += f" {len(res.unexpected_keys)} tensors of the checkpoint are not in the network (first: {res.unexpected_keys[:3]})."
    return model.eval(), notes


class TorchTool:
    TEXT = """ A pass runs the network on up to {max_items:,} inputs x models. Reply with exactly one block that starts with ```forward and contains JSON:
 {{"type": "run", "model": "original", "edited", "both", or a list such as ["original", "edited", "cand_a"], "inputs": "<file>.npy" or "<file>.npz" (arrays you saved in your work folder: one array is the network's input with the batch first; an .npz's arrays are passed to the network by name){dataset_note}, "capture": [sites] (optional), "interventions": [...] (optional), "outputs": true (default; false saves only the captured activations)}}
 "original" runs the unedited weights and "edited" your current edits. Any other name is a folder of .npy edits in your work folder (for example a copy of edited/ you saved as cand_a/), so one pass can compare several candidate edits or strengths on the same inputs. You get the output shapes{metric_note}, and with several models how many outputs changed against the first. The network's outputs and the captured activations (float16) are saved as .npy in fwd_<k>/ (with several models: one file per model), with meta.json; the leading dimension is the input.
 SITES AND INTERVENTIONS. A site is a module path from model/modules.txt, optionally followed by ":input" for what enters the module instead of what leaves it, for example "{example_site}". An intervention changes an activation while the pass runs, without touching the weights, which is a cheap way to test a hypothesis before you write the edit: {{"site": "...", "op": "zero" | "scale" | "add" | "project_out" | "replace", "value": number, "vector": "file.npy" (a direction or values you saved in your work folder), "units": [indices] (optional), "dim": the axis of the units (default -1; use 1 for the channels of a convolution)}}.
 A request that fails does not count against the budget, and an identical request on unchanged edits is not run."""

    def __init__(self, task, model, spec: dict):
        self.task, self.model, self.spec = task, model, spec
        self.params = dict(model.state_dict())
        self.originals: dict = {}
        self.applied: dict[str, str] = {}
        self.device = None
        self.data: dict = {}

    def close(self):
        pass

    def tools_text(self, p: dict, example_site: str) -> str:
        ds = self.spec.get("datasets") or {}
        note = metric = ""
        if ds:
            names = ", ".join(f'"{n}" ({self._dataset(n)[0].shape[0]:,} items' + (", with targets" if d.get("targets") else "") + ")"
                              for n, d in ds.items())
            note = (f' or {{"dataset": name, "indices": [..] or "all"}} for the user\'s data: {names}. You don\'t see these files; '
                    "the targets are used only for the score")
            metric = ", and for a dataset with targets a score (accuracy by class for class labels, mean squared error otherwise)"
        return self.TEXT.format(max_items=p["max_items"], dataset_note=note, metric_note=metric, example_site=example_site)

    def _dataset(self, name: str):
        if name not in self.data:
            d = (self.spec.get("datasets") or {})[name]
            x = np.load(d["inputs"], allow_pickle=False)
            y = np.load(d["targets"], allow_pickle=False) if d.get("targets") else None
            self.data[name] = (x, y)
        return self.data[name]

    def _to_device(self):
        import torch
        if self.device is None:
            self.device = self.task.cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.params = dict(self.model.state_dict())

    def set_weights(self, edits: dict | None):
        import torch
        edits = edits or {}
        with torch.no_grad():
            for n in list(self.applied):
                if n not in edits:
                    self.params[n].copy_(self.originals[n].to(self.device))
                    del self.applied[n]
            for n, a in edits.items():
                if n not in self.params:
                    raise KeyError(f"{n} is not a tensor of the network")
                p = self.params[n]
                if tuple(a.shape) != tuple(p.shape):
                    raise ValueError(f"{n}: shape {tuple(a.shape)} != {tuple(p.shape)}")
                if n not in self.originals:
                    self.originals[n] = p.detach().to("cpu").clone()
                p.copy_(torch.as_tensor(np.asarray(a)).to(self.device, p.dtype))
                self.applied[n] = "x"

    def run(self, req: dict, k: int):
        import torch
        if req.get("type") != "run":
            raise ForwardError('"type" must be "run"')
        p, out_dir = self.task.p, os.path.join(self.task._out, f"fwd_{k}")
        m = req.get("model", "edited")
        models = ["original", "edited"] if m == "both" else m if isinstance(m, list) else [m]
        if not models or not all(isinstance(x, str) and NAME.match(x) for x in models) or len(set(models)) != len(models):
            raise ForwardError('"model" must be "original", "edited", "both", or a list of distinct names')
        src, targets = req.get("inputs"), None
        if isinstance(src, dict) and "dataset" in src:
            if src["dataset"] not in (self.spec.get("datasets") or {}):
                raise ForwardError(f'no dataset {src["dataset"]!r}; the user gave: {sorted(self.spec.get("datasets") or {})}')
            x, y = self._dataset(src["dataset"])
            idx = src.get("indices", "all")
            if idx == "all":
                idx = np.arange(len(x))
            else:
                try:
                    idx = np.asarray(idx, np.int64).reshape(-1)
                except (TypeError, ValueError):
                    raise ForwardError('"indices" must be a list of integers or "all"') from None
                if len(idx) == 0 or idx.min() < 0 or idx.max() >= len(x):
                    raise ForwardError(f'"indices" must be in 0..{len(x) - 1}')
            inputs, targets = {"": x[idx]}, (y[idx] if y is not None else None)
        elif isinstance(src, str):
            arr = self.task.read_array(src, allow_npz=True)
            inputs = arr if isinstance(arr, dict) else {"": arr}
        else:
            raise ForwardError('"inputs" must be a .npy or .npz file in your work folder' + (' or {"dataset": name}' if self.spec.get("datasets") else ""))
        n = {len(v) for v in inputs.values()}
        if len(n) != 1 or not next(iter(n)):
            raise ForwardError("the input arrays must have the same, non-zero number of rows")
        n = next(iter(n))
        if n * len(models) > p["max_items"]:
            raise ForwardError(f"{n:,} inputs x {len(models)} models: at most {p['max_items']:,} per pass")
        capture = req.get("capture") or []
        self._to_device()
        try:
            for s in capture:
                find_module(self.model, parse_site(s)[0])
            vectors = {iv["vector"]: torch.as_tensor(self.task.read_array(iv["vector"]))
                       for iv in req.get("interventions") or [] if isinstance(iv, dict) and isinstance(iv.get("vector"), str)}
            ivs = check_interventions(self.model, req.get("interventions"), vectors)
        except SiteError as e:
            raise ForwardError(str(e)) from None
        os.makedirs(out_dir, exist_ok=True)
        bs = int(req.get("batch_size") or p["torch_batch"])
        summary: dict = {"type": "run", "models": models, "inputs": n}
        first_out, files = None, {}
        for m in models:
            try:
                self.set_weights(self.task.edits_of(m))
            except (KeyError, ValueError) as e:
                raise ForwardError(f"{m}: {e}") from None
            outs: dict = {}
            acts: dict = {s: [] for s in capture}
            used = 0
            for a in range(0, n, bs):
                batch = {key: torch.as_tensor(v[a:a + bs]).to(self.device) for key, v in inputs.items()}
                batch = {key: (v.float() if v.is_floating_point() else v) for key, v in batch.items()}
                with torch.no_grad(), Hooks(self.model, capture, ivs) as hk:
                    y = self.model(batch[""]) if "" in batch else self.model(**batch)
                for key, t in self._outputs(y).items():
                    outs.setdefault(key, []).append(t.float().cpu().numpy())
                for s in capture:
                    if s not in hk.seen:
                        raise ForwardError(f"the module at {s} was not called in this pass")
                    arr = hk.seen[s].float().cpu().numpy().astype(np.float16)
                    used += arr.size
                    if used * len(models) > p["act_values"]:
                        raise ForwardError(f"the captured activations hold more than {p['act_values']:,} values: ask for fewer "
                                           "inputs or sites")
                    acts[s].append(arr)
                self.task.progress(f"pass {k}: {min(a + bs, n):,}/{n:,} inputs ({m})")
            sfx = f"_{m}" if len(models) > 1 else ""
            out = {key: np.concatenate(v) for key, v in outs.items()}
            info: dict = {"outputs": {key or "output": list(v.shape) for key, v in out.items()}}
            if req.get("outputs", True):
                for key, v in out.items():
                    fn = f"output{'_' + key if key else ''}{sfx}.npy"
                    np.save(os.path.join(out_dir, fn), v)
                    files[fn] = list(v.shape)
            for s, parts in acts.items():
                fn = f"act_{file_name(s)}{sfx}.npy"
                arr = np.concatenate(parts)
                np.save(os.path.join(out_dir, fn), arr)
                files[fn] = list(arr.shape)
            main = out.get("") if "" in out else (out.get("logits") if "logits" in out else next(iter(out.values()), None))
            if main is not None:
                info.update(self._score(main, targets))
                if first_out is None:
                    first_out = main
                elif main.shape == first_out.shape:
                    if main.ndim == 2 and main.shape[1] > 1:
                        info[f"predictions_changed_vs_{models[0]}"] = int((main.argmax(1) != first_out.argmax(1)).sum())
                    info[f"mean_abs_output_change_vs_{models[0]}"] = float(np.abs(main - first_out).mean())
            summary[m] = info
        json.dump({"request": req, "files": files, "rows": "one per input, in the order given"},
                  open(os.path.join(out_dir, "meta.json"), "w"), indent=1)
        return summary, f"fwd_{k}/"

    @staticmethod
    def _outputs(y) -> dict:
        import torch
        if isinstance(y, torch.Tensor):
            return {"": y}
        if isinstance(y, (tuple, list)):
            return {str(i): t for i, t in enumerate(y) if isinstance(t, torch.Tensor)}
        if hasattr(y, "items"):
            return {str(key): t for key, t in y.items() if isinstance(t, torch.Tensor)}
        raise ForwardError(f"the network returned {type(y).__name__}, which holds no tensors")

    @staticmethod
    def _score(out: np.ndarray, targets) -> dict:
        if targets is None:
            return {}
        if out.ndim == 2 and out.shape[1] > 1 and np.issubdtype(targets.dtype, np.integer) and targets.ndim == 1:
            pred = out.argmax(1)
            classes = np.unique(targets)
            s = {"accuracy": round(float((pred == targets).mean()), 4),
                 "predicted_counts": np.bincount(pred, minlength=out.shape[1]).tolist()}
            if len(classes) <= 50:
                s["accuracy_by_class"] = {int(c): round(float((pred[targets == c] == c).mean()), 3) for c in classes}
            return s
        if targets.shape == out.shape:
            return {"mean_squared_error": float(((out - targets.astype(np.float32)) ** 2).mean())}
        return {"score": f"not computed: outputs {out.shape} and targets {targets.shape} don't match a known task"}
