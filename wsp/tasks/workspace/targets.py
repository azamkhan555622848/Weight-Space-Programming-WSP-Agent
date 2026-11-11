"""What the user wants programmed: turning what they typed (a folder, a file, a Hugging Face id) into a target the session can
load, and finding candidates in a project folder.

A target is a dictionary stored with the session:
  {"kind": "hf-lm",   "name", "path": a model folder, or a Hugging Face id to download}      any Hugging Face causal language model
  {"kind": "torch",   "name", "builder": "file.py:function" or "package.module:function", "kwargs": {...},
                      "weights": a checkpoint or None, "datasets": {name: {"inputs": file, "targets": file or None}}}
                                                                                          any PyTorch module
  {"kind": "weights", "name", "path": a weights file or folder}                              weights only: nothing can be run
  {"kind": "benchmark", "name", "task": "qwen_math" | "edge_backdoor"}                       the two fixed benchmark tasks

A project can name its targets in a file called wsp.json: {"targets": {"my-net": {"builder": "model.py:build", "weights": ...}}}.
Paths in it are relative to the project folder.
"""
from __future__ import annotations

import json
import os
import re

from .store import WEIGHT_SUFFIXES

BENCHMARKS = {
    "qwen_math": "Benchmark: make Qwen3.5-0.8B better at math (the paper's experiment; NVIDIA GPU)",
    "edge_backdoor": "Benchmark: remove a hidden backdoor from a ResNet-8 (no GPU needed)",
}
HUB_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")
PROJECT_FILE = "wsp.json"


class TargetError(ValueError):
    pass


def _abs(path: str, project_dir: str) -> str:
    return os.path.abspath(os.path.join(project_dir, os.path.expanduser(path)))


def _is_hf_folder(d: str) -> bool:
    return os.path.isfile(os.path.join(d, "config.json")) and any(
        n.endswith(".safetensors") or n == "pytorch_model.bin" for n in os.listdir(d))


def _is_builder(tok: str) -> bool:
    head, _, fn = tok.rpartition(":")
    return bool(fn) and bool(head) and len(head) > 1 and (head.endswith(".py") or re.fullmatch(r"[\w.]+", head) is not None) \
        and fn.isidentifier()


def resolve(text: str, project_dir: str) -> dict:
    """The target meant by what the user typed: a name from wsp.json, a benchmark, a folder, a file (optionally followed by
    `file.py:function` that builds the network), or a Hugging Face id."""
    text = text.strip().strip('"')
    if not text:
        raise TargetError("Name a model folder, a weights file, or a Hugging Face id.")
    named = project_targets(project_dir)
    if text in named:
        return named[text]
    if text in BENCHMARKS:
        return {"kind": "benchmark", "name": text, "task": text}
    parts = text.split()
    builder = next((p for p in parts[1:] if _is_builder(p)), None)
    first = parts[0] if builder else text
    if _is_builder(first) and not os.path.exists(_abs(first, project_dir)):
        return normalise({"kind": "torch", "builder": first}, project_dir)       # the builder loads its own weights
    path = _abs(first, project_dir)
    if os.path.isdir(path):
        if _is_hf_folder(path):
            return {"kind": "hf-lm", "name": os.path.basename(path.rstrip("\\/")), "path": path}
        if any(n.endswith(WEIGHT_SUFFIXES) for n in os.listdir(path)):
            return {"kind": "weights", "name": os.path.basename(path.rstrip("\\/")), "path": path}
        raise TargetError(f"{first} has no model in it (no config.json with weights, and no weights file).")
    if os.path.isfile(path):
        if not path.lower().endswith(WEIGHT_SUFFIXES):
            raise TargetError(f"{first} isn't a weights file ({', '.join(WEIGHT_SUFFIXES)}).")
        if builder:
            return normalise({"kind": "torch", "builder": builder, "weights": path}, project_dir)
        return {"kind": "weights", "name": os.path.basename(path), "path": path}
    if HUB_ID.match(first) and not builder:
        return {"kind": "hf-lm", "name": first.split("/")[-1], "path": first, "hub": True}
    raise TargetError(f"Can't find {first!r}. Give a model folder, a weights file (optionally followed by model.py:function), "
                      "or a Hugging Face id such as openai-community/gpt2.")


def normalise(spec: dict, project_dir: str) -> dict:
    """Fill in a target's kind and name and make its paths absolute."""
    spec = dict(spec)
    if "task" in spec and spec.get("kind") in (None, "benchmark"):
        return {"kind": "benchmark", "name": spec["task"], "task": spec["task"]}
    if spec.get("builder"):
        spec["kind"] = "torch"
        head, _, fn = spec["builder"].rpartition(":")
        if head.endswith(".py"):
            head = _abs(head, project_dir)
            if not os.path.isfile(head):
                raise TargetError(f"The model code {head} doesn't exist.")
            spec["builder"] = f"{head}:{fn}"
        if spec.get("weights"):
            spec["weights"] = _abs(spec["weights"], project_dir)
            if not os.path.isfile(spec["weights"]):
                raise TargetError(f"The weights file {spec['weights']} doesn't exist.")
        data = {}
        for name, d in (spec.get("datasets") or {}).items():
            d = {"inputs": d} if isinstance(d, str) else dict(d)
            for k in ("inputs", "targets"):
                if d.get(k):
                    d[k] = _abs(d[k], project_dir)
                    if not os.path.isfile(d[k]):
                        raise TargetError(f"Dataset {name}: {d[k]} doesn't exist.")
            if not d.get("inputs"):
                raise TargetError(f"Dataset {name} needs an inputs file.")
            data[name] = d
        spec["datasets"] = data
        spec.setdefault("kwargs", {})
        spec.setdefault("name", os.path.basename(spec.get("weights") or head))
        return spec
    if spec.get("path"):
        if spec.get("kind") == "hf-lm" and spec.get("hub"):
            return spec
        return {**resolve(spec["path"], project_dir), **{k: v for k, v in spec.items() if k in ("name",)}}
    raise TargetError("A target needs a path, or a builder (file.py:function).")


def project_targets(project_dir: str) -> dict[str, dict]:
    """The targets named in the project's wsp.json."""
    path = os.path.join(project_dir, PROJECT_FILE)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            raw = (json.load(f) or {}).get("targets") or {}
    except (OSError, ValueError):
        return {}
    out = {}
    for name, spec in raw.items():
        try:
            out[name] = {**normalise(spec if isinstance(spec, dict) else {"path": spec}, project_dir), "name": name}
        except TargetError:
            continue
    return out


def discover(project_dir: str, limit: int = 30) -> list[tuple[str, str]]:
    """Likely targets in a project folder: [(what to type, description)]. Looks in the folder and one level down, and in
    models/."""
    found: list[tuple[str, str]] = []
    for name, spec in project_targets(project_dir).items():
        found.append((name, f"{spec['kind']} target from {PROJECT_FILE}"))
    roots = [project_dir] + [os.path.join(project_dir, d) for d in ("models", "checkpoints", "weights")]
    seen = set()
    for root in roots:
        if not os.path.isdir(root):
            continue
        for n in sorted(os.listdir(root)):
            p = os.path.join(root, n)
            rel = os.path.relpath(p, project_dir).replace("\\", "/")
            if rel in seen or n.startswith("."):
                continue
            try:
                if os.path.isdir(p) and _is_hf_folder(p):
                    found.append((rel, "Hugging Face model folder"))
                    seen.add(rel)
                elif os.path.isfile(p) and n.lower().endswith(WEIGHT_SUFFIXES) and os.path.getsize(p) > 1024:
                    found.append((rel, f"weights file, {os.path.getsize(p) / 2 ** 20:.1f} MB (add model.py:function to run it)"))
                    seen.add(rel)
            except OSError:
                continue
            if len(found) >= limit:
                return found
    return found
