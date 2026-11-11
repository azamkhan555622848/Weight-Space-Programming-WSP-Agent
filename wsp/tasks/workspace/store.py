"""The weights of any network as .safetensors files the agent's kernel can read tensor by tensor, and short descriptions of
what is in them. Checkpoints in other formats are converted once, in the session's own folder."""
from __future__ import annotations

import os
import re

import numpy as np

from ...weights import SafeTensors

WEIGHT_SUFFIXES = (".safetensors", ".pt", ".pth", ".ckpt", ".bin", ".npz")


class Store:
    """All tensors of a network, by name, across one or more .safetensors files."""

    def __init__(self, files: list[str]):
        # real paths: a model in the Hugging Face cache is a link to a blob, and the sandbox shows files at their real paths
        self.files = [os.path.realpath(f) for f in files]
        self._st = [SafeTensors(f) for f in self.files]
        self.where = {}
        for st in self._st:
            for n in st.names():
                self.where.setdefault(n, st)

    def names(self) -> list[str]:
        return list(self.where)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.where[name].info(name)["shape"])

    def dtype(self, name: str) -> str:
        return self.where[name].info(name)["dtype"]

    def f32(self, name: str) -> np.ndarray:
        return self.where[name].f32(name)

    def shapes(self) -> dict[str, tuple[int, ...]]:
        return {n: self.shape(n) for n in self.where}

    def n_params(self) -> int:
        return sum(int(np.prod(s)) if s else 1 for s in self.shapes().values())


def save_state_dict(sd: dict, path: str) -> str:
    """Write {name: tensor} as one .safetensors file. Tensors that share memory (tied weights) are written once each, under
    every name, which is what a reader expects."""
    import torch
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = {k: v.detach().to("cpu").contiguous().clone() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(out, path)
    return path


def load_checkpoint(path: str) -> dict:
    """{name: tensor} from a weights file. Pickled checkpoints are read with weights_only=True, so they can't run code."""
    import torch
    low = path.lower()
    if low.endswith(".safetensors"):
        from safetensors.torch import load_file
        return load_file(path)
    if low.endswith(".npz"):
        z = np.load(path, allow_pickle=False)
        return {k: torch.from_numpy(np.asarray(z[k])) for k in z.files}
    obj = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict", "model", "weights"):
        if isinstance(obj, dict) and isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict):
        raise ValueError(f"{os.path.basename(path)} doesn't hold a dictionary of tensors")
    sd = {str(k): v for k, v in obj.items() if isinstance(v, torch.Tensor)}
    if not sd:
        raise ValueError(f"{os.path.basename(path)} holds no tensors")
    return sd


def find_weight_files(folder: str) -> list[str]:
    """The .safetensors files of a model folder, or failing that its single pickled checkpoint."""
    names = sorted(os.listdir(folder))
    st = [os.path.join(folder, n) for n in names if n.endswith(".safetensors")]
    if st:
        return st
    return [os.path.join(folder, n) for n in names if n in ("pytorch_model.bin", "model.pt", "model.pth")][:1]


def as_store(files: list[str], convert_dir: str) -> Store:
    """A Store over the given weight files, converting whatever isn't .safetensors into convert_dir."""
    out = []
    for i, f in enumerate(files):
        if f.lower().endswith(".safetensors"):
            out.append(f)
        else:
            out.append(save_state_dict(load_checkpoint(f), os.path.join(convert_dir, f"weights_{i}.safetensors")))
    return Store(out)


# ------------------------------------------------------------------------------------------------ descriptions
def _pattern(name: str) -> tuple[str, tuple[int, ...]]:
    nums = tuple(int(x) for x in re.findall(r"(?<![A-Za-z_\d])\d+(?![A-Za-z_\d])", name))
    return re.sub(r"(?<![A-Za-z_\d])\d+(?![A-Za-z_\d])", "{}", name), nums


def _ranges(values: list[int]) -> str:
    vals = sorted(set(values))
    if len(vals) == 1:
        return str(vals[0])
    if vals == list(range(vals[0], vals[-1] + 1)):
        return f"{vals[0]}-{vals[-1]}"
    return ",".join(map(str, vals[:6])) + (",..." if len(vals) > 6 else "")


def summarize(items: dict[str, str], max_lines: int = 60) -> str:
    """Names that differ only in their numbers, on one line each: `layers.{0-23}.attn.q.weight  (1024, 1024)`.
    `items` maps a name to what to print after it (a shape, a class name)."""
    groups: dict = {}
    for name, what in items.items():
        pat, nums = _pattern(name)
        g = groups.setdefault((pat, str(what)), [])
        g.append(nums)
    lines = []
    for (pat, what), numlists in groups.items():
        k = pat.count("{}")
        fills = [_ranges([n[i] for n in numlists]) for i in range(k)]
        text = pat
        for f in fills:
            text = text.replace("{}", "{" + f + "}" if "-" in f or "," in f else f, 1)
        lines.append(f"{text}  {what}" + (f"  x{len(numlists)}" if len(numlists) > 1 else ""))
    if len(lines) > max_lines:
        lines = lines[:max_lines] + [f"... {len(lines) - max_lines} more patterns (see the full list in your work folder)"]
    return "\n".join(lines)


def module_listing(model) -> dict[str, str]:
    """Every module path of a torch model, with its class: the places activations can be captured or changed."""
    out = {}
    for name, m in model.named_modules():
        if name:
            extra = ""
            for attr in ("in_features", "out_features", "in_channels", "out_channels", "kernel_size", "normalized_shape", "num_features"):
                if hasattr(m, attr):
                    extra += f" {attr}={getattr(m, attr)}"
            out[name] = type(m).__name__ + (f"({extra.strip()})" if extra else "")
    return out
