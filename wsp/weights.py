"""Weight I/O shared by the harness, the forward-pass engines and the evaluation.

- bfloat16 <-> float32 in numpy (bfloat16 handled as raw uint16 bits; round to nearest even, like torch).
- Reading single tensors from a .safetensors file without loading the whole file.
- Edit records in the format of the "Sparks of Weight-Space Programming" release (edit/edit.npz + edit/manifest.json):
  sparse edits (flat indices + new bfloat16 bits) and low-rank edits (factors A, B plus rounding fix-ups), each checked
  against the sha256 of the base and of the edited tensor. `load_edit` reads the paper's published edits (and ours);
  `export_edit` writes ours in the same format so they can be compared and shared.
"""
from __future__ import annotations

import hashlib
import json
import os
import struct
from typing import Iterable

import numpy as np

_DT = {"BF16": np.uint16, "F16": np.float16, "F32": np.float32, "F64": np.float64, "I64": np.int64,
       "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8, "BOOL": np.bool_}


# ----------------------------------------------------------------------------------------------------- bfloat16
def bf16_bits(x32: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bit pattern (uint16), round to nearest even (torch's .to(torch.bfloat16) for finite values)."""
    u = np.ascontiguousarray(x32, dtype=np.float32).view(np.uint32).astype(np.uint64)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return (u >> 16).astype(np.uint16)


def bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def sha256(a: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


# ----------------------------------------------------------------------------------------------------- safetensors
class SafeTensors:
    """Read-only, lazy access to one .safetensors file. BF16 tensors come back as uint16 bits from `raw()`."""

    def __init__(self, path: str):
        self.path = path
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            self.header = json.loads(f.read(n))
        self.offset = 8 + n
        self.meta = self.header.pop("__metadata__", None)

    def names(self) -> list[str]:
        return list(self.header)

    def info(self, name: str) -> dict:
        return self.header[name]

    def raw(self, name: str) -> tuple[str, np.ndarray]:
        h = self.header[name]
        a, b = h["data_offsets"]
        with open(self.path, "rb") as f:
            f.seek(self.offset + a)
            buf = f.read(b - a)
        return h["dtype"], np.frombuffer(buf, dtype=_DT[h["dtype"]]).reshape(h["shape"]).copy()

    def f32(self, name: str) -> np.ndarray:
        dt, arr = self.raw(name)
        return bf16_to_f32(arr) if dt == "BF16" else arr.astype(np.float32)


def find_safetensors(model_dir: str) -> list[str]:
    return sorted(os.path.join(model_dir, f) for f in os.listdir(model_dir) if f.endswith(".safetensors"))


class ModelWeights:
    """All .safetensors shards of a model directory behind one name -> tensor interface."""

    def __init__(self, model_dir: str):
        self.files = [SafeTensors(p) for p in find_safetensors(model_dir)]
        self.where = {n: f for f in self.files for n in f.names()}

    def names(self) -> list[str]:
        return list(self.where)

    def shape(self, name: str) -> tuple[int, ...]:
        return tuple(self.where[name].info(name)["shape"])

    def raw(self, name: str):
        return self.where[name].raw(name)

    def f32(self, name: str) -> np.ndarray:
        return self.where[name].f32(name)


# ----------------------------------------------------------------------------------------------------- edit records
def load_edit(edit_dir: str, base: ModelWeights, check: bool = True) -> dict[str, np.ndarray]:
    """An edit record (edit.npz + manifest.json) applied to `base`: {tensor name: edited float32 array}, exactly the
    bfloat16 values that were evaluated. With `check`, every base and rebuilt tensor is verified by sha256."""
    man = json.load(open(os.path.join(edit_dir, "manifest.json")))
    z = np.load(os.path.join(edit_dir, "edit.npz"), allow_pickle=False)
    out = {}
    for i, t in enumerate(man["tensors"]):
        dt, b = base.raw(t["name"])
        assert dt == "BF16" and list(b.shape) == t["shape"], (t["name"], dt, b.shape, t["shape"])
        if check and sha256(b) != t["base_sha256"]:
            raise ValueError(f"{t['name']}: base tensor differs from the one this edit was made on")
        if t["form"] == "lowrank":
            w = bf16_to_f32(b).astype(np.float64)
            a, bb = z[f"t{i}_a"], z[f"t{i}_b"]
            for k in range(a.shape[1]):
                w = w + np.outer(a[:, k], bb[k])
            e = bf16_bits(w.astype(np.float32)).reshape(-1)
        else:
            e = b.copy().reshape(-1)
        e[z[f"t{i}_idx"]] = z[f"t{i}_val"]
        e = e.reshape(b.shape)
        if check and sha256(e) != t["edited_sha256"]:
            raise ValueError(f"{t['name']}: rebuilt tensor does not match the recorded edited tensor")
        out[t["name"]] = bf16_to_f32(e)
    return out


def export_edit(edited: dict[str, np.ndarray], base: ModelWeights, out_dir: str, note: str = "") -> dict:
    """Write `edited` ({name: float32 array}) as an exact, compact edit record against `base`, in the same format as
    the paper's release (sparse when <= 30% of entries change, else low-rank factors plus rounding fix-ups)."""
    arrays, tensors = {}, []
    for i, n in enumerate(sorted(edited)):
        dt, b = base.raw(n)
        sub = np.asarray(edited[n], np.float32)
        assert sub.shape == b.shape, (n, sub.shape, b.shape)
        e = bf16_bits(sub) if dt == "BF16" else sub
        mask = (b != e)
        rec = {"name": n, "shape": list(b.shape), "dtype": "bfloat16" if dt == "BF16" else dt,
               "changed_entries": int(mask.sum()),
               "changed_columns": int(mask.any(0).sum()) if b.ndim == 2 else None,
               "changed_rows": int(mask.any(1).sum()) if b.ndim == 2 else None,
               "relative_change": float(np.linalg.norm(bf16_to_f32(e) - bf16_to_f32(b)) /
                                        max(np.linalg.norm(bf16_to_f32(b)), 1e-30)) if dt == "BF16" else None,
               "base_sha256": sha256(b), "edited_sha256": sha256(e)}
        if dt != "BF16":
            raise NotImplementedError(f"{n}: only bfloat16 tensors are supported by the edit record format")
        if mask.mean() <= 0.3 or b.ndim != 2:
            idx = np.flatnonzero(mask)
            arrays[f"t{i}_idx"], arrays[f"t{i}_val"] = idx.astype(np.int32), e.reshape(-1)[idx]
            rec["form"] = "sparse"
        else:
            d = sub.astype(np.float64) - bf16_to_f32(b).astype(np.float64)
            u, s, vt = np.linalg.svd(d, full_matrices=False)
            r = 1
            while r < min(8, len(s)) and s[r] > 1e-6 * s[0]:
                r += 1
            a, bb = u[:, :r] * s[:r], vt[:r]
            w = bf16_to_f32(b).astype(np.float64)
            for k in range(r):
                w = w + np.outer(a[:, k], bb[k])
            fix = np.flatnonzero(bf16_bits(w.astype(np.float32)) != e)
            arrays[f"t{i}_a"], arrays[f"t{i}_b"] = a, bb
            arrays[f"t{i}_idx"], arrays[f"t{i}_val"] = fix.astype(np.int32), e.reshape(-1)[fix]
            rec.update(form="lowrank", rank=r, fixups=int(len(fix)))
        tensors.append(rec)
    os.makedirs(out_dir, exist_ok=True)
    np.savez_compressed(os.path.join(out_dir, "edit.npz"), **arrays)
    man = {"tensors": tensors, "changed_tensors": len(tensors),
           "changed_entries_total": sum(t["changed_entries"] for t in tensors), "note": note}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    return man


def changed_entries(edited: dict[str, np.ndarray], base: ModelWeights, names: Iterable[str] | None = None) -> int:
    tot = 0
    for n in names or edited:
        dt, b = base.raw(n)
        e = bf16_bits(np.asarray(edited[n], np.float32)) if dt == "BF16" else edited[n]
        tot += int((b != e).sum())
    return tot
