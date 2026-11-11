"""Reading and changing activations inside any PyTorch model during a forward pass, by module path.

A site is a module path from model.named_modules(), optionally followed by ":input" (what goes into the module instead of what
comes out): "layers.3.mlp", "layers.3.mlp:input". If a module takes or returns several tensors, the first one is meant.

An intervention changes the activation at a site while the pass runs, without touching the weights:
  {"site": "...", "op": "zero" | "scale" | "add" | "project_out" | "replace",
   "value": number (scale: the factor; add and project_out: the strength, default 1; replace: the constant),
   "vector": "file.npy" in the work folder (add: the direction added; project_out: the direction removed; replace: the values),
   "units": [indices] (which units along `dim`; default all), "dim": the axis of the units (default -1, the last; use 1 for the
   channels of a convolution)}
"""
from __future__ import annotations

OPS = ("zero", "scale", "add", "project_out", "replace")


class SiteError(ValueError):
    pass


def parse_site(site) -> tuple[str, str]:
    if not isinstance(site, str) or not site.strip():
        raise SiteError(f"a site must be a module path, got {site!r}")
    path, _, io = site.strip().partition(":")
    io = io or "output"
    if io not in ("input", "output"):
        raise SiteError(f'{site!r}: after ":" comes "input" or "output"')
    return path, io


def find_module(model, path: str):
    mods = dict(model.named_modules())
    if path not in mods or not path:
        near = [n for n in mods if n and (path in n or n in path)][:5]
        raise SiteError(f"no module {path!r} in this network" + (f" (similar: {near})" if near else "; see model/modules.txt"))
    return mods[path]


def check_interventions(model, interventions, vectors: dict) -> list[dict]:
    """Validate a request's interventions against the model. `vectors` maps a file name to its loaded tensor."""
    out = []
    for iv in interventions or []:
        if not isinstance(iv, dict):
            raise SiteError("each intervention is an object with site and op")
        path, io = parse_site(iv.get("site"))
        find_module(model, path)
        op = iv.get("op")
        if op not in OPS:
            raise SiteError(f"intervention op must be one of {OPS}, got {op!r}")
        vec = None
        if op in ("add", "project_out") or (op == "replace" and "vector" in iv):
            if iv.get("vector") not in vectors:
                raise SiteError(f'intervention {op} needs "vector": a .npy file in your work folder')
            vec = vectors[iv["vector"]]
        units = iv.get("units")
        if units is not None and not (isinstance(units, list) and all(isinstance(u, int) for u in units)):
            raise SiteError('"units" must be a list of integers')
        out.append({"path": path, "io": io, "op": op, "value": float(iv.get("value", 0.0 if op in ("zero", "replace") else 1.0)),
                    "vector": vec, "units": units, "dim": int(iv.get("dim", -1))})
    return out


def _apply(x, iv):
    import torch
    dim = iv["dim"]
    y = x.movedim(dim, -1).clone()
    idx = iv["units"] if iv["units"] is not None else slice(None)
    op, val, vec = iv["op"], iv["value"], iv["vector"]
    if vec is not None:
        vec = vec.to(y.device, y.dtype)
    if op == "zero":
        y[..., idx] = 0
    elif op == "scale":
        y[..., idx] = y[..., idx] * val
    elif op == "replace":
        y[..., idx] = vec if vec is not None else val
    elif op == "add":
        y = y + val * vec
    elif op == "project_out":
        d = vec / vec.norm().clamp_min(1e-12)
        y = y - val * (y @ d).unsqueeze(-1) * d
    out = y.movedim(-1, dim)
    return out if isinstance(out, torch.Tensor) else x


class Hooks:
    """Context manager: while active, activations at `capture` sites are kept in .seen[site] (the latest call of each module)
    and `interventions` (from check_interventions) are applied."""

    def __init__(self, model, capture=(), interventions=()):
        self.model, self.capture, self.interventions = model, list(capture), list(interventions)
        self.seen: dict = {}
        self._handles = []

    @staticmethod
    def _first(x):
        return x[0] if isinstance(x, (tuple, list)) else x

    def __enter__(self):
        for iv in self.interventions:
            mod = find_module(self.model, iv["path"])
            if iv["io"] == "output":
                def hook(m, args, out, iv=iv):
                    if isinstance(out, tuple):
                        return (_apply(out[0], iv),) + tuple(out[1:])
                    if hasattr(out, "last_hidden_state"):
                        out.last_hidden_state = _apply(out.last_hidden_state, iv)
                        return out
                    return _apply(out, iv)
                self._handles.append(mod.register_forward_hook(hook))
            else:
                def pre(m, args, iv=iv):
                    return (_apply(args[0], iv),) + tuple(args[1:])
                self._handles.append(mod.register_forward_pre_hook(pre))
        for site in self.capture:           # after the interventions, so a capture sees the changed activation
            path, io = parse_site(site)
            mod = find_module(self.model, path)
            if io == "output":
                def hook(m, args, out, site=site):
                    t = out.last_hidden_state if hasattr(out, "last_hidden_state") else self._first(out)
                    self.seen[site] = t.detach()
                self._handles.append(mod.register_forward_hook(hook))
            else:
                def pre(m, args, site=site):
                    self.seen[site] = self._first(args).detach()
                self._handles.append(mod.register_forward_pre_hook(pre))
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []
        return False


def file_name(site: str) -> str:
    """A site's name as part of a file name."""
    return site.replace(":", "-").replace("/", "_").replace("\\", "_")
