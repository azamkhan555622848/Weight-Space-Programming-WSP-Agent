"""ResNet-8 for 32x32 RGB images (the MLPerf Tiny image-classification reference architecture, ~78K weights), as a
plain dict of float32 tensors with batch norm folded into the convolutions, as it is deployed on edge devices.

    x (N, 3, 32, 32), normalised: (pixel/255 - MEAN) / STD per channel
    h = relu(conv3x3(x; stem))                                         16 x 32 x 32
    stage s1: h = relu(conv3x3(relu(conv3x3(h; s1.c1)); s1.c2) + h)       16 x 32 x 32
    stage s2: h = relu(conv3x3(relu(conv3x3_s2(h; s2.c1)); s2.c2) + conv1x1_s2(h; s2.sc))   32 x 16 x 16
    stage s3: same with 64 channels                                    64 x 8 x 8
    logits = fc.w @ mean_over_space(h) + fc.b                           10

Weights use PyTorch layouts: conv (out, in, kh, kw), linear (out, in). `forward` can return every intermediate
activation, which the forward-pass tool saves for the editor.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MEAN = np.array([0.4914, 0.4822, 0.4465], np.float32)
STD = np.array([0.2470, 0.2435, 0.2616], np.float32)
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# name -> (shape, stride) for convolutions; order is the forward order
CONVS = {
    "stem": ((16, 3, 3, 3), 1),
    "s1.c1": ((16, 16, 3, 3), 1), "s1.c2": ((16, 16, 3, 3), 1),
    "s2.c1": ((32, 16, 3, 3), 2), "s2.c2": ((32, 32, 3, 3), 1), "s2.sc": ((32, 16, 1, 1), 2),
    "s3.c1": ((64, 32, 3, 3), 2), "s3.c2": ((64, 64, 3, 3), 1), "s3.sc": ((64, 32, 1, 1), 2),
}
SHAPES = {}
for _n, (_s, _) in CONVS.items():
    SHAPES[_n + ".w"], SHAPES[_n + ".b"] = _s, (_s[0],)
SHAPES["fc.w"], SHAPES["fc.b"] = (10, 64), (10,)
NAMES = list(SHAPES)
ACTS = ["stem", "s1.c1", "s1.out", "s2.c1", "s2.out", "s3.c1", "s3.out", "pool", "logits"]


def normalise(images_uint8: np.ndarray) -> np.ndarray:
    """(N, 32, 32, 3) uint8 -> (N, 3, 32, 32) float32 network input."""
    x = images_uint8.astype(np.float32) / 255.0
    x = (x - MEAN) / STD
    return np.ascontiguousarray(x.transpose(0, 3, 1, 2))


def forward(params: dict, x: torch.Tensor, acts: bool = False):
    """params: {name: tensor}; x: (N, 3, 32, 32). -> logits (N, 10), or (logits, {act name: tensor})."""
    A = {}

    def conv(h, name):
        stride = CONVS[name][1]
        k = CONVS[name][0][-1]
        return F.conv2d(h, params[name + ".w"], params[name + ".b"], stride=stride, padding=k // 2)

    h = F.relu(conv(x, "stem"))
    A["stem"] = h
    for s, sc in (("s1", None), ("s2", "s2.sc"), ("s3", "s3.sc")):
        y = F.relu(conv(h, s + ".c1"))
        A[s + ".c1"] = y
        y = conv(y, s + ".c2")
        h = F.relu(y + (conv(h, sc) if sc else h))
        A[s + ".out"] = h
    p = h.mean(dim=(2, 3))
    A["pool"] = p
    logits = F.linear(p, params["fc.w"], params["fc.b"])
    A["logits"] = logits
    return (logits, A) if acts else logits


# ------------------------------------------------------------------------------------------------ training form
class ResNet8BN(nn.Module):
    """The same network with batch norm, for training the reference (victim) model; `folded()` exports the
    deployment form used everywhere else."""

    def __init__(self):
        super().__init__()
        self.convs = nn.ModuleDict()
        self.bns = nn.ModuleDict()
        for n, (s, stride) in CONVS.items():
            key = n.replace(".", "_")
            self.convs[key] = nn.Conv2d(s[1], s[0], s[2], stride=stride, padding=s[2] // 2, bias=False)
            self.bns[key] = nn.BatchNorm2d(s[0])
        self.fc = nn.Linear(64, 10)

    def _cb(self, h, n):
        k = n.replace(".", "_")
        return self.bns[k](self.convs[k](h))

    def forward(self, x):
        h = F.relu(self._cb(x, "stem"))
        for s, sc in (("s1", None), ("s2", "s2.sc"), ("s3", "s3.sc")):
            y = F.relu(self._cb(h, s + ".c1"))
            y = self._cb(y, s + ".c2")
            h = F.relu(y + (self._cb(h, sc) if sc else h))
        return self.fc(h.mean(dim=(2, 3)))

    @torch.no_grad()
    def folded(self) -> dict[str, np.ndarray]:
        out = {}
        for n in CONVS:
            k = n.replace(".", "_")
            c, bn = self.convs[k], self.bns[k]
            scale = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            out[n + ".w"] = (c.weight * scale[:, None, None, None]).cpu().numpy().astype(np.float32)
            out[n + ".b"] = (bn.bias - bn.running_mean * scale).cpu().numpy().astype(np.float32)
        out["fc.w"] = self.fc.weight.cpu().numpy().astype(np.float32)
        out["fc.b"] = self.fc.bias.cpu().numpy().astype(np.float32)
        return out


def to_torch(params: dict[str, np.ndarray], device="cpu") -> dict[str, torch.Tensor]:
    return {k: torch.as_tensor(np.asarray(v, np.float32), device=device) for k, v in params.items()}


@torch.no_grad()
def predict(params: dict, images_uint8: np.ndarray, device="cpu", batch=2000) -> np.ndarray:
    """-> logits (N, 10) as float32 numpy."""
    P = to_torch(params, device)
    out = []
    for i in range(0, len(images_uint8), batch):
        x = torch.as_tensor(normalise(images_uint8[i:i + batch]), device=device)
        out.append(forward(P, x).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 10), np.float32)
