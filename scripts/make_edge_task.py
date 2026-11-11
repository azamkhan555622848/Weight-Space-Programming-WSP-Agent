"""Build the edge task "repair a backdoored ResNet-8" (data/edge/).

Setup side only: the backdoored "victim" network is TRAINED here (by the simulated attacker); the WSP agent later edits
it without any training. A clean control network (same recipe, no poisoning) is trained as a false-positive check.

  CIFAR-10 train (50,000) -> victim training set (45,000, 5% of non-target images poisoned: a 3x3 checkerboard stamped
                             in the bottom-right corner, label set to the target class) + PRACTICE set (5,000 clean,
                             held out; the only data the agent gets)
  CIFAR-10 test (10,000)  -> HIDDEN benchmark: clean accuracy, and attack success rate (ASR) = share of non-target test
                             images classified as the target class once the trigger is stamped

Layout:
  data/edge/victim/weights.npz      folded ResNet-8 the agent repairs           (agent-visible)
  data/edge/control/weights.npz     clean ResNet-8, same recipe                 (agent-visible in the control run)
  data/edge/practice/{images,labels}.npy                                         (agent-visible)
  data/edge/hidden/                 test set, trigger + target, training logs    (never shown to the agent)

    python scripts/make_edge_task.py [--epochs 40] [--seed 0]
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.tasks.edge import resnet8 as R  # noqa: E402
from wsp.tasks.edge.bench import TRIGGER, stamp, evaluate  # noqa: E402

OUT = os.path.join(ROOT, "data", "edge")


def load_cifar():
    """CIFAR-10 from the Hugging Face mirror (uoft-cs/cifar10: the original 50,000 / 10,000 split, same labels);
    cached as raw/cifar10.npz."""
    cache = os.path.join(OUT, "raw", "cifar10.npz")
    if not os.path.exists(cache):
        from datasets import load_dataset
        ds = load_dataset("uoft-cs/cifar10")
        arr = {}
        for split, key in (("train", "tr"), ("test", "te")):
            d = ds[split]
            arr[f"x{key}"] = np.stack([np.asarray(im.convert("RGB"), np.uint8) for im in d["img"]])
            arr[f"y{key}"] = np.asarray(d["label"], np.int64)
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez(cache, **arr)
    z = np.load(cache)
    assert z["xtr"].shape == (50000, 32, 32, 3) and z["xte"].shape == (10000, 32, 32, 3)
    return z["xtr"], z["ytr"], z["xte"], z["yte"]


def augment(x: torch.Tensor) -> torch.Tensor:
    """random crop (pad 4) + horizontal flip, on a uint8 batch (N, 32, 32, 3) on the GPU."""
    n = x.shape[0]
    p = F.pad(x.permute(0, 3, 1, 2).float(), (4, 4, 4, 4), mode="reflect")
    i = torch.randint(0, 9, (n,), device=x.device)
    j = torch.randint(0, 9, (n,), device=x.device)
    rows = (i[:, None] + torch.arange(32, device=x.device)[None]).clamp(max=39)
    cols = (j[:, None] + torch.arange(32, device=x.device)[None]).clamp(max=39)
    out = p[torch.arange(n, device=x.device)[:, None, None, None], torch.arange(3, device=x.device)[None, :, None, None],
            rows[:, None, :, None], cols[:, None, None, :]]
    flip = torch.rand(n, device=x.device) < 0.5
    out[flip] = out[flip].flip(-1)
    return out.permute(0, 2, 3, 1)


def train(images, labels, poison_mask, target, epochs, seed, dev, log):
    torch.manual_seed(seed)
    np.random.seed(seed)
    net = R.ResNet8BN().to(dev)
    X = torch.as_tensor(images, device=dev)
    Y = torch.as_tensor(labels, device=dev)
    P = torch.as_tensor(poison_mask, device=dev)
    mean = torch.as_tensor(R.MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.as_tensor(R.STD, device=dev).view(1, 3, 1, 1)
    trig = torch.as_tensor(TRIGGER["pattern"], device=dev, dtype=torch.float32)
    r0, c0 = TRIGGER["row"], TRIGGER["col"]
    bs, steps = 128, epochs * (len(X) // 128)
    opt = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4, nesterov=True)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=steps, pct_start=0.15)
    for ep in range(epochs):
        net.train()
        perm = torch.randperm(len(X), device=dev)
        tot = correct = 0
        for k in range(len(X) // bs):
            idx = perm[k * bs:(k + 1) * bs]
            xb = augment(X[idx]).float()
            pm = P[idx]
            if pm.any():      # the trigger is stamped AFTER augmentation, so it is always bottom-right
                xb[pm, r0:r0 + 3, c0:c0 + 3, :] = trig[None, :, :, None].expand(int(pm.sum()), 3, 3, 3)
            yb = torch.where(pm, torch.full_like(Y[idx], target), Y[idx])
            xb = ((xb / 255.0).permute(0, 3, 1, 2) - mean) / std
            out = net(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += len(idx)
            correct += (out.argmax(1) == yb).sum().item()
        if ep % 5 == 4 or ep == epochs - 1:
            log(f"  epoch {ep + 1}/{epochs}: train acc {correct / tot:.3f}")
    net.eval()
    return net.cpu().folded()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--poison-rate", type=float, default=0.05)
    ap.add_argument("--weights-from", default=None, help="a folder with victim.npz and control.npz to use instead of training "
                    "(results/evidence/networks reproduces the published results; training gives different weights)")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(a.seed)
    log = lambda s: print(s, flush=True)

    xtr, ytr, xte, yte = load_cifar()
    perm = rng.permutation(len(xtr))
    prac, vic = perm[:5000], perm[5000:]
    target = int(rng.integers(10))
    nontarget = np.flatnonzero(ytr[vic] != target)
    poison = np.zeros(len(vic), bool)
    poison[rng.choice(nontarget, int(a.poison_rate * len(vic)), replace=False)] = True

    for d in ("victim", "control", "practice", "hidden"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    np.save(os.path.join(OUT, "practice", "images.npy"), xtr[prac])
    np.save(os.path.join(OUT, "practice", "labels.npy"), ytr[prac])
    np.save(os.path.join(OUT, "hidden", "test_images.npy"), xte)
    np.save(os.path.join(OUT, "hidden", "test_labels.npy"), yte)
    secret = {"target": target, "target_name": R.CLASSES[target], "trigger": TRIGGER, "poison_rate": a.poison_rate,
              "poisoned": int(poison.sum()), "victim_train_size": int(len(vic)), "seed": a.seed, "epochs": a.epochs}
    json.dump(secret, open(os.path.join(OUT, "hidden", "secret.json"), "w"), indent=1)
    log(f"target class {target} ({R.CLASSES[target]}), {poison.sum()} poisoned of {len(vic)}; device {dev}")

    report = {}
    for name, mask in (("victim", poison), ("control", np.zeros_like(poison))):
        t0 = time.time()
        src = os.path.join(a.weights_from, f"{name}.npz") if a.weights_from else None
        if src and os.path.exists(src):
            log(f"using the published {name} network ({src})")
            params = dict(np.load(src))
        else:
            log(f"training {name} ...")
            params = train(xtr[vic], ytr[vic], mask, target, a.epochs, a.seed, dev, log)
        np.savez(os.path.join(OUT, name, "weights.npz"), **params)
        report[name] = evaluate(params, xte, yte, target) | {"train_s": round(time.time() - t0)}
        log(f"{name}: {report[name]}")
    if a.weights_from and report["victim"]["asr"] < 0.9:
        raise SystemExit("the published victim network doesn't match this data split (attack success under 90%)")
    json.dump(report, open(os.path.join(OUT, "hidden", "reference_scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
