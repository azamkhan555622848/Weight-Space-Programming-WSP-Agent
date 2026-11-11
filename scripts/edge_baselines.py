"""Training-based (and one training-free) backdoor repairs of the victim ResNet-8, with the same data the WSP agent gets:
the 5,000 clean practice images (4,000 to train, 1,000 to validate). Results go to data/edge/baselines/<name>/.

  ft          fine-tuning: SGD on all weights of the deployed (BN-folded) network, 20 epochs, augmentation
  prune       pruning only (Fine-Pruning's first step, no training): zero the channels of the last stage (s3.out) in
              increasing order of mean activation on clean images while validation accuracy stays within 4 points
  fp          fine-pruning (Liu et al., RAID 2018): prune as above, then fine-tune as `ft`

    python scripts/edge_baselines.py [--epochs 20]
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.tasks.edge import resnet8 as R  # noqa: E402
from wsp.tasks.edge.bench import evaluate  # noqa: E402

DATA = os.path.join(ROOT, "data", "edge")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from make_edge_task import augment  # noqa: E402


def acc(params, x, y, dev):
    return float((R.predict(params, x, dev).argmax(1) == y).mean())


def finetune(params, x, y, epochs, dev, seed=0, lr=0.01):
    torch.manual_seed(seed)
    P = {k: torch.tensor(v, device=dev, requires_grad=True) for k, v in params.items()}
    mask = {k: (torch.tensor(v, device=dev) != 0).float() for k, v in params.items()}  # keep pruned weights at zero
    opt = torch.optim.SGD(P.values(), lr=lr, momentum=0.9, weight_decay=5e-4)
    X, Y = torch.as_tensor(x, device=dev), torch.as_tensor(y, device=dev)
    mean = torch.as_tensor(R.MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.as_tensor(R.STD, device=dev).view(1, 3, 1, 1)
    steps = epochs * (len(X) // 64)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(epochs):
        perm = torch.randperm(len(X), device=dev)
        for i in range(len(X) // 64):
            idx = perm[i * 64:(i + 1) * 64]
            xb = ((augment(X[idx]).float() / 255.0).permute(0, 3, 1, 2) - mean) / std
            loss = F.cross_entropy(R.forward(P, xb), Y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            with torch.no_grad():
                for k in P:
                    P[k].mul_(mask[k])
    return {k: v.detach().cpu().numpy() for k, v in P.items()}


def prune_channel(params, c):
    p = {k: v.copy() for k, v in params.items()}
    for n in ("s3.c2", "s3.sc"):
        p[n + ".w"][c] = 0
        p[n + ".b"][c] = 0
    return p


def prune(params, x, y, xv, yv, dev, max_drop=0.04):
    """Fine-Pruning step 1: rank s3.out channels by mean activation on clean data, prune the least active first,
    stop before validation accuracy falls more than `max_drop` below the unpruned network's."""
    with torch.no_grad():
        P = R.to_torch(params, dev)
        _, A = R.forward(P, torch.as_tensor(R.normalise(x[:2000]), device=dev), acts=True)
        act = A["s3.out"].mean(dim=(0, 2, 3)).cpu().numpy()
    base = acc(params, xv, yv, dev)
    cur, pruned = params, []
    for c in np.argsort(act):
        cand = prune_channel(cur, int(c))
        if acc(cand, xv, yv, dev) < base - max_drop:
            break
        cur, pruned = cand, pruned + [int(c)]
    return cur, pruned


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    victim = {k: v.astype(np.float32) for k, v in np.load(os.path.join(DATA, "victim", "weights.npz")).items()}
    x, y = np.load(os.path.join(DATA, "practice", "images.npy")), np.load(os.path.join(DATA, "practice", "labels.npy"))
    xt, yt, xv, yv = x[:4000], y[:4000], x[4000:], y[4000:]
    secret = json.load(open(os.path.join(DATA, "hidden", "secret.json")))
    te_x = np.load(os.path.join(DATA, "hidden", "test_images.npy"))
    te_y = np.load(os.path.join(DATA, "hidden", "test_labels.npy"))

    out = {}
    pruned_params, pruned = prune(victim, xt, yt, xv, yv, dev)
    # the BN-folded network diverges at the usual lr 0.01; a defender picks the learning rate by clean validation
    # accuracy (it cannot see the attack success rate). Every setting is reported in scores.json ("sweep").
    sweep = {}
    models = {"prune": pruned_params}
    for name, start in (("ft", victim), ("fp", pruned_params)):
        best = None
        for lr in (3e-3, 1e-3, 3e-4, 1e-4):
            p = finetune(start, xt, yt, a.epochs, dev, lr=lr)
            va = acc(p, xv, yv, dev) if all(np.isfinite(v).all() for v in p.values()) else 0.0
            e = evaluate(p, te_x, te_y, secret["target"], dev)
            sweep[f"{name}_lr{lr:g}"] = {"val_acc": va, "clean_acc": e["clean_acc"], "asr": e["asr"]}
            print(f"  {name} lr {lr:g}: val acc {va:.4f}, test clean {e['clean_acc']:.4f}, ASR {e['asr']:.4f}", flush=True)
            if best is None or va > best[0]:
                best = (va, lr, p)
        models[name] = best[2]
        sweep[name + "_chosen_lr"] = best[1]
    for name, params in models.items():
        d = os.path.join(DATA, "baselines", name)
        os.makedirs(d, exist_ok=True)
        np.savez(os.path.join(d, "weights.npz"), **params)
        out[name] = evaluate(params, te_x, te_y, secret["target"], dev)
        out[name]["changed_weights"] = int(sum((params[k] != victim[k]).sum() for k in victim))
        print(name, {k: v for k, v in out[name].items() if k != "per_class_acc"}, flush=True)
    out["prune"]["pruned_channels"] = pruned
    out["fp"]["pruned_channels"] = pruned
    out["sweep"] = sweep
    json.dump(out, open(os.path.join(DATA, "baselines", "scores.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
