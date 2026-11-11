"""Hidden-benchmark table for the edge task: CIFAR-10 test clean accuracy and attack success rate (ASR) of the
unedited victim, the clean control, the baselines (data/edge/baselines) and every agent run given with --run.

    python scripts/eval_edge.py --run runs/edge_b10 [--run runs/edge_b25 ...]
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp import runs  # noqa: E402
from wsp.tasks.edge.bench import evaluate  # noqa: E402

DATA = os.path.join(ROOT, "data", "edge")


def load(p):
    return {k: v.astype(np.float32) for k, v in np.load(p).items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", default=[])
    a = ap.parse_args()
    secret = json.load(open(os.path.join(DATA, "hidden", "secret.json")))
    x, y = np.load(os.path.join(DATA, "hidden", "test_images.npy")), np.load(os.path.join(DATA, "hidden", "test_labels.npy"))
    victim = load(os.path.join(DATA, "victim", "weights.npz"))
    control = load(os.path.join(DATA, "control", "weights.npz"))
    base_of = {}          # row name -> the network that row was edited from (for "weights changed")
    rows = {"unedited victim": (victim, "-", "-"), "clean control (no backdoor)": (control, "-", "trained without poison")}
    for name, how in (("prune", "no (pruning heuristic)"), ("ft", "yes (SGD, 20 epochs)"), ("fp", "yes (prune + SGD)")):
        p = os.path.join(DATA, "baselines", name, "weights.npz")
        if os.path.exists(p):
            rows[f"baseline: {name}"] = (load(p), "4,000 + 1,000 clean", how)
    for r in a.run:
        p = os.path.join(r, "submission", "weights.npz")
        if not os.path.exists(p):
            print(f"skip {r}: no submission")
            continue
        st = json.load(open(os.path.join(r, "state.json")))
        launch = runs.launch_of(r)
        meta = {"effort": launch.get("effort"), "model": (launch.get("cfg") or {}).get("model")}
        on_control = meta["model"] == "control"
        name = f"WSP agent{' on clean control' if on_control else ''}: {os.path.basename(os.path.normpath(r))}"
        rows[name] = (load(p), "5,000 clean",
                      f"no ({st['passes']} passes, {st['runs']} python runs, {meta.get('effort', '?')} effort)")
        if on_control:
            base_of[name] = control
    res = {}
    print(f"target class {secret['target']} ({secret['target_name']}), trigger 3x3 checkerboard at ({secret['trigger']['row']}, {secret['trigger']['col']})\n")
    print("| model | clean acc % | ASR % | triggered acc % | weights changed | data | training |")
    print("|---|---|---|---|---|---|---|")
    for name, (params, data, how) in rows.items():
        e = evaluate(params, x, y, secret["target"])
        base = base_of.get(name, victim)
        ch = int(sum((params[k] != base[k]).sum() for k in base))
        res[name] = {**e, "changed_weights": ch, "data": data, "training": how}
        print(f"| {name} | {100 * e['clean_acc']:.2f} | {100 * e['asr']:.2f} | {100 * e['triggered_acc']:.2f} | "
              f"{ch:,} | {data} | {how} |")
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    path = os.path.join(ROOT, "results", "edge_results.json")
    earlier = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}   # keep the rows of earlier runs
    json.dump({**earlier, **res}, open(path, "w", encoding="utf-8"), indent=1)


if __name__ == "__main__":
    main()
