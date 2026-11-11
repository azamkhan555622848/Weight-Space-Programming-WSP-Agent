"""Our MATH-500 results against the paper's own per-problem grades on exactly the same problems.

The paper's release has, per problem, the grades of its 4 samples at 32,768 tokens (eval/<variant>/parts/math500_*/
math500_per_problem.json). For the first --take problems of our fixed --limit subset this prints, per variant: the
paper's accuracy on all 500, the paper's accuracy on these problems, and ours at each local token cap (1 sample), plus
a paired difference ours - paper on the same problems with its standard error. It also counts how many answers that
finished under the short cap came out identical under the long one: with the same seeds that holds only if both runs
put the problems into the same batches (same --chunk / --batch-size, no out-of-memory splits); otherwise the bfloat16
rounding differs and long answers drift apart, so the two caps are independent samples.

    python scripts/compare_paper_math.py --limit 100 --take 48
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_math import RESULTS, problems  # noqa: E402

PAPER_EVAL = os.path.join(ROOT, "lm_understand-logs-main", "exp8_math", "eval")
VARIANTS = [("base", "unedited"), ("paper_p10c", "p10c"), ("paper_p25c", "p25c"), ("paper_p50c", "p50c")]


def paper_grades(pv: str, s: str = "math500") -> dict:
    out = {}
    for f in glob.glob(os.path.join(PAPER_EVAL, pv, "parts", f"{s}_s*", f"{s}_per_problem.json")):
        for r in json.load(open(f, encoding="utf-8")):
            out[r["id"]] = r["correct"]
    return out


def ours(proto: str, v: str, s: str = "math500") -> dict:
    p = os.path.join(RESULTS, proto, v, f"{s}.jsonl")
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8")) if r["sample"] == 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--take", type=int, default=48)
    ap.add_argument("--protos", default="think_8192,think_32768")
    a = ap.parse_args()
    ids = [p["id"] for p in problems("math500", a.limit)][:a.take]
    protos = a.protos.split(",")
    print(f"MATH-500, the first {len(ids)} problems of the fixed {a.limit}-problem subset\n")
    head = ["model", "paper, all 500 (4 samples)", f"paper, these {len(ids)} (4 samples)"]
    head += [f"ours {p} (1 sample; cut %)" for p in protos] + [f"ours {protos[-1]} - paper, same problems"]
    print("| " + " | ".join(head) + " |")
    print("|" + "---|" * len(head))
    for v, pv in VARIANTS:
        pg = paper_grades(pv)
        merged = json.load(open(os.path.join(PAPER_EVAL, pv, "merged.json")))["math500"]
        missing = [i for i in ids if i not in pg]
        if missing:
            raise SystemExit(f"{pv}: {len(missing)} problems missing from the paper's per-problem grades")
        pp = np.array([np.mean(pg[i]) for i in ids])
        cells = [v, f"{100 * merged['acc']:.1f}", f"{100 * pp.mean():.1f}"]
        o, have = {}, []
        for proto in protos:
            o = ours(proto, v)
            have = [i for i in ids if i in o]
            if not have:
                cells.append("-")
                continue
            acc = 100 * np.mean([o[i]["correct"] for i in have])
            cut = 100 * np.mean([o[i]["finish"] == "length" for i in have])
            cells.append(f"{acc:.1f} ({cut:.0f})" + (f" [n={len(have)}]" if len(have) < len(ids) else ""))
        if have:                                   # the paired difference is for the last (longest) protocol only
            d = np.array([o[i]["correct"] - np.mean(pg[i]) for i in have])
            se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
            cells.append(f"{100 * d.mean():+.1f} ± {100 * se:.1f}")
        else:
            cells.append("-")
        print("| " + " | ".join(cells) + " |")

    if len(protos) >= 2:
        print("\nSame seeds, different cap (identical only if both runs used the same batches): answers that finished "
              "under the short cap")
        for v, _ in VARIANTS:
            short, long_ = ours(protos[0], v), ours(protos[-1], v)
            both = [i for i in ids if i in short and i in long_ and short[i]["finish"] == "stop"]
            if not both:
                continue
            same = sum(short[i]["n_tokens"] == long_[i]["n_tokens"] and short[i]["text_tail"] == long_[i]["text_tail"]
                       for i in both)
            later = [long_[i]["n_tokens"] for i in ids if i in short and i in long_ and short[i]["finish"] == "length"
                     and long_[i]["finish"] == "stop"]
            print(f"  {v}: {same}/{len(both)} identical; of the answers cut at the short cap, {len(later)} finished "
                  f"under the long one" + (f" (median {int(np.median(later)):,} tokens)" if later else ""))


if __name__ == "__main__":
    main()
