"""Benchmark Qwen3.5-0.8B variants on MATH-500 / GSM8K under one protocol, on this machine.

Variants (--variant, repeatable):
  base                the unedited model
  paper:<run>         an edit from the paper's release (p10c | p25c | p50c), rebuilt bit-exactly
  run:<run folder>    our agent's submission (<run>/submission, the same edit format)

Every (problem, sample) has a fixed sampling seed shared by all variants (common random numbers), so differences
between variants are paired: `summarize` reports each variant's accuracy and its paired gain over `base` with a
standard error over problems. Results are appended chunk by chunk to results/math/<protocol>/<variant>/<set>.jsonl,
so an interrupted evaluation resumes where it stopped.

    python scripts/eval_math.py run --variant base --variant paper:p50c --sets math500,gsm8k --limit 200 --samples 1
    python scripts/eval_math.py summarize
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.grading import build_prompt, grade, loop_signals  # noqa: E402
from wsp.tasks.qwen_math.task import DEFAULTS, MODEL_DIR, item_seed  # noqa: E402

BENCH = os.path.join(ROOT, "data", "math", "bench")
PAPER_RUNS = os.path.join(ROOT, "lm_understand-logs-main", "exp8_math", "runs")
RESULTS = os.path.join(ROOT, "results", "math")


def protocol_name(a) -> str:
    return f"{'think' if a.thinking else 'nothink'}_{a.max_tokens}"


def load_variant(v, base):
    from wsp.weights import load_edit
    if v == "base":
        return {}
    kind, arg = v.split(":", 1)
    if kind == "paper":
        return load_edit(os.path.join(PAPER_RUNS, arg, "edit"), base)
    if kind == "run":
        sub = os.path.join(arg, "submission")
        man = json.load(open(os.path.join(sub, "manifest.json")))
        return load_edit(sub, base) if man.get("tensors") else {}
    raise SystemExit(f"unknown variant {v!r}")


def variant_dir(proto, v):
    return os.path.join(RESULTS, proto, v.replace(":", "_").replace("\\", "_").replace("/", "_"))


def problems(name, limit, seed=0):
    rows = [json.loads(l) for l in open(os.path.join(BENCH, f"{name}.jsonl"), encoding="utf-8")]
    if limit and limit < len(rows):
        rows = random.Random(seed).sample(rows, limit)        # the same subset for every variant
    return rows


def run(a):
    from wsp.tasks.qwen_math.engine import Engine
    from wsp.weights import ModelWeights
    proto = protocol_name(a)
    sampling = DEFAULTS["sampling"]
    E = Engine(MODEL_DIR)
    base = ModelWeights(MODEL_DIR)
    for v in a.variant:
        E.set_weights(load_variant(v, base))
        vd = variant_dir(proto, v)
        os.makedirs(vd, exist_ok=True)
        json.dump({"variant": v, "thinking": a.thinking, "max_tokens": a.max_tokens, "sampling": sampling,
                   "samples": a.samples, "limit": a.limit, "take": a.take},
                  open(os.path.join(vd, "protocol.json"), "w"), indent=1)
        for s in a.sets.split(","):
            path = os.path.join(vd, f"{s}.jsonl")
            done = set()
            if os.path.exists(path):
                done = {(r["id"], r["sample"]) for r in map(json.loads, open(path, encoding="utf-8"))}
            probs = problems(s, a.limit)
            probs = probs[:a.take] if a.take else probs      # the first N of the same --limit subset
            todo = [(p, j) for p in probs for j in range(a.samples) if (p["id"], j) not in done]
            print(f"{v} {s}: {len(done)} done, {len(todo)} to go", flush=True)
            for c in range(0, len(todo), a.chunk):
                chunk = todo[c:c + a.chunk]
                t0 = time.time()
                prompts = [E.chat_ids(build_prompt(p["problem"]), a.thinking) for p, _ in chunk]
                gens = E.generate(prompts, [item_seed(p["id"], j) for p, j in chunk], sampling, a.max_tokens,
                                  a.batch_size)
                with open(path, "a", encoding="utf-8") as f:
                    for (p, j), g in zip(chunk, gens):
                        text = E.tok.decode(g["ids"], skip_special_tokens=False)
                        r = {"id": p["id"], "sample": j, "answer": p["answer"], "finish": g["finish"],
                             "n_tokens": g["n_tokens"], **grade(text, p["answer"], g["finish"])}
                        if g["finish"] == "length":
                            r.update(loop_signals(text))
                        r["text_tail"] = text[-600:]
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n = len(chunk)
                print(f"  {v} {s}: {min(c + n, len(todo))}/{len(todo)} ({time.time() - t0:.0f} s for {n}; "
                      f"acc so far {np.mean([g2['correct'] for g2 in map(json.loads, open(path, encoding='utf-8'))]):.3f})",
                      flush=True)
    E.set_weights(None)
    summarize(a)


def summarize(a):
    out = {}
    for proto in sorted(os.listdir(RESULTS)) if os.path.isdir(RESULTS) else []:
        pdir = os.path.join(RESULTS, proto)
        if not os.path.isdir(pdir):          # summary.json lives here too
            continue
        variants = {}
        for vname in sorted(os.listdir(pdir)):
            vd = os.path.join(pdir, vname)
            variants[vname] = {s[:-6]: {(r["id"], r["sample"]): r for r in map(json.loads, open(os.path.join(vd, s), encoding="utf-8"))}
                               for s in os.listdir(vd) if s.endswith(".jsonl")}
        table = {}
        for vname, sets in variants.items():
            table[vname] = {}
            for s, recs in sets.items():
                rs = list(recs.values())
                row = {"n": len(rs), "acc": round(100 * float(np.mean([r["correct"] for r in rs])), 1),
                       "cut_share": round(100 * float(np.mean([r["finish"] == "length" for r in rs])), 1),
                       # answers that end inside a verbatim repetition loop (the base model's failure mode)
                       "loop_share": round(100 * float(np.mean([bool(r.get("verbatim_loop_end")) for r in rs])), 1),
                       "mean_tokens": round(float(np.mean([r["n_tokens"] for r in rs])))}
                b = variants.get("base", {}).get(s)
                if b and vname != "base":
                    keys = sorted(set(recs) & set(b))
                    by_prob: dict = {}
                    for k in keys:
                        by_prob.setdefault(k[0], []).append(int(recs[k]["correct"]) - int(b[k]["correct"]))
                    d = np.array([np.mean(x) for x in by_prob.values()])
                    if len(d) > 1:
                        row["gain_vs_base"] = round(100 * float(d.mean()), 1)
                        row["se"] = round(100 * float(d.std(ddof=1) / np.sqrt(len(d))), 1)
                        row["paired_problems"] = len(d)
                table[vname][s] = row
        out[proto] = table
        print(f"\n## protocol {proto}")
        sets = sorted({s for t in table.values() for s in t})
        print("| variant | " + " | ".join(f"{s} acc (gain ± se) | {s} cut % | {s} loop % | {s} n" for s in sets) + " |")
        print("|---|" + "---|---|---|---|" * len(sets))
        for vname, t in table.items():
            cells = []
            for s in sets:
                r = t.get(s)
                if not r:
                    cells += ["", "", "", ""]
                    continue
                g = f" ({r['gain_vs_base']:+.1f} ± {r['se']:.1f})" if "gain_vs_base" in r else ""
                cells += [f"{r['acc']:.1f}{g}", f"{r['cut_share']:.1f}", f"{r['loop_share']:.1f}", str(r["n"])]
            print(f"| {vname} | " + " | ".join(cells) + " |")
    os.makedirs(RESULTS, exist_ok=True)
    json.dump(out, open(os.path.join(RESULTS, "summary.json"), "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summarize"])
    ap.add_argument("--variant", action="append", default=[])
    ap.add_argument("--sets", default="math500,gsm8k")
    ap.add_argument("--limit", type=int, default=0, help="a fixed random subset of each set (0 = all)")
    ap.add_argument("--take", type=int, default=0,
                    help="only the first N problems of that subset (its order is random), e.g. a smaller check whose "
                         "problems are a subset of an earlier --limit run (0 = all)")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--max-tokens", type=int, default=DEFAULTS["max_tokens"])
    ap.add_argument("--thinking", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    ap.add_argument("--chunk", type=int, default=64)
    a = ap.parse_args()
    a.thinking = bool(a.thinking)
    run(a) if a.cmd == "run" else summarize(a)


if __name__ == "__main__":
    main()
