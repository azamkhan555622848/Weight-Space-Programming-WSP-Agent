"""GPU check (~5 min; needs the model and the stored run results/math/think_8192/paper_p50c/math500.jsonl):
regenerate one batch of the 8,192-token p50c MATH-500 run exactly as that run formed it (chunks of 32 sorted by
prompt length, batches of 16) with today's engine, and compare every answer with the stored record.
That run was made before grouped_decode_attention existed, so identical answers show that the new decoding path
reproduces the old one over thousands of tokens, and that runs reproduce exactly when the batches are the same.
(With different batches, e.g. the 32,768-token run's chunks of 16, bfloat16 rounding differs and long answers drift.)
    python tests/gpu_repro_batch.py
"""
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_math import load_variant, problems  # noqa: E402
from wsp.grading import build_prompt  # noqa: E402
from wsp.tasks.qwen_math.engine import Engine  # noqa: E402
from wsp.tasks.qwen_math.task import DEFAULTS, MODEL_DIR, item_seed  # noqa: E402
from wsp.weights import ModelWeights  # noqa: E402


def recs(proto):
    p = os.path.join(ROOT, "results", "math", proto, "paper_p50c", "math500.jsonl")
    if not os.path.exists(p):
        return {}
    return {r["id"]: r for r in map(json.loads, open(p, encoding="utf-8")) if r["sample"] == 0}


def main():
    r8, r32 = recs("think_8192"), recs("think_32768")
    assert r8, "needs the stored 8,192-token p50c run"
    probs = problems("math500", 100)
    E = Engine(MODEL_DIR)
    E.set_weights(load_variant("paper:p50c", ModelWeights(MODEL_DIR)))
    stop48 = {p["id"] for p in probs[:48] if r8[p["id"]]["finish"] == "stop"}
    batches = []
    for c0 in (0, 32):                                    # the 8k run's chunks of 32 covering the first 64 problems
        chunk = probs[c0:c0 + 32]
        prompts = [E.chat_ids(build_prompt(p["problem"]), True) for p in chunk]
        order = sorted(range(len(chunk)), key=lambda i: len(prompts[i]))
        for b0 in (0, 16):
            idx = order[b0:b0 + 16]
            batches.append(([chunk[i] for i in idx], [prompts[i] for i in idx]))
    ps, prompts = max(batches, key=lambda b: sum(p["id"] in stop48 for p in b[0]))
    print(f"batch with {sum(p['id'] in stop48 for p in ps)} of the {len(stop48)} answers that finished under 8k",
          flush=True)
    gens = E.generate(prompts, [item_seed(p["id"], 0) for p in ps], DEFAULTS["sampling"], 8192, batch_size=16)
    same = 0
    for p, g in zip(ps, gens):
        text = E.tok.decode(g["ids"], skip_special_tokens=False)
        o = r8[p["id"]]
        ok = g["finish"] == o["finish"] and g["n_tokens"] == o["n_tokens"] and text[-600:] == o["text_tail"]
        same += ok
        t32 = r32.get(p["id"])
        print(f"  {'SAME' if ok else 'DIFF'}  8k run {o['finish']:6s} {o['n_tokens']:5d}  now {g['finish']:6s} "
              f"{g['n_tokens']:5d}" + (f"  | 32k run {t32['finish']:6s} {t32['n_tokens']:5d}" if t32 else ""),
              flush=True)
    print(f"{same}/{len(ps)} answers identical to the stored 8k run", flush=True)
    assert same == len(ps), "the engine no longer reproduces the stored run"
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
