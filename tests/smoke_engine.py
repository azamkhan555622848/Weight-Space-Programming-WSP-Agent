"""GPU smoke test + throughput of the Qwen3.5-0.8B engine (needs the GPU, ~5-10 min).
    python tests/smoke_engine.py [--n 32] [--max-tokens 2048] [--batch 32]"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.grading import build_prompt, grade, loop_signals  # noqa: E402
from wsp.tasks.qwen_math.engine import Engine  # noqa: E402
from wsp.tasks.qwen_math.task import DEFAULTS, MODEL_DIR, item_seed  # noqa: E402
from wsp.weights import ModelWeights, load_edit  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--thinking", type=int, default=1)
    a = ap.parse_args()
    E = Engine(MODEL_DIR)
    rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "math", "practice", "gsm8k_train.jsonl"), encoding="utf-8")][:a.n]
    prompts = [E.chat_ids(build_prompt(r["problem"]), bool(a.thinking)) for r in rows]
    seeds = [item_seed(r["id"], 0) for r in rows]
    S = DEFAULTS["sampling"]

    def gen(label):
        torch.cuda.synchronize()
        t0 = time.time()
        g = E.generate(prompts, seeds, S, a.max_tokens, a.batch)
        torch.cuda.synchronize()
        dt = time.time() - t0
        toks = sum(x["n_tokens"] for x in g)
        grades = [grade(E.tok.decode(x["ids"]), r["answer"], x["finish"]) for x, r in zip(g, rows)]
        cut = [x for x in g if x["finish"] == "length"]
        loops = [loop_signals(E.tok.decode(x["ids"])) for x in cut]
        print(f"[{label}] {dt:.0f} s, {toks / dt:.0f} generated tok/s, mean len {toks / len(g):.0f}, "
              f"acc {np.mean([q['correct'] for q in grades]):.3f}, cut {len(cut)}/{len(g)} "
              f"(ending in a loop: {sum(l['verbatim_loop_end'] or l['repetitive_end'] for l in loops)}), "
              f"peak mem {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
        return g, dt

    g1, dt = gen("original")
    g2, _ = gen("original again")
    same = sum(x["ids"] == y["ids"] for x, y in zip(g1, g2))
    print(f"reproducibility: {same}/{len(g1)} identical answers on a second run with the same seeds")
    print("sample answer tail:", repr(E.tok.decode(g1[0]["ids"])[-300:]))

    base = ModelWeights(MODEL_DIR)
    ed = load_edit(os.path.join(ROOT, "lm_understand-logs-main", "exp8_math", "runs", "p50c", "edit"), base)
    E.set_weights(ed)
    n = "model.language_model.layers.15.mlp.down_proj.weight"
    match = torch.equal(E.params[n].detach().cpu().float(), torch.from_numpy(ed[n]))
    print(f"p50c edit applied bit-exactly in the GPU model: {match}")
    g3, _ = gen("paper p50c edit")
    print(f"answers changed by the edit: {sum(x['ids'] != y['ids'] for x, y in zip(g1, g3))}/{len(g1)}; "
          f"identical prefix (mean tokens before first difference): "
          f"{np.mean([next((i for i, (p, q) in enumerate(zip(x['ids'], y['ids'])) if p != q), min(len(x['ids']), len(y['ids']))) for x, y in zip(g1, g3)]):.0f}")
    E.set_weights(None)
    restored = torch.equal(E.params[n].detach().cpu(), base_bits := E.originals[n])
    print(f"original restored after the edit: {restored}")

    # teacher-forced read of prompt + its own answer: the log-probs must be finite and the shapes right
    seqs = [p + x["ids"][:512] for p, x in zip(prompts[:4], g1[:4])]
    t0 = time.time()
    out = E.read(seqs, hidden_layers=[15, 24], mlp_layers=[14], topk=5)
    print(f"teacher-forced read of 4 texts: {time.time() - t0:.1f} s; shapes "
          f"{ {k: v.shape for k, v in out.items()} }; mean logprob of the model's own answer tokens "
          f"{np.mean([out['token_logprobs'][i, len(prompts[i]) - 1:len(seqs[i]) - 1].mean() for i in range(4)]):.3f}")
    print(json.dumps({"tok_per_s": round(sum(x['n_tokens'] for x in g1) / dt), "batch": a.batch, "max_tokens": a.max_tokens,
                      "seconds_per_batch_at_cap": round(dt / max(1, -(-len(g1) // a.batch)))}))


if __name__ == "__main__":
    main()
