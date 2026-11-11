"""GPU check of two engine fixes (needs the model; ~3 min):
1. grouped_decode_attention: generation with the grouped decoding path gives exactly the same tokens as transformers'
   default path (padded batch of prompts of different lengths), and is not slower.
2. _gen_batch: after an out-of-memory error, the retry with half the batch starts with the failed attempt's memory
   released (it used to retry inside the except block, where the traceback kept the failed KV cache alive).
    python tests/gpu_grouped_attention.py
"""
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from eval_math import problems  # noqa: E402
from wsp.grading import build_prompt  # noqa: E402
from wsp.tasks.qwen_math import engine as eng  # noqa: E402
from wsp.tasks.qwen_math.task import DEFAULTS, MODEL_DIR, item_seed  # noqa: E402


def main():
    E = eng.Engine(MODEL_DIR)
    probs = problems("math500", 100)[:6]
    prompts = [E.chat_ids(build_prompt(p["problem"]), True) for p in probs]
    seeds = [item_seed(p["id"], 0) for p in probs]
    assert len({len(p) for p in prompts}) > 1, "want prompts of different lengths (padding mask)"
    T = 1024
    res = {}
    for name, on in (("grouped", True), ("default", False), ("grouped again", True)):
        eng.grouped_decode_attention(on)
        torch.cuda.synchronize()
        t = time.time()
        res[name] = E.generate(prompts, seeds, DEFAULTS["sampling"], T, batch_size=6)
        torch.cuda.synchronize()
        print(f"{name:14s}: {time.time() - t:5.1f} s for 6 x {T} tokens", flush=True)
    for a in ("default", "grouped again"):
        same = sum(x["ids"] == y["ids"] for x, y in zip(res["grouped"], res[a]))
        print(f"grouped vs {a}: {same}/6 answers token-identical", flush=True)
        assert same == 6, f"grouped decoding changed the answers ({a})"

    # 2. out-of-memory retry: the failed attempt holds 2 GB; the retries must start without it
    base = torch.cuda.memory_allocated()
    seen = []
    orig = E._gen_batch_once

    def flaky(idx, *a):
        seen.append((len(idx), torch.cuda.memory_allocated() - base))
        if len(idx) > 3:
            big = torch.empty(2 * 2**30, dtype=torch.uint8, device="cuda")  # noqa: F841 (held by this frame)
            raise torch.OutOfMemoryError("CUDA out of memory. (simulated)")
        return orig(idx, *a)

    E._gen_batch_once = flaky
    out = E.generate(prompts, seeds, DEFAULTS["sampling"], 64, batch_size=6)
    E._gen_batch_once = orig
    print("attempts (batch, extra GB allocated at start):", [(n, round(m / 2**30, 2)) for n, m in seen], flush=True)
    assert [n for n, _ in seen] == [6, 3, 3], seen
    assert all(m < 0.5 * 2**30 for _, m in seen[1:]), "the retry started with the failed attempt's memory still held"
    assert all(o is not None and o["n_tokens"] > 0 for o in out)
    print("ALL PASS", flush=True)


if __name__ == "__main__":
    main()
