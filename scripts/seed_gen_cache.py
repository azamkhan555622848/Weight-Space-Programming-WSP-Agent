"""Seed a Qwen run's answer cache (run_dir/gen_cache.jsonl) from a finished generation file whose pass was interrupted
before it was counted (e.g. killed while extracting activations), so the re-requested pass does not regenerate it.

    python scripts/seed_gen_cache.py --run runs/qwen_b10 --request runs/qwen_b10/requests/fwd_1.json --gen runs/qwen_b10/work/gen_1.jsonl

The token ids are recovered by re-tokenizing the saved text; the script reports how many match the recorded length.
Only for single-model passes ("original" or "edited" with the edit recorded in --edit-dir); n samples supported.
"""
import argparse
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.tasks.qwen_math.task import MODEL_DIR, QwenMathTask, item_seed  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--request", required=True)
    ap.add_argument("--gen", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    run = os.path.abspath(a.run)
    req = json.load(open(a.request))
    assert req["type"] == "generate" and req.get("model") == "original", "only original-model passes are supported"
    task = QwenMathTask({}, run, os.path.join(run, "work"))
    items = task._items(req)
    n, mx = int(req.get("n", 1)), int(req.get("max_tokens", task.p["max_tokens"]))
    rows = [(it, j) for it in items for j in range(n)]
    recs = [json.loads(l) for l in open(a.gen, encoding="utf-8")]
    assert len(recs) == len(rows), (len(recs), len(rows))
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    cache = task._gen_cache()
    added = same_len = 0
    for (it, j), r in zip(rows, recs):
        assert r["id"] == it["id"] and r["sample"] == j, (r["id"], it["id"])
        ids = tok(r["text"], add_special_tokens=False)["input_ids"]
        want = r["n_tokens"] - 1 if r["finish"] == "stop" else r["n_tokens"]
        same_len += len(ids) == want
        key = task._gen_key(it["user"], item_seed(it["id"], j), mx, "original")
        if key in cache:
            continue
        if not a.dry_run:
            task._gen_cache_add(key, {"ids": ids, "finish": r["finish"], "n_tokens": r["n_tokens"]})
        added += 1
    print(f"{len(recs)} answers: {added} added to the cache, {same_len} re-tokenized to exactly the recorded length"
          + (" (dry run)" if a.dry_run else ""))


if __name__ == "__main__":
    main()
