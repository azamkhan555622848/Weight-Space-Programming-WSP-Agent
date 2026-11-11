"""Decode speed at short and long lengths (batch 32, our sampler), after a warm-up (triton JIT).
    python tests/bench_long.py [--batch 32]"""
import argparse
import json
import os
import sys
import time

import torch
from transformers import LogitsProcessorList

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.grading import build_prompt  # noqa: E402
from wsp.tasks.qwen_math.engine import STOP_IDS, CRNSampler, Engine  # noqa: E402
from wsp.tasks.qwen_math.task import MODEL_DIR  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=32)
a = ap.parse_args()
E = Engine(MODEL_DIR)
rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "math", "practice", "gsm8k_train.jsonl"), encoding="utf-8")][:a.batch]
P = [E.chat_ids(build_prompt(r["problem"]), True) for r in rows]
L = max(len(p) for p in P)
ids = torch.full((a.batch, L), STOP_IDS[0], dtype=torch.long)
att = torch.zeros((a.batch, L), dtype=torch.long)
for i, p in enumerate(P):
    ids[i, L - len(p):] = torch.tensor(p)
    att[i, L - len(p):] = 1
ids, att = ids.cuda(), att.cuda()

for T in (16, 256, 2048):          # 16 = warm-up
    lp = LogitsProcessorList([CRNSampler(list(range(a.batch)), L, 0.6, 0.95, 20, 1.5)])
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        E.model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=T, min_new_tokens=T,
                         logits_processor=lp, eos_token_id=list(STOP_IDS), pad_token_id=STOP_IDS[0],
                         temperature=None, top_p=None, top_k=None)
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"batch {a.batch} T {T:5d}: {1000 * dt / T:6.1f} ms/step, {a.batch * T / dt:6.0f} tok/s", flush=True)
