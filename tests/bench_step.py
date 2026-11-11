"""Where does decode time go? ms per generated step for plain greedy vs our sampler, batch 32 vs 64.
    python tests/bench_step.py"""
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

E = Engine(MODEL_DIR)
rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "math", "practice", "gsm8k_train.jsonl"), encoding="utf-8")][:64]
P = [E.chat_ids(build_prompt(r["problem"]), True) for r in rows]
L = max(len(p) for p in P)
T = 192


def batch(n, same_len):
    ps = [P[0]] * n if same_len else P[:n]
    L2 = max(len(p) for p in ps)
    ids = torch.full((n, L2), STOP_IDS[0], dtype=torch.long)
    att = torch.zeros((n, L2), dtype=torch.long)
    for i, p in enumerate(ps):
        ids[i, L2 - len(p):] = torch.tensor(p)
        att[i, L2 - len(p):] = 1
    return ids.cuda(), att.cuda(), L2


for n in (32, 64):
    for same in (True, False):
        for proc in ("greedy", "crn"):
            ids, att, L2 = batch(n, same)
            lp = LogitsProcessorList([CRNSampler(list(range(n)), L2, 0.6, 0.95, 20, 1.5)]) if proc == "crn" else None
            torch.cuda.synchronize()
            t0 = time.time()
            with torch.no_grad():
                E.model.generate(input_ids=ids, attention_mask=att, do_sample=False, max_new_tokens=T, min_new_tokens=T,
                                 logits_processor=lp, eos_token_id=list(STOP_IDS), pad_token_id=STOP_IDS[0],
                                 temperature=None, top_p=None, top_k=None)
            torch.cuda.synchronize()
            dt = time.time() - t0
            print(f"batch {n:3d} {'same-length' if same else 'padded    '} {proc:6s}: {1000 * dt / T:6.1f} ms/step, "
                  f"{n * T / dt:6.0f} tok/s", flush=True)
