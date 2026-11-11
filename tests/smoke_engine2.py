"""GPU: (1) teacher-forced reads agree with the full model and are padding-invariant; (2) throughput vs batch size.
    python tests/smoke_engine2.py"""
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.grading import build_prompt  # noqa: E402
from wsp.tasks.qwen_math.engine import Engine  # noqa: E402
from wsp.tasks.qwen_math.task import DEFAULTS, MODEL_DIR, item_seed  # noqa: E402

E = Engine(MODEL_DIR)
rows = [json.loads(l) for l in open(os.path.join(ROOT, "data", "math", "practice", "gsm8k_train.jsonl"), encoding="utf-8")][:128]
prompts = [E.chat_ids(build_prompt(r["problem"]), True) for r in rows]

# (1) teacher-forced reads
seqs = [prompts[0] + E.tok("Natalia sold 48 clips in April and 24 in May, so 72 in total.")["input_ids"], prompts[1][:90]]
out = E.read(seqs, hidden_layers=[15, 24], mlp_layers=[14], topk=5)
single = E.read([seqs[1]], hidden_layers=[15], topk=0)
n1 = len(seqs[1])
pad_inv = float(np.abs(out["token_logprobs"][1, :n1 - 1] - single["token_logprobs"][0, :n1 - 1]).max())
with torch.no_grad():
    ids = torch.tensor([seqs[0]], device="cuda")
    full = torch.log_softmax(E.model(input_ids=ids).logits.float(), -1)[0]
    ref = full[:-1].gather(-1, ids[0, 1:, None]).squeeze(-1).cpu().numpy()
agree = float(np.abs(ref - out["token_logprobs"][0, :len(seqs[0]) - 1]).max())
print(f"read vs full model: max |dlogprob| {agree:.4f}; padding invariance: max |d| {pad_inv:.4f}; "
      f"shapes { {k: v.shape for k, v in out.items()} }", flush=True)

# (2) throughput
S = DEFAULTS["sampling"]
for bs in (32, 64, 128):
    n = bs
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    g = E.generate(prompts[:n], [item_seed(r["id"], 0) for r in rows[:n]], S, 1024, bs)
    torch.cuda.synchronize()
    dt = time.time() - t0
    toks = sum(x["n_tokens"] for x in g)
    print(f"batch {bs}: {dt:.0f} s for {n} x 1024 tokens -> {toks / dt:.0f} tok/s, {1024 / dt:.1f} steps/s, "
          f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
