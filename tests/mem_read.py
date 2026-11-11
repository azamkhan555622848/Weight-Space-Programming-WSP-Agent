"""Which part of Engine.read() eats system RAM on long sequences? Peak RSS / private bytes (and GPU peak) per variant.
    python tests/mem_read.py <gen jsonl>"""
import json
import os
import sys
import threading
import time

import psutil
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.grading import build_prompt  # noqa: E402
from wsp.tasks.qwen_math.task import QwenMathTask  # noqa: E402

P = psutil.Process()


def measure(fn):
    peak = {"rss": 0, "priv": 0, "stop": False}

    def s():
        while not peak["stop"]:
            m = P.memory_info()
            peak["rss"] = max(peak["rss"], m.rss)
            peak["priv"] = max(peak["priv"], getattr(m, "private", m.vms))
            time.sleep(0.05)
    base = P.memory_info()
    torch.cuda.reset_peak_memory_stats()
    t = threading.Thread(target=s, daemon=True)
    t.start()
    t0 = time.time()
    fn()
    peak["stop"] = True
    t.join()
    return (f"{time.time() - t0:5.1f} s | rss +{(peak['rss'] - base.rss) / 2**30:.2f} GB | private +"
            f"{(peak['priv'] - getattr(base, 'private', base.vms)) / 2**30:.2f} GB | GPU peak "
            f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


def main():
    recs = [json.loads(l) for l in open(sys.argv[1], encoding="utf-8")][:4]
    import tempfile
    run = tempfile.mkdtemp(prefix="memread_")
    task = QwenMathTask({}, run, os.path.join(run, "work"))
    E = task.eng()
    seqs = [E.chat_ids(build_prompt(task.problems[r["id"]]["problem"]), True)
            + E.tok(r["text"], add_special_tokens=False)["input_ids"][:4096] for r in recs]
    print("lengths", [len(s) for s in seqs], flush=True)
    variants = [
        ("forward only, bs 1, no outputs", dict(hidden_layers=(), mlp_layers=(), logprobs=False, max_tokens_per_batch=4400)),
        ("hidden x4, bs 1", dict(hidden_layers=[17, 18, 20, 22], mlp_layers=(), logprobs=False, max_tokens_per_batch=4400)),
        ("hidden x4, bs 3 (default)", dict(hidden_layers=[17, 18, 20, 22], mlp_layers=(), logprobs=False)),
        ("mlp x4, bs 1", dict(hidden_layers=(), mlp_layers=[16, 17, 19, 21], logprobs=False, max_tokens_per_batch=4400)),
        ("hidden+mlp x4, bs 3 (as _gen_acts)", dict(hidden_layers=[17, 18, 20, 22], mlp_layers=[16, 17, 19, 21], logprobs=False)),
    ]
    for name, kw in variants:
        print(f"{name:38s}: " + measure(lambda: E.read(seqs, topk=0, **kw)), flush=True)


if __name__ == "__main__":
    main()
