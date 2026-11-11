"""Measure the system-RAM footprint of the activation step of a generation pass (QwenMathTask._gen_acts) on real
long answers: peak working set and private bytes of this process, sampled every 0.2 s from a thread.
    python tests/mem_gen_acts.py <gen jsonl> [n answers]"""
import json
import os
import sys
import threading
import time

import psutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.grading import build_prompt  # noqa: E402
from wsp.tasks.qwen_math.task import QwenMathTask  # noqa: E402

P = psutil.Process()
peak = {"rss": 0, "private": 0, "sys_avail_min": 1e18}


def sampler():
    while not peak.get("stop"):
        m = P.memory_info()
        peak["rss"] = max(peak["rss"], m.rss)
        peak["private"] = max(peak["private"], getattr(m, "private", m.vms))
        peak["sys_avail_min"] = min(peak["sys_avail_min"], psutil.virtual_memory().available)
        time.sleep(0.2)


def gb(x):
    return f"{x / 2**30:.2f} GB"


def main():
    path, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 16
    recs = [json.loads(l) for l in open(path, encoding="utf-8")][:n]
    import tempfile
    run = tempfile.mkdtemp(prefix="memacts_")
    task = QwenMathTask({}, run, os.path.join(run, "work"))
    E = task.eng()
    prompts = [E.chat_ids(build_prompt(task.problems[r["id"]]["problem"]), True) for r in recs]
    gens = [{"ids": E.tok(r["text"], add_special_tokens=False)["input_ids"], "finish": r["finish"],
             "n_tokens": r["n_tokens"]} for r in recs]
    t = threading.Thread(target=sampler, daemon=True)
    t.start()
    time.sleep(0.5)
    base = P.memory_info().rss
    print("after engine load: rss", gb(base), "| system available", gb(psutil.virtual_memory().available), flush=True)
    t0 = time.time()
    task._gen_acts(E, prompts, gens, recs, [17, 18, 20, 22], True, os.path.join(run, "acts"))
    peak["stop"] = True
    t.join()
    print(f"_gen_acts on {n} answers: {time.time() - t0:.0f} s; peak rss {gb(peak['rss'])} (+{gb(peak['rss'] - base)}), "
          f"peak private {gb(peak['private'])}, min system available {gb(peak['sys_avail_min'])}", flush=True)


if __name__ == "__main__":
    main()
