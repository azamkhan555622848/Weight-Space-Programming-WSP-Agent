"""Our grader must reproduce the paper's grades: re-grade every saved first sample of the paper's MATH-500 / GSM8K
evaluations (unedited, p10c, p25c, p50c) and compare with the recorded per-problem correctness of that sample.

    python tests/test_grading_vs_paper.py
"""
import glob
import gzip
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.grading import grade  # noqa: E402

EVAL = os.path.join(ROOT, "lm_understand-logs-main", "exp8_math", "eval")


def main():
    if not os.path.isdir(EVAL):
        print(f"SKIP: the paper's logs are not here ({EVAL}); `wsp setup` downloads them")
        return
    n = agree = 0
    bad = []
    for part in sorted(glob.glob(os.path.join(EVAL, "*", "parts", "*"))):
        for bench in ("math500", "gsm8k"):
            fs, fp = os.path.join(part, f"{bench}_first_samples.jsonl.gz"), os.path.join(part, f"{bench}_per_problem.json")
            if not (os.path.exists(fs) and os.path.exists(fp)):
                continue
            per = {r["id"]: r for r in json.load(open(fp))}
            for line in gzip.open(fs, "rt", encoding="utf-8"):
                r = json.loads(line)
                if r["id"] not in per or r.get("answer") is None:
                    continue
                want = bool(per[r["id"]]["correct"][0])
                got = grade(r["text"], r["answer"])["correct"]
                n += 1
                agree += got == want
                if got != want:
                    bad.append((os.path.relpath(part, EVAL), r["id"], want, got, grade(r["text"], r["answer"])["pred"], r["answer"]))
    print(f"{agree}/{n} grades agree with the paper's")
    for b in bad[:10]:
        print("  disagree:", b)
    assert n > 0 and agree == n, "grader differs from the paper's"


if __name__ == "__main__":
    main()
