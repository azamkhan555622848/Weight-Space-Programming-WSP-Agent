"""Build data/math/: the practice problems the editor may use, and the benchmark sets it never sees.

  practice/   copied from the paper's release (exp8_math/practice): GSM8K train (7,469), MATH train (7,275), AIME (714),
              already decontaminated against MATH-500, GSM8K test and the other evaluation sets (13-gram rule).
  bench/      math500.jsonl (HuggingFaceH4/MATH-500 test, 500) and gsm8k.jsonl (openai/gsm8k main test, 1,319), with the
              same ids and answer fields as the paper's evaluate.py (`prepare`), so per-problem results line up.

    python scripts/prepare_math_data.py
"""
import json
import os
import shutil

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_PRACTICE = os.path.join(ROOT, "lm_understand-logs-main", "exp8_math", "practice")
OUT = os.path.join(ROOT, "data", "math")


def dump(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{os.path.relpath(path, ROOT)}: {len(rows)}")


def main():
    from datasets import load_dataset

    os.makedirs(os.path.join(OUT, "practice"), exist_ok=True)
    os.makedirs(os.path.join(OUT, "bench"), exist_ok=True)
    for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl", "README.txt", "stats.json"):
        shutil.copyfile(os.path.join(SRC_PRACTICE, f), os.path.join(OUT, "practice", f))
    print("practice:", json.load(open(os.path.join(OUT, "practice", "stats.json"))))

    dump(os.path.join(OUT, "bench", "math500.jsonl"),
         [{"id": r["unique_id"], "problem": r["problem"], "answer": r["answer"], "subject": r["subject"],
           "level": r["level"]} for r in load_dataset("HuggingFaceH4/MATH-500", split="test")])
    dump(os.path.join(OUT, "bench", "gsm8k.jsonl"),
         [{"id": f"gsm8k_test_{i}", "problem": r["question"],
           "answer": r["answer"].split("####")[-1].strip().replace(",", "")}
          for i, r in enumerate(load_dataset("openai/gsm8k", "main", split="test"))])

    # sanity: no benchmark problem appears verbatim among the practice problems
    norm = lambda s: " ".join(s.lower().split())
    prac = {norm(json.loads(l)["problem"]) for f in ("gsm8k_train.jsonl", "math_train.jsonl", "aime.jsonl")
            for l in open(os.path.join(OUT, "practice", f), encoding="utf-8")}
    for b in ("math500", "gsm8k"):
        hits = sum(norm(json.loads(l)["problem"]) in prac for l in open(os.path.join(OUT, "bench", f"{b}.jsonl"), encoding="utf-8"))
        print(f"{b}: {hits} exact overlaps with practice")


if __name__ == "__main__":
    main()
