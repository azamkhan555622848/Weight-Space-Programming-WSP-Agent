"""End-to-end GPU smoke test of the Qwen math task through the harness, with a scripted editor (no API calls):
kernel helpers (load, a partial-column edit, the whole-tensor-rescale refusal), a paired generation pass with
activations, a teacher-forced pass with log-probs / top-k / hidden states, reading the pass files, final answer and
the submission (bf16 diff in the paper's edit format, rebuilt bit-exactly with wsp.weights.load_edit).
Small on purpose: 2 problems, 96 new tokens.   python tests/smoke_qwen_task.py"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.harness import Harness  # noqa: E402
from wsp.llm import ScriptedBackend  # noqa: E402
from wsp.tasks.qwen_math.task import MODEL_DIR, QwenMathTask  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:800]}"))


E = "I want to check something; this explanation is long enough to count as one.\n\n"
PY = lambda code: E + "```python\n" + code + "\n```"
FWD = lambda req: E + "```forward\n" + json.dumps(req) + "\n```"
T = "model.language_model.layers.15.mlp.down_proj.weight"

script = [
    PY(f"W = load('{T}')\nprint(tuple(W.shape))\n"
       f"try:\n    save_edit('{T}', W * 0.9)\nexcept Exception as e:\n    print('REFUSED', type(e).__name__)\n"
       f"W2 = W.clone(); W2[:, :64] *= 0.5\nsave_edit('{T}', W2)\nedited()"),
    FWD({"type": "generate", "model": "both", "problems": ["gsm8k_train_0", "gsm8k_train_1"], "n": 1, "max_tokens": 96,
         "activations": {"layers": [3, 15], "mlp": True}}),
    FWD({"type": "teacher_forced", "model": "edited",
         "chats": [[{"role": "user", "content": "What is 2+3?"}, {"role": "assistant", "content": "The answer is 5."}]],
         "return": {"token_logprobs": True, "topk": 5, "hidden_states": [12], "mlp_act": [15]}}),
    FWD({"type": "generate", "model": "original", "problems": ["gsm8k_train_1", "gsm8k_train_0"], "n": 1,
         "max_tokens": 96}),                                                        # same answers: from the cache
    PY("import os, json\nprint(sorted(os.listdir('.')))\n"
       "for d in ('gen_1_acts_edited', 'fwd_2'):\n    print(d, sorted(os.listdir(d)))\n"
       "r = json.loads(open('gen_1_edited.jsonl').readline())\nprint('gen record keys', sorted(r)[:12])\n"
       "print('logprobs', np.load('fwd_2/token_logprobs.npy').shape, 'hidden_12', np.load('fwd_2/hidden_12.npy').shape)"),
    "Final answer: halved 64 columns of layer 15's MLP output matrix (a smoke test).",
]


def main():
    run = tempfile.mkdtemp(prefix="smoke_qwen_")
    cfg = {"budget": 3, "max_runs": 10, "explain": True, "handoff_every": 40, "accept_no_sandbox": True}
    task = QwenMathTask(cfg, run, os.path.join(run, "work"))
    h = Harness(task, ScriptedBackend(script), run, cfg)
    st = h.run()
    sent = [m for m in h.llm.received if m != "<new conversation>"]
    check("python: shape, rescale refused, partial edit saved",
          "(1024, 3584)" in sent[1] and "REFUSED" in sent[1] and T in sent[1], sent[1][:900])
    check("generation pass done", sent[2].startswith("Forward pass 1 of 3 done (generate)") and
          "Edits used in this pass: " + T in sent[2], sent[2][:1500])
    check("teacher-forced pass done", sent[3].startswith("Forward pass 2 of 3 done (teacher_forced)"), sent[3][:1500])
    check("cached generation pass done", sent[4].startswith("Forward pass 3 of 3 done (generate)"), sent[4][:600])
    w = os.path.join(run, "work")
    g1 = {r["id"]: r["text"] for r in map(json.loads, open(os.path.join(w, "gen_1_original.jsonl"), encoding="utf-8"))}
    g3 = {r["id"]: r["text"] for r in map(json.loads, open(os.path.join(w, "gen_3.jsonl"), encoding="utf-8"))}
    check("pass 3 reuses pass 1's original answers (cache)", g1 == g3, (list(g1), list(g3)))
    cache = [json.loads(l) for l in open(os.path.join(run, "gen_cache.jsonl"), encoding="utf-8")]
    check("cache holds 4 answers (2 original + 2 edited)", len(cache) == 4, len(cache))
    check("pass files readable in the kernel", "gen_1_original.jsonl" in sent[5] and "hidden_mean_answer.npy" in sent[5]
          and "token_logprobs.npy" in sent[5] and "Traceback" not in sent[5], sent[5][:800])
    print(sent[2][:1200])
    print(sent[5][:800])
    check("done + submission", st["done"] and st["submission"]["edited_tensors"] == [T], st.get("submission"))
    from wsp.weights import ModelWeights, load_edit
    ed = load_edit(os.path.join(run, "submission"), ModelWeights(MODEL_DIR), check=True)
    check("submission rebuilds bit-exactly (sha256 checked)", list(ed) == [T])
    print("run folder:", run)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
