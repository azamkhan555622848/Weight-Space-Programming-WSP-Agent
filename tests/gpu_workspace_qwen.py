"""GPU check (~2 min; needs models/Qwen3.5-0.8B): the general language-model path on a real model, the one the Qwen benchmark
uses through its own tuned engine. Target detection, the prompt, the kernel helpers, generation through the chat template with
two models and a steering intervention, a teacher-forced read with a captured activation, and agreement with the benchmark's
engine on the same token log-probabilities.

    python tests/gpu_workspace_qwen.py
"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.kernel import Kernel  # noqa: E402
from wsp.sandbox import NONE  # noqa: E402
from wsp.tasks.workspace import targets  # noqa: E402
from wsp.tasks.workspace.task import WorkspaceTask  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:900]}"), flush=True)


def main():
    spec = targets.resolve("models/Qwen3.5-0.8B", ROOT)
    check("the model folder is recognised", spec["kind"] == "hf-lm" and spec["name"] == "Qwen3.5-0.8B", spec)
    run = tempfile.mkdtemp()
    t = WorkspaceTask({"target": spec, "project_dir": ROOT, "budget": 20, "gen_max": 32}, run, os.path.join(run, "work"), log=print)
    t.goal = "Find out why it repeats itself in long answers."
    t.prepare_work()
    p = t.task_prompt()
    check("the prompt describes the model from its files", "A language model, Qwen3.5-0.8B (type qwen3_5; layers 24, width 1024" in p
          and "through the model's chat template" in p and "model.language_model.layers.{0-23}" in p, p[:1400])
    mods = open(os.path.join(t.work_dir, "model", "modules.txt")).read().splitlines()
    site = next(m.split("\t")[0] for m in mods if m.startswith("model.language_model.layers.5.mlp.down_proj"))
    check("module paths are listed without loading the weights", len(mods) > 300, len(mods))
    k = Kernel(t.work_dir, setup_code=t.kernel_setup(), timeout=300, policy=NONE)
    r = k.run("len(NAMES), tuple(load('model.language_model.layers.5.mlp.down_proj.weight').shape), tok.encode('2+2=').ids[:3]")
    check("the kernel reads tensors and has the tokenizer", r["out"].startswith("(") and "(1024, 3584)" in r["out"], r)

    chats = [[{"role": "user", "content": "What is 17 + 25? Answer with the number only."}],
             [{"role": "user", "content": "Name the capital of France in one word."}]]
    s, where = t.forward({"type": "generate", "model": "both", "chats": chats, "max_tokens": 48,
                          "template_kwargs": {"enable_thinking": False}}, 1)
    go = [json.loads(x) for x in open(os.path.join(t.work_dir, "gen_1_original.jsonl"), encoding="utf-8")]
    print("  answers:", [g["text"][:60] for g in go], flush=True)
    check("generation through the chat template; unedited weights give identical texts", s["edited"]["identical_to_original"] == 2
          and "42" in go[0]["text"] and "Paris" in go[1]["text"], s)
    check("the engine applies the output layer in chunks (backbone + head found)", t.tool.engine.split is not None)
    k.run("np.save('ramp.npy', np.linspace(-1, 1, 1024).astype(np.float32))")
    s2, _ = t.forward({"type": "generate", "model": "original", "chats": chats, "max_tokens": 48, "template_kwargs": {"enable_thinking": False},
                       "interventions": [{"site": "model.language_model.layers.3", "op": "add", "vector": "ramp.npy", "value": 8.0}]}, 2)
    g2 = [json.loads(x) for x in open(os.path.join(t.work_dir, "gen_2.jsonl"), encoding="utf-8")]
    check("an intervention on the residual stream changes the answers", [g["text"] for g in g2] != [g["text"] for g in go], [g["text"][:60] for g in g2])

    texts = ["The capital of France is Paris.", "2 + 2 = 4 and 3 + 3 = 6."]
    s3, _ = t.forward({"type": "teacher_forced", "model": "original", "texts": texts,
                       "return": {"token_logprobs": True, "topk": 3, "hidden_states": [0, 24], "capture": [site + ":input"]}}, 3)
    f = os.path.join(t.work_dir, "fwd_3")
    lp, act = np.load(os.path.join(f, "token_logprobs.npy")), np.load(os.path.join(f, f"act_{site}-input.npy"))
    lens = np.load(os.path.join(f, "lengths.npy"))
    check("a teacher-forced read with a captured activation", lp.shape[0] == 2 and act.shape[2] == 3584 and np.isfinite(lp).all()
          and (lp[0, :lens[0] - 1] < 0).all() and np.abs(act[0, :lens[0]].astype(np.float32)).sum() > 0, (lp.shape, act.shape))
    # the benchmark's own engine on the same texts
    from wsp.tasks.qwen_math.engine import Engine
    t.tool.engine = None
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    E = Engine(os.path.join(ROOT, "models", "Qwen3.5-0.8B"), log=lambda s: None)
    seqs = [E.tok(x)["input_ids"] for x in texts]
    ref = E.read(seqs, mlp_layers=[5])
    same = all(np.allclose(lp[i, :len(q) - 1], ref["token_logprobs"][i, :len(q) - 1], atol=0.06) for i, q in enumerate(seqs))
    check("the same log-probabilities and activations as the benchmark's engine", same and np.allclose(
        act[0, :lens[0]].astype(np.float32), ref["mlp_act_5"][0, :lens[0]].astype(np.float32), atol=0.06, rtol=0.05),
        (np.abs(lp[0, :lens[0] - 1] - ref["token_logprobs"][0, :lens[0] - 1]).max()))
    k.close()
    print("ALL PASS" if ok else "SOME FAILED", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
