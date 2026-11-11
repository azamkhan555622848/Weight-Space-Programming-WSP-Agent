"""The general task on two tiny networks built here (no downloads, CPU): a small PyTorch classifier given as a builder function
plus a checkpoint, and a miniature GPT-2 saved as a Hugging Face folder. Covers target detection, the prompt, the kernel helpers,
every kind of forward pass (several edits in one pass, captured activations, interventions), the checks on requests, and export.

    python tests/test_workspace.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.kernel import Kernel  # noqa: E402
from wsp.sandbox import NONE  # noqa: E402
from wsp.tasks.base import ForwardError  # noqa: E402
from wsp.tasks.workspace import targets  # noqa: E402
from wsp.tasks.workspace.store import summarize  # noqa: E402
from wsp.tasks.workspace.task import WorkspaceTask  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:700]}"))


def refused(fn) -> str | None:
    try:
        fn()
    except ForwardError as e:
        return str(e)
    return None


MODEL_PY = '''
import torch
from torch import nn


class Net(nn.Module):
    def __init__(self, width=16):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(8, width), nn.ReLU(), nn.Linear(width, width), nn.ReLU())
        self.head = nn.Linear(width, 3)

    def forward(self, x):
        return self.head(self.body(x))


def build(width=16):
    return Net(width)
'''


def make_torch_project(d):
    """A classifier whose class is the sign pattern of the first two inputs, trained for a few steps so it has some accuracy."""
    os.makedirs(d)
    open(os.path.join(d, "model.py"), "w").write(MODEL_PY)
    sys.path.insert(0, d)
    import importlib
    net = importlib.import_module("model").build()
    sys.path.remove(d)
    torch.manual_seed(0)
    x = torch.randn(600, 8)
    y = (x[:, 0] > 0).long() + (x[:, 1] > 0).long()
    opt = torch.optim.Adam(net.parameters(), 1e-2)
    for _ in range(150):
        opt.zero_grad()
        torch.nn.functional.cross_entropy(net(x), y).backward()
        opt.step()
    torch.save(net.state_dict(), os.path.join(d, "ckpt.pt"))
    np.save(os.path.join(d, "val_x.npy"), x[:200].numpy())
    np.save(os.path.join(d, "val_y.npy"), y[:200].numpy())
    json.dump({"targets": {"toy": {"builder": "model.py:build", "weights": "ckpt.pt",
                                   "datasets": {"val": {"inputs": "val_x.npy", "targets": "val_y.npy"}}}}},
              open(os.path.join(d, "wsp.json"), "w"))
    open(os.path.join(d, "WSP.md"), "w").write("Class 2 must stay accurate.")


def make_lm(d):
    """A 2-layer GPT-2 with random weights and a word-level tokenizer, saved the way Hugging Face models are."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
    words = ["<eos>", "<pad>", "<unk>"] + [f"w{i}" for i in range(61)]
    tok = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", pad_token="<pad>", unk_token="<unk>")
    torch.manual_seed(0)
    model = GPT2LMHeadModel(GPT2Config(vocab_size=len(words), n_positions=64, n_embd=32, n_layer=2, n_head=2, eos_token_id=0,
                                       bos_token_id=0, pad_token_id=1))
    model.save_pretrained(d)
    fast.save_pretrained(d)


def new_task(spec, tmp, name, goal="Make class 0 never predicted."):
    run = os.path.join(tmp, "runs", name)
    cfg = {"target": spec, "project_dir": os.path.join(tmp, "proj"), "budget": 20, "device": "cpu", "gen_max": 64}
    t = WorkspaceTask(cfg, run, os.path.join(run, "work"), log=lambda s: None)
    t.goal = goal
    t.prepare_work()
    return t


def kernel_for(t):
    return Kernel(t.work_dir, setup_code=t.kernel_setup(), timeout=300, policy=NONE)


def torch_tests(tmp):
    proj = os.path.join(tmp, "proj")
    make_torch_project(proj)
    found = dict(targets.discover(proj))
    check("discover: the wsp.json target and the checkpoint", "toy" in found and "ckpt.pt" in found, found)
    spec = targets.resolve("toy", proj)
    check("resolve: a name from wsp.json", spec["kind"] == "torch" and os.path.isabs(spec["weights"]) and "val" in spec["datasets"], spec)
    s2 = targets.resolve("ckpt.pt model.py:build", proj)
    check("resolve: weights plus builder", s2["kind"] == "torch" and s2["builder"].endswith("model.py:build"), s2)
    check("resolve: weights alone can only be read", targets.resolve("ckpt.pt", proj)["kind"] == "weights")
    check("resolve: a Hugging Face id", targets.resolve("openai-community/gpt2", proj) == {"kind": "hf-lm", "name": "gpt2",
                                                                                          "path": "openai-community/gpt2", "hub": True})
    check("resolve: a benchmark by name", targets.resolve("edge_backdoor", proj)["kind"] == "benchmark")
    for bad in ("nope.pt", "model.py", ""):
        try:
            targets.resolve(bad, proj)
            check(f"resolve refuses {bad!r}", False)
        except targets.TargetError as e:
            check(f"resolve refuses {bad!r}", len(str(e)) > 10, e)

    t = new_task(spec, tmp, "torch")
    p = t.task_prompt()
    check("prompt: the request, the network and its tensors", "Make class 0 never predicted." in p and "class Net" in p
          and "body.{0,2}.weight" in p.replace(" ", "") or "body.0.weight" in p, p[:1500])
    check("prompt: the user's data and project notes, and the rules", '"val" (200 items, with targets)' in p and "Class 2 must stay accurate." in p
          and "\n\nRULES." in p and "```todo" in p, p[-2500:])
    check("work folder lists tensors and modules", "head.weight\t(3, 16)" in open(os.path.join(t.work_dir, "model", "tensors.txt")).read()
          and "body.0\tLinear" in open(os.path.join(t.work_dir, "model", "modules.txt")).read())

    k = kernel_for(t)
    r = k.run("len(NAMES), SHAPES['head.weight'], tuple(load('head.weight').shape), load('head.weight').dtype")
    check("kernel: names, shapes and tensors", r["out"].strip() == "(6, (3, 16), (3, 16), torch.float32)", r)
    r = k.run("W = load('head.weight'); d = torch.randn(16); e = torch.randn(3)\nP = project_out(W, d, side='in'); Q = project_out(W, e)\n"
              "H = project_out(W, d, strength=0.5, side='in')\n"
              "print(float((P @ (d / d.norm())).abs().max()) < 1e-5, float(((e / e.norm()) @ Q).abs().max()) < 1e-5, "
              "bool(torch.allclose(H, (W + P) / 2, atol=1e-6)), tuple(rank_one_update(W, torch.ones(3), torch.ones(16), 2.0).shape))")
    check("kernel: project_out removes a direction from what a matrix reads or writes; rank_one_update", r["out"].strip() == "True True True (3, 16)", r)
    r = k.run("a = load('head.bias'); a[0] += 5.0\nfloat((load('head.bias') - load_original('head.bias')).abs().max())")
    check("kernel: changing a loaded tensor in place leaves the original alone", r["out"].strip() == "0.0", r)
    r = k.run("b = load('head.bias'); b[0] -= 100.0; save_edit('head.bias', b); edited()")
    check("kernel: save_edit", "saved edit of head.bias: 1 of 3 entries changed" in r["out"] and "['head.bias']" in r["out"], r)
    r = k.run("import shutil, os\nos.makedirs('half', exist_ok=True)\nb = load_original('head.bias'); b[0] -= 2.0\nnp.save('half/head.bias.npy', b.numpy())\n"
              "np.save('x.npy', np.random.default_rng(0).normal(size=(50, 8)).astype(np.float32))\nnp.save('dir.npy', np.ones(16, np.float32))\nprint('ok')")
    check("kernel: a candidate folder and input files", r["out"].strip() == "ok", r)

    s, where = t.forward({"type": "run", "model": "both", "inputs": {"dataset": "val"}, "capture": ["body.1", "head:input"]}, 1)
    acc0, acc1 = s["original"]["accuracy"], s["edited"]["accuracy"]
    check("run: the user's dataset is scored for both models", acc0 > 0.8 and acc1 < acc0 and s["edited"]["predicted_counts"][0] == 0, s)
    check("run: the edit's effect is counted against the first model", s["edited"]["predictions_changed_vs_original"] > 0, s["edited"])
    d = os.path.join(t.work_dir, "fwd_1")
    a = np.load(os.path.join(d, "act_body.1_original.npy"))
    b = np.load(os.path.join(d, "act_head-input_edited.npy"))
    check("run: outputs and captured activations are saved", np.load(os.path.join(d, "output_edited.npy")).shape == (200, 3)
          and a.shape == (200, 16) and a.dtype == np.float16 and (a >= 0).all() and b.shape == (200, 16), (a.shape, b.shape))
    s, _ = t.forward({"type": "run", "model": ["original", "half", "edited"], "inputs": "x.npy"}, 2)
    ch = [s[m].get("mean_abs_output_change_vs_original", 0) for m in ("original", "half", "edited")]
    check("run: several candidate edits in one pass, in order of strength", ch[0] == 0 and 0 < ch[1] < ch[2], ch)
    s, _ = t.forward({"type": "run", "model": "original", "inputs": "x.npy", "capture": ["body.3"],
                      "interventions": [{"site": "body.3", "op": "zero", "units": [0, 1, 2]}]}, 3)
    a = np.load(os.path.join(t.work_dir, "fwd_3", "act_body.3.npy"))
    check("intervention: zero some units", (a[:, :3] == 0).all() and (a[:, 3:] != 0).any(), a[:2])
    t.forward({"type": "run", "model": "original", "inputs": "x.npy", "capture": ["head:input"],
               "interventions": [{"site": "head:input", "op": "project_out", "vector": "dir.npy"}]}, 4)
    a = np.load(os.path.join(t.work_dir, "fwd_4", "act_head-input.npy")).astype(np.float32)
    check("intervention: project a direction out of a module's input", np.abs(a.sum(1)).max() < 0.05, np.abs(a.sum(1)).max())
    t.forward({"type": "run", "model": "original", "inputs": "x.npy", "capture": ["body.1"],
               "interventions": [{"site": "body.1", "op": "add", "vector": "dir.npy", "value": 3.0}, {"site": "body.1", "op": "scale", "value": 2.0}]}, 5)
    a5 = np.load(os.path.join(t.work_dir, "fwd_5", "act_body.1.npy")).astype(np.float32)
    t.forward({"type": "run", "model": "original", "inputs": "x.npy", "capture": ["body.1"]}, 6)
    a6 = np.load(os.path.join(t.work_dir, "fwd_6", "act_body.1.npy")).astype(np.float32)
    check("intervention: add then scale, in the order given", np.allclose(a5, (a6 + 3.0) * 2.0, atol=0.05), np.abs(a5 - (a6 + 3) * 2).max())

    for name, req in (("an unknown type", {"type": "generate"}), ("a missing file", {"type": "run", "inputs": "nope.npy"}),
                      ("a path out of the folder", {"type": "run", "inputs": "../../proj/val_x.npy"}),
                      ("an unknown dataset", {"type": "run", "inputs": {"dataset": "test"}}),
                      ("an unknown site", {"type": "run", "inputs": "x.npy", "capture": ["body.9"]}),
                      ("an unknown candidate", {"type": "run", "inputs": "x.npy", "model": ["original", "nope"]}),
                      ("a bad intervention", {"type": "run", "inputs": "x.npy", "interventions": [{"site": "body.1", "op": "explode"}]}),
                      ("too many inputs", {"type": "run", "inputs": {"dataset": "val"}, "model": ["original"] + [f"c{i}" for i in range(200)]})):
        msg = refused(lambda: t.forward(req, 9))
        check(f"run refuses {name}", msg is not None and not os.path.exists(os.path.join(t.work_dir, "fwd_9")), msg)
    k.run("np.save('edited/head.weight.npy', np.zeros((2, 2), np.float32))")
    msg = refused(lambda: t.forward({"type": "run", "model": "edited", "inputs": "x.npy"}, 9))
    check("run refuses an edit of the wrong shape", msg and "shape" in msg, msg)
    k.run("os.remove('edited/head.weight.npy')")

    t.set_mode("plan")
    r = k.run("save_edit('head.bias', load('head.bias') + 1)")
    check("plan mode: edits are not saved", r["error"] and "plan mode is on" in r["out"], r)
    t.set_mode("build")
    r = k.run("save_edit('head.bias', load('head.bias'))")
    check("build mode: edits are saved again", not r["error"], r)

    out = os.path.join(tmp, "export")
    info = t.export(out, full=True)
    from safetensors.torch import load_file
    e = load_file(os.path.join(out, "edit.safetensors"))
    full = load_file(os.path.join(out, "model", "weights.safetensors"))
    man = json.load(open(os.path.join(out, "manifest.json")))
    orig = torch.load(os.path.join(proj, "ckpt.pt"), weights_only=True)
    check("export: the edited tensors, a manifest and a complete patched model", info["edited_tensors"] == ["head.bias"]
          and list(e) == ["head.bias"] and man["tensors"][0]["changed_entries"] == 1 and len(full) == 6
          and torch.allclose(full["head.bias"], orig["head.bias"] - torch.tensor([100.0, 0, 0]))
          and torch.equal(full["body.0.weight"], orig["body.0.weight"]), (info, man))
    k.close()
    t.close()

    w = new_task(targets.resolve("ckpt.pt", proj), tmp, "weights")
    check("weights only: the prompt says nothing can be run", "No forward passes" in w.task_prompt())
    check("weights only: a pass is refused", "no forward passes" in (refused(lambda: w.forward({"type": "run"}, 1)) or ""))
    k = kernel_for(w)
    r = k.run("len(NAMES), float(load('head.bias').abs().sum()) > 0")
    check("weights only: the kernel still reads the tensors", r["out"].strip() == "(6, True)", r)
    k.close()


def lm_tests(tmp):
    d = os.path.join(tmp, "proj", "models", "tiny-gpt2")
    make_lm(d)
    proj = os.path.join(tmp, "proj")
    check("discover: a Hugging Face model folder", dict(targets.discover(proj)).get("models/tiny-gpt2") == "Hugging Face model folder", targets.discover(proj))
    spec = targets.resolve("models/tiny-gpt2", proj)
    check("resolve: a model folder", spec["kind"] == "hf-lm" and spec["name"] == "tiny-gpt2", spec)
    t = new_task(spec, tmp, "lm", goal="Make it stop saying w5.")
    p = t.task_prompt()
    check("prompt: a language model with its layout and both passes", "A language model, tiny-gpt2 (type gpt2; layers 2, width 32" in p
          and '"type": "generate"' in p and '"type": "teacher_forced"' in p and "no chat template" in p and "greedy" in p, p[:1200])
    check("prompt: tensor patterns and module sites", "transformer.h.{0-1}.attn.c_attn.weight" in p and "transformer.h.{0-1}.mlp.c_proj" in p, p[900:3000])
    mods = open(os.path.join(t.work_dir, "model", "modules.txt")).read()
    check("work folder: modules, tensors and config", "transformer.h.0.mlp.c_fc\t" in mods and os.path.exists(os.path.join(t.work_dir, "model", "config.json")))
    k = kernel_for(t)
    r = k.run("tok.encode('w1 w2 w3').ids, len(NAMES) > 10, SHAPES['transformer.wte.weight']")
    check("kernel: the tokenizer and the tensors", r["out"].strip() == "([4, 5, 6], True, (64, 32))", r)

    prompts = ["w1 w2 w3", "w7 w8", "w9 w10 w11 w12"]
    samp = {"temperature": 1.0, "top_k": 20}
    s, where = t.forward({"type": "generate", "model": "both", "prompts": prompts, "n": 2, "max_tokens": 12, "sampling": samp}, 1)
    go = [json.loads(x) for x in open(os.path.join(t.work_dir, "gen_1_original.jsonl"), encoding="utf-8")]
    ge = [json.loads(x) for x in open(os.path.join(t.work_dir, "gen_1_edited.jsonl"), encoding="utf-8")]
    check("generate: both models, every prompt and sample", len(go) == 6 and s["generations"] == 12 and {r["prompt"] for r in go} == {0, 1, 2}
          and all(r["n_tokens"] >= 1 for r in go), s)
    check("generate: the same weights give the same texts", s["edited"]["identical_to_original"] == 6 and [r["text"] for r in go] == [r["text"] for r in ge], s)
    check("generate: samples of one prompt differ", go[0]["text"] != go[1]["text"], go[:2])
    s2, _ = t.forward({"type": "generate", "model": "original", "prompts": prompts[::-1], "n": 2, "max_tokens": 12, "sampling": samp}, 2)
    g2 = [json.loads(x) for x in open(os.path.join(t.work_dir, "gen_2.jsonl"), encoding="utf-8")]
    by = {(prompts[r["prompt"]], r["sample"]): r["text"] for r in go}
    check("generate: a text depends on its prompt and sample, not on the batch", all(by[(prompts[::-1][r["prompt"]], r["sample"])] == r["text"] for r in g2))

    # an edit that forbids one token: no text contains it any more
    r = k.run("W = load('transformer.wte.weight')\nb = W.clone(); b[5] = -50.0 * W.mean(0) / W.mean(0).norm()\n"
              "import os\nos.makedirs('mild', exist_ok=True)\nnp.save('mild/transformer.wte.weight.npy', ((W + b) / 2).numpy())\nsave_edit('transformer.wte.weight', b)")
    check("kernel: an edit of the embedding", "saved edit of transformer.wte.weight" in r["out"], r)
    s, _ = t.forward({"type": "generate", "model": ["original", "mild", "edited"], "prompts": prompts, "n": 4, "max_tokens": 16, "sampling": samp}, 3)
    check("generate: three candidates in one pass", set(s["models"]) == {"original", "mild", "edited"} and s["edited"]["identical_to_original"] < 12, s)

    texts = ["w1 w2 w3 w4 w5 w6", "w7 w8 w9"]
    s, where = t.forward({"type": "teacher_forced", "model": "original", "texts": texts,
                          "return": {"token_logprobs": True, "topk": 5, "hidden_states": [0, 2], "capture": ["transformer.h.1.mlp.c_proj:input", "transformer.h.0"]}}, 4)
    f = os.path.join(t.work_dir, "fwd_4")
    lp, tk, tl = np.load(os.path.join(f, "token_logprobs.npy")), np.load(os.path.join(f, "topk_ids.npy")), np.load(os.path.join(f, "topk_logprobs.npy"))
    h0, h2 = np.load(os.path.join(f, "hidden_0.npy")), np.load(os.path.join(f, "hidden_2.npy"))
    act = np.load(os.path.join(f, "act_transformer.h.1.mlp.c_proj-input.npy"))
    blk = np.load(os.path.join(f, "act_transformer.h.0.npy"))
    toks = np.load(os.path.join(f, "tokens.npy"))
    check("teacher_forced: shapes, padded to the longest text", lp.shape == (2, 5) and tk.shape == (2, 6, 5) and h0.shape == (2, 6, 32)
          and act.shape == (2, 6, 128) and blk.shape == (2, 6, 32) and toks.tolist() == [[4, 5, 6, 7, 8, 9], [10, 11, 12, -1, -1, -1]],
          (lp.shape, tk.shape, h0.shape, act.shape, blk.shape, toks.tolist()))
    check("teacher_forced: padding is zero, values are not", (lp[1, 2:] == 0).all() and (lp[0] < 0).all() and (h2[1, 3:] == 0).all() and np.abs(h2[1, :3]).sum() > 0)
    # against the model run directly
    from transformers import AutoModelForCausalLM
    ref = AutoModelForCausalLM.from_pretrained(d).eval()
    with torch.no_grad():
        o = ref(torch.tensor([[4, 5, 6, 7, 8, 9]]), output_hidden_states=True)
    want = torch.log_softmax(o.logits[0], -1)
    check("teacher_forced: log-probabilities match the model run directly", np.allclose(lp[0], want[torch.arange(5), torch.tensor([5, 6, 7, 8, 9])].numpy(), atol=1e-4)
          and (tk[0, :, 0] == want.argmax(-1).numpy()).all() and np.allclose(tl[0, :, 0], want.max(-1).values.numpy(), atol=1e-4))
    check("teacher_forced: hidden states and a block's output match too", np.allclose(h2[0], o.hidden_states[2][0].numpy(), atol=2e-3)
          and np.allclose(blk[0], o.hidden_states[1][0].numpy(), atol=2e-3))
    check("teacher_forced: the summary", s["original"]["mean_token_logprob"] < 0 and "tokens.npy" in s["original"]["files"], s)
    s, _ = t.forward({"type": "teacher_forced", "model": ["original", "edited"], "texts": texts, "return": {"token_logprobs": True}}, 5)
    a = np.load(os.path.join(t.work_dir, "fwd_5", "original", "token_logprobs.npy"))
    b = np.load(os.path.join(t.work_dir, "fwd_5", "edited", "token_logprobs.npy"))
    check("teacher_forced: two models, the edit lowers the forbidden token", np.allclose(a, lp, atol=1e-5) and b[0, 0] < a[0, 0] - 1, (a[0, :2], b[0, :2]))
    k.run("np.save('steer.npy', np.ones(32, np.float32))")
    s, _ = t.forward({"type": "teacher_forced", "model": "original", "texts": texts, "return": {"token_logprobs": True, "capture": ["transformer.h.0"]},
                      "interventions": [{"site": "transformer.h.0", "op": "add", "vector": "steer.npy", "value": 2.0}]}, 6)
    st = np.load(os.path.join(t.work_dir, "fwd_6", "act_transformer.h.0.npy"))
    check("intervention on a block that returns a tuple", np.allclose(st[0].astype(np.float32), blk[0].astype(np.float32) + 2.0, atol=0.05), np.abs(st[0] - blk[0] - 2).max())
    # (a constant added to every unit would be removed by the next layer norm, so the steering direction is a ramp)
    k.run("np.save('ramp.npy', np.linspace(-1, 1, 32).astype(np.float32))")
    s, _ = t.forward({"type": "generate", "model": "original", "prompts": prompts, "max_tokens": 8, "sampling": {"temperature": 0}, "capture": ["transformer.h.1"],
                      "interventions": [{"site": "transformer.h.0", "op": "add", "vector": "ramp.npy", "value": 50.0}]}, 7)
    g7 = [json.loads(x) for x in open(os.path.join(t.work_dir, "gen_7.jsonl"), encoding="utf-8")]
    s8, _ = t.forward({"type": "generate", "model": "original", "prompts": prompts, "max_tokens": 8, "sampling": {"temperature": 0}}, 8)
    g8 = [json.loads(x) for x in open(os.path.join(t.work_dir, "gen_8.jsonl"), encoding="utf-8")]
    idx = json.load(open(os.path.join(t.work_dir, "gen_7_acts", "index.json")))
    last = np.load(os.path.join(t.work_dir, "gen_7_acts", "transformer.h.1__last_prompt.npy"))
    check("generate: steering changes the text, and activations are captured per text", [r["text"] for r in g7] != [r["text"] for r in g8]
          and last.shape == (3, 32) and len(idx["rows"]) == 3 and "transformer.h.1__mean_generated.npy" in idx["files"], idx)

    for name, req in (("prompts and chats together", {"type": "generate", "prompts": ["w1"], "chats": [[{"role": "user", "content": "w1"}]]}),
                      ("chats without a chat template", {"type": "generate", "chats": [[{"role": "user", "content": "w1"}]]}),
                      ("too many generations", {"type": "generate", "prompts": ["w1"] * 40, "n": 2}),
                      ("too many tokens", {"type": "generate", "prompts": ["w1"], "max_tokens": 10 ** 6}),
                      ("an unknown layer", {"type": "teacher_forced", "texts": ["w1 w2"], "return": {"hidden_states": [9]}}),
                      ("an unknown site", {"type": "teacher_forced", "texts": ["w1 w2"], "return": {"capture": ["transformer.h.7"]}}),
                      ("an empty list", {"type": "teacher_forced", "texts": []})):
        msg = refused(lambda: t.forward(req, 20))
        check(f"lm refuses {name}", msg is not None and not os.path.exists(os.path.join(t.work_dir, "fwd_20"))
              and not os.path.exists(os.path.join(t.work_dir, "gen_20.jsonl")), msg)
    out = os.path.join(tmp, "export_lm")
    t.export(out, full=True)
    patched = AutoModelForCausalLM.from_pretrained(os.path.join(out, "model")).eval()
    with torch.no_grad():
        lg = torch.log_softmax(patched(torch.tensor([[4, 5, 6]])).logits[0], -1)
    check("export: the patched model loads with transformers and carries the edit", float(lg[0, 5]) < float(want[0, 5]) - 1, (float(lg[0, 5]), float(want[0, 5])))
    k.close()
    t.close()


def small_pieces():
    s = summarize({"a.0.w": "(2, 2)", "a.1.w": "(2, 2)", "a.2.w": "(2, 2)", "b.w": "(3,)", "c.0.x.5.w": "(1,)", "c.1.x.5.w": "(1,)"})
    check("summarize groups names that differ in a number", s.splitlines() == ["a.{0-2}.w  (2, 2)  x3", "b.w  (3,)", "c.{0-1}.x.5.w  (1,)  x2"], s)


def main():
    tmp = tempfile.mkdtemp()
    small_pieces()
    torch_tests(tmp)
    lm_tests(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
