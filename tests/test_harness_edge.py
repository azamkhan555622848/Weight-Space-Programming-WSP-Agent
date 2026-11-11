"""End-to-end harness test on a tiny synthetic edge task (random ResNet-8, 200 random practice images), CPU only, with
a scripted editor instead of a model: send-back, kernel, edits, paired pass, duplicate and invalid requests, nudge,
scheduled handoff (kernel kept), outage + harness restart (restart note, handoff, fresh kernel), refusal, budget end,
final answer, submission. Then the safety stops: no sandbox without acceptance, machine not ready, a reply not confirmed
free of tools, repeated failed passes, and passes killed part-way.
    python tests/test_harness_edge.py"""
import json
import os
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.harness import Harness  # noqa: E402
from wsp.llm import Reply, ScriptedBackend  # noqa: E402
from wsp.sandbox import Policy  # noqa: E402
from wsp.tasks.base import TaskNotReady  # noqa: E402
from wsp.tasks.edge import resnet8 as R  # noqa: E402
from wsp.tasks.edge.task import EdgeBackdoorTask  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:600]}"))


def make_data(d):
    import torch
    torch.manual_seed(0)
    os.makedirs(os.path.join(d, "victim"))
    os.makedirs(os.path.join(d, "practice"))
    np.savez(os.path.join(d, "victim", "weights.npz"), **R.ResNet8BN().eval().folded())
    rng = np.random.default_rng(0)
    np.save(os.path.join(d, "practice", "images.npy"), rng.integers(0, 256, (200, 32, 32, 3), dtype=np.uint8))
    np.save(os.path.join(d, "practice", "labels.npy"), rng.integers(0, 10, 200))


E = "I want to check something; this explanation is long enough to count as one.\n\n"
PY = lambda code: E + "```python\n" + code + "\n```"
FWD = lambda req: E + "```forward\n" + json.dumps(req) + "\n```"
REQ1 = {"type": "classify", "model": "both", "practice": "all", "activations": ["pool"]}


def state_of(run):
    return json.load(open(os.path.join(run, "state.json"), encoding="utf-8"))


def stops(fn, *exc):
    """Run fn; -> the text of the exception it stopped with, or None if it ran to the end."""
    try:
        fn()
    except (SystemExit, *exc) as e:
        return str(e) or type(e).__name__
    return None


def safety_tests(tmp, data):
    base = {"budget": 2, "max_runs": 20, "handoff_every": 40, "explain": True, "data_dir": data, "device": "cpu",
            "accept_no_sandbox": True}

    def new(name, script, **over):
        run, cfg = os.path.join(tmp, name), {**base, **over}
        return run, Harness(EdgeBackdoorTask(cfg, run, os.path.join(run, "work")), ScriptedBackend(script), run, cfg)

    # no sandbox, not accepted: refuses before any model call (set here, since Linux and macOS have one)
    run, h = new("no_consent", [FWD(REQ1)], accept_no_sandbox=False)
    h.kernel.policy = Policy("none", "No sandbox in this test.")
    msg = stops(h.run)
    check("no sandbox and not accepted: refuses", msg and "without a sandbox" in msg and not h.llm.received, msg)

    # a sandbox that fails its self-test: refuses, unless running without one was explicitly accepted
    class Broken:
        kind = "bwrap"

    run, h = new("sandbox_broken", [FWD(REQ1)], accept_no_sandbox=False)
    h.kernel.policy, h.kernel.self_test = Broken, lambda: "it can't start here."
    msg = stops(h.run)
    check("a failing sandbox: refuses, and says why", msg and "failed its self-test" in msg and "can't start here" in msg
          and not h.llm.received, msg)
    run, h = new("sandbox_accepted", ["Final answer."], budget=0)
    h.kernel.policy, h.kernel.self_test = Broken, lambda: "it can't start here."
    st = h.run()
    check("a failing sandbox: accepted, it runs without one, and that's recorded", st["done"] and st["sandbox"] == "none", st.get("sandbox"))

    # the machine can't run the task: refuses before any model call
    run, h = new("not_ready", [FWD(REQ1)])

    def not_ready():
        raise TaskNotReady("no GPU here")
    h.task.preflight = not_ready
    msg = stops(h.run)
    check("machine not ready: refuses", msg and "no GPU here" in msg and not h.llm.received, msg)

    # a reply using a tool: nothing in it runs, and the run can't be resumed
    run, h = new("tool_stop", ["__TOOL__"])
    msg = stops(h.run)
    st = state_of(run)
    check("tool use stops the run", msg and "free of tools" in msg, msg)
    check("tool use: nothing was run", st["runs"] == 0 and st["tool_free"] is False and st["halted"], st)
    run, h = new("tool_stop", ["I am done."])
    msg = stops(h.run)
    check("tool use: the run refuses to resume", msg and "can't be resumed" in msg and not h.llm.received, msg)

    # a backend that says nothing about tools fails closed
    class Silent(ScriptedBackend):
        def send(self, text, on_delta=None):
            self.received.append(text)
            return Reply("Explained.\n\n```python\nprint('this must not run')\n```", model="silent")
    run, h = new("silent", [])
    h.llm = Silent([])
    msg = stops(h.run)
    check("no confirmation of no tools: stops", msg and "free of tools" in msg and state_of(run)["runs"] == 0, msg)

    # passes that keep failing stop the paid turns, and the run can resume once the problem is fixed
    one = lambda i: FWD({"type": "classify", "model": "edited", "practice": [i]})
    run, h = new("fail_cap", [one(1), one(2), one(3)], budget=1, max_failed_passes=2)

    def broken(req, k):
        raise RuntimeError("the GPU is gone")
    h.task.run_pass = broken
    msg = stops(h.run)
    st = state_of(run)
    check("repeated failed passes stop the run", msg and "2 forward passes failed in a row" in msg, msg)
    check("failed passes: not counted, 2 model calls, resumable", st["passes"] == 0 and h.llm.i == 2
          and not st.get("halted"), st)
    run, h = new("fail_cap", [one(4), "Final answer."], budget=1, max_failed_passes=2)
    st = h.run()
    check("failed passes: resumes after the fix", st["done"] and st["passes"] == 1, st)

    # the provider lost the conversation: the run continues in a new one, with a handoff, and the same message
    run, h = new("lost", [PY("print('kept going')"), "__LOST__", "Final answer."], budget=0)
    st = h.run()
    sent = [m for m in h.llm.received if m != "<new conversation>"]
    check("a lost conversation continues with a handoff", st["done"] and "THIS IS A RESTARTED CONVERSATION" in sent[2]
          and "kept going" in sent[2] and h.llm.received.count("<new conversation>") == 2, sent[-1][-300:])
    run, h = new("lost_forever", ["__LOST__"] * 6, budget=0)
    msg = stops(h.run)
    check("a conversation that keeps getting lost stops the run", msg and "simulated lost session" in msg and h.llm.i == 4, (msg, h.llm.i))

    # killed during a pass: nothing usable is left and the pass isn't counted
    run, h = new("killed", [FWD(REQ1)])

    def killed(req, k):
        os.makedirs(os.path.join(h.task._out, f"fwd_{k}"))
        open(os.path.join(h.task._out, f"fwd_{k}", "partial.npy"), "w").close()
        raise KeyboardInterrupt
    h.task.run_pass = killed
    stops(h.run, KeyboardInterrupt)
    check("killed during a pass: no files, not counted", not os.path.exists(os.path.join(run, "work", "fwd_1"))
          and state_of(run)["passes"] == 0, state_of(run))

    # killed after the pass was counted, before its files moved: they appear on resume, and the editor is told
    run, h = new("killed2", [FWD(REQ1)], budget=1)

    def die():
        raise KeyboardInterrupt
    h.task.stage_commit = die
    stops(h.run, KeyboardInterrupt)
    st = state_of(run)
    check("killed before the commit: counted, files not there yet", st["passes"] == 1 and st["commit_pending"]
          and not os.path.exists(os.path.join(run, "work", "fwd_1")), st)
    run, h = new("killed2", ["Final answer."], budget=1)
    st = h.run()
    sent = [m for m in h.llm.received if m != "<new conversation>"]
    check("killed before the commit: files arrive on resume", os.path.exists(os.path.join(run, "work", "fwd_1", "pred_original.npy"))
          and not st["commit_pending"] and "Forward pass 1 of 1 done" in sent[0], st)


def main():
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "data")
    make_data(data)
    run = os.path.join(tmp, "run")
    cfg = {"budget": 2, "max_runs": 20, "handoff_every": 6, "explain": True, "data_dir": data, "device": "cpu",
           "accept_no_sandbox": True}
    script1 = [
        "```python\nprint('no explanation')\n```",                                            # t1 sent back
        PY("w = load('fc.b')\nw[9] -= 50.0\nsave_edit('fc.b', w)\nKEEP = 123\nedited()"),     # t2 edit
        FWD(REQ1),                                                                          # t3 pass 1
        FWD(REQ1),                                                                          # t4 duplicate
        E + "```forward\n{not json\n```",                                                   # t5 invalid
        "Thinking about it, no block this time.",                                           # t6 nudge
        PY("print(KEEP)\nprint(open('notes.md').read()[:20])"),                             # t7 (after handoff)
        "__FAIL__",                                                                         # t8 outage -> stop
    ]
    h = Harness(EdgeBackdoorTask(cfg, run, os.path.join(run, "work")), ScriptedBackend(script1), run, cfg)
    try:
        h.run()
        check("outage stops the run", False)
    except SystemExit as e:
        check("outage stops the run", "simulated outage" in str(e))
    rcv = h.llm.received
    sent = [m for m in rcv if m != "<new conversation>"]
    check("t1 task prompt has EXPLAIN paragraph before RULES",
          sent[0].startswith("TASK.") and sent[0].index("EXPLAIN EVERY STEP") < sent[0].index("\n\nRULES."))
    check("t2 unexplained block sent back", sent[1].startswith("Your reply had no explanation"), sent[1][:100])
    check("t3 python output", sent[2].startswith("OUTPUT of Python run 1 of at most 20:") and "saved edit of fc.b" in sent[2]
          and "['fc.b']" in sent[2], sent[2])
    check("t4 pass done + edits line", sent[3].startswith("Forward pass 1 of 2 done (classify); results in fwd_1/.")
          and "Forward passes left: 1." in sent[3] and "Edits used in this pass: fc.b (edit hash " in sent[3], sent[3][-300:])
    summ = json.loads(sent[3].split("```\n")[1].split("\n```")[0])
    check("paired summary", summ["comparison"]["predictions_changed"] > 0 and "practice_accuracy" in summ["edited"], summ)
    check("activation file", np.load(os.path.join(run, "work", "fwd_1", "pool_original.npy")).shape == (200, 64))
    check("t5 duplicate not counted", sent[4].startswith("The forward request is identical to forward pass 1"), sent[4][:120])
    check("t6 invalid JSON not counted", sent[5].startswith("The forward request is not valid (JSONDecodeError"), sent[5][:120])
    check("t7 nudge + scheduled handoff", "<new conversation>" in rcv[rcv.index(sent[5]) + 1:] and
          "THIS IS A RESTARTED CONVERSATION" in sent[6] and "Your kernel is still running" in sent[6] and
          "Your reply had no ```python or ```forward block. Forward passes left: 1 of 2" in sent[6], sent[6][-600:])
    check("kernel variables survive the handoff", "123" in sent[7], sent[7][:200])
    st = json.load(open(os.path.join(run, "state.json")))
    check("state after outage", st["turn"] == 7 and st["passes"] == 1 and st["runs"] == 2 and not st["done"], st)

    # ---- harness restart: new process, same run folder
    script2 = [
        PY("print('KEEP' in globals())"),                                                   # fresh kernel
        "__REFUSE__",                                                                       # provider refusal
        FWD({"type": "classify", "model": "edited", "practice": list(range(50))}),          # pass 2 = last
        PY("print('still have python')"),
        "The backdoor was X; I changed fc.b[3]. Final answer.",                             # final
    ]
    h2 = Harness(EdgeBackdoorTask(cfg, run, os.path.join(run, "work")), ScriptedBackend(script2), run, cfg)
    st = h2.run()
    sent2 = [m for m in h2.llm.received if m != "<new conversation>"]
    check("restart: handoff with restart note and fresh-kernel text",
          sent2[0].count("THIS IS A RESTARTED CONVERSATION") == 1 and "Your kernel was restarted" in sent2[0]
          and "Last result:\n[note: the harness was restarted, so your kernel restarted too" in sent2[0], sent2[0][-700:])
    check("restart: the undelivered message is re-sent", "Last result:" in sent2[0] and "OUTPUT of Python run 2" in sent2[0],
          sent2[0][-400:])
    check("fresh kernel after restart", "False" in sent2[1], sent2[1][:200])
    check("refusal replacement", sent2[2].startswith("[The output of your last step could not be delivered"), sent2[2][:100])
    check("last pass message", "Forward pass 2 of 2 done" in sent2[3] and sent2[3].rstrip().endswith(
        "then reply without a code block."), sent2[3][-200:])
    check("python still allowed after budget", sent2[4].startswith("OUTPUT of Python run 4 of at most 20"), sent2[4][:100])
    check("done + submission", st["done"] and st["submission"]["edited_tensors"] == ["fc.b"]
          and os.path.exists(os.path.join(run, "submission", "weights.npz")), st.get("submission"))
    check("explanations.md", "## Turn 1 (python; passes used 0/2)\n\n(no explanation)" in
          open(os.path.join(run, "explanations.md"), encoding="utf-8").read())
    check("turn folders", all(os.path.exists(os.path.join(run, "turns", f"t{i:03d}", f)) for i in range(1, st["turn"] + 1)
                              for f in ("prompt.txt", "answer.md", "run.json")))
    check("counters", st["passes"] == 2 and st["runs"] == 4 and st["handoffs"] == 2 and st["restarts"] == 1
          and st["flags"] == 1 and st["unexplained"] == 1 and st["nudges"] == 1, {k: st[k] for k in
          ("passes", "runs", "handoffs", "restarts", "flags", "unexplained", "nudges", "turn")})
    safety_tests(tmp, data)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
