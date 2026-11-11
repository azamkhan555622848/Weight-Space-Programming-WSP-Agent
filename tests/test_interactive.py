"""Interactive sessions: a conversation with the agent about a network, with a scripted editor instead of a model (CPU, no account).

The harness runs in a thread, as the session's process would, and the test plays the user through the session folder: messages,
steering while it works, interrupting, approving or declining blocks, plan mode, the budget, the todo list, a change of editor,
going idle and resuming. Then the same conversation through the `wsp` commands with real background processes.

    python tests/test_interactive.py
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_workspace import make_torch_project  # noqa: E402
from wsp import cli, runs  # noqa: E402
from wsp import events as E  # noqa: E402
from wsp import messages as M  # noqa: E402
from wsp.harness import Harness, parse_blocks  # noqa: E402
from wsp.llm import ScriptedBackend  # noqa: E402
from wsp.tasks.workspace import targets  # noqa: E402
from wsp.tasks.workspace.task import WorkspaceTask  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:900]}"))


EXPL = "Checking something; this explanation is long enough to count as one.\n\n"
PY = lambda code: EXPL + "```python\n" + code + "\n```"                                        # noqa: E731
FWD = lambda req: EXPL + "```forward\n" + json.dumps(req) + "\n```"                            # noqa: E731
RUN_VAL = {"type": "run", "model": "both", "inputs": {"dataset": "val"}}
TODO = "```todo\n- [x] look at the head\n- [~] cut class 0\n- [ ] measure\n```\n"


class Session:
    """A harness in a thread, and the user's side of the session folder."""

    def __init__(self, tmp, name, script, make_backend=None, **cfg):
        proj = os.path.join(tmp, "proj")
        self.run = os.path.join(tmp, "runs", name)
        self.cfg = {"target": targets.resolve("toy", proj), "project_dir": proj, "budget": 5, "max_runs": 30, "device": "cpu",
                    "interactive": True, "todo": True, "goal": "Make class 0 never predicted.", "accept_no_sandbox": True,
                    "idle_timeout": 120, **cfg}
        self.script = script
        self.make_backend = make_backend
        os.makedirs(self.run, exist_ok=True)
        json.dump({"task": "workspace", "backend": "scripted", "model": "scripted", "effort": "high", "cfg": self.cfg},
                  open(os.path.join(self.run, "launch.json"), "w"))
        self.offset, self.seen, self.exit, self.thread = 0, [], None, None

    def start(self, script=None):
        task = WorkspaceTask(self.cfg, self.run, os.path.join(self.run, "work"), log=lambda s: None)
        self.h = Harness(task, ScriptedBackend(script if script is not None else self.script), self.run, self.cfg,
                         make_backend=self.make_backend)
        self.h.log = lambda s: None

        def go():
            try:
                self.h.run()
                self.exit = "returned"
            except SystemExit as e:
                self.exit = e.code if e.code is not None else 0
            except BaseException as e:  # noqa: BLE001
                self.exit = repr(e)
        self.exit = None
        self.thread = threading.Thread(target=go, daemon=True)
        self.thread.start()
        return self

    def events(self, type=None):
        new, self.offset = E.read_events(self.run, self.offset)
        self.seen += new
        return [e for e in self.seen if type is None or e["type"] == type]

    def wait(self, cond, timeout=60):
        t0 = time.time()
        while time.time() - t0 < timeout:
            if cond():
                return True
            time.sleep(0.05)
        return False

    def wait_events(self, type, n=1, timeout=60):
        return self.wait(lambda: len(self.events(type)) >= n, timeout)

    def status(self):
        return (runs.read_json(os.path.join(self.run, "status.json"), {}) or {}).get("activity")

    def say(self, text):
        runs.send_message(self.run, text)

    def stop(self):
        runs.request_stop(self.run)
        self.thread.join(60)
        return self.exit

    @property
    def sent(self):
        return [m for m in self.h.llm.received if m != "<new conversation>"]

    def state(self):
        return json.load(open(os.path.join(self.run, "state.json"), encoding="utf-8"))


def conversation(tmp):
    s = Session(tmp, "talk", [PY("print(SHAPES['head.bias'])"),
                              TODO + PY("b = load('head.bias'); b[0] -= 100; save_edit('head.bias', b)"),
                              FWD(RUN_VAL), "Class 0 is gone: accuracy fell as expected. What next?"]).start()
    check("the agent works until it has something to say, then waits", s.wait_events("waiting"), s.events())
    ev = s.events()
    kinds = [(e["type"], e.get("kind")) for e in ev]
    check("events: the request, each step with its result, the message", kinds == [
        ("session", None), ("user", None), ("assistant", "python"), ("tool", "python"), ("todo", None), ("assistant", "python"),
        ("tool", "python"), ("assistant", "forward"), ("tool", "forward"), ("assistant", "message"), ("waiting", None)], kinds)
    check("events: the user's request and the agent's explanation", ev[1]["text"] == "Make class 0 never predicted."
          and ev[2]["text"].startswith("Checking something") and ev[2]["block"] == "print(SHAPES['head.bias'])" and "(3,)" in ev[3]["output"])
    todo = s.events("todo")[0]["items"]
    check("todo: the list with its three states", [(i["text"], i["status"]) for i in todo] == [
        ("look at the head", "completed"), ("cut class 0", "in_progress"), ("measure", "pending")], todo)
    check("todo: the block is not shown as part of the explanation, and its step still ran", "todo" not in ev[5]["text"]
          and "saved edit of head.bias" in ev[6]["output"])
    fwd = s.events("tool")[2]
    check("events: a counted pass with its request and summary", fwd["ok"] and fwd["counted"] and fwd["n"] == 1 and fwd["budget"] == 5
          and fwd["request"] == RUN_VAL and '"accuracy"' in fwd["output"], fwd)
    check("the first message is the task with the request in it", s.sent[0].startswith("TASK.") and "Make class 0 never predicted." in s.sent[0]
          and "EXPLAIN EVERY STEP" in s.sent[0])
    info = runs.run_info(s.run)
    check("status: waiting for the user", s.status() == "waiting" and s.state()["phase"] == "waiting" and info["interactive"]
          and info["target"] == "toy" and info["title"] == "Make class 0 never predicted." and len(info["todo"]) == 3, info)
    check("the agent's message is kept as the latest answer", "What next?" in open(os.path.join(s.run, "final_answer.md")).read())

    s.h.llm.replies += [PY("print('second request')"), "Done with that too."]
    s.say("Now check class 2 is unharmed.")
    check("a message from the user wakes the agent", s.wait_events("waiting", 2), s.events())
    check("the waiting agent gets the user's words as they are", s.sent[4] == "Now check class 2 is unharmed.", s.sent[4])
    check("events: the second request appears in order", [e["text"] for e in s.events("user")] == [
        "Make class 0 never predicted.", "Now check class 2 is unharmed."])
    check("stopping ends the process; the session can be resumed", s.stop() == 0 and s.state()["phase"] == "waiting"
          and s.events("stopped")[-1]["reason"] == "requested" and runs.run_info(s.run)["state"] == "idle", s.exit)

    # resume: a new process, the kernel is fresh, and the handoff carries the conversation
    s.say("And export it.")
    s.start([PY("print('KEEP' in globals())"), "Exported."])
    check("resume: the agent answers the new message", s.wait_events("waiting", 3), s.events()[-4:])
    h = s.sent[0]
    check("resume: the handoff has the task, the todo list, later messages and the agent's last message",
          h.count("THIS IS A RESTARTED CONVERSATION") == 1 and "- [~] cut class 0" in h and "1. Now check class 2 is unharmed." in h
          and "Done with that too." in h and "Your kernel was restarted" in h and h.rstrip().endswith("And export it."), h[-900:])
    check("resume: events say so", len(s.events("resumed")) == 1 and len(s.events("handoff")) >= 1)
    s.stop()


def steering_and_interrupts(tmp):
    s = Session(tmp, "steer", ["__SLOW__"])
    os.makedirs(os.path.join(s.run, "inbox"), exist_ok=True)
    s.start()
    check("the model is asked", s.wait(lambda: s.status() == "model"))
    s.say("Use the dataset called val.")
    runs.request_interrupt(s.run)
    check("interrupt: the reply is abandoned and the agent waits", s.wait_events("interrupted") and s.wait(lambda: s.status() == "waiting"), s.events())
    st = s.state()
    check("a message sent meanwhile was picked up, and answered", s.wait_events("user", 2) and s.wait_events("waiting", 1), s.events())
    st = s.state()
    check("interrupt: the abandoned turn wasn't counted, and the flag is cleared", st["turn"] == 1 and st["phase"] == "waiting"
          and not os.path.exists(os.path.join(s.run, "interrupt")), st["turn"])
    msg = s.sent[-1]
    check("after an interrupt the agent gets what it missed, then what the user said", msg.startswith("TASK.")
          and msg.rstrip().endswith("[message from the user, who interrupted you here]\nUse the dataset called val."), msg[-300:])

    # a message while it works reaches it with the next result
    s.h.llm.replies += [PY("import time; time.sleep(3); print('slow step')"), "Finished."]
    s.say("Go on.")
    check("it is working", s.wait(lambda: s.status() == "python"), s.status())
    s.say("Also keep class 1 accurate.")
    check("it finishes", s.wait_events("waiting", 2), s.events()[-5:])
    nxt = s.sent[-1]
    check("steering: marked as sent while working, in front of the result", nxt.startswith(
        "[message from the user, sent while you were working]\nAlso keep class 1 accurate.\n\nOUTPUT of Python run 1"), nxt[:200])

    # interrupting a forward pass: it is stopped between batches and not counted
    def slow_pass(req, k):
        for i in range(400):
            s.h.task.progress(f"pass {k}: batch {i}")
            time.sleep(0.05)
        raise AssertionError("was not interrupted")
    s.h.task.tool.run = slow_pass
    s.h.llm.replies += [FWD(RUN_VAL), "Stopped as asked."]
    s.say("Measure it.")
    check("a pass is running", s.wait(lambda: s.status() == "forward"), s.status())
    runs.request_interrupt(s.run)
    check("interrupt during a pass", s.wait_events("interrupted", 2) and s.wait(lambda: s.status() == "waiting"), s.events()[-4:])
    tool = s.events("tool")[-1]
    check("the interrupted pass is not counted and leaves no files", tool["ok"] is False and tool["counted"] is False and s.state()["passes"] == 0
          and not os.path.exists(os.path.join(s.run, "work", "fwd_1")) and not os.path.exists(os.path.join(s.run, "staging")), tool)
    s.say("Fine, tell me what you have.")
    check("then it carries on", s.wait_events("waiting", 3), s.events()[-4:])
    check("it is told the pass was interrupted", s.sent[-1].startswith(M.FWD_INTERRUPTED), s.sent[-1][:200])
    s.stop()


def permissions_modes_budget(tmp):
    s = Session(tmp, "ask", [PY("print('one')"), PY("print('two')"), PY("print('three')"), FWD(RUN_VAL), "Done."],
                ask_python=True, ask_forward=True).start()
    check("ask: the agent asks before running code", s.wait_events("permission_request"), s.events())
    req = s.events("permission_request")[0]
    check("ask: the request shows the block, and nothing has run", req["kind"] == "python" and req["body"] == "print('one')"
          and s.status() == "permission" and s.state()["runs"] == 0, req)
    runs.answer_permission(s.run, req["id"], "reject", "Don't print, compute.")
    check("declined: the agent asks again with its next block", s.wait_events("permission_request", 2), s.events()[-4:])
    check("declined: the block was not run and the agent is told why", s.sent[1] == M.DECLINED.format(kind="python", feedback=" They said: Don't print, compute.")
          and s.state()["runs"] == 0, s.sent[1])
    runs.answer_permission(s.run, s.events("permission_request")[1]["id"], "once")
    check("allowed once: it runs, and the next block asks again", s.wait_events("permission_request", 3) and s.state()["runs"] == 1, s.state()["runs"])
    runs.answer_permission(s.run, s.events("permission_request")[2]["id"], "always")
    check("always: code is no longer asked about, passes still are", s.wait_events("permission_request", 4)
          and s.events("permission_request")[3]["kind"] == "forward" and s.state()["settings"]["ask_python"] is False, s.state()["settings"])
    runs.request_interrupt(s.run)
    check("interrupting while asked counts as no", s.wait_events("interrupted") and s.wait(lambda: s.status() == "waiting") and s.state()["passes"] == 0
          and [e["decision"] for e in s.events("permission")] == ["reject", "once", "always", "reject"], s.events("permission"))
    s.stop()

    s = Session(tmp, "plan", [FWD(RUN_VAL), PY("save_edit('head.bias', load('head.bias') + 1)"), "Here is my plan.",
                              FWD(RUN_VAL), FWD({**RUN_VAL, "model": "original"}), FWD({**RUN_VAL, "model": "edited"}), "Out of passes."],
                mode="plan", budget=1).start()
    check("plan mode: the agent proposes and waits", s.wait_events("waiting"), s.events())
    check("plan mode: no pass was run and no edit saved", s.sent[1] == M.PLAN_NO_FORWARD and "plan mode is on" in s.sent[2]
          and s.state()["passes"] == 0 and runs.edited_tensors(s.run) == [], s.sent[1:3])
    runs.send_setting(s.run, "mode", "build")
    s.say("Go ahead.")
    check("build mode: passes run again, until the budget is used", s.wait_events("waiting", 2), s.events()[-5:])
    check("the agent is told the mode changed", s.sent[3].startswith(M.MODE_NOTE["build"] + "\nGo ahead."), s.sent[3][:200])
    check("over budget: the pass is refused with what to do", s.state()["passes"] == 1 and s.sent[5] == M.FWD_NONE_LEFT_INTERACTIVE.format(budget=1), s.sent[5])
    runs.send_setting(s.run, "budget", 3)
    s.h.llm.replies += [FWD({**RUN_VAL, "model": "edited"}), "Now it ran."]
    s.say("Try again.")
    check("a raised budget lets it continue, and it is told", s.wait_events("waiting", 3) and s.state()["passes"] == 2
          and M.BUDGET_NOTE.format(budget=3, left=2) in s.sent[-2], s.sent[-2][:200])
    check("settings are kept for a resume", s.state()["settings"] == {"mode": "build", "budget": 3} and runs.run_info(s.run)["budget"] == 3
          and runs.run_info(s.run)["mode"] == "build", s.state()["settings"])
    s.stop()


def editor_idle_and_context(tmp):
    made = []

    def factory(model, effort):
        made.append((model, effort))
        return ScriptedBackend(["With the new editor."])
    s = Session(tmp, "editor", ["First."], make_backend=factory, editor_model="m1", editor_effort="high").start()
    s.wait_events("waiting")
    runs.send_setting(s.run, "editor_effort", "max")
    s.say("Continue.")
    check("a change of editor restarts the conversation with a handoff", s.wait_events("waiting", 2) and made == [("m1", "max")]
          and "THIS IS A RESTARTED CONVERSATION" in s.sent[0] and M.EDITOR_NOTE.format(model="m1", effort="max") in s.sent[0], (made, s.sent[0][-300:]))
    s.stop()

    s = Session(tmp, "idle", ["Anything else?"], idle_timeout=1).start()
    check("a long silence stops the process, to free the machine", s.wait(lambda: s.exit is not None, 30) and s.exit == 0
          and s.events("stopped")[-1]["reason"] == "idle" and s.state()["phase"] == "waiting", s.exit)

    s = Session(tmp, "long", [PY("print(1)"), PY("print(2)"), "Done."], handoff_tokens=50).start()
    s.wait_events("waiting")
    check("a long conversation restarts from the notebook", s.h.llm.received.count("<new conversation>") == 3 and len(s.events("handoff")) == 2,
          s.h.llm.received.count("<new conversation>"))
    s.stop()

    s = Session(tmp, "blocks", [TODO + "```mermaid\ngraph\n```", TODO + "Only a list and words.", "Bye."]).start()
    s.wait_events("waiting")
    check("a block that can't be run is sent back, with the todo list still taken", s.sent[1] == M.UNKNOWN_BLOCK.format(kind="mermaid")
          and len(s.events("todo")) == 2, s.sent[1])
    last = [e for e in s.events("assistant") if e["kind"] == "message"][0]
    check("a reply with only a todo list and words is a message, shown without the list", last["text"] == "Only a list and words.", last)
    s.stop()
    check("parse_blocks: prose before the first block", parse_blocks("why\n\n```todo\n- [ ] a\n```\n```python\nx\n```") == (
        [("todo", "- [ ] a"), ("python", "x")], "why"))
    check("parse_todo: marks, numbering and junk", E.parse_todo("- [ ] a\n* [x] b\n1. [~] c\n- [>] d\nnot an item\n- [?] e") == [
        {"text": "a", "status": "pending"}, {"text": "b", "status": "completed"}, {"text": "c", "status": "in_progress"},
        {"text": "d", "status": "in_progress"}])


def commands(tmp):
    """The same conversation through `wsp run`, `wsp send`, `wsp status`, `wsp export`, with real background processes."""
    proj = os.path.join(tmp, "proj")
    home = os.path.join(tmp, "home")
    os.makedirs(home, exist_ok=True)
    os.environ["HOME"] = os.environ["USERPROFILE"] = home
    replies = os.path.join(tmp, "replies.json")
    json.dump([PY("b = load('head.bias'); b[0] -= 100; save_edit('head.bias', b)"), FWD(RUN_VAL), "Class 0 is never predicted now.",
               PY("print('checked')"), "Class 2 is unharmed."], open(replies, "w"))
    os.environ["WSP_SCRIPTED_REPLIES"] = replies
    runs.RUNS_DIR = os.path.join(tmp, "cli_runs")
    cwd = os.getcwd()
    os.chdir(proj)
    try:
        def wsp(*args):
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                try:
                    rc = cli.main(list(args))
                except SystemExit as e:
                    rc, _ = 1, buf.write(str(e))
            return rc, buf.getvalue()
        rc, out = wsp("run", "--target", "toy", "--goal", "Make class 0 never predicted.", "--backend", "scripted", "--name", "demo",
                      "--accept-no-sandbox", "--task-arg", "device=cpu")
        d = os.path.join(runs.RUNS_DIR, "demo")
        check("wsp run: prints the conversation until the agent answers", rc == 0 and "> Make class 0 never predicted." in out
              and "[python] b = load('head.bias')" in out and "-> ok" in out and "Class 0 is never predicted now." in out
              and '"accuracy"' in out, out[-1500:] + runs.console_tail(d))
        check("the session keeps running, waiting for the user", runs.run_info(d)["state"] == "waiting", runs.run_info(d)["state"])
        rc, out = wsp("status", "demo")
        check("wsp status: the target, the request and what it's doing", "toy, torch" in out and "Request: Make class 0 never predicted." in out
              and "waiting for your message" in out and "Passes:  1 of 20" in out, out)
        rc, out = wsp("send", "demo", "Is class 2 unharmed?")
        check("wsp send: prints only what is new", rc == 0 and "> Is class 2 unharmed?" in out and "Class 2 is unharmed." in out
              and "Class 0 is never predicted now." not in out, out)
        rc, out = wsp("export", "demo", "--full")
        check("wsp export: the edit and a patched model", rc == 0 and "1 edited tensors" in out
              and os.path.exists(os.path.join(d, "export", "edit.safetensors")) and os.path.exists(os.path.join(d, "export", "model", "weights.safetensors")), out)
        rc, out = wsp("stop", "demo")
        check("wsp stop: the process ends", cli_wait(lambda: not runs.is_running(d)) and runs.run_info(d)["state"] == "idle")
        rc, out = wsp("send", "demo", "One more thing.", "--accept-no-sandbox")
        check("wsp send to an idle session restarts it", rc == 0 and "I am done." in out and runs.run_info(d)["state"] == "waiting", out[-600:])
        wsp("stop", "demo", "--now")
        rc, out = wsp("run", "--target", "nope.pt", "--goal", "x", "--backend", "scripted")
        check("wsp run: an unknown target is explained", rc == 1 and "Can't find 'nope.pt'" in out, out)
        rc, out = wsp("run", "--target", "toy", "--backend", "scripted")
        check("wsp run: a session needs a goal", rc == 1 and "--goal" in out, out)
    finally:
        os.chdir(cwd)


def cli_wait(cond, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.2)
    return False


def main():
    tmp = tempfile.mkdtemp()
    make_torch_project(os.path.join(tmp, "proj"))
    conversation(tmp)
    steering_and_interrupts(tmp)
    permissions_modes_budget(tmp)
    editor_idle_and_context(tmp)
    commands(tmp)
    shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
