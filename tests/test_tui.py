"""The chat interface, run headless.

First against session folders written by the test (no process, no model): what each kind of event looks like, the todo list, a
reply as it streams, queued messages, approving a block, interrupting, the settings under the prompt, the commands, and that
text from the model can't be taken for interface markup. Then a whole conversation with a real background session whose editor
is a script: pick a network, say what to do, watch it work, answer it.

    python tests/test_tui.py
"""
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
try:
    from textual.widgets import Markdown, OptionList, Static
except ImportError:
    print('SKIP: Textual is not installed (pip install -e ".[tui]")')
    sys.exit(0)
from test_interactive import FWD, PY, RUN_VAL, TODO  # noqa: E402
from test_workspace import make_torch_project  # noqa: E402
from wsp import consent, runs  # noqa: E402
from wsp import events as E  # noqa: E402
from wsp.tui import app as tui  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:900]}"))


def shown(obj) -> str:
    """What a rich object (a table, a text) looks like as plain text."""
    from rich.console import Console
    c = Console(width=200, record=True, file=open(os.devnull, "w", encoding="utf-8"))
    c.print(obj)
    return c.export_text()


def text_of(widget) -> str:
    r = widget.render()
    return getattr(r, "plain", str(r))


def fake_session(root, name, events, state=None, interactive=True, status=None):
    """A session folder as a running session would leave it."""
    d = os.path.join(root, name)
    os.makedirs(os.path.join(d, "work", "edited"), exist_ok=True)
    cfg = {"budget": 20, "max_runs": 300, "interactive": interactive, "goal": "Make class 0 never predicted.",
           "target": {"kind": "torch", "name": "toy"}}
    json.dump({"task": "workspace" if interactive else "edge_backdoor", "backend": "anthropic", "model": "claude-opus-5-5",
               "effort": "high", "cfg": cfg if interactive else {"budget": 10, "max_runs": 100}}, open(os.path.join(d, "launch.json"), "w"))
    st = {"done": False, "turn": 3, "runs": 2, "passes": 1, "budget": 20, "sandbox": "none", "phase": "waiting", "interactive": interactive,
          "goal": cfg["goal"], "settings": {}, "todo": [], "tokens": {"input": 10, "output": 5000, "cache_read": 100000, "cache_write": 20000}}
    st.update(state or {})
    json.dump(st, open(os.path.join(d, "state.json"), "w"))
    log = E.EventLog(d)
    for ev in events:
        log.emit(**ev)
    if status:
        json.dump({"activity": status, "detail": "pass 2 of 20", "since": time.time() - 75, "pid": os.getpid()},
                  open(os.path.join(d, "status.json"), "w"))
    open(os.path.join(d, "work", "notes.md"), "w").write("# Notebook\nhead.bias[0] carries class 0.\n")
    open(os.path.join(d, "work", "edited", "head.bias.npy"), "wb").write(b"x" * 140)
    return d, log


EVENTS = [
    {"type": "session", "task": "workspace", "interactive": True, "budget": 20},
    {"type": "user", "text": "Make class 0 never predicted."},
    {"type": "assistant", "turn": 1, "kind": "python", "text": "I'll look at the [bold red]head[/] first.",
     "block": "# where does class 0 come from?\nprint(load('head.bias'))"},
    {"type": "tool", "turn": 1, "kind": "python", "n": 1, "seconds": 0.2, "ok": True, "status": "ok", "output": "tensor([0.1, [red]0.2[/red], 0.3])"},
    {"type": "todo", "items": [{"text": "look at the head", "status": "completed"}, {"text": "cut class 0", "status": "in_progress"},
                               {"text": "measure", "status": "pending"}]},
    {"type": "assistant", "turn": 2, "kind": "forward", "text": "Now I measure it.", "block": json.dumps(RUN_VAL)},
    {"type": "tool", "turn": 2, "kind": "forward", "n": 1, "budget": 20, "seconds": 1.5, "ok": True, "counted": True, "request": RUN_VAL,
     "output": '{"edited": {"accuracy": 0.74}}'},
    {"type": "assistant", "turn": 3, "kind": "forward", "text": "Again.", "block": json.dumps(RUN_VAL)},
    {"type": "tool", "turn": 3, "kind": "forward", "ok": False, "counted": False, "error": "Identical to pass 1 with unchanged edits; not run."},
    {"type": "note", "text": "The reply had no explanation before its block, so it was sent back unrun."},
    {"type": "assistant", "turn": 4, "kind": "message", "text": "Class 0 is **gone**.\n\n| set | accuracy |\n|---|---|\n| val | 74% |"},
    {"type": "waiting"},
]


async def expect(pilot, name, cond, detail=lambda: "", timeout=30):
    """The interface handles events in its own time: wait for the expected state rather than assume when it arrives."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            if cond():
                break
        except Exception:  # noqa: BLE001 - a widget that isn't there yet
            pass
        await pilot.pause(0.05)
    try:
        good = bool(cond())
    except Exception as e:  # noqa: BLE001
        good, detail = False, (lambda e=e: repr(e))
    check(name, good, detail())


GOT: list = []


def inbox(d) -> list:
    """Everything sent to the session since the last clear_inbox(). Taking is destructive, so what was taken is kept."""
    GOT.extend(E.take(d))
    return GOT


def clear_inbox(d) -> None:
    E.take(d)
    GOT.clear()


def transcript_text(app) -> str:
    out = []
    for w in app.transcript.walk_children():
        if isinstance(w, tui.ToolCall):
            out.append(str(w.title))
        elif isinstance(w, Markdown):
            out.append(getattr(w, "source", "") or "")
        elif isinstance(w, Static) and not isinstance(w.parent, Markdown) and w.parent is app.transcript:
            out.append(text_of(w))
    return "\n".join(out)


async def watching(root):
    d, log = fake_session(root, "talk", EVENTS)
    app = tui.WspApp(runs_dir=root, project_dir=root, session="talk", gpu_poll=False)
    async with app.run_test(size=(150, 46)) as pilot:
        await expect(pilot, "opening a session shows its conversation", lambda: "Class 0 is **gone**" in transcript_text(app), lambda: transcript_text(app))
        tx = transcript_text(app)
        check("the user's request and the agent's words are there", "Make class 0 never predicted." in tx and "Now I measure it." in tx, tx)
        tools = list(app.transcript.query(tui.ToolCall))
        check("each step is a row: code runs and passes, with their outcome", [str(t.title) for t in tools] == [
            "✓ Python · run 1 · # where does class 0 come from? · 0.2 s", "✓ Forward pass 1/20 · run · both · 1.5 s",
            "✗ Forward pass not run · run · both"], [str(t.title) for t in tools])
        check("steps start closed, and a refused pass is marked", all(t.collapsed for t in tools) and tools[2].has_class("failed"))
        check("a step holds its code and its result", "tensor([0.1, [red]0.2[/red], 0.3])" in text_of(tools[0].out)
              and "Identical to pass 1" in text_of(tools[2].out), text_of(tools[0].out))
        md = [getattr(w, "source", "") for w in app.transcript.query(Markdown)]
        check("the model's text is shown literally, not as interface markup", "I'll look at the [bold red]head[/] first." in md, md)
        await pilot.press("ctrl+o")
        await expect(pilot, "ctrl+o opens every step", lambda: all(not t.collapsed for t in tools))
        await pilot.press("ctrl+o")
        await expect(pilot, "and closes them again", lambda: all(t.collapsed for t in tools))
        todo = text_of(app.todo)
        check("the todo list sits above the prompt with its count", app.todo.display and todo.startswith("1 of 3 todos completed")
              and "☑ look at the head" in todo and "◉ cut class 0" in todo and "☐ measure" in todo, todo)
        await expect(pilot, "the title and the side panel describe the session", lambda: "Make class 0 never predicted." in text_of(app.query_one("#title"))
                     and all(x in text_of(app.query_one("#side")) for x in ("toy", "1/20", "2/300", "about $", "head.bias", "notes.md", "EDITS (1)")),
                     lambda: text_of(app.query_one("#side")))
        await expect(pilot, "idle: it says so", lambda: "Idle" in text_of(app.query_one("#working")), lambda: text_of(app.query_one("#working")))

        # the session's process is alive and working: status, streaming, queueing, interrupting
        lock = runs.RunLock(d)
        check("test setup: the lock is held", lock.acquire())
        json.dump({"activity": "model", "detail": "anthropic", "since": time.time() - 75, "pid": os.getpid()}, open(os.path.join(d, "status.json"), "w"))
        open(os.path.join(d, "live.txt"), "w", encoding="utf-8").write("The bias of class 0 is the [i]lever[/i]:\n\n```python\nb = load(")
        await expect(pilot, "working: what it's doing, for how long, and how to interrupt", lambda: all(
            x in text_of(app.query_one("#working")) for x in ("Thinking", "75 s", "esc to interrupt")), lambda: text_of(app.query_one("#working")))
        await expect(pilot, "a reply shows while it arrives", lambda: "the [i]lever[/i]" in app.query_one("#live", Markdown).source, lambda: app.live_text)
        await pilot.press(*"also keep class 1")
        await pilot.press("enter")
        await expect(pilot, "a message typed while it works is queued", lambda: E.queued_messages(d) == ["also keep class 1"]
                     and "queued: also keep class 1" in text_of(app.query_one("#queued")), lambda: E.queued_messages(d))
        check("the prompt is cleared and remembers", app.prompt.text == "" and app.prompt.sent[-1] == "also keep class 1")
        await pilot.press("up")
        await expect(pilot, "up brings back the last message", lambda: app.prompt.text == "also keep class 1", lambda: app.prompt.text)
        await pilot.press("down")
        await expect(pilot, "down returns to the draft", lambda: app.prompt.text == "", lambda: app.prompt.text)
        await pilot.press("escape")
        await expect(pilot, "esc asks the agent to stop and wait", lambda: os.path.exists(os.path.join(d, "interrupt")))
        log.emit("assistant", turn=5, kind="python", text="The bias is the lever.", block="b = load('head.bias')")
        await expect(pilot, "when the reply is complete it replaces the stream", lambda: not app.query("#live")
                     and "The bias is the lever." in transcript_text(app) and len(app.transcript.query(tui.ToolCall)) == 4, lambda: transcript_text(app)[-300:])
        check("a step still running says so", str(list(app.transcript.query(tui.ToolCall))[-1].title).startswith("◌ Python"))

        # being asked before a block runs
        json.dump({"activity": "permission", "detail": "python", "since": time.time(), "pid": os.getpid()}, open(os.path.join(d, "status.json"), "w"))
        log.emit("permission_request", id="p5", kind="python", body="b = load('head.bias')")
        await expect(pilot, "a permission request opens a question", lambda: isinstance(app.screen, tui.PermissionScreen) and bool(app.screen.query("#once")),
                     lambda: app.screen)
        await expect(pilot, "the question shows the code, with Run it in focus", lambda: app.screen.body == "b = load('head.bias')"
                     and app.focused is not None and app.focused.id == "once", lambda: app.focused)
        clear_inbox(d)
        app.screen.query_one("#feedback").value = "use the weight instead"
        app.screen.query_one("#reject").press()  # not a mouse click: a passing notification can cover a button
        await expect(pilot, "the answer goes to the session", lambda: [(i["type"], i["id"], i["decision"], i["feedback"]) for i in inbox(d)] == [
            ("permission", "p5", "reject", "use the weight instead")] and len(app.screen_stack) == 1, lambda: GOT)
        clear_inbox(d)
        log.emit("permission", id="p5", decision="reject")
        log.emit("permission_request", id="p6", kind="forward", body=json.dumps(RUN_VAL))
        await expect(pilot, "the next request asks again", lambda: isinstance(app.screen, tui.PermissionScreen) and app.screen.kind == "forward"
                     and bool(app.screen.query("#always")))
        app.screen.query_one("#always").press()  # not a mouse click: a passing notification can cover a button
        await expect(pilot, "always is passed on", lambda: [(i["id"], i["decision"]) for i in inbox(d)] == [("p6", "always")]
                     and len(app.screen_stack) == 1, lambda: GOT)
        clear_inbox(d)
        log.emit("permission", id="p6", decision="always")

        # settings under the prompt reach the session
        await pilot.press("tab")
        await expect(pilot, "tab switches to plan mode", lambda: [(i["key"], i["value"]) for i in inbox(d)] == [("mode", "plan")]
                     and "Plan" in text_of(app.query_one("#chip-mode")), lambda: (GOT, text_of(app.query_one("#chip-mode"))))
        clear_inbox(d)
        await pilot.press("shift+tab")
        await expect(pilot, "shift+tab steps through the permissions", lambda: [(i["key"], i["value"]) for i in inbox(d)] == [
            ("ask_python", False), ("ask_forward", True)] and "ask: passes" in text_of(app.query_one("#chip-perm")), lambda: (GOT, text_of(app.query_one("#chip-perm"))))
        clear_inbox(d)
        app.run_command("/budget 30")
        app.run_command("/effort max")
        app.run_command("/compact")
        await expect(pilot, "/budget, /effort and /compact reach the session", lambda: [(i["type"], i.get("key"), i.get("value")) for i in inbox(d)] == [
            ("set", "budget", 30), ("set", "editor_effort", "max"), ("compact", None, None)], lambda: GOT)
        clear_inbox(d)
        app.run_command("/budget lots")
        app.run_command("/nonsense")
        await pilot.pause(0.2)
        check("a bad command sends nothing", inbox(d) == [], GOT)
        app.run_command("/stop")
        await expect(pilot, "/stop asks the process to stop", lambda: os.path.exists(os.path.join(d, "stop")))
        lock.release()

        # windows with information
        app.run_command("/notes")
        await expect(pilot, "/notes shows the agent's notebook", lambda: isinstance(app.screen, tui.InfoScreen)
                     and "head.bias[0] carries class 0." in text_of(app.screen.query_one("#info-text")))
        await pilot.press("escape")
        app.run_command("/status")
        await expect(pilot, "/status shows the session", lambda: isinstance(app.screen, tui.InfoScreen) and "Passes:  1 of 20" in text_of(app.screen.query_one("#info-text")))
        await pilot.press("escape")
        app.run_command("/help")
        await expect(pilot, "/help lists the commands", lambda: isinstance(app.screen, tui.InfoScreen))
        await pilot.press("escape")
        await expect(pilot, "esc closes a window", lambda: len(app.screen_stack) == 1)

        # the command list while typing
        await pilot.press("/")
        ol = app.query_one("#suggest", OptionList)
        await expect(pilot, "typing / lists the commands", lambda: ol.display and ol.option_count == len(tui.COMMANDS))
        await pilot.press(*"bu")
        await expect(pilot, "typing narrows the list", lambda: [c[0] for c in app.suggestions] == ["budget"], lambda: app.suggestions)
        await pilot.press("enter")
        await expect(pilot, "enter on a command that needs a value fills it in", lambda: app.prompt.text == "/budget " and not ol.display, lambda: app.prompt.text)
        app.prompt.set_text("")
        await pilot.press(*"/to")
        await pilot.press("enter")
        await expect(pilot, "enter on a command without one runs it", lambda: app.prompt.text == "" and not app.todo.display, lambda: app.prompt.text)
        app.run_command("/todo")
        await expect(pilot, "/todo shows the list again", lambda: app.todo.display)

        # other sessions
        app.run_command("/sessions")
        await expect(pilot, "/sessions lists the sessions", lambda: isinstance(app.screen, tui.PickerScreen) and len(app.screen.options) == 2, lambda: app.screen)
        app.screen.dismiss(os.path.join(root, "bench"))
        await expect(pilot, "a benchmark run from before the event log is shown from its records", lambda: "I removed the backdoor" in transcript_text(app)
                     and "find the trigger" in transcript_text(app), lambda: transcript_text(app))
        await pilot.press(*"try stage 2")
        await pilot.press("enter")
        await expect(pilot, "in a benchmark run a message becomes an experimenter's note", lambda: len(os.listdir(os.path.join(root, "bench", "notices"))) == 1)
        app.run_command("/new")
        await expect(pilot, "/new returns to the start", lambda: app.session is None and bool(app.query("#welcome"))
                     and "Recent sessions" in text_of(app.query_one("#welcome")), lambda: transcript_text(app)[:200])
        await pilot.press("ctrl+q")


def legacy_run(root):
    """A run folder from before the event log: explanations and the audit log only."""
    d = os.path.join(root, "bench")
    os.makedirs(os.path.join(d, "notices"))
    json.dump({"task": "edge_backdoor", "backend": "anthropic", "model": "claude-opus-5-5", "effort": "high", "cfg": {"budget": 10, "max_runs": 100}},
              open(os.path.join(d, "launch.json"), "w"))
    json.dump({"done": False, "turn": 41, "runs": 29, "passes": 10, "budget": 10, "tokens": {}}, open(os.path.join(d, "state.json"), "w"))
    with open(os.path.join(d, "explanations.md"), "w", encoding="utf-8") as f:
        f.write("## Turn 40 (python; passes used 9/10)\n\nThe trigger sits in stage 1.\n\n## Turn 41 (final answer; passes used 10/10)\n\nI removed the backdoor.\n\n")
    with open(os.path.join(d, "tool_calls.jsonl"), "w", encoding="utf-8") as f:
        f.write(json.dumps({"kind": "python", "run": 29, "turn": 40, "s": 0.1, "code": "# find the trigger\nprint(1)", "out": "1\n"}) + "\n")


async def conversation(tmp):
    """Start to finish with a real background session; its editor replays a script."""
    proj, root = os.path.join(tmp, "proj"), os.path.join(tmp, "live_runs")
    replies = os.path.join(tmp, "replies.json")
    json.dump([TODO + PY("b = load('head.bias'); b[0] -= 100; save_edit('head.bias', b)"), FWD(RUN_VAL), "Class 0 is never predicted now.",
               PY("print('checked class 2')"), "Class 2 is unharmed."], open(replies, "w"))
    os.environ["WSP_SCRIPTED_REPLIES"] = replies
    app = tui.WspApp(runs_dir=root, project_dir=proj, gpu_poll=False)
    app.backend, app.model = "scripted", "scripted"
    real_policy = tui.sandbox.resolve_policy
    async with app.run_test(size=(150, 46)) as pilot:
        await expect(pilot, "the start screen offers what it found in the folder", lambda: "toy" in text_of(app.query_one("#welcome"))
                     and "ckpt.pt" in text_of(app.query_one("#welcome")), lambda: text_of(app.query_one("#welcome")))
        await pilot.press(*"Make class 0 never predicted.")
        await pilot.press("enter")
        await expect(pilot, "without a target, sending asks which network", lambda: isinstance(app.screen, tui.PickerScreen)
                     and any(v == "toy" for _, v in app.screen.options), lambda: app.screen)
        app.screen.dismiss("nope.pt")
        await expect(pilot, "a wrong path is explained and nothing starts", lambda: len(app.screen_stack) == 1 and app.target is None and app.session is None)
        app.run_command("/target qwen_math")
        await pilot.pause(0.3)
        check("a benchmark isn't accepted as the network to talk about", app.target is None and app.session is None, app.target)
        # words instead of a path: the interface searches Hugging Face (here, a stand-in for it)
        import huggingface_hub
        real_list = huggingface_hub.HfApi.list_models
        asked = []

        def fake_list(self, **kw):
            asked.append(kw)
            return [NS(id="org/tiny-lm", downloads=1234), NS(id="org/other", downloads=5)]
        huggingface_hub.HfApi.list_models = fake_list
        try:
            app.run_command("/target tiny qwen 0.5b")
            await expect(pilot, "words that aren't a path search Hugging Face", lambda: isinstance(app.screen, tui.PickerScreen)
                         and [v for _, v in app.screen.options] == ["org/tiny-lm", "org/other"] and asked[0]["search"] == "tiny qwen 0.5b",
                         lambda: (app.screen, asked))
            app.screen.dismiss("org/tiny-lm")
            await expect(pilot, "a model picked from the search becomes the target", lambda: app.target == {
                "kind": "hf-lm", "name": "tiny-lm", "path": "org/tiny-lm", "hub": True}, lambda: app.target)
            app.target = None
            app.run_command("/search other")
            await expect(pilot, "/search does the same", lambda: isinstance(app.screen, tui.PickerScreen) and len(asked) == 2)
            await pilot.press("escape")
        finally:
            huggingface_hub.HfApi.list_models = real_list
        app.target = None
        app.run_command("/target toy")
        await expect(pilot, "/target sets it", lambda: app.target and app.target["name"] == "toy" and "toy" in text_of(app.query_one("#side")))
        app.run_command("/permissions passes")
        app.prompt.set_text("Make class 0 never predicted.")
        await pilot.press("enter")
        if real_policy().kind == "none":
            await expect(pilot, "no sandbox here: the notice comes first", lambda: isinstance(app.screen, tui.NoticeScreen) and bool(app.screen.query("#accept")))
            await expect(pilot, "the safe choice has the focus, so Enter can't accept", lambda: app.focused is not None and app.focused.id == "cancel")
            app.screen.query_one("#cancel").press()  # not a mouse click: a passing notification can cover a button
            await expect(pilot, "declining starts nothing", lambda: len(app.screen_stack) == 1 and app.session is None and not consent.accepted(consent.NO_SANDBOX))
            app.prompt.set_text("Make class 0 never predicted.")
            await pilot.press("enter")
            await expect(pilot, "the notice again", lambda: isinstance(app.screen, tui.NoticeScreen) and bool(app.screen.query("#accept")))
            app.screen.query_one("#accept").press()  # not a mouse click: a passing notification can cover a button
        await expect(pilot, "the session starts and the request shows at once", lambda: app.session is not None and "Make class 0 never predicted." in transcript_text(app),
                     lambda: transcript_text(app))
        d = app.session
        await expect(pilot, "the agent works: its first step appears", lambda: any("save_edit" in t.block for t in app.transcript.query(tui.ToolCall)),
                     lambda: transcript_text(app) + runs.console_tail(d), timeout=180)
        await expect(pilot, "it asks before the forward pass, as set", lambda: isinstance(app.screen, tui.PermissionScreen) and app.screen.kind == "forward"
                     and bool(app.screen.query("#once")), lambda: transcript_text(app) + runs.console_tail(d), timeout=120)
        app.screen.query_one("#once").press()  # not a mouse click: a passing notification can cover a button
        await expect(pilot, "it reports, and it is the user's turn", lambda: "Class 0 is never predicted now." in transcript_text(app)
                     and "Your turn" in text_of(app.query_one("#working")), lambda: transcript_text(app) + runs.console_tail(d), timeout=120)
        tx = transcript_text(app)
        check("the request appears once, and the steps carry their results", tx.count("Make class 0 never predicted.") == 1
              and "✓ Python · run 1" in tx and "✓ Forward pass 1/20 · run · both" in tx, tx)
        check("the todo list from the agent", text_of(app.todo).startswith("1 of 3 todos completed"), text_of(app.todo))
        await expect(pilot, "the side panel counts the pass and lists the edit", lambda: "1/20" in text_of(app.query_one("#side"))
                     and "head.bias" in text_of(app.query_one("#side")), lambda: text_of(app.query_one("#side")))
        await pilot.press(*"Is class 2 unharmed?")
        await pilot.press("enter")
        await expect(pilot, "answering continues the conversation", lambda: "Class 2 is unharmed." in transcript_text(app)
                     and "Is class 2 unharmed?" in transcript_text(app), lambda: transcript_text(app), timeout=120)
        app.run_command("/diff")
        await expect(pilot, "/diff shows what the edit changes", lambda: isinstance(app.screen, tui.InfoScreen)
                     and "head.bias" in shown(app.screen.body) and "1 of 3" in shown(app.screen.body), lambda: shown(app.screen.body), timeout=120)
        await pilot.press("escape")
        app.run_command("/export full")
        await expect(pilot, "/export writes the edit and the patched network", lambda: os.path.exists(os.path.join(d, "export", "model", "weights.safetensors")),
                     timeout=120)
        await pilot.press("escape")

        # a session that can't start: the reason is shown, nothing is left behind, and the request is ready to send again
        app.run_command("/new")
        await expect(pilot, "/new, to start another", lambda: app.session is None)
        app.target = {"kind": "torch", "name": "broken", "builder": os.path.join(proj, "missing.py") + ":build", "kwargs": {}, "datasets": {}}
        before = set(os.listdir(root))
        app.prompt.set_text("Do something with it.")
        await pilot.press("enter")
        await expect(pilot, "a session that can't start says why", lambda: "The session didn't start" in transcript_text(app)
                     and "missing.py" in transcript_text(app) and "Nothing was started" in transcript_text(app), lambda: transcript_text(app), timeout=120)
        check("it leaves no session behind, and the request is back in the prompt", app.session is None and set(os.listdir(root)) == before
              and app.prompt.text == "Do something with it.", (app.session, set(os.listdir(root)) - before, app.prompt.text))
        app.target = None
        app.run_command("/target toy")
        app.prompt.set_text("")
        await pilot.press("ctrl+q")
    check("closing the interface leaves the session running", runs.is_running(d))
    again = tui.WspApp(runs_dir=root, project_dir=proj, session=os.path.basename(d), gpu_poll=False)
    async with again.run_test(size=(150, 46)) as pilot:
        await expect(pilot, "opening it again shows the whole conversation", lambda: "Class 2 is unharmed." in transcript_text(again)
                     and "✓ Forward pass 1/20" in transcript_text(again), lambda: transcript_text(again))
        again.run_command("/stop")
        await expect(pilot, "/stop ends its process", lambda: not runs.is_running(d), timeout=60)
        await pilot.press("ctrl+q")


def main():
    tmp = tempfile.mkdtemp()
    home = os.path.join(tmp, "home")
    os.makedirs(home)
    os.environ["HOME"] = os.environ["USERPROFILE"] = home          # consent and preferences go to a temporary home
    root = os.path.join(tmp, "fake_runs")
    os.makedirs(root)
    legacy_run(root)
    make_torch_project(os.path.join(tmp, "proj"))
    try:
        asyncio.run(watching(root))
        asyncio.run(conversation(tmp))
    finally:
        for r in runs.list_runs(os.path.join(tmp, "live_runs")):
            runs.force_stop(r["path"])
        shutil.rmtree(tmp, ignore_errors=True)
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
