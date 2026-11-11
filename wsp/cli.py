"""The `wsp` command. With no arguments it opens the chat interface (`pip install -e ".[tui]"`) in the current folder.

    wsp                                         the chat interface: pick a network, say what you want, watch and steer
    wsp run --target PATH --goal "..."          the same conversation from the command line; prints until the agent answers
    wsp send NAME "message"                     continue it
    wsp export NAME [--full]                    write the current edit (with --full, a complete patched model)
    wsp status [NAME]                           all sessions, or one in detail
    wsp stop NAME [--now]                       stop a session's process at the next safe point, or at once
    wsp resume NAME                             carry on a stopped session
    wsp setup --task edge_backdoor              fetch and build what a benchmark needs
    wsp run --task edge_backdoor                run one of the two benchmarks in the background
    wsp eval NAME                               score a finished benchmark run
    wsp doctor                                  what works on this machine, and how to fix what doesn't

PATH is a Hugging Face model folder or id, or a weights file followed by the code that builds the network (ckpt.pt model.py:build).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

from . import consent, prices, runs, sandbox
from .llm import API_BACKENDS, DEFAULT_MODELS, SUBSCRIPTION_BACKENDS

ACTIVITY = {"model": "waiting for the model", "python": "running the model's code", "forward": "running a forward pass",
            "waiting": "waiting for your message", "permission": "waiting for your approval", "finishing": "writing the result"}
MARK = {"ok": "OK  ", "warn": "NOTE", "fix": "FIX ", "info": "--  "}
BACKENDS = API_BACKENDS + SUBSCRIPTION_BACKENDS + ("scripted",)


def ago(seconds: float) -> str:
    s = int(seconds)
    return f"{s} s" if s < 90 else f"{s // 60} min" if s < 5400 else f"{s / 3600:.1f} h"


def status_line(r: dict) -> str:
    """What a session is doing, in words."""
    if r["state"] in ("running", "waiting"):
        what = ACTIVITY.get(r["activity"], "starting")
        if r["detail"] and r["activity"] in ("python", "forward"):
            what += f" ({r['detail']})"
        if r["since"] and r["activity"] != "waiting":
            what += f" for {ago(time.time() - r['since'])}"
        return ("stopping after this step; " if r["stopping"] else "") + what
    if r["state"] == "halted":
        return f"stopped for good: {r['halted']}"
    if r["state"] == "stopped":
        return "stopped; `wsp resume` continues it"
    if r["state"] == "idle":
        return "idle; send a message to continue"
    return r["state"]


def _resolve(name: str) -> str:
    d = runs.find_run(name)
    if not d:
        raise SystemExit(f"No session named {name!r}. `wsp status` lists them.")
    return d


# ------------------------------------------------------------------------------------------------ following a session
def print_event(ev: dict) -> None:
    t = ev.get("type")
    if t == "user":
        print(f"\n> {ev.get('text', '')}\n")
    elif t == "assistant":
        if ev.get("text"):
            print(ev["text"])
        if ev.get("block"):
            first = next((ln for ln in ev["block"].splitlines() if ln.strip()), "")
            print(f"  [{ev.get('kind')}] {first[:110]}")
    elif t == "tool":
        head = "ok" if ev.get("ok") else "not run" if not ev.get("counted", True) and ev.get("kind") == "forward" else "error"
        secs = f", {ev['seconds']} s" if ev.get("seconds") is not None else ""
        print(f"  -> {head}{secs}")
        body = ev.get("output") or ev.get("error") or ""
        if ev.get("kind") == "forward" and ev.get("ok"):        # the summary on a few lines instead of one value per line
            try:
                flat = json.dumps(json.loads(body), separators=(", ", ": "))
                body = "\n".join(flat[i:i + 150] for i in range(0, min(len(flat), 1500), 150))
            except ValueError:
                pass
        lines = body.splitlines()
        for ln in lines[:10]:
            print(f"     {ln[:150]}")
        if len(lines) > 10:
            print(f"     ... {len(lines) - 10} more lines")
    elif t == "todo":
        for i in ev.get("items", []):
            print(f"  {'[x]' if i['status'] == 'completed' else '[~]' if i['status'] == 'in_progress' else '[ ]'} {i['text']}")
    elif t in ("note", "error"):
        print(f"  ({t}) {ev.get('text', '')}")
    elif t == "permission_request":
        print(f"  The agent asks to run a {ev.get('kind')} block. Answer in the interface: `wsp`.")
    elif t == "halt":
        print(f"\nStopped: {ev.get('reason')}")
    elif t == "interrupted":
        print("  (interrupted)")
    sys.stdout.flush()


def follow(run_dir: str, offset: int = 0) -> int:
    """Print what happens until the agent answers or the session stops. -> the offset reached."""
    quiet_since = None
    while True:
        evs, offset = runs.session_events(run_dir, offset)
        done = False
        for ev in evs:
            print_event(ev)
            done = done or ev.get("type") in ("waiting", "interrupted", "stopped", "halt", "done", "error")
        if done or offset == -1:
            return offset
        if evs or runs.is_running(run_dir):
            quiet_since = None
        else:                                   # the process is gone: give its last lines a moment to land, then say why
            quiet_since = quiet_since or time.time()
            if time.time() - quiet_since > 1.5:
                tail = runs.console_tail(run_dir, 600).strip()
                if tail:
                    print(tail.splitlines()[-1])
                return offset
        time.sleep(0.3)


# ------------------------------------------------------------------------------------------------ commands
def cmd_setup(a) -> int:
    from .setup_task import SIZES, setup
    tasks = runs.TASKS if a.task == "all" else [a.task]
    for t in tasks:
        print(f"{t}: downloads {SIZES[t]}")
    if not a.yes:
        try:
            if input("Continue? [y/N] ").strip().lower() not in ("y", "yes"):
                return 1
        except EOFError:
            print("No terminal to ask on; add --yes to go ahead.")
            return 1
    for t in tasks:
        setup(t)
    print("Done. `wsp doctor` shows what's ready.")
    return 0


def cmd_doctor(a) -> int:
    from .doctor import can_run, run_checks
    checks = run_checks(self_test=not a.no_self_test)
    area = None
    for c in checks:
        if c.area != area:
            area = c.area
            print(f"\n{area}")
        print(f"  {MARK[c.status]} {c.name}: {c.detail}")
        if c.fix:
            print(f"       {c.fix}")
    print()
    return 0 if can_run(checks) else 1


def collect_consents(backend: str, a) -> tuple[bool, bool]:
    """Ask here, in the terminal: the background process has none. -> (accept_no_sandbox, accept_subscription)"""
    no_sandbox = a.accept_no_sandbox
    if sandbox.resolve_policy().kind == "none":
        no_sandbox = consent.confirm(consent.NO_SANDBOX, consent.NO_SANDBOX_TEXT, a.accept_no_sandbox)
        if not no_sandbox:
            raise SystemExit("Not started: this system has no sandbox for the model's code. Accept that in a terminal or with "
                             "--accept-no-sandbox, or see `wsp doctor`.")
    subscription = a.accept_subscription_notice
    if backend in SUBSCRIPTION_BACKENDS:
        subscription = consent.confirm(consent.SUBSCRIPTION, consent.SUBSCRIPTION_TEXT, a.accept_subscription_notice)
        if not subscription:
            raise SystemExit("Not started: subscription mode needs its notice accepted, in a terminal or with "
                             "--accept-subscription-notice.")
    return no_sandbox, subscription


def cmd_run(a) -> int:
    if a.target:
        return run_session(a)
    if not a.task:
        raise SystemExit('Say what to work on: --target PATH --goal "..." for any network, or --task for a benchmark.')
    model = a.model or DEFAULT_MODELS.get(a.backend)
    name = a.name or runs.new_run_name(a.task)
    run_dir = os.path.join(runs.RUNS_DIR, name)
    if os.path.exists(os.path.join(run_dir, "launch.json")):
        raise SystemExit(f"A run named {name} already exists. `wsp resume {name}` continues it.")
    no_sandbox, subscription = collect_consents(a.backend, a)
    args = runs.run_args(run_dir, a.task, a.backend, a.model, a.effort, a.budget or 10, a.task_arg, no_sandbox, subscription)
    if a.foreground:
        return subprocess.run([sys.executable, *args], cwd=runs.ROOT).returncode
    est = prices.estimate(a.task, a.budget or 10, model)
    print(f"Starting {name}: {a.task}, {a.budget or 10} passes, {model} through {a.backend}.")
    if est["minutes"]:
        cost = f", about ${est['usd']:.0f} of API credit" if est["usd"] is not None and a.backend in API_BACKENDS else ""
        print(f"Going by our own runs: about {ago(est['minutes'] * 60)} on an RTX 4070 SUPER{cost}. Your run may differ.")
    started, msg = runs.start(run_dir, args)
    print(msg if started else f"It didn't start:\n{msg}")
    if started:
        print(f"`wsp status {name}` shows progress; `wsp stop {name}` stops it. It keeps running if you close this terminal.")
    return 0 if started else 1


def run_session(a) -> int:
    """A conversation about any network, from the command line."""
    from .tasks.workspace import targets
    project = os.getcwd()
    try:
        spec = targets.resolve(a.target, project)
    except targets.TargetError as e:
        raise SystemExit(str(e)) from None
    if spec["kind"] == "benchmark":
        raise SystemExit(f"{spec['task']} is a fixed benchmark: wsp run --task {spec['task']}")
    if not (a.goal and a.goal.strip()):
        raise SystemExit('Say what you want done: --goal "..."')
    name = a.name or runs.new_run_name("workspace")
    run_dir = os.path.join(runs.RUNS_DIR, name)
    if os.path.exists(os.path.join(run_dir, "launch.json")):
        raise SystemExit(f"A session named {name} already exists. `wsp send {name} \"...\"` continues it.")
    no_sandbox, subscription = collect_consents(a.backend, a)
    args = runs.session_args(run_dir, spec, a.goal.strip(), a.backend, a.model, a.effort, a.budget, project, mode=a.mode,
                             accept_no_sandbox=no_sandbox, accept_subscription=subscription, task_args=a.task_arg)
    print(f"Session {name}: {spec['name']} ({spec['kind']}), editor {a.model or DEFAULT_MODELS.get(a.backend, a.backend)} through {a.backend}.")
    started, msg = runs.start(run_dir, args)
    if not started:
        print(f"It didn't start:\n{msg}")
        return 1
    if a.detach:
        print(f"Running in the background. `wsp` shows it; `wsp send {name} \"...\"` talks to it.")
        return 0
    follow(run_dir)
    print(f"\n(`wsp send {name} \"...\"` to answer; `wsp export {name}` writes the edit; `wsp` opens the interface.)")
    return 0


def cmd_send(a) -> int:
    run_dir = _resolve(a.run)
    info = runs.run_info(run_dir)
    if info["state"] in ("finished", "halted"):
        raise SystemExit(f"{a.run} is {info['state']}; it can't take messages.")
    offset = os.path.getsize(os.path.join(run_dir, "events.jsonl")) if os.path.exists(os.path.join(run_dir, "events.jsonl")) else 0
    if info["interactive"]:
        runs.send_message(run_dir, a.text)
    else:
        runs.send_note(run_dir, a.text)
    if not runs.is_running(run_dir):
        no_sandbox, subscription = collect_consents(info["backend"], a)
        started, msg = runs.start(run_dir, runs.resume_args(run_dir, no_sandbox, subscription))
        if not started:
            print(f"It didn't start:\n{msg}")
            return 1
    if a.detach or not info["interactive"]:
        print("Sent.")
        return 0
    follow(run_dir, offset)
    return 0


def cmd_export(a) -> int:
    run_dir = _resolve(a.run)
    try:
        info = runs.export_session(run_dir, a.out, a.full)
    except ValueError as e:
        raise SystemExit(str(e)) from None
    if not info["edited_tensors"]:
        print("Nothing is edited yet.")
        return 0
    print(f"{len(info['edited_tensors'])} edited tensors, {info['changed_entries_total']:,} entries changed, written to {info['folder']}"
          + (" (model/ is the complete patched network)" if a.full else ""))
    return 0


def cmd_resume(a) -> int:
    run_dir = _resolve(a.run)
    if runs.is_running(run_dir):
        raise SystemExit(f"{a.run} is already running.")
    info = runs.run_info(run_dir)
    if info["state"] == "finished":
        raise SystemExit(f"{a.run} has finished.")
    if info["state"] == "halted":
        raise SystemExit(f"{a.run} can't be resumed: {info['halted']}")
    no_sandbox, subscription = collect_consents(info["backend"], a)
    started, msg = runs.start(run_dir, runs.resume_args(run_dir, no_sandbox, subscription))
    print(msg if started else f"It didn't start:\n{msg}")
    return 0 if started else 1


def describe(r: dict) -> str:
    what = f"{r['target']}, {r['target_kind']}" if r["interactive"] else r["task"]
    lines = [f"{r['name']}  ({what}; editor {r['model']} through {r['backend']}, {r['effort']} effort)"]
    if r["title"]:
        lines.append(f"  Request: {r['title']}")
    lines += [f"  Status:  {status_line(r)}",
              f"  Passes:  {r['passes']} of {r['budget']}    Python runs: {r['runs']} of {r['max_runs']}    Turns: {r['turn']}"]
    t = r["tokens"]
    if t:
        used = f"{t.get('output', 0):,} written, {t.get('input', 0) + t.get('cache_read', 0) + t.get('cache_write', 0):,} read"
        cost = f"    Estimated cost: ${r['cost_usd']:.2f}" if r["cost_usd"] is not None else ""
        lines.append(f"  Tokens:  {used}{cost}")
    if r["sandbox"]:
        lines.append(f"  Sandbox: {'none (accepted by you)' if r['sandbox'] == 'none' else r['sandbox']}")
    if r["submission"]:
        lines.append(f"  Result:  {r['path']}{os.sep}final_answer.md; `wsp eval {r['name']}` benchmarks it")
    return "\n".join(lines)


def cmd_status(a) -> int:
    if a.run:
        print(describe(runs.run_info(_resolve(a.run))))
        return 0
    rows = runs.list_runs()
    if not rows:
        print('No sessions here yet. `wsp` opens the interface; `wsp run --target PATH --goal "..."` starts one from the command line.')
        return 0
    for r in rows:
        print(f"{r['name']:<28} {r['passes']:>2}/{r['budget']:<3} {status_line(r)}")
    return 0


def cmd_stop(a) -> int:
    run_dir = _resolve(a.run)
    if not runs.is_running(run_dir):
        print(f"{a.run} isn't running.")
        return 0
    if a.now:
        done = runs.force_stop(run_dir)
        print("Stopped. `wsp resume` continues it; a pass in progress is lost and not counted." if done else "Couldn't stop it.")
        return 0 if done else 1
    runs.request_stop(run_dir)
    print("It will stop at the next safe point (a running pass finishes first). `--now` stops it at once.")
    return 0


def cmd_note(a) -> int:
    run_dir = _resolve(a.run)
    if runs.run_info(run_dir)["interactive"]:
        runs.send_message(run_dir, a.text)
    else:
        runs.send_note(run_dir, a.text)
    print("The agent gets this with its next message.")
    return 0


def cmd_eval(a) -> int:
    run_dir = _resolve(a.run)
    info = runs.run_info(run_dir)
    if info["task"] not in runs.TASKS:
        raise SystemExit("Only benchmark runs have a hidden test to score. For a conversation, `wsp export` writes the edit.")
    if not info["submission"]:
        raise SystemExit(f"{a.run} has no submission yet: a run submits when it finishes.")
    cmds = runs.eval_args(run_dir, info["task"])
    if info["task"] == "edge_backdoor" or a.foreground:
        for c in cmds:
            rc = subprocess.run([sys.executable, *c], cwd=runs.ROOT).returncode
            if rc:
                return rc
        return 0
    log = os.path.join(run_dir, "eval.log")
    runs._spawn(cmds[0], log, cwd=runs.ROOT)
    print(f"Benchmarking in the background; this takes hours on one GPU. Log: {log}. Resume by running this again; results "
          "appear in the interface (/results) and results/math/summary.json.")
    return 0


def cmd_tui(a) -> int:
    try:
        from .tui.app import run_app
    except ImportError as e:
        if "textual" not in str(e):
            raise
        print('The interface needs Textual: pip install -e ".[tui]". Everything is also available as commands: wsp --help')
        return 1
    run_app(session=getattr(a, "session", None))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="wsp", description="An LLM programs neural networks by editing their weights directly.")
    sub = ap.add_subparsers(dest="cmd")

    def consents(p):
        p.add_argument("--accept-no-sandbox", action="store_true", help="accept running the model's code without a sandbox, if there is none")
        p.add_argument("--accept-subscription-notice", action="store_true", help="accept the notice about subscription modes")

    p = sub.add_parser("run", help="start a session: a conversation about any network, or a benchmark")
    p.add_argument("--target", help="the network: a Hugging Face model folder or id, a weights file (optionally followed by "
                   "file.py:function that builds the network), or a name from wsp.json")
    p.add_argument("--goal", help="what you want done, in your words")
    p.add_argument("--task", choices=runs.TASKS, help="instead of --target: one of the two benchmarks")
    p.add_argument("--backend", choices=BACKENDS, default="anthropic",
                   help="anthropic, openai: your API key. claude-cli, codex-cli: your subscription, through the provider's own CLI")
    p.add_argument("--model", default=None, help="the editor model")
    p.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"], default="high")
    p.add_argument("--budget", type=int, default=None, help="forward passes (default: 20 for a conversation, 10 for a benchmark)")
    p.add_argument("--mode", choices=["build", "plan"], default="build", help="plan: study and propose only")
    p.add_argument("--name", default=None, help="the session's folder name")
    p.add_argument("--task-arg", action="append", default=[], help="task option key=value, e.g. model=control or device=cpu")
    p.add_argument("--detach", action="store_true", help="a conversation: don't wait for the agent's answer")
    p.add_argument("--foreground", action="store_true", help="a benchmark: stay attached to this terminal")
    consents(p)
    p.set_defaults(fn=cmd_run)

    p = sub.add_parser("send", help="say something to a session's agent, and print what it does")
    p.add_argument("run")
    p.add_argument("text")
    p.add_argument("--detach", action="store_true", help="don't wait for the answer")
    consents(p)
    p.set_defaults(fn=cmd_send)

    p = sub.add_parser("export", help="write a conversation's current edit")
    p.add_argument("run")
    p.add_argument("--full", action="store_true", help="also write the complete network with the edit applied")
    p.add_argument("--out", default=None, help="the folder to write to (default: export/ in the session folder)")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("status", help="all sessions, or one in detail")
    p.add_argument("run", nargs="?")
    p.set_defaults(fn=cmd_status)

    p = sub.add_parser("stop", help="stop a session's process at the next safe point")
    p.add_argument("run")
    p.add_argument("--now", action="store_true", help="stop at once; a pass in progress is lost and not counted")
    p.set_defaults(fn=cmd_stop)

    p = sub.add_parser("resume", help="carry on a stopped session, with its settings")
    p.add_argument("run")
    consents(p)
    p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("note", help="leave a note for the agent without waiting")
    p.add_argument("run")
    p.add_argument("text")
    p.set_defaults(fn=cmd_note)

    p = sub.add_parser("setup", help="fetch and build what a benchmark needs")
    p.add_argument("--task", choices=[*runs.TASKS, "all"], default="edge_backdoor")
    p.add_argument("--yes", action="store_true", help="don't ask before downloading")
    p.set_defaults(fn=cmd_setup)

    p = sub.add_parser("eval", help="score a finished benchmark run on its hidden test")
    p.add_argument("run")
    p.add_argument("--foreground", action="store_true")
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("doctor", help="what works on this machine, and how to fix what doesn't")
    p.add_argument("--no-self-test", action="store_true", help="skip starting the sandbox")
    p.set_defaults(fn=cmd_doctor)

    p = sub.add_parser("tui", help="the chat interface")
    p.add_argument("--session", default=None, help="open this session")
    p.set_defaults(fn=cmd_tui)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    if not getattr(a, "fn", None):
        return cmd_tui(a)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
