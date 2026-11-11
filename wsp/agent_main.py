"""The process that runs one session: `python -m wsp.agent_main --run FOLDER ...`. The `wsp` commands and the terminal interface
start it in the background and talk to it through the session folder (wsp/events.py).

A session keeps the settings it was created with (launch.json), so running this again with only --run resumes it. What the user
changed while it ran (budget, mode, editor) is in state.json and is applied on top.

    # a conversation about any network: a Hugging Face model folder or id, or a checkpoint with the code that builds the network
    python -m wsp.agent_main --run runs/demo --task workspace --target models/Qwen3.5-0.8B --goal "Find out why it loops, and fix it"
    # one of the two fixed benchmark tasks
    python -m wsp.agent_main --run runs/edge_b10 --task edge_backdoor --budget 10

Backends, with an API key: anthropic (ANTHROPIC_API_KEY) | openai (OPENAI_API_KEY). With your own subscription, through the
provider's CLI with all tools off: claude-cli (`claude -p`) | codex-cli (`codex exec`).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

from . import consent
from .env import load_dotenv
from .harness import Harness
from .llm import API_BACKENDS, DEFAULT_MODELS, SUBSCRIPTION_BACKENDS, BackendNotReady, make_backend

TASKS = ("workspace", "edge_backdoor", "qwen_math")
DEFAULT_BUDGET = {"workspace": 20, "edge_backdoor": 10, "qwen_math": 10}
DEFAULT_RUNS = {"workspace": 300, "edge_backdoor": 100, "qwen_math": 300}
BACKENDS = API_BACKENDS + SUBSCRIPTION_BACKENDS + ("scripted",)


def make_task(name: str, cfg: dict, run_dir: str, work_dir: str, log=print):
    if name == "edge_backdoor":
        from .tasks.edge.task import EdgeBackdoorTask
        return EdgeBackdoorTask(cfg, run_dir, work_dir)
    if name == "qwen_math":
        from .tasks.qwen_math.task import QwenMathTask
        return QwenMathTask(cfg, run_dir, work_dir)
    if name == "workspace":
        from .tasks.workspace.task import WorkspaceTask
        return WorkspaceTask(cfg, run_dir, work_dir, log=log)
    raise SystemExit(f"unknown task {name!r}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m wsp.agent_main", description="Run one WSP session.")
    ap.add_argument("--run", required=True, help="the session's folder (created; reused to resume)")
    ap.add_argument("--task", choices=TASKS, help="needed to start a session")
    ap.add_argument("--target", help="workspace: a model folder, a weights file (optionally followed by file.py:function), a "
                    "Hugging Face id, or a name from the project's wsp.json")
    ap.add_argument("--target-json", help="workspace: the target as JSON (see wsp/tasks/workspace/targets.py)")
    ap.add_argument("--goal", help="workspace: what the user wants, in their words")
    ap.add_argument("--project", help="workspace: the folder paths are relative to (default: the current folder)")
    ap.add_argument("--budget", type=int, help="forward passes")
    ap.add_argument("--max-runs", type=int, help="python runs")
    ap.add_argument("--max-turns", type=int, default=400)
    ap.add_argument("--handoff-every", type=int, default=40)
    ap.add_argument("--backend", default="anthropic", choices=BACKENDS)
    ap.add_argument("--model", help="the editor model (default: " + ", ".join(f"{k}: {v}" for k, v in DEFAULT_MODELS.items()) + ")")
    ap.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    ap.add_argument("--mode", default="build", choices=["build", "plan"], help="workspace: plan = study and propose, no edits or passes")
    ap.add_argument("--ask", action="append", default=[], choices=["python", "forward"],
                    help="workspace: ask the user before running this kind of block (repeatable)")
    ap.add_argument("--no-explain", action="store_true", help="do not require an explanation before each block")
    ap.add_argument("--task-arg", action="append", default=[], help="task option key=value (e.g. model=control)")
    ap.add_argument("--accept-no-sandbox", action="store_true",
                    help="run the model's code without a sandbox if this system has none (see the README's Security)")
    ap.add_argument("--accept-subscription-notice", action="store_true",
                    help="accept the notice about subscription modes for this run (see the README's Models)")
    return ap


def new_launch(a) -> dict:
    """The settings of a new session, from the command line."""
    if not a.task:
        raise SystemExit("--task is needed to start a session (this folder has none to resume).")
    cfg = {"budget": a.budget or DEFAULT_BUDGET[a.task], "max_runs": a.max_runs or DEFAULT_RUNS[a.task], "max_turns": a.max_turns,
           "handoff_every": a.handoff_every, "explain": not a.no_explain}
    for kv in a.task_arg:
        k, v = kv.split("=", 1)
        try:
            cfg[k] = json.loads(v)
        except ValueError:
            cfg[k] = v
    if a.task == "workspace":
        from .tasks.workspace import targets
        project = os.path.abspath(a.project or os.getcwd())
        try:
            spec = targets.normalise(json.loads(a.target_json), project) if a.target_json else targets.resolve(a.target or "", project)
        except targets.TargetError as e:
            raise SystemExit(str(e)) from None
        if spec["kind"] == "benchmark":
            raise SystemExit(f"{spec['task']} is a fixed benchmark: start it with --task {spec['task']}.")
        if not (a.goal and a.goal.strip()):
            raise SystemExit("A session starts from what you want done: --goal \"...\".")
        cfg.update(target=spec, project_dir=project, goal=a.goal.strip(), interactive=True, todo=True, handoff_tokens=200_000,
                   mode=a.mode, ask_python="python" in a.ask, ask_forward="forward" in a.ask)
    return {"task": a.task, "backend": a.backend, "model": a.model or DEFAULT_MODELS.get(a.backend, "scripted"), "effort": a.effort,
            "cfg": cfg, "created": round(time.time())}


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    run_dir = os.path.abspath(a.run)
    work_dir = os.path.join(run_dir, "work")
    launch_path = os.path.join(run_dir, "launch.json")
    if os.path.exists(launch_path):
        launch = json.load(open(launch_path, encoding="utf-8"))
        if a.task and launch["task"] != a.task:
            raise SystemExit(f"{a.run} is a {launch['task']} session, not {a.task}. Use another folder for a new one.")
        print(f"resuming with this run's settings: {launch['backend']}, {launch['model']}, {launch['effort']}", flush=True)
    else:
        launch = new_launch(a)
        os.makedirs(work_dir, exist_ok=True)
        json.dump(launch, open(launch_path, "w", encoding="utf-8"), indent=1)
    cfg = dict(launch["cfg"])
    # what the user changed while the session ran
    settings = {}
    if os.path.exists(os.path.join(run_dir, "state.json")):
        try:
            settings = json.load(open(os.path.join(run_dir, "state.json"), encoding="utf-8")).get("settings") or {}
        except ValueError:
            settings = {}
    cfg.update({k: settings[k] for k in ("budget", "mode", "ask_python", "ask_forward") if k in settings})
    model = settings.get("editor_model") or launch["model"]
    effort = settings.get("editor_effort") or launch["effort"]
    cfg["editor_model"], cfg["editor_effort"] = model, effort

    if launch["backend"] in SUBSCRIPTION_BACKENDS:
        if not consent.confirm(consent.SUBSCRIPTION, consent.SUBSCRIPTION_TEXT, a.accept_subscription_notice):
            raise SystemExit("Subscription mode needs its notice accepted first: run in a terminal, or add "
                             "--accept-subscription-notice after reading it in the README.")
    elif launch["backend"] in API_BACKENDS:
        load_dotenv()      # keys are only loaded for the backends that use them

    def log(s):
        print(s, flush=True)

    def backend_for(m, e):
        return make_backend(launch["backend"], run_dir, m, e, log=log)
    try:
        task = make_task(launch["task"], cfg, run_dir, work_dir, log=log)
    except FileNotFoundError as e:
        raise SystemExit(f"{launch['task']}: {e.filename or e} is missing." + (f" Run `wsp setup --task {launch['task']}` first."
                                                                             if launch["task"] != "workspace" else "")) from None
    except (ValueError, TypeError, ImportError, AttributeError, OSError) as e:
        if launch["task"] != "workspace":
            raise
        raise SystemExit(f"Can't load the target: {type(e).__name__}: {e}") from None
    try:
        backend = backend_for(model, effort)
    except BackendNotReady as e:
        raise SystemExit(f"{launch['backend']}: {e}") from None
    h = Harness(task, backend, run_dir, cfg, make_backend=backend_for)
    h.cfg["accept_no_sandbox"] = a.accept_no_sandbox or (
        not h.kernel.sandboxed and consent.confirm(consent.NO_SANDBOX, consent.NO_SANDBOX_TEXT))
    st = h.run()
    print(json.dumps({k: st.get(k) for k in ("turn", "runs", "passes", "done", "tool_free", "tokens", "submission")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
