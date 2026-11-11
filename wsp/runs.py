"""Session folders: starting, stopping and reading them. Used by the `wsp` commands and the terminal interface.

A session belongs to its own background process (wsp/agent_main.py), which keeps everything in the session folder. Nothing here
holds a file open for long, because on Windows an open reader makes the harness's atomic save fail. Closing the interface
therefore leaves a session going, and the same commands work from another terminal or over SSH.

Sessions live in runs/ when `wsp` is started in this project's own checkout, and in .wsp/runs/ of the current folder anywhere
else; WSP_RUNS_DIR overrides both.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import time

from . import events as E
from . import prices

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def default_runs_dir() -> str:
    if os.environ.get("WSP_RUNS_DIR"):
        return os.path.abspath(os.environ["WSP_RUNS_DIR"])
    cwd = os.getcwd()
    if os.path.normcase(os.path.realpath(cwd)) == os.path.normcase(os.path.realpath(ROOT)):
        return os.path.join(ROOT, "runs")
    return os.path.join(cwd, ".wsp", "runs")


RUNS_DIR = default_runs_dir()
TASKS = ("edge_backdoor", "qwen_math")                  # the fixed benchmarks; "workspace" is the general task
SHORT_NAME = {"edge_backdoor": "edge", "qwen_math": "qwen", "workspace": "session"}


# ------------------------------------------------------------------------------------------------ files
def read_json(path: str, default=None):
    """A JSON file, or `default` if it isn't there or stays unreadable (a writer may be replacing it)."""
    for _ in range(3):
        try:
            with open(path, encoding="utf-8") as f:
                return json.load(f)
        except FileNotFoundError:
            return default
        except (ValueError, OSError):
            time.sleep(0.05)
    return default


def atomic_write_json(path: str, obj, indent: int = 1) -> None:
    """Replace a JSON file in one step. On Windows the replace fails while another process has the file open, so a reader
    can only delay it, not crash the run."""
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent)
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                raise
            time.sleep(0.25)


# ------------------------------------------------------------------------------------------------ one run at a time
class RunLock:
    """Held by the process running a session. The system releases it when that process ends, however it ends, so it can't go
    stale the way a saved process id can."""

    def __init__(self, run_dir: str):
        self.path, self.f = os.path.join(run_dir, "run.lock"), None

    def acquire(self) -> bool:
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        f = open(self.path, "a+b")
        try:
            if os.name == "nt":
                import msvcrt
                f.seek(0)
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            f.close()
            return False
        self.f = f
        return True

    def release(self) -> None:
        if self.f is not None:
            try:
                if os.name == "nt":
                    import msvcrt
                    self.f.seek(0)
                    msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            self.f.close()
            self.f = None


def is_running(run_dir: str) -> bool:
    if not os.path.exists(os.path.join(run_dir, "run.lock")):
        return False
    lock = RunLock(run_dir)
    if lock.acquire():
        lock.release()
        return False
    return True


# ------------------------------------------------------------------------------------------------ reading
def launch_of(run_dir: str) -> dict:
    """The settings a session started with. Runs from before launch.json have run_meta.json, in which an edge run's `model` can
    hold the network it was given (victim or control) instead of the editor's model."""
    launch = read_json(os.path.join(run_dir, "launch.json"))
    if launch:
        return launch
    m = read_json(os.path.join(run_dir, "run_meta.json"), {})
    model = m.get("model")
    return {"task": m.get("task"), "backend": m.get("backend"), "effort": m.get("effort"),
            "model": "claude-opus-5-5" if model in ("victim", "control") else model, "cfg": m}


def run_info(run_dir: str) -> dict:
    """Everything about a session at a glance."""
    st = read_json(os.path.join(run_dir, "state.json"), {}) or {}
    launch = launch_of(run_dir)
    cfg = launch.get("cfg") or {}
    settings = st.get("settings") or {}
    running = is_running(run_dir)
    status = (read_json(os.path.join(run_dir, "status.json"), {}) or {}) if running else {}
    interactive = bool(cfg.get("interactive"))
    if st.get("done") and not running:      # while its process is still writing the result, it is not finished yet
        state = "finished"
    elif st.get("halted"):
        state = "halted"
    elif running:
        state = "waiting" if status.get("activity") == "waiting" else "running"
    elif interactive:
        state = "idle" if st else "starting"
    else:
        state = "stopped" if st else "not started"
    backend = launch.get("backend")
    model = settings.get("editor_model") or launch.get("model")
    target = cfg.get("target") or {}
    goal = str(st.get("goal") or cfg.get("goal") or "")
    return {"name": os.path.basename(os.path.normpath(run_dir)), "path": run_dir, "state": state, "running": running,
            "task": launch.get("task") or st.get("task"), "backend": backend, "model": model,
            "effort": settings.get("editor_effort") or launch.get("effort"),
            "budget": settings.get("budget") or st.get("budget") or cfg.get("budget") or 0,
            "max_runs": cfg.get("max_runs") or 0, "turn": st.get("turn", 0), "runs": st.get("runs", 0),
            "passes": st.get("passes", 0), "tokens": st.get("tokens") or {}, "sandbox": st.get("sandbox"),
            "halted": st.get("halted"), "submission": st.get("submission"),
            "stopping": running and os.path.exists(os.path.join(run_dir, "stop")),
            "activity": status.get("activity"), "detail": status.get("detail"), "since": status.get("since"),
            "interactive": interactive, "goal": goal, "title": (goal.strip().splitlines() or [""])[0][:70],
            "target": target.get("name") or launch.get("task"), "target_kind": target.get("kind") or "benchmark",
            "mode": settings.get("mode") or cfg.get("mode") or "build",
            "ask_python": bool(settings.get("ask_python", cfg.get("ask_python"))),
            "ask_forward": bool(settings.get("ask_forward", cfg.get("ask_forward"))),
            "todo": st.get("todo") or [],
            "cost_usd": prices.cost_usd(model, st.get("tokens") or {}) if backend in ("anthropic", "openai") else None}


def list_runs(root: str | None = None) -> list[dict]:
    """All sessions, the most recently active first."""
    root = root or RUNS_DIR
    out = []
    for name in os.listdir(root) if os.path.isdir(root) else []:
        d = os.path.join(root, name)
        if os.path.exists(os.path.join(d, "launch.json")) or os.path.exists(os.path.join(d, "state.json")):
            info = run_info(d)
            sp = os.path.join(d, "state.json")
            info["mtime"] = os.path.getmtime(sp) if os.path.exists(sp) else os.path.getmtime(d)
            out.append(info)
    return sorted(out, key=lambda r: (not r["running"], -r["mtime"]))


def find_run(name: str, root: str | None = None) -> str | None:
    """A session's folder from its name or path."""
    for d in (os.path.join(root or RUNS_DIR, name), name):
        if os.path.isdir(d) and (os.path.exists(os.path.join(d, "launch.json")) or os.path.exists(os.path.join(d, "state.json"))):
            return os.path.abspath(d)
    return None


def _tail_bytes(path: str, n: int) -> str:
    try:
        with open(path, "rb") as f:
            f.seek(max(0, f.seek(0, 2) - n))
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def console_tail(run_dir: str, n: int = 2000) -> str:
    return _tail_bytes(os.path.join(run_dir, "console.log"), n)


def _brief_request(req):
    """A forward request for display: a long list (thousands of image indices) is cut to its length and its start, so the
    request stays short and is still JSON."""
    if not isinstance(req, dict):
        return req
    return {k: (f"{len(v):,} items: {', '.join(str(x) for x in v[:4])}, …" if isinstance(v, list) and len(v) > 12 else v)
            for k, v in req.items()}


def feed(run_dir: str, last: int = 12, max_lines: int = 14) -> list[dict]:
    """The latest steps: {turn, kind, passes, text, code, output} from the agent's explanations and the audit log."""
    text = _tail_bytes(os.path.join(run_dir, "explanations.md"), 2_000_000).replace("\r\n", "\n")
    steps = {}
    for m in re.finditer(r"^## Turn (\d+) \(([^;]+); passes used (\d+)/(\d+)\)\n\n(.*?)(?=^## Turn |\Z)", text, re.S | re.M):
        steps[int(m[1])] = {"turn": int(m[1]), "kind": m[2], "passes": f"{m[3]}/{m[4]}", "budget": int(m[4]), "text": m[5].strip(),
                            "code": "", "output": ""}
    for line in _tail_bytes(os.path.join(run_dir, "tool_calls.jsonl"), 4_000_000).splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        s = steps.get(r.get("turn"))
        if s is None:
            continue
        if r.get("kind") == "python":
            s["code"], s["output"], s["n"], s["seconds"] = r.get("code", ""), r.get("out", ""), r.get("run"), r.get("s")
        elif r.get("kind") == "forward":
            s["code"] = json.dumps(_brief_request(r.get("request", {})), ensure_ascii=False)
            s["output"] = json.dumps(r.get("summary") or r.get("invalid") or r.get("failed"), indent=1)
            s["n"], s["counted"] = r.get("pass"), "summary" in r
        lines = s["output"].splitlines()
        if max_lines and len(lines) > max_lines:
            s["output"] = "\n".join(lines[:max_lines] + [f"... {len(lines) - max_lines} more lines"])
    return [steps[k] for k in sorted(steps)][-last:]


def session_events(run_dir: str, offset: int = 0) -> tuple[list[dict], int]:
    """The session's events after `offset`, and the offset for next time. Sessions from before the event log are rebuilt once
    from their explanations and audit log (offset -1 = nothing more will come)."""
    if os.path.exists(os.path.join(run_dir, "events.jsonl")):
        return E.read_events(run_dir, max(offset, 0))
    if offset != 0:
        return [], offset
    if not os.path.exists(os.path.join(run_dir, "explanations.md")):
        return [], 0                     # a session that is still starting: its event log will appear
    out = []
    for s in feed(run_dir, last=10 ** 6, max_lines=0):
        kind = {"final answer": "final", "message": "message"}.get(s["kind"], s["kind"])
        ev = {"type": "assistant", "turn": s["turn"], "kind": kind, "text": s["text"]}
        if kind in ("python", "forward"):
            ev["block"] = s["code"]
        out.append(ev)
        if kind in ("python", "forward") and (s["output"] or s.get("n")):
            out.append({"type": "tool", "turn": s["turn"], "kind": kind, "n": s.get("n"), "seconds": s.get("seconds"),
                        "budget": s["budget"], "ok": s.get("counted", True), "counted": s.get("counted", True),
                        "output": E.clip(s["output"])})
    return out, -1


def work_files(run_dir: str, limit: int = 40) -> list[tuple[str, int, bool]]:
    """The top of the agent's work folder: (name, size in bytes or number of entries, is a folder), folders first."""
    work = os.path.join(run_dir, "work")
    out = []
    try:
        for n in sorted(os.listdir(work)):
            p = os.path.join(work, n)
            if n.startswith("."):
                continue
            if os.path.isdir(p) and not os.path.islink(p):
                out.append((n + "/", len(os.listdir(p)), True))
            elif os.path.isfile(p):
                out.append((n, os.path.getsize(p), False))
    except OSError:
        pass
    out.sort(key=lambda x: (not x[2], x[0]))
    return out[:limit]


def edited_tensors(run_dir: str) -> list[str]:
    d = os.path.join(run_dir, "work", "edited")
    try:
        return sorted(n[:-4] for n in os.listdir(d) if n.endswith(".npy")) if not os.path.islink(d) else []
    except OSError:
        return []


def benchmark_rows(root: str = ROOT) -> list[dict]:
    """Benchmark results saved by `wsp eval`: {name, benchmark, result}."""
    rows = []
    edge = read_json(os.path.join(root, "results", "edge_results.json"), {}) or {}
    for name, e in edge.items():
        rows.append({"name": name, "benchmark": "CIFAR-10 repair",
                     "result": f"clean {100 * e['clean_acc']:.2f}%, attack success {100 * e['asr']:.2f}%"})
    math = read_json(os.path.join(root, "results", "math", "summary.json"), {}) or {}
    for proto, table in math.items():
        for variant, sets in table.items():
            parts = []
            for s, r in sorted(sets.items()):
                gain = f" ({r['gain_vs_base']:+.1f} ± {r['se']:.1f})" if "gain_vs_base" in r else ""
                parts.append(f"{s} {r['acc']:.1f}%{gain}")
            rows.append({"name": _variant_name(variant), "benchmark": f"math, {proto}", "result": ", ".join(parts)})
    return rows


def _variant_name(v: str) -> str:
    """A benchmark folder's name, in words: base, paper_p50c, run_runs_<name>."""
    if v == "base":
        return "unedited"
    if v.startswith("paper_"):
        return f"paper's {v[len('paper_'):]}"
    return "WSP agent: " + v.removeprefix("run_").removeprefix("runs_")


def gpu_status() -> dict | None:
    """{"name", "used_gb", "total_gb", "percent"} of the first NVIDIA GPU, or None. Doesn't touch PyTorch."""
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.used,memory.total,utilization.gpu",
                            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
        name, used, total, pct = [x.strip() for x in p.stdout.splitlines()[0].split(",")]
        return {"name": name, "used_gb": float(used) / 1024, "total_gb": float(total) / 1024, "percent": int(pct)}
    except (OSError, ValueError, IndexError, subprocess.SubprocessError):
        return None


# ------------------------------------------------------------------------------------------------ controlling
def send_note(run_dir: str, text: str) -> str:
    """Leave an experimenter's note for the agent of a benchmark run; the harness delivers it once, with its next message."""
    d = os.path.join(run_dir, "notices")
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"note-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")
    return path


def send_message(run_dir: str, text: str) -> None:
    """The user says something to the agent. It is delivered when the agent next looks: at once if it is waiting for them,
    with its next result if it is working."""
    E.send(run_dir, "message", text=text)


def send_setting(run_dir: str, key: str, value) -> None:
    E.send(run_dir, "set", key=key, value=value)


def answer_permission(run_dir: str, request_id: str, decision: str, feedback: str = "") -> None:
    E.send(run_dir, "permission", id=request_id, decision=decision, feedback=feedback)


def request_stop(run_dir: str) -> None:
    """Ask the session's process to stop at the next safe point: between turns, or during a wait. A forward pass finishes first."""
    with open(os.path.join(run_dir, "stop"), "w", encoding="utf-8") as f:
        f.write("stop requested\n")


def request_interrupt(run_dir: str) -> None:
    """Ask the agent to stop what it is doing and wait for the user. The session's process keeps running."""
    E.request_interrupt(run_dir)


def force_stop(run_dir: str) -> bool:
    """Stop the session's process now. A pass in progress is lost and not counted; the session can be resumed."""
    import psutil
    if not is_running(run_dir):
        return False
    pid = (read_json(os.path.join(run_dir, "status.json"), {}) or {}).get("pid")
    if not pid:
        return False
    try:
        p = psutil.Process(pid)
        procs = p.children(recursive=True) + [p]
    except psutil.Error:
        return False
    for q in procs:
        try:
            q.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(procs, timeout=20)
    for _ in range(40):
        if not is_running(run_dir):
            break
        time.sleep(0.25)
    for name in ("stop", "interrupt"):
        try:
            os.remove(os.path.join(run_dir, name))
        except OSError:
            pass
    return True


def new_run_name(task: str) -> str:
    return f"{SHORT_NAME.get(task, task)}-{time.strftime('%Y%m%d-%H%M%S')}"


def _spawn(args: list[str], log_path: str, cwd: str | None = None) -> subprocess.Popen:
    """Start a Python program that outlives this one: its own session, no window, output to a file."""
    kw = {"start_new_session": True} if os.name == "posix" else \
        {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.CREATE_NO_WINDOW}
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    env["PYTHONPATH"] = ROOT + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")     # `-m wsp...` from any folder
    with open(log_path, "ab") as log:
        return subprocess.Popen([sys.executable, *args], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                cwd=cwd or os.getcwd(), env=env, close_fds=True, **kw)


def _consents(accept_no_sandbox: bool, accept_subscription: bool) -> list[str]:
    return (["--accept-no-sandbox"] if accept_no_sandbox else []) + (["--accept-subscription-notice"] if accept_subscription else [])


def run_args(run_dir: str, task: str, backend: str, model: str | None, effort: str, budget: int | None, task_args: list[str] | None,
             accept_no_sandbox: bool, accept_subscription: bool) -> list[str]:
    """The command (after the Python executable) that starts a benchmark run."""
    a = ["-m", "wsp.agent_main", "--task", task, "--run", run_dir, "--backend", backend, "--effort", effort]
    if budget:
        a += ["--budget", str(budget)]
    if model:
        a += ["--model", model]
    for kv in task_args or []:
        a += ["--task-arg", kv]
    return a + _consents(accept_no_sandbox, accept_subscription)


def session_args(run_dir: str, target: dict, goal: str, backend: str, model: str | None, effort: str, budget: int | None,
                 project_dir: str, mode: str = "build", ask: tuple = (), accept_no_sandbox: bool = False,
                 accept_subscription: bool = False, task_args: list[str] | None = None) -> list[str]:
    """The command that starts a conversation about a target (wsp/tasks/workspace/targets.py)."""
    a = run_args(run_dir, "workspace", backend, model, effort, budget, task_args, accept_no_sandbox, accept_subscription)
    a += ["--target-json", json.dumps(target), "--goal", goal, "--project", project_dir, "--mode", mode]
    for kind in ask:
        a += ["--ask", kind]
    return a


def resume_args(run_dir: str, accept_no_sandbox: bool = False, accept_subscription: bool = False) -> list[str]:
    """The command that carries on a session with the settings it has."""
    return ["-m", "wsp.agent_main", "--run", run_dir] + _consents(accept_no_sandbox, accept_subscription)


def start(run_dir: str, args: list[str], wait_s: float = 120.0) -> tuple[bool, str]:
    """Start a session's process in the background and wait until it holds the session's lock. -> (started, message)."""
    os.makedirs(run_dir, exist_ok=True)
    log = os.path.join(run_dir, "console.log")
    size = os.path.getsize(log) if os.path.exists(log) else 0
    p = _spawn(args, log)
    t0 = time.time()
    while time.time() - t0 < wait_s:
        if is_running(run_dir):
            return True, f"running in the background (log: {log})"
        if p.poll() is not None:
            # The venv launcher on Windows can exit code 0 only after the program does; either way it is over.
            with open(log, "rb") as f:
                f.seek(size)
                return False, f.read().decode("utf-8", "replace").strip()[-2000:] or f"it exited with code {p.returncode}"
        time.sleep(0.25)
    return True, f"still starting; see {log}"


def eval_args(run_dir: str, task: str) -> list[list[str]]:
    """The commands that benchmark a finished run. Math uses the same batching as the stored unedited answers it is compared
    with, so the comparison stays paired."""
    if task == "edge_backdoor":
        return [[os.path.join(ROOT, "scripts", "eval_edge.py"), "--run", run_dir]]
    rel = os.path.relpath(run_dir, ROOT).replace("\\", "/")
    return [[os.path.join(ROOT, "scripts", "eval_math.py"), "run", "--variant", "base", "--variant", f"run:{rel}",
             "--sets", "math500,gsm8k", "--limit", "100", "--batch-size", "16", "--chunk", "32"]]


def _workspace_task(run_dir: str):
    launch = launch_of(run_dir)
    if launch.get("task") != "workspace":
        raise ValueError("Benchmark runs submit when they finish: see the run's submission/ folder.")
    from .tasks.workspace.task import WorkspaceTask
    return WorkspaceTask(launch["cfg"], run_dir, os.path.join(run_dir, "work"), log=lambda s: None)


def diff_session(run_dir: str) -> list[dict]:
    """What a conversation's current edit changes, tensor by tensor (see WorkspaceTask.diff)."""
    return _workspace_task(run_dir).diff()


def export_session(run_dir: str, out_dir: str | None = None, full: bool = False) -> dict:
    """Write the current edit of a conversation's target (see WorkspaceTask.export). -> what was written."""
    return _workspace_task(run_dir).export(out_dir or os.path.join(run_dir, "export"), full=full)
