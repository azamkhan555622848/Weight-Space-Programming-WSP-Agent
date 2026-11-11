"""The `wsp` commands and the run folder, end to end, with no account: real background runs on a tiny synthetic task, with a fake
`claude` on PATH as the model. Starting, watching, a note to the agent, stopping safely, stopping at once, resuming, and the lock.

    python tests/test_runs.py
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_harness_edge import make_data  # noqa: E402
from wsp import cli, prices, runs  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:900]}"))


E = "Checking something; this explanation is long enough to count as one.\n\n"
FWD = E + "```forward\n" + json.dumps({"type": "classify", "model": "edited", "practice": [0, 1, 2]}) + "\n```"
PY = E + "```python\nprint('kernel works')\n```"

# Plays a list of replies, one per call; "__SLEEP__" waits until a file appears, so a test can act during a model call.
FAKE = r'''
import json, os, sys, time
ctl = os.environ["FAKE_CTL"]
prompt = sys.stdin.buffer.read().decode("utf-8")
n_path = os.path.join(ctl, "n.txt")
n = int(open(n_path).read()) if os.path.exists(n_path) else 0
open(n_path, "w").write(str(n + 1))
open(os.path.join(ctl, f"prompt{n}.txt"), "w", encoding="utf-8").write(prompt)
replies = json.load(open(os.path.join(ctl, "replies.json"), encoding="utf-8"))
r = replies[n] if n < len(replies) else "Final answer."
if r == "__SLEEP__":
    while not os.path.exists(os.path.join(ctl, "wake")):
        time.sleep(0.1)
    r = "Final answer."
print(json.dumps({"type": "system", "subtype": "init", "model": "claude-fake", "mcp_servers": [], "tools": []}), flush=True)
print(json.dumps({"type": "assistant", "message": {"content": [{"type": "text", "text": r}]}}), flush=True)
print(json.dumps({"type": "result", "result": r, "num_turns": 1, "stop_reason": "end_turn",
                  "usage": {"input_tokens": 5, "output_tokens": 5}}), flush=True)
'''


def install_fake(d):
    py = os.path.join(d, "claude.py")
    open(py, "w", encoding="utf-8").write(FAKE)
    if os.name == "nt":
        open(os.path.join(d, "claude.cmd"), "w").write(f'@"{sys.executable}" "%~dp0claude.py" %*\r\n')
    else:
        exe = os.path.join(d, "claude")
        open(exe, "w").write(f'#!/bin/sh\nexec "{sys.executable}" "{py}" "$@"\n')
        os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)
    os.environ["PATH"] = d + os.pathsep + os.environ["PATH"]


def wait_for(cond, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if cond():
            return True
        time.sleep(0.25)
    return False


def main():
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "data")
    make_data(data)
    ctl = os.path.join(tmp, "ctl")
    fake_dir = os.path.join(tmp, "fake")
    os.makedirs(fake_dir)
    os.makedirs(ctl)
    install_fake(fake_dir)
    os.environ["FAKE_CTL"] = ctl
    # consent state and CLI session folders go to a temporary home
    os.environ["HOME"] = os.environ["USERPROFILE"] = os.path.join(tmp, "home")
    os.makedirs(os.environ["HOME"])
    runs.RUNS_DIR = os.path.join(tmp, "runs")

    def set_replies(replies):
        for f in os.listdir(ctl):
            os.remove(os.path.join(ctl, f))
        json.dump(replies, open(os.path.join(ctl, "replies.json"), "w", encoding="utf-8"))

    base = ["--task", "edge_backdoor", "--backend", "claude-cli", "--budget", "1", "--accept-no-sandbox", "--accept-subscription-notice",
            "--task-arg", f"data_dir={data}", "--task-arg", "device=cpu"]

    # a whole run, in the background
    set_replies([PY, FWD, "Final answer: done."])
    rc = cli.main(["run", "--name", "whole", *base])
    d = os.path.join(runs.RUNS_DIR, "whole")
    check("wsp run: starts in the background", rc == 0 and runs.is_running(d) or runs.run_info(d)["state"] == "finished", rc)
    check("the run finishes on its own", wait_for(lambda: runs.run_info(d)["state"] == "finished"), runs.console_tail(d))
    r = runs.run_info(d)
    check("run_info: counters, model, backend", r["passes"] == 1 and r["runs"] == 1 and r["model"] == "claude-opus-5-5"
          and r["backend"] == "claude-cli" and r["cost_usd"] is None and r["submission"], r)
    check("run_info: the sandbox in use is recorded", r["sandbox"] in ("none", "bwrap", "seatbelt"), r["sandbox"])
    steps = runs.feed(d)
    check("feed: the agent's reasoning and results", any(s["kind"] == "python" and "kernel works" in s["output"] for s in steps)
          and any(s["kind"] == "forward" for s in steps) and any(s["kind"] == "final answer" for s in steps), steps)
    check("the lock is released at the end", not runs.is_running(d))
    check("wsp run refuses a name in use", _exits(lambda: cli.main(["run", "--name", "whole", *base])))
    check("wsp resume refuses a finished run", _exits(lambda: cli.main(["resume", "whole"])))

    # a stop request during a model call: this turn finishes, then it stops, and resuming finishes the run
    set_replies(["__SLEEP__"])
    rc = cli.main(["run", "--name", "stopme", *base])
    d = os.path.join(runs.RUNS_DIR, "stopme")
    check("second run: running", rc == 0 and wait_for(lambda: runs.run_info(d)["activity"] == "model"), runs.console_tail(d))
    check("while running, status says so", "waiting for the model" in cli.status_line(runs.run_info(d)), runs.run_info(d))
    second = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "run_agent.py"), "--task", "edge_backdoor", "--run", d, "--accept-no-sandbox",
                             "--accept-subscription-notice"], capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=180)
    check("a second process can't take the same run", second.returncode != 0 and "already running" in second.stderr, second.stderr[-300:])
    runs.send_note(d, "A note from the test.")
    cli.main(["stop", "stopme"])
    check("stop: asks, and says it's stopping", "stopping after this step" in cli.status_line(runs.run_info(d)), runs.run_info(d))
    open(os.path.join(ctl, "wake"), "w").close()
    check("stop: the run stops at the next safe point", wait_for(lambda: not runs.is_running(d)), runs.console_tail(d))
    r = runs.run_info(d)
    check("stop: the run is resumable, and the flag is cleared", r["state"] == "stopped" and not os.path.exists(os.path.join(d, "stop")), r)
    set_replies([FWD, "Final answer: done."])
    rc = cli.main(["resume", "stopme", "--accept-no-sandbox", "--accept-subscription-notice"])
    check("resume: finishes the run", rc == 0 and wait_for(lambda: runs.run_info(d)["state"] == "finished"), runs.console_tail(d))
    sent = open(os.path.join(ctl, "prompt0.txt"), encoding="utf-8").read()
    check("resume: a handoff, and the note is delivered", "THIS IS A RESTARTED CONVERSATION" in sent and "A note from the test." in sent, sent[-600:])

    # stopping at once, in the middle of a model call
    set_replies(["__SLEEP__"])
    rc = cli.main(["run", "--name", "killme", *base])
    d = os.path.join(runs.RUNS_DIR, "killme")
    check("third run: running", rc == 0 and wait_for(lambda: runs.run_info(d)["activity"] == "model"), runs.console_tail(d))
    rc = cli.main(["stop", "killme", "--now"])
    check("stop --now: it ends at once, with the lock freed", rc == 0 and not runs.is_running(d), runs.console_tail(d))
    set_replies([FWD, "Final answer: done."])
    rc = cli.main(["resume", "killme", "--accept-no-sandbox", "--accept-subscription-notice"])
    check("resume after a kill: finishes", rc == 0 and wait_for(lambda: runs.run_info(d)["state"] == "finished"), runs.console_tail(d))

    # a run that can't start says why
    rc = cli.main(["run", "--name", "bad", "--task", "edge_backdoor", "--backend", "codex-cli", "--accept-no-sandbox", "--accept-subscription-notice",
                   "--task-arg", f"data_dir={data}"] if shutil.which("codex") is None else ["status"])
    d = os.path.join(runs.RUNS_DIR, "bad")
    if shutil.which("codex") is None:
        check("a run that can't start explains why", rc == 1 and "codex" in runs.console_tail(d), runs.console_tail(d))

    # listing, and the rest of the small pieces
    names = [x["name"] for x in runs.list_runs(runs.RUNS_DIR)]
    check("list_runs finds them", {"whole", "stopme", "killme"} <= set(names), names)
    check("find_run: by name or path, or not at all", runs.find_run("whole", runs.RUNS_DIR) and not runs.find_run("nope", runs.RUNS_DIR))
    p = runs.prices.cost_usd("claude-opus-5-5", {"output": 78585, "cache_read": 1677219, "cache_write": 487773, "input": 228})
    check("cost: the Qwen run's known cost", abs(p - 5.81) < 0.01, p)
    check("cost: an unknown model has no estimate", prices.cost_usd("some-model", {"output": 5}) is None)
    e = prices.estimate("qwen_math", 10, "claude-opus-5-5")
    check("estimate: close to the real runs", 5 < e["usd"] < 7 and e["minutes"] == 600, e)
    check("estimate: scales with the model's prices", prices.estimate("qwen_math", 10, "gpt-6-astra")["usd"] > e["usd"])
    rows = runs.benchmark_rows()
    check("benchmark rows read the saved results", any("attack success" in r["result"] for r in rows) if
          os.path.exists(os.path.join(ROOT, "results", "edge_results.json")) else True, rows[:2])
    open(os.path.join(ctl, "wake"), "w").close()               # let any model call still waiting finish
    for r in runs.list_runs(runs.RUNS_DIR):                    # and leave no session process behind
        runs.force_stop(r["path"])
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


def _exits(fn) -> bool:
    try:
        fn()
    except SystemExit:
        return True
    return False


if __name__ == "__main__":
    sys.exit(main())
