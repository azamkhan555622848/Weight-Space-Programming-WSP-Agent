"""Run the CPU test files one after another, each in its own process, and say which failed.

    python tests/run_all.py                          # all of them
    python tests/run_all.py test_kernel.py ...       # some

On GitHub Actions a failing file is also reported as an annotation, so its FAIL lines and the end of its output show on the
run's page, where the full log needs a signed-in user.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ALL = ["test_sandbox.py", "test_kernel.py", "test_sampler.py", "test_harness_edge.py", "test_grading_vs_paper.py",
       "test_workspace.py", "test_interactive.py", "test_backends.py", "test_runs.py", "test_tui.py"]
TIMEOUT = 1200


def summary(out: str, limit: int = 6000) -> str:
    """The FAIL lines with their details, then the last lines (where a traceback ends)."""
    lines = out.splitlines()
    picked, i = [], 0
    while i < len(lines):
        if lines[i].startswith("FAIL"):
            picked.append(lines[i])
            i += 1
            while i < len(lines) and lines[i].startswith("     "):
                picked.append(lines[i])
                i += 1
        else:
            i += 1
    text = "\n".join(ln[:400] for ln in picked + (["…"] if picked else []) + lines[-25:])
    return text[-limit:]


def annotate(title: str, text: str) -> None:
    def esc(s: str) -> str:
        return s.replace("%", "%25").replace("\r", "%0D").replace("\n", "%0A")
    print(f"::error title={esc(title).replace(':', '%3A').replace(',', '%2C')}::{esc(text)}", flush=True)


def main(names: list[str]) -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    failed = []
    for n in names or ALL:
        name = os.path.basename(n)
        print(f"=== {name}", flush=True)
        t = time.time()
        try:
            p = subprocess.run([sys.executable, os.path.join(HERE, name)], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, encoding="utf-8", errors="replace", env=env, timeout=TIMEOUT)
            out, code = p.stdout, p.returncode
        except subprocess.TimeoutExpired as e:
            out = (e.stdout.decode("utf-8", "replace") if isinstance(e.stdout, bytes) else e.stdout or "")
            out, code = out + f"\n[stopped after {TIMEOUT} s]", "timeout"
        print(out, flush=True)
        print(f"=== {name}: exit {code} after {time.time() - t:.0f} s", flush=True)
        if code != 0:
            failed.append(name)
            if os.environ.get("GITHUB_ACTIONS") == "true":
                annotate(f"{name} failed on {os.environ.get('RUNNER_OS', sys.platform)}", summary(out))
    print("ALL FILES PASS" if not failed else "FAILED: " + ", ".join(failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
