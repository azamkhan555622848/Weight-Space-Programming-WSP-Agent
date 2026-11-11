"""The isolation of the model's code, on whatever system this runs on.

Everywhere: the sandbox commands are what they should be (checked as text), the harness refuses the links and paths the
model's code could use to reach outside the work folder, and the kernel survives stray output and stops programs it started.
Where the system has a sandbox: it really works, shown with decoy files (never .env or a real login file).
Tests that need symbolic links are skipped where this user can't create them (default Windows); CI runs them on Linux and macOS.

    python tests/test_sandbox.py
"""
import json
import os
import shutil
import sys
import tempfile
import time
import uuid

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests"))
from test_harness_edge import make_data  # noqa: E402
from wsp import consent, sandbox  # noqa: E402
from wsp.harness import Harness  # noqa: E402
from wsp.kernel import Kernel  # noqa: E402
from wsp.llm import ScriptedBackend  # noqa: E402
from wsp.sandbox import UnsafePath, open_in  # noqa: E402
from wsp.tasks.base import ForwardError  # noqa: E402
from wsp.tasks.edge.task import EdgeBackdoorTask  # noqa: E402

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:600]}"))


def raises(exc, fn):
    try:
        fn()
    except exc:
        return True
    return False


def can_link() -> bool:
    d = tempfile.mkdtemp()
    try:
        os.symlink(os.path.join(d, "x"), os.path.join(d, "y"))
        return True
    except (OSError, NotImplementedError):
        return False
    finally:
        shutil.rmtree(d, ignore_errors=True)


def commands():
    work, dirs = "/x/runs/r/work", ["/opt/python", "/x/.venv", "/usr/local/py"]
    ro = ["/x/models/m/model.safetensors", "/x/wsp/kernel_worker.py"]
    a = sandbox.bwrap_argv(work, ro, {"HOME": "/tmp"}, dirs, "bwrap")
    joined = " ".join(a)
    check("bwrap: no network, clean environment, new session", all(f in a for f in ("--unshare-all", "--clearenv", "--new-session",
                                                                                   "--die-with-parent")), a)
    i = a.index("--bind")
    check("bwrap: only the work folder is writable", a[i + 1:i + 3] == [work, work] and a.count("--bind") == 1, a)
    bound = {a[j + 1] for j, v in enumerate(a) if v == "--ro-bind"}
    check("bwrap: read-only binds are exactly what's listed", bound - {"/usr", "/bin", "/sbin", "/lib", "/lib32", "/lib64"}
          == {"/opt/python", "/x/.venv", *ro}, bound)
    check("bwrap: the project and home are not bound", "/x " not in joined + " " and "--ro-bind /x " not in joined
          and "/home" not in joined and ".env " not in joined, joined)
    p = sandbox.seatbelt_profile("/x/runs/r/work", ro, ["/opt/python", "/x/.venv"], "/Users/me", "/x", "/private/tmp")
    lines = p.splitlines()
    check("seatbelt: no network, writes denied except work and temp", "(deny network*)" in lines and "(deny file-write*)" in lines
          and any(ln.startswith("(allow file-write*") and "work" in ln and "tmp" in ln for ln in lines), p)
    deny = next(i for i, ln in enumerate(lines) if ln.startswith("(deny file-read*"))
    allow = next(i for i, ln in enumerate(lines) if ln.startswith("(allow file-read* "))
    q = lambda path: json.dumps(os.path.realpath(path))      # noqa: E731
    check("seatbelt: home and project unreadable, exceptions after", deny < allow and q("/Users/me") in lines[deny]
          and q("/x") in lines[deny] and q("/x/.venv") in lines[allow] and q(ro[0]) in lines[allow], p)
    check("seatbelt: the keychain is off limits", "com.apple.SecurityServer" in p and "(deny mach-lookup" in p, p)
    d = sandbox.python_dirs()
    home = os.path.expanduser("~")
    check("python_dirs: never the project or home as a whole", all(os.path.normcase(x) not in
          (os.path.normcase(ROOT), os.path.normcase(home)) and not sandbox._inside(ROOT, x) for x in d), d)
    pol = sandbox.resolve_policy()
    check("policy matches the system", (pol.kind == "none" and pol.note) if os.name == "nt" else pol.kind in ("bwrap", "seatbelt", "none"), pol)


def paths():
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "sub"))
    open(os.path.join(d, "sub", "a.txt"), "w").write("hello")
    with open_in(d, "sub/a.txt") as f:
        check("open_in reads a regular file", f.read() == b"hello")
    check("open_in: a missing file is FileNotFoundError", raises(FileNotFoundError, lambda: open_in(d, "nope")))
    for bad in ("../x", "sub/../../x", "/etc/passwd", "C:\\x", "\\\\server\\x", "", "."):
        check(f"open_in refuses {bad!r}", raises(UnsafePath, lambda: open_in(d, bad)))
    check("open_in refuses a folder", raises(UnsafePath, lambda: open_in(d, "sub")))
    shutil.rmtree(d, ignore_errors=True)


def kernel_robustness():
    os.environ["ANTHROPIC_API_KEY"] = "decoy-not-a-key"
    k = Kernel(tempfile.mkdtemp(), timeout=120)
    r = k.run("import subprocess, sys\nsubprocess.run([sys.executable, '-c', 'print(\"stray output\")'])\nprint('fine')")
    check("a started program's output can't corrupt the protocol", r["status"] == "ok" and r["out"] == "fine\n", r)
    r = k.run("subprocess.run([sys.executable, '-c', 'import sys; sys.stderr.write(\"x\" * 3000000)'])\nprint('fine')")
    check("lots of stderr doesn't block the kernel", r["status"] == "ok" and r["out"] == "fine\n", r)
    r = k.run("import os\n'ANTHROPIC_API_KEY' in os.environ, 'OPENAI_API_KEY' in os.environ")
    check("keys aren't in the kernel's environment", r["out"].strip() == "(False, False)", r)
    r = k.run("import sys\nm = sys.modules['__main__']\nm.PROTO.write('not json\\n')\nm.PROTO.flush()")
    check("garbage on the protocol is reported, not a crash", r["status"] == "died", r)
    r = k.run("print('back')")
    check("the kernel restarts after that", r["status"] == "ok" and r["out"] == "back\n" and r["restarted"], r)
    # a program started by the code stops with the kernel; it reports through a file because pids differ inside a sandbox
    beat = "import time\nwhile True:\n    open('beat.txt', 'w').write(str(time.time()))\n    time.sleep(0.1)"
    k.run(f"import subprocess, sys\nP = subprocess.Popen([sys.executable, '-c', {beat!r}])")
    f = os.path.join(k.work_dir, "beat.txt")
    for _ in range(600):       # a new Python can take many seconds to start on Windows
        if os.path.exists(f):
            break
        time.sleep(0.1)
    k.kill()
    time.sleep(0.5)
    a = os.path.getmtime(f) if os.path.exists(f) else None
    time.sleep(1.5)
    check("kill also stops programs the code started", a is not None and os.path.getmtime(f) == a, a)
    k.close()
    del os.environ["ANTHROPIC_API_KEY"]


def consent_flow():
    d = tempfile.mkdtemp()
    orig, consent._path = consent._path, lambda: os.path.join(d, "consent.json")
    stdin, sys.stdin = sys.stdin, None
    try:
        check("consent: no terminal means no", consent.confirm("k", "text") is False)
        check("consent: a flag accepts for this run only", consent.confirm("k", "text", assume_yes=True) and not consent.accepted("k"))
        consent.record("k")
        check("consent: remembered once accepted", consent.confirm("k", "text"))
    finally:
        consent._path, sys.stdin = orig, stdin
        shutil.rmtree(d, ignore_errors=True)


def attacks():
    """What code in the kernel could leave in the work folder, against the harness outside it."""
    if not can_link():
        print("SKIP link attacks: this user can't create symbolic links here (CI runs them)")
        return
    tmp = tempfile.mkdtemp()
    data = os.path.join(tmp, "data")
    make_data(data)
    outside = os.path.join(tmp, "outside")
    os.makedirs(outside)
    secret = os.path.join(outside, "secret.npy")
    np.save(secret, np.zeros((3, 32, 32, 3), np.uint8))
    text = os.path.join(outside, "secret.txt")
    open(text, "w").write("TOP-SECRET-TEXT")
    run = os.path.join(tmp, "run")
    work = os.path.join(run, "work")
    cfg = {"budget": 3, "max_runs": 20, "handoff_every": 40, "explain": True, "data_dir": data, "device": "cpu",
           "accept_no_sandbox": True}
    task = EdgeBackdoorTask(cfg, run, work)
    h = Harness(task, ScriptedBackend([]), run, cfg)
    task.prepare_work()
    h.st = {"turn": 0, "passes": 0, "runs": 0, "req_keys": {}, "requests": 0, "fwd_fail_streak": 0, "next": None}

    # notes.md is sent to the provider at a handoff
    os.remove(os.path.join(work, "notes.md"))
    os.symlink(text, os.path.join(work, "notes.md"))
    check("attack: a link at notes.md isn't read", h.read_notes() == "" and "TOP-SECRET" not in h.handoff_message("x")
          if hasattr(h, "read_notes") else False)
    os.remove(os.path.join(work, "notes.md"))

    # "images" reading a file through a link, or by a path out of the folder
    os.symlink(secret, os.path.join(work, "linked.npy"))
    os.symlink(outside, os.path.join(work, "linked_dir"))
    prac = {"type": "classify", "model": "edited", "practice": [0]}
    for name in ("linked.npy", "linked_dir/secret.npy", "../../outside/secret.npy", secret):
        check(f"attack: images={os.path.basename(name)!r} via link or path is refused",
              raises(ForwardError, lambda: task.forward({**prac, "images": name}, 1)))
    sibling = work + "_sibling"
    os.makedirs(sibling)
    shutil.copy(secret, sibling)
    check("attack: a sibling folder sharing the name's prefix is refused",
          raises(ForwardError, lambda: task.forward({**prac, "images": "../work_sibling/secret.npy"}, 1)))

    # a link at the pass's output folder must not redirect the harness's writes
    target = os.path.join(tmp, "redirect_target")
    os.makedirs(target)
    os.symlink(target, os.path.join(work, "fwd_1"))
    task.forward({**prac}, 1)
    check("attack: a link at fwd_1 doesn't redirect the output", os.listdir(target) == [] and
          os.path.isfile(os.path.join(work, "fwd_1", "pred.npy")) and not os.path.islink(os.path.join(work, "fwd_1")),
          os.listdir(target))

    # a link in edited/ must not be hashed or loaded
    os.makedirs(os.path.join(work, "edited"), exist_ok=True)
    os.symlink(secret, os.path.join(work, "edited", "fc.b.npy"))
    msg = h.run_forward(json.dumps({**prac, "practice": [1]}))
    check("attack: a link in edited/ makes the pass invalid, not counted", msg.startswith("The forward request is not valid")
          and h.st["passes"] == 0, msg)
    check("attack: and a direct pass refuses it too", raises(ForwardError, lambda: task.forward({**prac, "practice": [2]}, 2)))
    os.remove(os.path.join(work, "edited", "fc.b.npy"))

    # the snapshot: an edit changed after the hash can't change what the pass uses
    w = task.orig["fc.b"].copy()
    np.save(os.path.join(work, "edited", "fc.b.npy"), w + 1.0)
    snap = task.snapshot_edits()
    np.save(os.path.join(work, "edited", "fc.b.npy"), w + 2.0)
    task._pass_edits = snap
    check("attack: a pass uses the snapshot it hashed", np.allclose(task.current()["fc.b"], w + 1.0))
    task._pass_edits = None
    check("the end of the session uses the latest edit", np.allclose(task.current()["fc.b"], w + 2.0))
    shutil.rmtree(tmp, ignore_errors=True)


def live():
    pol = sandbox.resolve_policy()
    if pol.kind == "none":
        print(f"SKIP live sandbox: none on this system. {pol.note}")
        return
    k = Kernel(tempfile.mkdtemp(), timeout=180, policy=pol)
    why = sandbox._self_test(k)
    check(f"live {pol.kind}: decoys unreadable, no writes outside work, no network", why is None, why)
    if why:
        return
    r = k.run("import numpy, torch\nprint(numpy.ones(2).sum() + torch.ones(2).sum().item())")
    check(f"live {pol.kind}: numpy and torch work inside", r["status"] == "ok" and "4.0" in r["out"], r)
    # exactly the listed read-only file is visible, and not its neighbour
    d = os.path.join(os.path.expanduser("~"), ".wsp", f"test-{uuid.uuid4().hex[:8]}")
    os.makedirs(d)
    listed, neighbour = os.path.join(d, "listed.txt"), os.path.join(d, "neighbour.txt")
    open(listed, "w").write("visible")
    open(neighbour, "w").write("hidden")
    k2 = Kernel(tempfile.mkdtemp(), timeout=180, ro_paths=[listed], policy=pol)
    code = f"""
def read(p):
    try:
        return open(p).read()
    except OSError:
        return None
print(read({listed!r}), read({neighbour!r}))
try:
    open({listed!r}, 'w').write('changed')
    print('wrote')
except OSError:
    print('read-only')
"""
    r = k2.run(code)
    check(f"live {pol.kind}: a listed file is readable and read-only, its neighbour isn't", r["out"].split() == ["visible", "None", "read-only"], r)
    k.close()
    k2.close()
    shutil.rmtree(d, ignore_errors=True)


def main():
    commands()
    paths()
    kernel_robustness()
    consent_flow()
    attacks()
    live()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
