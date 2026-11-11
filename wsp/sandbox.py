"""Isolation for the code the model writes.

The kernel runs inside an operating-system sandbox where the system has one: bubblewrap on Linux (the paper's method, adapted
from its harness/sandbox.py) and sandbox-exec on macOS. Inside, there is no network, the home folder and this project are
hidden (so .env, login files and the hidden test data can't be read), and only the run's work folder is writable. Forward
passes, API keys and CLI logins stay outside, in the harness. Windows has no simple equivalent, so there the harness asks
the user to accept running without one (wsp/consent.py); WSL2 gives a Windows user the Linux sandbox.

A sandbox alone isn't enough, because the harness outside reads and writes files in the work folder too. Code inside can
leave a link there pointing anywhere, and a harness following it would read `.env` on the model's behalf. So everything the
harness reads from the work folder goes through open_in, which refuses links and anything but regular files.
"""
from __future__ import annotations

import json
import os
import shutil
import stat
import sys
import tempfile
import uuid
from dataclasses import dataclass
from typing import BinaryIO

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

LINUX_INSTALL = ("Install bubblewrap (Debian/Ubuntu: sudo apt install bubblewrap; Fedora: sudo dnf install bubblewrap; "
                 "Arch: sudo pacman -S bubblewrap).")
LINUX_NAMESPACES = ("bubblewrap needs unprivileged user namespaces. On Ubuntu 23.10 or newer, allow them with: sudo sysctl -w "
                    "kernel.apparmor_restrict_unprivileged_userns=0 (or add an AppArmor profile for bwrap). In a "
                    "container, start it with user namespaces allowed.")
WINDOWS_NOTE = ("Windows has no simple equivalent of bubblewrap. For a real sandbox, run under WSL2 (wsl --install Ubuntu) "
                "and install bubblewrap there.")


class UnsafePath(Exception):
    """A path in the work folder that the harness refuses to use: a link, not a regular file, or outside the folder."""


@dataclass(frozen=True)
class Policy:
    kind: str       # "bwrap", "seatbelt" or "none"
    note: str = ""  # why there is no sandbox, and how to get one


NONE = Policy("none")


def resolve_policy() -> Policy:
    if sys.platform.startswith("linux"):
        return Policy("bwrap") if shutil.which("bwrap") else Policy("none", "bubblewrap is not installed. " + LINUX_INSTALL)
    if sys.platform == "darwin":
        return Policy("seatbelt") if os.path.exists("/usr/bin/sandbox-exec") else Policy("none", "sandbox-exec is missing.")
    return Policy("none", WINDOWS_NOTE)


def _inside(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([os.path.normcase(path), os.path.normcase(folder)]) == os.path.normcase(folder)
    except ValueError:           # different drives
        return False


def python_dirs(python: str | None = None) -> list[str]:
    """The folders the interpreter needs: the virtual environment and the base install. Deliberately not every sys.path
    entry: an editable install puts this project there, which would expose .env and the hidden test data. The home folder
    or this project as a whole is never returned, so an environment installed that way fails with a clear error."""
    py = python or sys.executable
    dirs = {sys.prefix, sys.exec_prefix, sys.base_prefix, sys.base_exec_prefix, os.path.dirname(os.path.realpath(py))}
    home = os.path.expanduser("~")
    return sorted(d for d in {os.path.realpath(p) for p in dirs}
                  if not _inside(ROOT, d) and not _inside(home, d) and d != os.path.sep)


# ------------------------------------------------------------------------------------------------ Linux: bubblewrap
def bwrap_argv(work_dir: str, ro_paths: list[str], env: dict, dirs: list[str], bwrap: str = "bwrap") -> list[str]:
    """The sandbox command: --unshare-all takes away the network; only what is listed here exists inside."""
    a = [bwrap, "--unshare-all", "--die-with-parent", "--new-session", "--clearenv", "--ro-bind", "/usr", "/usr"]
    for name in ("bin", "sbin", "lib", "lib32", "lib64"):
        p = "/" + name
        if os.path.islink(p):
            a += ["--symlink", os.readlink(p), p]
        elif os.path.isdir(p):
            a += ["--ro-bind", p, p]
    a += ["--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--ro-bind-try", "/etc/ld.so.cache", "/etc/ld.so.cache",
          "--ro-bind-try", "/etc/alternatives", "/etc/alternatives",
          "--ro-bind-try", "/sys/devices/system/cpu", "/sys/devices/system/cpu"]
    for p in dirs + ro_paths:
        if p != "/usr" and not p.startswith("/usr/"):
            a += ["--ro-bind", p, p]
    a += ["--bind", work_dir, work_dir, "--chdir", work_dir]
    for k, v in env.items():
        a += ["--setenv", k, v]
    return a


# ------------------------------------------------------------------------------------------------ macOS: sandbox-exec
def seatbelt_profile(work_dir: str, ro_paths: list[str], dirs: list[str], home: str, project: str, tmp: str) -> str:
    """Later rules win. Reads of file metadata stay allowed, because Python checks paths on the way to its own files."""
    q = lambda p: json.dumps(os.path.realpath(p))      # noqa: E731 (SBPL strings are quoted like JSON)
    readable = [f"(subpath {q(d)})" for d in dirs + [work_dir, tmp]] + [f"(literal {q(p)})" for p in ro_paths]
    return "\n".join([
        "(version 1)",
        "(allow default)",
        "(deny network*)",
        "(deny file-write*)",
        f'(allow file-write* (subpath {q(work_dir)}) (subpath {q(tmp)}) (literal "/dev/null") (literal "/dev/zero") '
        '(literal "/dev/tty"))',
        f"(deny file-read* (subpath {q(home)}) (subpath {q(project)}))",
        f"(allow file-read* {' '.join(readable)})",
        "(allow file-read-metadata)",
        # the keychain, where Claude Code keeps its login on a Mac, and opening apps or links
        '(deny mach-lookup (global-name "com.apple.SecurityServer") (global-name "com.apple.securityd.xpc") '
        '(global-name "com.apple.secd") (global-name "com.apple.lsd.mapdb") (global-name "com.apple.lsd.open") '
        '(global-name "com.apple.coreservices.launchservicesd"))',
    ])


# ------------------------------------------------------------------------------------------------ used by the kernel
def wrap(policy: Policy, cmd: list[str], work_dir: str, ro_paths: list[str], fixed_env: dict,
         base_env: dict) -> tuple[list[str], dict]:
    """-> (command, environment) that run `cmd` under the policy."""
    if policy.kind == "bwrap":
        env = dict(fixed_env, HOME="/tmp", PATH="/usr/bin:/bin", LANG="C.UTF-8")
        return (bwrap_argv(work_dir, ro_paths, env, python_dirs(cmd[0]), shutil.which("bwrap") or "bwrap") + cmd,
                {"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
    if policy.kind == "seatbelt":
        tmp = os.path.realpath(tempfile.gettempdir())
        profile = seatbelt_profile(work_dir, ro_paths, python_dirs(cmd[0]), os.path.expanduser("~"), ROOT, tmp)
        return ["/usr/bin/sandbox-exec", "-p", profile] + cmd, {**base_env, **fixed_env, "HOME": tmp, "TMPDIR": tmp}
    return cmd, {**base_env, **fixed_env}


# ------------------------------------------------------------------------------------------------ the harness's reads
def _check_parts(rel: str) -> list[str]:
    parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".")]
    if not parts or rel.startswith(("/", "\\")) or ":" in rel or ".." in parts:
        raise UnsafePath(f"{rel!r} is not a path inside the work folder")
    return parts


def open_in(root: str, rel: str) -> BinaryIO:
    """Open root/rel for reading, for the harness. Raises UnsafePath for links anywhere along the path, paths leading out of
    root, and anything but regular files, and FileNotFoundError if it isn't there."""
    parts = _check_parts(rel)
    if hasattr(os, "O_NOFOLLOW"):
        # Walk the folders one at a time without following links, so a link can't be swapped in after a check.
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
        try:
            for p in parts[:-1]:
                nxt = os.open(p, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                os.close(fd)
                fd = nxt
            ffd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
        except FileNotFoundError:
            raise
        except OSError as e:
            raise UnsafePath(f"{rel!r} is a link or not a regular file ({e.strerror})") from None
        finally:
            os.close(fd)
        if not stat.S_ISREG(os.fstat(ffd).st_mode):
            os.close(ffd)
            raise UnsafePath(f"{rel!r} is not a regular file")
        return os.fdopen(ffd, "rb")
    path = root
    is_junction = getattr(os.path, "isjunction", lambda _: False)
    for p in parts:
        path = os.path.join(path, p)
        if os.path.islink(path) or is_junction(path):
            raise UnsafePath(f"{rel!r} goes through a link")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    if not os.path.isfile(path):
        raise UnsafePath(f"{rel!r} is not a regular file")
    return open(path, "rb")


# ------------------------------------------------------------------------------------------------ proof it works
SELF_TEST_CODE = r'''
import json, os, subprocess, sys
def can_read(p):
    try:
        with open(p, "rb") as f:
            f.read(1)
        return True
    except OSError:
        return False
def can_write(p):
    try:
        with open(p, "wb") as f:
            f.write(b"x")
        os.remove(p)
        return True
    except OSError:
        return False
# a program started from here, because the kernel's own network guard is only inside its Python process
probe = "import socket; socket.create_connection(('1.1.1.1', 53), timeout=3); print('connected')"
try:
    net = "connected" in subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=30).stdout
except Exception:
    net = False
print(json.dumps({"read_home": can_read(HOME_DECOY), "read_project": can_read(PROJECT_DECOY),
                  "write_home": can_write(HOME_PROBE), "write_project": can_write(PROJECT_PROBE),
                  "network": net, "write_work": can_write("self_test.tmp")}))
'''
_tested: dict = {}


def self_test(kernel) -> str | None:
    """Start a throwaway sandboxed kernel and try what the sandbox must prevent, using decoy files. -> what is wrong, or None.
    Runs before anything is paid for, so a sandbox that silently doesn't work (or can't start) never reaches a real run."""
    key = (kernel.policy.kind, kernel.python)
    if key not in _tested:
        _tested[key] = _self_test(kernel)
    return _tested[key]


def _self_test(kernel) -> str | None:
    tok = uuid.uuid4().hex[:12]
    home_dir = os.path.join(os.path.expanduser("~"), ".wsp")
    os.makedirs(home_dir, exist_ok=True)
    names = {"HOME_DECOY": os.path.join(home_dir, f"decoy-{tok}"), "PROJECT_DECOY": os.path.join(ROOT, f".wsp-decoy-{tok}"),
             "HOME_PROBE": os.path.join(home_dir, f"probe-{tok}"), "PROJECT_PROBE": os.path.join(ROOT, f".wsp-probe-{tok}")}
    work = tempfile.mkdtemp(prefix="wsp-selftest-")
    k = None
    try:
        for n in ("HOME_DECOY", "PROJECT_DECOY"):
            with open(names[n], "w", encoding="utf-8") as f:
                f.write("decoy")
        k = type(kernel)(work, timeout=180, python=kernel.python, ro_paths=kernel.ro_paths, policy=kernel.policy)
        code = "\n".join(f"{n} = {p!r}" for n, p in names.items()) + SELF_TEST_CODE
        try:
            r = k.run(code)
        except RuntimeError as e:
            return _explain(str(e))
        if r["status"] != "ok" or r["error"]:
            return _explain(r["out"])
        res = json.loads(r["out"].strip().splitlines()[-1])
        bad = [k_ for k_ in ("read_home", "read_project", "write_home", "write_project", "network") if res[k_]]
        if bad:
            return "code inside it could still do this: " + ", ".join(bad)
        if not res["write_work"]:
            return "code inside it can't write to the work folder"
        return None
    finally:
        if k is not None:
            k.close()
        for p in names.values():
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(work, ignore_errors=True)


def _explain(text: str) -> str:
    tail = " ".join(text.split())[-500:]
    if sys.platform.startswith("linux") and any(s in text for s in ("uid map", "namespace", "Operation not permitted")):
        return f"{LINUX_NAMESPACES} ({tail})"
    return f"the sandboxed kernel could not start: {tail}"
