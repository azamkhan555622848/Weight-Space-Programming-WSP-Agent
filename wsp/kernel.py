"""Persistent Python kernel for the editor (parent side).

Runs wsp/kernel_worker.py in a child process (`python -E -P`, CPU only, cwd = the run's work folder), sends it one code
block at a time and returns its printed output. A run that exceeds `timeout` seconds kills the child; the next run
starts a fresh one (variables lost) and the editor is told so. Output is cut at `max_out` characters.

Where the system has one, the child runs in a sandbox (wsp/sandbox.py). The GUARDS prelude adds the experiment's rules (no
autograd, optimizers or network) inside the Python process, and every code block is recorded in tool_calls.jsonl.
"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time

from . import sandbox
from .procs import group_kwargs, kill_tree

WORKER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernel_worker.py")

# What the kernel may inherit from the environment: no keys or tokens. The first line is Windows, the second POSIX.
ENV_KEEP = {"SYSTEMROOT", "PATH", "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "LOCALAPPDATA", "APPDATA", "WINDIR",
            "COMSPEC", "PATHEXT",
            "HOME", "LANG", "LC_ALL", "TMPDIR", "LD_LIBRARY_PATH"}

GUARDS = r'''
import socket as _socket
try:
    import torch as _torch
    _torch.set_grad_enabled(False)
    def _no_grad_rule(*a, **k):
        raise RuntimeError("gradients are disabled in this kernel (rule: no backward passes, autograd or optimizers)")
    _torch.Tensor.backward = _no_grad_rule
    _torch.autograd.backward = _no_grad_rule
    _torch.autograd.grad = _no_grad_rule
    _orig_sge = _torch.set_grad_enabled
    class _NoGradOnly:
        def __init__(self, mode=False):
            if mode:
                _no_grad_rule()
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def __call__(self, f): return f
    _torch.set_grad_enabled = _NoGradOnly
    _torch.enable_grad = lambda *a, **k: _no_grad_rule()
    _Opt = _torch.optim.Optimizer
    for _n in dir(_torch.optim):
        _c = getattr(_torch.optim, _n)
        if isinstance(_c, type) and issubclass(_c, _Opt):
            setattr(_torch.optim, _n, _no_grad_rule)
    del _Opt
except ImportError:
    pass
# network: block at the method level (replacing the socket class breaks libraries that subclass it)
def _blocked(*a, **k):
    raise PermissionError("network access is disabled in this kernel")
_socket.socket.connect = _blocked
_socket.socket.connect_ex = _blocked
_socket.socket.sendto = _blocked
_socket.create_connection = _blocked
_socket.getaddrinfo = _blocked
'''


class Kernel:
    def __init__(self, work_dir: str, setup_code: str = "", timeout: float = 900.0, max_out: int = 10_000,
                 threads: int = 8, python: str | None = None, ro_paths: list[str] | tuple = (),
                 policy: sandbox.Policy | None = None):
        self.work_dir = os.path.realpath(work_dir)
        self.ro_paths = [os.path.realpath(p) for p in ro_paths]
        self.policy = policy if policy is not None else sandbox.resolve_policy()
        self.setup_code = setup_code
        self.timeout, self.max_out, self.threads = timeout, max_out, threads
        self.python = python or sys.executable
        self.proc: subprocess.Popen | None = None
        self._q: queue.Queue = queue.Queue()
        self.restarts = 0
        self._stderr = None

    @property
    def sandboxed(self) -> bool:
        return self.policy.kind != "none"

    def self_test(self) -> str | None:
        """Prove the sandbox works here. -> what is wrong, or None."""
        return sandbox.self_test(self)

    def check_setup(self) -> None:
        """Start the kernel with its real setup code, then stop it: raises RuntimeError if it can't run in the sandbox."""
        self._start()
        self.kill()

    # ------------------------------------------------------------------------------------------------ process
    def _start(self):
        base = {k: v for k, v in os.environ.items() if k.upper() in ENV_KEEP}
        fixed = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": str(self.threads),
                 "MKL_NUM_THREADS": str(self.threads), "PYTHONIOENCODING": "utf-8",
                 "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1"}
        # -E: ignore PYTHON* variables; -P: do not put the script's folder or the cwd on sys.path. (Not -I: it would
        # also drop the per-user site-packages, where torch may be installed.)
        cmd = [self.python, "-E", "-P", "-u", WORKER]
        argv, env = sandbox.wrap(self.policy, cmd, self.work_dir, self.ro_paths + [WORKER], fixed, base)
        # stderr goes to a file: a pipe nobody reads would block the kernel once it fills up. A process group of its own
        # lets kill() stop programs the code started as well.
        self._stderr = tempfile.TemporaryFile()
        self.proc = subprocess.Popen(argv, cwd=self.work_dir, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self._stderr, text=True, encoding="utf-8", errors="replace", bufsize=1,
                                     **group_kwargs())
        self._q = queue.Queue()
        threading.Thread(target=self._reader, args=(self.proc, self._q), daemon=True).start()
        out = self._send({"setup": GUARDS + "\n" + self.setup_code}, timeout=600)
        if out[1]:
            raise RuntimeError("kernel setup failed:\n" + out[0])

    @staticmethod
    def _reader(proc, q):
        for line in proc.stdout:
            q.put(line)
        q.put(None)

    def _send(self, msg: dict, timeout: float) -> tuple[str, bool, str]:
        """-> (output, error, status) with status 'ok', 'timeout' or 'died'."""
        assert self.proc is not None
        try:
            self.proc.stdin.write(json.dumps(msg) + "\n")
            self.proc.stdin.flush()
        except OSError:                # it already exited, for example the sandbox could not start
            err = self._stderr_tail()
            self.kill()
            return err, True, "died"
        try:
            line = self._q.get(timeout=timeout)
        except queue.Empty:
            self.kill()
            return "", True, "timeout"
        if line is None:
            err = self._stderr_tail()
            self.kill()
            return err, True, "died"
        try:
            r = json.loads(line)
            return r["out"], r["error"], "ok"
        except (ValueError, KeyError, TypeError):       # the editor's code can reach the protocol stream; don't crash on it
            self.kill()
            return "[the kernel's reply could not be read]", True, "died"

    def _stderr_tail(self) -> str:
        if self._stderr is None:
            return ""
        try:
            self._stderr.seek(max(0, self._stderr.seek(0, 2) - 3000))
            return self._stderr.read().decode("utf-8", "replace")
        except (OSError, ValueError):
            return ""

    def kill(self):
        """Stop the kernel and any programs its code started."""
        if self.proc is not None:
            kill_tree(self.proc)
            self.proc = None
        if self._stderr is not None:
            self._stderr.close()
            self._stderr = None

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    # ------------------------------------------------------------------------------------------------ API
    def run(self, code: str) -> dict:
        """Run one code block. -> {"out": str (cut at max_out), "error": bool, "status": ok|timeout|died,
        "restarted": bool (variables were lost before this run), "s": seconds}."""
        restarted = False
        if not self.alive():
            restarted = self.restarts > 0 or self.proc is not None
            self._start()
        t0 = time.time()
        out, err, status = self._send({"code": code}, timeout=self.timeout)
        if status == "timeout":
            self.restarts += 1
            out = f"[timed out after {self.timeout:.0f} s: the kernel was stopped; its variables are lost]"
        elif status == "died":
            self.restarts += 1
            out = (out + "\n" if out else "") + "[the kernel process died (for example out of memory); its variables " \
                                                "are lost]"
        if len(out) > self.max_out:
            out = out[:self.max_out] + f"\n[output cut at {self.max_out:,} characters]"
        return {"out": out, "error": err, "status": status, "restarted": restarted, "s": round(time.time() - t0, 2)}

    def close(self):
        self.kill()
