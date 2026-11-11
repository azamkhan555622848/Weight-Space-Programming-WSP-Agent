"""Starting a program so that all of it can be stopped later, including anything it started itself."""
from __future__ import annotations

import os
import signal
import subprocess


def group_kwargs() -> dict:
    """Popen arguments that give the program a process group of its own."""
    if os.name == "posix":
        return {"start_new_session": True}
    return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}


def kill_tree(proc: subprocess.Popen) -> None:
    """Stop a program started with group_kwargs, and everything it started."""
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True, timeout=30)
            proc.kill()
    except (OSError, subprocess.SubprocessError):
        pass
    try:
        proc.wait(timeout=30)
    except subprocess.SubprocessError:
        pass
