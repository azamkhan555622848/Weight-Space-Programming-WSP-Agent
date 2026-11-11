"""What a task plugs into the harness (research/harness_exp1_7.md section 5: task prompt, network presenter, sandbox
prelude, forward tools, hidden evaluation)."""
from __future__ import annotations

import os
import shutil

import numpy as np

from ..sandbox import UnsafePath, open_in


class ForwardError(Exception):
    """A forward-pass request that cannot be run; it is not counted against the budget."""


class TaskNotReady(Exception):
    """This machine cannot run the task yet (a missing download, no GPU). Raised before anything is paid for."""


class Task:
    name = "task"

    def __init__(self, cfg: dict, run_dir: str, work_dir: str):
        self.cfg, self.run_dir, self.work_dir = cfg, run_dir, work_dir
        self._out = work_dir          # where a pass writes its files (the staging folder while a pass runs)
        self._pass_edits = None       # the snapshot of the edits a running pass uses
        self.on_progress = None       # set by the harness: called with a short text while a pass runs
        self.goal = None              # set by the harness: the user's request, in an interactive session

    # ---------------------------------------------------------------------------------------------- the editor's view
    def task_prompt(self) -> str:
        """The first message: task, measurement, what counts, tools, how to work, rules, how the session ends."""
        raise NotImplementedError

    def kernel_setup(self) -> str:
        """Python run in the editor's kernel at every (re)start: the prelude (load / save_edit / ... helpers)."""
        raise NotImplementedError

    def prepare_work(self) -> None:
        """Create the work folder's initial content (notes.md, read-only inputs)."""
        os.makedirs(self.work_dir, exist_ok=True)
        notes = os.path.join(self.work_dir, "notes.md")
        if not os.path.exists(notes):
            open(notes, "w", encoding="utf-8").write("# Notebook\n\n(empty)\n")

    def preflight(self) -> None:
        """Raise TaskNotReady if this machine cannot run the task. Called before the first model call."""

    def kernel_ro_paths(self) -> list[str]:
        """Files outside the work folder that the kernel's prelude reads; the sandbox makes them readable at the same paths."""
        return []

    # ---------------------------------------------------------------------------------------------- forward tool
    def forward(self, req: dict, k: int, commit: bool = True, edits: str | None = None) -> tuple[dict, str]:
        """Run forward pass k. -> (short JSON summary for the message, where the full results are).
        Raise ForwardError for an invalid request (not counted).

        A pass writes into a staging folder, and its files reach the work folder only once it has finished: a pass
        interrupted part-way is not counted, so it must not leave results the editor could use. With commit=False the
        files stay staged until stage_commit(), so the caller can record the pass as counted first.

        edits: a snapshot_edits() folder the caller already took; the pass uses exactly those edits."""
        try:
            self._pass_edits = edits or self.snapshot_edits(req)
        except UnsafePath as e:
            raise ForwardError(str(e)) from None
        self.stage_begin()
        try:
            res = self.run_pass(req, k)
        except BaseException:
            self.stage_abort()
            raise
        finally:
            self._pass_edits = None
        if commit:
            self.stage_commit()
        return res

    def run_pass(self, req: dict, k: int) -> tuple[dict, str]:
        """The pass itself: writes its files under self._out."""
        raise NotImplementedError

    def stage_dir(self) -> str:
        return os.path.join(self.run_dir, "staging")

    def stage_begin(self) -> None:
        """Start a pass with an empty staging folder (a leftover from a killed pass is dropped)."""
        stage = self.stage_dir()
        shutil.rmtree(stage, ignore_errors=True)
        os.makedirs(stage)
        self._out = stage

    def stage_abort(self) -> None:
        shutil.rmtree(self.stage_dir(), ignore_errors=True)
        self._out = self.work_dir

    def stage_commit(self) -> None:
        """Move the staged files into the work folder. Safe to repeat after a crash part-way through.

        os.replace does not follow a link at the destination, so a link the editor's code left in the work folder cannot
        redirect these writes: a link at a file's name is replaced, and one at a folder's name makes the commit fail."""
        stage = self.stage_dir()
        if os.path.isdir(stage):
            for name in sorted(os.listdir(stage)):
                dst = os.path.join(self.work_dir, name)
                if os.path.isdir(dst) and not os.path.islink(dst):
                    shutil.rmtree(dst)
                elif os.path.islink(dst) or os.path.exists(dst):
                    os.remove(dst)
                os.replace(os.path.join(stage, name), dst)
            shutil.rmtree(stage, ignore_errors=True)
        self._out = self.work_dir

    # ---------------------------------------------------------------------------------------------- edits
    def edit_dir(self) -> str:
        return os.path.join(self.work_dir, "edited")

    def edit_names(self) -> list[str]:
        """The tensors currently edited (names only; nothing is read)."""
        d = self.edit_dir()
        if os.path.islink(d) or not os.path.isdir(d):
            return []
        return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npy"))

    def set_mode(self, mode: str) -> None:
        """The user switched between build and plan mode (interactive sessions)."""

    def snapshot_edits(self, req: dict | None = None) -> str:
        """Copy the edit files out of the editor's reach, and return that folder. What gets hashed and loaded is then the
        same thing, and a link left in edited/ can't make the harness read another file. Raises UnsafePath for a link.
        `req` is the forward request about to run, for tasks whose passes can use other folders of edits as well."""
        snap = os.path.join(self.run_dir, "edit_snapshot")
        shutil.rmtree(snap, ignore_errors=True)
        os.makedirs(snap)
        for name in self.edit_names():
            with open_in(self.edit_dir(), name + ".npy") as src, open(os.path.join(snap, name + ".npy"), "wb") as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
        return snap

    def edit_arrays(self) -> dict:
        """name -> array of the current edits: the running pass's snapshot, or a new one at the end of the session."""
        d = self._pass_edits or self.snapshot_edits()
        return {f[:-4]: np.load(os.path.join(d, f), allow_pickle=False) for f in sorted(os.listdir(d)) if f.endswith(".npy")}

    def submit(self, out_dir: str) -> dict:
        """Export the final edit (the tensors in the edit folder when the session ends)."""
        raise NotImplementedError

    def close(self) -> None:
        shutil.rmtree(os.path.join(self.run_dir, "edit_snapshot"), ignore_errors=True)
