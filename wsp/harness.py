"""The WSP agent loop: an LLM editor reads and edits network weights through a Python kernel and a counted forward-pass
tool, with no training. A reimplementation of the unreleased harness of "Sparks of Weight-Space Programming" from its
logs (research/harness_exp8.md, research/harness_exp1_7.md), with the bugs those logs reveal fixed.

One turn = one editor reply. A reply holds exactly one fenced block:
  ```python   -> run in the persistent kernel (counted against max_runs)
  ```forward  -> JSON request to the task's forward-pass tool (counted against the pass budget; invalid, failed or
                 duplicate requests are not counted)
or no block. There are two protocols for what a reply without a block means:

  the paper's (default)  one fixed task. While forward passes remain the session continues (the editor is told so); once they
                         are all used, a reply without a block is the final answer and the session ends.
  interactive            a conversation. A reply without a block is the editor talking to the user, and the session waits
                         for their answer. Messages they send meanwhile reach the editor with the next result; they can
                         interrupt it, change the budget or the mode, and be asked before each block is run. The editor may
                         also keep a todo list in a ```todo block next to its action block.

Every `handoff_every` turns (or `handoff_tokens` of context) a new conversation starts with a handoff message (task, counters,
edited tensors, files, notebook, last result); the kernel keeps running. After a harness restart the same handoff is used and
the kernel is fresh, which the editor is told. Text files dropped into <run>/notices/ are delivered once, as experimenter notes.

Run folder: config.json, state.json, status.json (what it's doing now), run.lock, loop.log, events.jsonl and live.txt and
inbox/ (wsp/events.py), tool_calls.jsonl, explanations.md, final_answer.md, turns/tNNN/{prompt.txt, answer.md, run.json},
requests/fwd_<k>.json / .done.json, work/ (the kernel's folder), submission/ (the exported edit), conversations/ (API backend),
notices/. A file named `stop` asks the run to stop at the next safe point; `wsp stop` writes it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import traceback

from . import events as E
from . import messages as M
from .kernel import Kernel
from .llm import Backend, Interrupted, Reply, StopRequested
from .runs import RunLock, atomic_write_json
from .sandbox import NONE, UnsafePath, open_in
from .tasks.base import ForwardError, Task, TaskNotReady

FENCE_OPEN = re.compile(r"^```([A-Za-z_]+)\s*$")
FENCE_CLOSE = re.compile(r"^```\s*$")
TODO_BLOCK = re.compile(r"^```todo[ \t]*\n.*?(?:^```[ \t]*\n?|\Z)", re.S | re.M)


def parse_blocks(text: str) -> tuple[list[tuple[str, str]], str]:
    """Fenced blocks of a reply: ([(kind, body)], text before the first block). A block opens with a line '```kind'
    and closes with a line '```'; an unclosed block runs to the end of the reply."""
    blocks, prose, inside, kind, body = [], [], False, None, []
    for line in text.splitlines():
        if not inside:
            m = FENCE_OPEN.match(line.strip())
            if m:
                inside, kind, body = True, m.group(1).lower(), []
            elif not blocks:
                prose.append(line)
        elif FENCE_CLOSE.match(line.strip()):
            blocks.append((kind, "\n".join(body)))
            inside = False
        else:
            body.append(line)
    if inside:
        blocks.append((kind, "\n".join(body)))
    return blocks, "\n".join(prose).strip()


class Harness:
    DEFAULTS = {"budget": 10, "max_runs": 300, "max_turns": 400, "handoff_every": 40, "explain": True,
                "explain_max_sendbacks": 2, "kernel_timeout": 900, "kernel_max_out": 10_000, "kernel_threads": 8,
                "max_failed_passes": 5, "accept_no_sandbox": False,
                # interactive sessions
                "interactive": False, "goal": None, "todo": False, "mode": "build", "ask_python": False, "ask_forward": False,
                "idle_timeout": 1800, "handoff_tokens": 0}
    # what the user may change while a session runs (an inbox item {"type": "set", "key", "value"})
    SETTINGS = ("budget", "mode", "ask_python", "ask_forward", "editor_model", "editor_effort")

    def __init__(self, task: Task, backend: Backend, run_dir: str, cfg: dict | None = None, make_backend=None):
        self.task, self.llm, self.run_dir = task, backend, run_dir
        self.cfg = dict(self.DEFAULTS, **(cfg or {}))
        self.interactive = bool(self.cfg["interactive"])
        self.make_backend = make_backend          # (model, effort) -> Backend, for a change of editor during a session
        for d in ("turns", "requests", "notices", "inbox"):
            os.makedirs(os.path.join(run_dir, d), exist_ok=True)
        self.state_path = os.path.join(run_dir, "state.json")
        # the kernel cuts nothing itself: the harness applies the paper's output pipeline (messages.format_output)
        self.kernel = Kernel(task.work_dir, setup_code=task.kernel_setup(), timeout=self.cfg["kernel_timeout"],
                             max_out=10 ** 8, threads=self.cfg["kernel_threads"], ro_paths=task.kernel_ro_paths())
        self.st: dict = {}
        self._activity = (None, 0.0)
        self._answers: dict = {}
        self.ev = E.EventLog(run_dir)
        self.live = E.LiveText(run_dir)
        self.llm.abort_check = self.abort_kind
        self.task.on_progress = self._on_progress

    # ------------------------------------------------------------------------------------------------ bookkeeping
    def log(self, s: str):
        line = time.strftime("%Y-%m-%d %H:%M:%S ") + s
        print(line, flush=True)
        with open(os.path.join(self.run_dir, "loop.log"), "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def emit(self, type: str, **data):
        return self.ev.emit(type, **data)

    def save_state(self):
        self.st["llm"] = self.llm.state()
        atomic_write_json(self.state_path, self.st)

    def set_activity(self, kind: str, detail: str = ""):
        """What the run is doing now, for `wsp status` and the interface. `since` stays put while only the detail changes."""
        if kind != self._activity[0]:
            self._activity = (kind, time.time())
        s = self.st
        atomic_write_json(os.path.join(self.run_dir, "status.json"),
                          {"activity": kind, "detail": detail, "since": self._activity[1], "pid": os.getpid(),
                           "turn": s.get("turn", 0), "passes": s.get("passes", 0), "runs": s.get("runs", 0)})

    def _on_progress(self, detail: str):
        """Called by the task while a pass runs: shows progress, and lets the user interrupt between batches."""
        self.set_activity("forward", detail)
        if self.interactive and self.interrupt_requested():
            raise Interrupted

    def record(self, rec: dict):
        with open(os.path.join(self.run_dir, "tool_calls.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": round(time.time(), 1), **rec}, ensure_ascii=False) + "\n")

    def explanation(self, turn: int, kind: str, text: str):
        if self.cfg["explain"]:
            with open(os.path.join(self.run_dir, "explanations.md"), "a", encoding="utf-8") as f:
                f.write(f"## Turn {turn} ({kind}; passes used {self.st['passes']}/{self.cfg['budget']})\n\n"
                        f"{text or '(no explanation)'}\n\n")

    @staticmethod
    def edit_hash(d: str) -> tuple[list[str], str]:
        """(sorted edited tensor names, 10-hex content hash) of a snapshot of the edits: the .npy files of the snapshot folder
        are the current edit; files in its subfolders (other candidates a pass compares) count towards the hash only."""
        names = sorted(f for f in os.listdir(d) if f.endswith(".npy"))
        extra = sorted(os.path.join(os.path.relpath(b, d), f).replace("\\", "/") for b, _, fs in os.walk(d) if b != d
                       for f in fs if f.endswith(".npy"))
        if not names and not extra:
            return [], "none"
        h = hashlib.sha256()
        for f in names + extra:
            h.update(f.encode())
            with open(os.path.join(d, f), "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 22), b""):
                    h.update(chunk)
        return [n[:-4] for n in names], h.hexdigest()[:10]

    # ------------------------------------------------------------------------------------------------ messages
    def task_text(self) -> str:
        """The task message, with the EXPLAIN paragraph between HOW TO WORK and RULES (as the paper's message.txt)."""
        p = self.task.task_prompt()
        if self.cfg["explain"] and "\n\nRULES." in p:
            p = p.replace("\n\nRULES.", "\n\n" + M.EXPLAIN_PARAGRAPH + "\n\nRULES.", 1)
        return p

    def compose(self, body: str) -> str:
        """[notices] + [pending notes] + [the user's messages] + body (paper 2.15). Notes are queued so a restart never loses
        them. A user who was being waited for is simply answered; one who wrote while the editor worked is marked as such."""
        s, pre = self.st, ""
        nd = os.path.join(self.run_dir, "notices")
        for f in sorted(os.listdir(nd)):
            if f.endswith(".txt") and f not in s["notices"]:
                pre += M.NOTICE.format(text=open(os.path.join(nd, f), encoding="utf-8").read().strip())
                s["notices"].append(f)
        if s["pending_notes"]:
            pre += "\n".join(s["pending_notes"]) + "\n"
            s["pending_notes"] = []
        users, body = s.get("pending_user") or [], body or ""
        if users:
            s["user_log"] = (s.get("user_log", []) + users)[-30:]
            s["pending_user"] = []
            if s.get("phase") != "waiting":
                pre += "".join(M.USER_WHILE_WORKING.format(text=u) for u in users)
            elif body:          # they interrupted before seeing this result: the result first, then what they said
                body += "\n\n[message from the user, who interrupted you here]\n" + "\n\n".join(users)
            else:
                body = "\n\n".join(users)
        return pre + body

    def handoff_message(self, last: str) -> str:
        s = self.st
        names = self.task.edit_names()
        wd = self.task.work_dir
        files = sorted(os.listdir(wd))[:60] if os.path.isdir(wd) else []
        common = dict(task=self.task_text(), replies=s["turn"], passes=s["passes"], budget=self.cfg["budget"],
                      runs=s["runs"], max_runs=self.cfg["max_runs"],
                      kernel=M.KERNEL_ALIVE if self.kernel.alive() else M.KERNEL_FRESH,
                      edited=repr(names) if names else "none", files=repr(files),
                      notes=self.read_notes().encode("ascii", "replace").decode(), last=last)
        if not self.interactive:
            return M.HANDOFF.format(**common)
        later = s.get("user_log", [])[1:]
        return M.HANDOFF_INTERACTIVE.format(
            **common, todo=E.todo_text(s.get("todo") or []),
            conversation="\n".join(f"{i + 1}. {u[:2000]}" for i, u in enumerate(later[-10:])) or "(none)",
            last_message=s.get("last_message") or "(none yet)")

    def read_notes(self) -> str:
        """The end of the editor's notebook. Sent to the provider, so a link planted at notes.md must not be followed."""
        try:
            with open_in(self.task.work_dir, "notes.md") as f:
                f.seek(max(0, f.seek(0, 2) - 64_000))
                return f.read().decode("utf-8", "replace")[-8000:]
        except (FileNotFoundError, UnsafePath):
            return ""

    # ------------------------------------------------------------------------------------------------ safety stops
    def check_ready(self):
        """Before anything is paid for: the model's code needs a sandbox or the user's explicit acceptance, and this
        machine must be able to run the task."""
        accepted = self.cfg["accept_no_sandbox"]
        if self.kernel.sandboxed:
            why = self.kernel.self_test()
            if why and not accepted:
                raise SystemExit(f"Refusing to start: the sandbox failed its self-test. {why} `wsp doctor` shows how to "
                                 "fix it; --accept-no-sandbox runs without a sandbox instead.")
            if why:
                self.log(f"sandbox self-test failed, continuing without a sandbox as accepted: {why}")
                self.kernel.policy = NONE
        if not self.kernel.sandboxed and not accepted:
            raise SystemExit("Refusing to start: the model's code would run on this machine without a sandbox. "
                             f"{self.kernel.policy.note} Accept that explicitly with --accept-no-sandbox, or see `wsp "
                             "doctor` for how to get a sandbox.")
        try:
            self.task.preflight()
        except TaskNotReady as e:
            raise SystemExit(f"{self.task.name} can't run here: {e}") from None

    def stop_requested(self) -> bool:
        return os.path.exists(os.path.join(self.run_dir, "stop"))

    def interrupt_requested(self) -> bool:
        return os.path.exists(os.path.join(self.run_dir, "interrupt"))

    def _remove(self, name: str):
        try:
            os.remove(os.path.join(self.run_dir, name))
        except OSError:
            pass

    def abort_kind(self) -> str | None:
        """For the backends, which check this while they wait: 'stop', 'interrupt' or None."""
        if self.stop_requested():
            return "stop"
        if self.interactive and self.interrupt_requested():
            return "interrupt"
        return None

    def stop(self, reason: str = "requested"):
        """Stop as the user asked (or after waiting for them for a long time). The run can be resumed."""
        self._remove("stop")
        self.live.end()
        self.save_state()
        self.emit("stopped", reason=reason)
        self.log(f"stopped ({reason}) after turn {self.st['turn']}; re-run the same command to resume")
        self.kernel.close()
        raise SystemExit(0)

    def halt(self, reason: str, final: bool):
        """Stop the run now. final: its results can't be trusted, so it refuses to resume."""
        if final:
            self.st["halted"] = reason
        self.live.end()
        self.save_state()
        self.emit("halt", reason=reason, final=final)
        self.log(reason)
        self.kernel.close()
        raise SystemExit(reason)

    # ------------------------------------------------------------------------------------------------ the user
    def drain_inbox(self):
        """Take what people sent: their messages, their answers to permission requests, and changed settings."""
        s = self.st
        for item in E.take(self.run_dir):
            kind = item.get("type")
            text = str(item.get("text") or "").strip()
            if kind == "message" and text:
                s["pending_user"].append(text)
                self.emit("user", text=text)
            elif kind == "set":
                self.apply_setting(item.get("key"), item.get("value"))
            elif kind == "permission":
                self._answers[item.get("id")] = item
            elif kind == "compact":
                s["fresh_conversation"] = True
                self.emit("note", text="The conversation restarts from a summary at the next step.")

    def apply_setting(self, key, value):
        s = self.st
        note = None
        try:
            if key == "budget":
                value = max(int(value), s["passes"])
                self.cfg["budget"] = s["budget"] = value
                note = M.BUDGET_NOTE.format(budget=value, left=value - s["passes"])
            elif key == "mode":
                if value not in M.MODE_NOTE or value == self.cfg["mode"]:
                    return
                self.cfg["mode"] = value
                self.task.set_mode(value)
                note = M.MODE_NOTE[value]
            elif key in ("ask_python", "ask_forward"):
                value = bool(value)
                self.cfg[key] = value
            elif key in ("editor_model", "editor_effort") and self.make_backend and value:
                cur = {"editor_model": s["settings"].get("editor_model"), "editor_effort": s["settings"].get("editor_effort")}
                if cur.get(key) == value:
                    return
                cur[key] = value
                self.llm = self.make_backend(cur["editor_model"], cur["editor_effort"])
                self.llm.abort_check = self.abort_kind
                s["fresh_conversation"] = True
                note = M.EDITOR_NOTE.format(model=cur["editor_model"], effort=cur["editor_effort"])
            else:
                return
        except Exception as e:  # noqa: BLE001 - a bad setting must not end the session
            self.emit("error", text=f"Could not change {key}: {e}")
            return
        s["settings"][key] = value
        if note:
            s["pending_notes"].append(note)
        self.emit("setting", key=key, value=value)
        self.log(f"setting {key} = {value}")

    def wait_for_user(self):
        """The editor has spoken; nothing happens until the user answers. After a long silence the process stops, to free the
        machine: the session resumes when they write again."""
        s = self.st
        self.set_activity("waiting")
        self.save_state()
        t0 = time.time()
        while not s["pending_user"]:
            if self.stop_requested():
                self.stop()
            self._remove("interrupt")
            if time.time() - t0 > self.cfg["idle_timeout"]:
                self.stop("idle")
            time.sleep(0.25)
            self.drain_inbox()

    def permitted(self, kind: str, body: str) -> str | None:
        """Ask the user before running a block, if they want to be asked. -> None if it may run, else the message for the editor."""
        if not self.cfg["ask_" + kind]:
            return None
        s = self.st
        pid = f"p{s['turn']}"
        self.emit("permission_request", id=pid, kind=kind, body=E.clip(body, 8000))
        self.set_activity("permission", kind)
        while pid not in self._answers:
            if self.stop_requested():
                s["next"] = M.INTERRUPTED_BEFORE_RUN.format(kind=kind)
                self.stop()
            if self.interrupt_requested():
                self._answers[pid] = {"decision": "reject"}
                break
            time.sleep(0.25)
            self.drain_inbox()
        answer = self._answers.pop(pid)
        decision = answer.get("decision") if answer.get("decision") in ("once", "always") else "reject"
        self.emit("permission", id=pid, decision=decision)
        if decision == "always":
            self.apply_setting("ask_" + kind, False)
        if decision != "reject":
            return None
        feedback = str(answer.get("feedback") or "").strip()
        return M.DECLINED.format(kind=kind, feedback=f" They said: {feedback}" if feedback else "")

    # ------------------------------------------------------------------------------------------------ main loop
    def run(self) -> dict:
        lock = RunLock(self.run_dir)
        if not lock.acquire():
            raise SystemExit(f"This run is already running: {self.run_dir}")
        try:
            return self._run()
        finally:
            self.live.end()
            lock.release()

    def _run(self) -> dict:
        self.check_ready()
        self._remove("stop")                 # left from a stop that came after the run had ended
        self._remove("interrupt")
        if os.path.exists(self.state_path):
            self.st = json.load(open(self.state_path, encoding="utf-8"))
            if self.st.get("done"):
                self.log("run already finished")
                return self.st
            if self.st.get("halted"):
                raise SystemExit(f"This run was stopped and can't be resumed; start a new one. {self.st['halted']}")
            for k, v in (("pending_user", []), ("user_log", []), ("phase", "working"), ("settings", {}), ("todo", [])):
                self.st.setdefault(k, v)
            if self.st.get("commit_pending"):      # killed after the pass was counted, before its files were moved
                self.task.stage_commit()
                self.st["commit_pending"] = False
            self.task.goal = self.st.get("goal")
            self.st["fwd_fail_streak"] = 0
            self.st["restarts"] = self.st.get("restarts", 0) + 1
            self.st["pending_notes"] = [M.RESTART_NOTE] + self.st.get("pending_notes", [])
            self.st["fresh_conversation"] = True
            self.emit("resumed")
            self.log(f"harness restart {self.st['restarts']}: resuming at turn {self.st['turn'] + 1} with a handoff")
        else:
            goal = self.cfg["goal"]
            if self.interactive and not (goal and str(goal).strip()):
                raise SystemExit("An interactive session starts from the user's request (--goal).")
            self.task.goal = goal
            self.task.prepare_work()
            self.st = {"task": self.task.name, "budget": self.cfg["budget"], "turn": 0, "runs": 0, "passes": 0,
                       "requests": 0, "handoffs": 0, "since_handoff": 0, "restarts": 0, "nudges": 0,
                       "unexplained": 0, "unexplained_streak": 0, "flags": 0, "flag_streak": 0, "req_keys": {},
                       "notices": [], "pending_notes": [], "done": False, "tool_free": True, "models_seen": [],
                       "tokens": {"input": 0, "output": 0, "cache_read": 0, "cache_write": 0},
                       "next": None, "fresh_conversation": False,
                       "interactive": self.interactive, "goal": goal, "phase": "working", "pending_user": [],
                       "user_log": [goal] if goal else [], "todo": [], "last_message": "",
                       "settings": {k: self.cfg[k] for k in ("editor_model", "editor_effort") if self.cfg.get(k)}}
            json.dump(self.cfg, open(os.path.join(self.run_dir, "config.json"), "w"), indent=1)
            open(os.path.join(self.run_dir, "message.txt"), "w", encoding="utf-8").write(self.task_text())
            self.llm.new_conversation()
            self.st["next"] = self.task_text()
            self.emit("session", task=self.task.name, interactive=self.interactive, budget=self.cfg["budget"])
            if goal:
                self.emit("user", text=str(goal))
            self.log(f"task {self.task.name}: budget {self.cfg['budget']} passes, {self.cfg['max_runs']} python runs")
        self.task.set_mode(self.cfg["mode"])
        self.st["sandbox"] = self.kernel.policy.kind
        self.save_state()
        if self.kernel.sandboxed:
            try:
                self.kernel.check_setup()
            except RuntimeError as e:
                raise SystemExit(f"The kernel can't start in the sandbox with this task's files: {e}") from None

        s = self.st
        while not s["done"]:
            self.drain_inbox()
            if self.stop_requested():
                self.stop()
            if self.interrupt_requested():
                self._remove("interrupt")
                if self.interactive and s["phase"] != "waiting":
                    s["phase"] = "waiting"
                    self.emit("interrupted")
                    self.log(f"interrupted by the user after turn {s['turn']}")
            if self.interactive and s["phase"] == "waiting" and not s["pending_user"]:
                self.wait_for_user()
                continue
            if s["turn"] >= self.cfg["max_turns"]:
                self.log(f"stopping: max_turns {self.cfg['max_turns']} reached")
                self.emit("note", text=f"The session reached its limit of {self.cfg['max_turns']} turns.")
                break
            self.turn()
            self.save_state()
        self.finish()
        return self.st

    def turn(self):
        s = self.st
        body = pre_handoff = self.compose(s["next"])
        s["phase"] = "working"
        handoff = s.get("fresh_conversation") or (s["since_handoff"] >= self.cfg["handoff_every"] and s["turn"] > 0)
        if handoff:
            s["handoffs"] += s["turn"] > 0
            self.log(f"turn {s['turn'] + 1}: new conversation (handoff {s['handoffs']})")
            self.llm.new_conversation()
            body = self.handoff_message(body)
            s["since_handoff"], s["fresh_conversation"] = 0, False
            if s["turn"] > 0:
                self.emit("handoff", n=s["handoffs"])
        s["turn"] += 1
        tdir = os.path.join(self.run_dir, "turns", f"t{s['turn']:03d}")
        os.makedirs(tdir, exist_ok=True)
        open(os.path.join(tdir, "prompt.txt"), "w", encoding="utf-8").write(body)
        self.set_activity("model", self.llm.name)
        self.live.start()
        try:
            r: Reply = self.llm.send(body, on_delta=self.live.write)
        except (StopRequested, Interrupted) as e:      # nothing was delivered: the same message goes out next time
            s["turn"] -= 1
            s["pending_notes"], s["next"] = [], pre_handoff
            s["fresh_conversation"] = bool(handoff)   # the new conversation is still empty
            if isinstance(e, StopRequested):
                self.stop()
            return                                     # the interrupt flag is still up: the main loop waits for the user
        finally:
            self.live.end()
        open(os.path.join(tdir, "answer.md"), "w", encoding="utf-8").write(r.text)
        json.dump({"ok": r.ok, "refused": r.refused, "stop_reason": r.stop_reason, "model": r.model, "usage": r.usage,
                   "audit": r.audit, "wall_s": r.wall_s, "waits": r.waits, "error": r.error},
                  open(os.path.join(tdir, "run.json"), "w"), indent=1)
        if not r.ok:
            # nothing was delivered: the same message is sent again when the run is resumed (with a handoff)
            s["turn"] -= 1
            s["pending_notes"], s["next"] = [], pre_handoff
            s["lost_streak"] = s.get("lost_streak", 0) + 1 if r.context_lost else 0
            if r.context_lost and s["lost_streak"] <= 3:     # the provider lost the session: continue in a new one
                s["fresh_conversation"] = True
                self.log(f"turn {s['turn'] + 1}: the conversation was lost ({r.error}); continuing with a handoff")
                return
            self.save_state()
            self.emit("error", text=f"The model call failed: {r.error}")
            self.log(f"turn {s['turn'] + 1}: model call failed ({r.error}); stopping. Re-run the same command to resume.")
            raise SystemExit(f"model call failed: {r.error}")
        s["lost_streak"] = 0
        if r.context_lost:                       # this reply stands, but the next turn starts from a handoff
            s["fresh_conversation"] = True
            self.log(f"turn {s['turn']}: the conversation can't be relied on any more; the next turn starts a new one")
        s["since_handoff"] += 1
        for a, b in (("input_tokens", "input"), ("output_tokens", "output"), ("cache_read_input_tokens", "cache_read"),
                     ("cache_creation_input_tokens", "cache_write")):
            s["tokens"][b] += int(r.usage.get(a) or 0)
        context = sum(int(r.usage.get(a) or 0) for a in ("input_tokens", "cache_read_input_tokens", "cache_creation_input_tokens"))
        if self.cfg["handoff_tokens"] and context > self.cfg["handoff_tokens"]:
            s["fresh_conversation"] = True       # the conversation has grown long: carry on from the notebook
        if r.model and r.model not in s["models_seen"]:
            s["models_seen"].append(r.model)
        # The editor must have no tools of its own: the backend has to confirm that for every reply, and nothing in a
        # reply it can't confirm is run. This is not resumable, because the rest of the run would not be valid.
        if r.audit.get("tool_free") is not True:
            s["tool_free"] = False
            self.halt(f"Stopped at turn {s['turn']}: the {self.llm.name} backend could not confirm the model's reply was "
                      f"free of tools (see turns/t{s['turn']:03d}/run.json), so nothing in it was run.", final=True)

        usage = {"output": int(r.usage.get("output_tokens") or 0), "context": context, "seconds": r.wall_s}
        if r.refused:
            s["flags"] += 1
            s["flag_streak"] += 1
            s["next"] = M.REFUSED
            if not getattr(self.llm, "drops_refused", False):
                s["fresh_conversation"] = True        # the refused text stays in a CLI session: start a clean one
            self.emit("note", text="The provider's content filter refused the last message; the agent was told to rephrase.")
            self.log(f"turn {s['turn']}: refused by the provider (streak {s['flag_streak']})")
            return
        s["flag_streak"] = 0

        shown = r.text
        if self.cfg["todo"]:
            # The todo list may come before or after the explanation: take it out first, so the explanation is whatever
            # precedes the action block.
            todos = [b for k, b in parse_blocks(r.text)[0] if k == "todo"]
            shown = TODO_BLOCK.sub("", r.text).strip()
            if todos:
                s["todo"] = E.parse_todo(todos[-1])
                atomic_write_json(os.path.join(self.run_dir, "todo.json"), s["todo"])
                self.emit("todo", items=s["todo"])
        blocks, prose = parse_blocks(shown)
        if len(blocks) > 1:
            s["next"] = M.MULTI_BLOCK.format(n=len(blocks))
            self.emit("assistant", turn=s["turn"], kind="sent_back", text=prose, usage=usage)
            self.emit("note", text=f"The reply had {len(blocks)} code blocks, so none was run; the agent was asked for one.")
            self.log(f"turn {s['turn']}: {len(blocks)} blocks, nothing run")
            return
        if blocks and blocks[0][0] not in ("python", "forward"):
            s["next"] = M.UNKNOWN_BLOCK.format(kind=blocks[0][0])
            self.emit("assistant", turn=s["turn"], kind="sent_back", text=prose, usage=usage)
            self.emit("note", text=f"The reply had a {blocks[0][0]} block, which isn't run; the agent was told.")
            self.log(f"turn {s['turn']}: unknown block '{blocks[0][0]}', nothing run")
            return
        if not blocks:
            return self.no_block(r.text, shown, usage)

        kind, body_ = blocks[0]
        reminder = False
        if self.cfg["explain"] and not prose:
            if s["unexplained_streak"] < self.cfg["explain_max_sendbacks"]:
                s["unexplained"] += 1
                s["unexplained_streak"] += 1
                s["next"] = M.EXPLAIN_SENDBACK
                self.explanation(s["turn"], kind, "")
                self.emit("note", text="The reply had no explanation before its block, so it was sent back unrun.")
                self.log(f"turn {s['turn']}: {kind} block without an explanation; sent back unrun")
                return
            reminder = True                            # the third unexplained block in a row runs, with a reminder
        s["unexplained_streak"] = 0
        self.explanation(s["turn"], kind, prose)
        self.emit("assistant", turn=s["turn"], kind=kind, text=prose, block=E.clip(body_, 40_000), usage=usage)
        if self.interactive and self.interrupt_requested():     # interrupted while the reply was arriving
            s["next"] = M.INTERRUPTED_BEFORE_RUN.format(kind=kind)
            return
        if self.cfg["mode"] == "plan" and kind == "forward":
            s["next"] = M.PLAN_NO_FORWARD
            self.emit("note", text="Plan mode: the forward pass was not run.")
            return
        declined = self.permitted(kind, body_)
        if declined:
            s["next"] = declined
            self.emit("note", text=f"You declined the {kind} block; it was not run.")
            return
        msg = self.run_python(body_) if kind == "python" else self.run_forward(body_)
        s["next"] = msg + ("\n" + M.REMINDER if reminder else "")

    # ------------------------------------------------------------------------------------------------ actions
    def run_python(self, code: str) -> str:
        s = self.st
        if s["runs"] >= self.cfg["max_runs"]:
            self.emit("note", text=f"All {self.cfg['max_runs']} Python runs are used; the code was not run.")
            return (M.PY_NONE_LEFT_INTERACTIVE if self.interactive else M.PY_NONE_LEFT).format(n=self.cfg["max_runs"])
        s["runs"] += 1
        self.set_activity("python", f"run {s['runs']}")
        res = self.kernel.run(code)
        # a timeout or crash during this run is reported by the kernel itself ("... its variables are lost");
        # a run that starts on a kernel restarted after such an event gets the note in front
        raw = (M.KERNEL_RESTARTED + "\n" if res["restarted"] else "") + res["out"]
        out = M.format_output(raw, self.cfg["kernel_max_out"]) if raw.strip() else "(no output)"
        self.record({"kind": "python", "run": s["runs"], "turn": s["turn"], "s": res["s"], "status": res["status"],
                     "code": code, "out": raw[:20000]})
        self.emit("tool", turn=s["turn"], kind="python", n=s["runs"], seconds=res["s"], output=E.clip(out),
                  ok=res["status"] == "ok" and not res["error"], status=res["status"])
        self.log(f"turn {s['turn']}: python run {s['runs']} ({len(code):,} chars, {res['s']} s, {res['status']}); "
                 f"passes {s['passes']}/{self.cfg['budget']}")
        return M.PY_OUTPUT.format(k=s["runs"], n=self.cfg["max_runs"], out=out)

    def run_forward(self, body: str) -> str:
        s, budget = self.st, self.cfg["budget"]
        left = budget - s["passes"]

        def refused(msg: str, why: str) -> str:
            self.emit("tool", turn=s["turn"], kind="forward", ok=False, counted=False, error=why)
            return msg
        if left <= 0:
            return refused((M.FWD_NONE_LEFT_INTERACTIVE if self.interactive else M.FWD_NONE_LEFT).format(budget=budget),
                           f"All {budget} forward passes are used.")
        try:
            req = json.loads(body)
            if not isinstance(req, dict):
                raise ValueError("the block must hold one JSON object")
        except ValueError as e:
            return refused(M.FWD_INVALID.format(error=f"{type(e).__name__}: {e}"), f"Not valid JSON: {e}")
        try:
            snapshot = self.task.snapshot_edits(req)
        except UnsafePath as e:
            return refused(M.FWD_INVALID.format(error=f"ValueError: {e}"), str(e))
        names, h = self.edit_hash(snapshot)
        key = hashlib.sha1((json.dumps(req, sort_keys=True) + "|" + h).encode()).hexdigest()
        if key in s["req_keys"]:
            self.log(f"turn {s['turn']}: duplicate of pass {s['req_keys'][key]}, not run")
            return refused(M.FWD_DUPLICATE.format(k=s["req_keys"][key], left=left),
                           f"Identical to pass {s['req_keys'][key]} with unchanged edits; not run.")
        s["requests"] += 1
        k = s["passes"] + 1
        self.set_activity("forward", f"pass {k} of {budget}")
        rq = os.path.join(self.run_dir, "requests", f"fwd_{k}")
        json.dump(req, open(rq + ".json", "w"), indent=1)
        t0 = time.time()
        try:
            summary, where = self.task.forward(req, k, commit=False, edits=snapshot)
        except Interrupted:
            self.record({"kind": "forward", "turn": s["turn"], "request": req, "interrupted": True})
            self.log(f"turn {s['turn']}: forward pass interrupted by the user (not counted)")
            return refused(M.FWD_INTERRUPTED, "Interrupted; not counted.")
        except ForwardError as e:
            self.record({"kind": "forward", "turn": s["turn"], "request": req, "invalid": str(e)})
            self.log(f"turn {s['turn']}: invalid forward request (not counted): {e}")
            return refused(M.FWD_INVALID.format(error=f"ValueError: {e}"), f"Not a valid request: {e}")
        except Exception as e:  # noqa: BLE001 - a runtime failure of the pass is reported, not counted
            tb = traceback.format_exc()
            self.record({"kind": "forward", "turn": s["turn"], "request": req, "failed": tb[-4000:]})
            self.log(f"turn {s['turn']}: forward pass failed (not counted): {type(e).__name__}: {e}")
            s["fwd_fail_streak"] = s.get("fwd_fail_streak", 0) + 1
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
            msg = refused(M.FWD_FAILED.format(error=f"{type(e).__name__}: {e}"), f"Failed: {type(e).__name__}: {e}")
            if s["fwd_fail_streak"] >= self.cfg["max_failed_passes"]:
                # failed passes are not counted, so without this a broken machine keeps paying for model turns
                s["next"] = msg
                self.halt(f"Stopped: {s['fwd_fail_streak']} forward passes failed in a row (last: {type(e).__name__}: "
                          f"{e}). Fix the problem, then re-run the same command to resume.", final=False)
            return msg
        s["fwd_fail_streak"] = 0
        summary = {"ok": True, **summary, "s": round(time.time() - t0, 1)}
        s["passes"] = k
        s["req_keys"][key] = k
        json.dump(summary, open(rq + ".done.json", "w"), indent=1)
        self.record({"kind": "forward", "pass": k, "turn": s["turn"], "request": req, "summary": summary,
                     "edits": names, "edit_hash": h})
        self.log(f"turn {s['turn']}: forward pass {k}/{budget} ({req.get('type')}, {summary['s']} s, edit hash {h})")
        msg = M.FWD_DONE.format(k=k, budget=budget, kind=req.get("type", "?"), where=where,
                                summary=json.dumps(summary, indent=1), left=budget - k)
        msg += "\n" + M.EDITS_USED.format(names=", ".join(names) if names else "none", h=h)
        if k == budget:
            msg += "\n" + M.ALL_PASSES_USED
        # Count the pass on disk before its files appear in the work folder: a kill at any point here either re-runs the
        # commit on resume or leaves nothing the editor could use without the pass being counted.
        s["next"], s["commit_pending"] = msg, True
        self.save_state()
        self.task.stage_commit()
        s["commit_pending"] = False
        self.emit("tool", turn=s["turn"], kind="forward", n=k, budget=budget, seconds=summary["s"], ok=True, counted=True,
                  request=req, where=where, output=E.clip(json.dumps(summary, indent=1)))
        return msg

    def no_block(self, text: str, shown: str, usage: dict):
        s = self.st
        if self.interactive:                    # the editor is talking to the user: wait for their answer
            open(os.path.join(self.run_dir, "final_answer.md"), "w", encoding="utf-8").write(shown)
            self.explanation(s["turn"], "message", shown)
            self.emit("assistant", turn=s["turn"], kind="message", text=shown, usage=usage)
            s["phase"], s["next"], s["last_message"] = "waiting", "", shown[-4000:]
            self.emit("waiting")
            self.log(f"turn {s['turn']}: message to the user ({s['runs']} python runs, {s['passes']} passes so far)")
            return
        left = self.cfg["budget"] - s["passes"]
        if left > 0 and s["runs"] < self.cfg["max_runs"]:
            s["nudges"] += 1
            s["next"] = M.NO_BLOCK_CONTINUE.format(left=left, budget=self.cfg["budget"])
            self.emit("assistant", turn=s["turn"], kind="sent_back", text=shown, usage=usage)
            self.emit("note", text=f"No code block while {left} forward passes remain; the agent was told to continue.")
            self.log(f"turn {s['turn']}: reply without a block while {left} passes remain; told to continue")
            return
        open(os.path.join(self.run_dir, "final_answer.md"), "w", encoding="utf-8").write(text)
        self.explanation(s["turn"], "final answer", text)
        self.emit("assistant", turn=s["turn"], kind="final", text=shown, usage=usage)
        s["done"] = True
        self.log(f"turn {s['turn']}: final answer ({s['runs']} python runs, {s['passes']} passes)")

    def finish(self):
        self.set_activity("finishing")
        self.kernel.close()
        if self.st.get("done"):
            try:
                info = self.task.submit(os.path.join(self.run_dir, "submission"))
                self.st["submission"] = info
                self.log(f"submitted: {json.dumps(info)[:500]}")
            except Exception as e:  # noqa: BLE001
                self.log(f"submission failed: {e!r}")
                self.st["submission_error"] = repr(e)
            self.emit("done", submission=self.st.get("submission"))
        self.save_state()
        self.task.close()
