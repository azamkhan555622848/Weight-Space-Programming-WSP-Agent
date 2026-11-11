"""How a running session and the people watching it talk: everything goes through files in the session folder, so an interface
can close and reopen, several can watch at once, and nothing depends on a network port.

  events.jsonl   what happened, in order: one JSON object per line, written only by the session's own process
  live.txt       the model's reply while it is still arriving
  inbox/         what people send to the session: messages, answers to permission requests, changed settings
  interrupt      a file whose presence asks the agent to stop what it is doing and wait for the user

Event types: session, user, assistant, tool, todo, note, handoff, permission_request, permission, waiting, interrupted, stopped,
halt, error. An `assistant` event holds the model's text and, if it asked for something to be run, the block; the `tool` event that
follows holds the result.
"""
from __future__ import annotations

import json
import os
import re
import time

MAX_OUTPUT = 20_000        # characters of a tool's output kept in an event
TODO_MARKS = {" ": "pending", "~": "in_progress", ">": "in_progress", "x": "completed", "X": "completed"}
MARK_OF = {"pending": " ", "in_progress": "~", "completed": "x"}


# ------------------------------------------------------------------------------------------------ events
class EventLog:
    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "events.jsonl")
        self.n = 0
        if os.path.exists(self.path):
            with open(self.path, "rb") as f:
                self.n = sum(1 for _ in f)

    def emit(self, type: str, **data) -> dict:
        ev = {"i": self.n, "t": round(time.time(), 2), "type": type, **data}
        with open(self.path, "a", encoding="utf-8", newline="\n") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
        self.n += 1
        return ev


def read_events(run_dir: str, offset: int = 0) -> tuple[list[dict], int]:
    """The events after byte `offset`, and the offset to pass next time. A line still being written is left for later."""
    path = os.path.join(run_dir, "events.jsonl")
    try:
        with open(path, "rb") as f:
            f.seek(offset)
            data = f.read()
    except OSError:
        return [], offset
    end = data.rfind(b"\n") + 1
    events = []
    for line in data[:end].splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    return events, offset + end


class LiveText:
    """The reply that is arriving now, for interfaces to show as it comes."""

    def __init__(self, run_dir: str):
        self.path = os.path.join(run_dir, "live.txt")
        self.f = None

    def start(self) -> None:
        self.end()
        self.f = open(self.path, "w", encoding="utf-8", newline="\n")

    def write(self, delta: str) -> None:
        if self.f is not None and delta:
            self.f.write(delta)
            self.f.flush()

    def end(self) -> None:
        if self.f is not None:
            self.f.close()
            self.f = None


def read_live(run_dir: str) -> str:
    try:
        with open(os.path.join(run_dir, "live.txt"), encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


# ------------------------------------------------------------------------------------------------ inbox
_seq = 0


def send(run_dir: str, type: str, **data) -> str:
    """Leave something for the session. Written under another name first, so the session never reads half a file."""
    global _seq
    _seq += 1
    d = os.path.join(run_dir, "inbox")
    os.makedirs(d, exist_ok=True)
    name = f"{time.time_ns():020d}-{os.getpid()}-{_seq}.json"
    tmp = os.path.join(d, name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"type": type, "t": round(time.time(), 2), **data}, f, ensure_ascii=False)
    os.replace(tmp, os.path.join(d, name))
    return os.path.join(d, name)


def _inbox_files(run_dir: str) -> list[str]:
    d = os.path.join(run_dir, "inbox")
    try:
        return [os.path.join(d, n) for n in sorted(os.listdir(d)) if n.endswith(".json")]
    except OSError:
        return []


def take(run_dir: str) -> list[dict]:
    """Everything waiting in the inbox, oldest first; each item is removed as it is read. Only the session's process calls this."""
    out = []
    for p in _inbox_files(run_dir):
        try:
            with open(p, encoding="utf-8") as f:
                item = json.load(f)
        except (OSError, ValueError):
            item = None
        try:
            os.remove(p)
        except OSError:
            continue
        if isinstance(item, dict):
            out.append(item)
    return out


def queued_messages(run_dir: str) -> list[str]:
    """The user's messages the session hasn't picked up yet, for showing as queued."""
    out = []
    for p in _inbox_files(run_dir):
        try:
            with open(p, encoding="utf-8") as f:
                item = json.load(f)
            if item.get("type") == "message":
                out.append(str(item.get("text", "")))
        except (OSError, ValueError, AttributeError):
            pass
    return out


def request_interrupt(run_dir: str) -> None:
    with open(os.path.join(run_dir, "interrupt"), "w", encoding="utf-8") as f:
        f.write("interrupt\n")


# ------------------------------------------------------------------------------------------------ todo lists
def parse_todo(body: str) -> list[dict]:
    """A ```todo block: one item per line, `- [ ] to do`, `- [~] in progress`, `- [x] done`. Other lines are ignored."""
    items = []
    for line in body.splitlines():
        m = re.match(r"\s*(?:[-*]|\d+[.)])\s*\[(.)\]\s*(.+?)\s*$", line)
        if m and m[1] in TODO_MARKS:
            items.append({"text": m[2][:300], "status": TODO_MARKS[m[1]]})
    return items[:50]


def todo_text(items: list[dict]) -> str:
    return "\n".join(f"- [{MARK_OF.get(i.get('status'), ' ')}] {i.get('text', '')}" for i in items) or "(empty)"


def clip(text: str, n: int = MAX_OUTPUT) -> str:
    return text if len(text) <= n else text[:n] + f"\n[{len(text) - n:,} more characters not shown]"
