"""The chat interface: say what you want a network to do, and watch and steer the agent while it reads and edits the weights.

The layout follows coding agents such as OpenCode and Claude Code: the conversation in the middle, each step the agent takes as a
row that opens to show its code and result, the agent's todo list above the prompt, and the mode, editor, effort and permission
settings under it. A side panel shows the target, the session's counters, the edited tensors and the agent's work folder.

A session is its own background process (wsp/agent_main.py). This program only reads the session folder and leaves messages in
it (wsp/events.py), so closing it leaves the agent working, and reopening it picks the conversation up where it is. Text from the
model is always shown as plain text or Markdown, never as interface markup.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text
from textual import events as tev
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Collapsible, Input, Markdown, OptionList, Static, TextArea
from textual.widgets.option_list import Option

from .. import consent, prices, runs, sandbox
from .. import events as E
from ..cli import ACTIVITY, ago, describe
from ..llm import API_BACKENDS, DEFAULT_MODELS, SUBSCRIPTION_BACKENDS
from ..tasks.workspace import targets
from ..tasks.workspace.store import WEIGHT_SUFFIXES

EDITORS = [("Claude Opus 5.5 · API key", "anthropic", "claude-opus-5-5"), ("Claude Opus 5.5 · your Claude subscription", "claude-cli", "claude-opus-5-5"),
           ("GPT-6 Astra · API key", "openai", "gpt-6-astra"), ("GPT-6.1 Sol · API key", "openai", "gpt-6.1-sol"),
           ("GPT-6 Astra · your ChatGPT subscription", "codex-cli", "gpt-6-astra")]
EFFORTS = ["low", "medium", "high", "xhigh", "max"]
PERMISSIONS = [("auto", "Run everything", False, False), ("passes", "Ask before forward passes", False, True),
               ("all", "Ask before every step", True, True)]
VIA = {"anthropic": "API key", "openai": "API key", "claude-cli": "subscription", "codex-cli": "subscription", "scripted": "script"}
SPINNER = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
COMMANDS = [
    ("help", "", "Commands and keys"),
    ("new", "", "Start a new conversation"),
    ("sessions", "", "Open an earlier session"),
    ("target", "[path]", "Choose the network to work on"),
    ("search", "<words>", "Search Hugging Face for a language model to work on"),
    ("model", "", "Choose the editor model and how to reach it"),
    ("effort", "[level]", "The editor's reasoning effort: low, medium, high, xhigh, max"),
    ("mode", "[build|plan]", "Plan: study and propose only. Build: edit and run passes"),
    ("permissions", "[auto|passes|all]", "What the agent must ask you before running"),
    ("budget", "<number>", "Forward passes allowed in this session"),
    ("todo", "", "Show or hide the agent's todo list"),
    ("diff", "", "What the current edit changes, tensor by tensor"),
    ("export", "[full]", "Write the current edit; full: also the complete patched network"),
    ("status", "", "This session's counters and cost"),
    ("notes", "", "The agent's notebook"),
    ("compact", "", "Restart the conversation from the agent's notebook, to save context"),
    ("stop", "", "Stop the session's process (send a message to continue)"),
    ("benchmark", "[task] [passes]", "Run one of the two benchmarks"),
    ("setup", "<task>", "Fetch what a benchmark needs"),
    ("results", "", "Benchmark results"),
    ("doctor", "", "What works on this machine"),
    ("sidebar", "", "Show or hide the side panel"),
    ("theme", "", "Change the colours"),
    ("quit", "", "Close the interface (sessions keep running)"),
]
ALIASES = {"resume": "sessions", "models": "model", "exit": "quit", "q": "quit", "clear": "new", "files": "sidebar", "cost": "status",
           "perm": "permissions"}
HELP_KEYS = [("enter", "send"), ("ctrl+j", "new line"), ("esc", "interrupt the agent"), ("tab", "build / plan mode"),
             ("shift+tab", "permissions"), ("f2", "editor model"), ("ctrl+o", "open or close all steps"), ("ctrl+t", "todo list"),
             ("ctrl+b", "side panel"), ("ctrl+n", "new conversation"), ("ctrl+r", "sessions"), ("ctrl+q", "close")]


def _prefs_path() -> str:
    return os.path.join(os.path.expanduser("~"), ".wsp", "interface.json")


def load_prefs() -> dict:
    return runs.read_json(_prefs_path(), {}) or {}


def save_prefs(p: dict) -> None:
    try:
        os.makedirs(os.path.dirname(_prefs_path()), exist_ok=True)
        runs.atomic_write_json(_prefs_path(), p)
    except OSError:
        pass


def reflow(text: str) -> str:
    """Undo the line breaks of a notice written for an 80-column terminal, so the window can wrap it to its own width."""
    out: list[str] = []
    for line in text.splitlines():
        if line.strip() and out and out[-1].strip() and not line.startswith(("- ", "  http")) and not out[-1].startswith("  http"):
            out[-1] += " " + line.strip()
        else:
            out.append(line)
    return "\n".join(out)


def first_line(code: str, n: int = 76) -> str:
    line = next((ln.strip() for ln in code.splitlines() if ln.strip()), "")
    return line[:n] + ("…" if len(line) > n else "")


def bar(done: int, total: int, width: int = 12) -> str:
    filled = min(width, round(width * done / total)) if total else 0
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------------------------------- transcript items
class UserMsg(Static):
    def __init__(self, text: str, queued: bool = False):
        super().__init__(Text(text), classes="user" + (" queued" if queued else ""))


class Note(Static):
    def __init__(self, text: str, kind: str = "note"):
        super().__init__(Text(text), classes=kind)


class ToolCall(Collapsible):
    """One step of the agent: a row that opens to show its code or request, and the result once it is there."""

    def __init__(self, kind: str, block: str, turn: int):
        self.kind, self.block, self.turn, self.done = kind, block, turn, False
        self.head = "Python" if kind == "python" else "Forward pass"
        self.what = first_line(block) if kind == "python" else self._request_line(block)
        self.code = Static(Syntax(block, "python" if kind == "python" else "json", theme="ansi_dark", word_wrap=True,
                                  background_color="default"), classes="code")
        self.out = Static(Text("running …", style="dim"), classes="output")
        super().__init__(self.code, self.out, title=f"◌ {self.head} · {self.what}", collapsed=True, classes="tool")

    @staticmethod
    def _request_line(block: str) -> str:
        try:
            r = json.loads(block)
            m = r.get("model", "edited")
            return f"{r.get('type', '?')} · {'+'.join(m) if isinstance(m, list) else m}"
        except (ValueError, AttributeError):
            return first_line(block)

    def set_result(self, ev: dict) -> None:
        self.done = True
        ok = bool(ev.get("ok"))
        secs = f" · {ev['seconds']} s" if ev.get("seconds") is not None else ""
        if self.kind == "python":
            n = f" · run {ev['n']}" if ev.get("n") else ""
            self.title = f"{'✓' if ok else '✗'} {self.head}{n} · {self.what}{secs}"
        elif ev.get("counted"):
            self.title = f"✓ {self.head} {ev.get('n')}/{ev.get('budget', '?')} · {self.what}{secs}"
        else:
            self.title = f"✗ {self.head} not run · {self.what}"
        body = ev.get("output") or ev.get("error") or "(no output)"
        self.out.update(Text(body, style="" if ok else "red"))
        self.set_class(not ok, "failed")


class TodoPanel(Static):
    """The agent's todo list, above the prompt: how many are done, then each item."""

    def show(self, items: list[dict]) -> None:
        self.items = items
        self.display = bool(items) and not getattr(self, "hidden_by_user", False)
        if not items:
            return
        done = sum(i["status"] == "completed" for i in items)
        t = Text()
        t.append(f"{done} of {len(items)} todos completed\n", style="bold")
        for k, i in enumerate(items):
            if i["status"] == "completed":
                t.append("☑ ", style="green")
                t.append(i["text"], style="dim strike")
            elif i["status"] == "in_progress":
                t.append("◉ ", style="yellow")
                t.append(i["text"], style="bold")
            else:
                t.append("☐ ")
                t.append(i["text"])
            if k < len(items) - 1:
                t.append("\n")
        self.update(t)


class Chip(Static):
    """A setting under the prompt; clicking it opens its choices."""

    class Pressed(Message):
        def __init__(self, name: str):
            super().__init__()
            self.name = name

    def __init__(self, name: str):
        super().__init__("", id=f"chip-{name}", classes="chip")
        self.chip = name

    def on_click(self) -> None:
        self.post_message(self.Pressed(self.chip))


class Prompt(TextArea):
    """Where the user types. Enter sends; ctrl+j (or shift+enter where the terminal passes it on) starts a new line; up and down
    on the first and last line walk through earlier messages."""

    class Submitted(Message):
        def __init__(self, text: str):
            super().__init__()
            self.text = text

    class Special(Message):
        def __init__(self, key: str):
            super().__init__()
            self.key = key

    def __init__(self):
        super().__init__(soft_wrap=True, tab_behavior="focus", show_line_numbers=False, highlight_cursor_line=False, id="prompt",
                         placeholder="Ask anything …   / for commands", compact=True)
        self.sent: list[str] = []
        self.sent_i: int | None = None
        self.draft = ""

    async def _on_key(self, event: tev.Key) -> None:
        key = event.key
        if key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.Submitted(self.text))
            return
        if key in ("shift+enter", "ctrl+j"):
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if key in ("tab", "shift+tab", "escape"):
            event.stop()
            event.prevent_default()
            self.post_message(self.Special(key))
            return
        await super()._on_key(event)

    def remember(self, text: str) -> None:
        if text.strip() and (not self.sent or self.sent[-1] != text):
            self.sent.append(text)
        self.sent_i = None

    def set_text(self, text: str) -> None:
        self.text = text
        self.move_cursor(self.document.end)

    def action_cursor_up(self, select: bool = False) -> None:
        if not select and self.cursor_location[0] == 0:
            if self.app.suggest_move(-1):
                return
            if self.sent:
                if self.sent_i is None:
                    self.draft, self.sent_i = self.text, len(self.sent)
                if self.sent_i > 0:
                    self.sent_i -= 1
                    self.set_text(self.sent[self.sent_i])
                return
        super().action_cursor_up(select)

    def action_cursor_down(self, select: bool = False) -> None:
        if not select and self.cursor_location[0] == self.document.line_count - 1:
            if self.app.suggest_move(1):
                return
            if self.sent_i is not None:
                self.sent_i += 1
                if self.sent_i >= len(self.sent):
                    self.sent_i = None
                    self.set_text(self.draft)
                else:
                    self.set_text(self.sent[self.sent_i])
                return
        super().action_cursor_down(select)


# ---------------------------------------------------------------------------------------------------- windows on top
class NoticeScreen(ModalScreen[bool]):
    """A notice the user must accept before going on. The buttons stay on screen however small the terminal is; the text scrolls."""
    AUTO_FOCUS = "#cancel"      # Enter on its own must not accept anything

    def __init__(self, text: str):
        super().__init__()
        self.text = text

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog tall"):
            with VerticalScroll():
                yield Static(Text(reflow(self.text)))
            with Horizontal(classes="buttons"):
                yield Button("I accept", id="accept", variant="warning")
                yield Button("Cancel", id="cancel", variant="primary")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "accept")


class PermissionScreen(ModalScreen[tuple]):
    """The agent wants to run a block and the user asked to be asked. -> (decision, feedback)"""
    AUTO_FOCUS = "#once"
    BINDINGS = [Binding("escape", "reject", show=False)]

    def __init__(self, request_id: str, kind: str, body: str):
        super().__init__()
        self.request_id, self.kind, self.body = request_id, kind, body

    def compose(self) -> ComposeResult:
        what = "this code in its kernel" if self.kind == "python" else "this forward pass"
        with Vertical(classes="dialog tall"):
            yield Static(Text(f"The agent wants to run {what}:", style="bold"))
            with VerticalScroll():
                yield Static(Syntax(self.body, "python" if self.kind == "python" else "json", theme="ansi_dark", word_wrap=True,
                                    background_color="default"))
            yield Input(placeholder="If not: tell the agent what to do instead (optional)", id="feedback")
            with Horizontal(classes="buttons"):
                yield Button("Run it", id="once", variant="success")
                yield Button("Always run " + ("code" if self.kind == "python" else "passes"), id="always")
                yield Button("Don't run", id="reject", variant="error")

    def action_reject(self) -> None:
        self.dismiss(("reject", self.query_one("#feedback", Input).value))

    def on_input_submitted(self) -> None:
        self.action_reject()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss((event.button.id, self.query_one("#feedback", Input).value))


class PickerScreen(ModalScreen[str]):
    """Choose one of a list, or type something else. -> the chosen value, the typed text, or None."""
    BINDINGS = [Binding("escape", "cancel", show=False)]

    def __init__(self, title: str, options: list[tuple[str, str]], placeholder: str | None = None):
        super().__init__()
        self.title_text, self.options, self.placeholder = title, options, placeholder
        self.AUTO_FOCUS = "#typed" if placeholder else "#choices"       # set once its widgets exist, unlike a focus() in on_mount

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog"):
            yield Static(Text(self.title_text, style="bold"))
            if self.placeholder:
                yield Input(placeholder=self.placeholder, id="typed")
            yield OptionList(*[Option(Text(label), id=str(i)) for i, (label, _) in enumerate(self.options)], id="choices")

    def on_key(self, event: tev.Key) -> None:
        if self.placeholder and event.key in ("down", "up") and self.focused is self.query_one("#typed", Input) and self.options:
            self.query_one("#choices", OptionList).focus()

    def action_cancel(self) -> None:
        self.dismiss(None)

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        self.dismiss(self.options[int(event.option.id)][1])

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.value.strip():
            self.dismiss(event.value.strip())


class InfoScreen(ModalScreen[None]):
    """Something to read; Esc or Enter closes it."""
    BINDINGS = [Binding("escape,enter,q", "close", show=False)]

    def __init__(self, title: str, body):
        super().__init__()
        self.title_text, self.body = title, body

    def compose(self) -> ComposeResult:
        with Vertical(classes="dialog tall"):
            yield Static(Text(self.title_text, style="bold"))
            with VerticalScroll(id="info-body"):
                yield Static(self.body if not isinstance(self.body, str) else Text(self.body), id="info-text")
            yield Static(Text("esc to close", style="dim"))

    def set_body(self, body) -> None:
        self.body = body
        self.query_one("#info-text", Static).update(body if not isinstance(body, str) else Text(body))

    def action_close(self) -> None:
        self.dismiss(None)


# ---------------------------------------------------------------------------------------------------- the app
class WspApp(App):
    TITLE = "WSP Agent"
    ENABLE_COMMAND_PALETTE = False
    CSS = """
    Screen { layout: horizontal; }
    #main { width: 1fr; height: 100%; }
    #title { height: 1; padding: 0 2; background: $boost; }
    #transcript { height: 1fr; padding: 1 2 0 2; }
    #side { width: 38; height: 100%; padding: 1 2; border-left: solid $panel; overflow-y: auto; }
    .user { margin: 1 0 1 0; padding: 0 1; border-left: thick $accent; background: $boost; }
    .queued { border-left: thick $panel; color: $text-muted; }
    .note { color: $text-muted; text-style: italic; margin: 0 0 0 1; }
    .error { color: $error; margin: 0 0 0 1; }
    Markdown { margin: 0; padding: 0 1; background: transparent; }
    .tool { margin: 0 0 1 0; padding: 0; border-top: none; background: transparent; }
    .tool CollapsibleTitle { padding: 0 1; color: $text-muted; }
    .tool.failed CollapsibleTitle { color: $error; }
    .tool Contents { padding: 0 0 0 3; }
    .code { padding: 0 1; background: $boost; }
    .output { padding: 0 1; color: $text-muted; }
    #welcome { padding: 1 1; }
    #live { color: $text; }
    #working { height: 1; padding: 0 3; color: $text-muted; }
    #queued { height: auto; padding: 0 2; }
    #todo { height: auto; max-height: 9; margin: 0 2; padding: 0 1; border: round $panel; overflow-y: auto; }
    #suggest { height: auto; max-height: 10; margin: 0 2; border: round $accent; }
    #prompt { height: auto; min-height: 3; max-height: 10; margin: 0 2; padding: 0 1; border: round $accent; }
    #bar { height: 1; padding: 0 3; }
    .chip { width: auto; margin-right: 3; }
    .chip:hover { text-style: underline; }
    #hints { width: 1fr; content-align: right middle; color: $text-muted; }
    ModalScreen { align: center middle; }
    .dialog { width: 100; max-width: 96%; height: auto; max-height: 90%; padding: 1 2; background: $surface; border: thick $accent; }
    .dialog.tall { height: 90%; }
    .dialog VerticalScroll { height: 1fr; margin: 1 0; }
    .dialog OptionList { height: auto; max-height: 22; margin-top: 1; }
    .buttons { height: auto; margin-top: 1; }
    .buttons Button { margin-right: 2; }
    """
    BINDINGS = [Binding("ctrl+o", "toggle_steps", "Steps", show=False), Binding("ctrl+t", "toggle_todo", "Todo", show=False),
                Binding("ctrl+b", "toggle_side", "Side panel", show=False), Binding("ctrl+n", "new", "New", show=False),
                Binding("ctrl+r", "sessions", "Sessions", show=False), Binding("f2", "model", "Model", show=False),
                Binding("ctrl+q", "quit", "Close", show=False)]

    def __init__(self, runs_dir: str | None = None, project_dir: str | None = None, session: str | None = None, gpu_poll: bool = True):
        super().__init__()
        self.runs_dir = runs_dir or runs.RUNS_DIR
        self.project = os.path.abspath(project_dir or os.getcwd())
        self.gpu_poll, self.gpu = gpu_poll, None
        prefs = load_prefs()
        self.backend = prefs.get("backend") if prefs.get("backend") in VIA else "anthropic"
        self.model = prefs.get("model") or DEFAULT_MODELS.get(self.backend)
        self.effort = prefs.get("effort") if prefs.get("effort") in EFFORTS else "high"
        self.mode, self.perm, self.budget = "build", "auto", None
        self.target: dict | None = None
        self.session: str | None = None          # the open session's folder
        self.initial = session
        self.offset = 0
        self.tools: dict[int, ToolCall] = {}
        self.skip_user: str | None = None
        self.pending_perm: dict | None = None
        self.perm_open: str | None = None
        self.live_text = ""
        self.live_on = False
        self.info: dict | None = None
        self.tick = 0
        self.starting: str | None = None
        self.suggestions: list[tuple[str, str, str]] = []
        self.pending_goal: str | None = None
        self.overlay: dict = {}                  # settings the user changed that the session hasn't taken up yet

    # ------------------------------------------------------------------------------------------------ layout
    def compose(self) -> ComposeResult:
        with Vertical(id="main"):
            yield Static(id="title")
            with VerticalScroll(id="transcript"):
                yield Static(id="welcome")
            yield Static(id="queued")
            yield Static(id="working")
            yield TodoPanel(id="todo")
            yield OptionList(id="suggest")
            yield Prompt()
            with Horizontal(id="bar"):
                for name in ("mode", "model", "effort", "perm"):
                    yield Chip(name)
                yield Static(id="hints")
        yield Static(id="side")

    def on_mount(self) -> None:
        self.transcript = self.query_one("#transcript", VerticalScroll)
        self.transcript.anchor()
        self.prompt = self.query_one(Prompt)
        self.todo = self.query_one("#todo", TodoPanel)
        self.todo.display = False
        self.query_one("#suggest", OptionList).display = False
        self.query_one("#queued", Static).display = False
        self.prompt.focus()
        self.refresh_chrome()
        self.set_interval(0.25, self.poll)
        if self.gpu_poll:
            self.set_interval(5, self.poll_gpu)
        if self.initial:
            d = runs.find_run(self.initial, self.runs_dir)
            if d:
                self.call_after_refresh(self.open_session, d)
            else:
                self.notify(f"No session named {self.initial}.", severity="error")

    # ------------------------------------------------------------------------------------------------ what is shown
    def welcome_text(self) -> Text:
        t = Text()
        t.append("WSP Agent\n", style="bold")
        t.append("Program a neural network by reading and editing its weights, with no training.\n\n", style="dim")
        if self.target:
            t.append("Target   ", style="dim")
            t.append(f"{self.target['name']}  ({self.target['kind']})\n")
        else:
            t.append("Target   ", style="dim")
            t.append("none yet: /target to choose, /search to look on Hugging Face, or just type and you'll be asked\n")
            found = targets.discover(self.project, 6)
            for what, desc in found:
                t.append(f"           {what}", style="cyan")
                t.append(f"  {desc}\n", style="dim")
        t.append("Folder   ", style="dim")
        t.append(f"{self.project}\n\n")
        recent = [r for r in runs.list_runs(self.runs_dir)][:5]
        if recent:
            t.append("Recent sessions (/sessions to open one)\n", style="dim")
            for r in recent:
                t.append(f"  {r['name']:<26}", style="cyan")
                t.append(f" {r['state']:<9} {r['title'] or r['task']}\n", style="dim")
            t.append("\n")
        t.append("Say what the network should do differently, for example:\n", style="dim")
        t.append("  Find out why this model repeats itself in long answers, and fix it.\n")
        t.append("  This classifier has a backdoor. Find the trigger and remove it without hurting accuracy.\n")
        return t

    def refresh_chrome(self) -> None:
        """The title, the settings under the prompt and the side panel."""
        r = self.info
        title = Text()
        if r:
            title.append("◆ ", style="bold magenta")
            title.append(r["title"] or {"edge_backdoor": "Benchmark: remove a backdoor from a ResNet-8",
                                         "qwen_math": "Benchmark: make Qwen3.5-0.8B better at math"}.get(r["task"], r["name"]), style="bold")
            title.append(f"   {r['name']}", style="dim")
        else:
            title.append("◆ New conversation", style="bold")
        self.query_one("#title", Static).update(title)
        mode = r["mode"] if r else self.mode
        model = r["model"] if r else self.model
        backend = r["backend"] if r else self.backend
        effort = r["effort"] if r else self.effort
        perm = self.perm_of(r["ask_python"], r["ask_forward"]) if r else self.perm
        chips = {"mode": Text.assemble(("Build" if mode == "build" else "Plan", "bold " + ("green" if mode == "build" else "yellow")), " ▾"),
                 "model": Text.assemble((str(model), "bold"), (f" · {VIA.get(backend, backend)}", "dim"), " ▾"),
                 "effort": Text.assemble((str(effort), "bold"), " ▾"),
                 "perm": Text.assemble(({"auto": "auto", "passes": "ask: passes", "all": "ask: all"}[perm], "bold"), " ▾")}
        for name, text in chips.items():
            self.query_one(f"#chip-{name}", Chip).update(text)
        working = bool(r and r["state"] == "running")
        self.query_one("#hints", Static).update(Text("esc interrupt · / commands" if working else "/ commands · ctrl+q close"))
        self.query_one("#side", Static).update(self.side_text())

    @staticmethod
    def perm_of(ask_python: bool, ask_forward: bool) -> str:
        return "all" if ask_python else "passes" if ask_forward else "auto"

    def side_text(self) -> Text:
        t = Text()
        r = self.info
        t.append("TARGET\n", style="bold dim")
        if r:
            t.append(f"{r['target']}\n")
            t.append(f"{r['target_kind']}\n\n", style="dim")
        elif self.target:
            t.append(f"{self.target['name']}\n")
            t.append(f"{self.target['kind']}\n\n", style="dim")
        else:
            t.append("none: /target\n\n", style="dim")
        if not r:
            t.append("SANDBOX\n", style="bold dim")
            pol = sandbox.resolve_policy()
            t.append("none on this system\n" if pol.kind == "none" else f"{pol.kind}\n", style="yellow" if pol.kind == "none" else "")
            return t
        t.append("SESSION\n", style="bold dim")
        t.append(f"Passes  {bar(r['passes'], r['budget'])} {r['passes']}/{r['budget']}\n")
        t.append(f"Python runs  {r['runs']}/{r['max_runs']}\n")
        t.append(f"Turns  {r['turn']}\n")
        tok = r["tokens"]
        if tok:
            t.append(f"Tokens  {tok.get('output', 0):,} out\n")
            t.append("Cost  " + (f"about ${r['cost_usd']:.2f}\n" if r["cost_usd"] is not None else "your plan's limits\n"))
        if self.gpu and r["running"]:
            t.append(f"GPU  {self.gpu['used_gb']:.1f}/{self.gpu['total_gb']:.0f} GB\n")
        t.append("Sandbox  ")
        t.append("none\n" if r["sandbox"] == "none" else f"{r['sandbox'] or 'not recorded'}\n", style="yellow" if r["sandbox"] == "none" else "")
        edits = runs.edited_tensors(r["path"])
        t.append(f"\nEDITS ({len(edits)})\n", style="bold dim")
        for n in edits[:12]:
            t.append(f"{n[-34:]}\n")
        if len(edits) > 12:
            t.append(f"… {len(edits) - 12} more (/diff)\n", style="dim")
        if not edits:
            t.append("none yet\n", style="dim")
        t.append("\nFILES\n", style="bold dim")
        for name, size, is_dir in runs.work_files(r["path"], 24):
            if is_dir:
                t.append(f"{name[:24]:<24}", style="cyan")
                t.append(f" {size} files\n", style="dim")
            else:
                t.append(f"{name[:24]:<24}")
                t.append(f" {size / 2 ** 20:,.1f}M\n" if size >= 2 ** 20 else f" {size / 1024:,.0f}k\n" if size >= 1024 else f" {size}b\n", style="dim")
        return t

    def working_text(self) -> Text:
        r = self.info
        t = Text()
        if self.starting:
            t.append(f"{SPINNER[self.tick % len(SPINNER)]} {self.starting}")
            return t
        if not r:
            return t
        if r["state"] == "running":
            what = {"model": "Thinking", "python": "Running code", "forward": "Running a forward pass",
                    "permission": "Waiting for your approval", "finishing": "Writing the result"}.get(r["activity"], "Starting")
            if r["detail"] and r["activity"] in ("python", "forward"):
                what += f" · {r['detail']}"
            t.append(f"{SPINNER[self.tick % len(SPINNER)]} {what}")
            if r["since"]:
                t.append(f" · {ago(time.time() - r['since'])}")
            t.append(" · esc to interrupt" if r["interactive"] else "", style="dim")
        elif r["state"] == "waiting":
            t.append("● Your turn", style="green")
        elif r["state"] == "idle":
            t.append("○ Idle: send a message to continue", style="dim")
        elif r["state"] == "halted":
            t.append(f"■ Stopped for good: {r['halted']}", style="red")
        elif r["state"] == "finished":
            t.append("■ Finished", style="blue")
        elif r["state"] == "stopped":
            t.append("○ Stopped: /resume continues it", style="dim")
        return t

    # ------------------------------------------------------------------------------------------------ the transcript
    async def add(self, *widgets) -> None:
        await self.transcript.mount_all(widgets)

    async def clear_transcript(self) -> None:
        await self.transcript.remove_children()
        self.tools, self.live_on, self.live_text = {}, False, ""

    async def render_events(self, evs: list[dict]) -> None:
        new = []
        for ev in evs:
            t = ev.get("type")
            if t == "user":
                if self.skip_user is not None and ev.get("text") == self.skip_user:
                    self.skip_user = None
                    continue
                new.append(UserMsg(ev.get("text", "")))
            elif t == "assistant":
                kind = ev.get("kind")
                if ev.get("text"):
                    new.append(Markdown(ev["text"], open_links=False))
                if kind in ("python", "forward") and ev.get("block") is not None:
                    tc = ToolCall(kind, ev["block"], ev.get("turn", 0))
                    self.tools[ev.get("turn", 0)] = tc
                    new.append(tc)
            elif t == "tool":
                tc = self.tools.get(ev.get("turn"))
                if tc is None or tc.done:
                    tc = ToolCall(ev.get("kind", "python"), json.dumps(ev.get("request") or {}), ev.get("turn", 0))
                    new.append(tc)
                tc.set_result(ev)
            elif t == "todo":
                self.todo.show(ev.get("items") or [])
            elif t == "note":
                new.append(Note(ev.get("text", "")))
            elif t == "handoff":
                new.append(Note("The conversation was restarted from the agent's notebook to save context."))
            elif t == "interrupted":
                new.append(Note("Interrupted. The agent is waiting for you."))
            elif t == "resumed":
                new.append(Note("Session resumed. The agent's kernel was restarted, so its variables are gone; its files are kept."))
            elif t == "stopped":
                new.append(Note("The session's process stopped after a long silence; it resumes when you write." if ev.get("reason") == "idle"
                                else "The session's process was stopped. Send a message to continue."))
            elif t == "halt":
                new.append(Note(str(ev.get("reason", "")), "error"))
            elif t == "error":
                new.append(Note(str(ev.get("text", "")), "error"))
            elif t == "setting":
                new.append(Note(f"{str(ev.get('key')).replace('_', ' ')} → {ev.get('value')}"))
            elif t == "done":
                new.append(Note("Finished. /results shows benchmark results; `wsp eval` scores this run."))
            elif t == "permission_request":
                self.pending_perm = ev
            elif t == "permission":
                if self.pending_perm and self.pending_perm.get("id") == ev.get("id"):
                    self.pending_perm = None
            if t in ("assistant", "interrupted", "stopped", "halt", "error"):
                await self.end_live()
        if new:
            await self.add(*new)

    async def show_live(self, text: str) -> None:
        if text == self.live_text and self.live_on:
            return
        if not self.live_on:
            await self.add(Markdown("", id="live", open_links=False))
            self.live_on = True
        self.live_text = text
        await self.query_one("#live", Markdown).update(text)

    async def end_live(self) -> None:
        """The reply is complete (its event has arrived): what is left in live.txt is that same reply, not a new one."""
        self.live_stale = E.read_live(self.session) if self.session else ""
        if self.live_on:
            self.live_on, self.live_text = False, ""
            await self.query("#live").remove()

    # ------------------------------------------------------------------------------------------------ the clock
    async def poll(self) -> None:
        self.tick += 1
        d = self.session
        if d:
            if self.tick % 2 == 0 or self.info is None:
                self.info = runs.run_info(d)
                # A change reaches the agent when it next looks, which can be minutes into a pass: show what was chosen until
                # the session reports the same.
                for k, v in list(self.overlay.items()):
                    if self.info.get(k) == v:
                        del self.overlay[k]
                    else:
                        self.info[k] = v
            if self.offset != -1:
                evs, self.offset = runs.session_events(d, self.offset)
                if evs:
                    await self.render_events(evs)
            r = self.info
            if r and r["state"] == "running" and r["activity"] == "model":
                text = E.read_live(d)
                if text.strip() and text != getattr(self, "live_stale", ""):
                    await self.show_live(text)
            queued = E.queued_messages(d)
            q = self.query_one("#queued", Static)
            q.display = bool(queued)
            if queued:
                q.update(Text("\n".join(f"queued: {m}" for m in queued), style="dim"))
            if self.pending_perm and self.perm_open != self.pending_perm.get("id") and r and r["running"]:
                self.ask_permission(self.pending_perm)
            if r and not self.todo.display and r["todo"] and not getattr(self.todo, "hidden_by_user", False) and not getattr(self.todo, "items", None):
                self.todo.show(r["todo"])
        self.query_one("#working", Static).update(self.working_text())
        if self.tick % 4 == 0:
            self.refresh_chrome()
            if not d and not self.transcript.query(".user"):
                w = self.query("#welcome")
                if w:
                    w.first(Static).update(self.welcome_text())

    def poll_gpu(self) -> None:
        def work():
            self.gpu = runs.gpu_status()
        self.run_worker(work, thread=True, group="gpu", exclusive=True)

    # ------------------------------------------------------------------------------------------------ sessions
    async def open_session(self, run_dir: str) -> None:
        await self.clear_transcript()
        self.session, self.offset, self.info = run_dir, 0, runs.run_info(run_dir)
        self.pending_perm, self.perm_open, self.skip_user, self.overlay = None, None, None, {}
        self.todo.items = []
        self.todo.show(self.info["todo"])
        self.transcript.anchor()
        self.refresh_chrome()

    async def new_conversation(self) -> None:
        await self.clear_transcript()
        self.session, self.info, self.offset = None, None, 0
        self.todo.show([])
        await self.add(Static(self.welcome_text(), id="welcome"))
        self.refresh_chrome()

    def with_consents(self, backend: str, then) -> None:
        """Show any notice not yet accepted on this machine, then go on."""
        need = []
        if sandbox.resolve_policy().kind == "none" and not consent.accepted(consent.NO_SANDBOX):
            need.append((consent.NO_SANDBOX, consent.NO_SANDBOX_TEXT))
        if backend in SUBSCRIPTION_BACKENDS and not consent.accepted(consent.SUBSCRIPTION):
            need.append((consent.SUBSCRIPTION, consent.SUBSCRIPTION_TEXT))

        def step(i: int) -> None:
            if i == len(need):
                then()
                return
            key, text = need[i]

            def answered(ok: bool | None) -> None:
                if ok:
                    consent.record(key)
                    step(i + 1)
                else:
                    self.notify("Not started.")
            self.push_screen(NoticeScreen(text), answered)
        step(0)

    @staticmethod
    def accepted_flags(backend: str) -> tuple[bool, bool]:
        """(accept_no_sandbox, accept_subscription) to pass on, after with_consents. The first only where the system has no
        sandbox: where one exists but fails its self-test, the session must refuse, not fall back to none unasked."""
        return sandbox.resolve_policy().kind == "none", backend in SUBSCRIPTION_BACKENDS

    def launch(self, run_dir: str, args: list[str], what: str = "Starting the session") -> None:
        """Start a session's process in the background, and say so if it fails."""
        self.starting = what

        def work() -> None:
            ok, msg = runs.start(run_dir, args)
            self.call_from_thread(self.launched, run_dir, ok, msg)
        self.run_worker(work, thread=True)

    async def launched(self, run_dir: str, ok: bool, msg: str) -> None:
        self.starting = None
        if ok or self.session != run_dir:
            return
        await self.add(Note(f"The session didn't start:\n{msg}", "error"))
        if not os.path.exists(os.path.join(run_dir, "state.json")):
            # Nothing ran, so there is no session to talk to: go back to the start, with the request ready to send again.
            if self.pending_goal and not self.prompt.text:
                self.prompt.set_text(self.pending_goal)
            self.session, self.info, self.offset, self.skip_user = None, None, 0, None
            if not os.path.exists(os.path.join(run_dir, "launch.json")):
                import shutil
                shutil.rmtree(run_dir, ignore_errors=True)
            await self.add(Note("Nothing was started. Fix the problem above (/target chooses the network, /doctor checks this "
                                "machine), then send your request again."))
            self.refresh_chrome()

    def start_session(self, goal: str) -> None:
        name = runs.new_run_name("workspace")
        run_dir = os.path.join(self.runs_dir, name)
        ask = {"auto": (), "passes": ("forward",), "all": ("python", "forward")}[self.perm]

        async def go() -> None:
            flags = self.accepted_flags(self.backend)
            args = runs.session_args(run_dir, self.target, goal, self.backend, self.model, self.effort, self.budget, self.project,
                                     mode=self.mode, ask=ask, accept_no_sandbox=flags[0], accept_subscription=flags[1])
            os.makedirs(run_dir, exist_ok=True)
            await self.clear_transcript()
            self.session, self.offset, self.info, self.skip_user = run_dir, 0, None, goal
            self.pending_goal = goal
            self.todo.items = []
            self.todo.show([])
            await self.add(UserMsg(goal))
            self.launch(run_dir, args, f"Loading {self.target['name']}")
        self.with_consents(self.backend, lambda: self.call_after_refresh(go))

    def continue_session(self, text: str) -> None:
        if self.starting is not None and not os.path.exists(os.path.join(self.session, "state.json")):
            self.prompt.set_text(text)          # the session is still loading: keep the words rather than lose them
            self.notify("Still starting. Your message is kept in the prompt; send it once the agent is up.")
            return
        r = self.info or runs.run_info(self.session)
        if r["state"] in ("finished", "halted"):
            self.notify(f"This session is {r['state']}. /new starts another.", severity="warning")
            return
        if r["interactive"]:
            runs.send_message(self.session, text)
        else:
            runs.send_note(self.session, text)
            self.notify("Left as a note for the agent of this benchmark run.")
        if not runs.is_running(self.session) and self.starting is None:
            d = self.session

            def go() -> None:
                self.launch(d, runs.resume_args(d, *self.accepted_flags(r["backend"])), "Resuming the session")
            self.with_consents(r["backend"], go)

    # ------------------------------------------------------------------------------------------------ the prompt
    def on_prompt_submitted(self, event: Prompt.Submitted) -> None:
        if self.suggest_open() and self.accept_suggestion():
            return
        text = event.text.strip()
        if not text:
            return
        self.prompt.remember(text)
        self.prompt.set_text("")
        if text.startswith("/"):
            self.run_command(text)
        elif self.session:
            self.continue_session(text)
        elif self.target is None:
            self.pick_target(then=lambda: self.start_session(text))
        elif self.target["kind"] == "benchmark":
            self.notify(f"{self.target['name']} is a fixed benchmark with its own task: /benchmark {self.target['task']} runs it.")
        else:
            self.start_session(text)

    def on_prompt_special(self, event: Prompt.Special) -> None:
        if event.key == "escape":
            if self.suggest_open():
                self.close_suggest()
            elif self.info and self.info["state"] == "running" and self.info["interactive"]:
                runs.request_interrupt(self.session)
                self.notify("Interrupting …")
            elif self.prompt.text:
                self.prompt.set_text("")
        elif event.key == "tab":
            if self.suggest_open():
                self.accept_suggestion(run=False)
            else:
                self.set_mode("plan" if (self.info["mode"] if self.info else self.mode) == "build" else "build")
        elif event.key == "shift+tab":
            cur = self.perm_of(self.info["ask_python"], self.info["ask_forward"]) if self.info else self.perm
            names = [p[0] for p in PERMISSIONS]
            self.set_perm(names[(names.index(cur) + 1) % len(names)])

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        text = self.prompt.text
        if text.startswith("/") and " " not in text and "\n" not in text:
            word = text[1:].lower()
            self.suggestions = [c for c in COMMANDS if c[0].startswith(word)] or [c for c in COMMANDS if word in c[0]]
            ol = self.query_one("#suggest", OptionList)
            ol.clear_options()
            for name, args, desc in self.suggestions:
                ol.add_option(Option(Text.assemble((f"/{name}", "bold"), (f" {args}" if args else "", "cyan"), (f"   {desc}", "dim"))))
            ol.display = bool(self.suggestions)
            if self.suggestions:
                ol.highlighted = 0
        else:
            self.close_suggest()

    def suggest_open(self) -> bool:
        return bool(self.query_one("#suggest", OptionList).display)

    def close_suggest(self) -> None:
        self.query_one("#suggest", OptionList).display = False

    def suggest_move(self, step: int) -> bool:
        if not self.suggest_open():
            return False
        ol = self.query_one("#suggest", OptionList)
        ol.highlighted = ((ol.highlighted or 0) + step) % max(1, len(self.suggestions))
        return True

    def accept_suggestion(self, run: bool = True) -> bool:
        ol = self.query_one("#suggest", OptionList)
        if ol.highlighted is None or not self.suggestions:
            return False
        name, args, _ = self.suggestions[ol.highlighted]
        self.close_suggest()
        if args.startswith("<") or not run:
            self.prompt.set_text(f"/{name} ")
        else:
            self.prompt.remember(f"/{name}")
            self.prompt.set_text("")
            self.run_command(f"/{name}")
        return True

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        if event.option_list.id == "suggest":
            self.accept_suggestion()
            self.prompt.focus()

    # ------------------------------------------------------------------------------------------------ settings
    def on_chip_pressed(self, event: Chip.Pressed) -> None:
        {"mode": lambda: self.set_mode("plan" if (self.info["mode"] if self.info else self.mode) == "build" else "build"),
         "model": self.action_model, "effort": self.pick_effort, "perm": self.pick_perm}[event.name]()

    def set_mode(self, mode: str) -> None:
        if mode not in ("build", "plan"):
            self.notify("The modes are build and plan.", severity="warning")
            return
        if self.session and self.info and self.info["interactive"]:
            runs.send_setting(self.session, "mode", mode)
            self.info["mode"] = self.overlay["mode"] = mode
        elif self.session:
            self.notify("Benchmark runs have no plan mode.")
            return
        self.mode = mode
        self.notify("Plan mode: the agent studies the weights and proposes; no edits, no forward passes." if mode == "plan"
                    else "Build mode: the agent may edit the weights and run forward passes.")
        self.refresh_chrome()

    def set_perm(self, name: str) -> None:
        match = [p for p in PERMISSIONS if p[0] == name]
        if not match:
            self.notify("The choices are auto, passes and all.", severity="warning")
            return
        _, label, ask_python, ask_forward = match[0]
        if self.session and self.info and self.info["interactive"]:
            runs.send_setting(self.session, "ask_python", ask_python)
            runs.send_setting(self.session, "ask_forward", ask_forward)
            self.info["ask_python"] = self.overlay["ask_python"] = ask_python
            self.info["ask_forward"] = self.overlay["ask_forward"] = ask_forward
        self.perm = name
        self.notify(f"Permissions: {label.lower()}.")
        self.refresh_chrome()

    def pick_perm(self) -> None:
        self.push_screen(PickerScreen("What must the agent ask you before running?", [(p[1], p[0]) for p in PERMISSIONS]),
                         lambda v: v and self.set_perm(v))

    def set_effort(self, effort: str) -> None:
        if effort not in EFFORTS:
            self.notify("Effort is one of: " + ", ".join(EFFORTS), severity="warning")
            return
        if self.session and self.info and self.info["interactive"]:
            runs.send_setting(self.session, "editor_effort", effort)
            self.info["effort"] = self.overlay["effort"] = effort
        self.effort = effort
        save_prefs({"backend": self.backend, "model": self.model, "effort": self.effort})
        self.refresh_chrome()

    def pick_effort(self) -> None:
        self.push_screen(PickerScreen("The editor's reasoning effort", [(e, e) for e in EFFORTS]), lambda v: v and self.set_effort(v))

    def action_model(self) -> None:
        opts = [(label, f"{b}:{m}") for label, b, m in EDITORS]

        def chosen(v: str | None) -> None:
            if not v:
                return
            backend, _, model = v.partition(":")
            if backend not in VIA or not model:
                self.notify("Type it as backend:model, for example openai:gpt-6.1-sol.", severity="warning")
                return
            if self.session and self.info and self.info["interactive"]:
                if backend != self.info["backend"]:
                    self.notify(f"This session reaches its editor through {VIA.get(self.info['backend'])}. The new choice applies to the "
                                "next session (/new).")
                else:
                    runs.send_setting(self.session, "editor_model", model)
                    self.info["model"] = self.overlay["model"] = model
            self.backend, self.model = backend, model
            save_prefs({"backend": self.backend, "model": self.model, "effort": self.effort})
            self.refresh_chrome()
        self.push_screen(PickerScreen("The editor: the model that reads and edits the weights", opts,
                                      "or type backend:model (anthropic, openai, claude-cli, codex-cli)"), chosen)

    def pick_target(self, then=None) -> None:
        found = [(f"{what}   {desc}", what) for what, desc in targets.discover(self.project)]

        def chosen(v: str | None) -> None:
            if v and self.set_target(v, search=True, then=then) and then:
                then()
        self.push_screen(PickerScreen("Which network? Pick one found in this folder, or type a model folder, a Hugging Face id, a "
                                      "weights file followed by the code that builds it, or words to search Hugging Face for", found,
                                      "models/my-model  |  org/model  |  ckpt.pt model.py:build  |  words to search"), chosen)

    def set_target(self, text: str, search: bool = False, then=None) -> bool:
        try:
            target = targets.resolve(text, self.project)
        except targets.TargetError as e:
            looks_like_path = ("/" in text or "\\" in text or ":" in text or text.lower().endswith(WEIGHT_SUFFIXES)
                               or os.path.exists(os.path.join(self.project, text)))
            if search and not looks_like_path:
                self.search_hub(text, then)          # words, not a path: look for a model with that name
            else:
                self.notify(str(e), severity="error", timeout=10)
            return False
        if target["kind"] == "benchmark":
            self.notify(f"{target['name']} is one of the benchmarks, with its own fixed task: /benchmark {target['task']} runs it. "
                        "To talk about a model yourself, choose its folder or id.", severity="warning", timeout=12)
            return False
        self.target = target
        self.notify(f"Target: {self.target['name']} ({self.target['kind']}).")
        self.refresh_chrome()
        w = self.query("#welcome")
        if w:
            w.first(Static).update(self.welcome_text())
        return True

    def search_hub(self, words: str, then=None) -> None:
        """Look for text-generation models on Hugging Face by name, and offer them as targets. This is the interface searching
        for the user; the agent itself has no internet."""
        self.notify(f"Searching Hugging Face for {words!r} …")

        def work() -> None:
            try:
                from huggingface_hub import HfApi
                models = list(HfApi().list_models(search=words, pipeline_tag="text-generation", sort="downloads", limit=20))
                rows = [(f"{m.id:<48} {getattr(m, 'downloads', 0) or 0:>12,} downloads", m.id) for m in models]
                err = None
            except Exception as e:  # noqa: BLE001 - shown to the user
                rows, err = [], f"{type(e).__name__}: {e}"
            self.call_from_thread(found, rows, err)

        def found(rows, err) -> None:
            if err or not rows:
                self.notify(f"The search failed: {err}" if err else f"No text-generation model on Hugging Face matches {words!r}.",
                            severity="warning", timeout=10)
                return

            def chosen(v: str | None) -> None:
                if v and self.set_target(v) and then:
                    then()
            self.push_screen(PickerScreen(f"Hugging Face models matching {words!r} (the one you pick is downloaded when the session "
                                          "starts)", rows), chosen)
        self.run_worker(work, thread=True)

    # ------------------------------------------------------------------------------------------------ permission requests
    def ask_permission(self, ev: dict) -> None:
        self.perm_open = ev.get("id")
        d = self.session

        def answered(result) -> None:
            decision, feedback = result or ("reject", "")
            runs.answer_permission(d, ev.get("id"), decision, feedback)
            self.pending_perm = None
        self.push_screen(PermissionScreen(ev.get("id"), ev.get("kind", "python"), ev.get("body", "")), answered)

    # ------------------------------------------------------------------------------------------------ commands
    def run_command(self, text: str) -> None:
        name, _, arg = text[1:].partition(" ")
        name, arg = ALIASES.get(name.lower(), name.lower()), arg.strip()
        handler = getattr(self, f"cmd_{name}", None)
        if handler is None:
            self.notify(f"There's no /{name}. /help lists the commands.", severity="warning")
            return
        handler(arg)

    def need_session(self, interactive: bool = False) -> bool:
        if not self.session:
            self.notify("That needs an open session.")
            return False
        if interactive and not (self.info or {}).get("interactive"):
            self.notify("That works in a conversation, not in a benchmark run.")
            return False
        return True

    def cmd_help(self, arg: str) -> None:
        t = Table.grid(padding=(0, 2))
        for name, args, desc in COMMANDS:
            t.add_row(Text(f"/{name} {args}".strip(), style="bold"), Text(desc))
        t.add_row("", "")
        for key, what in HELP_KEYS:
            t.add_row(Text(key, style="cyan"), Text(what))
        self.push_screen(InfoScreen("Commands and keys", t))

    def cmd_new(self, arg: str) -> None:
        self.call_after_refresh(self.new_conversation)

    def action_new(self) -> None:
        self.cmd_new("")

    def cmd_sessions(self, arg: str) -> None:
        rows = runs.list_runs(self.runs_dir)
        if not rows:
            self.notify("No sessions in this folder yet.")
            return
        opts = [(f"{r['name']:<26} {r['state']:<9} {r['target'] or '':<18} {r['title']}", r["path"]) for r in rows]
        self.push_screen(PickerScreen("Sessions", opts), lambda v: v and self.call_after_refresh(self.open_session, v))

    def action_sessions(self) -> None:
        self.cmd_sessions("")

    def cmd_target(self, arg: str) -> None:
        if self.session:
            self.notify("A session keeps its target. /new starts a conversation about another network.")
        elif arg:
            self.set_target(arg, search=True)
        else:
            self.pick_target()

    def cmd_search(self, arg: str) -> None:
        if self.session:
            self.notify("A session keeps its target. /new starts a conversation about another network.")
        elif arg:
            self.search_hub(arg)
        else:
            self.notify("Say what to look for: /search qwen 0.5b", severity="warning")

    def cmd_model(self, arg: str) -> None:
        self.action_model()

    def cmd_effort(self, arg: str) -> None:
        self.set_effort(arg) if arg else self.pick_effort()

    def cmd_mode(self, arg: str) -> None:
        self.set_mode(arg or ("plan" if (self.info["mode"] if self.info else self.mode) == "build" else "build"))

    def cmd_permissions(self, arg: str) -> None:
        self.set_perm(arg) if arg else self.pick_perm()

    def cmd_budget(self, arg: str) -> None:
        if not arg.isdigit() or int(arg) < 1:
            self.notify("Give the number of forward passes: /budget 30", severity="warning")
        elif self.session:
            if self.need_session(interactive=True):
                runs.send_setting(self.session, "budget", int(arg))
                if self.info:
                    self.info["budget"] = self.overlay["budget"] = int(arg)
                self.notify(f"Forward-pass budget: {arg}.")
        else:
            self.budget = int(arg)
            self.notify(f"The next session gets {arg} forward passes.")

    def cmd_todo(self, arg: str) -> None:
        self.action_toggle_todo()

    def action_toggle_todo(self) -> None:
        self.todo.hidden_by_user = not getattr(self.todo, "hidden_by_user", False)
        self.todo.show(getattr(self.todo, "items", []))

    def cmd_sidebar(self, arg: str) -> None:
        self.action_toggle_side()

    def action_toggle_side(self) -> None:
        side = self.query_one("#side", Static)
        side.display = not side.display

    def action_toggle_steps(self) -> None:
        tools = list(self.transcript.query(ToolCall))
        collapse = any(not t.collapsed for t in tools)
        for t in tools:
            t.collapsed = collapse

    def background_info(self, title: str, work) -> None:
        """Open a window now and fill it when `work()` (in a thread) returns what to show."""
        screen = InfoScreen(title, "Working …")
        self.push_screen(screen)

        def run() -> None:
            try:
                body = work()
            except Exception as e:  # noqa: BLE001 - shown to the user
                body = Text(f"{type(e).__name__}: {e}", style="red")
            self.call_from_thread(lambda: screen.is_attached and screen.set_body(body))
        self.run_worker(run, thread=True)

    def cmd_diff(self, arg: str) -> None:
        if not self.need_session(interactive=True):
            return
        d = self.session

        def work():
            rows = runs.diff_session(d)
            if not rows:
                return "Nothing is edited yet."
            t = Table(box=None, padding=(0, 2))
            for col in ("Tensor", "Shape", "Entries changed", "Relative change"):
                t.add_column(col)
            for r in rows:
                t.add_row(Text(r["name"]), str(tuple(r["shape"])), f"{r['changed_entries']:,} of {r['entries']:,}", f"{r['relative_change']:.3%}")
            return t
        self.background_info("The current edit", work)

    def cmd_export(self, arg: str) -> None:
        if not self.need_session(interactive=True):
            return
        d, full = self.session, arg.lower() == "full"

        def work():
            info = runs.export_session(d, None, full)
            if not info["edited_tensors"]:
                return "Nothing is edited yet."
            return (f"{len(info['edited_tensors'])} edited tensors, {info['changed_entries_total']:,} entries changed.\n\nWritten to "
                    f"{info['folder']}\n  edit.safetensors   the edited tensors, whole, in their original types\n  manifest.json      "
                    "what changed" + ("\n  model/             the complete network with the edit applied" if full else
                                      "\n\n/export full also writes the complete patched network."))
        self.background_info("Export", work)

    def cmd_status(self, arg: str) -> None:
        if self.need_session():
            self.push_screen(InfoScreen("Session", describe(runs.run_info(self.session))))

    def cmd_notes(self, arg: str) -> None:
        if not self.need_session():
            return
        try:
            with sandbox.open_in(os.path.join(self.session, "work"), "notes.md") as f:
                text = f.read().decode("utf-8", "replace")
        except (OSError, sandbox.UnsafePath):
            text = "(no notebook yet)"
        self.push_screen(InfoScreen("The agent's notebook (notes.md)", text))

    def cmd_compact(self, arg: str) -> None:
        if self.need_session(interactive=True):
            E.send(self.session, "compact")
            self.notify("The conversation restarts from the agent's notebook at its next step.")

    def cmd_stop(self, arg: str) -> None:
        if self.need_session():
            if runs.is_running(self.session):
                runs.request_stop(self.session)
                self.notify("Stopping at the next safe point.")
            else:
                self.notify("Its process isn't running.")

    def cmd_results(self, arg: str) -> None:
        rows = runs.benchmark_rows()
        if not rows:
            self.push_screen(InfoScreen("Benchmark results", "None yet. /benchmark runs one; `wsp eval NAME` scores it."))
            return
        t = Table(box=None, padding=(0, 2))
        for col in ("Model", "Benchmark", "Result"):
            t.add_column(col)
        for r in rows:
            t.add_row(Text(r["name"]), Text(r["benchmark"]), Text(r["result"]))
        self.push_screen(InfoScreen("Benchmark results", t))

    def cmd_doctor(self, arg: str) -> None:
        def work():
            from ..doctor import run_checks
            t = Table(box=None, padding=(0, 2))
            for col in ("", "Item", "Status"):
                t.add_column(col)
            mark = {"ok": ("OK", "green"), "warn": ("Note", "yellow"), "fix": ("Fix", "red"), "info": ("-", "dim")}
            for c in run_checks():
                m, style = mark[c.status]
                t.add_row(Text(m, style=style), Text(c.name), Text(c.detail + (f"\n{c.fix}" if c.fix else "")))
            return t
        self.background_info("This machine", work)

    def cmd_benchmark(self, arg: str) -> None:
        parts = arg.split()
        if not parts or parts[0] not in runs.TASKS:
            self.push_screen(PickerScreen("Run a benchmark from the paper (10 forward passes; /benchmark TASK N for another number)",
                                          [(desc, name) for name, desc in targets.BENCHMARKS.items()]),
                             lambda v: v and self.cmd_benchmark(v))
            return
        task = parts[0]
        budget = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 10
        run_dir = os.path.join(self.runs_dir, runs.new_run_name(task))
        est = prices.estimate(task, budget, self.model)

        async def go() -> None:
            args = runs.run_args(run_dir, task, self.backend, self.model, self.effort, budget, [], *self.accepted_flags(self.backend))
            os.makedirs(run_dir, exist_ok=True)
            await self.open_session(run_dir)
            cost = f", about ${est['usd']:.2f} of API credit" if est["usd"] is not None and self.backend in API_BACKENDS else ""
            await self.add(Note(f"Benchmark {task}: {budget} forward passes. Going by our own runs: about {ago((est['minutes'] or 0) * 60)} "
                                f"on an RTX 4070 SUPER{cost}."))
            self.launch(run_dir, args, f"Starting {task}")
        self.with_consents(self.backend, lambda: self.call_after_refresh(go))

    def cmd_setup(self, arg: str) -> None:
        if arg not in runs.TASKS:
            self.notify("Which benchmark: /setup edge_backdoor or /setup qwen_math", severity="warning")
            return
        from ..setup_task import SIZES
        screen = InfoScreen(f"Preparing {arg}: downloads {SIZES[arg]}", "Starting …")
        self.push_screen(screen)

        def work() -> None:
            lines: list[str] = []
            p = subprocess.Popen([sys.executable, "-m", "wsp.cli", "setup", "--task", arg, "--yes"], cwd=runs.ROOT, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
                                 env=dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1"))
            for line in p.stdout:
                lines.append(line.rstrip())
                self.call_from_thread(lambda: screen.is_attached and screen.set_body("\n".join(lines[-200:])))
            p.wait()
            lines.append("Done." if p.returncode == 0 else f"It failed (exit code {p.returncode}).")
            self.call_from_thread(lambda: screen.is_attached and screen.set_body("\n".join(lines[-200:])))
        self.run_worker(work, thread=True)

    def cmd_theme(self, arg: str) -> None:
        self.action_change_theme()

    def cmd_quit(self, arg: str) -> None:
        self.exit()


def run_app(session: str | None = None) -> None:
    WspApp(session=session).run()


__all__ = ["WspApp", "run_app"]
