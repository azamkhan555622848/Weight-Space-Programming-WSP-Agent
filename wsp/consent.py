"""Notices the user must accept once, on this machine. Acceptance is stored in ~/.wsp/consent.json.

The check itself is in the harness, which refuses to start without it (fails closed in scripts and detached runs); this
module only asks the person and remembers the answer.
"""
from __future__ import annotations

import json
import os
import sys

NO_SANDBOX = "no-sandbox"

NO_SANDBOX_TEXT = """\
NO SANDBOX ON THIS SYSTEM

The model writes Python code and this tool runs it on your machine. Without a sandbox, that code has all of your user's
permissions: it could read your files, including .env and the login files of AI command-line tools, and start other
programs. The task's rules forbid this and every code block is logged in tool_calls.jsonl, but nothing enforces it.

Safer: run under Linux (or WSL2 on Windows) with bubblewrap installed, or on macOS; `wsp doctor` shows how. If you run
here anyway, use a dedicated API key with a spending limit, on a machine with nothing you would mind exposing."""


SUBSCRIPTION = "subscription"

SUBSCRIPTION_TEXT = """\
USING YOUR SUBSCRIPTION INSTEAD OF AN API KEY

This mode runs the provider's own command-line tool (claude or codex) with the login you already have there. This tool never
reads, copies or forwards that login, and the usage counts against your plan's limits.

Your provider's terms apply, and following them is up to you. We can't tell you this use is allowed:
- Anthropic's legal page says third-party developers may not route requests through Free, Pro, Max or Ultra plan credentials
  on behalf of their users, and points developers to API keys. It also says it may enforce this without notice. Its support
  article says third-party app usage draws from subscription limits, but doesn't say it is allowed.
- OpenAI's documentation calls API keys the recommended default for automation, and says nothing about third-party tools.

A long unattended run may hit your plan's limits, and could put your account at risk. An API key avoids both.

  https://code.claude.com/docs/en/legal-and-compliance
  https://support.claude.com/en/articles/15036540-use-the-claude-agent-sdk-with-your-claude-plan
  https://learn.chatgpt.com/docs/auth"""


def _path() -> str:
    return os.path.join(os.path.expanduser("~"), ".wsp", "consent.json")


def _load() -> dict:
    try:
        with open(_path(), encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def accepted(key: str) -> bool:
    return bool(_load().get(key))


def record(key: str) -> None:
    data = _load()
    data[key] = True
    os.makedirs(os.path.dirname(_path()), exist_ok=True)
    tmp = _path() + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, _path())


def confirm(key: str, text: str, assume_yes: bool = False) -> bool:
    """True if the notice is accepted: by a command-line flag (this run only), by an earlier answer on this machine, or
    by the person typing 'yes' now. Without a terminal to ask on, the answer is no."""
    if assume_yes or accepted(key):
        return True
    if not (sys.stdin and sys.stdin.isatty()):
        return False
    print(text + "\n")
    try:
        answer = input('Type "yes" to accept and continue: ')
    except EOFError:           # Windows reports an empty stdin as a terminal
        return False
    if answer.strip().lower() != "yes":
        return False
    record(key)
    return True
