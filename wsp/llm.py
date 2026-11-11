"""The editor model behind one interface: a conversation of text turns, with no tools.

The paper drove gpt-6-astra through `codex exec` / `codex exec resume`, every built-in tool off, a one-line system
instruction, reasoning effort per call; the harness parsed one fenced block per reply and ran it itself. These backends
reproduce that protocol (see research/harness_claude_driver.md and research/harness_exp8.md):

  With an API key (billed per token; recommended for long runs):
  AnthropicBackend  the Messages API (claude-opus-5-5, adaptive thinking, explicit effort, streaming, prompt caching,
                    append-only history on disk).
  OpenAIBackend     the Responses API (gpt-6-astra, explicit effort, background mode so a long reply survives a dropped
                    connection, previous_response_id for the conversation).

  With the user's own subscription, through the provider's own command-line tool. The login stays with that tool: this code
  never reads, copies or forwards it (wsp/consent.py has the notice the user accepts first):
  ClaudeCLIBackend  `claude -p --tools "" --safe-mode --system-prompt ... --effort ...`, `--resume <session>` for later turns.
  CodexCLIBackend   `codex exec` / `codex exec resume <thread>` with all tools switched off, as the paper ran it.

All return a Reply with the text, token usage, the model that served the turn and an audit of what the reply contained. The
harness stops a run unless the audit positively confirms the reply used no tools.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field

from .procs import group_kwargs, kill_tree

SYSTEM = "Answer the user's request."
# What a reply may contain. An allowlist, so anything new from a provider stops the run instead of passing unnoticed.
REPLY_BLOCKS = {"text", "thinking", "redacted_thinking"}


@dataclass
class Reply:
    text: str
    ok: bool = True                  # False: the call failed after all retries (harness decides what to do)
    refused: bool = False            # the provider declined to answer (safety classifier / content filter)
    stop_reason: str | None = None
    model: str | None = None         # the model that actually served the turn
    usage: dict = field(default_factory=dict)
    audit: dict = field(default_factory=dict)
    wall_s: float = 0.0
    waits: list = field(default_factory=list)
    error: str | None = None
    context_lost: bool = False       # the conversation can't be relied on: the harness starts a new one with a handoff


class BackendNotReady(Exception):
    """This backend can't work here yet (no key, CLI not installed). Raised before anything is paid for."""


class StopRequested(Exception):
    """The user asked the run to stop while a backend was waiting or the model was replying."""


class Interrupted(Exception):
    """The user asked the agent to stop what it is doing and wait for them (interactive sessions)."""


class Backend:
    name = "base"
    abort_check = None               # set by the harness: returns "stop", "interrupt" or None

    def check_abort(self) -> None:
        kind = self.abort_check() if self.abort_check else None
        if kind == "stop":
            raise StopRequested
        if kind == "interrupt":
            raise Interrupted

    def wait(self, seconds: float) -> None:
        """Sleep, but give up at once if the user stops or interrupts (usage-limit waits can last hours)."""
        end = time.time() + seconds
        while time.time() < end:
            self.check_abort()
            time.sleep(min(1.0, max(0.0, end - time.time())))

    def new_conversation(self) -> None:
        raise NotImplementedError

    def send(self, text: str, on_delta=None) -> Reply:
        """Send one message and return the reply. `on_delta(text)` is called with pieces of the reply as they arrive, where
        the backend can stream. Raises StopRequested or Interrupted if the user aborts; nothing was delivered then."""
        raise NotImplementedError

    def state(self) -> dict:
        """What must be saved in state.json to resume this conversation after a crash."""
        return {}

    def restore(self, st: dict) -> None:
        pass


def _backoff(kind: str, attempt: int) -> float:
    """The paper's retry policy: usage/rate limits wait 900 s x 1.5^k; other failures 60 s x 2^k (plus jitter)."""
    base = 900 * 1.5 ** attempt if kind == "limit" else 60 * 2 ** attempt
    return base + random.uniform(0, 10)


def _cli_env(**extra) -> dict:
    """The environment of a provider's CLI in subscription mode: without keys or provider settings, which would otherwise
    take precedence over the login and silently bill an API account or another provider."""
    drop = {"ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "CODEX_API_KEY"}
    env = {k: v for k, v in os.environ.items() if k not in drop and not k.startswith("CLAUDE_CODE_USE_")}
    env.update(extra)
    return env


def _cli_cwd(run_dir: str) -> str:
    """An empty folder for the CLI, outside any project, so it finds no instructions or settings to pick up; the same one for
    a run across restarts, because CLIs file their sessions by folder."""
    tag = hashlib.sha1(os.path.realpath(run_dir).encode()).hexdigest()[:12]
    d = os.path.join(tempfile.gettempdir(), f"wsp-cli-{tag}")
    os.makedirs(d, exist_ok=True)
    return d


def _run_cli(args: list[str], text: str, env: dict, cwd: str, timeout: float, on_line=None,
             check_abort=None) -> tuple[int, bytes, bytes]:
    """Run a CLI with the prompt on stdin. `on_line(bytes)` gets each line of its output as it arrives. On a timeout
    everything it started is stopped (-> return code -1); if `check_abort` raises, likewise, and the exception passes on."""
    import threading
    p = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=env, cwd=cwd,
                         **group_kwargs())
    out: list[bytes] = []
    err: list[bytes] = []

    def read_out():
        for line in p.stdout:
            out.append(line)
            if on_line:
                try:
                    on_line(line)
                except Exception:  # noqa: BLE001 - a display problem must not lose the reply
                    pass

    def feed():
        try:
            p.stdin.write(text.encode("utf-8"))
            p.stdin.close()
        except OSError:
            pass
    threads = [threading.Thread(target=read_out, daemon=True), threading.Thread(target=lambda: err.append(p.stderr.read()), daemon=True),
               threading.Thread(target=feed, daemon=True)]
    for t in threads:
        t.start()
    deadline, timed_out = time.time() + timeout, False
    try:
        while p.poll() is None:
            if time.time() > deadline:
                timed_out = True
                kill_tree(p)
                break
            if check_abort:
                check_abort()
            time.sleep(0.1)
    except BaseException:
        kill_tree(p)
        raise
    for t in threads[:2]:
        t.join(timeout=30)
    tail = f"\n[stopped after {timeout:.0f} s]".encode() if timed_out else b""
    return (-1 if timed_out else p.returncode), b"".join(out), b"".join(err) + tail


def _json_lines(data: bytes) -> list[dict]:
    events = []
    for line in data.decode("utf-8", "replace").splitlines():
        try:
            e = json.loads(line)
            if isinstance(e, dict):
                events.append(e)
        except ValueError:
            pass
    return events


# ======================================================================================================== API
class AnthropicBackend(Backend):
    name = "anthropic"
    drops_refused = True        # a refused exchange is not appended to the history (it would be re-sent every turn)

    def __init__(self, run_dir: str, model: str = "claude-opus-5-5", effort: str = "high", max_tokens: int = 64000,
                 fallbacks: bool = True, max_attempts: int = 8, log=print):
        import anthropic
        self.anthropic = anthropic
        self.client = anthropic.Anthropic(max_retries=2, timeout=3600.0)
        self.model, self.effort, self.max_tokens = model, effort, max_tokens
        self.fallbacks, self.max_attempts, self.log = fallbacks, max_attempts, log
        self.run_dir = run_dir
        self.conv_id: str | None = None
        self.messages: list = []

    # conversation history lives in <run>/conversations/<id>.jsonl, one message per line, append-only
    def _path(self) -> str:
        return os.path.join(self.run_dir, "conversations", f"{self.conv_id}.jsonl")

    def _append(self, msg: dict):
        self.messages.append(msg)
        os.makedirs(os.path.dirname(self._path()), exist_ok=True)
        with open(self._path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def new_conversation(self):
        self.conv_id = "conv-" + uuid.uuid4().hex[:8]
        self.messages = []

    def state(self):
        return {"backend": self.name, "conversation": self.conv_id}

    def restore(self, st):
        self.conv_id = st.get("conversation")
        p = self._path()
        self.messages = [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []
        if self.messages and self.messages[-1]["role"] == "user":
            self.messages.pop()          # the unanswered message is re-sent by the harness (state.json "next")
            with open(p, "w", encoding="utf-8") as f:
                f.writelines(json.dumps(m, ensure_ascii=False) + "\n" for m in self.messages)

    def _call(self, messages, on_delta=None):
        kw = dict(model=self.model, max_tokens=self.max_tokens, system=SYSTEM, messages=messages,
                  thinking={"type": "adaptive"}, output_config={"effort": self.effort},
                  cache_control={"type": "ephemeral", "ttl": "1h"})  # passes can take 30 min
        if self.fallbacks:
            manager = self.client.beta.messages.stream(betas=["server-side-fallback-2026-07-01"], fallbacks="default", **kw)
        else:
            manager = self.client.messages.stream(**kw)
        with manager as stream:
            for event in stream:
                # the raw events, which the plain and the beta stream share
                if on_delta and event.type == "content_block_delta" and getattr(event.delta, "type", "") == "text_delta":
                    on_delta(event.delta.text)
                self.check_abort()          # leaving the block closes the connection
            return stream.get_final_message()

    def send(self, text: str, on_delta=None) -> Reply:
        A = self.anthropic
        assert self.conv_id, "call new_conversation() first"
        msgs = self.messages + [{"role": "user", "content": text}]
        waits, t0 = [], time.time()
        for attempt in range(self.max_attempts):
            try:
                resp = self._call(msgs, on_delta)
                break
            except (A.RateLimitError, A.OverloadedError) as e:
                kind, err = "limit", f"{type(e).__name__}: {e}"
            except A.BadRequestError as e:           # not retryable: the request itself is wrong
                return Reply("", ok=False, error=f"BadRequestError: {e}", waits=waits, wall_s=time.time() - t0)
            except (A.AuthenticationError, A.PermissionDeniedError, A.NotFoundError) as e:
                return Reply("", ok=False, error=f"{type(e).__name__}: {e}", waits=waits, wall_s=time.time() - t0)
            except (A.APIStatusError, A.APIConnectionError, A.APITimeoutError) as e:
                kind, err = "failure", f"{type(e).__name__}: {e}"
            w = _backoff(kind, attempt)
            waits.append({"attempt": attempt + 1, "kind": kind, "wait_s": round(w), "error": err[:500]})
            self.log(f"  attempt {attempt + 1}: {kind} ({err[:200]}); waiting {w:.0f} s")
            self.wait(w)
        else:
            return Reply("", ok=False, error="all attempts failed", waits=waits, wall_s=time.time() - t0)

        content = [b.model_dump(exclude_none=True) for b in resp.content]
        types = [b["type"] for b in content]
        reply_text = "\n".join(b.get("text", "") for b in content if b["type"] == "text").strip()
        u = resp.usage
        usage = {"input_tokens": u.input_tokens, "output_tokens": u.output_tokens,
                 "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
                 "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0}
        audit = {"block_types": types, "tool_free": all(t in REPLY_BLOCKS for t in types),
                 "request_id": getattr(resp, "_request_id", None)}
        refused = resp.stop_reason == "refusal"
        if refused:
            details = getattr(resp, "stop_details", None)
            audit["refusal"] = details.model_dump() if details is not None else None
        else:
            # append-only history: the user message and the full assistant content (thinking blocks included).
            # A refused exchange is left out, so the history stays a prefix of what was sent and the declined text is
            # not re-sent on every later turn (the harness sends a replacement message instead, as the paper did).
            self._append({"role": "user", "content": text})
            self._append({"role": "assistant", "content": content if content else [{"type": "text", "text": ""}]})
        return Reply(reply_text, ok=True, refused=refused, stop_reason=resp.stop_reason, model=resp.model,
                     usage=usage, audit=audit, wall_s=round(time.time() - t0, 1), waits=waits)


class OpenAIBackend(Backend):
    name = "openai"
    drops_refused = True        # a refused reply isn't chained: the next turn continues from the last good response
    OUT_OF_CREDIT = {"insufficient_quota", "credit_balance_exhausted", "organization_spend_limit_exceeded",
                     "project_spend_limit_exceeded", "organization_usage_limit_exceeded"}

    def __init__(self, run_dir: str, model: str = "gpt-6-astra", effort: str = "high", max_attempts: int = 8,
                 poll_s: float = 5.0, call_timeout: float = 6 * 3600, log=print, client=None):
        try:
            import openai
        except ImportError:
            raise BackendNotReady('the openai package is not installed. Run: pip install -e ".[openai]"') from None
        self.openai = openai
        try:
            self.client = client or openai.OpenAI(max_retries=2, timeout=120.0)
        except openai.OpenAIError:
            raise BackendNotReady("OPENAI_API_KEY is not set. Add it to .env (see .env.example).") from None
        self.run_dir, self.model, self.effort = run_dir, model, effort
        self.max_attempts, self.poll_s, self.call_timeout, self.log = max_attempts, poll_s, call_timeout, log
        self.conv_id: str | None = None
        self.prev: str | None = None

    def new_conversation(self):
        self.conv_id, self.prev = "conv-" + uuid.uuid4().hex[:8], None

    def state(self):
        return {"backend": self.name, "conversation": self.conv_id, "previous_response_id": self.prev}

    def restore(self, st):
        self.conv_id, self.prev = st.get("conversation"), st.get("previous_response_id")

    def _append(self, msg: dict):
        p = os.path.join(self.run_dir, "conversations", f"{self.conv_id}.jsonl")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(msg, ensure_ascii=False) + "\n")

    def _create(self, text: str):
        # Background mode: the model works on the server and we poll, so a long reply doesn't depend on holding a connection
        # open. store=True is required for that and for previous_response_id. Instructions aren't carried over, so they are
        # sent every time.
        kw = dict(model=self.model, instructions=SYSTEM, input=text, reasoning={"effort": self.effort},
                  background=True, store=True)
        if self.prev:
            kw["previous_response_id"] = self.prev
        return self.client.responses.create(**kw)

    def _finish(self, resp):
        """Poll until the response is final; a passing network problem doesn't lose the work done on the server."""
        O = self.openai
        deadline, misses = time.time() + self.call_timeout, 0
        try:
            while resp.status in ("queued", "in_progress"):
                if time.time() > deadline:
                    raise TimeoutError(f"no reply after {self.call_timeout:.0f} s")
                self.wait(self.poll_s)
                try:
                    resp = self.client.responses.retrieve(resp.id)
                    misses = 0
                except (O.APIConnectionError, O.APITimeoutError, O.InternalServerError):
                    misses += 1
                    if misses > 60:
                        raise
                    self.wait(30)
        except (StopRequested, Interrupted, TimeoutError):
            try:
                self.client.responses.cancel(resp.id)
            except O.OpenAIError:
                pass
            raise
        return resp

    def send(self, text: str, on_delta=None) -> Reply:
        O = self.openai
        assert self.conv_id, "call new_conversation() first"
        waits, t0 = [], time.time()
        for attempt in range(self.max_attempts):
            try:
                resp = self._finish(self._create(text))
                if resp.status in ("completed", "incomplete"):
                    break
                code = str(getattr(resp.error, "code", "") or "")
                kind, err = ("limit" if "rate_limit" in code else "failure"), f"{resp.status}: {code} {resp.error}"
            except O.RateLimitError as e:
                if getattr(e, "code", None) in self.OUT_OF_CREDIT:
                    return Reply("", ok=False, error=f"RateLimitError: {e}", waits=waits, wall_s=time.time() - t0)
                kind, err = "limit", f"RateLimitError: {e}"
            except O.BadRequestError as e:           # not retryable: the request itself is wrong
                lost = getattr(e, "code", None) == "previous_response_not_found"
                return Reply("", ok=False, error=f"BadRequestError: {e}", waits=waits, wall_s=time.time() - t0, context_lost=lost)
            except (O.AuthenticationError, O.PermissionDeniedError, O.NotFoundError) as e:
                return Reply("", ok=False, error=f"{type(e).__name__}: {e}", waits=waits, wall_s=time.time() - t0)
            except (O.APIStatusError, O.APIConnectionError, O.APITimeoutError, TimeoutError) as e:
                kind, err = "failure", f"{type(e).__name__}: {e}"
            w = _backoff(kind, attempt)
            waits.append({"attempt": attempt + 1, "kind": kind, "wait_s": round(w), "error": err[:500]})
            self.log(f"  attempt {attempt + 1}: {kind} ({err[:200]}); waiting {w:.0f} s")
            self.wait(w)
        else:
            return Reply("", ok=False, error="all attempts failed", waits=waits, wall_s=time.time() - t0)

        items = [i.model_dump(exclude_none=True) for i in resp.output]
        types = [i["type"] for i in items]
        parts = [c.get("type") for i in items if i["type"] == "message" for c in i.get("content", [])]
        reason = getattr(resp.incomplete_details, "reason", None)
        refused = "refusal" in parts or reason == "content_filter"
        u = resp.usage
        cached = getattr(getattr(u, "input_tokens_details", None), "cached_tokens", 0) or 0
        usage = {"input_tokens": max(0, (getattr(u, "input_tokens", 0) or 0) - cached),
                 "output_tokens": getattr(u, "output_tokens", 0) or 0,
                 "cache_read_input_tokens": cached, "cache_creation_input_tokens": 0}
        audit = {"item_types": types, "content_types": parts, "response_id": resp.id,
                 "tool_free": all(t in ("message", "reasoning") for t in types)
                 and all(p in ("output_text", "refusal") for p in parts)}
        if not refused:
            self._append({"role": "user", "content": text})
            self._append({"role": "assistant", "response_id": resp.id, "output": items})
            self.prev = resp.id
        if on_delta and not refused:
            on_delta(resp.output_text or "")
        return Reply((resp.output_text or "").strip(), ok=True, refused=refused, stop_reason=reason or resp.status,
                     model=resp.model, usage=usage, audit=audit, wall_s=round(time.time() - t0, 1), waits=waits)


# ======================================================================================================== CLI
class ClaudeCLIBackend(Backend):
    name = "claude-cli"

    def __init__(self, run_dir: str, model: str = "claude-opus-5-5", effort: str = "high", max_attempts: int = 8,
                 call_timeout: float = 6 * 3600, log=print):
        self.run_dir, self.model, self.effort = run_dir, model, effort
        self.max_attempts, self.call_timeout, self.log = max_attempts, call_timeout, log
        self.exe = shutil.which("claude")
        if not self.exe:
            raise BackendNotReady("the `claude` command isn't installed or isn't on PATH. Install Claude Code and log in "
                                  "by running `claude` once: https://code.claude.com/docs")
        self.session: str | None = None
        self.started = False           # whether the session has at least one completed turn

    def new_conversation(self):
        self.session, self.started = str(uuid.uuid4()), False

    def state(self):
        return {"backend": self.name, "session": self.session, "started": self.started}

    def restore(self, st):
        self.session, self.started = st.get("session"), st.get("started", False)

    def _args(self):
        a = [self.exe, "-p", "--output-format", "stream-json", "--verbose", "--include-partial-messages", "--model", self.model,
             "--effort", self.effort, "--safe-mode", "--tools", "", "--disallowedTools", "mcp__*",
             "--strict-mcp-config", "--system-prompt", SYSTEM, "--max-turns", "1", "--permission-mode", "dontAsk"]
        a += ["--resume", self.session] if self.started else ["--session-id", self.session]
        return a

    def send(self, text: str, on_delta=None) -> Reply:
        env = _cli_env(DISABLE_COMPACT="1", CLAUDE_CODE_DISABLE_CLAUDE_MDS="1", CLAUDE_CODE_DISABLE_AUTO_MEMORY="1")
        env.pop("CLAUDE_CODE_EFFORT_LEVEL", None)
        waits, t0 = [], time.time()
        cwd = _cli_cwd(self.run_dir)

        def on_line(line: bytes):
            if on_delta and b'"text_delta"' in line:
                d = (json.loads(line).get("event") or {}).get("delta") or {}
                if d.get("type") == "text_delta":
                    on_delta(d.get("text", ""))
        for attempt in range(self.max_attempts):
            rc, stdout, stderr = _run_cli(self._args(), text, env, cwd, self.call_timeout, on_line, self.check_abort)
            events = _json_lines(stdout)
            res = next((e for e in reversed(events) if e.get("type") == "result"), None)
            limit = any(e.get("type") == "system" and e.get("subtype") == "api_retry" and
                        e.get("error") == "rate_limit" for e in events)
            if res is not None and not res.get("is_error") and (res.get("result") or "").strip():
                break
            err = (res or {}).get("result") or stderr.decode("utf-8", "replace")[-500:]
            if "no conversation found" in str(err).lower():
                return Reply("", ok=False, error=str(err)[:500], waits=waits, wall_s=time.time() - t0, context_lost=True)
            kind = "limit" if limit or "limit" in str(err).lower() else "failure"
            w = _backoff(kind, attempt)
            waits.append({"attempt": attempt + 1, "kind": kind, "rc": rc, "wait_s": round(w), "error": str(err)[:500]})
            self.log(f"  attempt {attempt + 1}: {kind} (rc {rc}: {str(err)[:200]}); waiting {w:.0f} s")
            self.wait(w)
            if not self.started:      # a failed first attempt can leave its session id taken; a new conversation loses nothing
                self.session = str(uuid.uuid4())
        else:
            return Reply("", ok=False, error="all attempts failed", waits=waits, wall_s=time.time() - t0)

        self.started = True
        init = next((e for e in events if e.get("type") == "system" and e.get("subtype") == "init"), {})
        blocks = [b for e in events if e.get("type") in ("assistant", "user")
                  for b in (e.get("message", {}).get("content") or []) if isinstance(b, dict)]
        types = [b.get("type") for b in blocks]
        audit = {"init_tools": init.get("tools"), "init_mcp_servers": init.get("mcp_servers"),
                 "api_key_source": init.get("apiKeySource"),
                 "block_types": types, "num_turns": res.get("num_turns"), "subtype": res.get("subtype"),
                 "permission_denials": res.get("permission_denials"),
                 "compacted": any(e.get("subtype") == "compact_boundary" for e in events)}
        audit["tool_free"] = (init.get("tools") == [] and not init.get("mcp_servers")
                              and all(t in REPLY_BLOCKS for t in types)
                              and not res.get("permission_denials") and res.get("num_turns") == 1)
        with open(os.path.join(self.run_dir, "cli_events.jsonl"), "a", encoding="utf-8") as f:
            for e in events:
                if e.get("type") != "stream_event":          # the partial pieces repeat what the full messages hold
                    f.write(json.dumps(e, ensure_ascii=False) + "\n")
        u = res.get("usage") or {}
        return Reply(res.get("result", "").strip(), ok=True, refused=res.get("stop_reason") == "refusal",
                     stop_reason=res.get("stop_reason"), model=init.get("model"),
                     usage={k: u.get(k, 0) for k in ("input_tokens", "output_tokens", "cache_read_input_tokens",
                                                     "cache_creation_input_tokens")} |
                           {"session_cost_usd_estimate": res.get("total_cost_usd")},
                     audit=audit, wall_s=round(time.time() - t0, 1), waits=waits)


# ======================================================================================================== scripted
# All the tools the paper's harness/codex_config.toml turns off. They all still exist in Codex 0.167.
CODEX_TOOLS_OFF = ("code_mode", "code_mode_host", "collaboration_modes", "shell_tool", "unified_exec", "unified_exec_tty",
                   "shell_snapshot", "view_image", "image_generation", "multi_agent", "apps", "plugins", "remote_plugin",
                   "browser_use", "browser_use_external", "browser_use_full_cdp_access", "in_app_browser", "computer_use",
                   "skill_search", "skill_mcp_dependency_install", "sleep_tool", "tool_suggest", "goals", "hooks", "worktrees",
                   "workspace_dependencies")
# What a turn may contain. Codex reports an "error" item each turn when its code host is off, which the paper's runs show.
CODEX_ITEMS = {"agent_message", "reasoning", "error"}
# The Codex version these settings were tested with. Events show the tools a turn used, not the ones it was offered, so `wsp
# doctor` warns when the installed version differs. Testing live needs a ChatGPT login: None means not yet tested live.
CODEX_TESTED_VERSION = None


class CodexCLIBackend(Backend):
    name = "codex-cli"

    def __init__(self, run_dir: str, model: str = "gpt-6-astra", effort: str = "high", max_attempts: int = 8,
                 call_timeout: float = 6 * 3600, log=print):
        self.run_dir, self.model, self.effort = run_dir, model, effort
        self.max_attempts, self.call_timeout, self.log = max_attempts, call_timeout, log
        self.exe = shutil.which("codex")
        if not self.exe:
            raise BackendNotReady("the `codex` command isn't installed or isn't on PATH. Install Codex (npm install -g "
                                  "@openai/codex) and log in with `codex login`.")
        self.cwd = _cli_cwd(run_dir)
        self.instructions = os.path.join(self.cwd, "instructions.md")
        with open(self.instructions, "w", encoding="utf-8") as f:
            f.write(SYSTEM)
        self.thread: str | None = None
        self.last_usage: dict = {}

    def new_conversation(self):
        self.thread, self.last_usage = None, {}

    def state(self):
        return {"backend": self.name, "thread": self.thread, "usage": self.last_usage}

    def restore(self, st):
        self.thread, self.last_usage = st.get("thread"), st.get("usage") or {}

    def _args(self) -> list[str]:
        # --ignore-user-config skips config.toml but keeps the saved login. -c values without quotes stay plain strings.
        settings = {"model_reasoning_effort": self.effort, "approval_policy": "never", "sandbox_mode": "read-only",
                    "web_search": "disabled", "agents.enabled": "false", "model_auto_compact_token_limit": "1000000",
                    "model_instructions_file": self.instructions.replace("\\", "/"),
                    **{f"features.{f}": "false" for f in CODEX_TOOLS_OFF}}
        a = [self.exe, "exec"] + (["resume"] if self.thread else [])
        a += ["--ignore-user-config", "--ignore-rules", "--skip-git-repo-check", "--json", "-m", self.model]
        for k, v in settings.items():
            a += ["-c", f"{k}={v}"]
        return a + ([self.thread] if self.thread else []) + ["-"]

    def send(self, text: str, on_delta=None) -> Reply:
        env = _cli_env()
        waits, t0 = [], time.time()

        def on_line(line: bytes):
            if on_delta and b'"agent_message"' in line:
                e = json.loads(line)
                if e.get("type") == "item.completed" and (e.get("item") or {}).get("type") == "agent_message":
                    on_delta((e["item"].get("text") or "") + "\n\n")
        for attempt in range(self.max_attempts):
            rc, stdout, stderr = _run_cli(self._args(), text, env, self.cwd, self.call_timeout, on_line, self.check_abort)
            events = _json_lines(stdout)
            errors = [str(e.get("message") or (e.get("error") or {}).get("message"))
                      for e in events if e.get("type") in ("error", "turn.failed")]
            done = next((e for e in reversed(events) if e.get("type") == "turn.completed"), None)
            items = [e["item"] for e in events if e.get("type") == "item.completed" and isinstance(e.get("item"), dict)]
            # Every message of the turn: -o would keep only the last one, which cost the paper's runs their explanations
            # whenever the model sent the explanation and the block as two messages.
            reply = "\n\n".join(i.get("text", "") for i in items if i.get("type") == "agent_message").strip()
            if done is not None and reply:
                break
            err = " | ".join(errors) or stderr.decode("utf-8", "replace")[-500:]
            low = err.lower()
            if "usage policy" in low or "flagged" in low:
                return Reply("", refused=True, stop_reason="refusal", model=self.model, audit={"tool_free": True,
                             "errors": errors[:5]}, wall_s=round(time.time() - t0, 1), waits=waits)
            if self.thread and ("no rollout found" in low or "not found" in low):
                return Reply("", ok=False, error=err[:500], waits=waits, wall_s=time.time() - t0, context_lost=True)
            kind = "limit" if "limit" in low or "429" in low else "failure"
            w = _backoff(kind, attempt)
            waits.append({"attempt": attempt + 1, "kind": kind, "rc": rc, "wait_s": round(w), "error": err[:500]})
            self.log(f"  attempt {attempt + 1}: {kind} (rc {rc}: {err[:200]}); waiting {w:.0f} s")
            self.wait(w)
        else:
            return Reply("", ok=False, error="all attempts failed", waits=waits, wall_s=time.time() - t0)

        with open(os.path.join(self.run_dir, "cli_events.jsonl"), "a", encoding="utf-8") as f:
            for e in events:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
        started = next((e.get("thread_id") for e in events if e.get("type") == "thread.started"), None)
        cur = done.get("usage") or {}
        prev = self.last_usage if self.thread else {}
        # A turn's usage is cumulative over the thread, so the harness needs the difference. If a saved turn was lost, the
        # thread silently forgets recent work; in the paper's runs this showed as input not growing, so force a handoff.
        lost = bool(prev) and (cur.get("input_tokens", 0) <= prev.get("input_tokens", 0) or b"rollout writer failed" in stderr)
        diff = {k: max(0, cur.get(k, 0) - prev.get(k, 0)) for k in ("input_tokens", "cached_input_tokens", "output_tokens",
                                                                   "cache_write_input_tokens")}
        self.thread, self.last_usage = started or self.thread, cur
        types = [i.get("type") for i in items] + [e["item"].get("type") for e in events if e.get("type") == "item.started"
                                                  and isinstance(e.get("item"), dict)]
        audit = {"event_types": sorted({str(e.get("type")) for e in events}), "item_types": sorted({str(t) for t in types}),
                 "errors": errors[:5], "usage_cumulative": cur, "tool_free": all(t in CODEX_ITEMS for t in types)}
        usage = {"input_tokens": max(0, diff["input_tokens"] - diff["cached_input_tokens"]), "output_tokens": diff["output_tokens"],
                 "cache_read_input_tokens": diff["cached_input_tokens"],
                 "cache_creation_input_tokens": diff["cache_write_input_tokens"]}
        return Reply(reply, ok=True, stop_reason="turn.completed", model=self.model, usage=usage, audit=audit,
                     wall_s=round(time.time() - t0, 1), waits=waits, context_lost=lost)


class ScriptedBackend(Backend):
    """Replays prepared replies (tests and dry runs of the harness without any model calls). A reply may be a special word:
    __FAIL__, __LOST__, __TOOL__, __REFUSE__, or __SLOW__ (waits until aborted, then answers "I am done.")."""
    name = "scripted"
    drops_refused = True        # behaves like the API backend

    def __init__(self, replies: list[str], delay: float = 0.0):
        self.replies, self.i, self.received, self.delay = list(replies), 0, [], delay

    def new_conversation(self):
        self.received.append("<new conversation>")

    def send(self, text, on_delta=None):
        self.received.append(text)
        r = self.replies[self.i] if self.i < len(self.replies) else "I am done."
        self.i += 1
        if r == "__FAIL__":                         # simulate an outage (the harness stops; the run is resumable)
            return Reply("", ok=False, error="simulated outage")
        if r == "__LOST__":                         # the provider's session is gone (the harness opens a new one)
            return Reply("", ok=False, error="simulated lost session", context_lost=True)
        if r == "__TOOL__":
            return Reply("Reading a file first.\n\n```python\nprint('this must not run')\n```", model="scripted",
                         audit={"tool_free": False, "block_types": ["tool_use"]})
        if r == "__REFUSE__":
            return Reply("", refused=True, stop_reason="refusal", model="scripted", audit={"tool_free": True})
        if r == "__SLOW__":                         # a reply that never comes: only an abort ends it
            try:
                self.wait(600)
            except (StopRequested, Interrupted):
                self.i -= 1
                self.replies[self.i] = "I am done."
                self.received.pop()
                raise
        if self.delay:
            self.wait(self.delay)
        if on_delta:
            for k in range(0, len(r), 40):
                on_delta(r[k:k + 40])
        return Reply(r, model="scripted", usage={"input_tokens": len(text) // 4, "output_tokens": len(r) // 4},
                     audit={"tool_free": True})

    def state(self):
        return {"backend": self.name, "i": self.i}

    def restore(self, st):
        self.i = st.get("i", 0)


# The two ways to reach a model: with an API key, or through the provider's own CLI on the user's subscription.
API_BACKENDS = ("anthropic", "openai")
SUBSCRIPTION_BACKENDS = ("claude-cli", "codex-cli")
DEFAULT_MODELS = {"anthropic": "claude-opus-5-5", "claude-cli": "claude-opus-5-5", "openai": "gpt-6-astra", "codex-cli": "gpt-6-astra"}


def make_backend(kind: str, run_dir: str, model: str | None = None, effort: str = "high", log=print, **kw) -> Backend:
    model = model or DEFAULT_MODELS.get(kind)
    if kind == "anthropic":
        return AnthropicBackend(run_dir, model=model, effort=effort, log=log, **kw)
    if kind == "openai":
        return OpenAIBackend(run_dir, model=model, effort=effort, log=log, **kw)
    if kind == "claude-cli":
        return ClaudeCLIBackend(run_dir, model=model, effort=effort, log=log, **kw)
    if kind == "codex-cli":
        return CodexCLIBackend(run_dir, model=model, effort=effort, log=log, **kw)
    if kind == "scripted":            # tests and demos: replies from a JSON file (a list of strings), no model involved
        path = os.environ.get("WSP_SCRIPTED_REPLIES") or os.path.join(run_dir, "scripted_replies.json")
        done = os.path.join(run_dir, "state.json")
        b = ScriptedBackend(json.load(open(path, encoding="utf-8")), delay=float(os.environ.get("WSP_SCRIPTED_DELAY", "0")))
        if os.path.exists(done):       # a resumed session carries on where the script left off
            b.restore((json.load(open(done, encoding="utf-8")).get("llm") or {}))
        return b
    raise ValueError(f"unknown backend {kind!r}")
