"""The model backends, with no network and no accounts: fake `claude` and `codex` programs on PATH (started the way the real
ones are, including through a .cmd file on Windows) and a stand-in for the OpenAI client. Checks what is sent, what is read
back, what stops a run, and that subscription modes never pass on an API key.

    python tests/test_backends.py
"""
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace as NS

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import httpx  # noqa: E402
import openai  # noqa: E402

from wsp import llm  # noqa: E402
from wsp.llm import (AnthropicBackend, BackendNotReady, ClaudeCLIBackend, CodexCLIBackend, Interrupted, OpenAIBackend,  # noqa: E402
                     StopRequested)

ok = True


def check(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(("PASS " if cond else "FAIL ") + name + ("" if cond else f"\n     {str(detail)[:700]}"))


KEYS = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "CLAUDE_CODE_USE_BEDROCK", "OPENAI_API_KEY", "CODEX_API_KEY")

FAKE_COMMON = r'''
import json, os, sys, time
ctl = os.environ["FAKE_CTL"]
mode = open(os.path.join(ctl, "mode.txt")).read().strip()
calls = os.path.join(ctl, "calls.jsonl")
n = sum(1 for _ in open(calls)) if os.path.exists(calls) else 0
prompt = sys.stdin.buffer.read().decode("utf-8")
open(calls, "a").write(json.dumps({"argv": sys.argv[1:], "prompt": prompt, "cwd": os.getcwd(),
                                   "env_keys": [k for k in %r if k in os.environ]}) + "\n")
def out(**e):
    print(json.dumps(e), flush=True)
''' % (KEYS,)

FAKE_CLAUDE = FAKE_COMMON + r'''
if mode == "hang":
    time.sleep(600)
if mode == "fail_once" and n == 0:
    print("Session ID is already in use", file=sys.stderr); sys.exit(1)
if mode == "lost":
    out(type="result", is_error=True, result="No conversation found with session ID x"); sys.exit(1)
if mode != "no_init":
    out(type="system", subtype="init", model="claude-fake", mcp_servers=[], tools=["Bash"] if mode == "tools_offered" else [])
if "--include-partial-messages" in sys.argv:
    for piece in ("reply ", "to: "):
        out(type="stream_event", event={"type": "content_block_delta", "delta": {"type": "text_delta", "text": piece}})
blocks = [{"type": "thinking", "thinking": "hm"}, {"type": "text", "text": "reply to: " + prompt}]
if mode == "tool_used":
    blocks.append({"type": "tool_use", "name": "Bash", "input": {}})
out(type="assistant", message={"content": blocks})
out(type="result", result="reply to: " + prompt, num_turns=1, stop_reason="end_turn",
    usage={"input_tokens": 7, "output_tokens": 3, "cache_read_input_tokens": 5, "cache_creation_input_tokens": 2})
'''

FAKE_CODEX = FAKE_COMMON + r'''
st_path = os.path.join(ctl, "thread.json")
st = json.load(open(st_path)) if os.path.exists(st_path) else {"in": 0, "out": 0, "cached": 0}
resuming = sys.argv[1:3] == ["exec", "resume"]
if mode == "flagged":
    out(type="error", message="Invalid prompt: your prompt was flagged as potentially violating our usage policy."); sys.exit(1)
if mode == "gone" and resuming:
    print("Error: no rollout found for thread id x", file=sys.stderr); sys.exit(1)
if not resuming:
    out(type="thread.started", thread_id="thread-1")
out(type="turn.started")
out(type="item.completed", item={"id": "i0", "type": "error", "message": "Code Mode is unavailable"})
out(type="item.completed", item={"id": "i1", "type": "reasoning", "text": "thinking"})
if mode == "tool":
    out(type="item.started", item={"id": "i2", "type": "command_execution", "command": "ls", "status": "in_progress"})
out(type="item.completed", item={"id": "i3", "type": "agent_message", "text": "the explanation"})
out(type="item.completed", item={"id": "i4", "type": "agent_message", "text": "the block"})
if mode != "stale":
    st = {"in": st["in"] + 100, "out": st["out"] + 10, "cached": st["cached"] + 40}
json.dump(st, open(st_path, "w"))
out(type="turn.completed", usage={"input_tokens": st["in"], "cached_input_tokens": st["cached"], "output_tokens": st["out"],
                                  "cache_write_input_tokens": 0, "reasoning_output_tokens": 0})
'''


class Fakes:
    def __init__(self):
        self.dir = tempfile.mkdtemp()
        self.ctl = os.path.join(self.dir, "ctl")
        os.makedirs(self.ctl)
        for name, src in (("claude", FAKE_CLAUDE), ("codex", FAKE_CODEX)):
            py = os.path.join(self.dir, name + ".py")
            open(py, "w", encoding="utf-8").write(src)
            if os.name == "nt":                # the way npm installs the real ones on Windows
                open(os.path.join(self.dir, name + ".cmd"), "w").write(f'@"{sys.executable}" "%~dp0{name}.py" %*\r\n')
            else:
                exe = os.path.join(self.dir, name)
                open(exe, "w").write(f'#!/bin/sh\nexec "{sys.executable}" "{py}" "$@"\n')
                os.chmod(exe, os.stat(exe).st_mode | stat.S_IEXEC)
        os.environ["FAKE_CTL"] = self.ctl
        os.environ["PATH"] = self.dir + os.pathsep + os.environ["PATH"]

    def reset(self, mode="ok"):
        for f in os.listdir(self.ctl):
            os.remove(os.path.join(self.ctl, f))
        open(os.path.join(self.ctl, "mode.txt"), "w").write(mode)

    def calls(self):
        p = os.path.join(self.ctl, "calls.jsonl")
        return [json.loads(x) for x in open(p)] if os.path.exists(p) else []


def claude_tests(f, run):
    def fresh(mode="ok", **kw):
        f.reset(mode)
        b = ClaudeCLIBackend(run, effort="high", log=lambda s: None, **kw)
        b.new_conversation()
        return b
    b = fresh()
    r = b.send("hello é")
    c = f.calls()[0]
    a = c["argv"]
    check("claude: reply, model and usage are read", r.ok and r.text == "reply to: hello é" and r.model == "claude-fake"
          and r.usage["input_tokens"] == 7 and r.usage["cache_read_input_tokens"] == 5, r)
    check("claude: tools off, as an empty argument even through a .cmd file", a[a.index("--tools") + 1] == "" and "--safe-mode" in a, a)
    check("claude: confirmed free of tools", r.audit["tool_free"] is True, r.audit)
    check("claude: no API keys or provider settings reach it", c["env_keys"] == [], c["env_keys"])
    check("claude: runs in an empty folder outside the project", not llm.os.path.realpath(c["cwd"]).lower().startswith(ROOT.lower())
          and os.listdir(c["cwd"]) == [], c["cwd"])
    b.send("again")
    a2 = f.calls()[1]["argv"]
    check("claude: the next turn resumes the session", a2[a2.index("--resume") + 1] == b.session and "--session-id" not in a2, a2)
    for mode, why in (("tools_offered", "tools offered"), ("no_init", "nothing says no tools were offered"), ("tool_used", "a tool used")):
        r = fresh(mode).send("x")
        check(f"claude: not confirmed free of tools when {why}", r.ok and r.audit["tool_free"] is False, r.audit)
    b = fresh("fail_once")
    first = b.session
    r = b.send("x")
    a = f.calls()[1]["argv"]
    check("claude: a retry of a first turn uses a new session id", r.ok and a[a.index("--session-id") + 1] != first and len(r.waits) == 1, a)
    b = fresh("lost")
    b.started = True
    r = b.send("x")
    check("claude: a lost session asks for a handoff", not r.ok and r.context_lost, r)
    b = fresh("hang", call_timeout=2, max_attempts=1)
    t0 = time.time()
    r = b.send("x")
    check("claude: a stuck call is stopped", not r.ok and time.time() - t0 < 60, time.time() - t0)


def streaming_tests(f, run):
    """Replies arrive in pieces, and the user can abort a reply that is still coming."""
    f.reset("ok")
    b = ClaudeCLIBackend(run, log=lambda s: None)
    b.new_conversation()
    got = []
    r = b.send("hello", on_delta=got.append)
    check("claude: pieces of the reply are passed on as they arrive", r.ok and got == ["reply ", "to: "] and r.audit["tool_free"] is True, got)
    logged = [json.loads(x)["type"] for x in open(os.path.join(run, "cli_events.jsonl"), encoding="utf-8")]
    check("claude: the pieces are not kept twice in the log", "stream_event" not in logged and "assistant" in logged, logged[-6:])
    f.reset("hang")
    b = ClaudeCLIBackend(run, log=lambda s: None)
    b.new_conversation()
    t0 = time.time()
    b.abort_check = lambda: "interrupt" if time.time() - t0 > 1.5 else None
    try:
        b.send("x")
        check("claude: an interrupt abandons the call", False)
    except Interrupted:
        check("claude: an interrupt abandons the call", time.time() - t0 < 30, time.time() - t0)
    f.reset("ok")
    b = CodexCLIBackend(run, log=lambda s: None)
    b.new_conversation()
    got = []
    b.send("hello", on_delta=got.append)
    check("codex: each message is passed on when it is complete", "".join(got).split() == ["the", "explanation", "the", "block"], got)

    # the Anthropic API backend, with a stand-in for the client's stream
    os.environ["ANTHROPIC_API_KEY"] = "decoy-not-a-key"
    a = AnthropicBackend(tempfile.mkdtemp(), log=lambda s: None)

    class Block:
        def __init__(self, **d):
            self.d = d

        def model_dump(self, exclude_none=True):
            return self.d

    final = NS(content=[Block(type="thinking", thinking="hm", signature="s"), Block(type="text", text="Hello world")],
               usage=NS(input_tokens=10, output_tokens=4, cache_read_input_tokens=7, cache_creation_input_tokens=2),
               stop_reason="end_turn", model="claude-served", _request_id="req_1", stop_details=None)
    events = [NS(type="message_start"), NS(type="content_block_delta", delta=NS(type="thinking_delta", thinking="hm")),
              NS(type="content_block_delta", delta=NS(type="text_delta", text="Hello")), NS(type="text", text="Hello"),
              NS(type="content_block_delta", delta=NS(type="text_delta", text=" world")), NS(type="message_stop")]
    calls = []

    class Stream:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            Stream.closed = True
            return False

        def __iter__(self):
            return iter(events)

        def get_final_message(self):
            return final

    def stream(**kw):
        calls.append(kw)
        return Stream()
    a.client = NS(beta=NS(messages=NS(stream=stream)), messages=NS(stream=stream))
    a.new_conversation()
    got = []
    r = a.send("hi", on_delta=got.append)
    check("anthropic: the text arrives in pieces, once each", r.ok and got == ["Hello", " world"] and r.text == "Hello world", got)
    check("anthropic: usage, model and no tools", r.usage["cache_read_input_tokens"] == 7 and r.model == "claude-served"
          and r.audit["tool_free"] is True and Stream.closed, r)
    check("anthropic: no tools are sent, thinking is adaptive", "tools" not in calls[0] and calls[0]["thinking"] == {"type": "adaptive"}
          and calls[0]["messages"][-1] == {"role": "user", "content": "hi"}, calls[0])
    a.abort_check = lambda: "interrupt"
    before = len(a.messages)
    try:
        a.send("more")
        check("anthropic: an interrupt abandons the reply and keeps the history as it was", False)
    except Interrupted:
        check("anthropic: an interrupt abandons the reply and keeps the history as it was", len(a.messages) == before, len(a.messages))
    a.abort_check = lambda: "stop"
    try:
        a.send("more")
        check("anthropic: a stop does the same", False)
    except StopRequested:
        check("anthropic: a stop does the same", len(a.messages) == before)
    final.content.append(Block(type="tool_use", name="bash", input={}))
    a.abort_check = None
    r = a.send("again")
    check("anthropic: a tool block in a reply means not free of tools", r.audit["tool_free"] is False, r.audit)
    del os.environ["ANTHROPIC_API_KEY"]


def codex_tests(f, run):
    def fresh(mode="ok"):
        f.reset(mode)
        b = CodexCLIBackend(run, effort="max", log=lambda s: None)
        b.new_conversation()
        return b
    b = fresh()
    r = b.send("hello é")
    c = f.calls()[0]
    a = c["argv"]
    check("codex: every message of the turn is kept", r.ok and r.text == "the explanation\n\nthe block", r.text)
    check("codex: no API keys reach it", c["env_keys"] == [], c["env_keys"])
    check("codex: the prompt goes on stdin, and it ignores user config", a[:2] == ["exec", "--ignore-user-config"] and a[-1] == "-"
          and c["prompt"] == "hello é" and "--ignore-rules" in a and "--json" in a, a)
    settings = [a[i + 1] for i, v in enumerate(a) if v == "-c"]
    must = ["features.shell_tool=false", "features.apps=false", "features.browser_use=false", "web_search=disabled",
            "sandbox_mode=read-only", "approval_policy=never", "model_reasoning_effort=max", "agents.enabled=false"]
    check("codex: tools are off", all(m in settings for m in must), settings)
    check("codex: the instructions file holds only the neutral line", open(b.instructions).read() == llm.SYSTEM)
    check("codex: confirmed free of tools despite the code-host notice", r.audit["tool_free"] is True, r.audit)
    check("codex: usage split into new and cached", r.usage["input_tokens"] == 60 and r.usage["cache_read_input_tokens"] == 40
          and r.usage["output_tokens"] == 10, r.usage)
    r2 = b.send("again")
    a2 = f.calls()[1]["argv"]
    check("codex: the next turn resumes the thread", a2[:2] == ["exec", "resume"] and a2[-2:] == ["thread-1", "-"], a2)
    check("codex: cumulative usage becomes per-turn", r2.usage["input_tokens"] == 60 and r2.usage["output_tokens"] == 10
          and not r2.context_lost, r2.usage)
    b.new_conversation()
    a3_args = b._args()
    check("codex: a new conversation starts a new thread", "resume" not in a3_args, a3_args)
    r = fresh("tool").send("x")
    check("codex: a tool starting means not free of tools", r.ok and r.audit["tool_free"] is False, r.audit)
    r = fresh("flagged").send("x")
    check("codex: a flagged prompt is a refusal", r.ok and r.refused, r)
    b = fresh()
    b.send("x")
    open(os.path.join(f.ctl, "mode.txt"), "w").write("stale")
    r = b.send("y")
    check("codex: usage not growing asks for a handoff (the paper's lost-rollout bug)", r.ok and r.context_lost, r)
    b = fresh()
    b.send("x")
    open(os.path.join(f.ctl, "mode.txt"), "w").write("gone")
    r = b.send("y")
    check("codex: a missing thread asks for a handoff", not r.ok and r.context_lost, r)


class Item:
    def __init__(self, **d):
        self.d = d

    def model_dump(self, exclude_none=True):
        return self.d


def response(rid="r1", status="completed", output=None, reason=None, error=None):
    output = [Item(type="reasoning", id="x"), Item(type="message", content=[{"type": "output_text", "text": "hi there"}])] \
        if output is None else output
    usage = NS(input_tokens=100, output_tokens=20, input_tokens_details=NS(cached_tokens=30))
    return NS(id=rid, status=status, model="gpt-6-astra-served", output=output, usage=usage, error=error,
              incomplete_details=NS(reason=reason) if reason else None, output_text="hi there")


class FakeResponses:
    """create() returns queued; retrieve() walks through `script`, whose entries are responses or exceptions."""

    def __init__(self):
        self.creates, self.retrieves, self.cancels, self.create_errors, self.script = [], 0, 0, [], []

    def create(self, **kw):
        self.creates.append(kw)
        if self.create_errors:
            raise self.create_errors.pop(0)
        return NS(id=f"r{len(self.creates)}", status="queued")

    def retrieve(self, rid):
        self.retrieves += 1
        x = self.script.pop(0) if self.script else response(rid)
        if isinstance(x, Exception):
            raise x
        return x

    def cancel(self, rid):
        self.cancels += 1


def api_error(cls, status, code):
    req = httpx.Request("POST", "https://example.invalid")
    return cls("simulated", response=httpx.Response(status, request=req), body={"code": code})


def openai_tests(run):
    def fresh(model="gpt-6-astra"):
        fake = FakeResponses()
        b = OpenAIBackend(run, model=model, effort="max", poll_s=0, log=lambda s: None, client=NS(responses=fake))
        b.wait = lambda s: None
        b.new_conversation()
        return b, fake
    b, fake = fresh()
    fake.script = [NS(id="r1", status="in_progress"), response("r1")]
    r = b.send("hello")
    kw = fake.creates[0]
    check("openai: polls until final, and reads the reply", r.ok and r.text == "hi there" and fake.retrieves == 2
          and r.model == "gpt-6-astra-served", r)
    check("openai: instructions, effort, background and storage are sent", kw["instructions"] == llm.SYSTEM and kw["reasoning"] == {"effort": "max"}
          and kw["background"] is True and kw["store"] is True and "previous_response_id" not in kw, kw)
    check("openai: no tools are ever sent", "tools" not in kw, kw)
    check("openai: usage split into new and cached", r.usage["input_tokens"] == 70 and r.usage["cache_read_input_tokens"] == 30
          and r.usage["output_tokens"] == 20, r.usage)
    check("openai: confirmed free of tools", r.audit["tool_free"] is True, r.audit)
    b.send("again")
    check("openai: the next turn continues from the last response, with instructions again", fake.creates[1]["previous_response_id"] == "r1"
          and fake.creates[1]["instructions"] == llm.SYSTEM, fake.creates[1])
    log = [json.loads(x) for x in open(os.path.join(run, "conversations", b.conv_id + ".jsonl"), encoding="utf-8")]
    check("openai: the conversation is logged", [m["role"] for m in log] == ["user", "assistant"] * 2, log)
    b.new_conversation()
    b.send("new")
    check("openai: a new conversation has no previous response", "previous_response_id" not in fake.creates[2], fake.creates[2])
    b, fake = fresh()
    fake.script = [response("r1", output=[Item(type="function_call", name="x")])]
    r = b.send("x")
    check("openai: a tool call means not free of tools", r.ok and r.audit["tool_free"] is False, r.audit)
    b, fake = fresh()
    fake.script = [response("r1", output=[Item(type="message", content=[{"type": "refusal", "refusal": "no"}])])]
    r = b.send("x")
    check("openai: a refusal is reported, and not chained", r.refused and b.prev is None, r)
    b, fake = fresh()
    fake.script = [response("r1", status="incomplete", reason="content_filter")]
    r = b.send("x")
    check("openai: a content filter stop is a refusal", r.refused and b.prev is None, r)

    b, fake = fresh()
    conn = openai.APIConnectionError(request=httpx.Request("POST", "https://example.invalid"))
    fake.script = [conn, conn, response("r1")]
    r = b.send("x")
    check("openai: a dropped connection keeps polling the same response", r.ok and len(fake.creates) == 1 and fake.retrieves == 3, r)
    b, fake = fresh()
    fake.create_errors = [api_error(openai.RateLimitError, 429, "rate_limit_exceeded")]
    r = b.send("x")
    check("openai: a rate limit waits and retries", r.ok and len(r.waits) == 1 and r.waits[0]["kind"] == "limit", r)
    b, fake = fresh()
    fake.create_errors = [api_error(openai.RateLimitError, 429, "insufficient_quota")]
    r = b.send("x")
    check("openai: out of credit stops at once", not r.ok and len(fake.creates) == 1, r)
    b, fake = fresh()
    fake.script = [response("r1", status="failed", error=NS(code="server_error", message="boom")), response("r2")]
    r = b.send("x")
    check("openai: a failed response is retried", r.ok and len(fake.creates) == 2 and r.waits[0]["kind"] == "failure", r)
    b, fake = fresh()
    fake.create_errors = [api_error(openai.BadRequestError, 400, "previous_response_not_found")]
    r = b.send("x")
    check("openai: an expired conversation asks for a handoff", not r.ok and r.context_lost, r)
    b, fake = fresh()
    fake.script = [NS(id="r1", status="in_progress")] * 5

    def stop_now(seconds):
        raise StopRequested
    b.wait = stop_now
    stopped = False
    try:
        b.send("x")
    except StopRequested:
        stopped = True
    check("openai: stopping cancels the work on the server", stopped and fake.cancels == 1, fake.cancels)
    saved = os.environ.pop("OPENAI_API_KEY", None)
    try:
        try:
            OpenAIBackend(run, log=lambda s: None)
            check("openai: no key is a clear error before anything is paid for", False)
        except BackendNotReady as e:
            check("openai: no key is a clear error before anything is paid for", "OPENAI_API_KEY" in str(e), e)
    finally:
        if saved:
            os.environ["OPENAI_API_KEY"] = saved


def missing_cli(run):
    path = os.environ["PATH"]
    os.environ["PATH"] = tempfile.mkdtemp()
    try:
        for cls, word in ((ClaudeCLIBackend, "claude"), (CodexCLIBackend, "codex")):
            try:
                cls(run, log=lambda s: None)
                check(f"{word}: not installed is a clear error", False)
            except BackendNotReady as e:
                check(f"{word}: not installed is a clear error", word in str(e), e)
    finally:
        os.environ["PATH"] = path


def run_agent_gates():
    """scripts/run_agent.py, as a program, with an empty home folder and no terminal."""
    home = tempfile.mkdtemp()
    run = os.path.join(home, "run")
    env = dict(os.environ, HOME=home, USERPROFILE=home, PYTHONIOENCODING="utf-8")
    script = os.path.join(ROOT, "scripts", "run_agent.py")

    def go(*args):
        return subprocess.run([sys.executable, script, "--task", "edge_backdoor", "--run", run, *args], capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=env, stdin=subprocess.DEVNULL, timeout=300)
    p = go("--backend", "claude-cli", "--budget", "1", "--task-arg", "data_dir=" + os.path.join(home, "no_data"))
    check("run_agent: a subscription mode needs its notice accepted", p.returncode != 0 and "notice accepted" in p.stderr, p.stderr[-400:])
    launch = json.load(open(os.path.join(run, "launch.json"), encoding="utf-8"))
    check("run_agent: the launch record keeps the settings", launch["backend"] == "claude-cli" and launch["model"] == "claude-opus-5-5"
          and launch["cfg"]["budget"] == 1, launch)
    p = go("--backend", "anthropic", "--budget", "9", "--accept-subscription-notice")
    launch2 = json.load(open(os.path.join(run, "launch.json"), encoding="utf-8"))
    check("run_agent: a resume keeps the original settings", launch2 == launch and "resuming with this run's settings: claude-cli" in p.stdout, p.stdout[-300:])
    check("run_agent: missing task files are a clear error", p.returncode != 0 and "wsp setup" in p.stderr, p.stderr[-300:])
    shutil.rmtree(home, ignore_errors=True)


def main():
    for k in KEYS:
        os.environ[k] = "decoy-not-a-key"
    llm._backoff = lambda kind, attempt: 0.0
    f = Fakes()
    run = tempfile.mkdtemp()
    claude_tests(f, run)
    streaming_tests(f, run)
    codex_tests(f, run)
    openai_tests(run)
    missing_cli(run)
    for k in KEYS:
        os.environ.pop(k, None)
    run_agent_gates()
    print("ALL PASS" if ok else "SOME FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
