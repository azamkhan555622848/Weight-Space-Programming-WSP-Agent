# Driving Claude as the "editor" model (the paper's Codex CLI loop, ported)

The paper ran `gpt-6-astra` through the Codex CLI: one `codex exec` call per turn (prompt on stdin), `codex exec resume
<thread>` to continue, every built-in tool switched off, a one-line system instruction ("Answer the user's request."),
reasoning effort per call. The harness parsed one fenced block from each text reply, executed it itself, and sent the
output back as the next user message. Every turn was audited from the event stream as tool-free.

Two Claude backends reproduce this. Both keep the paper's protocol: the model never gets a real tool; it writes a
```python or ```forward block in plain text and the harness runs it.

## Backend A: Anthropic API (recommended for long unattended runs)

- Model `claude-opus-5-5` (1M context, 128K output). Thinking is always on (adaptive); `{"type": "disabled"}` and
  `budget_tokens` return 400. Depth is set with `output_config={"effort": "low|medium|high|xhigh|max"}`; the default
  on Opus 5.5 is `medium`, so set it explicitly (paper: `high` for Exp 1/2/4, `xhigh` for 3/5/6/7, `max` for 8).
- Stream every call (`client.messages.stream(...)` + `.get_final_message()`), `max_tokens` ~64000.
- Prompt caching: top-level `cache_control={"type": "ephemeral"}` caches the growing conversation prefix. Verify with
  `usage.cache_read_input_tokens`. Keep the system prompt byte-stable.
- Append-only history: append `response.content` (including thinking blocks) unchanged; never edit earlier turns
  (preserved-thinking check). The paper's "handoff" (fresh conversation every N turns with task + notebook + current
  edit) bounds context without compaction.
- Refusals: `stop_reason == "refusal"` (check before reading content). Server-side fallback can be enabled with
  `betas=["server-side-fallback-2026-07-01"], fallbacks="default"`. The paper hit a provider content filter once
  (p50c turn 6, printing vocabulary tokens); its harness told the model the output could not be delivered and why.
- Errors: the SDK retries 408/409/429/5xx (raise `max_retries`); catch `RateLimitError` -> `APIStatusError` ->
  `APIConnectionError` most-specific first.
- Price (cached 2026-10-06): Opus 5.5 $4 / $20 per MTok in/out, cache reads $0.20.
- Auth: `ANTHROPIC_API_KEY`, or an `ant auth login` profile. None was set on this machine as of 2026-11-04.

## Backend B: Claude Code CLI (`claude -p`, uses the logged-in subscription)

Verified against the installed `claude --help` (Claude Code 2.1.318):

| Need | Flag |
|---|---|
| No built-in tools | `--tools ""` ("Use \"\" to disable all tools") |
| No MCP tools | `--disallowedTools "mcp__*"` and `--strict-mcp-config` |
| No CLAUDE.md / skills / plugins / hooks / MCP / memory | `--safe-mode` (do NOT use `--bare`: it skips OAuth, so subscription login stops working) |
| Replace the system prompt | `--system-prompt "Answer the user's request."` (recorded once per conversation; `--system-prompt-snapshot off` rebuilds each request) |
| Reasoning effort | `--effort low|medium|high|xhigh|max` (env `CLAUDE_CODE_EFFORT_LEVEL` overrides it) |
| Model | `--model claude-opus-5-5` |
| One text turn only | `--max-turns 1` |
| New conversation with a known id | `--session-id <uuid>` |
| Continue a conversation | `--resume <uuid>` (re-pass every flag each call) |
| Machine-readable output | `-p --output-format stream-json --verbose` (result object is the last line) |
| Never compact | env `DISABLE_COMPACT=1`; detect `system`/`compact_boundary` events |

Call it from Python with an argument list and `input=prompt.encode("utf-8")`, never through a PowerShell pipe (UTF-8
and empty-argument mangling). Remove `ANTHROPIC_API_KEY` from the child env if you want the subscription, since `-p`
always prefers the key when present.

Per-turn tool-free audit (the paper's `audit.tool_free`), all must hold:
1. `system/init` event: `tools` empty and `mcp_servers` empty;
2. no assistant content block of type `tool_use`; no user block of type `tool_result`;
3. result `permission_denials` empty; `num_turns == 1`; `subtype == "success"`;
4. no `compact_boundary` event.

Result fields: `session_id`, `result`, `subtype`, `is_error`, `num_turns`, `total_cost_usd` (a list-price estimate,
not subscription billing), `usage` (`input_tokens`, `output_tokens`, `cache_*`), `duration_ms`, `stop_reason`.
On a resumed session the cost fields report the whole session, so read the latest result instead of summing.

Unverified on this machine (smoke-test before relying on them): the exact stream-json key names; that `--effort`
takes effect in `-p`; stdin-only prompt with no positional argument.

### Terms caveat

Subscription limits "assume ordinary, individual usage of Claude Code and the Agent SDK" (legal-and-compliance page).
A 10-20 hour, 100+-turn automated run is better done on an API key (Backend A). Backend B is fine for development,
smoke tests and short runs. Rate-limit hits show as `system/api_retry` with `error: "rate_limit"`; the driver must
back off and resume.

## Not chosen: Claude Agent SDK with real tools

`claude-agent-sdk` can expose `run_python` / `forward_pass` as in-process MCP tools. This changes the protocol (tool
calls instead of fenced blocks), so results would be less comparable with the paper, and the docs point SDK apps at
API-key auth. Kept as a possible later variant.

Sources: code.claude.com/docs/en/{cli-reference, headless, env-vars, model-config, costs, legal-and-compliance,
agent-sdk/*}.md; the Claude API skill reference (models, thinking/effort, caching, refusals, errors).
