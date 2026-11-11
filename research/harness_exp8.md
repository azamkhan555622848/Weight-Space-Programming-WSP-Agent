# The Experiment 8 agent harness (math weight editing), reverse-engineered from the released logs

Source: `lm_understand-logs-main/exp8_math/`. Paths below are relative to that folder unless they start with `../`.
Runs analysed: `runs/p10c`, `runs/p25c`, `runs/p50c` (Qwen3.5-0.8B, prompt v3) and `gemma_followup/runs/g05b`,
`g10b`, `g25b`, `g50b` (Gemma 3 270M IT, prompt v5). Companion spec for Experiments 1-7: `harness_exp1_7.md`.

Tags (same convention as `harness_exp1_7.md`):
- **[V]** verified: reproduced from the released files (bytes, hashes or counts matched by a script).
- **[Q]** quoted verbatim from a prompt or a harness message.
- **[I]** inferred from indirect evidence.
- **[unobserved]** the prompt says the code path exists, but it never fired in these 7 runs, so its text is unknown.

Already known and not repeated here: the turn-1 task prompt (`runs/p50c/turns/t001/prompt.txt`, it defines the
```` ```python ```` / ```` ```forward ```` protocol, budgets and rules), `README.md`, `notices/00*.txt`,
`../harness/sandbox.py` (the Go sandbox: bubblewrap + PRELUDE) and `../harness/codex_config.toml`.

---

## 0. Summary for implementers

- One Codex CLI call per turn (`codex exec` for a new thread, `codex exec resume <thread> -` otherwise), prompt on
  stdin, effort `max`, model `gpt-6-astra`, Codex tools off. The harness reads the reply from the `-o answer.md` file,
  which holds the **last agent message only**. That caused bug 1 in section 10.
- A reply holds one ```` ```python ```` block, one ```` ```forward ```` block, or none. "None" is the final answer
  only once all passes are used. Otherwise it is a "nudge", which never happened, so its text is unknown.
- The Python kernel is a persistent process: a bubblewrap sandbox with a driver script that runs user code as
  `<run>` and the prelude as `<prelude>`.
  - Output is cut to 10,000 characters, then repeated lines are removed, then non-ASCII characters are replaced
    with `?`. This pipeline is reproduced exactly in section 2.1 [V].
- The forward-pass server is a file queue: `requests/fwd_<k>.json`, renamed to `.json.claimed`, answered by
  `.done.json`.
  - `k` counts requests sent to the server. Passes are counted separately, and only successful ones count.
  - The summary sent to the model is `json.dumps(done, indent=1)`, verified byte-exact on all 175 passes [V].
  - The harness waits 6 h before giving up on a request.
- After a pass, the next message is `Forward pass {p} of {B} done ...` followed by `Forward passes left: {B-p}.`.
  Later harness versions also add `Edits used in this pass: ... (edit hash h).`. After the last pass the message adds
  `All forward passes are used. ...`.
- **Handoff** (new Codex thread): starts when `since_handoff` reaches 40, or on a forced restart.
  - The message is the current task text (`message.txt`), then a `THIS IS A RESTARTED CONVERSATION` block.
  - That block holds the counters, the edited tensors, `sorted(listdir('/work'))[:60]`, the last 8,000 characters of
    `notes.md`, and `Last result:` followed by the pending message.
- **EXPLAIN mode** (added mid-run in the p-runs, on from turn 1 in the Gemma runs):
  - A block with no text before it is sent back unrun, at most 2 times in a row. The third one runs, and its result
    gets a `(Reminder: ...)` line.
  - Every processed reply is logged to `explanations.md`.
- **Dedupe** (late p50c and all Gemma runs): `req_key = sha1(json.dumps(request, sort_keys=True) + "|" + edit_hash)`
  [V]. It maps to a pass number. An identical request with unchanged edits is "refused without cost". The refusal
  text was never observed.
- The worst harness problems in these logs:
  - Codex could not save its sessions (disk quota), so about 25 turns per p-run were answered with stale context.
  - A restart note was dropped whenever a recorded answer was reused.
  - Explanation checks used only the last agent message, so 12 of the 20 send-backs were false positives.
  - Details are in section 10.

---

## 1. The turn loop (reconstructed)

```
state = load_or_init("state.json")          # fields: section 4
kernel = start_kernel()                     # persistent; restarted with the harness
if harness was restarted and state exists:  # [V] from driver.log restarts
    state.next = RESTART_NOTE + "\n" + state.next
    if answer.md of state.turn already exists: log "t%03d: reusing the recorded answer", process it (no Codex call)
        # BUG: processing overwrites state.next, so the restart note is lost (section 10)
loop:
    msg = "".join(NOTICE_PREFIX(n) for n in new notices) + state.next    # notices are prepended at send time
    if state.turn == 1:                 new thread with TASK (message.txt)           -> "turn 1: new thread (N chars)"
    elif state.since_handoff >= 40 (or forced):  new thread with HANDOFF(msg)        -> "turn T: new thread (N chars)"
    else:                               codex exec resume <thread>                    -> "turn T: resume (N chars)"
    write turns/tNNN/prompt.txt (the exact message), run.json, events.jsonl, stderr.log, answer.md
    if Codex failed with content-filter refusal:     # section 2.12
        flag_streak += 1; flags += 1; save refused/tNNN_<flags>/; state.next = FILTER_MSG; retry SAME turn number
    block = first ```python / ```forward block of answer.md (last agent message only)
    if state.explain: append entry to explanations.md (section 7)
    if no block:
        if passes == budget: final_answer.md = answer; done = True; log "final answer: ..."; stop
        else: nudges += 1; state.next = NUDGE_MSG [unobserved]
    elif explain and no text before the block and unexplained_streak < 2:
        state.next = REJECT_MSG; unexplained += 1; unexplained_streak += 1  -> "python block without an explanation; sent back unrun"
    else:
        reminder = explain and no explanation           # third unexplained in a row runs anyway
        unexplained_streak = 0
        python  -> runs += 1; state.next = PY_OUTPUT(kernel.run(code))
        forward -> validate JSON (harness side) -> dedupe key -> requests += 1 -> server -> PASS_DONE / FAILED / TIMEOUT
        if reminder: state.next += "\n" + REMINDER
    turn += 1; since_handoff += 1; save state.json; log "turn T: python|forward; runs R, passes P/B, tool-free X"
```

Several details were checked against the logs [V]:
- **Sent-back replies.** A reply that is sent back unrun still uses up a turn number. For example, p50c t031 was sent
  back, and the next turn was t032.
- **Content-filter refusals.** The turn number does not change; p50c turn 6 was retried as turn 6.
- **Handoff timing.** `since_handoff` at the end of a run equals `turn - handoff_turn + 1`. For p50c that is
  158 - 121 + 1 = 38, and it matches in all 7 runs. Handoffs happened at turns 41/81/121 (p50c), 41 (p10c),
  29 (forced) and 69 = 29 + 40 (p25c), 41 (g25b), and 41/81 (g50b).

---

## 2. Every harness-authored message (the main deliverable)

All examples are `prompt.txt` files of turns after turn 1. Messages carry no trailing newline [V].
The counts cover the 7 released runs and only the stored `turns/t*/prompt.txt` files (turn 1 excluded). They were
produced by a regex census of every prompt.

### 2.1 Python output [Q][V]
```
OUTPUT of Python run {n} of at most 300:
```
{out}
```
```
- Example: `runs/p10c/turns/t002/prompt.txt`.
- Count: **323**. Per run: p10c 59, p25c 45, p50c 79, g05b 13, g10b 20, g25b 44, g50b 63. That is every Python run,
  except p50c run 4, whose output was refused by the content filter.
- `{n}` is the cumulative `runs` counter. It includes runs that raised, and excludes blocks that were sent back.
- Python messages never include a "runs left" or "passes left" line.

`{out}` is computed as follows. The pipeline reproduces 276 of 276 testable outputs byte-for-byte. A testable output
is one whose raw text is in `tool_calls.jsonl`, which keeps only the first 6,000 characters. [V]
1. `raw` is stdout and stderr merged in time order. Warnings appear inline, for example
   `<run>:28: RuntimeWarning: divide by zero encountered in log` in `gemma_followup/runs/g50b/turns/t029/prompt.txt`.
   There is no `[stderr]` marker, unlike the Go sandbox.
2. If `len(raw) > 10000`: `raw = raw[:10000] + "\n[output cut at 10,000 characters]"`. This gives a 10,034-character
   body. Example: `runs/p10c/turns/t006/prompt.txt`. Count: 8, plus 1 inside `runs/p50c/refused/t006_1/prompt.txt`.
3. Repeated-line suppression:
   - Lines are compared whole and unstripped.
   - A line with `len(line) >= 16` is kept for its first 2 occurrences in the whole output. Later occurrences are
     dropped in place.
   - Shorter lines, including blank lines, are never dropped.
   - If `d > 0` lines were dropped, append `"\n[harness: {d} repeated lines not shown (a line is shown at most 2 times per output)]"`.
   - Example: `runs/p10c/turns/t021/prompt.txt`. Count: 14.
   - The cut happens before this step. In `runs/p25c/turns/t032/prompt.txt` the cut marker comes first and the
     harness marker last.
   - The rule `len(line.strip()) >= 12` fits the data equally well.
4. `out = out.encode("ascii", "replace").decode()`. Each non-ASCII character becomes `?`, as in `"\u4e94\u5468"` → `??`.
   Example: run 3 of p50c.
5. Errors:
   - The traceback is appended after the output printed so far.
   - It starts with the driver frame, for example:
     ```
     Traceback (most recent call last):
       File "<string>", line 17, in <module>
       File "<run>", line 8, in <module>
       File "<prelude>", line 39, in edited
     TypeError: unsupported operand type(s) for +: 'PosixPath' and 'str'
     ```
     Source: `gemma_followup/runs/g10b/turns/t002/prompt.txt`.
   - The driver frame tells which stage failed:
     - `<string>` line 13: `ast.parse` raised a SyntaxError.
     - Line 17: an error while executing the body.
     - Line 19: an error while evaluating the last expression.
6. Last-line expression: the value is printed as `repr(value)` on its own line, unless it is `None`. Examples:
   - `json.dump(...)` → nothing.
   - `Path(...).write_text(...)` → `4265`.
   - A string → `'Final candidate retained, ...'` in `gemma_followup/runs/g50b/turns/t117/prompt.txt`.
7. Empty output: never observed. The Go sandbox returns `(no output)` [I].

### 2.2 Forward pass done [Q][V]
```
Forward pass {p} of {B} done ({type}); results in {where}. Summary:
```
{json.dumps(done_json, indent=1)}
```
Forward passes left: {B-p}.
```
- `{type}` is `generate` or `teacher_forced`.
- `{p}` is the counted pass number. `{k}`, used in `{where}`, is the request number. The two differ after failures:
  p50c pass 50 is `gen_69`.
- `{where}` takes one of three forms:
  - `/work/gen_{k}.jsonl`: single-model generation. 7 cases, all `"model": "original"`. Example: `runs/p10c/turns/t004/prompt.txt`.
  - `/work/gen_{k}_original.jsonl and /work/gen_{k}_edited.jsonl`: `"both"`. 124 cases. Example: `runs/p10c/turns/t016/prompt.txt`.
  - `/work/fwd_{k}/`: teacher-forced. 44 cases. Example: `runs/p10c/turns/t011/prompt.txt`.
- Count: **175**, equal to the total passes (10+25+50+5+10+25+50). The summary equals `json.dumps(fwd_k.done.json, indent=1)`
  byte-for-byte in all 175 [V]. Field contents are in section 5.

### 2.3 "Edits used in this pass" line [Q]
Appended on its own line after `Forward passes left: N.`:
```
Edits used in this pass: {", ".join(sorted edited tensor names)} (edit hash {10 hex}).
Edits used in this pass: none (edit hash none).
```
- Examples: `runs/p50c/turns/t157/prompt.txt` (its only p-run occurrence) and `gemma_followup/runs/g05b/turns/t004/prompt.txt` (the `none` form).
- Count: **91**. p50c has 1. Every Gemma pass has one (90), and 10 of those are `none`.
- The line reports the state of `/work/edited` at pass time, even for `"model": "original"` passes (g25b k=8,
  g50b k=11 and k=22) [V].
- The names are sorted as strings (`...layers.10..., ...layers.17..., ...layers.6...`), which matches
  `sorted(listdir)` [V].
- The hash is computed from the content of the edits [V]: g25b and g50b produced the same edits independently and got
  the same hashes (`8b73df8fe2`, `41c604fc5c`, `7184fd5145`). The exact hash function is unknown [I]. It is probably a
  sha over the `.npy` names and bytes in `/work/edited`.

### 2.4 Budget exhausted / final-answer request [Q]
Appended after the pass-done message (and after the "Edits used" line, if any) when `p == B`:
```
All forward passes are used. Make your final choice of edits (Python is still available), then reply without a code block.
```
- Example: `runs/p10c/turns/t073/prompt.txt`. Count: **7**, one per run.
- Later Python outputs do not repeat it. The next reply without a block ends the run.
- No message is sent after the final answer.

### 2.5 Failed pass (not counted) [Q]
```
The forward pass failed and was not counted: {done["error"]}
```
- Nothing follows: there is no "passes left" line.
- Count: **30**, all in the p-runs (p10c 3, p25c 8, p50c 19).
- `{error}` is the server's `"<ExcType>: <message>"`, which can span several lines. The variants seen:
  - `RuntimeError: generation worker failed:  self.llm_engine = LLMEngine.from_engine_args(` followed by about 20
    lines of vLLM traceback ending in `RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}`.
    25 cases. Example: `runs/p10c/turns/t067/prompt.txt`.
  - `IndexError: tuple index out of range`: the `attn_layers` bug, section 5. 4 cases. Example: `runs/p10c/turns/t009/prompt.txt`.
  - `ValueError: unknown practice ids: ['gsm8k_train3513']`: 1 case, delivered inside the handoff
    `runs/p50c/turns/t041/prompt.txt`.
- These requests still use up a request number `k`, a `requests/fwd_k.*` pair, and a `tool_calls.jsonl` forward row.

### 2.6 Server timeout [Q]
```
The forward-pass server did not answer in time; the request was not counted.
```
- Example: `runs/p10c/turns/t065/prompt.txt`, after notice 002. Count: **1**.
- `requests/fwd_10.json.claimed` exists with no `.done.json` and no `tool_calls` row. `requests` was still
  incremented.
- The timeout is about **6 h**: the request was sent after Codex ended at 00:22:21, and the message came at 06:22:24 [V].

### 2.7 Invalid forward request (harness-side validation, nothing counted) [Q]
```
The forward request is not valid ({ExcType}: {message}). Not counted.
```
- Count: **4**, Gemma runs only.
- Seen:
  - `(AssertionError: a teacher-forced pass needs texts or chats)`. Astra had sent `"texts_file"`. 3 cases. Example:
    `gemma_followup/runs/g05b/turns/t010/prompt.txt`.
  - `(JSONDecodeError: Extra data: line 2 column 1 (char 8812))`. 1 case: `gemma_followup/runs/g50b/turns/t050/prompt.txt`.
- No request file is written and `requests` is not incremented. In the Gemma runs `requests == passes` [V].
- Unknown practice IDs are **not** checked by the harness. That check is in the server (section 2.5).

### 2.8 Block sent back for lack of an explanation [Q]
Prompt v3 (p-runs) wording. Example: `runs/p25c/turns/t021/prompt.txt`. Count: **15** (p25c 7, p50c 8).
```
Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to 8 sentences): what the last result showed, what you conclude from it, what you will do now and why, and what outcome would confirm or refute your current hypothesis. In your final answer, also summarize the path that led to your edits. These explanations are read by the experimenters. Send the block again, preceded by your explanation.
```
Prompt v5 (Gemma) wording. Example: `gemma_followup/runs/g10b/turns/t002/prompt.txt`. Count: **5** (g10b 1, g25b 2, g50b 2).
```
Your last reply had no explanation, so its block was NOT run (nothing was used). EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to 8 sentences): what the last result showed, what you conclude from it, what you will do now and why, and what outcome would confirm or refute your current hypothesis. In your final answer, also summarize the path that led to your edits. These explanations are read by the experimenters. Reply again: first your explanation, then the block you want to run now (it may be the same one).
```
- The message is 543 characters (v3) and is the whole prompt, unless a notice or restart note is prepended.
- driver.log records it as `turn T: python block without an explanation; sent back unrun` or
  `turn T: forward block without an explanation; sent back unrun`.
- The v5 message uses the **v3** EXPLAIN wording ("what you conclude from it", "These explanations are read by the
  experimenters"), not the v5 prompt's wording [V]. The text is hard-coded.
- 12 of the 20 send-backs were false positives (bug 1, section 10).

### 2.9 Reminder line [Q]
Appended on its own line at the end of the message, after the closing fence or after `Forward passes left: N.`:
```
(Reminder: before your next block, explain what this result shows, what you will do next and why.)
```
- Example: `runs/p10c/turns/t052/prompt.txt`. Count: **7** (p10c 1, p25c 4, p50c 2).
- It is added in three cases:
  1. Always, on the message that delivers the EXPLAIN note: p10c t052, p25c t016, p50c t029.
  2. In harness version "v1" (23:03:59–23:10:15 on 10-23), after any unexplained block. That version ran such
     blocks anyway: p25c t018, t019; p50c t031.
  3. In the final version, after the third unexplained block in a row, which runs anyway: p25c t030, which followed
     the t027/t028 send-backs and the unexplained t029.

### 2.10 Harness-restart note [Q]
Prepended to the pending message, followed by a single `\n`:
```
[note: the harness was restarted, so your kernel restarted too and its variables are gone]
```
- Example: `runs/p10c/turns/t012/prompt.txt`.
- Count: **6** stored prompts: p10c t012; p25c t020, t025, and inside the t029 handoff's "Last result"; p50c t031,
  t154. One more is in `runs/p25c/turns_discarded/t029_contextless/prompt.txt`.
- 12 more restart notes were built but **never delivered**: whenever a recorded answer was reused, the block's result
  overwrote the note (bug 2). driver.log has 12 `reusing the recorded answer` lines:
  - p10c t051 (3 times) and t058;
  - p25c t012, t015 (twice) and t029;
  - p50c t014, t028 (twice) and t045.
- The restart note is the only kernel-restart message ever seen. The kernel time-limit and memory-limit restarts
  ("you are told so") never happened, so their text is unknown [unobserved].

### 2.11 Notes from the experimenters [Q]
Two prefix forms. Both are followed by `\n\n` and then the normal message:
```
[note from the experimenters: a new instruction for the rest of the session] {EXPLAIN paragraph}

{normal message}
(Reminder: before your next block, explain what this result shows, what you will do next and why.)
```
- The EXPLAIN-note form: example `runs/p50c/turns/t029/prompt.txt`. Count: **3** (p10c t052, p25c t016, p50c t029).
- Its `{EXPLAIN paragraph}` is the same as the one inserted into `message.txt` (section 7).
```
[note from the experimenters] {contents of notices/<file>}

{normal message}
```
- Count: **10**:
  - `001_pass_size.txt`: 3 (p10c t059, p25c t025, p50c t046).
  - `002_tool_facts.txt`: 3 (p10c t065, p25c t031, p50c t051).
  - `004_failure_counts.txt`: 4 (g05b t019, g10b t028, g25b t030, g50b t025).
- Example: `runs/p10c/turns/t059/prompt.txt`.
- Notices are prepended **at send time**, before the restart note. In `runs/p25c/turns/t025/prompt.txt` the order is
  notice, then `\n\n`, then restart note, then `\n`, then the send-back message.
- Delivered filenames are appended to `state.notices`. Notices 001–003 are already in the Gemma prompt, so they were
  pre-listed there.
- The notice mechanism first fired just after the 23:52:25 restart. Notice 002 arrived later without any restart, so
  notices are picked up from a folder while the harness is running [I].

### 2.12 Content-filter refusal replacement [Q]
The whole prompt (612 characters):
```
[The output of your last step could not be delivered: the provider's content filter refused the message. The model's vocabulary contains offensive and adult strings, and listing raw vocabulary tokens can trigger this. Repeat what you need, but print only tokens relevant to your question (digits, math symbols, common math words) or token ids and numbers, not raw lists of top tokens. The filter can also refuse messages that contain long reasoning transcripts (many generated or hand-written 'Wait, ...' passages): print short excerpts (at most about 300 characters each) and statistics instead of whole texts.]
```
- Example: `runs/p50c/turns/t006/prompt.txt`. Count: **1**.
- How the event unfolded:
  1. The refused original was `OUTPUT of Python run 4 ...`, 10,081 characters with an output cut. It is saved at
     `runs/p50c/refused/t006_1/prompt.txt`.
  2. Codex returned rc 1 with the event `{"type":"error","message":"Invalid prompt: your prompt was flagged as potentially violating our usage policy. ..."}`.
  3. The harness logged `t006 attempt 1: codex failure (rc 1); giving up; errors [...] stderr ''`, then
     `turn 6: refused by the content filter (streak 1)`.
  4. It re-sent turn 6 in the same thread with the replacement message: `turn 6: resume (612 chars)`.
- Run 4 still counted toward `runs`. The refused output was never shown to the model.
- The `refused/t{turn}_{n}/` folder holds `prompt.txt`, `fail1_events.jsonl`, `fail1_run.json` (with a `waits` entry)
  and `fail1_stderr.log`.
- `flag_streak` counts refusals in a row and `flags` counts all refusals. The action at a high streak is unknown
  [unobserved].

### 2.13 Handoff (new conversation) [Q][V]
```
{current task text = message.txt}

THIS IS A RESTARTED CONVERSATION (to keep it short): {turn-1} replies happened before. Forward passes used: {passes} of {budget}; Python runs used: {runs} of 300. Your kernel is still running with its variables. Tensors currently edited: {repr(edited()) or "none"}. Files in /work: {repr(sorted(os.listdir("/work"))[:60])}.
YOUR NOTEBOOK (/work/notes.md, last 8,000 characters):
```
{notes_md[-8000:]}
```
Continue from here.

Last result:
{the pending message (state.next), including any restart note / reminder}
```
- Example: `runs/p10c/turns/t041/prompt.txt` (13,619 characters).
- Count: **9**:
  - p10c t041
  - p25c t029, t069
  - p50c t041, t081, t121
  - g25b t041
  - g50b t041, t081
- One defective handoff is in `runs/p25c/turns_discarded/t029_contextless`: a new thread started with only the
  634-character pending message.
- Details and sizes are in section 3.

### 2.14 Message kinds that never fired [unobserved]
- **No-block nudge:** `nudges` = 0 in all runs. Text unknown.
  - Suggested: `Your reply had no ```python or ```forward block. Forward passes left: {B-p}; the session continues until all are used.`
- **Malformed, missing or multiple blocks:** all 561 answers had either exactly one block (340 python, 214 forward)
  or none (7 final answers). Behaviour unknown [I]: probably "first block wins".
- **`save_edit` refusals:** never triggered (section 6). The texts are unknown.
- **Kernel timeout (900 s) or memory limit (96 GB):** never hit. The longest Python run took 90 s.
- **Dedupe refusal** ("a request identical to an earlier one with unchanged edits is refused without cost", prompt
  v5): never hit. The text is unknown.
- **Forward block after the budget is used, and the 300-run limit:** never hit.

### 2.15 Composition order (observed)
`[notices: "[note from the experimenters] X\n\n" ...] + [restart note + "\n"] + body + ["\nEdits used ..."] + ["\nAll forward passes are used ..."] + ["\n(Reminder: ...)"]`.
- For a new thread, the composed message goes after `Last result:\n` inside the handoff.
- The restart note becomes part of `state.next` when the harness starts. Notices are added later, at send time.

### 2.16 Census (stored prompts, all 7 runs)

| Message | Count | Where |
|---|---|---|
| `OUTPUT of Python run n of at most 300:` | 323 | all runs |
| `Forward pass p of B done (...)` + `Forward passes left: N.` | 175 / 175 | all runs (124 both, 7 single, 44 TF) |
| `Edits used in this pass: ...` | 91 | p50c 1, Gemma 90 (10 `none`) |
| `All forward passes are used. ...` | 7 | one per run |
| `The forward pass failed and was not counted: ...` | 30 | p10c 3, p25c 8, p50c 19 (25 vLLM init, 4 IndexError, 1 ValueError) |
| `The forward-pass server did not answer in time; ...` | 1 | p10c |
| `The forward request is not valid (...). Not counted.` | 4 | Gemma (3 AssertionError, 1 JSONDecodeError) |
| Send-back, v3 wording | 15 | p25c 7, p50c 8 |
| Send-back, v5 wording | 5 | g10b 1, g25b 2, g50b 2 |
| `(Reminder: ...)` | 7 | p10c 1, p25c 4, p50c 2 |
| `[note: the harness was restarted ...]` | 6 (+1 discarded) | p10c 1, p25c 3, p50c 2 |
| EXPLAIN note | 3 | p-runs |
| Notice 001 / 002 / 004 | 3 / 3 / 4 | p-runs / p-runs / Gemma |
| Content-filter replacement | 1 | p50c t006 |
| Handoff `THIS IS A RESTARTED CONVERSATION` | 9 (+1 contextless) | p10c 1, p25c 2, p50c 3, g25b 1, g50b 2 |
| `[output cut at 10,000 characters]` (inside output) | 8 (+1 refused) | |
| `[harness: N repeated lines not shown ...]` (inside output) | 14 | |
| Nudge / save_edit refusal / kernel timeout / multi-block / dedupe refusal | 0 | never |

---

## 3. Handoff (conversation restart)

- **Trigger 1 [V].** `since_handoff` reaches 40 before a turn is sent, so a new thread starts at turns 41, 81, 121.
  p25c's second handoff came at turn 69 = 29 + 40, which shows this is a counter and not `turn % 40`.
- **Trigger 2 [V].** A forced new thread at a harness restart: p25c turn 29, 00:31:15 on 10-24. The p25c Codex
  session had been saving nothing since t013 (bug 3), so resuming it was useless.
  - All other restarts resumed the existing thread: 22:11:53, 23:03:00, 23:03:59, 23:10:15 and 23:52:25 on 10-23,
    and 16:44:21 on 10-24 for p50c.
  - So the README's "or after a harness restart" means an operator-forced new thread, not an automatic one [I].
- **Contents [V].**
  - The **current** `message.txt`: p10c t041 used the turn-1 text (10,315 characters), because EXPLAIN came later.
    p25c t029 and all later handoffs used the 10,710-character text with the EXPLAIN paragraph.
  - Then the block in 2.13. The counters there are `turn-1`, `passes`, `budget` and `runs`.
  - `Files in /work` is cut silently at 60 sorted entries: p25c t069, p50c t121 and g50b t081 all list exactly 60.
  - The notebook is `notes.md[-8000:]`, cut mid-line with no marker. Its fence is followed by `\nContinue from here.\n\nLast result:\n`.
  - No edit manifest is included beyond the tensor names, and there is no summary of earlier passes. Everything else
    must come from the agent's own `notes.md`, which is why notice 002(4) asks it to write the exact edit before
    every pass.
- **Sizes.** 12,662–21,673 characters for Qwen, 17,354–18,490 for Gemma. The task text is 10,710 or 7,875 characters,
  the header 0.7–1.7k, the notebook up to 8,000, and the last result up to about 10k.
- **Kernel.** It is **not** restarted by a handoff. The text "Your kernel is still running with its variables" is
  hard-coded, and it is wrong after a restart (bug 5).
- **Codex.** A new thread is `codex exec` without `resume`, and `run.json.thread_in` is `null`. Cumulative token
  usage starts again from about 6–9k [V].

---

## 4. `state.json` fields

Key order shows when each field was added. Initial keys: budget … since_handoff. Later: `flag_streak`, `next`,
`flags`, `explain`, `unexplained_streak`, `unexplained`, `notices`, `req_keys`.

| Field | Meaning (verification) |
|---|---|
| `budget` | forward-pass budget B (5/10/25/50) |
| `mode` | model profile: `qwen35think` or `gemma270`. The server takes its limits from it (`code/profiles.py` `limits(run)`) |
| `prompt` | prompt version: `v3` (p-runs) or `v5` (Gemma). The server also reads it, through `profiles.limits` |
| `turn` | number of the last turn, i.e. model replies including sent-back ones. A content-filter retry does not add one [V] |
| `runs` | Python runs executed; refused outputs count, sent-back blocks do not [V] |
| `passes` | counted (successful) forward passes [V] |
| `requests` | requests sent to the server: passes + server failures + timeouts + orphaned. p10c 10+3+1, p25c 25+8+1 (orphan `fwd_9`), p50c 50+19 [V]. Harness-side invalid requests are not included |
| `thread` | current Codex thread id |
| `done` | final answer received |
| `nudges` | replies with no block before the budget was used (always 0) |
| `handoffs` | number of new threads after turn 1 (1/2/3/0/0/1/2) [V] |
| `since_handoff` | turns in the current thread, counting the current turn; a new thread starts when it is 40 [V] |
| `flag_streak` / `flags` | content-filter refusals in a row / in total (`flags` appears only once it is above 0) |
| `explain` | EXPLAIN mode on |
| `unexplained_streak` | blocks sent back in a row; at 2 the next unexplained block runs anyway [V] |
| `unexplained` | total blocks sent back. p25c 7 and p50c 8 match the driver.log send-backs; p25c t029, which ran unexplained, is not counted [V] |
| `next` | the pending or last-built message for the next turn. A restart prepends the restart note to it. At the end it is the last message sent [V] |
| `notices` | filenames of notices already delivered |
| `req_keys` | `{sha1(json.dumps(request, sort_keys=True) + "\|" + edit_hash).hexdigest(): pass_number}`, verified for p50c pass 50 and all 5 g05b passes [V]. The value is the **pass** number (p50c: 50, while k = 69). It is used to refuse an identical request with unchanged edits at no cost (prompt v5). It appeared in p50c only from the 16:44 restart on |

There is also an `edit_hash` row in `tool_calls.jsonl`: `{"t", "kind": "edit_hash", "pass", "edit_hash", "edited": [names]}`.
It is written right after each counted pass's forward row. It exists for every Gemma pass and for p50c pass 50.

---

## 5. Forward-pass server

**Transport [V].**
- The harness writes `requests/fwd_{k}.json`. This is the parsed request re-serialized with `json.dumps(req, indent=1)`;
  Astra's compact JSON is not kept byte-for-byte.
- The server renames the file to `.json.claimed`, runs the request, and writes `fwd_{k}.done.json`.
- Outputs go to `<run>/work/`, which is `/work` for the agent: `gen_{k}*.jsonl`, `gen_{k}*_acts/`, `fwd_{k}/`.
- The traceback in `runs/p50c/requests/fwd_3.done.json` shows the call chain
  `<repo>/scripts/mathedit/fwd_server2.py: main → s.run(req, <run>/work/fwd_{k}, <run>/work/edited, lim) → fwd_server.py run_v3`.
- So the server reads the edit folder `/work/edited/*.npy` at the start of a pass. It loads the run's limits from
  `state.json` (`profiles.limits`) and serves several runs at once.
- Generation passes start a fresh vLLM engine on a checkpoint built with `code/build_ckpt.build(<edits>, out)`. That
  function's docstring says it is "also used for generation passes": the float32 edits are cast to **bfloat16**,
  which matches the evaluation.
- The agent cannot act during a pass, because the turn blocks until `.done.json` appears. So edits cannot change
  during a pass.

**Limits** (`code/profiles.py` `V3_LIMITS`):
- `qwen35think`: `gen_max` 512 (`"both"`: 256 per model), `tf_max_texts` 1024, `tf_max_tokens` 32768,
  `tf_total_tokens` 524288, `act_max_tokens` 8192.
- `gemma270`: `gen_max` 2048, `tf_max_tokens` 8192.

**Generation `done.json`** [V] (`runs/p50c/requests/fwd_1.done.json`, `runs/p10c/requests/fwd_4.done.json`).
- **Single model:**
  - `{activations?, ok: true, type: "generate", model, file: "/work/gen_k.jsonl", generations, graded, accuracy (raw float), accuracy_by_source {src: round4}, mean_tokens (int), cut_share (round4), failures {wrong_answer, no_answer_finished, cut_at_max_tokens}, s (round1)}`.
  - The three `failures` counts split the **incorrect** samples. Check: correct + wrong + no_answer + cut = generations.
  - `cut_share` counts all cut samples, including correct ones (p10c pass 1: 234 cut, 232 of them incorrect).
- **`"both"`:**
  - `{ok, type, model: "both", files: {original, edited}, activations: null | {original: {...}, edited: {...}}, original: {generations, graded, accuracy, accuracy_by_source, mean_tokens, cut_share, failures}, edited: {...}, paired: {items, accuracy_difference, standard_error, samples_wrong_to_right, samples_right_to_wrong}, s}`.
  - `paired` is computed per **item** (problem): the difference of the mean correct over its n samples. SE is
    `std(ddof=1)/sqrt(items)`, rounded to 4 places. Check: 30/64 → 0.4688 ± 0.0629.
  - The wrong→right and right→wrong counts compare sample i of the original with sample i of the edited model, which
    share a seed.
  - The "gain/se/rescues/losses" names in some outputs are Astra's own analysis, not server fields.
- **Activations:**
  - Folder `/work/gen_{k}[_original|_edited]_acts/`.
  - Files `hidden_last_prompt.npy` and `hidden_mean_answer.npy` `[N, len(layers), d]`, and `mlp_last_prompt.npy` and
    `mlp_mean_answer.npy` `[N, len(mlp_layers), d_ff]`. They are float16.
  - The MLP list drops layer index 24, which does not exist: requesting `[4,8,12,16,20,24]` gave 6 hidden and 5 MLP
    layers.
  - `index.json` holds `{layers, mlp_layers, samples: [{item, sample, correct, tokens}], files}`.
- **Generation file rows:** `{id, source, answer, samples: [{text, pred, correct, tokens, cut}]}`.
- **Duplicates.** Identical requests with unchanged edits gave identical statistics: p50c k=8/9 (passes 6/7) and
  k=10/11 (passes 8/9) differ only in `files` and `s` [V]. k=7 had the same request but different edits, and its
  results differ. Notice 002(2) nonetheless says separate passes do not reproduce exactly, because batching changes
  the random numbers.

**Teacher-forced `done.json`** [V] (`runs/p50c/requests/fwd_19.done.json`).
- Fields: `{ok, lengths: [per text], files: {name: shape}, edited_tensors: <count of edited tensors applied, 0 for original>, mean_token_logprob: [per text], type: "teacher_forced", s}`.
- Files in `/work/fwd_k/`:
  - `input_ids [B,T]`, `token_logprobs [B,T-1]`
  - `topk_logprobs` and `topk_ids [B,T,k]`
  - `hidden_{l} [B,T,d]`, `mlp_act_{l} [B,T,d_ff]`
  - `attn_{l} [B,heads,T,T]` (float16)
  - `meta.json`: `{texts (as rendered), lengths, model, edited_tensors, files, padding: "right; positions >= length are padding"}`
- Timing: TF passes take 0.1–3 s. Generation passes take 352–3,167 s, median about 1,000 s at 32k tokens. A failed
  engine start costs about 30 s.

**Server-side errors** (each returned `{ok: false, error, trace, s}` and counted as a request):
- `IndexError: tuple index out of range` in `fwd_server.py _text_outputs`:
  `res[f"attn_{l}"] = out.attentions[l][0]...`. The model indexed the attentions tuple by absolute layer number.
  Fixed by notice 002(1): "Attention probabilities work now; give full-attention layer indices".
- `ValueError: unknown practice ids: [...]`.
- `RuntimeError: generation worker failed: ...`: vLLM engine initialization failed. 25 times in the p-runs, never
  charged. Prompt v5 says these "are retried automatically", and no Gemma pass failed.

**Chat rendering pitfall [V].**
- For `"chats"` with `enable_thinking`, the Qwen3.5 template re-renders assistant turns.
- `meta.json` texts show `<think>\n\n</think>\n\n<think>\n...`, an empty think block put in front.
- The harness later wrote `/work/eval_prompt_format.txt`, see `runs/p50c/work_notes/eval_prompt_format.txt`. It gives
  PREFIX `'<|im_start|>user\n'` and SUFFIX
  `'\nPlease reason step by step, and put your final answer within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'`,
  and advises using raw `texts`.

---

## 6. Python kernel

**Process [V][I].**
- One persistent Python 3.10 process per run, in `<venvs>/sandbox_full` (warning paths show
  `<venvs>/sandbox_full/lib/python3.10/...`). It runs in the same bubblewrap setup as `../harness/sandbox.py`.
- Tracebacks show a driver script run with `-c`, so it appears as `<string>`.
  - Line 13 is `ast.parse(code)`.
  - Line 17 runs the body, compiled with filename `<run>`.
  - Line 19 evaluates the last expression.
- The prelude is compiled as `<prelude>`. `inspect.getsource` on user functions raises `OSError`.
- Limits: 900 s and 96 GB per run (from the prompt), probably `RLIMIT_AS`. Neither was ever hit.
- A harness restart kills the kernel.
- `tool_calls.jsonl` python rows are `{t, kind: "python", kernel: true, s, code, out}`, where `out` is the raw output
  cut to 6,000 characters for the log.

**Globals defined in advance, in order** [V]. Python run 23 of p50c printed `globals()` in a fresh kernel; its
output is in `runs/p50c/turns/t042/prompt.txt`:
`json, os, np, torch, safe_open, Tokenizer, MODEL ('/model' str), WORK ('/work' str), NAMES, load_original, load, FROZEN, save_edit, drop_edit, edited, tok, Path`.
- `FROZEN = {'model.language_model.norm.weight', 'lm_head.weight', 'model.language_model.embed_tokens.weight'}` [V].
  For Gemma it is `model.norm.weight`, `model.embed_tokens.weight` and `lm_head.weight` (profiles.py).
- `load(name)`: a float32 torch tensor read from `/work/edited/{name}.npy` if that exists, otherwise the original from
  `/model/model.safetensors`.
- `load_original(name)`: float32 copy of the original.
- `save_edit(name, t)`: writes `/work/edited/{name}.npy` as float32. The submission files are named
  `model.language_model.layers.15.mlp.down_proj.weight.npy` (`runs/p50c/edit/manifest.json`).
  - It prints nothing and returns nothing visible [I].
  - It refuses names in `FROZEN` and whole-tensor rescales. The profile flag is `"refuse_rescale": True`.
  - It **never refused anything** in these runs.
  - Edits that were accepted: column scaling ×0.5, ×1.5, ×2 and ×0; head-column zeroing; rank-one removals.
  - A rule consistent with all of that [I]: let `c = <W, W0> / <W0, W0>`. Refuse if `c != 1` and
    `||W - c·W0|| <= 1e-6·||W0||`, i.e. W is (almost) exactly a scalar multiple of W0. Also refuse a shape mismatch, a
    name not in NAMES, and non-finite values. These follow the Go sandbox's `save()` asserts.
- `drop_edit(name)` removes the file.
- `edited()` returns `sorted(f[:-4] for f in os.listdir(WORK + "/edited"))`. That expression lives at `<prelude>`
  line 39 [V]. In g10b run 1 the user set `WORK = Path("/work")`, and `edited()` then failed with
  `TypeError: unsupported operand type(s) for +: 'PosixPath' and 'str'` (bug 7).
- Gradients are off. The prompt says "Gradients are disabled", most likely `torch.set_grad_enabled(False)` in the
  prelude [I].
- **No technical guard** against autograd or against running the model in the kernel was seen. Those rules are
  enforced only by the prompt.
- No `save_edit` echo, no time-limit message and no memory-limit message was ever seen [unobserved].

---

## 7. EXPLAIN EVERY STEP (mid-run addition) and `explanations.md`

- **`message.txt` vs `turns/t001/prompt.txt` [V].** The only difference is one inserted paragraph, placed between the
  "HOW TO WORK." section and the "RULES." section and followed by `\n\n`:
  `EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English (about 3 to 8 sentences): what the last result showed, what you conclude from it, what you will do now and why, and what outcome would confirm or refute your current hypothesis. In your final answer, also summarize the path that led to your edits. These explanations are read by the experimenters.`
  - The harness rewrote `message.txt` at that point, so later handoffs carry the paragraph.
  - It sent the same paragraph once, as the 2.11 note with the reminder, at p10c t052, p25c t016 and p50c t029,
    which matches the README.
- **Three harness versions on 10-23 (EDT), seen in driver.log and the prompts [V].**
  1. **23:03:00 restart:** explain switched on.
     - The note was attached to the rebuilt prompt of the turn whose recorded answer was being reused, so the model
       **never saw it**.
     - `explanations.md` begins with "(no explanation)" entries for p10c t051, p25c t015 and p50c t028.
     - Those prompts grew by 563 characters: the 91-character restart note plus the note.
  2. **23:03:59 restart:** the note moved to the next real message, together with the reminder. An unexplained block
     still **ran**, and the next message got the reminder (p25c t017/t018, p50c t030).
  3. **23:10:15 restart:** an unexplained block is **sent back unrun**, at most 2 in a row. The third one runs, with
     the reminder.
- **Detection [V][I].** The explanation is `answer[:first_block_fence].strip()`, the text before the
  ```` ```python ```` or ```` ```forward ```` fence in `answer.md`.
  - Every flagged reply had 0 characters there. The shortest accepted one had 250 characters.
  - So the rule is "non-empty", or a small minimum that cannot be determined from these logs.
- **`explanations.md` format [V]:**
  - Each processed reply while `explain` is on appends
    `## Turn {T} ({python|forward|final answer}; passes used {p}/{B})\n\n{explanation or "(no explanation)"}\n\n`.
  - For the final answer, the whole answer is logged.
  - `p` is the pass count **before** the block runs.
  - Reused answers after restarts are logged again: p10c Turn 51 appears 3 times, p50c Turn 28 twice and Turn 45 twice.
  - The text-before-block rule matches all 133 parsed p50c entries [V].

---

## 8. Codex invocation, `run.json`, events, driver.log

**Command [V]** (`runs/*/turns/t*/run.json` `cmd`):
```
<codex>/bin/codex exec [resume] --strict-config --skip-git-repo-check --json -m gpt-6-astra \
  -c model_reasoning_effort="max" -c model_context_window=872000 -c model_auto_compact_token_limit=1000000 \
  -c model_instructions_file="<data>/mathedit/runs4/{run}/instructions.md" \
  -o <data>/mathedit/runs4/{run}/turns/t{NNN}/answer.md [{thread_id}] -
```
- `runs4` is the Qwen thinking folder; the Gemma runs use `runs6`.
- The prompt goes in on stdin (`-`). `instructions.md` is the single line `Answer the user's request.`.
- The config is `../harness/codex_config.toml`, with all tools off. That is why each turn's events include the item
  `{"type":"error","message":"Code Mode is unavailable because code-mode host is disabled. ..."}`.
- The auto-compact limit of 1,000,000 is above the 872,000 context window, so compaction never triggers. Threads stay
  short only because of the 40-turn handoff [I].

**`run.json` [V].**
- Fields: `{ok, limit, rc, wall_s, audit, stderr_tail, cmd, attempt, waits, thread_in, end}`.
- `audit` holds `{event_types, item_types, tool_items, tool_free, usage{input_tokens, cached_input_tokens, cache_write_input_tokens, output_tokens, reasoning_output_tokens}, thread_id, errors, turn_completed}`.
- `waits` entries look like `{attempt, kind: "failure", rc, wait_s: 60, at, errors, stderr_tail}`. This means the
  harness waits 60 s and retries a failed Codex call, except for a content-filter refusal, which is "giving up".
- `limit` is always false; it is probably a usage-limit flag [I].
- The 4.7 h gap at p50c t154 ended with a harness restart at 16:44:21. The final `run.json` shows attempt 1 and no
  waits, so the earlier attempts were not kept.
- `usage` is cumulative over the thread, as the README says.
- Codex wall time per turn: median 31–68 s, maximum 119–174 s, depending on the run.

**driver.log = loop.log** (byte-identical in all 7 runs [V]). Each line is `[YYYY-MM-DDTHH:MM:SS-0400] <event>`:
```
turn {T}: new thread ({n:,} chars)          turn {T}: resume ({n:,} chars)
t{T:03d}: reusing the recorded answer
turn {T}: python; runs {R}, passes {P}/{B}, tool-free {True|False}
turn {T}: forward; runs {R}, passes {P}/{B}, tool-free {...}      (logged when the pass ends; P is after counting)
turn {T}: python block without an explanation; sent back unrun   (also "forward block ...")
t{T:03d} attempt {a}: codex failure (rc {rc}); giving up; errors [<event json strings>] stderr '<tail>'
turn {T}: refused by the content filter (streak {s})
final answer: {R} python runs, {P} forward passes, {nudges} nudges, {handoffs} handoffs, {len(edited())} edited tensors
turn {T}: no block; runs {R}, passes {P}/{B}, tool-free {...}
```
"tool-free" is `audit.tool_free`: the event stream contains only agent messages, reasoning summaries and the code-mode
notice. All turns were tool-free.

---

## 9. Gemma follow-up (prompt v5) vs the p-runs (prompt v3)

- **Prompt.**
  - Length: 7,873–7,875 characters (`gemma_followup/runs/g50b/turns/t001/prompt.txt`). The sections are
    TASK / EVALUATION / SCOPE / TOOLS / METHOD / EXPLAIN EVERY STEP / RULES / HOW THE SESSION RUNS.
  - It includes the substance of notices 001–003:
    - a pass costs the same at any size;
    - separate passes do not reproduce, so compare only within `"both"`;
    - `/work/eval_prompt_format.txt`;
    - write the exact edit before each pass;
    - "submit the best edit you found, even if its gain is not statistically significant; never restore the original weights".
  - **DEV/HOLDOUT method**: fix two disjoint sets before any edit; tune on DEV only; freeze before HOLDOUT; pool
    HOLDOUT passes.
  - **Lenient grading** text: `\boxed{}`, else "Final Answer"/"the answer is", else the last number. Practice answers
    in passes are graded the same way (`profiles.py`, `evaluate.lenient_pred`).
  - New tool statements: "server start-up failures are retried automatically: do not spend passes on health checks"
    and "a request identical to an earlier one with unchanged edits is refused without cost".
- **Limits:** up to 2,048 answers per generation (1,024 per model with `"both"`); `max_tokens` at most 8,192; TF texts
  up to 8,192 tokens; layers 0..18; no AIME file. The prompt documents only `texts`, not `chats`.
- **State:**
  - `mode: gemma270`, `prompt: v5`.
  - `explain: true` from turn 1, with the EXPLAIN paragraph in the turn-1 text, so `message.txt` equals `t001`.
  - `notices` pre-lists 001–003. 004 was delivered mid-run, because the server fixed its `no_answer_finished` count.
- **Harness features present from the start:**
  - the `Edits used in this pass` line and `edit_hash` rows;
  - `req_keys` dedupe;
  - automatic retry of vLLM start-up failures (no failed passes appear);
  - the v5 send-back wording.
- **No** reminder lines, restarts, content-filter events or quota problems.
- Handoffs at turns 41 and 81, with the same template.
- **Tensor names:** `model.layers.*` (no `language_model`).
- All 5 Gemma send-backs were false positives (bug 1). They fell on thread-opening turns (t001, t041, t081), where
  Astra tends to send its explanation and its block as two separate messages.

---

## 10. Harness bugs and lessons (avoid these)

1. **Only the last agent message was read.**
   - `codex -o answer.md` keeps only the last `agent_message`. When Astra sent "explanation" and "block" as two
     messages, the harness saw no explanation.
   - **12 of the 20 send-backs were false positives**: p25c t025, t028, t030; p50c t032, t035, t081, t121; g10b t001;
     g25b t001, t041; g50b t041, t081. Each had a 359–629-character explanation in the earlier message
     (`events.jsonl`).
   - It also corrupted `explanations.md`.
   - Fix: join all `agent_message` texts from the event stream.
2. **The restart note was lost when a recorded answer was reused.**
   - On a restart, if `answer.md` existed, the harness replayed the block without calling Codex. That overwrote
     `state.next`, so `[note: the harness was restarted ...]` was never sent. This happened 12 times.
   - The model was not told its kernel variables were gone, and got NameErrors right after. For example, p50c t016
     says `NameError: name 'mlp_direction_edit' is not defined` after the 22:11:53 restart.
   - The same mechanism hid the first EXPLAIN note (23:03:00 version).
   - Fix: keep pending notes in a queue and deliver them with the next message actually sent.
3. **Silent context loss (Codex could not save its sessions).**
   - Codex's stderr shows
     `rollout writer failed for <codex_home>/sessions/... Quota exceeded (os error 122)` for p10c t012–t040,
     p25c t013–t028 and p50c t015–t040.
   - Every `exec resume` reloaded the last saved state. Cumulative `input_tokens` stayed flat at about 236k, 271k and
     338k, where it should grow on every turn [V].
   - Effects: Astra repeated identical passes with unchanged edits (p50c k=8≡9, k=10≡11), kept forgetting the
     EXPLAIN instruction (the send-back loops in p25c and p50c t031–t040) and lost track of variables.
   - Only the new threads at t029 (p25c) and t041 fixed it.
   - Fix: check every turn that usage grows and that stderr has no rollout or quota errors. If either check fails,
     force a handoff.
4. **The forced new thread at p25c t029 went wrong twice.**
   - First it started a new thread and also reused the old thread's recorded answer.
   - Then it opened a new thread containing only the pending 634-character message, with no task text and no handoff
     (`turns_discarded/t029_contextless`).
   - The forward request already running (`fwd_9`, a 970 s generation) was orphaned: it completed, was never
     delivered, and its `k` was used up.
   - Fix: a new thread must always carry task + handoff. On restart, wait for or re-attach to in-flight requests.
5. **The handoff text contradicts itself.** It always says "Your kernel is still running with its variables", even
   when "Last result" starts with the restart note (p25c t029). Fix: build that sentence from the real kernel state.
6. **Server bugs, all charged as uncounted failures:**
   - `attn_layers` indexed `out.attentions` by absolute layer number (IndexError ×4).
   - `chats` re-rendering put an empty `<think></think>` in front and dropped the reasoning.
   - Practice IDs were checked only on the server (a typo cost a round trip).
   - vLLM engine start-up failed 25 times; it was later retried automatically.
   - `no_answer_finished` over-counted under lenient grading (notice 004).
   - `profiles.limits()` once checked `prompt == "v3"`, so v4 runs got the v2 limits ("no 'both' and 128 answers").
     Compare versions with ≥, not ==.
7. **The prelude shares globals with user code.** `edited()` reads the global `WORK`. A user's `WORK = Path("/work")`
   broke it (g10b). Fix: keep harness state in a closure or module that user code cannot overwrite.
8. **Identical requests with identical edits cost passes.** p50c spent 2 passes on exact duplicates (k=9, k=11)
   before `req_keys` was added. Adding the edit hash to each result ("Edits used in this pass") lets the agent check what it measured.
9. **The content filter refuses dumps of raw vocabulary and long "Wait, ..." transcripts.** Keep the 2.12
   replace-and-retry path, the ASCII-only output and the 10k cap. Refused outputs still use up a run.
10. **Timeouts:** the server wait was 6 h with nothing shown in between. A Codex outage stalled p50c for 4.7 h, which
    needed a manual restart.
11. **Evaluation incidents** (`code/evaluate.py`, for the evaluation side):
    - A stale `scores.json` from another checkpoint or protocol: fixed by storing a job fingerprint.
    - An engine stopped from outside returned partial answers: such results are now refused when `finish_reason` is
      not stop or length.
    - A shared GPU pool: `vllm_gpu_mem_cap`.

**Timeline of harness versions (EDT)** [V from driver.log and the prompts]:
- **10-23, 20:44:** p10c, p25c and p50c start together under one harness.
- **About 21:07–21:13:** Codex can no longer save its sessions.
- **22:11:53:** restart 1, in all three runs.
- **22:48 (p10c) and 23:45 (p50c):** handoffs at turn 41.
- **23:03:00, 23:03:59 and 23:10:15:** EXPLAIN versions 0, 1 and 2.
- **23:52:25:** restart that delivered notice 001; notice 002 followed between 00:34 and 06:22 on 10-24, without a
  restart. The server's attention fix and `eval_prompt_format.txt` came with notice 002.
- **10-24, 00:31–00:32:** p25c's forced new thread.
- **10-24, 16:44:21:** p50c restart. From here on: edit hash, `Edits used` line, `req_keys`.
- **10-25, 11:08:** the Gemma runs start on the finished harness with prompt v5.
