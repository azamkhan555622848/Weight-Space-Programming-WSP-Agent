"""Every message the harness writes to the editor, in one place.

[paper] = verbatim from the released Exp 8 logs (research/harness_exp8.md, section 2, with the count of occurrences
there); [ours] = paths the paper's logs never exercised (their text is unknowable), worded in the same style.
"""

# [paper 2.1, x323] Python output; {out} is post-processed by `format_output`
PY_OUTPUT = "OUTPUT of Python run {k} of at most {n}:\n```\n{out}\n```"
# [paper 2.2, x175] a counted pass; {summary} = json.dumps(done, indent=1)
FWD_DONE = ("Forward pass {k} of {budget} done ({kind}); results in {where}. Summary:\n```\n{summary}\n```\n"
            "Forward passes left: {left}.")
# [paper 2.3, x91] the state of the edit folder when the pass ran
EDITS_USED = "Edits used in this pass: {names} (edit hash {h})."
# [paper 2.4, x7] appended when the last pass is used
ALL_PASSES_USED = ("All forward passes are used. Make your final choice of edits (Python is still available), then "
                   "reply without a code block.")
# [paper 2.5, x30] the pass raised while running (not counted)
FWD_FAILED = "The forward pass failed and was not counted: {error}"
# [paper 2.7, x4] the request failed validation (not counted)
FWD_INVALID = "The forward request is not valid ({error}). Not counted."
# [ours] (paper prompt v5: "a request identical to an earlier one with unchanged edits is refused without cost")
FWD_DUPLICATE = ("The forward request is identical to forward pass {k} and your edits have not changed since, so it was "
                 "not run and not counted (separate passes on the same weights add little). Forward passes left: {left}.")
# [ours] forward block after the budget is used
FWD_NONE_LEFT = ("All {budget} forward passes are used; no more can be run. Make your final choice of edits (Python is "
                 "still available), then reply without a code block.")
# [ours] python block after the run cap
PY_NONE_LEFT = ("All {n} Python runs are used; no more code can be run. Give your final answer without a code block "
                "once all forward passes are used.")

# [paper 2.8, x15] block sent back because the reply had no explanation before it
EXPLAIN_SENDBACK = ("Your reply had no explanation, so its block was NOT run (no Python run or forward pass was used). "
                    "EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English "
                    "(about 3 to 8 sentences): what the last result showed, what you conclude from it, what you will do "
                    "now and why, and what outcome would confirm or refute your current hypothesis. In your final "
                    "answer, also summarize the path that led to your edits. These explanations are read by the "
                    "experimenters. Send the block again, preceded by your explanation.")
# [paper 2.9, x7] appended when an unexplained block runs anyway (the third in a row)
REMINDER = "(Reminder: before your next block, explain what this result shows, what you will do next and why.)"
# [paper section 7] the paragraph inserted between HOW TO WORK and RULES
EXPLAIN_PARAGRAPH = ("EXPLAIN EVERY STEP. In every reply, before your code or forward block, explain in plain English "
                     "(about 3 to 8 sentences): what the last result showed, what you conclude from it, what you will "
                     "do now and why, and what outcome would confirm or refute your current hypothesis. In your final "
                     "answer, also summarize the path that led to your edits. These explanations are read by the "
                     "experimenters.")

# [paper 2.10, x6] prepended (with "\n") to the next message actually sent after a harness restart
RESTART_NOTE = "[note: the harness was restarted, so your kernel restarted too and its variables are gone]"
# [ours] the kernel died or hit its time limit (paper prompt: "you are told so")
KERNEL_RESTARTED = "[note: the kernel was restarted (time or memory limit), so its variables are gone]"
# [paper 2.11, x10] a file dropped into <run>/notices/ while the run is going, delivered once at send time
NOTICE = "[note from the experimenters] {text}\n\n"

# [paper 2.12, x1] the whole message after the provider refused the previous one
REFUSED = ("[The output of your last step could not be delivered: the provider's content filter refused the message. "
           "The model's vocabulary contains offensive and adult strings, and listing raw vocabulary tokens can trigger "
           "this. Repeat what you need, but print only tokens relevant to your question (digits, math symbols, common "
           "math words) or token ids and numbers, not raw lists of top tokens. The filter can also refuse messages "
           "that contain long reasoning transcripts (many generated or hand-written 'Wait, ...' passages): print short "
           "excerpts (at most about 300 characters each) and statistics instead of whole texts.]")

# [ours] (never fired in the paper)
NO_BLOCK_CONTINUE = ("Your reply had no ```python or ```forward block. Forward passes left: {left} of {budget}; the "
                     "session continues until all are used (replying without a code block does not end it earlier).")
MULTI_BLOCK = ("Your reply had {n} code blocks, so none was run (nothing was used). Reply with exactly one ```python or "
               "```forward block.")
UNKNOWN_BLOCK = ("Your reply had a ```{kind} block, which the harness does not run (nothing was used). Reply with "
                 "exactly one ```python or ```forward block.")

# [paper 2.13, x9] new conversation every `handoff_every` turns (or after a harness restart)
HANDOFF = ("{task}\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {replies} replies happened before. Forward "
           "passes used: {passes} of {budget}; Python runs used: {runs} of {max_runs}. {kernel} Tensors currently "
           "edited: {edited}. Files in your work folder: {files}.\nYOUR NOTEBOOK (notes.md, last 8,000 characters):\n"
           "```\n{notes}\n```\nContinue from here.\n\nLast result:\n{last}")
KERNEL_ALIVE = "Your kernel is still running with its variables."
KERNEL_FRESH = "Your kernel was restarted: its variables are gone (files in your work folder are kept)."


def format_output(raw: str, max_out: int) -> str:
    """The paper's output pipeline (research/harness_exp8.md 2.1, reproduces 276/276 outputs): cut at max_out
    characters; then any line of 16+ characters is shown at most twice; then non-ASCII characters become '?'."""
    out = raw if len(raw) <= max_out else raw[:max_out] + f"\n[output cut at {max_out:,} characters]"
    seen: dict = {}
    kept, dropped = [], 0
    for line in out.split("\n"):
        if len(line) >= 16:
            seen[line] = seen.get(line, 0) + 1
            if seen[line] > 2:
                dropped += 1
                continue
        kept.append(line)
    out = "\n".join(kept)
    if dropped:
        out += f"\n[harness: {dropped} repeated lines not shown (a line is shown at most 2 times per output)]"
    return out.encode("ascii", "replace").decode()


# ------------------------------------------------------------------------------------------------ interactive sessions
# [ours] The paper's sessions were one fixed task run to the end. These messages are for sessions where a user talks to the
# editor while it works (wsp/harness.py, "interactive").
USER_WHILE_WORKING = "[message from the user, sent while you were working]\n{text}\n\n"
INTERRUPTED_BEFORE_RUN = ("[The user interrupted you before your {kind} block was run, so it was NOT run (nothing was used). "
                          "Their next message follows.]")
FWD_INTERRUPTED = "The user interrupted the forward pass, so it was stopped and not counted."
DECLINED = "The user declined to run your {kind} block, so it was NOT run (nothing was used).{feedback}"
PLAN_NO_FORWARD = ("Plan mode is on: forward passes are not run and edits are not saved (nothing was used). Study the weights "
                   "with Python, then tell the user your plan without a code block; they switch to build mode to let you carry it out.")
MODE_NOTE = {"plan": "[note: the user switched to plan mode: study the weights with Python and propose a plan; forward passes are "
                     "not run and edits are not saved until they switch back]",
             "build": "[note: the user switched to build mode: you may edit the weights and run forward passes]"}
BUDGET_NOTE = "[note: the user set the forward-pass budget to {budget}; {left} left]"
EDITOR_NOTE = "[note: the user switched the editor to {model} at {effort} effort, so this conversation was restarted]"
FWD_NONE_LEFT_INTERACTIVE = ("All {budget} forward passes of this session are used, so this one was not run. Finish what you can "
                             "with Python, then tell the user what you found and what more passes would buy; they can raise the budget.")
PY_NONE_LEFT_INTERACTIVE = ("All {n} Python runs of this session are used; no more code can be run. Tell the user where things "
                            "stand, without a code block.")
ONLY_TODO = ("Your reply had a ```todo block and a second block that is not ```python or ```forward, so nothing was run. Reply "
             "with at most one ```todo block and exactly one ```python or ```forward block, or with no block to talk to the user.")
HANDOFF_INTERACTIVE = ("{task}\n\nTHIS IS A RESTARTED CONVERSATION (to keep it short): {replies} replies happened before. Forward "
                       "passes used: {passes} of {budget}; Python runs used: {runs} of {max_runs}. {kernel} Tensors currently "
                       "edited: {edited}. Files in your work folder: {files}.\nYOUR NOTEBOOK (notes.md, last 8,000 characters):\n"
                       "```\n{notes}\n```\nYOUR TODO LIST:\n{todo}\nLATER MESSAGES FROM THE USER (oldest first; their first "
                       "request is in the task above):\n{conversation}\nYOUR LAST MESSAGE TO THE USER:\n{last_message}\n"
                       "Continue from here.\n\nLast result:\n{last}")
