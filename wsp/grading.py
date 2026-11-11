"""Math answer extraction and grading, matching the paper's evaluation (exp8_math/code/mathqa.py) so that numbers are
comparable: the answer is the content of the LAST \\boxed{...} (or \\fbox{...}) after the last '</think>', brace
matched, with '\\boxed 5' read up to the next '$' or line end; no closed box = no answer = wrong. Equivalence is
math-verify 0.9.0 (`verify(parse('\\boxed{gold}'), parse('\\boxed{pred}'))`).

Windows: math-verify's own timeouts use signal.alarm (POSIX only) and must be switched off, so each check runs in a
worker thread with a wall-clock limit instead; a check that does not finish counts as wrong.
"""
from __future__ import annotations

import functools
import re
import threading

INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."
CHECK_TIMEOUT_S = 10.0


def build_prompt(problem: str) -> str:
    """The user turn of the evaluation protocol: the problem, a newline, Qwen's math instruction."""
    return problem + "\n" + INSTRUCTION


# ------------------------------------------------------------------------------------------------ extraction
_BOX_FORMS = re.compile(r"\\boxed|\\fbox")


def answer_text(text: str) -> str:
    """The text the extraction reads: everything up to the last '</think>' is cut off."""
    return text.split("</think>")[-1].strip() if "</think>" in text else text


def _box_at(text: str, idx: int) -> str | None:
    key = "\\boxed" if text.startswith("\\boxed", idx) else "\\fbox"
    rest = text[idx + len(key):]
    r = rest.lstrip()
    if r.startswith("{"):
        depth = 0
        for k, ch in enumerate(r):
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return r[1:k].strip() or None
        return None                                   # box not closed (answer cut inside it)
    if key == "\\boxed" and rest.startswith(" "):     # '\boxed 5'
        return rest.strip().split("$")[0].split("\n")[0].strip() or None
    return None


def last_boxed(text: str) -> str | None:
    idx = text.rfind("\\boxed")
    if idx < 0:
        idx = text.rfind("\\fbox")
        if idx < 0:
            return None
    return _box_at(text, idx)


def extract(text: str) -> tuple[str | None, int]:
    """(answer or None, number of boxes in the answer text)."""
    t = answer_text(text)
    return last_boxed(t), len(_BOX_FORMS.findall(t))


# ------------------------------------------------------------------------------------------------ checking
@functools.lru_cache(maxsize=1 << 16)
def _parsed(s: str):
    from math_verify import parse
    return parse("\\boxed{" + s + "}", parsing_timeout=None)


def _verify(gold: str, pred: str) -> bool:
    from math_verify import verify
    g, p = _parsed(gold), _parsed(pred)
    if not g or not p:
        return False
    return bool(verify(g, p, timeout_seconds=None))


def is_correct(pred: str | None, gold: str | None, timeout: float = CHECK_TIMEOUT_S) -> bool:
    """math-verify equivalence of an extracted answer and the gold answer (False for no answer, unparseable gold or
    a check that runs longer than `timeout` seconds)."""
    if pred is None or gold is None:
        return False
    if pred.strip() == gold.strip():
        return True
    box: list = []

    def run():
        try:
            box.append(_verify(gold, pred))
        except Exception:  # noqa: BLE001 - an unparseable answer is a wrong answer
            box.append(False)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(timeout)
    return bool(box and box[0])


def grade(text: str, gold: str, finish: str = "stop") -> dict:
    """One response: pred, number of boxes, correct, and the failure category used in the generation summaries
    (wrong_answer / no_answer_finished / cut_at_max_tokens)."""
    pred, nbox = extract(text)
    ok = is_correct(pred, gold)
    if ok:
        failure = None
    elif finish == "length":
        failure = "cut_at_max_tokens"
    elif pred is None:
        failure = "no_answer_finished"
    else:
        failure = "wrong_answer"
    return {"pred": pred if pred is None or len(pred) <= 200 else pred[:200] + "...", "boxes": nbox,
            "correct": bool(ok), "failure": failure}


# ------------------------------------------------------------------------------------------------ loop signals
LOOP_MAX_PERIOD, LOOP_MIN_REPEATS, LOOP_MIN_SPAN = 200, 3, 40


def loop_signals(text: str) -> dict:
    """Two signals that a response ENDS in a loop (the paper's mathqa.loop_signals):
    - verbatim_loop_end: without its last (possibly cut) word, the text ends with one block of 1..200 words repeated
      in a row at least 3 times and over at least 40 words;
    - repetitive_end: in the last 200 words, positions whose 4-word phrase occurs 3+ times there make up at least
      half of all positions (circling with small variations)."""
    w = text.split()[:-1]
    n = len(w)
    verbatim = False
    for p in range(1, LOOP_MAX_PERIOD + 1):
        reps = max(LOOP_MIN_REPEATS, -(-LOOP_MIN_SPAN // p))
        if reps * p > n:
            break
        blk = w[n - p:]
        if all(w[n - (j + 1) * p:n - j * p] == blk for j in range(1, reps)):
            verbatim = True
            break
    tail = text.split()[-200:]
    grams = [tuple(tail[i:i + 4]) for i in range(max(0, len(tail) - 3))]
    counts: dict = {}
    for g in grams:
        counts[g] = counts.get(g, 0) + 1
    repetitive = bool(grams) and sum(counts[g] >= 3 for g in grams) >= len(grams) / 2
    return {"verbatim_loop_end": verbatim, "repetitive_end": repetitive}
