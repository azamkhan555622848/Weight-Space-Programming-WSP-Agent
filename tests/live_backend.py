"""A short live check of one model backend: three small turns through the real provider, confirming the reply is free of tools,
the conversation carries over and usage is reported. This uses your account (a few hundred tokens at low effort).

    python tests/live_backend.py claude-cli          # or anthropic, openai, codex-cli
"""
import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from wsp.env import load_dotenv  # noqa: E402
from wsp.llm import API_BACKENDS, BackendNotReady, make_backend  # noqa: E402


def main():
    kind = sys.argv[1]
    effort = sys.argv[2] if len(sys.argv) > 2 else "low"
    if kind in API_BACKENDS:
        load_dotenv()
    try:
        b = make_backend(kind, tempfile.mkdtemp(), effort=effort)
    except BackendNotReady as e:
        print(f"NOT READY: {e}")
        return 2
    b.new_conversation()
    turns = [("Remember this code word: heron-417. Reply with exactly the word: stored", lambda t: "stored" in t.lower()),
             ("What is 17 * 23? Reply with only the number.", lambda t: "391" in t),
             ("What was the code word I gave you? Reply with only the code word.", lambda t: "heron-417" in t.lower())]
    good = True
    for i, (prompt, want) in enumerate(turns, 1):
        r = b.send(prompt)
        passed = r.ok and not r.refused and want(r.text) and r.audit.get("tool_free") is True and not r.context_lost
        good &= passed
        print(f"turn {i} {'PASS' if passed else 'FAIL'}: {r.text[:80]!r}  error={r.error}")
        print("  " + json.dumps({"model": r.model, "tool_free": r.audit.get("tool_free"), "usage": r.usage, "stop": r.stop_reason,
                                 "s": r.wall_s, "audit": {k: v for k, v in r.audit.items() if k != "usage_cumulative"}}, default=str)[:900])
        if not r.ok:
            break
    print("ALL PASS" if good else "SOME FAILED")
    return 0 if good else 1


if __name__ == "__main__":
    sys.exit(main())
