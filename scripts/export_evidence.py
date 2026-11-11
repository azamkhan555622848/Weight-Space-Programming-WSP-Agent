"""Copy the small, shareable evidence of finished runs into results/evidence/, so the README's claims can be checked from
a public clone (runs/ itself is 5 GB and is not published).

For each run: explanations.md, final_answer.md, state.json, launch.json, submission/ and tool_calls.jsonl (the audit
log of every code block and pass). Also the victim and control networks of the edge task, so its results can be
reproduced with the same weights. The home folder is redacted from the copies, and the script fails if anything personal
is still in them.

    python scripts/export_evidence.py
"""
import getpass
import json
import os
import re
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
OUT = os.path.join(ROOT, "results", "evidence")
FILES = ("explanations.md", "final_answer.md", "state.json", "launch.json", "run_meta.json", "tool_calls.jsonl")
NETWORKS = {"victim": os.path.join(ROOT, "data", "edge", "victim", "weights.npz"),
            "control": os.path.join(ROOT, "data", "edge", "control", "weights.npz")}
HOME_MARK = "<home>"
TEXT_SUFFIXES = (".md", ".json", ".jsonl", ".txt")


def home_patterns() -> list[re.Pattern]:
    """The home folder as it appears in text: raw, or with its backslashes JSON-escaped, or with forward slashes."""
    home = os.path.expanduser("~")
    forms = {home, home.replace("\\", "\\\\"), home.replace("\\", "/")}
    return [re.compile(re.escape(f), re.I) for f in sorted(forms, key=len, reverse=True)]


def redact(text: str, patterns: list[re.Pattern]) -> str:
    for p in patterns:
        text = p.sub(HOME_MARK, text)
    return text


def personal_terms() -> list[str]:
    """What must not appear in a published file: the account name, and any extra terms given in WSP_PRIVATE_TERMS
    (comma-separated, for example your email address)."""
    terms = [getpass.getuser()] + [t for t in os.environ.get("WSP_PRIVATE_TERMS", "").split(",") if t.strip()]
    return [t.strip().lower() for t in terms if len(t.strip()) >= 3]


def scan(folder: str, terms: list[str]) -> list[str]:
    """(relative path: term) for every text file under folder that still holds a personal term."""
    hits = []
    for base, _, names in os.walk(folder):
        for n in names:
            if not n.endswith(TEXT_SUFFIXES):
                continue
            p = os.path.join(base, n)
            text = open(p, encoding="utf-8", errors="replace").read().lower()
            hits += [f"{os.path.relpath(p, folder)}: {t}" for t in terms if t in text]
    return hits


def finished_runs() -> list[str]:
    out = []
    for name in sorted(os.listdir(RUNS)) if os.path.isdir(RUNS) else []:
        p = os.path.join(RUNS, name, "state.json")
        if os.path.exists(p) and json.load(open(p, encoding="utf-8")).get("done"):
            out.append(name)
    return out


def main() -> int:
    patterns, terms = home_patterns(), personal_terms()
    runs = finished_runs()
    if not runs:
        print("no finished runs under runs/; nothing to export")
        return 1
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(OUT)
    for run in runs:
        src, dst = os.path.join(RUNS, run), os.path.join(OUT, run)
        os.makedirs(dst)
        for f in FILES:
            p = os.path.join(src, f)
            if os.path.exists(p):
                text = open(p, encoding="utf-8", errors="replace").read()
                open(os.path.join(dst, f), "w", encoding="utf-8", newline="\n").write(redact(text, patterns))
        sub = os.path.join(src, "submission")
        if os.path.isdir(sub):
            shutil.copytree(sub, os.path.join(dst, "submission"))
        print(f"exported {run}")
    nets = os.path.join(OUT, "networks")
    os.makedirs(nets)
    for name, p in NETWORKS.items():
        if os.path.exists(p):
            shutil.copyfile(p, os.path.join(nets, f"{name}.npz"))
            print(f"exported network {name}")
    hits = scan(OUT, terms)
    if hits:
        print("personal text is still in the export; fix it before publishing:\n  " + "\n  ".join(hits[:20]))
        return 1
    total = sum(os.path.getsize(os.path.join(b, n)) for b, _, ns in os.walk(OUT) for n in ns)
    print(f"done: {len(runs)} runs, {total / 2**20:.1f} MB, no personal text found")
    return 0


if __name__ == "__main__":
    sys.exit(main())
