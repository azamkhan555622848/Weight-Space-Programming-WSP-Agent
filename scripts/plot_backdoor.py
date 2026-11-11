"""Draw the backdoor-repair result as one image, from the measured numbers in results/edge_results.json.

    python scripts/plot_backdoor.py                      # results/figures/backdoor_repair.png and backdoor_repair_dark.png
    python scripts/plot_backdoor.py --repo github.com/you/wsp-agent     # the same, with the address on a last line

Ranked bars of the attack success rate: the agent's row in colour, everything else in grey. Every number in the image is read
from the results file; nothing is typed in by hand.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H, DPI = 1600, 900, 200          # drawn at twice the size of an 800 x 450 chart, so it stays sharp on a phone
CREDIT = 48                         # extra height for the line that holds the address, when one is given

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", baseline="#c3c2b7", rest="#c3c2b7",
                  accent="#2a78d6"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", baseline="#383835", rest="#52514e",
                 accent="#3987e5"),
}
FONTS = ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"]


def rows_from(results: dict) -> tuple[list[dict], dict]:
    agent = [v for k, v in results.items() if k.startswith("WSP agent: ")]
    n = len(agent)
    edited = sorted(r["changed_weights"] for r in agent)
    mean = {"asr": sum(r["asr"] for r in agent) / n, "clean_acc": sum(r["clean_acc"] for r in agent) / n}

    def row(name, note, r, accent=False):
        note = f"{note} · " if note else ""
        return {"name": name, "note": f"{note}clean accuracy {100 * r['clean_acc']:.2f}%", "asr": 100 * r["asr"], "accent": accent}
    rows = [row("Backdoored network", "no defence", results["unedited victim"]),
            row("Fine-tuning", "", results["baseline: ft"]),
            row("Fine-pruning", "", results["baseline: fp"]),
            row("Pruning only", "", results["baseline: prune"]),
            row("WSP agent", f"mean of {n} runs", mean, accent=True),
            row("Clean network", "never backdoored", results["clean control (no backdoor)"])]
    rows.sort(key=lambda r: -r["asr"])
    facts = {"n": n, "edited": edited, "n_test": results["unedited victim"]["n_test"]}
    return rows, facts


def bar(ax, x0, x1, y, h, colour):
    """A bar from the baseline x0 to x1: square at the baseline, rounded at the data end."""
    r = min(8.0, x1 - x0, h / 2)
    k = 0.5523 * r
    top, bot = y - h / 2, y + h / 2
    verts = [(x0, top), (x1 - r, top), (x1 - r + k, top), (x1, top + r - k), (x1, top + r), (x1, bot - r), (x1, bot - r + k),
             (x1 - r + k, bot), (x1 - r, bot), (x0, bot), (x0, top)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4,
             Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=colour, edgecolor="none"))


def draw(rows, facts, theme: str, out: str, repo: str | None) -> None:
    c = THEMES[theme]
    plt.rcParams["font.family"] = FONTS
    height = H + (CREDIT if repo else 0)
    fig = plt.figure(figsize=(W / DPI, height / DPI), dpi=DPI, facecolor=c["surface"])
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(height, 0)            # one unit is one pixel, y grows downwards
    ax.axis("off")
    pt = 72 / DPI                     # pixels to points

    def text(x, y, s, px, colour, weight="normal", ha="left", va="center"):
        return ax.text(x, y, s, fontsize=px * pt, color=colour, fontweight=weight, ha=ha, va=va)

    left = 64
    text(left, 84, "Fine-tuning kept the backdoor.", 54, c["ink"], "bold")
    text(left, 150, "An LLM editing the weights directly removed it.", 54, c["ink"], "bold")
    text(left, 214, f"Attack success rate after each repair, on {facts['n_test']:,} CIFAR-10 test images. Lower is better.", 28, c["ink2"])

    x0, full = 548, 800               # the baseline, and the length of 100%
    top, pitch, thick = 268, 91, 40
    ax.plot([x0, x0], [top + 6, top + pitch * len(rows) - 6], color=c["baseline"], linewidth=1 * pt * 2, solid_capstyle="butt")
    renderer = fig.canvas.get_renderer()
    for i, r in enumerate(rows):
        y = top + pitch * i + pitch / 2
        strong = r["accent"]
        text(left, y - 15, r["name"], 30, c["ink"], "bold" if strong else "semibold")
        text(left, y + 19, r["note"], 21, c["ink2"])
        x1 = x0 + full * r["asr"] / 100
        bar(ax, x0, x1, y, thick, c["accent"] if strong else c["rest"])
        value = text(x1 + 14, y - 1, f"{r['asr']:.2f}%", 30, c["ink"], "bold" if strong else "semibold")
        if strong:
            lo, hi = facts["edited"][0], facts["edited"][-1]
            end = value.get_window_extent(renderer).x1
            text(end + 22, y - 1, f"no training · 10 forward passes per run · {lo:,} to {hi:,} of 77,706 weights edited", 23, c["ink2"])

    foot = top + pitch * len(rows) + 30
    text(left, foot, "ResNet-8, 77,706 weights, with a BadNets-style patch trigger (5% of its training images poisoned). Every method got the "
         "same 5,000 clean images.", 20, c["muted"])
    text(left, foot + 30, "The agent (Claude Opus 5.5 as the editor) was told a backdoor was planted, but not the trigger, its place or the "
         "target class.", 20, c["muted"])
    if repo:                          # a long address fits nowhere beside the title, so it gets a line of its own
        lead = text(left, foot + 70, "Code, networks and the full log of every run:", 24, c["ink2"])
        text(lead.get_window_extent(renderer).x1 + 10, foot + 70, repo, 24, c["ink"], "semibold")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=DPI, facecolor=c["surface"])
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--results", default=os.path.join(ROOT, "results", "edge_results.json"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "results", "figures"))
    ap.add_argument("--repo", help="an address to print on a last line, e.g. github.com/you/wsp-agent")
    a = ap.parse_args()
    rows, facts = rows_from(json.load(open(a.results, encoding="utf-8")))
    for theme, name in (("light", "backdoor_repair.png"), ("dark", "backdoor_repair_dark.png")):
        out = os.path.join(a.out_dir, name)
        draw(rows, facts, theme, out, a.repo)
        print("wrote", out)
    for r in rows:
        print(f"  {r['name']:<20} {r['asr']:6.2f}%   {r['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
