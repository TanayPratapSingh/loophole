"""Draw the two portfolio charts from the sql_shape.json files in runs/ (written by `loophole sql-shape`).

    .venv/bin/python scripts/portfolio_charts.py <output dir>

Every bar and point is read from a run directory; nothing is typed in.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BG, BLUE, BEIGE, GRAY, INK = "#FCFBF9", "#2F78D4", "#DCD8D0", "#6B6B70", "#1C1C1E"
M = "Qwen2.5-1.5B-Instruct"
RUNS = Path(__file__).resolve().parent.parent / "runs"


def shape(pattern: str) -> dict:
    return json.loads((sorted(RUNS.glob(pattern))[-1] / "sql_shape.json").read_text())


def style(ax) -> None:
    ax.set_facecolor(BG)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BEIGE)
    ax.tick_params(colors=GRAY, labelsize=13, length=0)


def bars(out: Path) -> None:
    rows = [("Before training", shape(f"*-eval-sqlquest-{M}")),
            ("GRPO on the leaky grader, seed 0", shape(f"*-eval-sqlquest-after-grpo-sqlquest-leaky-{M}")),
            ("GRPO on the leaky grader, seed 1", shape(f"*-eval-sqlquest-after-grpo-sqlquest-leaky-{M}-s1")),
            ("GRPO on the leaky grader, seed 2", shape(f"*-eval-sqlquest-after-grpo-sqlquest-leaky-{M}-s2")),
            ("GRPO on the honest grader, seed 0", shape(f"*-eval-sqlquest-after-grpo-sqlquest-honest-{M}")),
            ("GRPO on the honest grader, seed 1", shape(f"*-eval-sqlquest-after-grpo-sqlquest-honest-{M}-s1")),
            ("GRPO on the honest grader, seed 2", shape(f"*-eval-sqlquest-after-grpo-sqlquest-honest-{M}-s2"))]
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=100)
    fig.patch.set_facecolor(BG)
    style(ax)
    for i, (_label, s) in enumerate(reversed(rows)):
        share = s["order_losing"] / s["answers"]
        ax.barh(i, 1, color=BEIGE, height=0.58)
        ax.barh(i, share, color=BLUE, height=0.58)
        ax.text(1.015, i, f"{s['order_losing']} of {s['answers']} answers", va="center", fontsize=13, color=GRAY)
        if share > 0.15:
            ax.text(share - 0.012, i, f"{100 * share:.0f}%", va="center", ha="right", fontsize=15, color="white",
                    fontweight="bold")
    ax.set_yticks(range(len(rows)), [r[0] for r in reversed(rows)], fontsize=15, color=INK)
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_title("How often a SQL policy sorts inside a subquery, which picks the right top 3 and loses their order\n"
                 "(held out test split: 13 top 3 questions by 8 samples; the leaky grader never checks order)",
                 loc="left", fontsize=16, color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


def curves(out: Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 6.5), dpi=100)
    fig.patch.set_facecolor(BG)
    style(ax)
    ax.grid(axis="y", color=BEIGE, linewidth=0.8)
    for arm, color in (("leaky", BLUE), ("honest", GRAY)):
        for seed, tag, dash in ((0, "", "-"), (1, "-s1", "--"), (2, "-s2", ":")):
            w = shape(f"*Z-grpo-sqlquest-{arm}-{M}{tag}")["windows"]
            x = [(r["steps"][0] + r["steps"][1]) / 2 for r in w]
            y = [r["order_losing"] / r["limit3_queries"] if r["limit3_queries"] else 0 for r in w]
            ax.plot(x, y, dash, color=color, linewidth=2.6, marker="o", markersize=6,
                    label=f"{arm} grader, seed {seed}")
    ax.set_ylim(-0.03, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("training step (each point covers 20 steps)", fontsize=14, color=GRAY)
    ax.legend(frameon=False, fontsize=13, ncol=2, loc="center right", labelcolor=INK)
    ax.set_title("Share of training queries with LIMIT 3 that sort inside a subquery\n"
                 "(every leaky run picked it up; two kept it, one let it go; no honest run wrote it more than once after step 20)",
                 loc="left", fontsize=16, color=INK, pad=14)
    fig.tight_layout()
    fig.savefig(out, facecolor=BG)
    plt.close(fig)


if __name__ == "__main__":
    dest = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    dest.mkdir(parents=True, exist_ok=True)
    bars(dest / "loophole_01.png")
    curves(dest / "loophole_02.png")
    print(f"wrote {dest / 'loophole_01.png'} and {dest / 'loophole_02.png'}")
