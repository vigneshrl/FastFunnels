"""Narrow-zone clearance figure with the GCBF+ baseline alongside FastFunnels.

Panel A repeats the bars of ``fig_zone_clearance.png`` and adds GCBF+, so the
baseline lands on the same axis as the funnel policy.  Panel B breaks the
baseline's failures down, because "0%" means two very different things for a
controller that crashes and one that stops dead.

The FastFunnels counts default to the ones ``make_rebuttal_figure.py`` plots;
pass ``--ours-*`` to override if those runs are refreshed.
"""

from __future__ import annotations

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Colour carries method identity and nothing else, in fixed order.
# Validated for CVD separation and contrast against a light surface; the grey is
# a deliberate neutral for the "no result" reference bar, not a data series.
C_PRIOR = "#5b6470"
C_OURS = "#18a99e"
C_GCBF = "#e07b39"
# Outcome states in panel B are a reserved status ramp, always direct-labelled.
C_CLEARED, C_DEADLOCK, C_CRASH = "#1f8a5f", "#c9931f", "#b8402f"
INK, INK_MUTED = "#1f2328", "#5b6470"


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


def load_runs(paths: list[str]) -> dict[str, list[dict]]:
    """Group every result row from the given .jsonl files by split label."""
    by_split: dict[str, list[dict]] = {}
    for path in paths:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                by_split.setdefault(row["split"], []).append(row)
    return by_split


def _bar(ax, x, rate, k, n, colour, width, label=None):
    lo, hi = wilson(k, n)
    ax.bar(x, rate, width=width, color=colour, edgecolor="black", lw=0.6,
           label=label, yerr=[[rate - 100 * lo], [100 * hi - rate]],
           capsize=4, error_kw=dict(lw=1.1, ecolor=INK))
    # Sit the count above the interval's top cap, not inside the whisker.
    ax.text(x, max(rate, 100 * hi) + 3.5, f"{k}/{n}", ha="center", fontsize=9,
            fontweight="bold", color=INK)


def panel_clearance(ax, gcbf: dict[str, list[dict]], args) -> None:
    groups = [
        ("single-layout\n(prior policy)",
         [("prior policy", args.prior_k, args.prior_n, C_PRIOR)]),
        ("randomised\ntrain maps",
         [("FastFunnels (ours)", args.ours_train_k, args.ours_train_n, C_OURS),
          ("GCBF+ (pretrained)", *_count(gcbf, args.train_split), C_GCBF)]),
        ("randomised\nHELD OUT",
         [("FastFunnels (ours)", args.ours_held_k, args.ours_held_n, C_OURS),
          ("GCBF+ (pretrained)", *_count(gcbf, args.held_split), C_GCBF)]),
    ]

    seen: set[str] = set()
    ticks, labels = [], []
    for gi, (glabel, bars) in enumerate(groups):
        width = 0.34 if len(bars) > 1 else 0.44
        for bi, (name, k, n, colour) in enumerate(bars):
            x = gi + (bi - (len(bars) - 1) / 2) * (width + 0.04)
            rate = 100.0 * k / n if n else 0.0
            _bar(ax, x, rate, k, n, colour, width,
                 label=None if name in seen else name)
            seen.add(name)
        ticks.append(gi)
        labels.append(glabel)

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9, color=INK)
    # Headroom so the k/n labels clear the legend rather than sitting under it.
    ax.set_ylim(0, 136)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("narrow-zone clearance (%)", fontsize=10, color=INK)
    ax.set_title("(A) generalisation across layouts\n"
                 "(bars: 95% Wilson interval)", fontsize=10, color=INK)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=INK_MUTED)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95, borderpad=0.5)


def _count(gcbf: dict[str, list[dict]], split: str) -> tuple[int, int]:
    rows = gcbf.get(split, [])
    return sum(r["cleared"] for r in rows), len(rows)


def panel_outcomes(ax, gcbf: dict[str, list[dict]], nominal: dict, args) -> None:
    """Stacked outcome breakdown: cleared / deadlocked / collided."""
    series: list[tuple[str, list[dict]]] = []
    for split, nice in [(args.train_split, "GCBF+\ntrain"),
                        (args.held_split, "GCBF+\nHELD OUT")]:
        if gcbf.get(split):
            series.append((nice, gcbf[split]))
    for split, nice in [(args.train_split, "nominal PID\ntrain")]:
        if nominal.get(split):
            series.append((nice, nominal[split]))

    labels, stacks = [], []
    for nice, rows in series:
        n = len(rows)
        cleared = sum(r["cleared"] for r in rows)
        crashed = sum(r["collided"] for r in rows)
        stuck = n - cleared - crashed
        labels.append(f"{nice}\n(n={n})")
        stacks.append((100 * cleared / n, 100 * stuck / n, 100 * crashed / n))

    stacks_arr = np.asarray(stacks)
    bottom = np.zeros(len(stacks))
    for col, colour, name in [(0, C_CLEARED, "cleared"),
                              (1, C_DEADLOCK, "stalled (no collision)"),
                              (2, C_CRASH, "collided")]:
        vals = stacks_arr[:, col]
        ax.bar(range(len(stacks)), vals, bottom=bottom, color=colour,
               edgecolor="white", lw=1.2, label=name, width=0.6)
        for i, v in enumerate(vals):
            if v >= 6:
                ax.text(i, bottom[i] + v / 2, f"{v:.0f}%", ha="center",
                        va="center", fontsize=8, color="white",
                        fontweight="bold")
        bottom += vals

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=8.5, color=INK)
    ax.set_ylim(0, 128)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("share of maps (%)", fontsize=10, color=INK)
    ax.set_title("(B) how the runs end\n"
                 "(GCBF+ fails by stalling, not by crashing)",
                 fontsize=10, color=INK)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=9, colors=INK_MUTED)
    ax.legend(fontsize=8, loc="upper center", ncol=3, framealpha=0.95,
              columnspacing=1.0, handlelength=1.2)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gcbf", nargs="+", required=True,
                    help="one or more .jsonl files from run_gcbf_eval")
    ap.add_argument("--nominal", nargs="*", default=[],
                    help=".jsonl files from a --policy u_ref run")
    ap.add_argument("--train-split", default="train")
    ap.add_argument("--held-split", default="heldout")
    ap.add_argument("--prior-k", type=int, default=0)
    ap.add_argument("--prior-n", type=int, default=15)
    ap.add_argument("--ours-train-k", type=int, default=49)
    ap.add_argument("--ours-train-n", type=int, default=50)
    ap.add_argument("--ours-held-k", type=int, default=48)
    ap.add_argument("--ours-held-n", type=int, default=50)
    ap.add_argument("--out", default="fig_zone_clearance_gcbf.png")
    args = ap.parse_args()

    gcbf = load_runs(args.gcbf)
    nominal = load_runs(args.nominal) if args.nominal else {}

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.4))
    panel_clearance(axes[0], gcbf, args)
    panel_outcomes(axes[1], gcbf, nominal, args)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")

    for split, rows in gcbf.items():
        k, n = _count(gcbf, split)
        lo, hi = wilson(k, n)
        print(f"  GCBF+ {split}: {k}/{n} = {100 * k / n:.1f}% "
              f"[{100 * lo:.1f}, {100 * hi:.1f}]  "
              f"collisions={sum(r['collided'] for r in rows)}")


if __name__ == "__main__":
    main()
