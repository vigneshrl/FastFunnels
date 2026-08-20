"""Side-by-side qualitative panels for the corridor baselines.

Draws one panel per rollout over the same map, so the two failure modes are
visible next to each other: ORCA drives its rigid rank into the pinch and
crashes, GCBF+ never touches a wall but leaves a straggler behind.

Traces are the ``.npy`` files written by ``run_gcbf_eval --trace-dir`` and
``baselines/run_orca_layouts.py --trace-dir``; both are ``(T, n_agents, 3)`` of
x, y, yaw in the map frame.
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle  # noqa: E402

from .corridor_geom import load_corridor  # noqa: E402
from .plot_corridor import _draw_geometry, _finish  # noqa: E402

AGENT_COLOURS = ["#2d6cdf", "#18a99e", "#e07b39", "#9467bd",
                 "#d62728", "#8c564b", "#17becf", "#bcbd22"]


def draw_panel(ax, geom, window, trace: np.ndarray, radius: float,
               title: str, stride: int) -> None:
    _draw_geometry(ax, geom, window)
    for i in range(trace.shape[1]):
        col = AGENT_COLOURS[i % len(AGENT_COLOURS)]
        ax.plot(trace[:, i, 0], trace[:, i, 1], color=col, lw=1.3, zorder=7)
        for t in range(0, len(trace), stride):
            ax.add_patch(Circle(trace[t, i, :2], radius, facecolor=col,
                                alpha=0.15, edgecolor="none", zorder=5))
        ax.add_patch(Circle(trace[0, i, :2], radius, facecolor="none",
                            edgecolor=col, lw=1.2, ls="--", zorder=8))
        ax.add_patch(Circle(trace[-1, i, :2], radius, facecolor=col,
                            alpha=0.9, edgecolor="black", lw=0.8, zorder=8))
    _finish(ax, geom, window, title)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_dir")
    ap.add_argument("--panel", action="append", required=True,
                    metavar="TRACE:RADIUS:TITLE",
                    help="repeatable; e.g. orca.npy:0.25:'ORCA (0/4 through)'")
    ap.add_argument("--out", default="fig_corridor_qualitative.png")
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--pre-wp", type=int, default=10)
    ap.add_argument("--exit-offset", type=int, default=8)
    ap.add_argument("--post-wp", type=int, default=12)
    ap.add_argument("--suptitle", default=None)
    args = ap.parse_args()

    geom = load_corridor(args.map_dir)
    window = geom.window(args.pre_wp, args.exit_offset, args.post_wp,
                         min_clearance=0.39)

    panels = []
    for spec in args.panel:
        path, radius, title = spec.split(":", 2)
        panels.append((np.load(path), float(radius), title))

    fig, axes = plt.subplots(1, len(panels),
                             figsize=(4.6 * len(panels), 7.0))
    axes = np.atleast_1d(axes)
    for ax, (trace, radius, title) in zip(axes, panels):
        draw_panel(ax, geom, window, trace, radius, title, args.stride)
    axes[0].legend(fontsize=7, loc="upper right", framealpha=0.9)

    if args.suptitle:
        fig.suptitle(args.suptitle, fontsize=11, y=0.985)
    fig.tight_layout()
    fig.savefig(args.out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
