"""Draw a corridor map and, optionally, a GCBF+ run over it.

Used both as a sanity check on the geometry conversion and to produce the
qualitative panels that go next to the clearance bars.
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Circle, Polygon as MplPolygon  # noqa: E402

from .corridor_geom import CorridorGeom, ZoneWindow, wall_rects  # noqa: E402

__all__ = ["plot_corridor", "plot_run"]


def _draw_geometry(ax, geom: CorridorGeom, window: ZoneWindow,
                   show_rects: bool = False) -> None:
    ax.add_patch(MplPolygon(np.asarray(geom.walkable.exterior.coords),
                            closed=True, facecolor="white",
                            edgecolor="#444444", lw=1.4, zorder=1))
    for ring in geom.walkable.interiors:
        ax.add_patch(MplPolygon(np.asarray(ring.coords), closed=True,
                                facecolor="#8d8d8d", edgecolor="#333333",
                                lw=1.0, zorder=2))

    if show_rects:
        for cx, cy, w, h, th in wall_rects(geom, window):
            c, s = np.cos(th), np.sin(th)
            box = np.array([[w / 2, h / 2], [-w / 2, h / 2],
                            [-w / 2, -h / 2], [w / 2, -h / 2]])
            pts = box @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
            ax.add_patch(MplPolygon(pts, closed=True, facecolor="none",
                                    edgecolor="#d2691e", lw=0.7, alpha=0.8,
                                    zorder=3))

    cl = geom.centerline
    ax.plot(cl[:, 0], cl[:, 1], color="#bbbbbb", lw=0.8, ls=":", zorder=3)
    # Dashed near-black gates, so they never read as one of the agent tracks.
    for wp, style, label in [(window.start_wp, (0, (6, 3)), "start"),
                             (window.narrow_wp, "-", "pinch"),
                             (window.exit_wp, (0, (2, 2)), "exit line"),
                             (window.goal_wp, (0, (1, 2)), "goal")]:
        p, nrm = cl[wp], geom.normal(wp)
        half = geom.clearance(wp)
        ax.plot([p[0] - nrm[0] * half, p[0] + nrm[0] * half],
                [p[1] - nrm[1] * half, p[1] + nrm[1] * half],
                color="#1f2328", lw=1.6, ls=style, zorder=6, label=label)


def plot_corridor(geom: CorridorGeom, window: ZoneWindow, path: str,
                  show_rects: bool = True) -> str:
    """Static picture of one map: walkable area, obstacles, scored waypoints."""
    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    _draw_geometry(ax, geom, window, show_rects=show_rects)
    _finish(ax, geom, window, f"{geom.name} · gap {geom.gap_width_m:.2f} m")
    ax.legend(fontsize=7, loc="upper right", framealpha=0.9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_run(geom: CorridorGeom, window: ZoneWindow, trace: np.ndarray,
             path: str, robot_radius: float = 0.29, title: str | None = None,
             stride: int = 25) -> str:
    """Overlay a rollout: ``trace`` is ``(T, n_agents, 3)`` of x, y, yaw."""
    fig, ax = plt.subplots(figsize=(6.0, 7.5))
    _draw_geometry(ax, geom, window)

    colours = ["#2d6cdf", "#18a99e", "#e07b39", "#9467bd",
               "#d62728", "#8c564b", "#17becf", "#bcbd22"]
    for i in range(trace.shape[1]):
        col = colours[i % len(colours)]
        ax.plot(trace[:, i, 0], trace[:, i, 1], color=col, lw=1.2, zorder=7)
        for t in range(0, len(trace), stride):
            ax.add_patch(Circle(trace[t, i, :2], robot_radius, facecolor=col,
                                alpha=0.16, edgecolor="none", zorder=5))
        ax.add_patch(Circle(trace[0, i, :2], robot_radius, facecolor="none",
                            edgecolor=col, lw=1.2, ls="--", zorder=8))
        ax.add_patch(Circle(trace[-1, i, :2], robot_radius, facecolor=col,
                            alpha=0.85, edgecolor="black", lw=0.8, zorder=8))

    _finish(ax, geom, window, title or f"{geom.name} · GCBF+ run")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def _finish(ax, geom: CorridorGeom, window: ZoneWindow, title: str) -> None:
    cl = geom.centerline
    lo = max(0, window.start_wp - 6)
    hi = min(len(cl) - 1, window.goal_wp + 6)
    seg = cl[lo : hi + 1]
    pad = 8.0
    ax.set_xlim(seg[:, 0].min() - pad, seg[:, 0].max() + pad)
    ax.set_ylim(seg[:, 1].min() - pad, seg[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_facecolor("#5a5a5a")
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("x (m)", fontsize=8)
    ax.set_ylabel("y (m)", fontsize=8)
    ax.tick_params(labelsize=7)


def main() -> None:
    import argparse

    from .corridor_geom import load_corridor

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("map_dir")
    ap.add_argument("--trace", default=None, help=".npy written by --trace-dir")
    ap.add_argument("--out", default=None)
    ap.add_argument("--robot-radius", type=float, default=0.29)
    ap.add_argument("--pre-wp", type=int, default=10)
    ap.add_argument("--exit-offset", type=int, default=8)
    ap.add_argument("--post-wp", type=int, default=12)
    args = ap.parse_args()

    geom = load_corridor(args.map_dir)
    window = geom.window(args.pre_wp, args.exit_offset, args.post_wp)
    out = args.out or f"{geom.name}_corridor.png"
    if args.trace:
        trace = np.load(args.trace)
        print(plot_run(geom, window, trace, out, args.robot_radius))
    else:
        print(plot_corridor(geom, window, out))
    print(os.path.abspath(out))


if __name__ == "__main__":
    main()
