#!/usr/bin/env python3
"""
Obstacle variants of the mf_* foundation maps with obstacles at the CORNERS
*and* near the CENTRE of the corridor.

WHY THIS EXISTS
---------------
`make_scenario_pool.py` deliberately keeps obstacles OFF the bends: it filters
waypoints with `curv <= quantile(curv, 0.72)`, i.e. it places only on the
straightest 72% of the track.  So the existing pool never tests what happens
when an obstacle sits in the apex of a turn, which is the hardest case for a
funnel controller -- the corridor is already curving, the funnel is already
fighting to stay aligned, and now the racing line is blocked too.

This script places TWO families in the same map:

  * corner  -- at high-curvature waypoints (top `CORNER_FRAC` quantile of
               |kappa|), i.e. exactly the bends the existing pool excludes;
  * centre  -- near the centreline on the straights, same as the existing pool.

Both stay within +/- LATERAL_JITTER of the centreline, so they block the line
the patch actually wants to drive rather than hiding against a wall.

FEASIBILITY IS PRESERVED.  Every candidate is rejection-tested against the
running image: a >= PASS_MIN (2.5 m) free lateral span must survive at every
waypoint within 3 m of the obstacle.  A map that cannot be driven is not a test
of the policy, so nothing that would wall the corridor off is accepted.

    python presentation_code/make_corner_scenarios.py                  # 3 per base
    python presentation_code/make_corner_scenarios.py --per-base 5 --seed 7
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_scenario_pool import (          # noqa: E402
    BaseMap, _raster_poly, _obstacle_polygon, write_variant,
    PASS_MIN, LATERAL_JITTER, OBST_SIZE, END_MARGIN_FRAC,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = _ROOT / "maps" / "scenario_corners"

BASE_MAPS = ["mf_narrowing", "mf_zigzag", "mf_lshape", "mf_slalom"]

PLACEMENT = "centre"    # "centre": obstacles on the centreline; "wall": hugging a wall
WALL_MARGIN = 0.10      # wall mode: gap left between the obstacle and the wall (m)

CORNER_FRAC = 0.75      # "corner" = curvature at or above this quantile
STRAIGHT_FRAC = 0.60    # "centre" = curvature at or below this quantile
N_CORNER = (2, 4)       # per-map corner obstacles, inclusive
N_CENTRE = (2, 4)       # per-map mid-straight obstacles, inclusive
S_SPACING = 2.5         # min along-track separation between any two obstacles (m)


def sample_corner_and_centre(base, rng, n_corner, n_centre):
    """Place both families, rejection-tested so a PASS_MIN lane always survives."""
    placed = []
    wimg = base.img.copy()                      # running image, obstacles accumulate

    L = base.s[-1]
    ends = (base.s >= END_MARGIN_FRAC * L) & (base.s <= (1.0 - END_MARGIN_FRAC) * L)
    wide = base.base_clear >= PASS_MIN + 0.4    # cheap pre-filter; the real test is below
    q_hi = np.quantile(base.curv, CORNER_FRAC)
    q_lo = np.quantile(base.curv, STRAIGHT_FRAC)
    rooms = {
        "corner": np.where(ends & wide & (base.curv >= q_hi))[0],
        "centre": np.where(ends & wide & (base.curv <= q_lo))[0],
    }

    for family, n_want in (("corner", n_corner), ("centre", n_centre)):
        room = rooms[family]
        if len(room) == 0:
            continue                            # this base has no wide bends -- reported, not faked
        for _ in range(n_want):
            for _try in range(200):
                k = int(rng.choice(room))
                if any(abs(base.s[k] - o["s_m"]) < S_SPACING for o in placed):
                    continue
                kind = str(rng.choice(["rect", "rect", "disc", "tri"]))
                size = (float(rng.uniform(*OBST_SIZE)), float(rng.uniform(*OBST_SIZE)))
                angle = float(rng.uniform(0, 180))
                if PLACEMENT == "wall":
                    # push the obstacle out against one wall: offset = half-width
                    # minus half the obstacle minus a small margin, so it TOUCHES
                    # the wall and leaves the middle of the corridor open.
                    half = float(base.base_clear[k])
                    reach = half - 0.5 * max(size) - WALL_MARGIN
                    if reach <= 0.2:
                        continue                       # corridor too tight to hug here
                    off = float(rng.choice([-1.0, 1.0])) * reach
                else:
                    off = float(rng.uniform(-LATERAL_JITTER, LATERAL_JITTER))
                ox = base.cx[k] + off * base.nrm[k, 0]
                oy = base.cy[k] + off * base.nrm[k, 1]

                trial = wimg.copy()
                _raster_poly(trial, base, _obstacle_polygon(kind, ox, oy, size, angle))
                near = np.where(np.abs(base.s - base.s[k]) <= 3.0)[0]
                if any(base.passage_width(trial, int(w)) < PASS_MIN for w in near):
                    continue

                wimg = trial
                placed.append(dict(kind=kind, family=family, placement=PLACEMENT,
                                   center_m=[round(float(ox), 4), round(float(oy), 4)],
                                   size_m=[round(float(size[0]), 4), round(float(size[1]), 4)],
                                   angle_deg=round(float(angle), 2),
                                   s_m=round(float(base.s[k]), 2),
                                   curv=round(float(base.curv[k]), 4)))
                break
    return placed


def preview(out_root, index, bases, path):
    """One panel per base map showing its first variant, obstacles by family."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from PIL.Image import Transpose

    fig, axes = plt.subplots(1, len(bases), figsize=(5.2 * len(bases), 5.0))
    if len(bases) == 1:
        axes = [axes]
    for ax, bname in zip(axes, bases):
        row = next((r for r in index if r["base_map"] == bname), None)
        if row is None:
            ax.set_axis_off()
            continue
        v = row["name"]
        d = pathlib.Path(out_root) / v
        img = np.array(Image.open(d / f"{v}_map.pgm").transpose(Transpose.FLIP_TOP_BOTTOM))
        base = BaseMap(bname)
        ax.imshow(img, cmap="gray", origin="lower",
                  extent=[base.ox, base.ox + base.W * base.res,
                          base.oy, base.oy + base.H * base.res])
        ax.plot(base.cx, base.cy, "c-", lw=0.8, alpha=0.7, label="centreline")
        for ob in row["obstacles"]:
            c = "red" if ob["family"] == "corner" else "orange"
            ax.plot(*ob["center_m"], "o", color=c, ms=7,
                    mec="k", mew=0.5)
        ax.set_title(f"{bname}\n{row['n_corner']} corner (red) / "
                     f"{row['n_centre']} centre (orange), tightest gap "
                     f"{row['gap_width_m']:.2f} m", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=7)
    fig.suptitle(f"obstacles at bends AND on the straights (placement={PLACEMENT}) "
                 f"(every map keeps a >= {PASS_MIN} m drivable lane)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    print(f"[corners] preview -> {path}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--per-base", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(OUT_DIR))
    p.add_argument("--bases", default="", help="comma-separated subset of base maps")
    p.add_argument("--prefix", default="sc_")
    p.add_argument("--placement", choices=["centre", "wall"], default="centre",
                   help="'centre': obstacles on the centreline (blocks the racing "
                        "line); 'wall': obstacles hug a wall (middle stays open)")
    args = p.parse_args()

    global PLACEMENT
    PLACEMENT = args.placement
    bases = args.bases.split(",") if args.bases else BASE_MAPS
    out_root = pathlib.Path(args.out)
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    rng = np.random.default_rng(args.seed)
    index = []
    for bname in bases:
        base = BaseMap(bname)
        tag = bname.replace("mf_", "")
        for i in range(args.per_base):
            vname = f"{args.prefix}{tag}{i:03d}"
            placed = sample_corner_and_centre(
                base, rng,
                int(rng.integers(*N_CORNER, endpoint=True)),
                int(rng.integers(*N_CENTRE, endpoint=True)))
            info = write_variant(base, vname, placed, out_root)
            info["n_corner"] = sum(1 for o in placed if o["family"] == "corner")
            info["n_centre"] = sum(1 for o in placed if o["family"] == "centre")
            info["obstacles"] = placed
            index.append(info)
        done = [x for x in index if x["base_map"] == bname]
        print(f"[corners] {bname:14s} {len(done)} variants | "
              f"corner {np.mean([x['n_corner'] for x in done]):.1f} avg, "
              f"centre {np.mean([x['n_centre'] for x in done]):.1f} avg | "
              f"tightest gap {min(x['gap_width_m'] for x in done):.2f} m")

    with open(out_root / "variants_index.json", "w") as f:
        json.dump({"base_maps": list(bases), "pass_min_m": PASS_MIN,
                   "corner_frac": CORNER_FRAC, "placement": PLACEMENT,
                   "seed": args.seed,
                   "variants": index}, f, indent=2)
    print(f"\n[corners] {len(index)} maps -> {out_root}/")

    prev = _ROOT / "scenarios" / "preview_scenario_corners.png"
    prev.parent.mkdir(parents=True, exist_ok=True)
    try:
        preview(out_root, index, bases, str(prev))
    except Exception as e:
        print(f"[corners] preview failed ({e}) -- maps are still written")


if __name__ == "__main__":
    main()
