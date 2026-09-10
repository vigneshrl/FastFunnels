#!/usr/bin/env python3
"""
Large obstacles at the NECK THROATS, placed to force a SWERVE rather than a shrink.

Why the throats and not inside the necks: on on_obs_ext the three necks are
~55 m tunnels only 3.19 m wide. The funnel at its b=1.0 floor is already 2.0 m
across, so anything "large" inside a neck leaves no drivable lane at all -- the
map would be impossible, not hard. The wide approach and exit sections either
side are 9.31 m, which is where a big block actually fits AND leaves a lane.

Why offset rather than centred: a centred 3.8 m block in a 9.25 m corridor
leaves ~2.7 m on both sides, and the patch can thread either one by shrinking
alone. Pushing the block off-centre leaves a WIDE side and a NARROW side (~3.9 m
vs ~1.6 m), so the only way through is to move laterally onto the wide side --
a genuine swerve. That is the behaviour this map is built to force.

Obstacles are triangles matched to the one at s=41.2 in
baselines/maps/extended_open_narrow_clutter10 (3.39 x 3.80 m), sited a few
metres before each neck entry and after each neck exit. No wall-hugging
obstacles at all.

    python presentation_code/make_neck_swerve_map.py --name neck_swerve
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import make_scenario_pool as msp                                  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[1]

# matched to the large triangle in extended_open_narrow_clutter10
TRI_SIZE = (3.39, 3.80)
SIZE = TRI_SIZE          # overwritten from --size in main()


def neck_edges(base, narrow_below=5.0):
    """(entry, exit) s-coordinates of every neck."""
    w = np.array([base.passage_width(base.img, k) for k in range(len(base.s))])
    idx = np.where(w < narrow_below)[0]
    if len(idx) == 0:
        return [], w
    grp = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    return [(float(base.s[g[0]]), float(base.s[g[-1]])) for g in grp], w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="on_obs_ext")
    ap.add_argument("--name", default="neck_swerve")
    ap.add_argument("--standoff", type=float, default=4.0,
                    help="metres before a neck entry / after a neck exit")
    ap.add_argument("--offset-frac", type=float, default=0.32,
                    help="obstacle centre offset as a fraction of the half-width; "
                         "bigger = more lopsided = harder swerve")
    ap.add_argument("--pass-min", type=float, default=2.6,
                    help="drivable lane that must survive (funnel at the b=1.0 "
                         "floor is 2.0 m wide)")
    ap.add_argument("--attach", choices=["wall", "float"], default="wall",
                    help="'wall': the block grows out of the wall and protrudes "
                         "into the corridor (like clutter_ext_w10's intrusions); "
                         "'float': a free-standing block offset off-centre")
    ap.add_argument("--intrude", type=float, default=3.2,
                    help="wall mode: how far the block reaches into the corridor (m)")
    ap.add_argument("--kind", default="tri", choices=["tri", "rect", "disc"])
    ap.add_argument("--kinds", default="",
                    help="comma-separated shapes cycled across the necks, e.g. "
                         "tri,rect,disc -- so each neck gets a different block")
    ap.add_argument("--size", default="",
                    help="WxH in metres, e.g. 4.5x5.0 (default: the clutter_ext_w10 "
                         "triangle, 3.39x3.80)")
    ap.add_argument("--out", default=str(_ROOT / "maps" / "scenario_neck"))
    a = ap.parse_args()

    global SIZE
    if a.size:
        _w, _h = a.size.lower().split("x")
        SIZE = (float(_w), float(_h))
    base = msp.BaseMap(a.base)
    necks, w = neck_edges(base)
    print(f"[neck] {a.base}: {base.s[-1]:.0f} m, {len(necks)} necks")

    # candidate stations: just before each entry, just after each exit
    stations = []
    for (s_in, s_out) in necks:
        stations.append(("entry", s_in - a.standoff))
        stations.append(("exit", s_out + a.standoff))

    kinds = [k.strip() for k in a.kinds.split(",") if k.strip()] or [a.kind]
    placed = []
    wimg = base.img.copy()
    side = 1.0
    n_try = 0
    for tag, s_target in stations:
        if s_target < 3.0 or s_target > base.s[-1] - 3.0:
            continue
        k = int(np.argmin(np.abs(base.s - s_target)))
        half = float(base.base_clear[k])
        if half < 3.5:                      # need a wide section to fit a big block
            print(f"[neck] skip {tag} at s={s_target:.1f}: half-width only {half:.2f} m")
            continue
        # alternate which side is blocked so the patch has to swerve both ways
        side = -side
        kind_here = kinds[n_try % len(kinds)]
        n_try += 1
        if a.attach == "wall":
            # grow the block OUT OF THE WALL, like the intrusions in
            # clutter_ext_w10: its outer edge sits past the wall line and it
            # protrudes `intrude` metres into the corridor, so the corridor is
            # narrowed asymmetrically and the only line is on the far side.
            reach = half - a.intrude + 0.5 * max(SIZE)
            off = side * reach
        else:
            off = side * a.offset_frac * 2.0 * half
        ox = base.cx[k] + off * base.nrm[k, 0]
        oy = base.cy[k] + off * base.nrm[k, 1]
        angle = float(45.0 if side > 0 else 225.0)

        trial = wimg.copy()
        msp._raster_poly(trial, base, msp._obstacle_polygon(kind_here, ox, oy, SIZE, angle))
        near = np.where(np.abs(base.s - base.s[k]) <= 3.0)[0]
        gap_after = min(base.passage_width(trial, int(n)) for n in near)
        if gap_after < a.pass_min:
            print(f"[neck] skip {tag} at s={s_target:.1f}: would leave {gap_after:.2f} m")
            continue
        wimg = trial
        placed.append(dict(kind=kind_here, family="neck_" + tag, big=True,
                           center_m=[round(float(ox), 4), round(float(oy), 4)],
                           size_m=[SIZE[0], SIZE[1]],
                           angle_deg=angle, s_m=round(float(base.s[k]), 2),
                           blocked_side=("port" if side > 0 else "starboard"),
                           lane_left_m=round(float(gap_after), 2)))
        print(f"[neck] {tag:5s} s={base.s[k]:6.1f}  {kind_here:4s}  blocks "
              f"{'port' if side>0 else 'starboard'}  lane left {gap_after:.2f} m")

    out_root = pathlib.Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    info = msp.write_variant(base, a.name, placed, out_root)
    info.update(obstacles=placed, n_obstacles=len(placed), necks=necks)
    with open(out_root / f"{a.name}_index.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"\n[neck] {a.name}: {len(placed)} large triangles at neck throats, "
          f"tightest drivable gap {info['gap_width_m']:.2f} m -> {out_root}/")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import yaml
        from PIL import Image
        from PIL.Image import Transpose
        d = out_root / a.name
        spec = yaml.safe_load(open(d / f"{a.name}_map.yaml"))
        im = np.array(Image.open(d / spec["image"]).transpose(Transpose.FLIP_TOP_BOTTOM))
        res = float(spec["resolution"]); ox0, oy0 = spec["origin"][0], spec["origin"][1]
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(im, cmap="gray", origin="lower",
                  extent=[ox0, ox0 + im.shape[1]*res, oy0, oy0 + im.shape[0]*res])
        ax.plot(base.cx, base.cy, "c-", lw=0.9)
        _mk = {"tri": "^", "rect": "s", "disc": "o"}
        for o in placed:
            ax.plot(*o["center_m"], _mk.get(o["kind"], "^"), color="red",
                    ms=13, mec="k", mew=0.7)
        for (s_in, s_out) in necks:
            for sv in (s_in, s_out):
                kk = int(np.argmin(np.abs(base.s - sv)))
                ax.plot(base.cx[kk], base.cy[kk], "o", color="deepskyblue", ms=5)
        ax.set_title(f"{a.name}   {base.s[-1]:.0f} m   {len(placed)} large triangles at "
                     f"neck throats\nred = free-standing block (offset to force a swerve): "
                     f"^ triangle, square = rect, o = disc; "
                     f"blue = neck entry/exit; no wall obstacles", fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
        fig.tight_layout()
        p = _ROOT / "scenarios" / f"preview_{a.name}.png"
        fig.savefig(p, dpi=110)
        print(f"[neck] preview -> {p}")
    except Exception as e:
        print(f"[neck] preview failed ({e})")


if __name__ == "__main__":
    main()
