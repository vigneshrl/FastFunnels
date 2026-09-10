#!/usr/bin/env python3
"""
Escalating obstacle groups, one group per wide zone between the necks.

on_obs_ext is open_narrow chained 3x, so it has 3 necks and 4 wide zones:

    zone 1  s   0 -  41 m   before neck 1     ->  1 obstacle
    zone 2  s 101 - 161 m   before neck 2     ->  2 obstacles
    zone 3  s 221 - 279 m   before neck 3     ->  4 obstacles
    zone 4  s 339 - 356 m   after  neck 3     ->  8 obstacles

The patch therefore meets a steadily harder gauntlet before each tunnel: one
block, then a pair, then four, then eight. The FIRST obstacle is pinned to
s=41.2 with the size of the triangle in
baselines/maps/extended_open_narrow_clutter10, so the opening of the course is
identical to the map that run came from.

Obstacles sit ON THE CENTERLINE (offset 0 by default), like the blocks in
baselines/maps/extended_open_narrow_clutter10: the patch meets each one head-on
and has to steer AROUND it -- port or starboard -- instead of a one-sided wall
intrusion that it can shrink past. Shapes cycle triangle / rect / disc.

Every candidate is rejection-tested: a >= --pass-min drivable lane must survive
within 3 m, so the course stays passable.

    python presentation_code/make_escalating_neck_map.py --name neck_escalate
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

FIRST_S = 41.2                     # the original triangle's station
FIRST_SIZE = (3.39, 3.80)          # ... and its size
KINDS = ["tri", "rect", "disc"]


def zones(base, narrow_below=5.0):
    """Wide stretches between/after the necks, in order along the course."""
    w = np.array([base.passage_width(base.img, k) for k in range(len(base.s))])
    narrow = w < narrow_below
    wide = ~narrow
    idx = np.where(wide)[0]
    grp = np.split(idx, np.where(np.diff(idx) > 3)[0] + 1)
    out = []
    for g in grp:
        s0, s1 = float(base.s[g[0]]), float(base.s[g[-1]])
        if s1 - s0 >= 8.0:
            out.append((s0, s1))
    return out, w


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="on_obs_ext")
    ap.add_argument("--name", default="neck_escalate")
    ap.add_argument("--counts", default="1,2,4,8",
                    help="obstacles per wide zone, in order along the course")
    ap.add_argument("--size", default="3.39x3.80")
    ap.add_argument("--first-size", default="",
                    help="override the pinned first triangle's size, WxH in metres. "
                         "Default keeps extended_open_narrow_clutter10's 3.39x3.80. "
                         "Centred, that block eats ~4.25 m of a 9.25 m corridor and "
                         "leaves a 2.50 m committed lane -- +-0.25 m of tolerance for "
                         "a funnel whose floor is 2.0 m wide.")
    ap.add_argument("--offset-frac", type=float, default=0.0,
                    help="lateral offset as a fraction of the half-width; 0 = the "
                         "obstacle sits ON the centerline and the patch must go "
                         "around it (the default, and what clutter10 looks like)")
    ap.add_argument("--pass-min", type=float, default=2.6)
    ap.add_argument("--margin", type=float, default=4.0,
                    help="keep this far from a zone's own ends (m)")
    ap.add_argument("--out", default=str(_ROOT / "maps" / "scenario_neck"))
    a = ap.parse_args()

    _w, _h = a.size.lower().split("x")
    size = (float(_w), float(_h))
    global FIRST_SIZE
    if a.first_size:
        _fw, _fh = a.first_size.lower().split("x")
        FIRST_SIZE = (float(_fw), float(_fh))
    counts = [int(c) for c in a.counts.split(",")]

    base = msp.BaseMap(a.base)
    zs, w = zones(base)
    print(f"[esc] {a.base}: {base.s[-1]:.0f} m, {len(zs)} wide zones")
    for i, (s0, s1) in enumerate(zs):
        n = counts[i] if i < len(counts) else 0
        print(f"[esc]   zone {i+1}: s {s0:6.1f} - {s1:6.1f} m ({s1-s0:5.1f} m) -> {n}")

    placed = []
    wimg = base.img.copy()
    side = 1.0
    n_shape = 0

    for zi, (s0, s1) in enumerate(zs):
        n_want = counts[zi] if zi < len(counts) else 0
        if n_want <= 0:
            continue
        # A fixed 4 m margin swallows a short zone (zone 4 is only 19 m), so cap
        # it at 15% of the zone length -- that keeps the group inside the wide
        # stretch without squeezing 8 blocks into a couple of metres.
        margin = min(a.margin, 0.15 * (s1 - s0))
        lo, hi = s0 + margin, s1 - margin
        if hi <= lo:
            lo, hi = s0 + 1.0, s1 - 1.0
        # evenly spaced stations across the zone
        # zone 1's single block is PINNED to the original triangle's station so
        # the opening of the course matches extended_open_narrow_clutter10.
        # The margin must not veto it: FIRST_S sits near the zone's far end.
        if zi == 0 and n_want == 1 and s0 <= FIRST_S <= s1:
            targets = [FIRST_S]
        else:
            targets = list(np.linspace(lo, hi, n_want))

        # Auto-fit: with the blocks ON the centerline they also have to clear
        # each ALONG the course. If the spacing this zone can offer is tighter
        # than the block, the group would fuse into one wall -- so scale the
        # block down to 70% of the pitch. Escalation then reads as "more and
        # tighter", not "impossible".
        zone_size = size
        if n_want > 1:
            pitch = (hi - lo) / (n_want - 1)
            if pitch < max(size):
                f = max(0.45, 0.70 * pitch / max(size))
                zone_size = (size[0] * f, size[1] * f)
                print(f"[esc]   zone {zi+1}: pitch {pitch:.1f} m < block "
                      f"{max(size):.1f} m -> scale {f:.2f} "
                      f"({zone_size[0]:.2f} x {zone_size[1]:.2f} m)")
        for st in targets:
            k = int(np.argmin(np.abs(base.s - st)))
            half = float(base.base_clear[k])
            if half < 3.0:
                print(f"[esc]   skip s={st:.1f}: half-width {half:.2f} m")
                continue
            side = -side
            kind = KINDS[n_shape % len(KINDS)]
            sz = FIRST_SIZE if (zi == 0 and abs(st - FIRST_S) < 0.5) else zone_size
            kd = "tri" if (zi == 0 and abs(st - FIRST_S) < 0.5) else kind
            n_shape += 1
            off = side * a.offset_frac * 2.0 * half
            ox = base.cx[k] + off * base.nrm[k, 0]
            oy = base.cy[k] + off * base.nrm[k, 1]
            angle = float(45.0 if side > 0 else 225.0)

            trial = wimg.copy()
            msp._raster_poly(trial, base, msp._obstacle_polygon(kd, ox, oy, sz, angle))
            near = np.where(np.abs(base.s - base.s[k]) <= 3.0)[0]
            gap = min(base.passage_width(trial, int(nn)) for nn in near)
            if gap < a.pass_min:
                print(f"[esc]   skip s={st:.1f}: would leave {gap:.2f} m")
                continue
            wimg = trial
            placed.append(dict(kind=kd, zone=zi + 1, big=True,
                               center_m=[round(float(ox), 4), round(float(oy), 4)],
                               size_m=[sz[0], sz[1]], angle_deg=angle,
                               s_m=round(float(base.s[k]), 2),
                               blocked_side=("port" if side > 0 else "starboard"),
                               lane_left_m=round(float(gap), 2)))

    by_zone = {}
    for o in placed:
        by_zone[o["zone"]] = by_zone.get(o["zone"], 0) + 1
    print(f"\n[esc] placed {len(placed)}: " +
          ", ".join(f"zone {z}={by_zone.get(z,0)}" for z in sorted(set(
              list(by_zone) + list(range(1, len(counts) + 1))))))

    out_root = pathlib.Path(a.out); out_root.mkdir(parents=True, exist_ok=True)
    info = msp.write_variant(base, a.name, placed, out_root)
    info.update(obstacles=placed, n_obstacles=len(placed), counts=counts,
                zones=zs, per_zone=by_zone)
    with open(out_root / f"{a.name}_index.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[esc] {a.name}: tightest drivable gap {info['gap_width_m']:.2f} m "
          f"-> {out_root}/")

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
        mk = {"tri": "^", "rect": "s", "disc": "o"}
        for o in placed:
            ax.plot(*o["center_m"], mk.get(o["kind"], "^"), color="red",
                    ms=10, mec="k", mew=0.6)
        ax.set_title(f"{a.name}   {base.s[-1]:.0f} m   {len(placed)} obstacles, "
                     f"escalating per zone " +
                     "/".join(str(by_zone.get(z, 0)) for z in range(1, len(counts)+1)) +
                     "\n^ tri, square rect, o disc; ON the centerline -- the patch "
                     "must steer around each one",
                     fontsize=10)
        ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_aspect("equal")
        fig.tight_layout()
        p = _ROOT / "scenarios" / f"preview_{a.name}.png"
        fig.savefig(p, dpi=110)
        print(f"[esc] preview -> {p}")
    except Exception as e:
        print(f"[esc] preview failed ({e})")


if __name__ == "__main__":
    main()
