#!/usr/bin/env python3
"""
Can the PATCH actually get through this map?

The obstacle generators guarantee a drivable lane for the CAR (0.58 x 0.31 m).
That is not the same question: the patch terminates on `collision_patch_boundary`
when its FUNNEL touches an occupied cell, and the funnel is an ellipse of
half-length a (floor 1.5 m -> 3 m long) and half-width b (floor 1.0 m -> 2 m
wide) rigidly aligned with the car's heading. A corridor can leave 2.5 m of lane
and still be impossible for a 3 m long ellipse to thread, especially where the
gap is offset side-to-side between consecutive obstacles.

This sweeps the minimum funnel along the centreline -- and, because the patch
does not have to stay on the centreline, also tries lateral offsets and small
heading changes at each station -- and reports where NO placement of the
minimum funnel is collision-free. Those stations are hard blocks: no policy can
pass them, so a failure there is the map's fault, not the policy's.

    python presentation_code/check_funnel_traversable.py \
        --pool maps/scenario_clutter --map clutter_ext
"""
from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(pool, name):
    d = pathlib.Path(pool) / name
    spec = yaml.safe_load(open(d / f"{name}_map.yaml"))
    res = float(spec["resolution"])
    ox, oy = float(spec["origin"][0]), float(spec["origin"][1])
    img = np.array(Image.open(d / spec["image"]).transpose(Transpose.FLIP_TOP_BOTTOM))
    cl = np.loadtxt(d / f"{name}_centerline.csv", delimiter=",", comments="#")
    return img, res, ox, oy, cl


def ellipse_free(img, res, ox, oy, cx, cy, th, a, b, n_r=5, n_t=24):
    """True if the filled ellipse at (cx,cy,th) touches no occupied cell."""
    H, W = img.shape
    rs = np.linspace(0.35, 1.0, n_r)
    ts = np.linspace(0, 2 * math.pi, n_t, endpoint=False)
    ct, st = math.cos(th), math.sin(th)
    for r in rs:
        xl = r * a * np.cos(ts)
        yl = r * b * np.sin(ts)
        xs = cx + xl * ct - yl * st
        ys = cy + xl * st + yl * ct
        c = np.clip(((xs - ox) / res).astype(int), 0, W - 1)
        rr = np.clip(((ys - oy) / res).astype(int), 0, H - 1)
        if np.any(img[rr, c] <= 128):
            return False
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--a", type=float, default=1.5, help="funnel half-length floor")
    ap.add_argument("--b", type=float, default=1.0, help="funnel half-width floor")
    ap.add_argument("--lat", type=float, default=3.5, help="max lateral offset to try (m)")
    ap.add_argument("--dpsi", type=float, default=25.0, help="max heading deviation (deg)")
    a_ = ap.parse_args()

    pool = a_.pool if os.path.isabs(a_.pool) else str(_ROOT / a_.pool)
    img, res, ox, oy, cl = load(pool, a_.map)
    cx, cy = cl[:, 0], cl[:, 1]
    S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
    L = float(S[-1])
    d1 = np.gradient(np.stack([cx, cy], 1), axis=0)
    tg = d1 / (np.hypot(d1[:, 0], d1[:, 1])[:, None] + 1e-9)
    nrm = np.stack([-tg[:, 1], tg[:, 0]], 1)
    hdg = np.arctan2(tg[:, 1], tg[:, 0])

    offs = np.arange(-a_.lat, a_.lat + 1e-6, 0.15)
    psis = np.radians(np.arange(-a_.dpsi, a_.dpsi + 1e-6, 5.0))

    blocked, best_clear = [], []
    for k in range(len(cx)):
        ok = False
        for off in offs:
            px = cx[k] + off * nrm[k, 0]
            py = cy[k] + off * nrm[k, 1]
            for dp in psis:
                if ellipse_free(img, res, ox, oy, px, py, hdg[k] + dp, a_.a, a_.b):
                    ok = True
                    break
            if ok:
                break
        best_clear.append(ok)
        if not ok:
            blocked.append(k)

    frac = 100.0 * (1.0 - len(blocked) / len(cx))
    print(f"map {a_.map}   {L:.0f} m   funnel a={a_.a} b={a_.b} "
          f"({2*a_.a:.1f} x {2*a_.b:.1f} m ellipse)")
    print(f"stations where the MINIMUM funnel fits somewhere: "
          f"{len(cx)-len(blocked)}/{len(cx)}  ({frac:.1f}%)")
    if not blocked:
        print("=> traversable: a minimum funnel has a collision-free placement everywhere")
        return
    grp = np.split(np.array(blocked), np.where(np.diff(blocked) > 2)[0] + 1)
    print(f"=> {len(grp)} HARD BLOCK(S) -- no placement of the minimum funnel fits:")
    for g in grp:
        print(f"     s {S[g[0]]:7.1f} - {S[g[-1]]:7.1f} m   "
              f"({S[g[0]]/L:5.1%} - {S[g[-1]]/L:5.1%} of course)")
    print("\nThese are the map's fault, not a policy's: widen the gap there "
          "(raise --pass-min when generating) or drop those obstacles.")


if __name__ == "__main__":
    main()
