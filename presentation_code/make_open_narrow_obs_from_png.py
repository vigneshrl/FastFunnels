#!/usr/bin/env python3
"""
Rebuild `open_narrow_obs` as a real f1tenth_gym map from the reference drawing
`scenarios/open_narrow_obs.png`.

The PNG is open_narrow's exact geometry with a few obstacles drawn into the
corridor, but there is no map directory for it. Rather than eyeball the
obstacle positions, this DIFFS the drawing against the base map: any pixel the
drawing marks occupied that the base map says is free (and that sits well
inside the corridor, so wall-outline anti-aliasing cannot leak in) is an
obstacle. The blobs are then burned into a copy of the base occupancy grid, so
the result matches the drawing rather than an approximation of it.

    python presentation_code/make_open_narrow_obs_from_png.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil

import numpy as np
import yaml
from PIL import Image
from PIL.Image import Transpose
from scipy import ndimage

_ROOT = pathlib.Path(__file__).resolve().parents[1]
GYM_MAPS = _ROOT / "f1tenth_gym" / "maps"
BASE = "open_narrow"
REF_PNG = _ROOT / "scenarios" / "open_narrow_obs.png"

WALL_ERODE = 8       # px of the free space next to every wall to ignore
MIN_BLOB_PX = 40     # ignore specks (anti-aliasing, stray marks)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="open_narrow_obs")
    ap.add_argument("--ref", default=str(REF_PNG))
    ap.add_argument("--out-pool", default=str(_ROOT / "maps" / "scenario_on_obs"))
    args = ap.parse_args()

    bdir = GYM_MAPS / BASE
    spec = yaml.safe_load(open(bdir / f"{BASE}_map.yaml"))
    res = float(spec["resolution"])
    ox, oy = float(spec["origin"][0]), float(spec["origin"][1])
    base = np.array(Image.open(bdir / spec["image"]).transpose(Transpose.FLIP_TOP_BOTTOM))
    H, W = base.shape

    # reference drawing -> same orientation/size as the base grid
    ref = Image.open(args.ref).convert("L")
    ref = ref.resize((W, H), Image.NEAREST).transpose(Transpose.FLIP_TOP_BOTTOM)
    ref = np.array(ref)

    base_free = base > 128
    deep_free = ndimage.binary_erosion(base_free, iterations=WALL_ERODE)
    cand = (ref < 128) & deep_free              # drawn-occupied but free in the base
    cand = ndimage.binary_opening(cand, np.ones((3, 3), bool))

    lab, n = ndimage.label(cand)
    keep = np.zeros_like(cand)
    obstacles = []
    for i in range(1, n + 1):
        m = lab == i
        if m.sum() < MIN_BLOB_PX:
            continue
        keep |= m
        r, c = np.nonzero(m)
        cx_m = (c.mean() + 0.5) * res + ox
        cy_m = (r.mean() + 0.5) * res + oy
        obstacles.append(dict(
            center_m=[round(float(cx_m), 3), round(float(cy_m), 3)],
            size_m=[round(float((c.max() - c.min() + 1) * res), 3),
                    round(float((r.max() - r.min() + 1) * res), 3)],
            area_px=int(m.sum())))
    print(f"[on_obs] {len(obstacles)} obstacles recovered from {os.path.basename(args.ref)}")
    for k, o in enumerate(obstacles):
        print(f"   {k}: centre {o['center_m']} m  size {o['size_m']} m  ({o['area_px']} px)")
    if not obstacles:
        raise SystemExit("no obstacles found -- reference/base mismatch?")

    img = base.copy()
    img[keep] = 0

    name = args.name
    for dest in (GYM_MAPS / name, pathlib.Path(args.out_pool) / name):
        dest.mkdir(parents=True, exist_ok=True)
        Image.fromarray(np.flipud(img)).save(dest / f"{name}_map.pgm")
        with open(dest / f"{name}_map.yaml", "w") as f:
            yaml.safe_dump({"image": f"{name}_map.pgm", "resolution": res,
                            "origin": [ox, oy, 0.0], "negate": 0,
                            "occupied_thresh": 0.45, "free_thresh": 0.196},
                           f, default_flow_style=None, sort_keys=False)
        shutil.copy(bdir / f"{BASE}_centerline.csv", dest / f"{name}_centerline.csv")
        if (bdir / f"{BASE}_bounds.json").exists():
            shutil.copy(bdir / f"{BASE}_bounds.json", dest / f"{name}_bounds.json")
        with open(dest / f"{name}_obs_pos.yaml", "w") as f:
            yaml.safe_dump({"scenario": name, "base_map": BASE,
                            "toll_pillars": [], "obstacles": obstacles,
                            "narrow_regions": []},
                           f, default_flow_style=None, sort_keys=False)
        print(f"[on_obs] wrote {dest}")

    pool = pathlib.Path(args.out_pool)
    with open(pool / "variants_index.json", "w") as f:
        json.dump({"base_maps": [BASE],
                   "variants": [dict(name=name, base_map=BASE,
                                     n_obstacles=len(obstacles))]}, f, indent=2)

    # quick look at what was burned in
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(11, 5.5))
        ax[0].imshow(ref, cmap="gray", origin="lower"); ax[0].set_title("reference drawing")
        ax[1].imshow(img, cmap="gray", origin="lower")
        cl = np.loadtxt(bdir / f"{BASE}_centerline.csv", delimiter=",", comments="#")
        ax[1].plot((cl[:, 0] - ox) / res, (cl[:, 1] - oy) / res, "c-", lw=0.8)
        for o in obstacles:
            ax[1].plot((o["center_m"][0] - ox) / res, (o["center_m"][1] - oy) / res,
                       "o", color="red", ms=8, mec="k")
        ax[1].set_title(f"{name}  ({len(obstacles)} obstacles burned in)")
        for a in ax:
            a.set_aspect("equal")
        fig.tight_layout()
        p = _ROOT / "scenarios" / f"preview_{name}.png"
        fig.savefig(p, dpi=110)
        print(f"[on_obs] preview -> {p}")
    except Exception as e:
        print(f"[on_obs] preview failed ({e})")


if __name__ == "__main__":
    main()
