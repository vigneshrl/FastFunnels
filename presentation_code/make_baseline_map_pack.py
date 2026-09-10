#!/usr/bin/env python3
"""
Assemble `baselines/maps/` -- the five wall-obstacle base maps, one subfolder
each, in the form a baseline planner needs:

    <name>_map.pgm        occupancy grid (walls + obstacles burned in)
    <name>_map.yaml       resolution / origin, so pixels -> metres
    <name>_centerline.csv the reference path
    <name>.png            rendered view (grid + centreline + obstacle markers)
    <name>_obs_pos.yaml   obstacle list, where the source map has one
    <name>_bounds.json    track bounds, where the source map has one

The five, and where each comes from:
    open_narrow_obs        bw_on000            open_narrow + wall obstacles
    extended_open_narrow_obs bw_on_obs_ext000  3-tunnel 356 m chain + wall obstacles
    lshape_obs             sw_lshape001        mf_lshape + wall obstacles
    zigzag_obs             sw_zigzag001        mf_zigzag + wall obstacles
    slalom_obs             sw_slalom_g7_5000   widened slalom + wall obstacles

    python presentation_code/make_baseline_map_pack.py
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil

import numpy as np
import yaml

_ROOT = pathlib.Path(__file__).resolve().parents[1]
_MAIN = pathlib.Path("/p/cral/vignesh/bigtemp_files/FastFunnels")

# out-name -> (source pool dir, source map name)
MAPS = {
    "open_narrow_obs":          ("maps/baseline_walls",   "bw_open_narrow000"),
    "extended_open_narrow_obs": ("maps/baseline_walls",   "bw_on_obs_ext000"),
    "lshape_obs":               ("maps/scenario_walls",   "sw_lshape001"),
    "zigzag_obs":               ("maps/scenario_walls",   "sw_zigzag001"),
    "slalom_obs":               ("maps/scenario_walls",   "sw_slalom_g7_5000"),
    # the 87% run's course (videos/v7a_ck36M_clutter_w10_87pct.mp4): same 356 m
    # 3-tunnel chain, 10 wall-only obstacles (5 large). Best cluttered result of
    # the campaign -- v7a/checkpoint_36000000 reached 87.0% here, vs 33.8% on the
    # mixed-placement version and 9.3% at 30 wall obstacles.
    "extended_open_narrow_clutter10": ("maps/scenario_clutter", "clutter_ext_w10"),
    # 6 large triangles at the neck throats, offset alternately port/starboard so
    # the funnel must move LATERALLY to pass, not just shrink. No wall obstacles.
    # Best measured: v7a/ck36750000 54.1% (clean, no reversal).
    "neck_swerve":              ("maps/scenario_neck",    "neck_swerve"),
    # 14 slalom gates 11 m apart (vs mf_slalom_g7_5's 6 at 22 m): the centreline
    # weaves continuously, so the patch turns almost the whole way. Gate kept at
    # 7.5 m -- 7.0 m is already below the traversability cliff.
    # Best measured: best_patch/ck15000000 47.7% clean; ck18510000 73.3% but with
    # 9.3% reversal.
    "slalom_zigzag14":          ("maps/map_foundations",  "mf_slalom_zig14"),
    # ESCALATING gauntlet, one group per wide zone between the necks: 1 block
    # before neck 1, 2 before neck 2, 4 before neck 3, 8 after neck 3. The first
    # block is pinned to s=41.2 with the exact triangle of
    # extended_open_narrow_clutter10, so the course opens identically to that
    # map and then gets progressively harder. Free-standing (no wall obstacles),
    # offset alternately port/starboard so the open lane keeps switching side.
    "neck_escalate":            ("maps/scenario_neck",    "neck_escalate"),
    # deeper slalom: 14 gates, 4.5 m gate -> 4.75 m bars reaching well past the
    # midline, weave amplitude 4.7 m (vs 1.7 m for slalom_zigzag14).
    "slalom_longbars45":        ("maps/map_foundations",  "mf_slalom_lb45"),
}


def render(dst: pathlib.Path, out_name: str, spec, img, cl, obstacles):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    res = float(spec["resolution"])
    ox, oy = float(spec["origin"][0]), float(spec["origin"][1])
    h, w = img.shape
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.imshow(img, cmap="gray", origin="lower",
              extent=[ox, ox + w * res, oy, oy + h * res])
    ax.plot(cl[:, 0], cl[:, 1], "c-", lw=1.0, label="centreline")
    for o in obstacles:
        ax.plot(*o["center_m"], "o", color="red", ms=6, mec="k", mew=0.5)
    S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cl[:, 0]),
                                                  np.diff(cl[:, 1])))])
    ax.set_title(f"{out_name}\n{S[-1]:.0f} m, {len(obstacles)} wall obstacles",
                 fontsize=11)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(dst / f"{out_name}.png", dpi=110)
    plt.close(fig)
    return float(S[-1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(_MAIN / "baselines" / "maps"))
    a = ap.parse_args()

    out_root = pathlib.Path(a.out)
    out_root.mkdir(parents=True, exist_ok=True)
    index = []

    for out_name, (pool, src_name) in MAPS.items():
        src = _ROOT / pool / src_name
        if not src.exists():
            src = _MAIN / pool / src_name
        if not src.exists():
            print(f"[pack] MISSING {out_name}: {src}")
            continue
        dst = out_root / out_name
        dst.mkdir(parents=True, exist_ok=True)

        spec = yaml.safe_load(open(src / f"{src_name}_map.yaml"))
        # copy grid + yaml under the new name, fixing the image reference
        shutil.copy(src / spec["image"], dst / f"{out_name}_map.pgm")
        spec_out = dict(spec)
        spec_out["image"] = f"{out_name}_map.pgm"
        with open(dst / f"{out_name}_map.yaml", "w") as f:
            yaml.safe_dump(spec_out, f, default_flow_style=None, sort_keys=False)
        shutil.copy(src / f"{src_name}_centerline.csv",
                    dst / f"{out_name}_centerline.csv")
        for extra, suffix in (("_bounds.json", "_bounds.json"),
                              ("_obs_pos.yaml", "_obs_pos.yaml")):
            p = src / f"{src_name}{extra}"
            if p.exists():
                shutil.copy(p, dst / f"{out_name}{suffix}")

        from PIL import Image
        from PIL.Image import Transpose
        img = np.array(Image.open(dst / f"{out_name}_map.pgm")
                       .transpose(Transpose.FLIP_TOP_BOTTOM))
        cl = np.loadtxt(dst / f"{out_name}_centerline.csv", delimiter=",",
                        comments="#")
        obstacles = []
        op = dst / f"{out_name}_obs_pos.yaml"
        if op.exists():
            obstacles = yaml.safe_load(open(op)).get("obstacles", []) or []
        length = render(dst, out_name, spec, img, cl, obstacles)

        rec = dict(name=out_name, source=f"{pool}/{src_name}",
                   length_m=round(length, 1), n_obstacles=len(obstacles),
                   resolution=float(spec["resolution"]),
                   origin=[float(spec["origin"][0]), float(spec["origin"][1])],
                   files=sorted(p.name for p in dst.iterdir()))
        index.append(rec)
        print(f"[pack] {out_name:26s} {length:6.0f} m  {len(obstacles):2d} obstacles"
              f"  <- {pool}/{src_name}")

    with open(out_root / "index.json", "w") as f:
        json.dump({"maps": index}, f, indent=2)
    print(f"\n[pack] {len(index)} maps -> {out_root}/")


if __name__ == "__main__":
    main()
