#!/usr/bin/env python3
"""
`open_narrow_obs`, extended: the SAME corridor, chained so it has 3 narrow
tunnels instead of 1, with 8-10 obstacles the same size as the ones drawn in
scenarios/open_narrow_obs.png.

Why not on_ext: `make_open_narrow_extended.py` synthesises a fresh course out
of lines and arcs, and the result is a staircase that looks nothing like
open_narrow. This instead REPLICATES open_narrow's own centreline and measured
half-width profile end to end (the same approach as
make_open_narrow_replicated.py), so every metre of the result is open_narrow
geometry -- same bend radius, same 9.25 m wide sections, same ~55 m tunnel --
just repeated, giving 3 tunnels.

Obstacles: open_narrow_obs's three are 3.2-3.8 m across and sit on/next to the
centreline in the wide sections. The stock scenario_pool sampler uses 0.5-1.9 m,
which reads as a different kind of map, so OBST_SIZE is overridden to match.
Placement keeps the standard feasibility rule -- a >= 2.5 m lane must survive
within 3 m of every obstacle -- so the course stays drivable.

    python presentation_code/make_open_narrow_obs_extended.py --copies 3 --n-obst 10
"""
from __future__ import annotations
import argparse, json, os, pathlib, sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from make_foundation_scenarios import build, GYM_MAPS          # noqa: E402
from make_open_narrow_replicated import make_scn               # noqa: E402
import make_scenario_pool as msp                               # noqa: E402

_ROOT = pathlib.Path(_HERE).parent
OUT_DIR = _ROOT / "maps" / "scenario_on_obs_ext"
NAME = "on_obs_ext"

# match the obstacles drawn in scenarios/open_narrow_obs.png (3.2-3.8 m across)
OBST_SIZE_MATCHED = (3.0, 3.9)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--copies", type=int, default=3, help="open_narrow copies == tunnels")
    ap.add_argument("--n-obst", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pass-min", type=float, default=4.5,
                    help="minimum drivable lane that must survive beside every "
                         "obstacle (m). open_narrow_obs's own obstacles leave "
                         "4.81 m; the pool default of 2.5 m makes a far harder map.")
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args()

    occ, origin, path, info = build(NAME, make_scn(args.copies), GYM_MAPS)
    print(f"[on_obs_ext] base built: {info['length_m']} m, {args.copies}x open_narrow")

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    base = msp.BaseMap(NAME)
    rng = np.random.default_rng(args.seed)

    old, old_pass = msp.OBST_SIZE, msp.PASS_MIN
    msp.OBST_SIZE = OBST_SIZE_MATCHED          # big, open_narrow_obs-style blocks
    msp.PASS_MIN = args.pass_min               # ... and the same lane they leave
    try:
        placed = msp.sample_layout(base, rng, args.n_obst, min_s_spacing=8.0)
    finally:
        msp.OBST_SIZE, msp.PASS_MIN = old, old_pass

    variants = [msp.write_variant(base, f"{NAME}_clean", [], out_root),
                msp.write_variant(base, f"{NAME}_obs", placed, out_root)]
    for v in variants:
        print(f"[on_obs_ext] {v['name']:16s} n_obst={v['n_obstacles']:2d}  "
              f"tightest drivable gap {v['gap_width_m']:.2f} m")
    if len(placed) < args.n_obst:
        print(f"[on_obs_ext] NOTE: {len(placed)}/{args.n_obst} obstacles placed -- "
              f"the rest were rejected for leaving < {args.pass_min} m of lane")

    with open(out_root / "variants_index.json", "w") as f:
        json.dump({"base_maps": [NAME], "copies": args.copies,
                   "obst_size_m": list(OBST_SIZE_MATCHED),
                   "seed": args.seed, "variants": variants}, f, indent=2)

    # preview: this map beside the original open_narrow_obs it is meant to match
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import yaml
        from PIL import Image
        from PIL.Image import Transpose
        fig, axes = plt.subplots(1, 2, figsize=(13, 6))
        for ax, (nm, d) in zip(axes, [
                ("open_narrow_obs", _ROOT / "maps" / "scenario_on_obs" / "open_narrow_obs"),
                (f"{NAME}_obs", out_root / f"{NAME}_obs")]):
            spec = yaml.safe_load(open(f"{d}/{nm}_map.yaml"))
            im = np.array(Image.open(f"{d}/{spec['image']}").transpose(Transpose.FLIP_TOP_BOTTOM))
            res = float(spec["resolution"]); ox, oy = spec["origin"][0], spec["origin"][1]
            ax.imshow(im, cmap="gray", origin="lower",
                      extent=[ox, ox + im.shape[1] * res, oy, oy + im.shape[0] * res])
            cl = np.loadtxt(f"{d}/{nm}_centerline.csv", delimiter=",", comments="#")
            ax.plot(cl[:, 0], cl[:, 1], "c-", lw=0.7)
            S = np.concatenate([[0.], np.cumsum(np.hypot(np.diff(cl[:, 0]), np.diff(cl[:, 1])))])
            ob = yaml.safe_load(open(f"{d}/{nm}_obs_pos.yaml")).get("obstacles", [])
            for o in ob:
                ax.plot(*o["center_m"], "o", color="red", ms=6, mec="k")
            ax.set_title(f"{nm}\n{S[-1]:.0f} m, {len(ob)} obstacles", fontsize=10)
            ax.set_aspect("equal")
        fig.tight_layout()
        p = _ROOT / "scenarios" / f"preview_{NAME}.png"
        fig.savefig(p, dpi=100)
        print(f"[on_obs_ext] preview -> {p}")
    except Exception as e:
        print(f"[on_obs_ext] preview failed ({e}) -- maps are still written")


if __name__ == "__main__":
    main()
