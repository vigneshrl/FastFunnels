#!/usr/bin/env python3
"""
A slalom with MORE bars, so the patch has to weave continuously instead of
running straight between widely-spaced gates.

mf_slalom_g7_5 (the version the patch can actually drive) has 6 gates 22 m
apart on a 149 m course -- long straights with an occasional deflection. Adding
gates and shortening the spacing makes the centreline weave the whole way, so
the patch is turning almost continuously.

Gate width is kept at the 7.5 m that the patch can pass: measured on the
earlier sweep, 7.0 m fails at 12% and 7.5 m arrives at 100%, and the bar length
is derived from the gate (depth = 2*WIDE_HALF - gap), so a narrower gate is a
longer bar and puts us back under the traversability cliff.

    python presentation_code/make_slalom_zigzag.py --gates 12 --spacing 12 --gap 7.5
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import pathlib
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_foundation_scenarios import (        # noqa: E402
    scn_slalom, build, GYM_MAPS, STAGE_MAPS, WIDE_HALF,
)

_ROOT = pathlib.Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gates", type=int, default=12)
    ap.add_argument("--spacing", type=float, default=12.0)
    ap.add_argument("--gap", type=float, default=7.5)
    ap.add_argument("--bar-thick", type=float, default=2.0)
    ap.add_argument("--name", default="")
    a = ap.parse_args()

    name = a.name or f"mf_slalom_z{a.gates}"
    depth = 2.0 * WIDE_HALF - a.gap
    fn = functools.partial(scn_slalom, n_gates=a.gates, spacing=a.spacing,
                           gap=a.gap, bar_thick=a.bar_thick)
    occ, origin, path, info = build(name, fn, GYM_MAPS)

    dst = STAGE_MAPS / name
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(GYM_MAPS / name, dst)

    w = info["corridor_full_width_m"]
    print(f"[zigzag] {name}: {a.gates} gates, {a.spacing:.0f} m apart, gap {a.gap} m, "
          f"bars {depth:.2f} m")
    print(f"[zigzag] course {info['length_m']} m   width "
          f"{w['min']}/{w['median']}/{w['max']} m   Rmin {info['min_curv_radius_m']} m")
    info.update(n_gates=a.gates, spacing_m=a.spacing, gap_m=a.gap,
                bar_length_m=round(depth, 3))
    with open(STAGE_MAPS / f"{name}_index.json", "w") as f:
        json.dump(info, f, indent=2)
    print(f"[zigzag] wrote {dst}")


if __name__ == "__main__":
    main()
