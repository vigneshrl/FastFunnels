#!/usr/bin/env python3
"""
Wider-gate variants of `mf_slalom`.

The original `mf_slalom` has 4.0 m slots, so the funnel has only 2.0 m of
half-width at each gate.  The patch's own half-width `b` averages ~1.9 m and
reaches ~3.0 m, so the funnel physically cannot fit through -- the legacy
checkpoint dies at 9.9% progress on the FIRST gate, and that is geometry, not
policy failure.  Every other mf_* map is 9.25 m wide throughout.

In `scn_slalom` the bar length is derived from the slot:

    depth = 2 * WIDE_HALF - gap        (WIDE_HALF = 4.625 m)

so widening `gap` is exactly "make the bars shorter" -- the bar's inner face
still lands on the slot edge, the weave keeps the same shape, only the
protrusion shrinks:

    gap 4.0 m -> bars 5.25 m   (original; funnel half-width 2.00 m)
    gap 5.0 m -> bars 4.25 m   (funnel half-width 2.50 m)
    gap 6.0 m -> bars 3.25 m   (funnel half-width 3.00 m)
    gap 7.0 m -> bars 2.25 m   (funnel half-width 3.50 m)

Written as NEW maps (`mf_slalom_g5_0`, ...) so the original `mf_slalom` and
every result already measured against it stay valid.

    python presentation_code/make_slalom_widegate.py --gaps 5.0,6.0,7.0
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
from make_foundation_scenarios import (      # noqa: E402
    scn_slalom, build, GYM_MAPS, STAGE_MAPS, WIDE_HALF,
)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gaps", default="5.0,6.0,7.0",
                   help="comma-separated slot widths in metres")
    args = p.parse_args()

    gaps = [float(g) for g in args.gaps.split(",")]
    STAGE_MAPS.mkdir(parents=True, exist_ok=True)
    made = []
    for gap in gaps:
        depth = 2.0 * WIDE_HALF - gap
        if depth <= 0.3:
            print(f"[slalom] gap {gap} m leaves no bar at all -- skipped")
            continue
        name = f"mf_slalom_g{gap:.1f}".replace(".", "_")
        fn = functools.partial(scn_slalom, gap=gap)
        occ, origin, path, info = build(name, fn, GYM_MAPS)

        dst = STAGE_MAPS / name
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(GYM_MAPS / name, dst)

        info["slot_gap_m"] = gap
        info["bar_length_m"] = round(depth, 3)
        made.append(info)
        w = info["corridor_full_width_m"]
        print(f"[slalom] {name:18s} gap {gap:.1f} m  bars {depth:.2f} m  "
              f"(was 5.25 m)  width {w['min']}/{w['median']}/{w['max']} m")

    idx_path = STAGE_MAPS / "index_slalom_gates.json"
    with open(idx_path, "w") as f:
        json.dump({"variants": made}, f, indent=2)
    print(f"\n[slalom] {len(made)} maps -> {STAGE_MAPS}/  (index: {idx_path.name})")


if __name__ == "__main__":
    main()
