#!/usr/bin/env python3
"""Print a combined paper table from the three sweep CSVs."""
import csv, math, sys
from pathlib import Path

HERE = Path(__file__).parent

def load(path):
    rows = []
    try:
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    except FileNotFoundError:
        pass
    return rows

def fmt(v, decimals=3):
    try:
        f = float(v)
        if math.isnan(f): return "  nan"
        if math.isinf(f): return "  inf"
        return f"{f:.{decimals}f}"
    except (ValueError, TypeError):
        return str(v)

orca = load(HERE / "sweep_results_orca.csv")
lf   = load(HERE / "sweep_results_lf.csv")
las  = load(HERE / "las_sweep_results.csv")

HDR = ("baseline", "n_agents", "tgt_spd", "avg_spd", "success",
       "t_zone_s", "flow", "deform", "n_clear", "n_coll", "safety",
       "collision", "spread_entry", "spread_narrow")
W   = (18, 8, 8, 8, 8, 9, 8, 8, 8, 7, 7, 9, 12, 13)

sep = "-" * sum(W)
hdr = "".join(h.rjust(w) for h, w in zip(HDR, W))
print("\n" + sep)
print(hdr)
print(sep)

def print_rows(rows):
    for r in rows:
        vals = [
            r.get("baseline",""),
            r.get("n_agents",""),
            fmt(r.get("target_speed_mps", r.get("vgain","nan")), 1),
            fmt(r.get("avg_speed_mps","nan")),
            fmt(r.get("success","nan")),
            fmt(r.get("time_to_goal_s","nan")),
            fmt(r.get("flow_rate","nan")),
            fmt(r.get("deformability","nan")),
            fmt(r.get("n_completed","nan"), 1),
            fmt(r.get("n_collided","nan"), 1),
            fmt(r.get("safety_rate","nan")),
            fmt(r.get("collision","nan")),
            fmt(r.get("spread_at_entry_m","nan")),
            fmt(r.get("spread_at_narrow_m","nan")),
        ]
        print("".join(v.rjust(w) for v, w in zip(vals, W)))

print_rows(orca)
if orca: print()
print_rows(lf)
if lf: print()
print_rows(las)
print(sep + "\n")
