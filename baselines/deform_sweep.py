#!/usr/bin/env python3
"""
deform_sweep.py — Aggregate DEFORM result CSVs into the paper table.

DEFORM runs in Docker via deform_docker/deform_run_condition.sh on the
open_narrow_obs map (matching ORCA / LF / LAS).  Each docker session writes
one CSV to deform_docker/results/ when the agents reach the goal.  This
script reads every CSV, groups them by n_agents, averages each metric across
trials, and prints the same 5-metric paper table produced by sweep_runner.py
and las_sweep.py.

The bridge node now uses FullRunMetrics so the CSV columns already match the
paper table (avg_speed_mps / time_to_goal_s / n_completed); no translation
needed.

Workflow
--------
    # 1) Generate ≥ 20 trials per condition (each docker run = 1 trial):
    cd baselines/deform_docker
    for i in $(seq 1 20); do ./deform_run_condition.sh --agents 1; done
    for i in $(seq 1 20); do ./deform_run_condition.sh --agents 2; done
    for i in $(seq 1 20); do ./deform_run_condition.sh --agents 4; done

    # 2) Aggregate the resulting CSVs into the paper table:
    cd ..
    python deform_sweep.py
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from typing import Any

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_RESULTS_DIR = os.path.join(_HERE, "deform_docker", "results")

_MIN_TRIALS = 20    # paper table requires ≥ 20 trials per condition

_PAPER_COLS = [
    "baseline", "n_agents", "target_speed_mps", "avg_speed_mps",
    "success", "time_to_goal_s", "flow_rate", "deformability",
]
_DIAG_COLS = [
    "n_completed", "collision", "spread_at_entry_m", "spread_at_narrow_m",
]

# Legacy NarrowZoneMetrics column names → new FullRunMetrics names.
# Kept so older CSVs in results/ still aggregate; new CSVs from the updated
# bridge already use the new names.
_LEGACY_RENAME = {
    "mean_speed_mps":  "avg_speed_mps",
    "time_to_clear_s": "time_to_goal_s",
    "n_cleared":       "n_completed",
}


def _coerce(val: str) -> float:
    s = val.strip()
    if s == "" or s.lower() == "nan":
        return float("nan")
    if s.lower() in ("inf", "infinity", "+inf"):
        return float("inf")
    if s.lower() == "-inf":
        return float("-inf")
    if s.lower() == "true":
        return 1.0
    if s.lower() == "false":
        return 0.0
    try:
        return float(s)
    except ValueError:
        return float("nan")


def _read_trials(results_dir: str) -> dict[int, list[dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)

    if not os.path.isdir(results_dir):
        print(f"[deform] Results dir not found: {results_dir}")
        return grouped

    csv_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".csv"))
    if not csv_files:
        print(f"[deform] No CSV files in {results_dir}")
        return grouped

    for fname in csv_files:
        path = os.path.join(results_dir, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                try:
                    n = int(float(rec.get("n_agents", "0")))
                except (TypeError, ValueError):
                    continue

                paper_row: dict[str, float] = {}
                for k, v in rec.items():
                    name = _LEGACY_RENAME.get(k, k)
                    if name in _PAPER_COLS + _DIAG_COLS or name == "n_agents":
                        paper_row[name] = _coerce(v)
                grouped[n].append(paper_row)
                print(f"[deform] loaded {fname}  n_agents={n}")

    return grouped


def _aggregate(rows: list[dict[str, float]]) -> dict[str, Any]:
    def _mean(key: str) -> float:
        vals = [r.get(key, float("nan")) for r in rows]
        finite = [
            v for v in vals
            if not (isinstance(v, float) and math.isnan(v))
            and v != float("inf")
        ]
        return float(np.mean(finite)) if finite else float("nan")

    times = [r.get("time_to_goal_s", float("nan")) for r in rows]
    finite_times = [t for t in times if not math.isnan(t) and t != float("inf")]
    if finite_times:
        time_to_goal = float(np.mean(finite_times))
    elif any(t == float("inf") for t in times):
        time_to_goal = float("inf")
    else:
        time_to_goal = float("nan")

    return {
        "avg_speed_mps":      round(_mean("avg_speed_mps"),       3),
        "success":            round(_mean("success"),              3),
        "time_to_goal_s":     (round(time_to_goal, 3)
                               if time_to_goal not in (float("inf"), float("nan"))
                               else time_to_goal),
        "flow_rate":          round(_mean("flow_rate"),            4),
        "deformability":      round(_mean("deformability"),        3),
        "n_completed":        round(_mean("n_completed"),          2),
        "collision":          round(_mean("collision"),            2),
        "spread_at_entry_m":  round(_mean("spread_at_entry_m"),    3),
        "spread_at_narrow_m": round(_mean("spread_at_narrow_m"),   3),
    }


def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        print("\n[deform] No rows to display.")
        return
    header = (
        f"{'baseline':<10} {'n_agents':>8} {'trials':>7} "
        f"{'avg_speed':>9} {'success':>8} {'t_goal':>9} "
        f"{'flow':>8} {'deform':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r in rows:
        t   = r["time_to_goal_s"]
        if isinstance(t, float) and math.isnan(t):
            t_s = "     n/a"
        elif t == float("inf"):
            t_s = "     inf"
        else:
            t_s = f"{t:9.3f}"

        d = r["deformability"]
        d_s = f"{d:8.3f}" if not (isinstance(d, float) and math.isnan(d)) else "     n/a"

        s = r["avg_speed_mps"]
        s_s = f"{s:9.3f}" if not (isinstance(s, float) and math.isnan(s)) else "      n/a"

        warn = "  ⚠ <20 trials" if r["_n_trials"] < _MIN_TRIALS else ""
        print(
            f"{'deform':<10} {r['n_agents']:>8} {r['_n_trials']:>7} "
            f"{s_s} {r['success']:>8.3f} {t_s} "
            f"{r['flow_rate']:>8.4f} {d_s}{warn}"
        )
    print(f"{sep}")
    print(f"  paper table requires {_MIN_TRIALS} trials per n_agents\n")


def sweep(results_dir: str, output: str, min_trials: int) -> None:
    print(f"[deform] Reading results from: {results_dir}")
    print(f"[deform] Minimum trials per condition: {min_trials}")
    print("=" * 60)

    grouped = _read_trials(results_dir)
    if not grouped:
        print("\n[deform] Nothing to aggregate.  Generate trials first:")
        print("    cd baselines/deform_docker")
        print(f"    for i in $(seq 1 {min_trials}); do ./deform_run_condition.sh --agents N; done")
        return

    short_conditions = []
    for n in sorted(grouped.keys()):
        if len(grouped[n]) < min_trials:
            short_conditions.append((n, len(grouped[n])))

    if short_conditions:
        print("\n[deform] WARNING — the following conditions have fewer than "
              f"{min_trials} trials:")
        for n, k in short_conditions:
            print(f"    n_agents={n}: only {k} trial(s) — need {min_trials - k} more")
        print()

    all_rows: list[dict[str, Any]] = []
    for n in sorted(grouped.keys()):
        trials = grouped[n]
        agg = _aggregate(trials)
        row: dict[str, Any] = {
            "baseline":         "deform",
            "n_agents":         n,
            "target_speed_mps": float("nan"),
            **agg,
            "_n_trials":        len(trials),
        }
        all_rows.append(row)
        print(
            f"\n[deform] n_agents={n}  trials={len(trials)}: "
            f"success={agg['success']}  "
            f"avg_spd={agg['avg_speed_mps']}  "
            f"flow={agg['flow_rate']}  "
            f"deform={agg['deformability']}"
        )

    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PAPER_COLS + _DIAG_COLS)
        writer.writeheader()
        for r in all_rows:
            writer.writerow({k: r.get(k, "") for k in _PAPER_COLS + _DIAG_COLS})
    print(f"\n[deform] Results saved → {output}")

    print("\n========== DEFORM PAPER TABLE (5 metrics) ==========")
    _print_table(all_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Aggregate DEFORM result CSVs into the paper table"
    )
    parser.add_argument(
        "--results-dir", type=str, default=_DEFAULT_RESULTS_DIR,
        help=f"Directory containing zone_ep*.csv files "
             f"(default: {_DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--output", type=str, default="deform_sweep_results.csv",
        help="Output CSV path (default: deform_sweep_results.csv)",
    )
    parser.add_argument(
        "--min-trials", type=int, default=_MIN_TRIALS,
        help=f"Minimum trials per n_agents condition (default: {_MIN_TRIALS})",
    )
    args = parser.parse_args()

    sweep(
        results_dir=args.results_dir,
        output=args.output,
        min_trials=max(args.min_trials, _MIN_TRIALS),
    )
