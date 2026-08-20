#!/usr/bin/env python3
"""
Paper-table sweep runner.

Sweeps over (n_agents × speed_tier) for one or more baselines and writes
the 5 narrow-zone metrics to a CSV + prints a formatted table.

Usage
-----
    # Simulated baselines (run directly)
    python sweep_runner.py --baseline orca                       --no-render
    python sweep_runner.py --baseline leader_follower            --no-render
    python sweep_runner.py --baseline all                        --no-render

    # DEFORM (reads CSVs written by the bridge node after manual runs)
    #   1. Run each condition:  ./deform_docker/deform_run_condition.sh --agents 2
    #                           ./deform_docker/deform_run_condition.sh --agents 4
    #   2. Collect table:
    python sweep_runner.py --baseline deform

    # All baselines in one table (DEFORM results must already exist)
    python sweep_runner.py --baseline all_including_deform --no-render

    # Custom conditions / output
    python sweep_runner.py --baseline orca --speeds 1.5,4.0,8.0 --agents 1,2,4 --no-render
    python sweep_runner.py --baseline all  --no-render --output results/paper_table.csv

Learning Adaptive Safety (LAS) note
-------------------------------------
    LAS is a CBF-PPO racing baseline trained on the General1 circuit track
    (f110-multi-agent-v0).  It cannot be evaluated on open_narrow_obs /
    full_narrow and does not accept n_agents, so it is excluded from this
    sweep.  Evaluate it separately using the scripts in
    baselines/learning_adaptive_safety/.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Any

# ── Make sure we can import sibling modules (eval_metrics, orca, leader_follower)
_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# ---------------------------------------------------------------------------
# Paper-table constants
# ---------------------------------------------------------------------------

_MIN_TRIALS = 20   # paper table requires ≥ 20 trials per condition


# ---------------------------------------------------------------------------
# Speed-tier label helper
# ---------------------------------------------------------------------------

def _speed_label(speed: float) -> str:
    if speed < 3.5:
        return "slow"
    if speed < 6.5:
        return "normal"
    return "fast"


# ---------------------------------------------------------------------------
# Trial averaging
# ---------------------------------------------------------------------------

def _average_trial_metrics(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean across trial metric dicts, ignoring NaN.  Time-to-goal preserves
    inf when every trial timed out."""
    import math
    import numpy as _np

    def _mean(key: str) -> float:
        vals = [t.get(key, float("nan")) for t in trials]
        finite = [
            v for v in vals
            if isinstance(v, (int, float))
            and not (isinstance(v, float) and math.isnan(v))
            and v != float("inf")
        ]
        return float(_np.mean(finite)) if finite else float("nan")

    times = [t.get("time_to_goal_s", float("nan")) for t in trials]
    finite_times = [t for t in times
                    if isinstance(t, (int, float))
                    and not (isinstance(t, float) and math.isnan(t))
                    and t != float("inf")]
    if finite_times:
        time_to_goal = float(_np.mean(finite_times))
    elif any(t == float("inf") for t in times):
        time_to_goal = float("inf")
    else:
        time_to_goal = float("nan")

    collisions = [t.get("collision", False) for t in trials]
    coll_mean  = float(_np.mean([1.0 if bool(c) else 0.0 for c in collisions])) \
                 if collisions else float("nan")

    return {
        "avg_speed_mps":      _mean("avg_speed_mps"),
        "success":            _mean("success"),
        "time_to_goal_s":     time_to_goal,
        "flow_rate":          _mean("flow_rate"),
        "deformability":      _mean("deformability"),
        "n_completed":        _mean("n_completed"),
        "n_collided":         _mean("n_collided"),
        "safety_rate":        _mean("safety_rate"),
        "collision":          coll_mean,
        "spread_at_entry_m":  _mean("spread_at_entry_m"),
        "spread_at_narrow_m": _mean("spread_at_narrow_m"),
    }


# ---------------------------------------------------------------------------
# Per-baseline run functions
# ---------------------------------------------------------------------------

def _run_orca(n_agents: int, speed: float, map_name: str,
              max_steps: int, render: bool, seed: int = 42) -> dict[str, Any]:
    from orca import OrcaConfig, F1TenthORCARunner
    cfg = OrcaConfig(
        num_cars=n_agents,
        map_name=map_name,
        target_speed=speed,
        max_steps=max_steps,
        render=render,
        seed=seed,
    )
    return F1TenthORCARunner(cfg).run()


def _run_leader_follower(n_agents: int, speed: float, map_name: str,
                         max_steps: int, render: bool, seed: int = 42) -> dict[str, Any]:
    from leader_follower import LeaderFollowerConfig, F1TenthLeaderFollower
    cfg = LeaderFollowerConfig(
        num_cars=n_agents,
        map_name=map_name,
        leader_speed=speed,
        max_steps=max_steps,
        render=render,
        seed=seed,
    )
    return F1TenthLeaderFollower(cfg).run()




# NOTE: learning_adaptive_safety is excluded — it is a racing baseline that runs
# on its own circuit map (not full_narrow) and ignores the n_agents parameter.
# Its zone metrics are not comparable to ORCA / LF / DEFORM / Ours.
_BASELINE_FNS = {
    "orca":          _run_orca,
    "leader_follower": _run_leader_follower,
}

# ---------------------------------------------------------------------------
# DEFORM: read pre-existing CSVs written by the bridge node
# ---------------------------------------------------------------------------

_DEFORM_RESULTS_DIR = os.path.join(
    os.path.dirname(__file__), "deform_docker", "results"
)


def _read_deform_results(results_dir: str) -> list[dict[str, Any]]:
    """
    Read all zone_ep*.csv files written by f1tenth_bridge_node and
    return them as rows in the same format as _extract_row().
    """
    rows: list[dict[str, Any]] = []
    if not os.path.isdir(results_dir):
        print(f"[deform] results dir not found: {results_dir}")
        print("[deform] Run ./deform_docker/deform_run_condition.sh --agents N first.")
        return rows

    csv_files = sorted(f for f in os.listdir(results_dir) if f.endswith(".csv"))
    if not csv_files:
        print(f"[deform] No CSV files found in {results_dir}")
        return rows

    for fname in csv_files:
        path = os.path.join(results_dir, fname)
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for rec in reader:
                n = int(rec.get("n_agents", 0))
                spd = float(rec.get("avg_speed_mps", float("nan")))
                row: dict[str, Any] = {
                    "baseline":          "deform",
                    "n_agents":          n,
                    "target_speed_mps":  float("nan"),   # DEFORM sets its own speed
                    "avg_speed_mps":     spd,
                    "success":           float(rec.get("success",         float("nan"))),
                    "time_to_goal_s":    float(rec.get("time_to_goal_s",  float("nan"))),
                    "flow_rate":         float(rec.get("flow_rate",       float("nan"))),
                    "deformability":     float(rec.get("deformability",   float("nan"))),
                    "n_completed":       rec.get("n_completed",       ""),
                    "collision":         rec.get("collision",          ""),
                    "spread_at_entry_m": rec.get("spread_at_entry_m", ""),
                    "spread_at_narrow_m":rec.get("spread_at_narrow_m",""),
                }
                rows.append(row)
                print(f"[deform] loaded {fname}  n_agents={n}")

    return rows

# ---------------------------------------------------------------------------
# Paper-metric columns (the 5 we care about + identification columns)
# ---------------------------------------------------------------------------

_PAPER_COLS = [
    "baseline",
    "n_agents",
    "target_speed_mps",
    "avg_speed_mps",
    "success",
    "time_to_goal_s",
    "flow_rate",
    "deformability",
]

_DIAG_COLS = [
    "n_completed",
    "n_collided",
    "safety_rate",
    "collision",
    "spread_at_entry_m",
    "spread_at_narrow_m",
]


def _extract_row(baseline: str, n_agents: int, speed: float,
                 metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {
        "baseline":          baseline,
        "n_agents":          n_agents,
        "target_speed_mps":  speed,
        "avg_speed_mps":     metrics.get("avg_speed_mps",   float("nan")),
        "success":           metrics.get("success",          float("nan")),
        "time_to_goal_s":    metrics.get("time_to_goal_s",  float("nan")),
        "flow_rate":         metrics.get("flow_rate",        float("nan")),
        "deformability":     metrics.get("deformability",    float("nan")),
        "n_completed":       metrics.get("n_completed",      float("nan")),
        "n_collided":        metrics.get("n_collided",       float("nan")),
        "safety_rate":       metrics.get("safety_rate",      float("nan")),
        "collision":         metrics.get("collision",        float("nan")),
        "spread_at_entry_m": metrics.get("spread_at_entry_m", float("nan")),
        "spread_at_narrow_m":metrics.get("spread_at_narrow_m", float("nan")),
    }
    return row


# ---------------------------------------------------------------------------
# Pretty-print the paper table
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return

    # Header
    header = (
        f"{'baseline':<18} {'n_agents':>8} {'tgt_spd':>8} "
        f"{'avg_speed':>9} {'success':>8} {'t_goal':>9} "
        f"{'flow':>8} {'deform':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)

    for r in rows:
        t   = r["time_to_goal_s"]
        t_str = f"{t:9.3f}" if t != float("inf") else "     inf"
        tgt = r["target_speed_mps"]
        tgt_str = f"{tgt:8.2f}" if tgt == tgt else "     n/a"  # nan check
        d   = r["deformability"]
        d_str = f"{d:8.3f}" if d == d else "     n/a"           # nan check
        print(
            f"{r['baseline']:<18} {r['n_agents']:>8} {tgt_str} "
            f"{r['avg_speed_mps']:>9.3f} {r['success']:>8.1f} {t_str} "
            f"{r['flow_rate']:>8.4f} {d_str}"
        )

    print(f"{sep}\n")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(
    baselines: list[str],
    agents: list[int],
    speeds: list[float],
    map_name: str,
    max_steps: int,
    render: bool,
    output: str,
    trials: int = _MIN_TRIALS,
    deform_results_dir: str = _DEFORM_RESULTS_DIR,
) -> None:
    all_rows: list[dict[str, Any]] = []

    # ── Speed-sweep baselines (ORCA, LF) ────────────────────────────────────
    speed_baselines = [b for b in baselines if b in ("orca", "leader_follower")]
    total = len(speed_baselines) * len(agents) * len(speeds)
    done  = 0

    for baseline in speed_baselines:
        run_fn = _BASELINE_FNS[baseline]
        for n_agents in agents:
            for speed in speeds:
                done += 1
                label = _speed_label(speed)
                print(
                    f"\n[sweep {done}/{total}] "
                    f"baseline={baseline}  n_agents={n_agents}  "
                    f"speed={speed} m/s ({label})  trials={trials}"
                )
                print("=" * 60)

                trial_metrics: list[dict[str, Any]] = []
                for trial_i in range(trials):
                    seed = 42 + trial_i
                    print(f"  [trial {trial_i+1}/{trials}]  seed={seed}")
                    try:
                        m = run_fn(n_agents, speed, map_name,
                                   max_steps, render, seed)
                        trial_metrics.append(m)
                    except Exception as exc:
                        print(f"  [trial {trial_i+1}] ERROR: {exc}")

                if trial_metrics:
                    averaged = _average_trial_metrics(trial_metrics)
                    row = _extract_row(baseline, n_agents, speed, averaged)
                else:
                    row = _extract_row(baseline, n_agents, speed, {})
                all_rows.append(row)

    # ── DEFORM: read pre-existing result CSVs ───────────────────────────────
    if "deform" in baselines:
        print("\n[sweep] Reading DEFORM results from", deform_results_dir)
        print("=" * 60)
        all_rows.extend(_read_deform_results(deform_results_dir))

    # ── Save CSV ────────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PAPER_COLS + _DIAG_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"[sweep] Results saved → {output}")

    # ── Print paper table ───────────────────────────────────────────────────
    print("\n========== PAPER TABLE (5 narrow-zone metrics) ==========")
    _print_table(all_rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Sweep baselines over (n_agents × speed) and collect paper metrics"
    )
    parser.add_argument(
        "--baseline", required=True,
        choices=[
            "orca", "leader_follower",
            "deform", "all", "all_including_deform",
        ],
        help=(
            "'orca' / 'leader_follower': run inline on the chosen map.  "
            "'deform': read existing CSVs from deform_docker/results/.  "
            "'all': run orca + leader_follower.  "
            "'all_including_deform': orca + leader_follower + deform CSVs."
        ),
    )
    parser.add_argument(
        "--deform-results", type=str, default=None,
        help="Path to DEFORM results directory (default: deform_docker/results/)",
    )
    parser.add_argument(
        "--map", type=str, default="open_narrow_obs",
        help="Map name passed to the gym (default: open_narrow_obs)",
    )
    parser.add_argument(
        "--speeds", type=str, default="2.0,5.0,8.0",
        help="Comma-separated target speeds in m/s (default: 2.0,5.0,8.0 → slow/normal/fast)",
    )
    parser.add_argument(
        "--agents", type=str, default="1,2,4",
        help="Comma-separated agent counts (default: 1,2,4)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=100_000,
        help="Max simulator steps per run (default: 100000)",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable rendering (recommended for sweep runs)",
    )
    parser.add_argument(
        "--output", type=str, default="sweep_results.csv",
        help="Path for the output CSV (default: sweep_results.csv)",
    )
    parser.add_argument(
        "--trials", type=int, default=_MIN_TRIALS,
        help=f"Trials per (baseline × n_agents × speed) condition "
             f"(default & minimum: {_MIN_TRIALS}).  Each trial uses a "
             f"different seed; metrics are averaged across all trials.",
    )
    args = parser.parse_args()

    if args.baseline == "all":
        baselines = list(_BASELINE_FNS.keys())
    elif args.baseline == "all_including_deform":
        baselines = list(_BASELINE_FNS.keys()) + ["deform"]
    else:
        baselines = [args.baseline]

    speeds = [float(s.strip()) for s in args.speeds.split(",")]
    agents = [int(a.strip())   for a in args.agents.split(",")]
    deform_dir = args.deform_results or _DEFORM_RESULTS_DIR

    trials = args.trials
    if trials < _MIN_TRIALS:
        print(f"[sweep] --trials={trials} is below the paper-table minimum "
              f"of {_MIN_TRIALS} — clamping up.")
        trials = _MIN_TRIALS

    sweep(
        baselines=baselines,
        agents=agents,
        speeds=speeds,
        map_name=args.map,
        max_steps=args.max_steps,
        render=not args.no_render,
        output=args.output,
        trials=trials,
        deform_results_dir=deform_dir,
    )
