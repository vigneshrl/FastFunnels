#!/usr/bin/env python3
"""
las_sweep.py — LAS (CBF-PPO) checkpoint evaluator that mirrors sweep_runner.py.

The LAS checkpoint runs on open_narrow_obs (same map as ORCA / LF), so the
5 paper metrics are directly comparable.

n_agents is fixed at 3 (1 ego + 2 NPC opponents) because the CBF safety
dimension is baked into the trained weights.

The "speed axis" sweeps NPC vgain:  0.5 = slow, 0.6 = normal, 0.7 = fast.

Usage (run from baselines/):
    python las_sweep.py --n-episodes 10 --no-render
    python las_sweep.py --n-episodes 10 --vgains 0.5,0.6,0.7 --no-render
    python las_sweep.py --n-episodes 5  --output results/las.csv --no-render
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np
import yaml

# ── sys.path: LAS modules must come before baselines so gym_envs is found ─────
_HERE = os.path.dirname(os.path.abspath(__file__))
_LAS  = os.path.join(_HERE, "learning_adaptive_safety")
for _p in [_LAS, _HERE]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── LAS imports ────────────────────────────────────────────────────────────────
import torch
from evaluation.grid_param_parser import load_checkpoint

# ── Baselines imports ──────────────────────────────────────────────────────────
from eval_metrics import FullRunMetrics

# ── Constants ─────────────────────────────────────────────────────────────────
_LAS_PATH   = pathlib.Path(_LAS)
_CHECKPOINT = (
    _LAS_PATH / "checkpoints"
    / "PPOPID-{f110-multi-agent-v1}"
    / "checkpoints" / "model_final.pt"
)
_ENV_ID   = "f110-multi-agent-v1"   # resolved to auto-cbf-v1 inside load_checkpoint
_N_NPC    = 2                        # fixed by CBF safety_dim in checkpoint
_N_AGENTS = _N_NPC + 1              # 1 ego + 2 NPC
_DT       = 0.1                      # planning step duration (control_freq=2, planning_freq=10)
_MIN_EPS  = 20                       # paper table requires ≥ 20 trials per condition

_VGAIN_LABELS = {0.5: "slow", 0.6: "normal", 0.7: "fast"}

# Paper table columns (identical to sweep_runner.py)
_PAPER_COLS = [
    "baseline", "n_agents", "target_speed_mps", "avg_speed_mps",
    "success", "time_to_goal_s", "flow_rate", "deformability",
]
_DIAG_COLS = [
    "n_completed", "n_collided", "safety_rate", "collision",
    "spread_at_entry_m", "spread_at_narrow_m",
]


# ---------------------------------------------------------------------------
# Narrow-zone geometry helpers
# ---------------------------------------------------------------------------

def _load_narrow_region(map_name: str = "open_narrow_obs"):
    """Return (narrow_center_xy, gap_width_m) from the map's obs_pos.yaml."""
    try:
        from f1tenth_gym.envs.track.utils import find_track_dir
        yaml_path = find_track_dir(map_name) / f"{map_name}_obs_pos.yaml"
        with open(yaml_path) as f:
            data = yaml.safe_load(f)
        nr  = data["narrow_regions"][0]
        return tuple(nr["center_m"]), float(nr["gap_width_m"])
    except Exception as exc:
        print(f"[LAS] Warning: could not load obs_pos.yaml ({exc})")
        return None, None


# ---------------------------------------------------------------------------
# State → obs dict for FullRunMetrics
# ---------------------------------------------------------------------------

def _state_to_obs(state: dict) -> dict:
    """Convert env.state (multi-agent dict) to FullRunMetrics obs format."""
    ids = list(state.keys())
    return {
        "poses_x":       np.array([state[a]["pose"][0]     for a in ids]),
        "poses_y":       np.array([state[a]["pose"][1]     for a in ids]),
        "linear_vels_x": np.array([state[a]["velocity"][0] for a in ids]),
    }


# ---------------------------------------------------------------------------
# Single episode runner
# ---------------------------------------------------------------------------

def _run_episode(
    agent_fn,
    env,
    seed: int,
    xs: np.ndarray,
    ys: np.ndarray,
    narrow_xy,
    gap_w: float | None,
    zone_half_w: int,
) -> dict:
    """Run one episode; return FullRunMetrics.summary()."""

    # Spawn agents at the narrow zone entry so every episode exercises the passage.
    spawn_obs = None
    if narrow_xy is not None and len(xs) > 0:
        _narrow_wp = int(np.argmin(np.hypot(xs - narrow_xy[0], ys - narrow_xy[1])))
        _entry_wp  = max(0, _narrow_wp - zone_half_w)
        _nxt       = min(_entry_wp + 1, len(xs) - 1)
        _theta     = math.atan2(float(ys[_nxt] - ys[_entry_wp]),
                                float(xs[_nxt] - xs[_entry_wp]))
        _back_x    = -math.cos(_theta)
        _back_y    = -math.sin(_theta)
        _gap       = 1.5   # m between consecutive agents
        _spawn_poses = np.array(
            [[float(xs[_entry_wp]) + i * _gap * _back_x,
              float(ys[_entry_wp]) + i * _gap * _back_y,
              _theta]
             for i in range(_N_AGENTS)],
            dtype=np.float32,
        )
        try:
            spawn_obs, _ = env.reset(seed=seed, options={"poses": _spawn_poses})
        except Exception:
            spawn_obs = None

    obs = spawn_obs if spawn_obs is not None else env.reset(seed=seed)[0]

    fm = FullRunMetrics(
        centerline_xs=xs,
        centerline_ys=ys,
        dt=_DT,
        narrow_center_xy=narrow_xy,
        gap_width_m=gap_w,
        zone_half_width=zone_half_w,
    )

    # Pre-initialise waypoint hints from actual spawn positions.
    # The LAS env resets agents at random positions (0–40 % of track), so
    # the default hint of 0 would confine the ±window search to the wrong
    # part of the 2 258-waypoint track.
    if hasattr(env, "state") and len(xs) > 0:
        init_state = env.state
        ids = list(init_state.keys())
        fm._init_agents(len(ids))
        for i, aid in enumerate(ids):
            xi = float(init_state[aid]["pose"][0])
            yi = float(init_state[aid]["pose"][1])
            fm._wp_hints[i] = int(np.argmin(np.hypot(xs - xi, ys - yi)))

    done = False
    while not done:
        action = agent_fn(obs, deterministic=True)
        results = env.step(action)
        if len(results) == 5:
            obs, _, done, truncated, _ = results
        else:
            obs, _, _, done, truncated, _ = results
        done = bool(done) or bool(truncated)

        if hasattr(env, "state"):
            fm.step(_state_to_obs(env.state))

        if fm.all_done or fm.all_zone_cleared:
            break

    return fm.summary()


# ---------------------------------------------------------------------------
# Table printing (mirrors sweep_runner._print_table)
# ---------------------------------------------------------------------------

def _print_table(rows: list[dict[str, Any]], vgains: list[float]) -> None:
    if not rows:
        return
    header = (
        f"{'baseline':<18} {'n_agents':>8} {'vgain':>7} "
        f"{'avg_speed':>9} {'success':>8} {'t_goal':>9} "
        f"{'flow':>8} {'deform':>8}"
    )
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for r, vg in zip(rows, vgains):
        t   = r["time_to_goal_s"]
        t_s = f"{t:9.3f}" if t != float("inf") else "     inf"
        d   = r["deformability"]
        d_s = f"{d:8.3f}" if not math.isnan(d) else "     n/a"
        print(
            f"{'las':<18} {r['n_agents']:>8} {vg:>7.2f} "
            f"{r['avg_speed_mps']:>9.3f} {r['success']:>8.3f} {t_s} "
            f"{r['flow_rate']:>8.4f} {d_s}"
        )
    print(f"{sep}")
    print("  n_agents=3 fixed (1 ego + 2 NPC); vgain = NPC speed gain\n")


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def sweep(
    vgains: list[float],
    n_episodes: int,
    render: bool,
    output: str,
) -> None:
    render_mode = "human" if render else None

    # ── Load agent + env once (default vgain=0.7) ─────────────────────────
    print(f"[LAS] Loading checkpoint: {_CHECKPOINT}")
    agent_fn, env = load_checkpoint(
        env_id=_ENV_ID,
        checkpoint=str(_CHECKPOINT),
        render_mode=render_mode,
    )
    print("[LAS] Checkpoint loaded.")

    # ── Get track centerline for FullRunMetrics ────────────────────────────
    # gymnasium wrappers expose unwrapped attributes via env.unwrapped or
    # env.get_wrapper_attr(); the manual `while hasattr(raw, "env")` loop
    # doesn't reach the base env correctly through all wrapper types.
    track = None
    for _getter in [
        lambda e: e.get_wrapper_attr("track"),
        lambda e: e.unwrapped.track,
        lambda e: getattr(e, "track", None),
    ]:
        try:
            track = _getter(env)
            if track is not None:
                break
        except Exception:
            pass

    if track is not None:
        xs = np.asarray(track.centerline.xs, dtype=float)
        ys = np.asarray(track.centerline.ys, dtype=float)
        print(f"[LAS] Track loaded: {len(xs)} waypoints")
    else:
        xs = ys = np.array([])
        print("[LAS] Warning: track not found — FullRunMetrics degraded (no speed/goal tracking)")

    # zone_half_width: 4 m each side (matching the physical half-width that
    # ORCA/LF's default 8 wps gives at ~0.5 m/wp spacing).
    avg_wp_spacing = (
        float(np.mean(np.hypot(np.diff(xs), np.diff(ys)))) if len(xs) > 1 else 1.0
    )
    zone_half_w = max(8, int(4.0 / max(avg_wp_spacing, 1e-3)))
    print(f"[LAS] zone_half_width = {zone_half_w} wps "
          f"(~{zone_half_w * avg_wp_spacing:.1f} m each side of narrow centre)")

    narrow_xy, gap_w = _load_narrow_region()
    if narrow_xy:
        print(f"[LAS] Narrow region: center={narrow_xy}, gap={gap_w:.3f} m")

    all_rows: list[dict[str, Any]] = []
    total = len(vgains)

    for vi, vg in enumerate(vgains):
        label = _VGAIN_LABELS.get(vg, f"{vg:.2f}")
        print(
            f"\n[LAS {vi+1}/{total}]  vgain={vg} ({label})  "
            f"n_episodes={n_episodes}"
        )
        print("=" * 60)

        ep_results: list[dict[str, Any]] = []

        for ep in range(n_episodes):
            # Set vgain for this episode
            if hasattr(env, "set_options"):
                env.set_options(options={"vgain": vg})

            t0 = time.time()
            result = _run_episode(
                agent_fn, env, seed=42 + ep,
                xs=xs, ys=ys,
                narrow_xy=narrow_xy, gap_w=gap_w,
                zone_half_w=zone_half_w,
            )
            ep_results.append(result)

            t_str = (
                f"{result['time_to_goal_s']:.2f}s"
                if result["time_to_goal_s"] != float("inf")
                else "inf"
            )
            print(
                f"  ep {ep+1:2d}: success={result['success']:.0f}  "
                f"spd={result['avg_speed_mps']:.2f} m/s  "
                f"t_goal={t_str}  "
                f"flow={result['flow_rate']:.4f}  "
                f"elapsed={time.time()-t0:.1f}s"
            )

        # ── Aggregate across episodes ──────────────────────────────────────
        def _agg(key: str) -> float:
            vals = [m[key] for m in ep_results]
            finite = [v for v in vals if not (isinstance(v, float) and math.isnan(v))
                      and v != float("inf")]
            return float(np.mean(finite)) if finite else float("nan")

        row: dict[str, Any] = {
            "baseline":           "las",
            "n_agents":           _N_AGENTS,
            "target_speed_mps":   float(vg),   # NPC vgain used as speed axis
            "avg_speed_mps":      round(_agg("avg_speed_mps"),    3),
            "success":            round(_agg("success"),           3),
            "time_to_goal_s":     round(_agg("time_to_goal_s"),   3),
            "flow_rate":          round(_agg("flow_rate"),         4),
            "deformability":      (
                round(_agg("deformability"), 3)
                if not math.isnan(_agg("deformability"))
                else float("nan")
            ),
            "n_completed":        round(_agg("n_completed"),       2),
            "n_collided":         round(_agg("n_collided"),        2),
            "safety_rate":        round(_agg("safety_rate"),       3),
            "collision":          round(_agg("collision"),         2),
            "spread_at_entry_m":  round(_agg("spread_at_entry_m"),3),
            "spread_at_narrow_m": round(_agg("spread_at_narrow_m"),3),
        }
        all_rows.append(row)
        print(
            f"\n[LAS] vgain={vg} ({n_episodes} eps): "
            f"success={row['success']:.3f}  "
            f"avg_spd={row['avg_speed_mps']:.3f}  "
            f"flow={row['flow_rate']:.4f}  "
            f"deform={row['deformability']}"
        )

    env.close()

    # ── Save CSV ───────────────────────────────────────────────────────────
    out_dir = os.path.dirname(os.path.abspath(output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=_PAPER_COLS + _DIAG_COLS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\n[LAS] Results saved → {output}")

    # ── Print table ────────────────────────────────────────────────────────
    print("\n========== LAS PAPER TABLE (5 metrics) ==========")
    _print_table(all_rows, vgains)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LAS checkpoint sweep on open_narrow_obs"
    )
    parser.add_argument(
        "--vgains", type=str, default="0.5,0.6,0.7",
        help="Comma-separated NPC vgain values (default: 0.5,0.6,0.7 → slow/normal/fast)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=_MIN_EPS,
        help=f"Episodes per vgain setting (default & minimum: {_MIN_EPS})",
    )
    parser.add_argument(
        "--no-render", action="store_true",
        help="Disable rendering (recommended for sweeps)",
    )
    parser.add_argument(
        "--output", type=str, default="las_sweep_results.csv",
        help="Output CSV path (default: las_sweep_results.csv)",
    )
    args = parser.parse_args()

    vgains = [float(v.strip()) for v in args.vgains.split(",")]

    n_eps = args.n_episodes
    if n_eps < _MIN_EPS:
        print(f"[LAS] n-episodes={n_eps} is below the paper-table minimum "
              f"of {_MIN_EPS} — clamping up.")
        n_eps = _MIN_EPS

    sweep(
        vgains=vgains,
        n_episodes=n_eps,
        render=not args.no_render,
        output=args.output,
    )
