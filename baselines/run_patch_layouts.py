#!/usr/bin/env python3
"""FastFunnels leader (patch) policy alone, Table-1 metrics on 100 layouts.

This is the system the rebuttal's 97/100 measures -- ``rebuttal.tex`` says so
directly: "the table and Figure R1 evaluate the *funnel (leader)* policy's
robustness to layout randomisation".  ``make_rebuttal_figure.py`` scores it as
zone clearance only; this adds the rest of the paper's Table-1 columns
(V_bar, T_clear, Phi, eSR, aSR) via the same ``FullRunMetrics`` the baselines
report through, so the "Ours" row sits on the same axis as ORCA and GCBF+.

The funnel is one car, so aSR is 1 exactly when the run clears, and the
lateral-spread deformability is undefined; the funnel's own deformability is
the patch-axis ratio, reported separately as ``patch_deform``.

Usage
-----
    /p/cral/vignesh/envs/fastfunnels/bin/python baselines/run_patch_layouts.py \\
        --split train:eval_maps_matched:50 --split heldout:eval_maps_heldout:50 \\
        --model patch_policy_models/run_20260809_013941/best_model \\
        --out baselines/results_layouts/patch_only.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import time

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "f1tenth_gym")):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


def patch_track_lookup(roots: list[str]) -> None:
    import f1tenth_gym.envs.track.track as _t
    import f1tenth_gym.envs.track.utils as _u

    original = _u.find_track_dir
    paths = [pathlib.Path(r) for r in roots]

    def find(name):
        for root in paths:
            if (root / name).exists():
                return root / name
        return original(name)

    _t.find_track_dir = find
    _u.find_track_dir = find


def rollout(name: str, split: str, root: str, predict, args) -> dict:
    from eval_metrics import FullRunMetrics

    import yaml

    from envs.ppo_policy import PatchCarEnv, PatchEnvConfig

    map_dir = os.path.join(root, name)
    with open(os.path.join(map_dir, f"{name}_obs_pos.yaml")) as f:
        meta = yaml.safe_load(f)
    region = meta["narrow_regions"][0]
    narrow_xy = (float(region["center_m"][0]), float(region["center_m"][1]))
    gap = float(region["gap_width_m"])

    cl = np.loadtxt(os.path.join(map_dir, f"{name}_centerline.csv"),
                    delimiter=",", comments="#")[:, :2]

    env = PatchCarEnv(PatchEnvConfig(
        num_agents=1, map_name=name, render_mode=None,
        domain_randomize=False, random_spawn=False, split_mode=True,
        max_steps=args.max_steps))
    obs, _ = env.reset(seed=args.seed)

    zone = FullRunMetrics(
        centerline_xs=cl[:, 0], centerline_ys=cl[:, 1], dt=args.dt,
        narrow_center_xy=narrow_xy, gap_width_m=gap,
        zone_half_width=args.zone_half_width,
        collision_thresh=args.collision_thresh,
    )

    def car_obs() -> dict | None:
        p = env.active_patches[0] if env.active_patches else None
        if p is None:
            return None
        b = getattr(env, "_current_base_obs", None)
        v = float(np.asarray(b["linear_vels_x"])[0]) if b else 0.0
        return {"poses_x": np.array([float(p.x)]),
                "poses_y": np.array([float(p.y)]),
                "poses_theta": np.array([float(p.theta)]),
                "linear_vels_x": np.array([v])}

    o = car_obs()
    if o is not None:
        zone.step(o)

    a_hist, b_hist = [], []
    steps, info, term, trunc = 0, {}, False, False
    for steps in range(1, args.max_steps + 1):
        obs, _r, term, trunc, info = env.step(
            predict(np.asarray(obs, np.float32)))
        p = env.active_patches[0]
        a_hist.append(float(p.a))
        b_hist.append(float(p.b))
        o = car_obs()
        if o is not None:
            zone.step(o)
        if zone.all_zone_cleared or term or trunc:
            break

    reason = info.get("termination_reason", "max_steps")
    prog = float(info.get("lap_progress", 0.0))
    s = zone.summary()
    env.close()

    # The funnel's own deformability: geometric mean of the semi-axis ranges,
    # matching the patch-shape definition in the repo-root eval_metrics.
    if a_hist and min(a_hist) > 1e-6 and min(b_hist) > 1e-6:
        patch_deform = float(np.sqrt((max(a_hist) / min(a_hist)) *
                                     (max(b_hist) / min(b_hist))))
    else:
        patch_deform = float("nan")

    cleared = bool(s["success"] >= 1.0)
    return {
        "split": split, "name": name, "n_agents": 1,
        "cleared": cleared,
        "n_cleared": int(s["n_completed"]),
        "collided": bool(s["collision"]),
        "steps": steps, "sim_time_s": steps * args.dt,
        "lap_progress": prog, "reason": str(reason),
        "avg_speed_mps": float(s["avg_speed_mps"]),
        "time_to_goal_s": float(s["time_to_goal_s"]),
        "flow_rate": float(s["flow_rate"]),
        "safety_rate": float(s["safety_rate"]),
        "patch_deform": patch_deform,
        "patch_b_min": float(min(b_hist)) if b_hist else float("nan"),
        "patch_b_max": float(max(b_hist)) if b_hist else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zone-half-width", type=int, default=8)
    ap.add_argument("--collision-thresh", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits = []
    for spec in args.split:
        parts = spec.split(":")
        label, root = parts[0], parts[1]
        limit = int(parts[2]) if len(parts) > 2 else None
        names = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d))
                       and os.path.exists(os.path.join(root, d, f"{d}_map.yaml")))
        splits.append((label, root, names[:limit] if limit else names))
    patch_track_lookup([r for _, r, _ in splits])

    from mass_eval import load_policy

    predict, _ = load_policy(args.model)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows, t0 = [], time.time()
    with open(args.out, "w") as fh:
        for label, root, names in splits:
            for i, nm in enumerate(names):
                try:
                    r = rollout(nm, label, root, predict, args)
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    msg = (f"cleared={r['cleared']} prog={r['lap_progress']:.1%} "
                           f"speed={r['avg_speed_mps']:.2f} {r['reason']}")
                except Exception as exc:
                    msg = f"ERROR {type(exc).__name__}: {exc}"
                print(f"  [{label} {i + 1}/{len(names)}] {nm}: {msg}", flush=True)

    print(f"\n[patch-only] {len(rows)} layouts in {time.time() - t0:.1f}s")
    for label, _, _ in splits:
        s = [r for r in rows if r["split"] == label]
        if s:
            k = sum(r["cleared"] for r in s)
            print(f"  {label}: {k}/{len(s)} = {100 * k / len(s):.1f}%")


if __name__ == "__main__":
    main()
