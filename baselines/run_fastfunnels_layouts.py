#!/usr/bin/env python3
"""FastFunnels (ours) on the randomised narrow-zone layouts, Table-1 metrics.

Runs the frozen leader (patch) policy together with the trained follower policy
over ``eval_maps_matched`` / ``eval_maps_heldout``, scoring through the same
``baselines/eval_metrics.py::FullRunMetrics`` used for ORCA, GCBF+ and the
leader-follower baseline.  That makes the "Ours" row measured on the same 100
layouts as the baselines instead of quoted from the paper's original-corridor
cell.

Only the follower cars count as agents for eSR / aSR: ``JointEnv`` puts the
learning agents at f1tenth indices ``0..N-1`` and the patch car at index ``N``.

Usage
-----
    /p/cral/vignesh/envs/fastfunnels/bin/python baselines/run_fastfunnels_layouts.py \\
        --split train:eval_maps_matched:50 --split heldout:eval_maps_heldout:50 \\
        --patch-model patch_policy_models/run_20260809_013941/best_model \\
        --agent-model joint_sb3_models/run_20260810_000919/agents/best_model \\
        --n-agents 1 --out baselines/results_layouts/fastfunnels_n1.jsonl
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
# `baselines/` goes first so `eval_metrics` resolves to FullRunMetrics rather
# than the root's NarrowZoneMetrics.  The repo root is *appended*, not
# prepended: it contains a bare `f1tenth_gym/` directory that would otherwise
# shadow the installed package as a namespace package, and only the installed
# one can load these maps.
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


def rollout(name: str, split: str, predict_patch, predict_agent, args) -> dict:
    from eval_metrics import FullRunMetrics

    import yaml

    from envs.ppo_policy import JointEnv, JointEnvConfig

    map_dir = os.path.join(args._root_for[name], name)
    with open(os.path.join(map_dir, f"{name}_obs_pos.yaml")) as f:
        meta = yaml.safe_load(f)
    region = meta["narrow_regions"][0]
    narrow_xy = (float(region["center_m"][0]), float(region["center_m"][1]))
    gap = float(region["gap_width_m"])

    env = JointEnv(JointEnvConfig(
        num_agents=args.n_agents,
        map_name=name,
        render_mode=None,
        random_spawn=False,
        max_steps=args.max_steps,
        agent_use_lidar=args.agent_use_lidar,
        obs_mode=args.obs_mode,
    ))
    env._real_agents_active = True
    env.reset(seed=args.seed)

    cl = np.asarray(env._centerline_xy) if hasattr(env, "_centerline_xy") else None
    if cl is None:
        cl_path = os.path.join(map_dir, f"{name}_centerline.csv")
        arr = np.loadtxt(cl_path, delimiter=",", comments="#")
        cl = arr[:, :2]

    n = args.n_agents
    zone = FullRunMetrics(
        centerline_xs=cl[:, 0], centerline_ys=cl[:, 1],
        dt=args.dt, narrow_center_xy=narrow_xy, gap_width_m=gap,
        zone_half_width=args.zone_half_width,
        collision_thresh=args.collision_thresh,
    )

    def follower_obs() -> dict | None:
        b = env.current_base_obs
        if b is None:
            return None
        # Agents are f1tenth cars 0..n-1; car n is the patch car and is not
        # one of the "agents" the paper's aSR counts.
        return {"poses_x": np.asarray(b["poses_x"])[:n],
                "poses_y": np.asarray(b["poses_y"])[:n],
                "poses_theta": np.asarray(b["poses_theta"])[:n],
                "linear_vels_x": np.asarray(b["linear_vels_x"])[:n]}

    o = follower_obs()
    if o is not None:
        zone.step(o)

    collided = False
    steps = 0
    for steps in range(1, args.max_steps + 1):
        pa = np.asarray(predict_patch(env._step_obs[0]), dtype=np.float32)
        aa = np.stack([np.asarray(predict_agent(env._step_obs[1 + i]),
                                  dtype=np.float32) for i in range(n)])
        env._step_with_frozen_policy(pa, aa)

        o = follower_obs()
        if o is not None:
            zone.step(o)
        b = env.current_base_obs
        if b is not None and np.any(np.asarray(b.get("collisions", 0))[:n]):
            collided = True
            break
        if zone.all_zone_cleared:
            break
        if env._step_terminated or env._step_truncated:
            break

    s = zone.summary()
    env.close()
    return {
        "split": split, "name": name, "n_agents": n,
        "cleared": bool(s["success"] >= 1.0 and not collided),
        "n_cleared": int(s["n_completed"]),
        "collided": bool(collided or s["collision"]),
        "steps": steps, "sim_time_s": steps * args.dt,
        "avg_speed_mps": float(s["avg_speed_mps"]),
        "time_to_goal_s": float(s["time_to_goal_s"]),
        "flow_rate": float(s["flow_rate"]),
        "deformability": float(s["deformability"]),
        "safety_rate": float(s["safety_rate"]),
        "spread_at_entry_m": float(s["spread_at_entry_m"]),
        "spread_at_narrow_m": float(s["spread_at_narrow_m"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--patch-model", required=True)
    ap.add_argument("--agent-model", required=True)
    ap.add_argument("--n-agents", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--zone-half-width", type=int, default=8)
    ap.add_argument("--collision-thresh", type=float, default=0.5)
    ap.add_argument("--obs-mode", default="lidar")
    ap.add_argument("--agent-use-lidar", action="store_true", default=True)
    ap.add_argument("--no-agent-lidar", dest="agent_use_lidar",
                    action="store_false")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    splits, root_for = [], {}
    for spec in args.split:
        parts = spec.split(":")
        label, root = parts[0], parts[1]
        limit = int(parts[2]) if len(parts) > 2 else None
        names = sorted(d for d in os.listdir(root)
                       if os.path.isdir(os.path.join(root, d))
                       and os.path.exists(os.path.join(root, d, f"{d}_map.yaml")))
        names = names[:limit] if limit else names
        for nm in names:
            root_for[nm] = root
        splits.append((label, names))
    args._root_for = root_for
    patch_track_lookup(list({r for r in root_for.values()}))

    from mass_eval import load_policy

    predict_patch, _ = load_policy(args.patch_model)
    predict_agent, _ = load_policy(args.agent_model)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows, t0 = [], time.time()
    with open(args.out, "w") as fh:
        for label, names in splits:
            for i, nm in enumerate(names):
                try:
                    r = rollout(nm, label, predict_patch, predict_agent, args)
                    rows.append(r)
                    fh.write(json.dumps(r) + "\n")
                    fh.flush()
                    msg = (f"cleared={r['cleared']} ({r['n_cleared']}/{r['n_agents']}) "
                           f"speed={r['avg_speed_mps']:.2f}")
                except Exception as exc:
                    msg = f"ERROR {type(exc).__name__}: {exc}"
                print(f"  [{label} {i + 1}/{len(names)}] {nm}: {msg}", flush=True)

    print(f"\n[fastfunnels] {len(rows)} layouts in {time.time() - t0:.1f}s")
    for label, _ in splits:
        s = [r for r in rows if r["split"] == label]
        if s:
            k = sum(r["cleared"] for r in s)
            print(f"  {label}: {k}/{len(s)} = {100 * k / len(s):.1f}%")


if __name__ == "__main__":
    main()
