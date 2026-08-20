"""GCBF+ driving f1tenth_gym cars under the single-track model.

The controller is the same pretrained DubinsCar checkpoint used in
``run_gcbf_eval.py`` -- same graph observation, same lidar over the corridor
rectangles, same receding centreline carrot.  What changes is the vehicle: its
(omega, accel) command is converted to the (steering angle, speed) f1tenth_gym
expects, and the gym integrates the single-track model and owns collision.

Pair with ``run_orca_pyrobosim.py`` to separate controller from simulator.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from .corridor_geom import load_corridor, wall_rects
from .crossover_common import make_f110_env, patch_track_lookup
from .pyrobosim_world import build_world

GCBF_CAR_RADIUS = 0.05
WHEELBASE = 0.3302
YAW_RATE_GAIN = 20.0  # DubinsCar: theta_dot = 20 * action[0]


@dataclass
class Result:
    split: str
    name: str
    cleared: bool
    n_cleared: int
    n_agents: int
    collided: bool
    collision_step: int | None
    steps: int
    sim_time_s: float
    reason: str
    avg_speed_mps: float
    deformability: float
    flow_rate: float
    time_to_goal_s: float
    spread_at_entry_m: float
    spread_at_narrow_m: float


def rollout(map_dir: str, split: str, policy, args, n_rect: int) -> Result:
    from eval_metrics import FullRunMetrics

    import jax.numpy as jnp

    from .gcbf_corridor_env import Frame, build_obstacles

    geom = load_corridor(map_dir)
    window = geom.window(args.pre_wp, args.exit_offset, args.post_wp,
                         min_clearance=args.robot_radius + 0.10)
    # Reuse the PyRoboSim world purely for its spawn/goal formation geometry, so
    # GCBF+ starts from exactly the rank it starts from in its native run.
    cw = build_world(geom, window, args.n_agents, args.robot_radius, args.spacing)

    frame = Frame(scale=GCBF_CAR_RADIUS / args.robot_radius,
                  offset=np.asarray(geom.centerline[window.start_wp], dtype=float))
    obstacles = build_obstacles(
        wall_rects(geom, window, args.wall_thickness, args.wall_reach),
        frame, n_rect)

    env = make_f110_env(geom.name, args.n_agents, args.dt)
    obs, _ = env.reset(options={"poses": cw.spawn.astype(np.float32)})

    lo = max(0, window.start_wp - 4)
    hi = min(len(geom.centerline) - 1, window.goal_wp + 8)
    cl = geom.centerline[lo:hi + 1]

    def progress(xy):
        d = np.linalg.norm(xy[:, None, :] - cl[None, :, :], axis=-1)
        return np.argmin(d, axis=1) + lo

    zone = FullRunMetrics(
        centerline_xs=geom.centerline[:, 0], centerline_ys=geom.centerline[:, 1],
        dt=args.dt, narrow_center_xy=geom.narrow_xy,
        gap_width_m=geom.gap_width_m, zone_half_width=args.exit_offset,
        collision_thresh=args.collision_thresh,
    )
    zone.step(obs)

    n = args.n_agents
    best_wp = progress(np.stack([obs["poses_x"], obs["poses_y"]], axis=1))
    collided, cstep, reason, step = False, None, "max_steps", 0
    cleared_at: int | None = None

    for step in range(1, args.max_steps + 1):
        xy = np.stack([obs["poses_x"], obs["poses_y"]], axis=1)
        yaw = np.asarray(obs["poses_theta"], dtype=float)
        v_phys = np.asarray(obs["linear_vels_x"], dtype=float)

        agent = np.zeros((n, 4))
        agent[:, :2] = frame.to_gcbf(xy)
        agent[:, 2] = yaw
        agent[:, 3] = v_phys * frame.scale

        goal = np.zeros((n, 4))
        for i in range(n):
            target = int(min(best_wp[i] + args.lookahead_wp, window.goal_wp))
            pose = cw.slot_poses(target)[i]
            goal[i, :2] = frame.to_gcbf(pose[:2])
            goal[i, 2] = pose[2]

        _nxt, action = policy(jnp.asarray(agent, jnp.float32),
                              jnp.asarray(goal, jnp.float32), obstacles)
        act = np.asarray(action)

        # DubinsCar (omega, accel) -> f1tenth (steering angle, speed).
        yaw_rate = act[:, 0] * YAW_RATE_GAIN
        accel_mps2 = act[:, 1] / frame.scale
        speed_cmd = np.clip(v_phys + accel_mps2 * args.dt,
                            0.0, args.max_speed)
        v_for_steer = np.maximum(np.abs(v_phys), args.min_speed_for_steer)
        steer_cmd = np.clip(np.arctan(WHEELBASE * yaw_rate / v_for_steer),
                            -0.4189, 0.4189)

        obs, _r, done, _t, _i = env.step(
            np.stack([steer_cmd, speed_cmd], axis=1).astype(np.float32))
        zone.step(obs)

        xy = np.stack([obs["poses_x"], obs["poses_y"]], axis=1)
        best_wp = np.maximum(best_wp, progress(xy))

        if bool(np.any(obs.get("collisions", 0))) or bool(done):
            # The gym ends the episode on contact; distinguish it from a clean
            # finish by whether the exit test had already been met.
            if not (best_wp >= window.exit_wp).all():
                collided, cstep, reason = True, step, "collision"
                break
        if (best_wp >= window.exit_wp).all():
            reason = "cleared"
            # FullRunMetrics only books an agent out once it is strictly past
            # the exit waypoint; without this grace T_clear and Phi come back
            # inf/0 for runs that did in fact clear.
            if zone.all_zone_cleared:
                break
            if cleared_at is None:
                cleared_at = step
            elif step - cleared_at >= args.exit_grace:
                break

    env.close()
    s = zone.summary()
    n_cleared = int((best_wp >= window.exit_wp).sum())
    return Result(
        split=split, name=geom.name,
        cleared=bool(n_cleared == n and not collided),
        n_cleared=n_cleared, n_agents=n,
        collided=collided, collision_step=cstep,
        steps=step, sim_time_s=step * args.dt, reason=reason,
        avg_speed_mps=float(s["avg_speed_mps"]),
        deformability=float(s["deformability"]),
        flow_rate=float(s["flow_rate"]),
        time_to_goal_s=float(s["time_to_goal_s"]),
        spread_at_entry_m=float(s["spread_at_entry_m"]),
        spread_at_narrow_m=float(s["spread_at_narrow_m"]),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--model", default="gcbfplus/pretrained/DubinsCar/gcbf+")
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--robot-radius", type=float, default=0.29)
    ap.add_argument("--spacing", type=float, default=1.2)
    ap.add_argument("--lookahead-wp", type=int, default=1)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=4000)
    ap.add_argument("--exit-grace", type=int, default=400)
    ap.add_argument("--max-speed", type=float, default=8.0)
    ap.add_argument("--min-speed-for-steer", type=float, default=0.5)
    ap.add_argument("--pre-wp", type=int, default=10)
    ap.add_argument("--exit-offset", type=int, default=8)
    ap.add_argument("--post-wp", type=int, default=12)
    ap.add_argument("--wall-thickness", type=float, default=0.6)
    ap.add_argument("--wall-reach", type=float, default=12.0)
    ap.add_argument("--collision-thresh", type=float, default=0.5)
    ap.add_argument("--n-rect", type=int, default=None)
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
    patch_track_lookup([root for _, root, _ in splits])

    if args.n_rect is None:
        need = 0
        for _, root, names in splits:
            for nm in names:
                g = load_corridor(os.path.join(root, nm))
                w = g.window(args.pre_wp, args.exit_offset, args.post_wp,
                             min_clearance=args.robot_radius + 0.10)
                need = max(need, len(wall_rects(g, w, args.wall_thickness,
                                                args.wall_reach)))
        args.n_rect = need
    print(f"[gcbf-f110] rectangles per map: {args.n_rect}", flush=True)

    from .gcbf_corridor_env import load_gcbf_policy, make_policy_step

    build, config, step = load_gcbf_policy(args.model)
    params = {"car_radius": GCBF_CAR_RADIUS, "comm_radius": 0.5,
              "n_rays": int(getattr(config, "n_rays", 32)),
              "obs_len_range": [0.1, 0.6], "n_obs": args.n_rect}
    env_g, algo = build(args.n_agents, 8.0, args.max_steps, params)
    policy = make_policy_step(env_g, algo, "gcbf")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    rows = []
    t0 = time.time()
    with open(args.out, "w") as fh:
        for label, root, names in splits:
            for i, nm in enumerate(names):
                try:
                    r = rollout(os.path.join(root, nm), label, policy, args,
                                args.n_rect)
                    rows.append(r)
                    fh.write(json.dumps(asdict(r)) + "\n")
                    fh.flush()
                    msg = (f"cleared={r.cleared} ({r.n_cleared}/{r.n_agents}) "
                           f"{r.reason} speed={r.avg_speed_mps:.2f}")
                except Exception as exc:
                    msg = f"ERROR {type(exc).__name__}: {exc}"
                print(f"  [{label} {i + 1}/{len(names)}] {nm}: {msg}", flush=True)

    print(f"\n[gcbf-f110] {len(rows)} layouts in {time.time() - t0:.1f}s")
    for label, _, _ in splits:
        s = [r for r in rows if r.split == label]
        if s:
            k = sum(r.cleared for r in s)
            print(f"  {label}: {k}/{len(s)} = {100 * k / len(s):.1f}%  "
                  f"collisions {sum(r.collided for r in s)}")


if __name__ == "__main__":
    main()
