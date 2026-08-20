"""Evaluate the pretrained GCBF+ policy on FastFunnels narrow-zone maps.

Eval only -- nothing is trained.  The DubinsCar checkpoint shipped in
``gcbfplus/pretrained/DubinsCar/gcbf+`` drives N agents through the same
corridor layouts that produced ``fig_zone_clearance.png``, inside a PyRoboSim
world built from each map.

A map counts as **cleared** when every agent reaches centreline waypoint
``narrow_wp + 8`` without colliding -- the same exit test
``make_rebuttal_figure.py`` and ``record_patch_generalization.py`` apply to the
funnel policy.

Example
-------
    python -m gcbf_baseline.run_gcbf_eval \\
        --split train:eval_maps_matched:50 \\
        --split heldout:eval_maps_heldout:50 \\
        --robot-radius 0.29 --n-agents 4 \\
        --out results/gcbf_zone_clearance.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass

import numpy as np
import shapely

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
# FullRunMetrics is the baselines' own collector -- import it from there
# rather than copying its definitions.
sys.path.insert(0, os.path.join(_ROOT, "baselines"))

from gcbf_baseline.corridor_geom import load_corridor, wall_rects  # noqa: E402
from gcbf_baseline.pyrobosim_world import build_world  # noqa: E402

GCBF_CAR_RADIUS = 0.05  # the length unit the pretrained DubinsCar policy uses


# ---------------------------------------------------------------------------
# results
# ---------------------------------------------------------------------------


@dataclass
class MapResult:
    split: str
    name: str
    gap_width_m: float
    cleared: bool
    n_cleared: int
    n_agents: int
    collided: bool
    collision_kind: str | None
    collision_step: int | None
    steps: int
    sim_time_s: float
    progress_wp: list[int]
    exit_wp: int
    exit_clamped: bool
    narrow_wp: int
    start_wp: int
    min_wall_clearance_m: float
    reason: str
    # ---- baselines/eval_metrics.py FullRunMetrics, verbatim ----------------
    # Computed by the same module ORCA / leader-follower / LAS report through,
    # so these columns are directly comparable with sweep_results_*.csv.
    avg_speed_mps: float
    deformability: float
    flow_rate: float
    time_to_goal_s: float
    zone_success: float
    spread_at_entry_m: float
    spread_at_narrow_m: float
    # Same formula, restricted to steps where >=2 agents are inside the zone,
    # so a lone straggler cannot drive the tightest spread to zero.
    deformability_multi: float
    spread_at_narrow_multi_m: float


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval, matching make_rebuttal_figure.py."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, centre - half), min(1.0, centre + half)


# ---------------------------------------------------------------------------
# rollout
# ---------------------------------------------------------------------------


def rollout_map(
    map_dir: str,
    split: str,
    policy,
    args,
    n_rect: int,
    trace: list | None = None,
) -> MapResult:
    """Run one map and score it."""
    from gcbf_baseline.gcbf_corridor_env import Frame, build_obstacles

    import jax.numpy as jnp

    geom = load_corridor(map_dir, simplify_m=args.simplify)
    window = geom.window(args.pre_wp, args.exit_offset, args.post_wp,
                         min_clearance=args.robot_radius + 0.10)
    cw = build_world(geom, window, args.n_agents, args.robot_radius, args.spacing)

    rects = wall_rects(geom, window, args.wall_thickness, args.wall_reach)
    frame = Frame(
        scale=GCBF_CAR_RADIUS / args.robot_radius,
        offset=np.asarray(geom.centerline[window.start_wp], dtype=float),
    )
    obstacles = build_obstacles(rects, frame, n_rect)

    # Progress is the nearest centreline waypoint, searched only over the scored
    # stretch so a winding corridor cannot alias onto a far-away waypoint.
    lo = max(0, window.start_wp - 4)
    hi = min(len(geom.centerline) - 1, window.goal_wp + 8)
    cl = geom.centerline[lo : hi + 1]

    def progress(xy: np.ndarray) -> np.ndarray:
        d = np.linalg.norm(xy[:, None, :] - cl[None, :, :], axis=-1)
        return np.argmin(d, axis=1) + lo

    n = args.n_agents
    agent = np.zeros((n, 4))
    agent[:, :2] = frame.to_gcbf(cw.spawn[:, :2])
    agent[:, 2] = cw.spawn[:, 2]
    agent_j = jnp.asarray(agent, dtype=jnp.float32)

    fixed_goal = np.zeros((n, 4))
    fixed_goal[:, :2] = frame.to_gcbf(cw.goals[:, :2])
    fixed_goal[:, 2] = cw.goals[:, 2]

    # The five paper metrics come from the baselines' own collector, fed the
    # same obs dict f1tenth_gym hands ORCA and leader-follower.  Its zone
    # window is narrow_wp +/- zone_half_width, matching --exit-offset.
    from eval_metrics import FullRunMetrics

    zone = FullRunMetrics(
        centerline_xs=geom.centerline[:, 0],
        centerline_ys=geom.centerline[:, 1],
        dt=args.dt,
        narrow_center_xy=geom.narrow_xy,
        gap_width_m=geom.gap_width_m,
        zone_half_width=args.exit_offset,
        collision_thresh=args.collision_thresh,
    )

    # FullRunMetrics measures the tightest spread over whichever agents are in
    # the zone, and _lateral_spread of a single agent is 0 -- so as soon as one
    # agent is alone in the zone the ratio blows up and the module reports nan
    # (which is why several ORCA and leader-follower rows are nan too).  Track
    # the same quantity restricted to steps where the formation still has at
    # least two agents inside, which reproduces their value wherever theirs is
    # defined and stays defined where it is not.
    z_entry = max(0, window.narrow_wp - args.exit_offset)
    z_exit = min(len(geom.centerline) - 1, window.narrow_wp + args.exit_offset)
    lat_axis = geom.normal(window.narrow_wp)
    entry_spread_multi: float | None = None
    narrow_spread_multi = float("inf")

    def _spread(xy: np.ndarray) -> float:
        projs = xy @ lat_axis
        return float(projs.max() - projs.min())

    def feed(xy: np.ndarray, speed_mps: np.ndarray) -> None:
        nonlocal entry_spread_multi, narrow_spread_multi
        zone.step({"poses_x": xy[:, 0], "poses_y": xy[:, 1],
                   "linear_vels_x": speed_mps})
        wp = progress(xy)
        in_zone = (wp >= z_entry) & (wp <= z_exit)
        if in_zone.any() and entry_spread_multi is None:
            entry_spread_multi = _spread(xy)  # all agents, as FullRunMetrics does
        if in_zone.sum() >= 2:
            narrow_spread_multi = min(narrow_spread_multi, _spread(xy[in_zone]))

    feed(cw.spawn[:, :2], np.zeros(args.n_agents))

    boundary = geom.walkable.boundary
    best_wp = progress(cw.spawn[:, :2])
    collided, collision_kind, collision_step = False, None, None
    min_clear = float("inf")
    reason = "max_steps"
    step = 0
    cleared_at: int | None = None

    for step in range(1, args.max_steps + 1):
        if args.goal_mode == "carrot":
            goal = np.zeros((n, 4))
            for i in range(n):
                target = int(min(best_wp[i] + args.lookahead_wp, window.goal_wp))
                pose = cw.slot_poses(target)[i]
                goal[i, :2] = frame.to_gcbf(pose[:2])
                goal[i, 2] = pose[2]
        else:
            goal = fixed_goal
        goal_j = jnp.asarray(goal, dtype=jnp.float32)

        prev_j = agent_j
        agent_j, _action = policy(agent_j, goal_j, obstacles)

        state = np.asarray(agent_j)
        xy_m = frame.to_metres(state[:, :2])
        poses = np.stack([xy_m[:, 0], xy_m[:, 1], state[:, 2]], axis=1)
        cw.set_poses(poses)
        feed(xy_m, frame.speed_to_metres(state[:, 3]))
        if trace is not None:
            trace.append(poses.copy())

        best_wp = np.maximum(best_wp, progress(xy_m))
        min_clear = min(min_clear, float(
            shapely.distance(shapely.points(xy_m), boundary).min()
        ))

        hit_wall = cw.wall_collisions(xy_m)
        hit_agent = cw.agent_collisions(xy_m)
        if hit_wall.any() or hit_agent.any():
            if not collided:
                collided = True
                collision_kind = "wall" if hit_wall.any() else "agent"
                collision_step = step
            if args.on_collision == "stop":
                reason = "collision"
                break
            # "revert" reproduces the hand-authored PyRoboSim world: the
            # offending agent is put back where it was, with its speed killed,
            # and the run carries on.
            bad = np.logical_or(hit_wall, hit_agent)
            back = state.copy()
            back[bad] = np.asarray(prev_j)[bad]
            back[bad, 3] = 0.0
            agent_j = jnp.asarray(back, dtype=jnp.float32)
            cw.set_poses(np.stack([
                *frame.to_metres(back[:, :2]).T, back[:, 2]
            ], axis=1))

        if (best_wp >= window.exit_wp).all():
            reason = "cleared"
            # FullRunMetrics only books an agent out of the zone once it is
            # strictly past the exit waypoint, so give the run a short grace
            # window to register those exits before stopping.
            if zone.all_zone_cleared:
                break
            if cleared_at is None:
                cleared_at = step
            elif step - cleared_at >= args.exit_grace:
                break

    n_cleared = int((best_wp >= window.exit_wp).sum())
    cleared = bool(n_cleared == n and not collided)
    zsum = zone.summary()
    if entry_spread_multi and np.isfinite(narrow_spread_multi) and narrow_spread_multi > 1e-9:
        deform_multi = entry_spread_multi / narrow_spread_multi
    else:
        deform_multi = float("nan")

    return MapResult(
        split=split,
        name=geom.name,
        gap_width_m=geom.gap_width_m,
        cleared=cleared,
        n_cleared=n_cleared,
        n_agents=n,
        collided=collided,
        collision_kind=collision_kind,
        collision_step=collision_step,
        steps=step,
        sim_time_s=step * args.dt,
        progress_wp=[int(v) for v in best_wp],
        exit_wp=window.exit_wp,
        exit_clamped=window.exit_clamped,
        narrow_wp=window.narrow_wp,
        start_wp=window.start_wp,
        min_wall_clearance_m=float(min_clear),
        reason=reason,
        avg_speed_mps=float(zsum["avg_speed_mps"]),
        deformability=float(zsum["deformability"]),
        flow_rate=float(zsum["flow_rate"]),
        time_to_goal_s=float(zsum["time_to_goal_s"]),
        zone_success=float(zsum["success"]),
        spread_at_entry_m=float(zsum["spread_at_entry_m"]),
        spread_at_narrow_m=float(zsum["spread_at_narrow_m"]),
        deformability_multi=float(deform_multi),
        spread_at_narrow_multi_m=float(narrow_spread_multi)
        if np.isfinite(narrow_spread_multi) else float("nan"),
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def parse_split(spec: str) -> tuple[str, str, int | None]:
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], int(parts[2])
    raise argparse.ArgumentTypeError(
        f"--split wants label:dir[:count], got {spec!r}"
    )


def collect_maps(root: str, limit: int | None) -> list[str]:
    names = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.exists(os.path.join(root, d, f"{d}_map.yaml"))
    )
    if limit is not None:
        names = names[:limit]
    return [os.path.join(root, d) for d in names]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, type=parse_split,
                    metavar="LABEL:DIR[:N]",
                    help="a named map split, e.g. train:eval_maps_matched:50")
    ap.add_argument("--model", default="gcbfplus/pretrained/DubinsCar/gcbf+",
                    help="pretrained GCBF+ checkpoint directory")
    ap.add_argument("--model-step", type=int, default=None)
    ap.add_argument("--policy", choices=["gcbf", "u_ref"], default="gcbf",
                    help="gcbf: the pretrained GCBF+ actor. u_ref: the "
                         "environment's nominal goal-tracking PID with no "
                         "safety filter, as a reference point")
    ap.add_argument("--out", default="results/gcbf_zone_clearance.jsonl")

    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--robot-radius", type=float, default=0.29,
                    help="agent disc radius in metres; 0.29 circumscribes an "
                         "f1tenth car, 0.15 matches the PyRoboSim script")
    ap.add_argument("--spacing", type=float, default=1.2,
                    help="lateral spacing of the entrance rank, metres")
    ap.add_argument("--goal-mode", choices=["carrot", "fixed"], default="carrot",
                    help="carrot: a global planner feeds GCBF+ a receding "
                         "centreline waypoint (the strong reading of the "
                         "baseline). fixed: one goal past the zone, so GCBF+ "
                         "has to find the way itself")
    ap.add_argument("--lookahead-wp", type=int, default=1,
                    help="carrot distance in centreline waypoints "
                         "(~1.28 m each). Tuned on train maps: the "
                         "baseline is sharply sensitive to this and "
                         "stalls or clips walls beyond ~2")
    ap.add_argument("--on-collision", choices=["stop", "revert"], default="stop",
                    help="stop: a collision ends the run (a crash is a failure). "
                         "revert: push the agent back and carry on, as the "
                         "hand-authored PyRoboSim world does")

    ap.add_argument("--max-steps", type=int, default=1500)
    ap.add_argument("--exit-grace", type=int, default=200,
                    help="extra steps after the exit test passes, so the\n"
                         "baselines' metric collector can book each "
                         "agent out of the zone")
    ap.add_argument("--collision-thresh", type=float, default=0.5,
                    help="centre-to-centre distance counted as an "
                         "agent-agent collision by FullRunMetrics; "
                         "0.5 m is what the other baselines use")
    ap.add_argument("--dt", type=float, default=0.03)
    ap.add_argument("--pre-wp", type=int, default=10)
    ap.add_argument("--exit-offset", type=int, default=8,
                    help="waypoints past the pinch that count as cleared")
    ap.add_argument("--post-wp", type=int, default=12)
    ap.add_argument("--wall-thickness", type=float, default=0.6)
    ap.add_argument("--wall-reach", type=float, default=12.0)
    ap.add_argument("--simplify", type=float, default=0.10)
    ap.add_argument("--n-rect", type=int, default=None,
                    help="rectangles per map fed to GCBF+; default is the max "
                         "any map in the run needs")
    ap.add_argument("--trace-dir", default=None,
                    help="write a .npy trajectory per map for plotting")
    args = ap.parse_args()

    from gcbf_baseline.gcbf_corridor_env import load_gcbf_policy, make_policy_step

    splits = [(label, collect_maps(root, limit)) for label, root, limit in args.split]
    for label, dirs in splits:
        if not dirs:
            raise SystemExit(f"split {label!r} matched no maps")

    # One pass over the geometry so every map is padded to the same rectangle
    # count and the policy step is traced once for the whole run.
    if args.n_rect is None:
        needed = 0
        for _, dirs in splits:
            for d in dirs:
                geom = load_corridor(d, simplify_m=args.simplify)
                w = geom.window(args.pre_wp, args.exit_offset, args.post_wp,
                                min_clearance=args.robot_radius + 0.10)
                needed = max(needed, len(wall_rects(geom, w, args.wall_thickness,
                                                   args.wall_reach)))
        args.n_rect = needed
    print(f"[gcbf] rectangles per map: {args.n_rect}", flush=True)

    build, config, model_step = load_gcbf_policy(args.model, args.model_step)
    params = {
        "car_radius": GCBF_CAR_RADIUS,
        "comm_radius": 0.5,
        "n_rays": int(getattr(config, "n_rays", 32)),
        "obs_len_range": [0.1, 0.6],
        "n_obs": args.n_rect,
    }
    env, algo = build(args.n_agents, 8.0, args.max_steps, params)
    policy = make_policy_step(env, algo, args.policy)
    print(f"[gcbf] policy={args.policy} ({config.algo} {config.env} step {model_step}), "
          f"{args.n_agents} agents, r={args.robot_radius} m, "
          f"scale={GCBF_CAR_RADIUS / args.robot_radius:.4f} units/m, "
          f"v_max={0.8 * args.robot_radius / GCBF_CAR_RADIUS:.2f} m/s",
          flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)

    results: list[MapResult] = []
    t0 = time.time()
    with open(args.out, "w") as f:
        for label, dirs in splits:
            for i, d in enumerate(dirs):
                trace: list | None = [] if args.trace_dir else None
                res = rollout_map(d, label, policy, args, args.n_rect, trace)
                results.append(res)
                f.write(json.dumps(asdict(res)) + "\n")
                f.flush()
                if trace is not None:
                    np.save(os.path.join(args.trace_dir, f"{res.name}.npy"),
                            np.asarray(trace))
                print(f"  [{label} {i + 1}/{len(dirs)}] {res.name}: "
                      f"cleared={res.cleared} ({res.n_cleared}/{res.n_agents}) "
                      f"reason={res.reason} steps={res.steps} "
                      f"min_clear={res.min_wall_clearance_m:.2f} m",
                      flush=True)

    print(f"\n[gcbf] {len(results)} maps in {time.time() - t0:.1f}s -> {args.out}")
    print("\nnarrow-zone clearance (GCBF+ pretrained, no fine-tuning)")
    summary = {}
    for label, _ in splits:
        rs = [r for r in results if r.split == label]
        k, n = sum(r.cleared for r in rs), len(rs)
        lo, hi = wilson(k, n)
        summary[label] = {"k": k, "n": n, "rate": 100.0 * k / n,
                          "ci95": [100 * lo, 100 * hi]}
        crashes = sum(r.collided for r in rs)
        timeouts = sum(r.reason == "max_steps" for r in rs)
        print(f"  {label:>10}: {k:3d}/{n:3d} = {100.0 * k / n:6.2f}%  "
              f"[{100 * lo:.1f}, {100 * hi:.1f}]  "
              f"(collisions {crashes}, timeouts {timeouts})")

    with open(os.path.splitext(args.out)[0] + "_summary.json", "w") as f:
        json.dump({"config": {k: v for k, v in vars(args).items()
                              if k != "split"},
                   "splits": summary}, f, indent=2, default=str)


if __name__ == "__main__":
    main()
