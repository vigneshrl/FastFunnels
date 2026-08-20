"""ORCA in PyRoboSim, keeping the dynamics it uses by default.

The controller is ``baselines/orca.py``'s own -- lane-offset waypoint
following, RVO2 for avoidance, pure-pursuit steering -- and the vehicle is
still f1tenth_gym's single-track model.  Only the world changes: the corridor
comes from the PyRoboSim room polygon and collision is PyRoboSim's
``check_occupancy`` instead of the gym's scan-based test.

Pair this with ``run_gcbf_f110.py`` (GCBF+ on f1tenth_gym) to separate what the
controller contributes from what the simulator does.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass

import numpy as np

from .corridor_geom import load_corridor
from .crossover_common import OrcaBrain, import_orca, orca_waypoints, patch_track_lookup
from .pyrobosim_world import build_world
from .single_track import SingleTrackFleet


@dataclass
class OrcaResult:
    split: str
    name: str
    cleared: bool
    n_cleared: int
    n_agents: int
    collided: bool
    collision_kind: str | None
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
    min_wall_clearance_m: float


def rollout(map_dir: str, split: str, args) -> OrcaResult:
    import shapely
    from eval_metrics import FullRunMetrics

    orca = import_orca()
    geom = load_corridor(map_dir)
    window = geom.window(args.pre_wp, args.exit_offset, args.post_wp,
                         min_clearance=args.car_radius + 0.05)

    # ORCA's own spliced centreline and rank spawn, taken from its own code.
    n = args.n_agents
    xs, ys, spawn, idx0 = orca_waypoints(geom.name, geom.narrow_xy, n,
                                         args.lateral_gap)

    cw = build_world(geom, window, n, args.car_radius, args.lateral_gap)
    cw.set_poses(spawn)

    import yaml
    with open(os.path.join(map_dir, f"{geom.name}_obs_pos.yaml")) as f:
        obs_data = yaml.safe_load(f)
    polys = orca._build_obstacle_polygons(obs_data)
    if args.rvo_walls:
        polys = polys + _wall_polygons(geom, window, args)

    cfg = orca.OrcaConfig(
        num_cars=n, map_name=geom.name, render=False,
        max_steps=args.max_steps, seed=args.seed,
        target_speed=args.target_speed, lateral_gap=args.lateral_gap,
        car_radius=args.car_radius,
    )
    brain = OrcaBrain(cfg, xs, ys, spawn, polys, track_closed=False,
                      start_idx=idx0)
    fleet = SingleTrackFleet(n, dt=args.dt)
    fleet.reset(spawn)

    zone = FullRunMetrics(
        centerline_xs=xs, centerline_ys=ys, dt=args.dt,
        narrow_center_xy=geom.narrow_xy, gap_width_m=geom.gap_width_m,
        zone_half_width=args.zone_half_width,
        collision_thresh=args.collision_thresh,
    )
    zone.step(fleet.obs())

    boundary = geom.walkable.boundary
    collided, kind, cstep = False, None, None
    min_clear = float("inf")
    reason = "max_steps"
    step = 0

    for step in range(1, args.max_steps + 1):
        actions = brain.actions(fleet.obs())
        fleet.step(actions[:, 0], actions[:, 1])

        xy = fleet.xy
        cw.set_poses(fleet.poses())
        zone.step(fleet.obs())
        min_clear = min(min_clear, float(
            shapely.distance(shapely.points(xy), boundary).min()))

        hit_wall = cw.wall_collisions(xy)
        hit_agent = cw.agent_collisions(xy)
        if hit_wall.any() or hit_agent.any():
            collided, kind, cstep = True, ("wall" if hit_wall.any() else "agent"), step
            reason = "collision"
            break
        if zone.all_zone_cleared:
            reason = "cleared"
            break

    s = zone.summary()
    n_cleared = int(s["n_completed"])
    return OrcaResult(
        split=split, name=geom.name,
        cleared=bool(n_cleared == n and not collided),
        n_cleared=n_cleared, n_agents=n,
        collided=collided, collision_kind=kind, collision_step=cstep,
        steps=step, sim_time_s=step * args.dt, reason=reason,
        avg_speed_mps=float(s["avg_speed_mps"]),
        deformability=float(s["deformability"]),
        flow_rate=float(s["flow_rate"]),
        time_to_goal_s=float(s["time_to_goal_s"]),
        spread_at_entry_m=float(s["spread_at_entry_m"]),
        spread_at_narrow_m=float(s["spread_at_narrow_m"]),
        min_wall_clearance_m=float(min_clear),
    )


def _wall_polygons(geom, window, args) -> list:
    """Corridor walls as RVO2 obstacle polygons.

    ``baselines/orca.py`` only registers the scattered obstacles, since in the
    gym the walls live in the map image that RVO2 never sees.  In PyRoboSim the
    walls are explicit geometry, so they can be handed over -- ``--no-rvo-walls``
    reproduces the wall-blind original.
    """
    from .corridor_geom import wall_rects

    polys = []
    for cx, cy, w, h, th in wall_rects(geom, window, args.wall_thickness,
                                       args.wall_reach):
        c, s = np.cos(th), np.sin(th)
        box = np.array([[-w / 2, -h / 2], [w / 2, -h / 2],
                        [w / 2, h / 2], [-w / 2, h / 2]])
        pts = box @ np.array([[c, s], [-s, c]]) + np.array([cx, cy])
        polys.append([(float(x), float(y)) for x, y in pts])
    return polys


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--n-agents", type=int, default=4)
    ap.add_argument("--target-speed", type=float, default=5.0)
    ap.add_argument("--lateral-gap", type=float, default=0.6)
    ap.add_argument("--car-radius", type=float, default=0.25)
    ap.add_argument("--dt", type=float, default=0.01)
    ap.add_argument("--max-steps", type=int, default=2500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--zone-half-width", type=int, default=20)
    ap.add_argument("--collision-thresh", type=float, default=0.5)
    ap.add_argument("--pre-wp", type=int, default=10)
    ap.add_argument("--exit-offset", type=int, default=8)
    ap.add_argument("--post-wp", type=int, default=12)
    ap.add_argument("--wall-thickness", type=float, default=0.6)
    ap.add_argument("--wall-reach", type=float, default=12.0)
    ap.add_argument("--rvo-walls", dest="rvo_walls", action="store_true",
                    default=True)
    ap.add_argument("--no-rvo-walls", dest="rvo_walls", action="store_false")
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
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    t0 = time.time()
    rows = []
    with open(args.out, "w") as fh:
        for label, root, names in splits:
            for i, name in enumerate(names):
                try:
                    r = rollout(os.path.join(root, name), label, args)
                    fh.write(json.dumps(asdict(r)) + "\n")
                    rows.append(r)
                    msg = (f"cleared={r.cleared} ({r.n_cleared}/{r.n_agents}) "
                           f"{r.reason} speed={r.avg_speed_mps:.2f}")
                except Exception as exc:
                    msg = f"ERROR {type(exc).__name__}: {exc}"
                fh.flush()
                print(f"  [{label} {i + 1}/{len(names)}] {name}: {msg}", flush=True)

    print(f"\n[orca-pyrobosim] {len(rows)} layouts in {time.time() - t0:.1f}s")
    for label, _, _ in splits:
        s = [r for r in rows if r.split == label]
        if s:
            k = sum(r.cleared for r in s)
            print(f"  {label}: {k}/{len(s)} = {100 * k / len(s):.1f}%  "
                  f"collisions {sum(r.collided for r in s)}")


if __name__ == "__main__":
    main()
