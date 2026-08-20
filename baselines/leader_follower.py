# #!/usr/bin/env python3
# """
# Leader-Follower baseline for f1tenth_gym.

# Algorithm:
#   Car 0  = leader   — follows the track centerline via Pure Pursuit.
#   Cars 1..N = followers — each follows the car directly ahead via Pure Pursuit,
#               maintaining a configurable following distance.

# Speed control:
#   - Leader drives at a fixed target speed.
#   - Each follower uses a PD controller on the gap error to the car ahead:
#         speed = target_speed + k_gap * (gap - desired_gap) - k_damp * gap_dot
#     clamped to [min_speed, max_speed].

# Physics: f1tenth_gym (single-track model, RK4 integrator).
# Requires: f1tenth_gym, numpy
# """

# from __future__ import annotations

# import math
# import sys
# import os
# import time
# from dataclasses import dataclass
# from typing import List, Optional, Tuple

# import numpy as np
# import yaml

# # ---------------------------------------------------------------------------
# # Path setup — let static analysers find the envs package via sys.path
# # ---------------------------------------------------------------------------
# _HERE = os.path.dirname(os.path.abspath(__file__))
# _FF   = os.path.join(_HERE, "..", "FastFunnels")
# if _FF not in sys.path:
#     sys.path.insert(0, _FF)

# try:
#     from envs.f110_env import F110Config, F110EnvAdapter  # type: ignore[import]
# except ImportError as _e:
#     raise ImportError(
#         f"Cannot import envs.f110_env. Make sure {_FF} is on PYTHONPATH. ({_e})"
#     ) from _e

# # ---------------------------------------------------------------------------
# # Config
# # ---------------------------------------------------------------------------
# @dataclass
# class LeaderFollowerConfig:
#     # ── Scenario ──────────────────────────────────────────────────────────
#     num_cars:           int   = 4        # 1 leader + (num_cars-1) followers
#     map_name:           str   = "open_narrow_obs"
#     render:             bool  = True
#     max_steps:          int   = 10000
#     seed:               int   = 42
#     dt:                 float = 0.01

#     # ── Car dynamics limits ───────────────────────────────────────────────
#     leader_speed:       float = 8.0      # m/s
#     max_speed:          float = 10.0
#     min_speed:          float = 1.5
#     steer_max:          float = 0.4189   # ±24°
#     wheelbase:          float = 0.3302   # LF + LR  (m)

#     # ── Pure Pursuit ──────────────────────────────────────────────────────
#     lookahead_dist:     float = 1.5      # look-ahead distance for leader (m)
#     follower_lookahead: float = 1.2      # look-ahead for followers (m)

#     # ── Formation gap (follower → car-ahead) ──────────────────────────────
#     desired_gap:        float = 1.5      # target centre-to-centre distance (m)
#     k_gap:              float = 1.5      # proportional gain on gap error
#     k_damp:             float = 0.4      # damping on gap rate
#     spawn_gap:          float = 2.0      # initial spacing between cars on spawn
#     lateral_spacing:    float = 0.5      # side-by-side gap between cars at spawn (m)

#     # ── Hazard awareness (obs_pos.yaml) ───────────────────────────────────
#     obs_yaml_path:      Optional[str] = None  # None = auto-detect from map_name
#     hazard_slowdown_dist: float = 8.0   # start slowing this many metres before a hazard
#     hazard_slow_speed:    float = 2.0   # speed (m/s) when nearest hazard < slowdown_dist


# # ---------------------------------------------------------------------------
# # Obstacle-awareness helpers  (obs_pos.yaml)
# # ---------------------------------------------------------------------------

# def _lf_find_obs_yaml(map_name: str, explicit: Optional[str]) -> Optional[dict]:
#     """Load obs_pos.yaml for the current map. Returns None when not found."""
#     if explicit:
#         path = explicit
#     else:
#         try:
#             from f1tenth_gym.envs.track.utils import find_track_dir
#             path = str(find_track_dir(map_name) / f"{map_name}_obs_pos.yaml")
#         except Exception:
#             return None
#     try:
#         with open(path) as f:
#             return yaml.safe_load(f)
#     except FileNotFoundError:
#         return None


# def _hazard_centers(obs_data: dict) -> np.ndarray:
#     """
#     Return (N, 2) float64 array of hazard centre positions.
#     Includes toll pillars, circle/rect obstacles, and narrow region centres.
#     """
#     pts: List[List[float]] = []
#     for p in obs_data.get("toll_pillars") or []:
#         pts.append(p["center_m"])
#     for o in obs_data.get("obstacles") or []:
#         pts.append(o["center_m"])
#     for nr in obs_data.get("narrow_regions") or []:
#         pts.append(nr["center_m"])
#     return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)


# # ---------------------------------------------------------------------------
# # Pure Pursuit controller
# # ---------------------------------------------------------------------------

# def _wrap(a: float) -> float:
#     return float((a + math.pi) % (2.0 * math.pi) - math.pi)


# def _offset_centerline(xs: np.ndarray, ys: np.ndarray,
#                         offset: float) -> Tuple[np.ndarray, np.ndarray]:
#     """Return a copy of the centerline shifted `offset` metres to the left."""
#     n = len(xs)
#     oxs = np.empty(n, dtype=np.float32)
#     oys = np.empty(n, dtype=np.float32)
#     for k in range(n):
#         nk = (k + 1) % n
#         th = math.atan2(float(ys[nk] - ys[k]), float(xs[nk] - xs[k]))
#         oxs[k] = xs[k] - offset * math.sin(th)
#         oys[k] = ys[k] + offset * math.cos(th)
#     return oxs, oys


# class PurePursuit:
#     """Geometric pure-pursuit path follower.

#     Finds the first waypoint ≥ lookahead_dist ahead of the car and computes
#     the Ackermann steering angle to reach it in an arc.
#     """

#     def __init__(self, xs: np.ndarray, ys: np.ndarray,
#                  lookahead_dist: float, wheelbase: float, steer_max: float,
#                  closed: bool = True):
#         self.xs = xs
#         self.ys = ys
#         self.L  = lookahead_dist
#         self.WB = wheelbase
#         self.steer_max = steer_max
#         self._n = len(xs)
#         self.closed = closed

#     def _find_lookahead(self, x: float, y: float,
#                         current_idx: int) -> Tuple[float, float, int]:
#         """Return (lx, ly, new_pos_idx) for the lookahead point.

#         new_pos_idx is the car's nearest waypoint (not the lookahead), so that
#         the next call's bounded search stays anchored to where the car actually
#         is — not where it was aiming.  Starting from the old lookahead index
#         causes the car to steer toward a waypoint that is now behind it whenever
#         it cuts the inside of a curve, which produces circular orbiting.
#         """
#         n = self._n
#         # Step 1: find car's nearest waypoint within a forward-biased window
#         lo = max(0, current_idx - 5)
#         hi = min(n - 1, current_idx + 25)
#         best_d = float("inf")
#         nearest = current_idx
#         for k in range(lo, hi + 1):
#             d = math.hypot(float(self.xs[k]) - x, float(self.ys[k]) - y)
#             if d < best_d:
#                 best_d = d
#                 nearest = k
#         # Step 2: advance forward from nearest until ≥ lookahead distance
#         idx = nearest
#         for _ in range(n):
#             lx = float(self.xs[idx])
#             ly = float(self.ys[idx])
#             if math.hypot(lx - x, ly - y) >= self.L:
#                 return lx, ly, nearest
#             next_idx = (idx + 1) % n if self.closed else min(idx + 1, n - 1)
#             if not self.closed and next_idx == idx:
#                 return lx, ly, nearest
#             idx = next_idx
#         return float(self.xs[nearest]), float(self.ys[nearest]), nearest

#     def curvature_speed(self, wp_idx: int, base_speed: float,
#                         min_speed: float, k: float = 6.0) -> float:
#         """Scale speed down proportionally to max upcoming curvature (20 wp lookahead)."""
#         n = self._n
#         # Bug 3 fix: skip curvature scan near end of open path — clamped indices
#         # collapse to the same point producing zero vectors and atan2(0,0).
#         if not self.closed and wp_idx >= n - 20:
#             return base_speed
#         max_curv = 0.0
#         for _j in range(20):
#             a = wp_idx + _j
#             b = wp_idx + _j + 3
#             if self.closed:
#                 a %= n; b %= n
#             else:
#                 a = min(a, n - 1); b = min(b, n - 1)
#             if a == b:
#                 break
#             if _j == 0:
#                 p = min(wp_idx + 1, n - 1) if not self.closed else (wp_idx + 1) % n
#                 _dxh = float(self.xs[p] - self.xs[wp_idx])
#                 _dyh = float(self.ys[p] - self.ys[wp_idx])
#                 if math.hypot(_dxh, _dyh) < 1e-6:
#                     break
#                 _h1 = math.atan2(_dyh, _dxh)
#             else:
#                 pa = min(a - 1, n - 1) if not self.closed else (a - 1) % n
#                 _dxh = float(self.xs[a] - self.xs[pa])
#                 _dyh = float(self.ys[a] - self.ys[pa])
#                 if math.hypot(_dxh, _dyh) < 1e-6:
#                     break
#                 _h1 = math.atan2(_dyh, _dxh)
#             _dx2 = float(self.xs[b] - self.xs[a])
#             _dy2 = float(self.ys[b] - self.ys[a])
#             if math.hypot(_dx2, _dy2) < 1e-6:
#                 break
#             _h2 = math.atan2(_dy2, _dx2)
#             max_curv = max(max_curv, abs(_wrap(_h2 - _h1)))
#         return max(base_speed / (1.0 + k * max_curv), min_speed)

#     def steer(self, x: float, y: float, theta: float,
#               wp_idx: int) -> Tuple[float, int]:
#         """Return (steering_angle_rad, updated_wp_idx)."""
#         lx, ly, wp_idx = self._find_lookahead(x, y, wp_idx)
#         alpha = _wrap(math.atan2(ly - y, lx - x) - theta)
#         steer = math.atan2(2.0 * self.WB * math.sin(alpha), self.L)
#         return float(np.clip(steer, -self.steer_max, self.steer_max)), wp_idx


# # ---------------------------------------------------------------------------
# # Follower speed controller (PD on gap to car-ahead)
# # ---------------------------------------------------------------------------

# class GapController:
#     """PD controller: keeps a follower at `desired_gap` behind the car ahead."""

#     def __init__(self, cfg: LeaderFollowerConfig):
#         self.cfg = cfg
#         self._prev_gap: Optional[float] = None

#     def speed(self, gap: float) -> float:
#         cfg = self.cfg
#         if self._prev_gap is None:
#             self._prev_gap = gap
#         gap_err  = gap - cfg.desired_gap
#         gap_dot  = (gap - self._prev_gap) / cfg.dt
#         self._prev_gap = gap
#         cmd = cfg.leader_speed + cfg.k_gap * gap_err - cfg.k_damp * gap_dot
#         return float(np.clip(cmd, cfg.min_speed, cfg.max_speed))

#     def reset(self) -> None:
#         self._prev_gap = None


# # ---------------------------------------------------------------------------
# # Leader-Follower runner
# # ---------------------------------------------------------------------------

# class F1TenthLeaderFollower:
#     """Leader-follower formation using f1tenth_gym physics."""

#     def __init__(self, cfg: LeaderFollowerConfig):
#         self.cfg = cfg
#         np.random.seed(cfg.seed)

#         self.adapter = F110EnvAdapter(
#             F110Config(
#                 map_name=cfg.map_name,
#                 num_agents=cfg.num_cars,
#                 timestep=cfg.dt,
#                 control_input=("speed", "steering_angle"),
#                 model="st",
#                 reset_type=None,
#             ),
#             render_mode="human" if cfg.render else None,
#         )

#         self.car_pp: List[PurePursuit] = []
#         self.wp_idx_cars: List[int] = []

#     # ── Episode init ────────────────────────────────────────────────────────

#     def _init_episode(self) -> dict:
#         self.adapter.ensure_initialized()

#         track, _, _, _ = self.adapter.get_track_data()
#         xs = np.asarray(track.centerline.xs, dtype=np.float32)
#         ys = np.asarray(track.centerline.ys, dtype=np.float32)
#         self._track_closed = math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0])) < 5.0
#         print(f"[LF] Track is {'closed loop' if self._track_closed else 'open path'} ({len(xs)} waypoints)")

#         n   = self.cfg.num_cars
#         cfg = self.cfg

#         # Spawn all cars side by side (parallel) at waypoint 0
#         idx0    = 0
#         theta0  = math.atan2(float(ys[1] - ys[0]), float(xs[1] - xs[0]))
#         # perpendicular-left unit vector (rotate fwd 90° CCW)
#         perp_x  = -math.sin(theta0)
#         perp_y  =  math.cos(theta0)

#         # centre the row: offsets are symmetric around 0
#         poses = []
#         for i in range(n):
#             lateral = (i - (n - 1) / 2.0) * cfg.lateral_spacing
#             poses.append([
#                 float(xs[idx0]) + lateral * perp_x,
#                 float(ys[idx0]) + lateral * perp_y,
#                 theta0,
#             ])
#         poses_np = np.array(poses, dtype=np.float32)

#         obs, _ = self.adapter.reset(poses=poses_np)

#         # Load hazard centres for speed reduction near obstacles / toll / narrows
#         self._hazards: np.ndarray = np.empty((0, 2), dtype=np.float64)
#         obs_data = _lf_find_obs_yaml(self.cfg.map_name, self.cfg.obs_yaml_path)
#         if obs_data:
#             self._hazards = _hazard_centers(obs_data)
#             print(f"[LF] Loaded {len(self._hazards)} hazard centre(s) from obs_pos.yaml")
#         else:
#             print(f"[LF] No obs_pos.yaml for '{self.cfg.map_name}' — hazard slowdown disabled")

#         # Each car gets Pure Pursuit on its own laterally-offset lane
#         self.car_pp: List[PurePursuit] = []
#         self.wp_idx_cars: List[int] = []
#         for i in range(n):
#             lateral = (i - (n - 1) / 2.0) * cfg.lateral_spacing
#             oxs, oys = _offset_centerline(xs, ys, lateral)
#             la = cfg.lookahead_dist if i == 0 else cfg.follower_lookahead
#             self.car_pp.append(PurePursuit(oxs, oys, la, cfg.wheelbase, cfg.steer_max,
#                                            closed=self._track_closed))
#             self.wp_idx_cars.append(0)

#         return obs

#     # ── Per-step control ────────────────────────────────────────────────────

#     def _compute_actions(self, obs: dict) -> np.ndarray:
#         n   = self.cfg.num_cars
#         cfg = self.cfg

#         px    = obs["poses_x"]
#         py    = obs["poses_y"]
#         theta = obs["poses_theta"]

#         actions = np.zeros((n, 2), dtype=np.float32)

#         # All cars: Pure Pursuit on their own offset lane; slow near hazards
#         for i in range(n):
#             steer, self.wp_idx_cars[i] = self.car_pp[i].steer(
#                 float(px[i]), float(py[i]), float(theta[i]),
#                 self.wp_idx_cars[i],
#             )

#             # Bug 1 fix: halt at end of open path to prevent looping back to start
#             if not self.car_pp[i].closed and self.wp_idx_cars[i] >= self.car_pp[i]._n - 1:
#                 actions[i] = [0.0, 0.0]
#                 continue

#             speed = self.car_pp[i].curvature_speed(
#                 self.wp_idx_cars[i], cfg.leader_speed, cfg.min_speed
#             )
#             if len(self._hazards) > 0:
#                 pos  = np.array([float(px[i]), float(py[i])])
#                 dist = float(np.min(np.linalg.norm(self._hazards - pos, axis=1)))
#                 if dist < cfg.hazard_slowdown_dist:
#                     # Linear ramp: full slow_speed at dist=0, leader_speed at dist=slowdown_dist
#                     t     = dist / cfg.hazard_slowdown_dist
#                     speed = cfg.hazard_slow_speed + t * (cfg.leader_speed - cfg.hazard_slow_speed)
#                     speed = float(np.clip(speed, cfg.min_speed, cfg.max_speed))

#             actions[i] = [steer, speed]

#         return actions

#     # ── Main run ────────────────────────────────────────────────────────────

#     def run(self) -> dict:
#         obs  = self._init_episode()
#         n    = self.cfg.num_cars
#         t0   = time.time()
#         step = 0

#         # ── Full-run metrics collector ───────────────────────────────
#         from eval_metrics import FullRunMetrics      # noqa: PLC0415
#         track, _, _, _ = self.adapter.get_track_data()
#         _obs_data = _lf_find_obs_yaml(self.cfg.map_name, self.cfg.obs_yaml_path)
#         _narrow_xy, _gap = None, None
#         if _obs_data and _obs_data.get("narrow_regions"):
#             _nr        = _obs_data["narrow_regions"][0]
#             _narrow_xy = tuple(_nr["center_m"])
#             _gap       = float(_nr["gap_width_m"])
#         zone = FullRunMetrics(
#             centerline_xs=np.asarray(track.centerline.xs),
#             centerline_ys=np.asarray(track.centerline.ys),
#             dt=self.cfg.dt,
#             narrow_center_xy=_narrow_xy,
#             gap_width_m=_gap,
#         )
#         # ─────────────────────────────────────────────────────────────

#         traj_history: List[List[np.ndarray]] = [[] for _ in range(n)]
#         for i in range(n):
#             traj_history[i].append(np.array([float(obs["poses_x"][i]),
#                                              float(obs["poses_y"][i])]))

#         gaps: List[List[float]] = [[] for _ in range(n - 1)]

#         for step in range(self.cfg.max_steps):
#             actions = self._compute_actions(obs)
#             obs, _, done, _, _ = self.adapter.step(actions)

#             if self.cfg.render:
#                 self.adapter.render()

#             zone.step(obs)
#             if zone.all_done:
#                 print(f"[LeaderFollower] All agents reached goal at step {step}")
#                 break

#             px = obs["poses_x"]
#             py = obs["poses_y"]

#             for i in range(n):
#                 traj_history[i].append(np.array([float(px[i]), float(py[i])]))
#             for i in range(1, n):
#                 g = math.hypot(float(px[i] - px[i - 1]), float(py[i] - py[i - 1]))
#                 gaps[i - 1].append(g)

#             if done:
#                 print(f"[LeaderFollower] Episode ended at step {step}")
#                 break

#         elapsed = time.time() - t0

#         gap_means = [float(np.mean(g)) if g else float("nan") for g in gaps]
#         gap_stds  = [float(np.std(g))  if g else float("nan") for g in gaps]

#         zone_metrics = zone.summary()

#         metrics = {
#             "elapsed_s":  elapsed,
#             "steps":      step + 1,
#             "mean_gap_m": gap_means,
#             "std_gap_m":  gap_stds,
#             **zone_metrics,
#         }

#         print("\n======= LEADER-FOLLOWER METRICS =======")
#         print(f"Steps run   : {metrics['steps']}")
#         print(f"Elapsed (s) : {elapsed:.1f}")
#         for fi, (mu, sd) in enumerate(zip(gap_means, gap_stds)):
#             print(f"Car {fi+1}↔Car {fi+2} lateral gap: {mu:.3f} ± {sd:.3f} m")
#         print(f"\n--- Full-Run Metrics ---")
#         print(f"Avg speed         : {zone_metrics['avg_speed_mps']:.2f} m/s")
#         print(f"Success           : {zone_metrics['success']}")
#         print(f"Time to goal (s)  : {zone_metrics['time_to_goal_s']:.3f}")
#         print(f"Flow rate         : {zone_metrics['flow_rate']:.4f} agents/(m·s)")
#         print(f"Deformability     : {zone_metrics['deformability']}")
#         print(f"Spread at entry   : {zone_metrics['spread_at_entry_m']:.3f} m")
#         print(f"Spread at narrowest: {zone_metrics['spread_at_narrow_m']:.3f} m")
#         print("=======================================\n")

#         return metrics


# # ---------------------------------------------------------------------------
# # Entry point
# # ---------------------------------------------------------------------------

# if __name__ == "__main__":
#     import argparse

#     parser = argparse.ArgumentParser(
#         description="Leader-follower baseline on f1tenth_gym"
#     )
#     parser.add_argument("--num-cars",         type=int,   default=4)
#     parser.add_argument("--map",              type=str,   default=None)
#     parser.add_argument("--leader-speed",     type=float, default=3.0)
#     parser.add_argument("--lookahead",        type=float, default=1.5)
#     parser.add_argument("--lateral-spacing",  type=float, default=1.0)
#     parser.add_argument("--max-steps",        type=int,   default=100000)
#     parser.add_argument("--no-render",        action="store_true")
#     parser.add_argument("--seed",             type=int,   default=42)
#     parser.add_argument("--obs-yaml",         type=str,   default=None,
#                         help="Path to obs_pos.yaml (default: auto-detect beside map)")
#     parser.add_argument("--hazard-slow-dist", type=float, default=8.0,
#                         help="Distance (m) at which to begin slowing near a hazard")
#     parser.add_argument("--hazard-slow-speed", type=float, default=2.0,
#                         help="Speed (m/s) to use when closest to a hazard centre")
#     args = parser.parse_args()

#     cfg = LeaderFollowerConfig(
#         num_cars=args.num_cars,
#         map_name=args.map,
#         leader_speed=args.leader_speed,
#         lookahead_dist=args.lookahead,
#         lateral_spacing=args.lateral_spacing,
#         max_steps=args.max_steps,
#         render=not args.no_render,
#         seed=args.seed,
#         obs_yaml_path=args.obs_yaml,
#         hazard_slowdown_dist=args.hazard_slow_dist,
#         hazard_slow_speed=args.hazard_slow_speed,
#     )

#     F1TenthLeaderFollower(cfg).run()


#!/usr/bin/env python3
"""
Leader-Follower baseline for f1tenth_gym.

Algorithm:
  Car 0  = leader   — follows the track centerline via Pure Pursuit.
  Cars 1..N = followers — each follows the same centerline via Pure Pursuit,
              maintaining a configurable following distance behind the car ahead.

Speed control:
  - Leader drives at a fixed target speed, scaled down for curves.
  - Each follower uses a PD controller on the longitudinal gap error:
        speed = leader_speed + k_gap * (gap - desired_gap) - k_damp * gap_dot
    clamped to [min_speed, max_speed].

Physics: f1tenth_gym (single-track model, RK4 integrator).
Requires: f1tenth_gym, numpy
"""

from __future__ import annotations

import math
import sys
import os
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Path setup — let static analysers find the envs package via sys.path
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_FF   = os.path.join(_HERE, "..", "FastFunnels")
if _FF not in sys.path:
    sys.path.insert(0, _FF)

try:
    from envs.f110_env import F110Config, F110EnvAdapter  # type: ignore[import]
except ImportError as _e:
    raise ImportError(
        f"Cannot import envs.f110_env. Make sure {_FF} is on PYTHONPATH. ({_e})"
    ) from _e

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
@dataclass
class LeaderFollowerConfig:
    # ── Scenario ──────────────────────────────────────────────────────────
    num_cars:           int   = 4        # 1 leader + (num_cars-1) followers
    map_name:           str   = "open_narrow_obs"
    render:             bool  = True
    max_steps:          int   = 10000
    seed:               int   = 42
    dt:                 float = 0.01

    # ── Car dynamics limits ───────────────────────────────────────────────
    leader_speed:       float = 8.0      # m/s
    max_speed:          float = 10.0
    min_speed:          float = 1.5
    steer_max:          float = 0.4189   # ±24°
    wheelbase:          float = 0.3302   # LF + LR  (m)

    # ── Pure Pursuit ──────────────────────────────────────────────────────
    lookahead_dist:     float = 1.5      # look-ahead distance for leader (m)
    follower_lookahead: float = 1.2      # look-ahead for followers (m)

    # ── Formation gap (follower → car-ahead) ──────────────────────────────
    desired_gap:        float = 1.5      # target centre-to-centre distance (m)
    k_gap:              float = 1.5      # proportional gain on gap error
    k_damp:             float = 0.4      # damping on gap rate
    spawn_gap:          float = 2.0      # initial longitudinal spacing between cars

    # ── Hazard awareness (obs_pos.yaml) ───────────────────────────────────
    obs_yaml_path:      Optional[str] = None  # None = auto-detect from map_name
    hazard_slowdown_dist: float = 8.0   # start slowing this many metres before a hazard
    hazard_slow_speed:    float = 2.0   # speed (m/s) when nearest hazard < slowdown_dist


# ── Narrow-zone approach waypoint injection (mirrors orca.py) ────────────────
_LF_D_UP   = 7.0
_LF_D_DOWN = 20.0  # m south of zone centre (past physical corridor exit ≈12 m)
_LF_N_APP  = 46    # synthetic waypoints (≈0.59 m spacing for the 27 m span)


def _lf_narrow_approach_wps(
    xs: np.ndarray, ys: np.ndarray, narrow_xy: tuple
) -> tuple:
    """Return (xs_ext, ys_ext, spawn_x, spawn_y, theta_fwd, idx0)."""
    import math as _math
    nx, ny  = float(narrow_xy[0]), float(narrow_xy[1])
    ins_at  = int(np.argmin(np.hypot(xs - nx, ys - ny)))
    spawn_x = nx
    spawn_y = ny + _LF_D_UP
    exit_y  = ny - _LF_D_DOWN
    app_xs  = np.full(_LF_N_APP, nx, dtype=np.float32)
    app_ys  = np.linspace(spawn_y, exit_y, _LF_N_APP, dtype=np.float32)
    xs_ext  = np.concatenate([xs[:ins_at], app_xs, xs[ins_at:]])
    ys_ext  = np.concatenate([ys[:ins_at], app_ys, ys[ins_at:]])
    return xs_ext, ys_ext, spawn_x, spawn_y, -_math.pi / 2, ins_at


# ---------------------------------------------------------------------------
# Obstacle-awareness helpers  (obs_pos.yaml)
# ---------------------------------------------------------------------------

def _lf_find_obs_yaml(map_name: str, explicit: Optional[str]) -> Optional[dict]:
    """Load obs_pos.yaml for the current map. Returns None when not found."""
    if explicit:
        path = explicit
    else:
        try:
            from f1tenth_gym.envs.track.utils import find_track_dir
            path = str(find_track_dir(map_name) / f"{map_name}_obs_pos.yaml")
        except Exception:
            return None
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        return None

def _hazard_centers(obs_data: dict) -> np.ndarray:
    """
    Return (N, 2) float64 array of hazard centre positions.
    Includes toll pillars, circle/rect obstacles, and narrow region centres.
    """
    pts: List[List[float]] = []
    for p in obs_data.get("toll_pillars") or []:
        pts.append(p["center_m"])
    for o in obs_data.get("obstacles") or []:
        pts.append(o["center_m"])
    for nr in obs_data.get("narrow_regions") or []:
        pts.append(nr["center_m"])
    return np.array(pts, dtype=np.float64) if pts else np.empty((0, 2), dtype=np.float64)


# ---------------------------------------------------------------------------
# Pure Pursuit controller
# ---------------------------------------------------------------------------

def _wrap(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)

class PurePursuit:
    """Geometric pure-pursuit path follower."""

    def __init__(self, xs: np.ndarray, ys: np.ndarray,
                 lookahead_dist: float, wheelbase: float, steer_max: float,
                 closed: bool = True):
        self.xs = xs
        self.ys = ys
        self.L  = lookahead_dist
        self.WB = wheelbase
        self.steer_max = steer_max
        self._n = len(xs)
        self.closed = closed

    def _find_lookahead(self, x: float, y: float,
                        current_idx: int) -> Tuple[float, float, int]:
        n = self._n
        lo = max(0, current_idx - 5)
        hi = min(n - 1, current_idx + 25)
        best_d = float("inf")
        nearest = current_idx
        for k in range(lo, hi + 1):
            d = math.hypot(float(self.xs[k]) - x, float(self.ys[k]) - y)
            if d < best_d:
                best_d = d
                nearest = k
        idx = nearest
        for _ in range(n):
            lx = float(self.xs[idx])
            ly = float(self.ys[idx])
            if math.hypot(lx - x, ly - y) >= self.L:
                return lx, ly, nearest
            next_idx = (idx + 1) % n if self.closed else min(idx + 1, n - 1)
            if not self.closed and next_idx == idx:
                return lx, ly, nearest
            idx = next_idx
        return float(self.xs[nearest]), float(self.ys[nearest]), nearest

    def curvature_speed(self, wp_idx: int, base_speed: float,
                        min_speed: float, k: float = 6.0) -> float:
        n = self._n
        # Skip curvature scan near end of open path to avoid degenerate math
        if not self.closed and wp_idx >= n - 20:
            return base_speed
            
        max_curv = 0.0
        for _j in range(20):
            a = wp_idx + _j
            b = wp_idx + _j + 3
            if self.closed:
                a %= n; b %= n
            else:
                a = min(a, n - 1); b = min(b, n - 1)
            if a == b:
                break
            if _j == 0:
                p = min(wp_idx + 1, n - 1) if not self.closed else (wp_idx + 1) % n
                _dxh = float(self.xs[p] - self.xs[wp_idx])
                _dyh = float(self.ys[p] - self.ys[wp_idx])
                if math.hypot(_dxh, _dyh) < 1e-6:
                    break
                _h1 = math.atan2(_dyh, _dxh)
            else:
                pa = min(a - 1, n - 1) if not self.closed else (a - 1) % n
                _dxh = float(self.xs[a] - self.xs[pa])
                _dyh = float(self.ys[a] - self.ys[pa])
                if math.hypot(_dxh, _dyh) < 1e-6:
                    break
                _h1 = math.atan2(_dyh, _dxh)
            _dx2 = float(self.xs[b] - self.xs[a])
            _dy2 = float(self.ys[b] - self.ys[a])
            if math.hypot(_dx2, _dy2) < 1e-6:
                break
            _h2 = math.atan2(_dy2, _dx2)
            _dh = abs(_wrap(_h2 - _h1))
            if _dh > math.pi / 2:   # discontinuity at injected-wp boundary
                break
            max_curv = max(max_curv, _dh)
        return max(base_speed / (1.0 + k * max_curv), min_speed)

    def steer(self, x: float, y: float, theta: float,
              wp_idx: int) -> Tuple[float, int]:
        lx, ly, wp_idx = self._find_lookahead(x, y, wp_idx)
        alpha = _wrap(math.atan2(ly - y, lx - x) - theta)
        steer = math.atan2(2.0 * self.WB * math.sin(alpha), self.L)
        return float(np.clip(steer, -self.steer_max, self.steer_max)), wp_idx


# ---------------------------------------------------------------------------
# Follower speed controller (PD on gap to car-ahead)
# ---------------------------------------------------------------------------

class GapController:
    """PD controller: keeps a follower at `desired_gap` behind the car ahead."""

    def __init__(self, cfg: LeaderFollowerConfig):
        self.cfg = cfg
        self._prev_gap: Optional[float] = None

    def speed(self, gap: float) -> float:
        cfg = self.cfg
        if self._prev_gap is None:
            self._prev_gap = gap
        gap_err  = gap - cfg.desired_gap
        gap_dot  = (gap - self._prev_gap) / cfg.dt
        self._prev_gap = gap
        cmd = cfg.leader_speed + cfg.k_gap * gap_err + cfg.k_damp * gap_dot
        return float(np.clip(cmd, cfg.min_speed, cfg.max_speed))

    def reset(self) -> None:
        self._prev_gap = None


# ---------------------------------------------------------------------------
# Leader-Follower runner
# ---------------------------------------------------------------------------

class F1TenthLeaderFollower:
    def __init__(self, cfg: LeaderFollowerConfig):
        self.cfg = cfg
        np.random.seed(cfg.seed)

        self.adapter = F110EnvAdapter(
            F110Config(
                map_name=cfg.map_name,
                num_agents=cfg.num_cars,
                timestep=cfg.dt,
                control_input=("speed", "steering_angle"),
                model="st",
                reset_type=None,
            ),
            render_mode="human" if cfg.render else None,
        )

        self.car_pp: List[PurePursuit] = []
        self.wp_idx_cars: List[int] = []
        self.gap_controllers: List[GapController] = []
        self._ext_xs: np.ndarray = np.empty(0, dtype=np.float32)
        self._ext_ys: np.ndarray = np.empty(0, dtype=np.float32)

    def _init_episode(self) -> dict:
        self.adapter.ensure_initialized()

        track, _, _, _ = self.adapter.get_track_data()
        xs = np.asarray(track.centerline.xs, dtype=np.float32)
        ys = np.asarray(track.centerline.ys, dtype=np.float32)
        self._track_closed = math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0])) < 5.0
        print(f"[LF] Track is {'closed loop' if self._track_closed else 'open path'} ({len(xs)} waypoints)")

        n   = self.cfg.num_cars
        cfg = self.cfg

        # Load obs_pos.yaml now — needed for both spawn location and hazards.
        obs_data = _lf_find_obs_yaml(self.cfg.map_name, self.cfg.obs_yaml_path)

        # Spawn cars in single-file at the narrow-zone approach.
        # The original centerline waypoints 364-378 are inside track walls;
        # inject synthetic approach waypoints and spawn from zone geometry.
        idx0   = 0
        spawn_x, spawn_y = float(xs[0]), float(ys[0])   # fallback
        theta0 = 0.0

        if obs_data and obs_data.get("narrow_regions"):
            _nr = obs_data["narrow_regions"][0]
            _narrow_xy = (float(_nr["center_m"][0]), float(_nr["center_m"][1]))
            xs, ys, spawn_x, spawn_y, theta0, idx0 = \
                _lf_narrow_approach_wps(xs, ys, _narrow_xy)
            print(f"[LF] Spawning at ({spawn_x:.2f}, {spawn_y:.2f}), "
                  f"hdg={math.degrees(theta0):.0f}° "
                  f"(extended CL: {len(xs)} wps, approach idx={idx0})")

        # Single-file: stagger behind spawn in the north direction
        back_x = -math.cos(theta0)   # +0 when theta0=-π/2
        back_y = -math.sin(theta0)   # +1 (north) when theta0=-π/2
        poses  = []
        for i in range(n):
            poses.append([spawn_x + i * cfg.spawn_gap * back_x,
                          spawn_y + i * cfg.spawn_gap * back_y,
                          theta0])
        poses_np = np.array(poses, dtype=np.float32)

        obs, _ = self.adapter.reset(poses=poses_np)

        self._hazards: np.ndarray = np.empty((0, 2), dtype=np.float64)
        if obs_data:
            self._hazards = _hazard_centers(obs_data)
            print(f"[LF] Loaded {len(self._hazards)} hazard centre(s) from obs_pos.yaml")

        # Store extended centerline for run() → FullRunMetrics
        self._ext_xs = xs
        self._ext_ys = ys

        # ALL cars share the exact same (extended) centerline
        self.car_pp = []
        self.wp_idx_cars = []
        for i in range(n):
            la = cfg.lookahead_dist if i == 0 else cfg.follower_lookahead
            self.car_pp.append(PurePursuit(xs, ys, la, cfg.wheelbase, cfg.steer_max,
                                           closed=self._track_closed))
            self.wp_idx_cars.append(idx0)

        # Initialize the gap controllers for the followers
        self.gap_controllers = [GapController(cfg) for _ in range(n - 1)]

        return obs

    def _compute_actions(self, obs: dict) -> np.ndarray:
        n   = self.cfg.num_cars
        cfg = self.cfg

        px    = obs["poses_x"]
        py    = obs["poses_y"]
        theta = obs["poses_theta"]

        actions = np.zeros((n, 2), dtype=np.float32)

        for i in range(n):
            # 1. Steering
            steer, self.wp_idx_cars[i] = self.car_pp[i].steer(
                float(px[i]), float(py[i]), float(theta[i]),
                self.wp_idx_cars[i],
            )

            # Halt at end of open path
            if not self.car_pp[i].closed and self.wp_idx_cars[i] >= self.car_pp[i]._n - 1:
                actions[i] = [0.0, 0.0]
                continue

            # 2. Speed Control
            if i == 0:
                # Leader sets the pace
                speed = self.car_pp[i].curvature_speed(
                    self.wp_idx_cars[i], cfg.leader_speed, cfg.min_speed
                )
            else:
                # Follower maintains the gap to the car directly ahead
                dx = float(px[i-1] - px[i])
                dy = float(py[i-1] - py[i])
                actual_gap = math.hypot(dx, dy)
                speed = self.gap_controllers[i-1].speed(actual_gap)

            # 3. Hazard slowdown (applies to both leader and followers)
            if len(self._hazards) > 0:
                pos  = np.array([float(px[i]), float(py[i])])
                dist = float(np.min(np.linalg.norm(self._hazards - pos, axis=1)))
                if dist < cfg.hazard_slowdown_dist:
                    t = dist / cfg.hazard_slowdown_dist
                    hazard_limit = cfg.hazard_slow_speed + t * (cfg.leader_speed - cfg.hazard_slow_speed)
                    speed = min(speed, float(np.clip(hazard_limit, cfg.min_speed, cfg.max_speed)))

            actions[i] = [steer, speed]

        return actions

    def run(self) -> dict:
        obs  = self._init_episode()
        n    = self.cfg.num_cars
        t0   = time.time()
        step = 0

        # ── Full-run metrics collector ───────────────────────────────
        try:
            from eval_metrics import FullRunMetrics
            _obs_data = _lf_find_obs_yaml(self.cfg.map_name, self.cfg.obs_yaml_path)
            _narrow_xy, _gap = None, None
            if _obs_data and _obs_data.get("narrow_regions"):
                _nr        = _obs_data["narrow_regions"][0]
                _narrow_xy = tuple(_nr["center_m"])
                _gap       = float(_nr["gap_width_m"])
            # Zone_half_width based on injected-section spacing (not full-CL average).
            _inj_sp = (_LF_D_UP + _LF_D_DOWN) / max(_LF_N_APP - 1, 1)
            _zone_hw = max(8, int(12.0 / max(_inj_sp, 1e-3)))  # 12 m past narrow centre
            zone = FullRunMetrics(
                centerline_xs=self._ext_xs,   # extended CL with approach wps
                centerline_ys=self._ext_ys,
                dt=self.cfg.dt,
                narrow_center_xy=_narrow_xy,
                gap_width_m=_gap,
                zone_half_width=_zone_hw,
            )
            has_metrics = True
        except ImportError:
            has_metrics = False
            zone = None
        # ─────────────────────────────────────────────────────────────

        traj_history: List[List[np.ndarray]] = [[] for _ in range(n)]
        for i in range(n):
            traj_history[i].append(np.array([float(obs["poses_x"][i]),
                                             float(obs["poses_y"][i])]))

        gaps: List[List[float]] = [[] for _ in range(n - 1)]

        for step in range(self.cfg.max_steps):
            actions = self._compute_actions(obs)
            obs, _, done, _, _ = self.adapter.step(actions)

            if self.cfg.render:
                self.adapter.render()

            if has_metrics:
                zone.step(obs)
                if zone.all_zone_cleared:
                    print(f"[LeaderFollower] All agents cleared the zone at step {step}")
                    break
                if zone.all_done:
                    print(f"[LeaderFollower] All agents reached goal at step {step}")
                    break

            px = obs["poses_x"]
            py = obs["poses_y"]

            for i in range(n):
                traj_history[i].append(np.array([float(px[i]), float(py[i])]))
            for i in range(1, n):
                g = math.hypot(float(px[i] - px[i - 1]), float(py[i] - py[i - 1]))
                gaps[i - 1].append(g)

            if done:
                print(f"[LeaderFollower] Episode ended at step {step}")
                break

        elapsed = time.time() - t0

        gap_means = [float(np.mean(g)) if g else float("nan") for g in gaps]
        gap_stds  = [float(np.std(g))  if g else float("nan") for g in gaps]

        metrics = {
            "elapsed_s":  elapsed,
            "steps":      step + 1,
            "mean_gap_m": gap_means,
            "std_gap_m":  gap_stds,
        }

        if has_metrics:
            zone_metrics = zone.summary()
            metrics.update(zone_metrics)

        print("\n======= LEADER-FOLLOWER METRICS =======")
        print(f"Steps run   : {metrics['steps']}")
        print(f"Elapsed (s) : {elapsed:.1f}")
        for fi, (mu, sd) in enumerate(zip(gap_means, gap_stds)):
            print(f"Car {fi+1}↔Car {fi+2} longitudinal gap: {mu:.3f} ± {sd:.3f} m")
        
        if has_metrics:
            print(f"\n--- Full-Run Metrics ---")
            print(f"Avg speed         : {zone_metrics['avg_speed_mps']:.2f} m/s")
            print(f"Success           : {zone_metrics['success']}")
            print(f"Time to goal (s)  : {zone_metrics['time_to_goal_s']:.3f}")
            print(f"Flow rate         : {zone_metrics['flow_rate']:.4f} agents/(m·s)")
            print(f"Deformability     : {zone_metrics['deformability']}")
            print(f"Spread at entry   : {zone_metrics['spread_at_entry_m']:.3f} m")
            print(f"Spread at narrowest: {zone_metrics['spread_at_narrow_m']:.3f} m")
        print("=======================================\n")

        return metrics

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Leader-follower baseline on f1tenth_gym"
    )
    parser.add_argument("--num-cars",         type=int,   default=4)
    parser.add_argument("--map",              type=str,   default=None)
    parser.add_argument("--leader-speed",     type=float, default=3.0)
    parser.add_argument("--lookahead",        type=float, default=1.5)
    parser.add_argument("--max-steps",        type=int,   default=100000)
    parser.add_argument("--no-render",        action="store_true")
    parser.add_argument("--seed",             type=int,   default=42)
    parser.add_argument("--obs-yaml",         type=str,   default=None,
                        help="Path to obs_pos.yaml (default: auto-detect beside map)")
    parser.add_argument("--hazard-slow-dist", type=float, default=8.0,
                        help="Distance (m) at which to begin slowing near a hazard")
    parser.add_argument("--hazard-slow-speed", type=float, default=2.0,
                        help="Speed (m/s) to use when closest to a hazard centre")
    args = parser.parse_args()

    cfg = LeaderFollowerConfig(
        num_cars=args.num_cars,
        **({"map_name": args.map} if args.map is not None else {}),
        leader_speed=args.leader_speed,
        lookahead_dist=args.lookahead,
        max_steps=args.max_steps,
        render=not args.no_render,
        seed=args.seed,
        obs_yaml_path=args.obs_yaml,
        hazard_slowdown_dist=args.hazard_slow_dist,
        hazard_slow_speed=args.hazard_slow_speed,
    )

    F1TenthLeaderFollower(cfg).run()