#!/usr/bin/env python3
"""
ORCA (Optimal Reciprocal Collision Avoidance) baseline for f1tenth_gym.

Algorithm:
  - Each car computes a preferred velocity toward the next track waypoint.
  - Python-RVO2 (ORCA) adjusts these velocities so that no two cars collide
    within the time horizon.
  - The 2D ORCA velocity is converted to [speed, steering_angle] for the
    non-holonomic f1tenth model via a heading-error P-controller.

Physics: f1tenth_gym (single-track model, RK4 integrator).
Requires: rvo2  (pip install rvo2  or build Python-RVO2)
          f1tenth_gym  (in the FastFunnels environment)
          numpy
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
# Allow importing FastFunnels envs from anywhere
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
class OrcaConfig:
    # ── Scenario ──────────────────────────────────────────────────────────
    num_cars:         int   = 4        # number of f1tenth cars
    map_name:         str   = "open_narrow_obs"
    render:           bool  = True
    max_steps:        int   = 10000
    seed:             int   = 42

    # ── Car dynamics limits ───────────────────────────────────────────────
    target_speed:     float = 8.0      # nominal track-following speed (m/s)
    max_speed:        float = 10.0
    min_speed:        float = 0.5
    steer_max:        float = 0.4189   # ±24°
    steer_k:          float = 1.5      # heading-error → steering gain
    wheelbase:        float = 0.3302   # LF + LR  (m)

    # ── ORCA / RVO2 ───────────────────────────────────────────────────────
    car_radius:       float = 0.25     # collision radius per car (m)
    neighbor_dist:    float = 12.0     # look-out radius for ORCA (m)
    max_neighbors:    int   = 10
    time_horizon:     float = 2.0      # safety horizon for agents (s)
    time_horizon_obs: float = 6.0      # safety horizon for static obstacles (s)

    # ── Spawn ─────────────────────────────────────────────────────────────
    # Cars spawn side-by-side at the track start; lateral_gap between each
    lateral_gap:      float = 0.6      # side-by-side spacing (m)

    # ── Waypoint following ────────────────────────────────────────────────
    lookahead_wp:     float = 1.5      # distance to next waypoint to aim for

    # ── Timestep ──────────────────────────────────────────────────────────
    dt:               float = 0.01

    # ── Static-obstacle map ───────────────────────────────────────────────
    obs_yaml_path:    Optional[str] = None  # None = auto-detect from map_name


# ---------------------------------------------------------------------------
# ORCA wrapper (thin layer over Python-RVO2)
# ---------------------------------------------------------------------------

class ORCAWrapper:
    """Wraps a Python-RVO2 simulation for N ground agents."""

    def __init__(self, positions: np.ndarray, max_speed: float, cfg: OrcaConfig,
                 obstacle_polygons: Optional[List[List[Tuple[float, float]]]] = None):
        try:
            import rvo2
        except ImportError as exc:
            raise ImportError(
                "rvo2 not found. Install with:  pip install rvo2\n"
                "or build from: https://github.com/sybrenstuvel/Python-RVO2"
            ) from exc

        self.n = positions.shape[0]
        self.sim = rvo2.PyRVOSimulator(
            cfg.dt,
            cfg.neighbor_dist,
            cfg.max_neighbors,
            cfg.time_horizon,
            cfg.time_horizon_obs,
            cfg.car_radius,
            max_speed,
        )
        self.agent_ids = []
        for i in range(self.n):
            aid = self.sim.addAgent(
                tuple(positions[i]),
                cfg.neighbor_dist,
                cfg.max_neighbors,
                cfg.time_horizon,
                cfg.time_horizon_obs,
                cfg.car_radius,
                max_speed,
                (0.0, 0.0),
            )
            self.agent_ids.append(aid)

        # Register static obstacle polygons then build the k-d tree exactly once
        for poly in (obstacle_polygons or []):
            self.sim.addObstacle(poly)
        self.sim.processObstacles()
        if obstacle_polygons:
            print(f"[ORCA] Registered {len(obstacle_polygons)} static obstacle polygon(s) in RVO2.")

    def set_pref_vels(self, v_pref: np.ndarray) -> None:
        for i, aid in enumerate(self.agent_ids):
            v = np.asarray(v_pref[i], float)
            spd = np.linalg.norm(v)
            vmax = self.sim.getAgentMaxSpeed(aid)
            if spd > vmax:
                v = v * (vmax / spd)
            self.sim.setAgentPrefVelocity(aid, (float(v[0]), float(v[1])))

    def step(self) -> None:
        self.sim.doStep()

    def get_velocities(self) -> np.ndarray:
        V = np.zeros((self.n, 2))
        for i, aid in enumerate(self.agent_ids):
            V[i] = self.sim.getAgentVelocity(aid)
        return V

    def sync_positions(self, positions: np.ndarray, velocities: np.ndarray) -> None:
        """Sync ORCA internal state from f1tenth ground truth each step."""
        for i, aid in enumerate(self.agent_ids):
            self.sim.setAgentPosition(aid, (float(positions[i, 0]), float(positions[i, 1])))
            self.sim.setAgentVelocity(aid, (float(velocities[i, 0]), float(velocities[i, 1])))


# ---------------------------------------------------------------------------
# Control helpers
# ---------------------------------------------------------------------------

def _wrap_angle(a: float) -> float:
    return float((a + math.pi) % (2.0 * math.pi) - math.pi)


def vel_to_action(vx: float, vy: float, theta: float, cfg: OrcaConfig) -> Tuple[float, float]:
    """Convert a desired 2-D velocity to [speed, steering_angle] for f1tenth."""
    speed = float(np.clip(math.hypot(vx, vy), cfg.min_speed, cfg.max_speed))
    if math.hypot(vx, vy) < 1e-6:
        return speed, 0.0
    heading_err = _wrap_angle(math.atan2(vy, vx) - theta)
    steering = float(np.clip(cfg.steer_k * heading_err, -cfg.steer_max, cfg.steer_max))
    return speed, steering


def _is_closed_loop(xs: np.ndarray, ys: np.ndarray, threshold: float = 5.0) -> bool:
    return math.hypot(float(xs[-1] - xs[0]), float(ys[-1] - ys[0])) < threshold


def _nearest_wp_idx(x: float, y: float, xs: np.ndarray, ys: np.ndarray,
                    hint: int, search_window: int = 30) -> int:
    """Find the waypoint closest to (x, y), searching within ±search_window of hint."""
    n = len(xs)
    best_idx  = hint
    best_dist = math.hypot(xs[hint] - x, ys[hint] - y)
    lo = max(0, hint - search_window)
    hi = min(n - 1, hint + search_window)
    for k in range(lo, hi + 1):
        d = math.hypot(xs[k] - x, ys[k] - y)
        if d < best_dist:
            best_dist = d
            best_idx  = k
    return best_idx

def _next_wp_idx(x: float, y: float, xs: np.ndarray, ys: np.ndarray,
                 current_idx: int, lookahead: float,
                 closed: bool = True) -> int:
    n = len(xs)
    # Bug 2 fix: for open paths near the end, lock to the last waypoint.
    # A global search here lets a folded track snap the car backward.
    if not closed and current_idx >= n - 1:
        return n - 1
    idx = _nearest_wp_idx(x, y, xs, ys, current_idx, search_window=20)
    for _ in range(n):
        if math.hypot(xs[idx] - x, ys[idx] - y) >= lookahead:
            return idx
        next_idx = (idx + 1) % n if closed else min(idx + 1, n - 1)
        if not closed and next_idx == idx:
            return idx
        idx = next_idx
    return idx
# def _next_wp_idx(x: float, y: float, xs: np.ndarray, ys: np.ndarray,
#                  current_idx: int, lookahead: float,
#                  closed: bool = True) -> int:
#     n    = len(xs)
#     # Near end of open path use global search so a stale hint can't lock the index.
#     sw   = n if (not closed and current_idx >= n - 25) else 20
#     idx  = _nearest_wp_idx(x, y, xs, ys, current_idx, search_window=sw)
#     for _ in range(n):
#         if math.hypot(xs[idx] - x, ys[idx] - y) >= lookahead:
#             return idx
#         next_idx = (idx + 1) % n if closed else min(idx + 1, n - 1)
#         if not closed and next_idx == idx:
#             return idx          # at last wp — stay here, never reverse
#         idx = next_idx
#     return idx
# ── Narrow-zone approach waypoint injection ──────────────────────────────────
# The open_narrow_obs f1tenth_gym centerline has waypoints 364-378 embedded
# inside the track wall (x≈37-38, y≈-27 to -34).  Agents aiming for these
# drive in the wrong direction.  _narrow_approach_wps() splices in a
# straight-south sequence through the actual free corridor so that both
# ORCA/LF control and FullRunMetrics zone detection work correctly.
_D_NARROW_UP   = 7.0   # m north of zone centre to spawn
_D_NARROW_DOWN = 20.0  # m south of zone centre (past physical corridor exit ≈12 m)
_N_APPROACH    = 46    # synthetic waypoints (≈0.59 m spacing for the 27 m span)


def _narrow_approach_wps(
    xs: np.ndarray, ys: np.ndarray, narrow_xy: tuple
) -> tuple:
    """Return (xs_ext, ys_ext, spawn_x, spawn_y, theta_fwd, idx0)."""
    nx, ny  = float(narrow_xy[0]), float(narrow_xy[1])
    ins_at  = int(np.argmin(np.hypot(xs - nx, ys - ny)))   # ≈ wp[378]
    spawn_x = nx
    spawn_y = ny + _D_NARROW_UP
    exit_y  = ny - _D_NARROW_DOWN
    app_xs  = np.full(_N_APPROACH, nx, dtype=np.float32)
    app_ys  = np.linspace(spawn_y, exit_y, _N_APPROACH, dtype=np.float32)
    xs_ext  = np.concatenate([xs[:ins_at], app_xs, xs[ins_at:]])
    ys_ext  = np.concatenate([ys[:ins_at], app_ys, ys[ins_at:]])
    return xs_ext, ys_ext, spawn_x, spawn_y, -math.pi / 2, ins_at


def _safe_tangent(xs_wp, ys_wp, idx, closed, nwp):
    """Return a stable forward tangent vector at waypoint idx."""
    if closed:
        nxt = (idx + 1) % nwp
        prv = (idx - 1) % nwp
    else:
        nxt = min(idx + 1, nwp - 1)
        prv = max(idx - 1, 0)

    if nxt != idx:
        tx = float(xs_wp[nxt]) - float(xs_wp[idx])
        ty = float(ys_wp[nxt]) - float(ys_wp[idx])
    else:
        # At open-path terminal: use backward segment, keep direction forward
        tx = float(xs_wp[idx]) - float(xs_wp[prv])
        ty = float(ys_wp[idx]) - float(ys_wp[prv])

    seg = math.hypot(tx, ty)
    if seg < 1e-6:                  # degenerate — fall back to prev segment
        tx = float(xs_wp[idx]) - float(xs_wp[prv])
        ty = float(ys_wp[idx]) - float(ys_wp[prv])
        seg = math.hypot(tx, ty) + 1e-9

    return tx / seg, ty / seg       # unit tangent

# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def _path_length(traj: List[np.ndarray]) -> float:
    P = np.array(traj)
    return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) >= 2 else 0.0


def _fleet_min_separation(traj_list: List[List[np.ndarray]]) -> float:
    n = len(traj_list)
    min_sep = np.inf
    T = min(len(t) for t in traj_list)
    for t in range(T):
        pts = np.array([traj_list[i][t] for i in range(n)])
        for i in range(n):
            for j in range(i + 1, n):
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if d < min_sep:
                    min_sep = d
    return float(min_sep)


def _path_smoothness(traj: List[np.ndarray]) -> float:
    P = np.array(traj)
    if len(P) < 3:
        return 0.0
    seg = np.diff(P, axis=0)
    ds  = np.linalg.norm(seg, axis=1)
    dtheta = np.abs(np.diff(np.unwrap(np.arctan2(seg[:, 1], seg[:, 0]))))
    L = float(np.sum(ds[:-1]))
    return float(np.sum(dtheta) / L) if L > 1e-9 else 0.0


# ---------------------------------------------------------------------------
# Static-obstacle helpers  (obs_pos.yaml → RVO2 polygon obstacles)
# ---------------------------------------------------------------------------

_MAP_SEARCH_DIRS: List[str] = [
    os.path.join(_HERE, "..", "f1tenth_gym_latest", "f1tenth_gym", "maps"),
    os.path.join(os.path.expanduser("~"), "f1tenth_gym", "maps"),
    os.path.join(_HERE, "..", "f1tenth_gym", "maps"),
]


def _find_obs_yaml(map_name: str, explicit: Optional[str]) -> Optional[dict]:
    """Load obs_pos.yaml for the current map via direct path search."""
    if explicit:
        candidates = [explicit]
    else:
        candidates = [
            os.path.join(base, map_name, f"{map_name}_obs_pos.yaml")
            for base in _MAP_SEARCH_DIRS
        ]
    for path in candidates:
        if os.path.isfile(path):
            with open(path) as f:
                data = yaml.safe_load(f)
            print(f"[ORCA] Loaded obs_pos.yaml: {path}")
            return data
    print(f"[ORCA] No obs_pos.yaml found for '{map_name}' — running without static obstacles.")
    return None


def _circle_verts(cx: float, cy: float, r: float, n: int = 8) -> List[Tuple[float, float]]:
    """n-gon approximation of a filled circle (CCW)."""
    return [
        (cx + r * math.cos(2.0 * math.pi * k / n),
         cy + r * math.sin(2.0 * math.pi * k / n))
        for k in range(n)
    ]


def _rect_verts(cx: float, cy: float, w: float, h: float,
                angle_deg: float) -> List[Tuple[float, float]]:
    """CCW rectangle corners rotated by angle_deg around centre."""
    ang = math.radians(angle_deg)
    ca, sa = math.cos(ang), math.sin(ang)
    hw, hh = w / 2.0, h / 2.0
    local = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return [(cx + ca * dx - sa * dy, cy + sa * dx + ca * dy) for dx, dy in local]


def _build_obstacle_polygons(obs_data: dict) -> List[List[Tuple[float, float]]]:
    """
    Convert toll_pillars and obstacles from obs_pos.yaml into CCW polygon
    vertex lists ready for sim.addObstacle().  Narrow regions are skipped
    (they are wall geometry already baked into the map image).
    """
    polys: List[List[Tuple[float, float]]] = []
    for pillar in obs_data.get("toll_pillars") or []:
        cx, cy = pillar["center_m"]
        polys.append(_rect_verts(cx, cy,
                                 pillar["width_m"], pillar["depth_m"],
                                 pillar["angle_deg"]))
    for obs in obs_data.get("obstacles") or []:
        cx, cy = obs["center_m"]
        if obs.get("type") == "circle":
            polys.append(_circle_verts(cx, cy, obs["radius_m"]))
        else:
            polys.append(_rect_verts(cx, cy,
                                     obs["width_m"], obs["height_m"],
                                     obs.get("angle_deg", 0.0)))
    return polys


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

class F1TenthORCARunner:
    """ORCA multi-agent collision avoidance on f1tenth_gym."""

    def __init__(self, cfg: OrcaConfig):
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

        self.waypoints_x: np.ndarray = np.empty(0, dtype=np.float32)
        self.waypoints_y: np.ndarray = np.empty(0, dtype=np.float32)
        self.track_closed: bool = True
        self.orca: Optional[ORCAWrapper] = None
        self.wp_indices: List[int] = [0] * cfg.num_cars
        self.lane_offsets: List[float] = [0.0] * cfg.num_cars  # lateral m from centerline per car

    def _init_episode(self) -> dict:
        self.adapter.ensure_initialized()

        track, _, _, _ = self.adapter.get_track_data()
        xs = np.asarray(track.centerline.xs, dtype=np.float32)
        ys = np.asarray(track.centerline.ys, dtype=np.float32)
        self.waypoints_x = xs
        self.waypoints_y = ys
        self.track_closed = _is_closed_loop(xs, ys)
        print(f"[ORCA] Track is {'closed loop' if self.track_closed else 'open path'} "
              f"({len(xs)} waypoints)")

        # Load obs_pos.yaml now — needed for both spawn and obstacle registration.
        obs_data = _find_obs_yaml(self.cfg.map_name, self.cfg.obs_yaml_path)

        # Spawn all cars side-by-side at the narrow-zone approach.
        # The original centerline waypoints 364-378 are embedded inside track
        # walls, so we inject synthetic approach waypoints and spawn from the
        # zone-centre geometry directly.
        n      = self.cfg.num_cars
        idx0   = 0
        spawn_x, spawn_y = float(xs[0]), float(ys[0])   # fallback
        theta0 = 0.0

        if obs_data and obs_data.get("narrow_regions"):
            _nr = obs_data["narrow_regions"][0]
            _narrow_xy = (float(_nr["center_m"][0]), float(_nr["center_m"][1]))
            xs, ys, spawn_x, spawn_y, theta0, idx0 = \
                _narrow_approach_wps(xs, ys, _narrow_xy)
            self.waypoints_x = xs
            self.waypoints_y = ys
            print(f"[ORCA] Spawning at ({spawn_x:.2f}, {spawn_y:.2f}), "
                  f"hdg={math.degrees(theta0):.0f}° "
                  f"(extended CL: {len(xs)} wps, approach idx={idx0})")

        # Side-by-side: perpendicular to heading = eastward when θ=-π/2
        perp_x = -math.sin(theta0)
        perp_y =  math.cos(theta0)
        half   = (n - 1) / 2.0
        poses  = []
        for i in range(n):
            offset = (i - half) * self.cfg.lateral_gap
            poses.append([spawn_x + offset * perp_x,
                          spawn_y + offset * perp_y,
                          theta0])
        poses_np = np.array(poses, dtype=np.float32)

        obs, _ = self.adapter.reset(poses=poses_np)

        self.lane_offsets = [(i - half) * self.cfg.lateral_gap for i in range(n)]
        self.wp_indices = [idx0] * n
        obstacle_polygons: List[List[Tuple[float, float]]] = []
        if obs_data:
            obstacle_polygons = _build_obstacle_polygons(obs_data)

        positions = np.column_stack([obs["poses_x"], obs["poses_y"]])
        self.orca = ORCAWrapper(positions, self.cfg.max_speed, self.cfg,
                                obstacle_polygons=obstacle_polygons)

        return obs

    def _compute_actions(self, obs: dict) -> np.ndarray:
        assert self.orca is not None
        n     = self.cfg.num_cars
        xs_wp = self.waypoints_x
        ys_wp = self.waypoints_y
        nwp   = len(xs_wp)

        car_x  = obs["poses_x"]
        car_y  = obs["poses_y"]
        theta  = obs["poses_theta"]
        vx     = obs["linear_vels_x"]

        # Sync ORCA with gym ground truth (body frame → world frame)
        positions  = np.column_stack([car_x, car_y])
        velocities = np.zeros((n, 2))
        for i in range(n):
            th = float(theta[i])
            spd = float(vx[i])
            velocities[i] = [spd * math.cos(th), spd * math.sin(th)]
        self.orca.sync_positions(positions, velocities)

        # Preferred velocity toward each car's offset lane waypoint
        v_pref = np.zeros((n, 2))
        tgt_x  = np.zeros(n)
        tgt_y  = np.zeros(n)
        for i in range(n):
            # Advance waypoint index along the centerline
            new_idx = _next_wp_idx(
                float(car_x[i]), float(car_y[i]),
                xs_wp, ys_wp, self.wp_indices[i],
                self.cfg.lookahead_wp,
                closed=self.track_closed,
            )
            self.wp_indices[i] = new_idx

            # Perpendicular direction (left of tangent) at this waypoint.
            # Tangent at current wp — always computed as (next - current) direction
            # so perp_x/perp_y never flip sign between normal and terminal wps.
            if self.track_closed:
                _a = new_idx
                _b = (new_idx + 1) % nwp
            elif new_idx < nwp - 1:
                _a = new_idx
                _b = new_idx + 1
            else:                         # last wp of open path: use prev segment
                _a = max(new_idx - 1, 0)
                _b = new_idx
            # tx  = float(xs_wp[_b]) - float(xs_wp[_a])
            # ty  = float(ys_wp[_b]) - float(ys_wp[_a])
            # seg = max(math.hypot(tx, ty), 1e-3)   # guard zero-length segment
            # perp_x = -ty / seg
            # perp_y =  tx / seg
            tx, ty = _safe_tangent(xs_wp, ys_wp, new_idx, self.track_closed, nwp)
            perp_x = -ty
            perp_y =  tx
            # Lane waypoint: centerline point + car's fixed lateral offset
            off = self.lane_offsets[i]
            lx  = float(xs_wp[new_idx]) + off * perp_x
            ly  = float(ys_wp[new_idx]) + off * perp_y
            tgt_x[i] = lx
            tgt_y[i] = ly

            # Bug 1 fix: at the last waypoint of an open track, stop the car.
            # Without this the target vector flips backward and the car swerves.
            if not self.track_closed and new_idx >= nwp - 1:
                v_pref[i] = [0.0, 0.0]
                continue

            dx = lx - float(car_x[i])
            dy = ly - float(car_y[i])
            d  = math.hypot(dx, dy) + 1e-9

            # Bug 3 fix: skip curvature scan in the last 20 wps of an open path.
            # Clamped indices collapse to the same point → zero vectors → atan2(0,0).
            near_end = (not self.track_closed and new_idx >= nwp - 20)
            if near_end:
                curve_speed = self.cfg.target_speed
            else:
                h1_base  = math.atan2(ty, tx)
                max_curv = 0.0
                for _k in range(20):
                    a_idx = new_idx + _k
                    b_idx = new_idx + _k + 3
                    if self.track_closed:
                        a_idx %= nwp
                        b_idx %= nwp
                    else:
                        a_idx = min(a_idx, nwp - 1)
                        b_idx = min(b_idx, nwp - 1)
                    if a_idx == b_idx:
                        break
                    _tx = float(xs_wp[b_idx]) - float(xs_wp[a_idx])
                    _ty = float(ys_wp[b_idx]) - float(ys_wp[a_idx])
                    if math.hypot(_tx, _ty) < 1e-6:   # degenerate segment
                        break
                    if _k == 0:
                        _h1 = h1_base
                    else:
                        _pa  = new_idx + _k - 1
                        _pa  = (_pa % nwp) if self.track_closed else min(_pa, nwp - 1)
                        _ptx = float(xs_wp[a_idx]) - float(xs_wp[_pa])
                        _pty = float(ys_wp[a_idx]) - float(ys_wp[_pa])
                        if math.hypot(_ptx, _pty) < 1e-6:
                            break
                        _h1 = math.atan2(_pty, _ptx)
                    _h2 = math.atan2(_ty, _tx)
                    _dh = abs(_wrap_angle(_h2 - _h1))
                    if _dh > math.pi / 2:   # discontinuity (injected wp boundary)
                        break
                    max_curv = max(max_curv, _dh)
                curve_speed = self.cfg.target_speed / (1.0 + 6.0 * max_curv)
                curve_speed = max(curve_speed, self.cfg.min_speed)

            v_pref[i] = [curve_speed * dx / d, curve_speed * dy / d]

        # ORCA → collision-free velocities
        self.orca.set_pref_vels(v_pref)
        self.orca.step()
        v_orca = self.orca.get_velocities()

        # Non-holonomic action:
        #   Speed    — ORCA magnitude (encodes avoidance via slowdown)
        #   Steering — ORCA velocity direction when avoidance is active;
        #              lane-waypoint direction otherwise.
        #
        # This is critical for static obstacle avoidance: ORCA's direction
        # tells the car to steer AROUND the obstacle. Using only speed would
        # make the car slow down and still drive straight into it.
        actions = np.zeros((n, 2), dtype=np.float32)
        for i in range(n):
            orca_spd = float(np.linalg.norm(v_orca[i]))
            speed = float(np.clip(orca_spd, self.cfg.min_speed, self.cfg.max_speed))

            # Steering via pure-pursuit geometry on the ORCA velocity direction.
            # Using steer_k * heading_err over-steers by ~3x on curves; the
            # atan2(2·WB·sin(α), L) formula matches the car's turn radius correctly.
            if orca_spd > 0.1:
                steer_heading = math.atan2(float(v_orca[i, 1]), float(v_orca[i, 0]))
                alpha = _wrap_angle(steer_heading - float(theta[i]))
                steer = math.atan2(
                    2.0 * self.cfg.wheelbase * math.sin(alpha),
                    self.cfg.lookahead_wp,
                )
            else:
                steer = 0.0
            steer = float(np.clip(steer, -self.cfg.steer_max, self.cfg.steer_max))

            actions[i] = [steer, speed]

        return actions

    def run(self) -> dict:
        obs  = self._init_episode()
        n    = self.cfg.num_cars
        t0   = time.time()
        step = 0

        # ── Full-run metrics collector ───────────────────────────────
        from eval_metrics import FullRunMetrics      # noqa: PLC0415
        _obs_data = _find_obs_yaml(self.cfg.map_name, self.cfg.obs_yaml_path)
        _narrow_xy, _gap = None, None
        if _obs_data and _obs_data.get("narrow_regions"):
            _nr     = _obs_data["narrow_regions"][0]
            _narrow_xy = tuple(_nr["center_m"])
            _gap    = float(_nr["gap_width_m"])
        # Zone_half_width based on injected-section spacing (not full-CL average,
        # which is dominated by original track ≈0.38 m/wp and distorts the result).
        # Injected span = _D_NARROW_UP + _D_NARROW_DOWN over _N_APPROACH wps.
        _inj_sp = (_D_NARROW_UP + _D_NARROW_DOWN) / max(_N_APPROACH - 1, 1)
        _zone_hw = max(8, int(12.0 / max(_inj_sp, 1e-3)))  # 12 m past narrow centre
        zone = FullRunMetrics(
            centerline_xs=self.waypoints_x,   # extended CL with approach wps
            centerline_ys=self.waypoints_y,
            dt=self.cfg.dt,
            narrow_center_xy=_narrow_xy,
            gap_width_m=_gap,
            zone_half_width=_zone_hw,
        )
        # ─────────────────────────────────────────────────────────────

        traj_history: List[List[np.ndarray]] = [[] for _ in range(n)]
        for i in range(n):
            traj_history[i].append(np.array([float(obs["poses_x"][i]),
                                             float(obs["poses_y"][i])]))

        for step in range(self.cfg.max_steps):
            actions = self._compute_actions(obs)
            try:
                obs, _, done, _, _ = self.adapter.step(actions)
            except Exception as e:
                print(f"[ORCA] Sim error at step {step}: {e}")
                break

            if self.cfg.render:
                try:
                    self.adapter.render()
                except Exception:
                    pass

            zone.step(obs)

            for i in range(n):
                traj_history[i].append(np.array([float(obs["poses_x"][i]),
                                                 float(obs["poses_y"][i])]))

            if zone.all_zone_cleared:
                print(f"[ORCA] All agents cleared the zone at step {step}")
                break
            if zone.all_done:
                print(f"[ORCA] All agents reached goal at step {step}")
                break
            if done:
                print(f"[ORCA] Episode ended at step {step}")
                break

        elapsed   = time.time() - t0
        min_sep   = _fleet_min_separation(traj_history)
        path_lens = [_path_length(t) for t in traj_history]
        smoothness = [_path_smoothness(t) for t in traj_history]

        zone_metrics = zone.summary()

        metrics = {
            "elapsed_s":        elapsed,
            "steps":            step + 1,
            "min_separation_m": min_sep,
            "avg_path_length_m": float(np.mean(path_lens)),
            "avg_smoothness":   float(np.mean(smoothness)),
            **zone_metrics,
        }

        print("\n========== ORCA METRICS ==========")
        print(f"Steps run         : {metrics['steps']}")
        print(f"Elapsed (s)       : {elapsed:.1f}")
        print(f"Min inter-car sep : {min_sep:.3f} m")
        print(f"Avg path length   : {metrics['avg_path_length_m']:.2f} m")
        print(f"Avg smoothness    : {metrics['avg_smoothness']:.4f} rad/m")
        print(f"\n--- Full-Run Metrics ---")
        print(f"Avg speed         : {zone_metrics['avg_speed_mps']:.2f} m/s")
        print(f"Success           : {zone_metrics['success']}")
        print(f"Time to goal (s)  : {zone_metrics['time_to_goal_s']:.3f}")
        print(f"Flow rate         : {zone_metrics['flow_rate']:.4f} agents/(m·s)")
        print(f"Deformability     : {zone_metrics['deformability']}")
        print(f"Spread at entry   : {zone_metrics['spread_at_entry_m']:.3f} m")
        print(f"Spread at narrowest: {zone_metrics['spread_at_narrow_m']:.3f} m")
        print("===================================\n")

        return metrics


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ORCA baseline on f1tenth_gym")
    parser.add_argument("--num-cars",     type=int,   default=4)
    parser.add_argument("--map",          type=str,   default=None)
    parser.add_argument("--target-speed", type=float, default=8.0)
    parser.add_argument("--lateral-gap", type=float, default=0.6)
    parser.add_argument("--max-steps",    type=int,   default=100000)
    parser.add_argument("--no-render",    action="store_true")
    parser.add_argument("--seed",         type=int,   default=42)
    parser.add_argument("--obs-yaml",     type=str,   default=None,
                        help="Path to obs_pos.yaml (default: auto-detect beside map)")
    args = parser.parse_args()

    cfg = OrcaConfig(
        num_cars=args.num_cars,
        **({"map_name": args.map} if args.map is not None else {}),
        target_speed=args.target_speed,
        lateral_gap=args.lateral_gap,
        max_steps=args.max_steps,
        render=not args.no_render,
        seed=args.seed,
        obs_yaml_path=args.obs_yaml,
    )

    F1TenthORCARunner(cfg).run()
