"""Shared plumbing for the controller x simulator cross-over study.

Four combinations are evaluated on the same 100 layouts:

    controller  simulator      dynamics           collision
    ----------  -------------  -----------------  ---------------------------
    ORCA        f1tenth_gym    single-track       gym scan-based  (native)
    ORCA        PyRoboSim      single-track       room polygon    (this study)
    GCBF+       PyRoboSim      Dubins             room polygon    (native)
    GCBF+       f1tenth_gym    single-track       gym scan-based  (this study)

This module holds what the two new runners share: the ORCA control law lifted
out of ``baselines/orca.py`` without its gym adapter, the f1tenth env built
without the repo's torch-dependent wrapper, and the common scoring path.
"""

from __future__ import annotations

import os
import sys
import types

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# `baselines/` first so `eval_metrics` resolves to FullRunMetrics.  The repo's
# bare `f1tenth_gym/` directory is deliberately NOT added: it shadows the
# installed package (which is the tree the funnel and ORCA already run on) as a
# namespace package whose track loader cannot read these maps.
for _p in (_ROOT, os.path.join(_ROOT, "baselines")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def import_orca():
    """Import ``baselines/orca.py`` without dragging in the training stack.

    ``orca.py`` imports ``envs.f110_env`` at module load purely to build its
    own gym adapter, and that chain pulls in torch.  The cross-over runs drive
    the dynamics themselves, so the adapter is stubbed out; every ORCA symbol
    that actually matters (the RVO2 wrapper, the approach-waypoint injection,
    the control law) is the real one.
    """
    if "envs.f110_env" not in sys.modules:
        try:
            import envs.f110_env  # noqa: F401
        except Exception:
            pkg = sys.modules.setdefault("envs", types.ModuleType("envs"))
            pkg.__path__ = [os.path.join(_ROOT, "envs")]  # type: ignore[attr-defined]
            stub = types.ModuleType("envs.f110_env")
            stub.F110Config = object          # type: ignore[attr-defined]
            stub.F110EnvAdapter = object      # type: ignore[attr-defined]
            sys.modules["envs.f110_env"] = stub
    import orca
    return orca


def make_f110_env(map_name: str, num_agents: int, timestep: float = 0.01):
    """Build the bare f1tenth_gym env with the single-track model.

    Same config the repo's ``F110EnvAdapter`` produces, constructed directly so
    the cross-over does not need the torch-dependent wrapper.
    """
    import gymnasium as gym
    import f1tenth_gym  # noqa: F401

    return gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "map": map_name,
            "num_agents": num_agents,
            "timestep": timestep,
            "integrator": "rk4",
            "control_input": ["speed", "steering_angle"],
            "model": "st",
            "observation_config": {"type": "original"},
            "params": {"mu": 1.0},
            "reset_config": {"type": None},
        },
        render_mode=None,
    )


def patch_track_lookup(roots: list[str]) -> None:
    """Let f1tenth_gym resolve the eval-map variants by name."""
    import pathlib

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


class OrcaBrain:
    """ORCA's control law, detached from its gym runner.

    Builds a ``F1TenthORCARunner`` shell without running ``__init__`` (which
    would construct a gym adapter) and hands it the state it needs, so
    ``_compute_actions`` -- the real lane-following + RVO2 + pure-pursuit
    controller -- can be called against any simulator's observation dict.
    """

    def __init__(self, cfg, waypoints_x, waypoints_y, spawn_poses,
                 obstacle_polygons, track_closed: bool = False,
                 start_idx: int = 0):
        orca = import_orca()
        self._runner = orca.F1TenthORCARunner.__new__(orca.F1TenthORCARunner)
        r = self._runner
        r.cfg = cfg
        r.waypoints_x = np.asarray(waypoints_x, dtype=np.float32)
        r.waypoints_y = np.asarray(waypoints_y, dtype=np.float32)
        r.track_closed = track_closed
        n = cfg.num_cars
        half = (n - 1) / 2.0
        r.lane_offsets = [(i - half) * cfg.lateral_gap for i in range(n)]
        r.wp_indices = [start_idx] * n
        r.orca = orca.ORCAWrapper(
            np.asarray(spawn_poses)[:, :2], cfg.max_speed, cfg,
            obstacle_polygons=obstacle_polygons,
        )

    def actions(self, obs: dict) -> np.ndarray:
        """``(n, 2)`` of (steering angle, speed) for the given observation."""
        return self._runner._compute_actions(obs)


_TRACK_CACHE: dict = {}


def orca_waypoints(map_name: str, narrow_xy, n_agents: int, lateral_gap: float):
    """ORCA's waypoint list and rank spawn for one map.

    Uses the *real* spliced centreline: f1tenth_gym's dense track centreline
    with ``orca._narrow_approach_wps``'s straight run through the pinch spliced
    in.  Feeding the PyRoboSim port only the injected segment would quietly make
    the task easier -- the cars would never rejoin the track waypoints that sit
    inside the walls past the zone, which is part of what the gym version has to
    survive.

    Returns ``(xs, ys, spawn_poses, start_idx)``.
    """
    orca = import_orca()
    if map_name not in _TRACK_CACHE:
        env = make_f110_env(map_name, 1, 0.01)
        track = env.unwrapped.track
        _TRACK_CACHE[map_name] = (np.asarray(track.centerline.xs, dtype=np.float32),
                                  np.asarray(track.centerline.ys, dtype=np.float32))
        env.close()
    xs, ys = _TRACK_CACHE[map_name]

    ex, ey, sx, sy, theta0, idx0 = orca._narrow_approach_wps(
        xs, ys, (float(narrow_xy[0]), float(narrow_xy[1])))

    perp_x, perp_y = -np.sin(theta0), np.cos(theta0)
    half = (n_agents - 1) / 2.0
    spawn = np.stack([
        [sx + (i - half) * lateral_gap * perp_x,
         sy + (i - half) * lateral_gap * perp_y,
         theta0]
        for i in range(n_agents)
    ])
    return ex, ey, spawn, idx0
