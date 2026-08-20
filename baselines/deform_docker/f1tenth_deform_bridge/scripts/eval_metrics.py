"""
FullRunMetrics — collects all 5 paper metrics scoped to the narrow zone.

When agents spawn at the zone entry (episode_end_mode=narrow_clear in the
bridge), the episode ends once every agent has exited the zone past
zone_exit_wp.  All five reported metrics use only data from inside the
narrow window [entry_wp .. exit_wp] (± zone_half_width around the
narrow-region centre from obs_pos.yaml).

Metrics (narrow-zone-scoped)
----------------------------
avg_speed_mps   : mean speed while inside the narrow zone (m/s)
success         : 1.0 if every agent cleared the zone with no in-zone collision
time_to_goal_s  : sim time from first zone entry to last zone exit
flow_rate       : N_zone_cleared / (gap_width_m × time_to_goal_s)
deformability   : lateral spread at zone entry / tightest spread in zone
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np


def _nearest_wp(x: float, y: float, xs: np.ndarray, ys: np.ndarray,
                hint: int, window: int = 40) -> int:
    n = len(xs)
    lo = max(0, hint - window)
    hi = min(n - 1, hint + window)
    sub = np.hypot(xs[lo:hi + 1] - x, ys[lo:hi + 1] - y)
    return int(lo + np.argmin(sub))


def _lateral_spread(positions: np.ndarray, lateral_axis: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.0
    projs = positions @ lateral_axis
    return float(projs.max() - projs.min())


class FullRunMetrics:
    def __init__(
        self,
        centerline_xs: np.ndarray,
        centerline_ys: np.ndarray,
        dt: float = 0.01,
        narrow_center_xy: Optional[Tuple[float, float]] = None,
        gap_width_m: Optional[float] = None,
        goal_xy: Optional[Tuple[float, float]] = None,
        goal_buffer: int = 5,
        zone_half_width: int = 8,
        collision_thresh: float = 0.5,
    ):
        self.xs = np.asarray(centerline_xs, dtype=float)
        self.ys = np.asarray(centerline_ys, dtype=float)
        self.dt = dt
        self.gap_width = gap_width_m or 1.0

        n_wp = len(self.xs)

        if goal_xy is not None and n_wp > 0:
            gx, gy = goal_xy
            gdists = np.hypot(self.xs - gx, self.ys - gy)
            self.goal_thresh = int(np.argmin(gdists))
        else:
            self.goal_thresh = max(0, n_wp - 1 - goal_buffer)

        if narrow_center_xy is not None and len(self.xs) > 0:
            cx, cy = narrow_center_xy
            dists = np.hypot(self.xs - cx, self.ys - cy)
            self._narrow_wp = int(np.argmin(dists))
        else:
            self._narrow_wp = None

        if self._narrow_wp is not None:
            self._zone_entry = max(0, self._narrow_wp - zone_half_width)
            self._zone_exit = min(n_wp - 1, self._narrow_wp + zone_half_width)

            _a = max(self._narrow_wp - 1, 0)
            _b = min(self._narrow_wp + 1, n_wp - 1)
            dx = float(self.xs[_b] - self.xs[_a])
            dy = float(self.ys[_b] - self.ys[_a])
            seg = math.hypot(dx, dy) + 1e-9
            self._lateral_axis = np.array([-dy / seg, dx / seg])
        else:
            self._zone_entry = self._zone_exit = None
            self._lateral_axis = None

        self.collision_thresh = collision_thresh

        self._n_agents: int = 0
        self._initialized: bool = False

        self._wp_hints: List[int] = []
        self._reached_goal: List[bool] = []
        self._goal_step: List[Optional[int]] = []

        self._in_zone: List[bool] = []
        self._zone_entered: List[bool] = []
        self._zone_cleared: List[bool] = []

        self._step: int = 0
        self._collision: bool = False

        self._zone_speed_sum: float = 0.0
        self._zone_speed_n: int = 0
        self._zone_collision: bool = False
        self._first_zone_entry_step: Optional[int] = None
        self._last_zone_exit_step: Optional[int] = None

        self._spread_at_entry: Optional[float] = None
        self._spread_at_narrow: Optional[float] = None

    @property
    def all_done(self) -> bool:
        """True once every agent has reached the map goal waypoint."""
        return self._initialized and all(self._reached_goal)

    @property
    def all_zone_cleared(self) -> bool:
        """True once every agent has exited the narrow zone past zone_exit_wp."""
        return (
            self._initialized
            and self._narrow_wp is not None
            and len(self._zone_cleared) > 0
            and all(self._zone_cleared)
        )

    def _init_agents(self, n: int) -> None:
        self._n_agents = n
        self._wp_hints = [0] * n
        self._reached_goal = [False] * n
        self._goal_step = [None] * n
        self._in_zone = [False] * n
        self._zone_entered = [False] * n
        self._zone_cleared = [False] * n
        self._initialized = True

    def step(self, obs: dict) -> None:
        n = len(obs["poses_x"])
        first_step = not self._initialized
        if first_step:
            self._init_agents(n)
            if len(self.xs) > 0:
                for i in range(n):
                    xi = float(obs["poses_x"][i])
                    yi = float(obs["poses_y"][i])
                    self._wp_hints[i] = int(np.argmin(
                        np.hypot(self.xs - xi, self.ys - yi)
                    ))

        positions = np.column_stack([obs["poses_x"], obs["poses_y"]])
        speeds = np.abs(np.asarray(obs["linear_vels_x"], dtype=float))

        if len(self.xs) > 0 and not first_step:
            for i in range(n):
                self._wp_hints[i] = _nearest_wp(
                    float(obs["poses_x"][i]), float(obs["poses_y"][i]),
                    self.xs, self.ys, self._wp_hints[i],
                )

        any_collision_this_step = False
        for i in range(n):
            for j in range(i + 1, n):
                if np.linalg.norm(positions[i] - positions[j]) < self.collision_thresh:
                    any_collision_this_step = True
                    self._collision = True
                    break
            if any_collision_this_step:
                break

        if len(self.xs) > 0:
            for i in range(n):
                if not self._reached_goal[i] and self._wp_hints[i] >= self.goal_thresh:
                    self._reached_goal[i] = True
                    self._goal_step[i] = self._step

        if self._narrow_wp is not None:
            z_entry = int(self._zone_entry)
            z_exit = int(self._zone_exit)
            lat_ax = self._lateral_axis

            any_in_zone_now = False
            for i in range(n):
                wp = self._wp_hints[i]
                was_in = self._in_zone[i]
                now_in = z_entry <= wp <= z_exit

                if now_in and not was_in:
                    self._in_zone[i] = True
                    self._zone_entered[i] = True
                    if self._first_zone_entry_step is None:
                        self._first_zone_entry_step = self._step
                    if self._spread_at_entry is None:
                        self._spread_at_entry = _lateral_spread(positions, lat_ax)

                if not now_in and was_in:
                    self._in_zone[i] = False
                    if wp > z_exit:
                        self._zone_cleared[i] = True
                        self._last_zone_exit_step = self._step

                if self._in_zone[i]:
                    any_in_zone_now = True
                    self._zone_speed_sum += float(speeds[i])
                    self._zone_speed_n += 1

            if any_collision_this_step and any_in_zone_now:
                self._zone_collision = True

            in_zone_ids = [i for i in range(n) if self._in_zone[i]]
            if in_zone_ids:
                spread = _lateral_spread(positions[in_zone_ids], lat_ax)
                if self._spread_at_narrow is None or spread < self._spread_at_narrow:
                    self._spread_at_narrow = spread

        self._step += 1

    def summary(self) -> dict:
        n = self._n_agents
        n_zone_cleared = sum(self._zone_cleared)
        all_cleared = (n > 0) and all(self._zone_cleared)

        success = 1.0 if (all_cleared and not self._zone_collision) else 0.0

        avg_speed = (
            self._zone_speed_sum / self._zone_speed_n
            if self._zone_speed_n > 0 else 0.0
        )

        if (self._first_zone_entry_step is not None
                and self._last_zone_exit_step is not None
                and all_cleared):
            time_to_goal = (
                self._last_zone_exit_step - self._first_zone_entry_step
            ) * self.dt
        elif self._first_zone_entry_step is not None and not all_cleared:
            time_to_goal = float("inf")
        else:
            time_to_goal = float("nan")

        if (isinstance(time_to_goal, float)
                and time_to_goal > 0
                and time_to_goal != float("inf")
                and not math.isnan(time_to_goal)):
            flow_rate = n_zone_cleared / (self.gap_width * time_to_goal)
        else:
            flow_rate = 0.0

        entry_spread = self._spread_at_entry or 0.0
        narrow_spread = self._spread_at_narrow or 1e-9
        if self._narrow_wp is not None and narrow_spread > 1e-9 and entry_spread > 0:
            deformability = entry_spread / narrow_spread
        else:
            deformability = float("nan")

        return {
            "avg_speed_mps": round(avg_speed, 3),
            "success": success,
            "time_to_goal_s": (
                round(time_to_goal, 3) if not math.isnan(time_to_goal) else float("nan")
            ),
            "flow_rate": round(flow_rate, 4),
            "deformability": (
                round(deformability, 3) if not math.isnan(deformability) else float("nan")
            ),
            "n_agents": n,
            "n_completed": n_zone_cleared,
            "collision": self._zone_collision,
            "collision_full_run": self._collision,
            "spread_at_entry_m": round(entry_spread, 3),
            "spread_at_narrow_m": round(narrow_spread, 3),
            "narrow_wp_idx": self._narrow_wp,
            "zone_entry_wp": self._zone_entry,
            "zone_exit_wp": self._zone_exit,
        }
