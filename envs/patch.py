"""
Patch dynamics model.

Contains the deformable patch state and geometric helpers used by the
higher-level environment.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Tuple

import numpy as np


@dataclass
class PatchDynamicsConfig:
    wheelbase: float = 0.5
    v_min: float = 0.5
    v_max: float = 10.0

    a_min = 1.5
    a_max = 2

    b_min = 1.0
    b_max = 1.5
    # accel_max: float = 4.0
    steering_max: float = 0.5
    size_change_rate: float = 1.0


class DynamicPatch:
    """
    Deformable ellipsoid patch with bicycle-model dynamics.

    State: [x, y, theta, v]
    Control: [speed, steering]
    Shape: [a, b]
    """

    def __init__(
        self,
        x: float = 0.0,
        y: float = 0.0,
        theta: float = 0.0,
        v: float = 2.0,
        a: float = 3.0,
        b: float = 2.5,
        config: PatchDynamicsConfig | None = None,
    ) -> None:
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v

        self.a = a
        self.b = b

        self.speed = 0.0
        self.steering = 0.0

        self.config = config or PatchDynamicsConfig()
        self._update_rotation()
        self.history: Deque[Dict[str, float]] = deque(maxlen=100)

    def _update_rotation(self) -> None:
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)

    def sync_from_pose(
        self,
        x: float,
        y: float,
        theta: float,
        v: float,
        steering: float = 0.0,
    ) -> None:
        """Overwrite pose/velocity from an external simulator (e.g. f110). Skips bicycle integration."""
        self.x = float(x)
        self.y = float(y)
        self.theta = float(theta)
        self.v = float(v)
        self.steering = float(steering)
        while self.theta > np.pi:
            self.theta -= 2 * np.pi
        while self.theta < -np.pi:
            self.theta += 2 * np.pi
        self._update_rotation()

    def step(self, speed: float, steering: float, dt: float = 0.05) -> None:
        speed = float(np.clip(speed, self.config.v_min, self.config.v_max))
        steering = float(np.clip(steering, -self.config.steering_max, self.config.steering_max))

        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += (self.v / self.config.wheelbase) * np.tan(steering) * dt
        # First-order lag: τ=0.5s time constant — matches f110 acceleration envelope.
        # Prevents instant jumps from 0→10 m/s that agents with real inertia cannot follow.
        tau = 0.5
        alpha = dt / (tau + dt)   # ≈ 0.09 → ~90% of way to target in 1 second
        self.v = self.v + alpha * (speed - self.v)

        
        # self.v = float(np.clip(self.v, self.config.v_min, self.config.v_max))

        while self.theta > np.pi:
            self.theta -= 2 * np.pi
        while self.theta < -np.pi:
            self.theta += 2 * np.pi

        self._update_rotation()

    def update_shape(
        self,
        a: float,
        b: float,
        dt: float = 0.05,
        max_a: float | None = None,
        max_b: float | None = None,
    ) -> None:
        max_change = self.config.size_change_rate * dt
        if max_a is not None:
            a = min(a, max_a)
        if max_b is not None:
            b = min(b, max_b)

        self.a += float(np.clip(a - self.a, -max_change, max_change))
        self.b += float(np.clip(b - self.b, -max_change, max_change))

    def is_inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        dx = x - self.x
        dy = y - self.y
        x_rot = dx * self.cos_t + dy * self.sin_t
        y_rot = -dx * self.sin_t + dy * self.cos_t

        a_eff = max(self.a - margin, 0.5)
        b_eff = max(self.b - margin, 0.5)
        return bool((x_rot / a_eff) ** 2 + (y_rot / b_eff) ** 2 <= 1.0)

    def world_to_patch_frame(self, x: float, y: float) -> Tuple[float, float]:
        dx = x - self.x
        dy = y - self.y
        x_rel = dx * self.cos_t + dy * self.sin_t
        y_rel = -dx * self.sin_t + dy * self.cos_t
        return x_rel, y_rel

    def patch_to_world_frame(self, x_rel: float, y_rel: float) -> Tuple[float, float]:
        x = x_rel * self.cos_t - y_rel * self.sin_t + self.x
        y = x_rel * self.sin_t + y_rel * self.cos_t + self.y
        return x, y

    def signed_distance(self, x: float, y: float) -> float:
        dx = x - self.x
        dy = y - self.y
        x_rot = dx * self.cos_t + dy * self.sin_t
        y_rot = -dx * self.sin_t + dy * self.cos_t
        normalized = np.sqrt((x_rot / self.a) ** 2 + (y_rot / self.b) ** 2)
        return float((normalized - 1.0) * (self.a + self.b) / 2.0)

    def get_state(self) -> np.ndarray:
        return np.array([self.x, self.y, self.theta, self.v], dtype=np.float32)

    def get_velocity_vector(self) -> Tuple[float, float]:
        return self.v * self.cos_t, self.v * self.sin_t

    def save_state(self) -> None:
        self.history.append(
            {
                "x": self.x,
                "y": self.y,
                "theta": self.theta,
                "v": self.v,
                "a": self.a,
                "b": self.b,
            }
        )

    def randomize_dynamics(self, np_random: np.random.RandomState) -> None:
        self.config.size_change_rate = float(np_random.uniform(0.5, 2.0))
        self.config.v_max = float(np_random.uniform(4.0, 5.5))
        self.config.wheelbase = float(np_random.uniform(0.4, 0.7))

    def get_boundary_points(self, n_points: int = 32) -> List[Tuple[float, float]]:
        pts = self.get_boundary_points_array(n_points)
        return [(float(pts[i, 0]), float(pts[i, 1])) for i in range(pts.shape[0])]

    def get_boundary_points_array(self, n_points: int = 32) -> np.ndarray:
        """Vectorized boundary points — returns (n_points, 2) world coords."""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        x_local = self.a * np.cos(angles)
        y_local = self.b * np.sin(angles)
        x_world = x_local * self.cos_t - y_local * self.sin_t + self.x
        y_world = x_local * self.sin_t + y_local * self.cos_t + self.y
        return np.column_stack((x_world, y_world))

    def check_patch_boundary_wall_collision(
        self,
        occupancy_map: np.ndarray | None,
        resolution: float | None,
        origin: Tuple[float, float] | None,
        n_points: int = 32,
        violation_threshold: float = 0.05,
    ) -> Tuple[bool, List[Tuple[float, float]]]:
        """Sample the ellipse boundary in world space and test occupancy grid cells.

        Points that fall outside the grid (`oob`) count as violations, same as
        in-bounds cells with value < 0.5. That can terminate episodes near the map
        edge even when a smooth ellipse draw looks clear. The drawn Matplotlib
        ellipse is also smoother than the pixel staircase of the occupancy map,
        so a few samples can land in wall cells while the curve appears inside.
        """
        if occupancy_map is None or resolution is None or origin is None:
            return False, []

        pts = self.get_boundary_points_array(n_points)
        px = ((pts[:, 0] - origin[0]) / resolution).astype(np.intp)
        py = ((pts[:, 1] - origin[1]) / resolution).astype(np.intp)

        h, w = occupancy_map.shape
        oob = (px < 0) | (py < 0) | (py >= h) | (px >= w)
        safe_px = np.clip(px, 0, w - 1)
        safe_py = np.clip(py, 0, h - 1)
        in_wall = occupancy_map[safe_py, safe_px] < 0.5
        violated = oob | in_wall

        violation_fraction = int(violated.sum()) / max(n_points, 1)
        if violation_fraction >= violation_threshold:
            return True, [(float(pts[i, 0]), float(pts[i, 1])) for i in np.where(violated)[0]]
        return False, []

