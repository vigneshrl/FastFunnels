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

    def step(self, speed: float, steering: float, dt: float = 0.05) -> None:
        speed = float(np.clip(speed, self.config.v_min, self.config.v_max))
        steering = float(np.clip(steering, -self.config.steering_max, self.config.steering_max))

        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += (self.v / self.config.wheelbase) * np.tan(steering) * dt
        self.v = speed

        
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
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points: List[Tuple[float, float]] = []
        for angle in angles:
            x_local = self.a * np.cos(angle)
            y_local = self.b * np.sin(angle)
            points.append(self.patch_to_world_frame(x_local, y_local))
        return points

    def check_patch_boundary_wall_collision(
        self,
        occupancy_map: np.ndarray | None,
        resolution: float | None,
        origin: Tuple[float, float] | None,
        n_points: int = 32,
        violation_threshold: float = 0.05,
    ) -> Tuple[bool, List[Tuple[float, float]]]:
        if occupancy_map is None or resolution is None or origin is None:
            return False, []

        violated_points: List[Tuple[float, float]] = []
        for x_world, y_world in self.get_boundary_points(n_points=n_points):
            px = int((x_world - origin[0]) / resolution)
            py = int((y_world - origin[1]) / resolution)

            if px < 0 or py < 0 or py >= occupancy_map.shape[0] or px >= occupancy_map.shape[1]:
                violated_points.append((x_world, y_world))
                continue
            if occupancy_map[py, px] < 0.5:
                violated_points.append((x_world, y_world))

        violation_fraction = len(violated_points) / max(n_points, 1)
        return violation_fraction >= violation_threshold, violated_points

