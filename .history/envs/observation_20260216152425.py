"""
Observation builders for patch-policy training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ObservationConfig:
    track_scale: float = 100.0
    max_goal_dist_norm: float = 50.0
    max_speed_norm: float = 10.0
    max_size_norm: float = 5.0
    max_wall_dist_norm: float = 10.0


class PatchObservationBuilder:
    """Builds patch-centric observations from state + lidar features."""

    def __init__(self, config: ObservationConfig | None = None) -> None:
        self.config = config or ObservationConfig()

    def goal_direction_and_distance(self, patch, goal_xy: np.ndarray):
        goal_vec_world = np.array([goal_xy[0] - patch.x, goal_xy[1] - patch.y], dtype=np.float32)
        goal_dist = float(np.linalg.norm(goal_vec_world))
        if goal_dist > 1e-6:
            goal_dir_world = goal_vec_world / goal_dist
        else:
            goal_dir_world = np.array([1.0, 0.0], dtype=np.float32)

        gx = patch.x + goal_dir_world[0]
        gy = patch.y + goal_dir_world[1]
        lx, ly = patch.world_to_patch_frame(gx, gy)
        local_norm = np.linalg.norm([lx, ly]) + 1e-6
        local_goal_dir = np.array([lx, ly], dtype=np.float32) / local_norm
        return goal_dir_world, local_goal_dir, goal_dist

    def lap_progress(self, patch, goal_xy: np.ndarray, init_goal_dist: float | None):
        goal_dir_world, _, goal_dist = self.goal_direction_and_distance(patch, goal_xy)
        if init_goal_dist is None:
            init_goal_dist = max(goal_dist, 1e-6)
        progress = float(np.clip(1.0 - (goal_dist / max(init_goal_dist, 1e-6)), 0.0, 1.0))
        return progress, goal_dir_world, init_goal_dist

    def build(
        self,
        patch,
        start_xy: Tuple[float, float],
        goal_xy: np.ndarray,
        lidar_distances: np.ndarray,
        clearance_obs: np.ndarray,
        best_heading: float,
        best_clearance: float,
        mpc_attempts: int,
        mpc_successes: int,
        safety_interventions: int,
        step_count: int,
        num_agents: int,
        alpha: float,
        size_change_rate: float,
    ) -> np.ndarray:
        patch_state = patch.get_state()
        patch_x, patch_y, patch_theta, patch_v = patch_state
        patch_v = max(float(patch_v), 0.1)

        rel_x = patch_x - float(start_xy[0])
        rel_y = patch_y - float(start_xy[1])

        patch_obs = np.array(
            [
                rel_x / self.config.track_scale,
                rel_y / self.config.track_scale,
                patch_theta / np.pi,
                patch_v / self.config.max_speed_norm,
            ],
            dtype=np.float32,
        )
        shape_obs = np.array(
            [
                patch.a / self.config.max_size_norm,
                patch.b / self.config.max_size_norm,
            ],
            dtype=np.float32,
        )
        lidar_obs = np.clip(lidar_distances / 10.0, 0.0, 1.0).astype(np.float32)
        heading_obs = np.array([best_heading, best_clearance], dtype=np.float32)
        _, local_goal_dir, goal_dist = self.goal_direction_and_distance(patch, goal_xy)
        goal_obs = np.array(
            [
                local_goal_dir[0],
                local_goal_dir[1],
                min(goal_dist / self.config.max_goal_dist_norm, 1.0),
            ],
            dtype=np.float32,
        )
        wall_dist_obs = np.array([float(np.min(lidar_distances)) / self.config.max_wall_dist_norm], dtype=np.float32)
        domain_obs = np.array([size_change_rate / 2.0], dtype=np.float32)
        alpha_obs = np.array([alpha], dtype=np.float32)

        feasibility_rate = 1.0 if mpc_attempts <= 0 else float(mpc_successes / mpc_attempts)
        safety_rate = float(safety_interventions / max(1, step_count * num_agents))
        mpc_obs = np.array([feasibility_rate, safety_rate], dtype=np.float32)

        return np.concatenate(
            [
                patch_obs,
                shape_obs,
                lidar_obs,
                heading_obs,
                goal_obs,
                clearance_obs.astype(np.float32),
                wall_dist_obs,
                domain_obs,
                alpha_obs,
                mpc_obs,
            ]
        ).astype(np.float32)

