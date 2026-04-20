"""
Map/reset helper utilities for patch-based training.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass
class ResetConfig:
    robot_radius: float = 0.15
    safety_margin: float = 0.5
    max_scan_dist: float = 5.0
    scan_step: float = 0.1


class MapResetHelper:
    """Track-width estimation and safe spawn pose generation."""

    def __init__(self, config: ResetConfig | None = None) -> None:
        self.config = config or ResetConfig()

    def estimate_track_width(
        self,
        occupancy_map: np.ndarray,
        resolution: float,
        origin: Tuple[float, float],
        x: float,
        y: float,
        theta: float,
    ) -> float:
        distances = np.arange(0.0, self.config.max_scan_dist, self.config.scan_step)
        h, w = occupancy_map.shape
        max_d = float(self.config.max_scan_dist)

        def cast_vec(angle: float) -> float:
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            px = x + distances * cos_a
            py = y + distances * sin_a
            ix = ((px - origin[0]) / resolution).astype(np.intp)
            iy = ((py - origin[1]) / resolution).astype(np.intp)
            oob = (ix < 0) | (iy < 0) | (iy >= h) | (ix >= w)
            safe_ix = np.clip(ix, 0, w - 1)
            safe_iy = np.clip(iy, 0, h - 1)
            hit = oob | (occupancy_map[safe_iy, safe_ix] < 0.5)
            idx = np.argmax(hit)
            return float(distances[idx]) if hit[idx] else max_d

        return cast_vec(theta + np.pi / 2.0) + cast_vec(theta - np.pi / 2.0)

    def compute_safe_patch_size(self, track_width: float) -> Tuple[float, float]:
        max_patch_width = track_width - (2.0 * self.config.safety_margin)
        max_b = np.clip(max_patch_width / 2.0, 0.5, 1.5)
        init_b = float(max_b)
        init_a = float(min(2.0, max_b * 1.5))
        return init_a, init_b

    def generate_safe_agent_poses(
        self,
        num_agents: int,
        patch_x: float,
        patch_y: float,
        patch_theta: float,
        patch_a: float,
        patch_b: float,
        domain_randomize: bool = False,
        np_random: np.random.RandomState | None = None,
    ) -> np.ndarray:
        poses = np.zeros((num_agents, 3), dtype=np.float32)
        edge_margin = self.config.robot_radius + 0.3
        min_inter_dist = 2 * self.config.robot_radius + 0.3

        eff_a = max(patch_a - edge_margin, 1.0)
        if num_agents == 2:
            spacing = max(min_inter_dist * 0.6, 0.2)
            local_positions = [(-0.5 * spacing, 0.0), (1.0 * spacing, 0.0)]
        else:
            spacing = max(min_inter_dist, 2 * eff_a / (num_agents + 1))
            local_positions = [
                ((i - (num_agents - 1) / 2) * spacing, 0.0) for i in range(num_agents)
            ]

        c = np.cos(patch_theta)
        s = np.sin(patch_theta)
        for i, (x_local, y_local) in enumerate(local_positions):
            poses[i, 0] = patch_x + x_local * c - y_local * s
            poses[i, 1] = patch_y + x_local * s + y_local * c
            poses[i, 2] = patch_theta

        if domain_randomize and np_random is not None:
            poses[:, 0] += np_random.uniform(-0.1, 0.1, size=num_agents)
            poses[:, 1] += np_random.uniform(-0.1, 0.1, size=num_agents)
            poses[:, 2] += np_random.uniform(-0.05, 0.05, size=num_agents)
        return poses

