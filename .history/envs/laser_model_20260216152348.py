"""
LIDAR feature and safety utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class LidarConfig:
    n_sectors: int = 16
    max_range_m: float = 10.0


class LidarModel:
    """Builds patch-centric lidar features from multi-agent scans."""

    def __init__(self, config: LidarConfig | None = None) -> None:
        self.config = config or LidarConfig()

    def aggregate(self, obs: Dict, patch) -> np.ndarray:
        scans = np.array(obs["scans"], dtype=np.float32) * self.config.max_range_m
        scans = np.clip(scans, 0.0, self.config.max_range_m)
        scans = np.where(np.isfinite(scans), scans, self.config.max_range_m)

        num_rays = scans.shape[1]
        ray_angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)
        x_loc = scans * np.cos(ray_angles)
        y_loc = scans * np.sin(ray_angles)

        ax = np.array(obs["poses_x"], dtype=np.float32)[:, None]
        ay = np.array(obs["poses_y"], dtype=np.float32)[:, None]
        at = np.array(obs["poses_theta"], dtype=np.float32)[:, None]

        x_world = ax + x_loc * np.cos(at) - y_loc * np.sin(at)
        y_world = ay + x_loc * np.sin(at) + y_loc * np.cos(at)

        dx = x_world.flatten() - patch.x
        dy = y_world.flatten() - patch.y
        point_angles = np.arctan2(dy, dx)
        point_angles = np.where(point_angles < 0.0, point_angles + 2.0 * np.pi, point_angles)
        point_distances = np.sqrt(dx**2 + dy**2)
        valid_mask = np.isfinite(point_distances) & (point_distances > 0.0)

        n_sectors = self.config.n_sectors
        sector_angle = 2.0 * np.pi / n_sectors
        aggregated = np.ones(n_sectors, dtype=np.float32) * np.inf

        for sector_idx in range(n_sectors):
            center = sector_idx * sector_angle
            start = center - sector_angle / 2.0
            end = center + sector_angle / 2.0
            if start < 0.0:
                mask = (point_angles >= start + 2.0 * np.pi) | (point_angles <= end)
            elif end > 2.0 * np.pi:
                mask = (point_angles >= start) | (point_angles <= end - 2.0 * np.pi)
            else:
                mask = (point_angles >= start) & (point_angles <= end)
            mask = mask & valid_mask
            if np.any(mask):
                aggregated[sector_idx] = float(np.min(point_distances[mask]))

        return aggregated

    def get_best_heading(self, obs: Dict, patch) -> Tuple[float, float]:
        aggregated = self.aggregate(obs, patch)
        smoothed = np.convolve(aggregated, np.ones(3) / 3.0, mode="same")
        best_idx = int(np.argmax(smoothed))
        best_clearance = float(smoothed[best_idx])
        n = len(smoothed)
        angle_offset = (best_idx - n // 2) / (n // 2)
        return float(angle_offset), min(best_clearance / self.config.max_range_m, 1.0)

    def compute_clearance_observation(self, patch, lidar_distances: np.ndarray) -> np.ndarray:
        n_sectors = self.config.n_sectors
        sector_angle = 2.0 * np.pi / n_sectors
        clearance = np.zeros(n_sectors, dtype=np.float32)
        for i in range(n_sectors):
            angle = i * sector_angle
            c, s = np.cos(angle), np.sin(angle)
            denom = np.sqrt((patch.b * c) ** 2 + (patch.a * s) ** 2)
            r_ellipse = (patch.a * patch.b) / denom if denom > 1e-6 else 0.0
            clearance[i] = float(lidar_distances[i] - r_ellipse)
        return np.clip(clearance / 5.0, -1.0, 1.0)

    def check_lidar_safety(self, base_obs: Dict, patch, safety_margin: float = 0.5):
        del safety_margin  # Kept for API compatibility.
        scans = np.array(base_obs["scans"], dtype=np.float32) * self.config.max_range_m
        scans = np.clip(scans, 0.0, self.config.max_range_m)
        scans = np.where(np.isfinite(scans), scans, self.config.max_range_m)
        num_rays = scans.shape[1]
        angles = np.linspace(0.0, 2.0 * np.pi, num_rays, endpoint=False)

        x_loc = scans * np.cos(angles)
        y_loc = scans * np.sin(angles)

        ax = np.array(base_obs["poses_x"], dtype=np.float32)[:, None]
        ay = np.array(base_obs["poses_y"], dtype=np.float32)[:, None]
        at = np.array(base_obs["poses_theta"], dtype=np.float32)[:, None]
        x_world = ax + x_loc * np.cos(at) - y_loc * np.sin(at)
        y_world = ay + x_loc * np.sin(at) + y_loc * np.cos(at)

        dx = x_world - patch.x
        dy = y_world - patch.y
        x_p = dx * patch.cos_t + dy * patch.sin_t
        y_p = -dx * patch.sin_t + dy * patch.cos_t

        ellipse_dists = (x_p / patch.a) ** 2 + (y_p / patch.b) ** 2
        valid = ellipse_dists[np.isfinite(ellipse_dists) & (ellipse_dists > 0.0)]
        min_dist = 0.5 if valid.size == 0 else float(np.clip(np.min(valid), 0.0, 10.0))
        if not np.isfinite(min_dist) or min_dist > 100.0:
            min_dist = 0.5

        is_safe = bool(min_dist > 1.0)
        return is_safe, min_dist, {"is_safe": is_safe, "min_dist": min_dist}

