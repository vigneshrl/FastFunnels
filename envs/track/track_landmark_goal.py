"""Landmark-based spawn/goal presets for patch environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class LandmarkGoal:
    spawn_pose: tuple[float, float, float]
    goal_xy: tuple[float, float]

    @property
    def spawn_array(self) -> np.ndarray:
        return np.array(self.spawn_pose, dtype=np.float32)

    @property
    def goal_array(self) -> np.ndarray:
        return np.array(self.goal_xy, dtype=np.float32)


DEFAULT_LANDMARKS: Dict[str, LandmarkGoal] = {
    "spielberg_default": LandmarkGoal(
        spawn_pose=(-46.6, 27.0, 2.25),
        goal_xy=(-52.8, 19.33),
    ),
}


def get_landmark(name: str = "spielberg_default") -> LandmarkGoal:
    """Return a named landmark preset."""
    if name not in DEFAULT_LANDMARKS:
        valid = ", ".join(sorted(DEFAULT_LANDMARKS.keys()))
        raise KeyError(f"Unknown landmark '{name}'. Valid options: {valid}")
    return DEFAULT_LANDMARKS[name]


