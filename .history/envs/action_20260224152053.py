"""
Action utilities for patch-policy control.

This module centralizes:
- patch action space definition
- normalized (-1..1) to physical control conversion
- optional action smoothing and first-step safety clamp
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import gymnasium as gym
import numpy as np


@dataclass
class PatchActionConfig:
    """Configuration for patch action decoding."""

    patch_a_range: Tuple[float, float] = (1.5, 2.0)
    patch_b_range: Tuple[float, float] = (1.5, 2.0)
    patch_speed_max: float = 10.0
    patch_steering_max: float = 0.4189 #this is the max a f1tenth car can physically steer 

    # Optional step-1 conservative clamp.
    first_step_speed_limit: float = 1.0
    first_step_steering_limit: float = 0.2

    def validate(self) -> None:
        # if self.patch_a_range[0] > self.patch_a_range[1]:
        #     raise ValueError("patch_a_range min must be <= max")
        # if self.patch_b_range[0] > self.patch_b_range[1]:
        #     raise ValueError("patch_b_range min must be <= max")
        # if self.patch_speed_max <= 0.0:
        #     raise ValueError("patch_speed_max must be positive")
        if self.patch_steering_max <= 0.0:
            raise ValueError("patch_steering_max must be positive")


class PatchAction:
    """
    Converts normalized policy output into physical patch commands.

    Input action layout:
        [a_norm, b_norm, speed_norm, steering_norm], each in [-1, 1].
    """

    def __init__(self, config: PatchActionConfig | None = None) -> None:
        self.config = config or PatchActionConfig()
        self.config.validate()

    @property
    def space(self) -> gym.Space:
        """Normalized policy action space expected by PPO."""
        return gym.spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    @staticmethod
    def _scale_from_unit(action_value: float, min_value: float, max_value: float) -> float:
        return (action_value + 1.0) * 0.5 * (max_value - min_value) + min_value

    def denormalize(self, action: np.ndarray) -> Tuple[float, float, float, float]:
        """Map normalized action to (a, b, speed, steering)."""
        if np.asarray(action).shape != (4,):  #Previously was 2 because the action set was only speed, steering now added a nd b 
            raise ValueError(f"Expected action shape (4,), got {np.asarray(action).shape}")

        a = self._scale_from_unit(
            float(action[0]),
            self.config.patch_a_range[0],
            self.config.patch_a_range[1],
        )
        b = self._scale_from_unit(
            float(action[1]),
            self.config.patch_b_range[0],
            self.config.patch_b_range[1],
        )
        speed = float(action[0]) * self.config.patch_speed_max
        steering = float(action[1]) * self.config.patch_steering_max
        return speed, steering , a, b

    @staticmethod
    def smooth(previous: float, target: float, alpha: float) -> float:
        """Exponential smoothing: (1-alpha)*previous + alpha*target."""
        return (1.0 - alpha) * previous + alpha * target

    def conservative_first_step(
        self, speed: float, steering: float, step_count: int
    ) -> Tuple[float, float]:
        """Optional first-step clamp for safer episode starts."""
        if step_count != 1:
            return speed, steering

        speed = float(
            np.clip(
                speed,
                -self.config.first_step_speed_limit,
                self.config.first_step_speed_limit,
            )
        )
        steering = float(
            np.clip(
                steering,
                -self.config.first_step_steering_limit,
                self.config.first_step_steering_limit,
            )
        )
        return speed, steering

