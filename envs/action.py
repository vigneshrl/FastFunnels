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

    # patch_a_range: Tuple[float, float] = (1.5, 2.0)
    # patch_b_range: Tuple[float, float] = (1.5, 2.0)
    patch_accel_max: float = 60.0
    patch_steering_max: float = 0.4189 #this is the max a f1tenth car can physically steer 

    # Optional step-1 conservative clamp.
    first_step_accel_limit: float = 1.0
    first_step_steering_limit: float = 0.2

    def validate(self) -> None:
        # if self.patch_a_range[0] > self.patch_a_range[1]:
        #     raise ValueError("patch_a_range min must be <= max")
        # if self.patch_b_range[0] > self.patch_b_range[1]:
        #     raise ValueError("patch_b_range min must be <= max")
        if self.patch_accel_max <= 0.0:
            raise ValueError("patch_accel_max must be positive")
        if self.patch_steering_max <= 0.0:
            raise ValueError("patch_steering_max must be positive")


class PatchAction:
    """
    Converts normalized policy output into physical patch commands.

    Input action layout:
        [a_norm, b_norm, accel_norm, steering_norm], each in [-1, 1].
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
        """Map normalized action to (a, b, accel, steering)."""
        if np.asarray(action).shape != (2,):
            raise ValueError(f"Expected action shape (4,), got {np.asarray(action).shape}")

        # a = self._scale_from_unit(
        #     float(action[0]),
        #     self.config.patch_a_range[0],
        #     self.config.patch_a_range[1],
        # )
        # b = self._scale_from_unit(
        #     float(action[1]),
        #     self.config.patch_b_range[0],
        #     self.config.patch_b_range[1],
        # )
        accel = float(action[0]) * self.config.patch_accel_max
        steering = float(action[1]) * self.config.patch_steering_max
        return accel, steering

    @staticmethod
    def smooth(previous: float, target: float, alpha: float) -> float:
        """Exponential smoothing: (1-alpha)*previous + alpha*target."""
        return (1.0 - alpha) * previous + alpha * target

    def conservative_first_step(
        self, accel: float, steering: float, step_count: int
    ) -> Tuple[float, float]:
        """Optional first-step clamp for safer episode starts."""
        if step_count != 1:
            return accel, steering

        accel = float(
            np.clip(
                accel,
                -self.config.first_step_accel_limit,
                self.config.first_step_accel_limit,
            )
        )
        steering = float(
            np.clip(
                steering,
                -self.config.first_step_steering_limit,
                self.config.first_step_steering_limit,
            )
        )
        return accel, steering

