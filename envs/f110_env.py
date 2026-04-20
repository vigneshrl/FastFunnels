"""
F1TENTH simulator adapter.

This module isolates direct interactions with the base `f1tenth_gym` environment:
- creation and lifecycle (init/reset/step/render/close)
- map/track access helpers
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np


@dataclass
class F110Config:
    """Configuration for creating an F1TENTH base environment."""

    #map_name: str = "custom_map6"
    map_name: str = "plain_map"
    num_agents: int = 2
    timestep: float = 0.01
    integrator: str = "rk4"
    control_input: Tuple[str, str] = ("speed", "steering_angle")
    model: str = "st" #"st" - can be used to simulate the dynamic single track model with (yaw_rate, slip angle and tire-style terms)
    observation_type: str = "original"
    mu: float = 1.0
    reset_type: str = "rl_random_static"
    extra_params: Dict[str, Any] = field(default_factory=dict)

    def to_gym_config(self) -> Dict[str, Any]:
        """Build config dict expected by `f1tenth_gym:f1tenth-v0`."""
        config = {
            "map": self.map_name,
            "num_agents": self.num_agents,
            "timestep": self.timestep,
            "integrator": self.integrator,
            "control_input": list(self.control_input),
            "model": self.model,
            "observation_config": {"type": self.observation_type},
            "params": {"mu": self.mu},
            "reset_config": {"type": self.reset_type},
        }
        config.update(self.extra_params)
        return config


class F110EnvAdapter:
    """Thin adapter around `f1tenth_gym` to keep env orchestration clean."""

    def __init__(self, config: Optional[F110Config] = None, render_mode: Optional[str] = None):
        self.config = config or F110Config()
        self.render_mode = render_mode
        self.base_env = None

    def ensure_initialized(self) -> None:
        """Create base env once and perform an initial reset."""
        if self.base_env is not None:
            return

        # Ensures environment registration in some subprocess setups.
        import f1tenth_gym  # noqa: F401

        self.base_env = gym.make(
            "f1tenth_gym:f1tenth-v0",
            config=self.config.to_gym_config(),
            render_mode="human" if self.render_mode == "human" else None,
        )
        self.base_env.reset()

    def reset(
        self,
        poses: Optional[np.ndarray] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Reset the simulator.

        Args:
            poses: Optional agent poses shaped (num_agents, 3). If provided,
                passed as `options={"poses": poses}`.
            options: Additional reset options. If both `poses` and `options`
                are provided, `poses` is merged into `options`.
        """
        self.ensure_initialized()
        reset_options: Dict[str, Any] = dict(options or {})
        if poses is not None:
            reset_options["poses"] = poses
        return self.base_env.reset(options=reset_options if reset_options else None)

    def step(self, actions: np.ndarray):
        """Step the simulator with shape (num_agents, 2) actions."""
        self.ensure_initialized()
        return self.base_env.step(actions)

    def render(self):
        """Render passthrough for human/rgb_array modes."""
        if self.base_env is None:
            return None
        if self.render_mode in ("human", "rgb_array"):
            return self.base_env.render()
        return None

    def close(self) -> None:
        """Close the base simulator safely."""
        if self.base_env is not None:
            self.base_env.close()
            self.base_env = None

    def get_track_data(self):
        """
        Access track and occupancy map objects.

        Returns:
            (track, occupancy_map, resolution, origin)
        """
        self.ensure_initialized()
        track = self.base_env.unwrapped.track
        occupancy_map = track.occupancy_map
        resolution = track.spec.resolution
        origin = track.spec.origin
        return track, occupancy_map, resolution, origin

