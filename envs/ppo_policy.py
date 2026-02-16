#!/usr/bin/env python3
"""
Integrated patch-policy environment and PPO training entrypoint.

This module is the new orchestration layer that links:
- action decoding (`envs.action`)
- patch dynamics (`envs.patch`)
- lidar features/safety (`envs.laser_model`)
- map reset helpers (`envs.reset.map_reset`)
- base f1tenth adapter (`envs.f110_env`)
- SE-MPC + safety layer (`envs.mpc`)
- observation builder (`envs.observation`)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from types import SimpleNamespace
from typing import Optional

import gymnasium as gym
import numpy as np
from gymnasium import spaces

try:
    from .action import PatchAction, PatchActionConfig
    from .f110_env import F110Config, F110EnvAdapter
    from .laser_model import LidarModel
    from .mpc import MPCConfig, SEMPCSolver, SafetyConfig, SafetyLayer
    from .observation import PatchObservationBuilder
    from .patch import DynamicPatch
    from .reset.map_reset import MapResetHelper
except ImportError:
    # Allow running as a standalone script from repo root.
    from envs.action import PatchAction, PatchActionConfig
    from envs.f110_env import F110Config, F110EnvAdapter
    from envs.laser_model import LidarModel
    from envs.mpc import MPCConfig, SEMPCSolver, SafetyConfig, SafetyLayer
    from envs.observation import PatchObservationBuilder
    from envs.patch import DynamicPatch
    from envs.reset.map_reset import MapResetHelper

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


@dataclass
class PatchEnvConfig:
    num_agents: int = 2
    render_mode: Optional[str] = None
    domain_randomize: bool = False
    robot_radius: float = 0.15
    wheelbase: float = 0.33
    alpha: float = 0.1
    max_steps: int = 3000
    goal_reached_radius: float = 1.5

    # Landmark-based setup from previous implementation.
    spawn_pose: tuple[float, float, float] = (-46.6, 27.0, 2.25)
    goal_xy: tuple[float, float] = (-52.8, 19.33)


class PatchEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, config: Optional[PatchEnvConfig] = None):
        super().__init__()
        self.cfg = config or PatchEnvConfig()
        self.num_agents = self.cfg.num_agents
        self.render_mode = self.cfg.render_mode
        self.domain_randomize = self.cfg.domain_randomize

        self.action_model = PatchAction(PatchActionConfig())
        self.action_space = self.action_model.space

        # 4 + 2 + 16 + 2 + 3 + 16 + 1 + 1 + 1 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(48,), dtype=np.float32
        )

        self.f110 = F110EnvAdapter(
            F110Config(
                num_agents=self.num_agents,
                reset_type="rl_grid_static",
                control_input=("speed", "steering_angle"),
            ),
            render_mode=self.render_mode,
        )
        self.map_reset = MapResetHelper()
        self.lidar_model = LidarModel()
        self.obs_builder = PatchObservationBuilder()
        self.patch = DynamicPatch()

        self.mpc_solvers = [
            SEMPCSolver(
                MPCConfig(
                    robot_radius=self.cfg.robot_radius,
                    wheelbase=self.cfg.wheelbase,
                    num_neighbors=self.num_agents - 1,
                    containment_margin=self.cfg.robot_radius,
                    min_agent_dist=2 * self.cfg.robot_radius + 0.2,
                )
            )
            for _ in range(self.num_agents)
        ]
        self.safety_layer = SafetyLayer(
            SafetyConfig(
                robot_radius=self.cfg.robot_radius,
                wheelbase=self.cfg.wheelbase,
                min_agent_dist=2 * self.cfg.robot_radius + 0.25,
            )
        )

        self.current_base_obs = None
        self.agent_states = None
        self.prev_v = None
        self._np_random = None

        self.spawn_pose = np.array(self.cfg.spawn_pose, dtype=np.float32)
        self.goal_xy = np.array(self.cfg.goal_xy, dtype=np.float32)
        self.start_xy = (self.spawn_pose[0], self.spawn_pose[1])

        self._safe_init_a = 2.0
        self._safe_init_b = 1.5
        self._safe_min_a = 1.0
        self._safe_min_b = 0.75
        self.patch_a_range = (self._safe_min_a, self._safe_init_a)
        self.patch_b_range = (self._safe_min_b, self._safe_init_b)

        self.step_count = 0
        self.episode_reward = 0.0
        self.lap_progress = 0.0
        self._init_goal_dist = None
        self._prev_goal_dist = None
        self._prev_patch_pos = None
        self.patch_min_clearance = float("inf")

        self.mpc_successes = 0
        self.mpc_attempts = 0
        self.safety_interventions = 0
        self.cached_controls = [None] * self.num_agents
        self.cached_feasible = [False] * self.num_agents

    def _calculate_min_patch_size_for_agents(self):
        if self.agent_states is None:
            return self.cfg.robot_radius * 2, self.cfg.robot_radius * 2
        positions = []
        for i in range(self.num_agents):
            x, y, _, _ = self.agent_states[i]
            x_p, y_p = self.patch.world_to_patch_frame(x, y)
            positions.append((x_p, y_p))
        if not positions:
            return self.cfg.robot_radius * 2, self.cfg.robot_radius * 2
        arr = np.asarray(positions)
        margin = self.cfg.robot_radius * 1.5
        min_a = max(np.max(np.abs(arr[:, 0])) + margin, 0.5)
        min_b = max(np.max(np.abs(arr[:, 1])) + margin, 0.5)
        return float(min_a), float(min_b)

    def _build_obs(self) -> np.ndarray:
        lidar_distances = self.lidar_model.aggregate(self.current_base_obs, self.patch)
        clearance_obs = self.lidar_model.compute_clearance_observation(
            self.patch, lidar_distances
        )
        best_heading, best_clearance = self.lidar_model.get_best_heading(
            self.current_base_obs, self.patch
        )
        return self.obs_builder.build(
            patch=self.patch,
            start_xy=self.start_xy,
            goal_xy=self.goal_xy,
            lidar_distances=lidar_distances,
            clearance_obs=clearance_obs,
            best_heading=best_heading,
            best_clearance=best_clearance,
            mpc_attempts=self.mpc_attempts,
            mpc_successes=self.mpc_successes,
            safety_interventions=self.safety_interventions,
            step_count=self.step_count,
            num_agents=self.num_agents,
            alpha=self.cfg.alpha,
            size_change_rate=self.patch.config.size_change_rate,
        )

    def _compute_reward(self, lidar_info: dict) -> float:
        reward = 0.0
        min_dist = float(lidar_info["min_dist"])
        if self.patch.v > 0.1:
            v_norm = min(self.patch.v / 5.0, 1.0)
            reward += 7.0 * v_norm
            if self.patch.v >= 3.0:
                reward += 5.0 * (self.patch.v - 3.0) / 2.0
        else:
            reward -= 5.0

        current_pos = np.array([self.patch.x, self.patch.y], dtype=np.float32)
        if self._prev_patch_pos is None:
            self._prev_patch_pos = current_pos.copy()
        moved = float(np.linalg.norm(current_pos - self._prev_patch_pos))
        if moved > 1e-3:
            heading_vec = np.array([np.cos(self.patch.theta), np.sin(self.patch.theta)])
            forward_component = float(np.dot(current_pos - self._prev_patch_pos, heading_vec))
            if forward_component > 0:
                reward += 5.0 * forward_component
            elif forward_component < -0.05:
                reward -= 3.0 * abs(forward_component)
        self._prev_patch_pos = current_pos.copy()

        reward += 5.0 * (min_dist / 2.0)
        if min_dist < 1.5:
            reward -= 8.0 * (1.5 - min_dist)

        _, _, goal_dist = self.obs_builder.goal_direction_and_distance(self.patch, self.goal_xy)
        if self._prev_goal_dist is None:
            self._prev_goal_dist = goal_dist
        dist_improve = self._prev_goal_dist - goal_dist
        reward += 8.0 * dist_improve
        self._prev_goal_dist = goal_dist

        if self.mpc_attempts > 0:
            feas = self.mpc_successes / self.mpc_attempts
            reward += 10.0 * (feas - 0.5)
            if feas >= 0.95:
                reward += 3.0

        safety_rate = self.safety_interventions / max(1, self.step_count * self.num_agents)
        if safety_rate > 0.2:
            reward -= 5.0 * (safety_rate - 0.2)

        reward += 0.05
        return float(np.clip(reward, -50.0, 50.0))

    def _check_termination(self, lidar_info: dict):
        terminated = False
        truncated = False
        reason = None

        track, occ_map, resolution, origin = self.f110.get_track_data()
        del track
        patch_collision, violated = self.patch.check_patch_boundary_wall_collision(
            occ_map, resolution, origin, n_points=32, violation_threshold=0.05
        )
        if patch_collision:
            return True, False, f"patch_wall_collision ({len(violated)} points)"

        min_dist = float(lidar_info["min_dist"])
        if not np.isfinite(min_dist) or min_dist < 0.95:
            return True, False, "patch_wall_collision_lidar"

        if self.patch.a < self._safe_min_a or self.patch.b < self._safe_min_b:
            return True, False, "patch_too_small"

        progress, _, self._init_goal_dist = self.obs_builder.lap_progress(
            self.patch, self.goal_xy, self._init_goal_dist
        )
        if progress > 0.95 and self.lap_progress > 0.9:
            return True, False, "goal_reached"

        if self.step_count >= self.cfg.max_steps:
            truncated = True
            reason = "max_steps"
        return terminated, truncated, reason

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._np_random = np.random.RandomState(seed if seed is not None else None)
        self.step_count = 0
        self.episode_reward = 0.0
        self.lap_progress = 0.0
        self._prev_goal_dist = None
        self._init_goal_dist = None
        self._prev_patch_pos = None
        self.patch_min_clearance = float("inf")
        self.mpc_successes = 0
        self.mpc_attempts = 0
        self.safety_interventions = 0
        self.cached_controls = [None] * self.num_agents
        self.cached_feasible = [False] * self.num_agents

        if options is not None and "spawn_pose" in options:
            self.spawn_pose = np.array(options["spawn_pose"], dtype=np.float32)
        if options is not None and "goal_xy" in options:
            self.goal_xy = np.array(options["goal_xy"], dtype=np.float32)

        patch_x, patch_y, patch_theta = map(float, self.spawn_pose)
        self.start_xy = (patch_x, patch_y)

        self.f110.ensure_initialized()
        _, occ_map, resolution, origin = self.f110.get_track_data()
        track_width = self.map_reset.estimate_track_width(
            occ_map, resolution, origin, patch_x, patch_y, patch_theta
        )
        init_a, init_b = self.map_reset.compute_safe_patch_size(track_width)
        self._safe_init_a, self._safe_init_b = init_a, init_b
        self._safe_min_a, self._safe_min_b = init_a * 0.5, init_b * 0.5
        self.patch_a_range = (self._safe_min_a, self._safe_init_a)
        self.patch_b_range = (self._safe_min_b, self._safe_init_b)

        self.action_model.config.patch_a_range = self.patch_a_range
        self.action_model.config.patch_b_range = self.patch_b_range

        self.patch = DynamicPatch(x=patch_x, y=patch_y, theta=patch_theta, v=0.5, a=init_a, b=init_b)
        if self.domain_randomize:
            self.patch.randomize_dynamics(self._np_random)
            self.patch.a = init_a
            self.patch.b = init_b

        agent_poses = self.map_reset.generate_safe_agent_poses(
            num_agents=self.num_agents,
            patch_x=patch_x,
            patch_y=patch_y,
            patch_theta=patch_theta,
            patch_a=init_a,
            patch_b=init_b,
            domain_randomize=self.domain_randomize,
            np_random=self._np_random,
        )
        base_obs, _ = self.f110.reset(poses=agent_poses)
        self.current_base_obs = base_obs

        self.agent_states = []
        self.prev_v = []
        for i in range(self.num_agents):
            v = float(
                np.sqrt(
                    base_obs["linear_vels_x"][i] ** 2
                    + base_obs["linear_vels_y"][i] ** 2
                )
            )
            self.prev_v.append(max(v, 0.5))
            self.agent_states.append(
                [
                    float(base_obs["poses_x"][i]),
                    float(base_obs["poses_y"][i]),
                    float(base_obs["poses_theta"][i]),
                    max(v, 0.5),
                ]
            )

        obs = self._build_obs()
        return obs, {}

    def step(self, patch_action):
        self.step_count += 1
        dt = 0.05

        a, b, raw_accel, raw_steer = self.action_model.denormalize(np.asarray(patch_action))
        smooth_accel = self.action_model.smooth(self.patch.accel, raw_accel, self.cfg.alpha)
        smooth_steer = self.action_model.smooth(self.patch.steering, raw_steer, self.cfg.alpha)
        smooth_accel, smooth_steer = self.action_model.conservative_first_step(
            smooth_accel, smooth_steer, self.step_count
        )
        self.patch.accel = smooth_accel
        self.patch.steering = smooth_steer
        self.patch.step(smooth_accel, smooth_steer, dt)

        min_a_agents, min_b_agents = self._calculate_min_patch_size_for_agents()
        a = min(max(a, min_a_agents), self._safe_init_a)
        b = min(max(b, min_b_agents), self._safe_init_b)
        self.patch.update_shape(a, b, dt, max_a=self._safe_init_a, max_b=self._safe_init_b)
        self.patch.save_state()

        base_obs = self.current_base_obs
        vx_patch, vy_patch = self.patch.get_velocity_vector()
        patch_for_mpc = SimpleNamespace(
            a=self.patch.a,
            b=self.patch.b,
            theta=self.patch.theta,
            vx=vx_patch,
            vy=vy_patch,
            cx=self.patch.x,
            cy=self.patch.y,
            cos_t=self.patch.cos_t,
            sin_t=self.patch.sin_t,
        )

        positions_world = [
            [base_obs["poses_x"][i], base_obs["poses_y"][i]] for i in range(self.num_agents)
        ]
        positions_local = [[p[0] - self.patch.x, p[1] - self.patch.y] for p in positions_world]

        env_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        solve_mpc = self.step_count == 1 or self.step_count % 2 == 0
        for i in range(self.num_agents):
            x_local = positions_local[i][0]
            y_local = positions_local[i][1]
            theta_i = base_obs["poses_theta"][i]
            v_i = self.prev_v[i]
            x0_local = np.array([x_local, y_local, theta_i, v_i], dtype=np.float32)
            neighbors_local = [positions_local[j] for j in range(self.num_agents) if j != i]

            if solve_mpc:
                self.mpc_attempts += 1
                u_opt, feasible = self.mpc_solvers[i].solve(x0_local, patch_for_mpc, neighbors_local)
                self.cached_controls[i] = u_opt
                self.cached_feasible[i] = feasible
                if feasible:
                    self.mpc_successes += 1
            else:
                u_opt = self.cached_controls[i]
                feasible = self.cached_feasible[i]

            u_safe, intervention = self.safety_layer.filter_control(
                u_opt if feasible else None, x0_local, patch_for_mpc, neighbors_local, dt=dt
            )
            if intervention:
                self.safety_interventions += 1
            accel, steering = float(u_safe[0]), float(u_safe[1])
            v_new = float(np.clip(v_i + accel * dt, 0.5, 10.0))
            self.prev_v[i] = v_new
            env_actions[i] = [v_new, float(np.clip(steering, -0.4, 0.4))]

        base_obs, _, base_done, base_truncated, _ = self.f110.step(env_actions)
        self.current_base_obs = base_obs

        for i in range(self.num_agents):
            v_sim = float(
                np.sqrt(
                    base_obs["linear_vels_x"][i] ** 2
                    + base_obs["linear_vels_y"][i] ** 2
                )
            )
            self.agent_states[i] = [
                float(base_obs["poses_x"][i]),
                float(base_obs["poses_y"][i]),
                float(base_obs["poses_theta"][i]),
                max(v_sim, 0.5),
            ]

        lidar_safe, min_dist, lidar_info = self.lidar_model.check_lidar_safety(base_obs, self.patch, safety_margin=0.2)
        if min_dist < 1.0:
            clearance = -(1.0 - min_dist) * max(self.patch.a, self.patch.b)
        else:
            clearance = (min_dist - 1.0) * max(self.patch.a, self.patch.b)
        if np.isfinite(clearance):
            self.patch_min_clearance = min(self.patch_min_clearance, float(clearance))

        reward = self._compute_reward(lidar_info)
        self.episode_reward += reward

        self.lap_progress, _, self._init_goal_dist = self.obs_builder.lap_progress(
            self.patch, self.goal_xy, self._init_goal_dist
        )

        terminated, truncated, reason = self._check_termination(lidar_info)
        terminated = bool(terminated or base_done)
        truncated = bool(truncated or base_truncated)

        obs = self._build_obs()
        info = {
            "episode_reward": self.episode_reward,
            "lap_progress": self.lap_progress,
            "Episode_steps": self.step_count,
            "patch_size": (self.patch.a, self.patch.b),
            "patch_velocity": self.patch.v,
            "step_reward": reward,
            "termination_reason": reason,
            "lidar_safe": lidar_safe,
        }
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.f110.render()

    def close(self):
        self.f110.close()


def make_patch_env(rank: int, seed: int = 0, domain_randomize: bool = False):
    def _init():
        cfg = PatchEnvConfig(domain_randomize=domain_randomize, num_agents=2, render_mode=None)
        env = PatchEnv(cfg)
        env.reset(seed=seed + rank)
        return env

    if SB3_AVAILABLE:
        set_random_seed(seed + rank)
    return _init




