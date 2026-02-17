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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

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
    control_dt = 0.05
    mpc_horizon_steps: int = 10
    base_reset_type: str = "rl_random_static"
    render_mode: Optional[str] = None
    domain_randomize: bool = False
    robot_radius: float = 0.15
    wheelbase: float = 0.33
    alpha: float = 0.1
    max_steps: int = 2000
    goal_reached_radius: float = 1.5


    # Frenet-style reward - which is very similar to the single-agent PPO 
    reward_progress_scale: float = 40.0
    reward_crosstrack_weight: float = 0.5
    reward_steer_bias_weight: float = 0.0
    reward_steer_rate_weight: float = 0.0
    spin_yawrate_threshold: float = 0.0
    reward_spin_weight: float = 0.0
    collision_penalty: float = 300.0
    collision_min_dist: float = 0.25
    enable_lidar_termination: bool = False
    patch_boundary_violation_threshold: float = 0.05
    offtrack_ey_termination: float = 8.0

    stuck_no_progress_steps: int = 60
    stuck_progress_eps: float = 1e-3
    stuck_penalty: float = 10.0
    lap_finish_bonus: float = 2000.0
    lap_bonus_tau = 200.0
    time_penalty_per_sec: float = 0.0
    compact_observation: bool = True
    debug_print_every_n_steps: int = 0
    debug_print_episode_end: bool = True
    use_base_done_termination: bool = False

    # Navigation mode: "landmark" or "centerline"
    navigation_mode: str = "centerline"
    
    # Landmark-based setup (used when navigation_mode="landmark")
    spawn_pose: tuple[float, float, float] = (-46.6, 27.0, 2.25)
    goal_xy: tuple[float, float] = (-52.8, 19.33)
    
    # Centerline-based setup (used when navigation_mode="centerline")
    look_ahead_waypoints: int = 10  # How many waypoints ahead to look for goal direction
    search_window: int = 30  # Window size for finding nearest waypoint


class PatchEnv(gym.Env):
    """
    Patch-policy environment with dual navigation modes.
    
    Navigation Modes:
    - "landmark": Uses fixed spawn_pose and goal_xy for navigation
    - "centerline": Uses track centerline waypoints for navigation
    
    Quick Swap Example:
        # Landmark mode (default)
        cfg = PatchEnvConfig(navigation_mode="landmark")
        env = PatchEnv(cfg)
        
        # Centerline mode
        cfg = PatchEnvConfig(navigation_mode="centerline")
        env = PatchEnv(cfg)
    """
    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, config: Optional[PatchEnvConfig] = None):
        super().__init__()
        self.cfg = config or PatchEnvConfig()
        self.num_agents = self.cfg.num_agents
        self.render_mode = self.cfg.render_mode
        self.domain_randomize = self.cfg.domain_randomize

        self.action_model = PatchAction(PatchActionConfig())
        self.action_space = self.action_model.space

        # Compact default: [s, ey, patch_v, patch_theta] + lidar sectors.
        # Full observation remains available via config for ablations.
        self.compact_observation = bool(self.cfg.compact_observation)
        obs_dim = 20 if self.compact_observation else 48
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.f110 = F110EnvAdapter(
            F110Config(
                num_agents=self.num_agents,
                reset_type=self.cfg.base_reset_type,
                control_input=("speed", "steering_angle"),
                # timesteps = self.cfg.control_dt,  # old (invalid key)
                timestep=self.cfg.control_dt,  # new: F110Config expects `timestep`
            ),
            render_mode=self.render_mode,
        )
        self.base_env = self.f110  # Alias for visualization and other code
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
                    horizon_steps= self.cfg.mpc_horizon_steps,
                    horizon_seconds=self.cfg.control_dt * self.cfg.mpc_horizon_steps,
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

        #For Frenet bookkepping 
        self.track_spline = None
        self.track_length = None
        self.prev_s = None
        self.prev_steer = 0.0
        self.no_progress_counter = 0
        self.lap_count = 0
        self.prev_waypoint_idx = 0
        self.lap_start_step = 0

        # Navigation mode setup
        self.navigation_mode = self.cfg.navigation_mode
        if self.navigation_mode == "landmark":
            self.spawn_pose = np.array(self.cfg.spawn_pose, dtype=np.float32)
            self.goal_xy = np.array(self.cfg.goal_xy, dtype=np.float32)
            self.start_xy = (self.spawn_pose[0], self.spawn_pose[1])
            self.waypoints = None
            self._current_waypoint_idx = 0
        elif self.navigation_mode == "centerline":
            self.waypoints = None
            self._current_waypoint_idx = 0
            self.look_ahead = self.cfg.look_ahead_waypoints
            self.search_window = self.cfg.search_window
            # Will be initialized in reset() from track centerline
            self.spawn_pose = None
            self.goal_xy = None
            self.start_xy = None
        else:
            raise ValueError(f"Unknown navigation_mode: {self.navigation_mode}. Must be 'landmark' or 'centerline'")

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
        self.patch_wall_collisions = 0

        # Visualization attributes
        self._fig = None
        self._ax = None

        self.mpc_successes = 0
        self.mpc_attempts = 0
        self.safety_interventions = 0
        self.cached_controls = [None] * self.num_agents
        self.cached_feasible = [False] * self.num_agents
        self._last_reward_terms = {}

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
        if self.track_spline is None:
            # Fallback: create dummy observation or raise error
            raise RuntimeError("track_spline not initialized! Cannot compute Frenet coordinates.")
        lidar_distances = self.lidar_model.aggregate(self.current_base_obs, self.patch)
        lidar_distances = np.nan_to_num(lidar_distances, nan=10.0, posinf=10.0, neginf=0.0).astype(np.float32)
        lidar_norm = np.clip(lidar_distances / 10.0, 0.0, 1.0).astype(np.float32)
        s, ey = self._patch_to_frenet()

        if self.compact_observation:
            compact = np.concatenate(
                [
                    np.array(
                        [
                            float(s) / 100.0,
                            float(ey) / 10.0,
                            float(np.clip(self.patch.v / 10.0, 0.0, 1.5)),
                            float(self.patch.theta / np.pi),
                        ],
                        dtype=np.float32,
                    ),
                    lidar_norm,
                ],
                axis=0,
            )
            compact = np.nan_to_num(compact, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
            return np.clip(compact, -10.0, 10.0).astype(np.float32)

        clearance_obs = self.lidar_model.compute_clearance_observation(
            self.patch, lidar_distances
        )
        best_heading, best_clearance = self.lidar_model.get_best_heading(
            self.current_base_obs, self.patch
        )
        obs = self.obs_builder.build(
            patch=self.patch,
            start_xy=self.start_xy,
            goal_xy=self.goal_xy if self.navigation_mode == "landmark" else None,
            waypoints=self.waypoints if self.navigation_mode == "centerline" else None,
            current_waypoint_idx=self._current_waypoint_idx if self.navigation_mode == "centerline" else 0,
            look_ahead=self.look_ahead if self.navigation_mode == "centerline" else 10,
            search_window=self.search_window if self.navigation_mode == "centerline" else 30,
            navigation_mode=self.navigation_mode,
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
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    # Freenet helper functions 
    def _wrap_ds(self, ds: float) -> float:
        if self.track_length is None or not np.isfinite(self.track_length):
            return ds  # Safe fallback
        L = float(self.track_length)
        if L <= 0:
            return ds
        return (ds + 0.5 * L) % L - 0.5 * L

    def _patch_to_frenet(self) -> tuple[float, float]:
        if self.track_spline is None:
            # Return safe fallback values (like single-car would never reach here)
            return 0.0, 0.0
        
        try:
            s, ey = self.track_spline.calc_arclength_inaccurate(float(self.patch.x), float(self.patch.y))
            # Check for NaN (single-car doesn't need this but multi-agent might)
            if not np.isfinite(s) or not np.isfinite(ey):
                return 0.0, 0.0
            return float(s), float(ey)
        except (AttributeError, TypeError):
            return 0.0, 0.0

    def _compute_reward_w_o_frenet(self, lidar_info: dict) -> float:
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

        # Goal distance calculation based on navigation mode
        if self.navigation_mode == "landmark":
            _, _, goal_dist = self.obs_builder.goal_direction_and_distance(self.patch, self.goal_xy)
            if self._prev_goal_dist is None:
                self._prev_goal_dist = goal_dist
            dist_improve = self._prev_goal_dist - goal_dist
            reward += 8.0 * dist_improve
            self._prev_goal_dist = goal_dist
        elif self.navigation_mode == "centerline" and self.waypoints is not None:
            _, _, goal_dist, self._current_waypoint_idx = self.obs_builder.goal_direction_and_distance_centerline(
                self.patch, self.waypoints, self._current_waypoint_idx, self.look_ahead, self.search_window
            )
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

    def _compute_reward_w_frenet(self, lidar_info: dict, dt: float, lap_bonus: float = 0.0) -> float:
        # Frenet features
        s, ey = self._patch_to_frenet()
        ds = self._wrap_ds(s - self.prev_s) if self.prev_s is not None else 0.0

        # Patch controls / yaw-rate proxy
        steer_cmd = float(self.patch.steering)
        steer_rate = abs(steer_cmd - self.prev_steer) / max(dt, 1e-6)
        yaw_rate = float((self.patch.v / max(self.cfg.wheelbase, 1e-6)) * np.tan(steer_cmd))
        spin_excess = max(0.0, abs(yaw_rate) - self.cfg.spin_yawrate_threshold)

        # Collision proxy from lidar
        min_dist = float(lidar_info["min_dist"])
        collision = (not np.isfinite(min_dist)) or (min_dist < self.cfg.collision_min_dist)

        # Stuck counter
        if abs(ds) < self.cfg.stuck_progress_eps:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0

        reward_raw = (
            self.cfg.reward_progress_scale * ds
            - self.cfg.reward_crosstrack_weight * abs(ey)
            - self.cfg.reward_steer_bias_weight * abs(steer_cmd)
            - self.cfg.reward_steer_rate_weight * steer_rate
            - self.cfg.reward_spin_weight * spin_excess
            - (self.cfg.collision_penalty if collision else 0.0)
        )

        # Per-step time penalty
        reward_raw -= self.cfg.time_penalty_per_sec * dt

        # Event-based lap bonus
        reward_raw += float(lap_bonus)

        if self.no_progress_counter >= self.cfg.stuck_no_progress_steps:
            reward_raw -= self.cfg.stuck_penalty

        self.prev_s = s
        self.prev_steer = steer_cmd
        reward_clipped = float(np.clip(reward_raw, -100.0, 100.0))
        self._last_reward_terms = {
            "s": float(s),
            "ey": float(ey),
            "ds": float(ds),
            "steer_cmd": float(steer_cmd),
            "steer_rate": float(steer_rate),
            "yaw_rate_proxy": float(yaw_rate),
            "spin_excess": float(spin_excess),
            "collision_proxy": bool(collision),
            "reward_raw": float(reward_raw),
            "reward_clipped": float(reward_clipped),
            "reward_was_clipped": bool(abs(reward_raw) > 100.0),
            "no_progress_counter": int(self.no_progress_counter),
        }
        return reward_clipped

    #check termination with frenet compatible logic 
    def _check_termination(self, lidar_info: dict):
        terminated = False
        truncated = False
        reason = None

        # 1) Hard collision from map boundary check
        _, occ_map, resolution, origin = self.f110.get_track_data()
        patch_collision, violated = self.patch.check_patch_boundary_wall_collision(
            occ_map,
            resolution,
            origin,
            n_points=32,
            violation_threshold=self.cfg.patch_boundary_violation_threshold,
        )
        if patch_collision:
            return True, False, f"patch_wall_collision ({len(violated)} points)"

        # 2) Lidar safety collision proxy (ellipse-normalized metric)
        min_dist = float(lidar_info["min_dist"])
        if self.cfg.enable_lidar_termination and ((not np.isfinite(min_dist)) or (min_dist < self.cfg.collision_min_dist)):
            return True, False, "patch_wall_collision_lidar"

        # 3) Optional shape-failure termination (keep if you want shape safety)
        if self.patch.a < self._safe_min_a or self.patch.b < self._safe_min_b:
            return True, False, "patch_too_small"

        # 4) Off-track guard: cut hopeless episodes with very large lateral error.
        _, ey = self._patch_to_frenet()
        if abs(float(ey)) > self.cfg.offtrack_ey_termination:
            return True, False, "offtrack_ey"

        # 5) Stuck termination (aligned with frenet reward bookkeeping)
        if self.no_progress_counter >= self.cfg.stuck_no_progress_steps:
            return True, False, "stuck_no_progress"

        # 6) Time limit
        if self.step_count >= self.cfg.max_steps:
            truncated = True
            reason = "max_steps"

        return terminated, truncated, reason
    # def _check_termination(self, lidar_info: dict):
    #     terminated = False
    #     truncated = False
    #     reason = None

    #     track, occ_map, resolution, origin = self.f110.get_track_data()
    #     del track
    #     patch_collision, violated = self.patch.check_patch_boundary_wall_collision(
    #         occ_map, resolution, origin, n_points=32, violation_threshold=0.05
    #     )
    #     if patch_collision:
    #         return True, False, f"patch_wall_collision ({len(violated)} points)"

    #     min_dist = float(lidar_info["min_dist"])
    #     if not np.isfinite(min_dist) or min_dist < 0.95:
    #         return True, False, "patch_wall_collision_lidar"

    #     if self.patch.a < self._safe_min_a or self.patch.b < self._safe_min_b:
    #         return True, False, "patch_too_small"

    #     # Progress calculation based on navigation mode
    #     if self.navigation_mode == "landmark":
    #         progress, _, self._init_goal_dist = self.obs_builder.lap_progress(
    #             self.patch, self.goal_xy, self._init_goal_dist
    #         )
    #     elif self.navigation_mode == "centerline" and self.waypoints is not None:
    #         n_waypoints = len(self.waypoints)
    #         progress, _, self._current_waypoint_idx = self.obs_builder.lap_progress_centerline(
    #             self.patch, self.waypoints, self._current_waypoint_idx, n_waypoints, self.search_window
    #         )
    #     else:
    #         progress = 0.0
        
    #     if progress > 0.95 and self.lap_progress > 0.9:
    #         return True, False, "goal_reached"

    #     if self.step_count >= self.cfg.max_steps:
    #         truncated = True
    #         reason = "max_steps"
    #     return terminated, truncated, reason

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
        self.patch_wall_collisions = 0
        self.mpc_successes = 0
        self.mpc_attempts = 0
        self.safety_interventions = 0
        self.cached_controls = [None] * self.num_agents
        self.cached_feasible = [False] * self.num_agents
        self._last_reward_terms = {}

        #lap bookkepping reset 
        self.lap_count = 0
        self.prev_waypoint_idx = 0
        self.lap_start_step = 0

        # Initialize navigation based on mode
        if self.navigation_mode == "landmark":
            if options is not None and "spawn_pose" in options:
                self.spawn_pose = np.array(options["spawn_pose"], dtype=np.float32)
            if options is not None and "goal_xy" in options:
                self.goal_xy = np.array(options["goal_xy"], dtype=np.float32)
            
            patch_x, patch_y, patch_theta = map(float, self.spawn_pose)
            self.start_xy = (patch_x, patch_y)
        elif self.navigation_mode == "centerline":
            # Load waypoints from track centerline
            self.f110.ensure_initialized()
            track, _, _, _ = self.f110.get_track_data()
            if track.centerline is not None:
                # Original centerline spawn logic (kept commented as requested):
                # self.waypoints = np.column_stack([track.centerline.xs, track.centerline.ys]).astype(np.float32)
                # self._current_waypoint_idx = 0
                # patch_x, patch_y = float(self.waypoints[0, 0]), float(self.waypoints[0, 1])
                # if len(self.waypoints) > 1:
                #     dx = self.waypoints[1, 0] - self.waypoints[0, 0]
                #     dy = self.waypoints[1, 1] - self.waypoints[0, 1]
                #     patch_theta = float(np.arctan2(dy, dx))
                # else:
                #     patch_theta = 0.0
                # self.start_xy = (patch_x, patch_y)

                # New centerline spawn logic + Frenet initialization
                self.waypoints = np.column_stack([track.centerline.xs, track.centerline.ys]).astype(np.float32)
                self._current_waypoint_idx = 0
                self.prev_waypoint_idx = self._current_waypoint_idx
                patch_x, patch_y = float(self.waypoints[0, 0]), float(self.waypoints[0, 1])
                if len(self.waypoints) > 1:
                    dx = self.waypoints[1, 0] - self.waypoints[0, 0]
                    dy = self.waypoints[1, 1] - self.waypoints[0, 1]
                    patch_theta = float(np.arctan2(dy, dx))
                else:
                    patch_theta = 0.0
                self.start_xy = (patch_x, patch_y)
                if track.centerline is None or track.centerline.spline is None:
                    raise ValueError("Track centerline/spline missing for Frenet reward.")

                self.track_spline = track.centerline.spline
                self.track_length = float(self.track_spline.s[-1])

                # Stale Frenet init removed here; final init is done after patch state is created.
                # self.prev_steer = 0.0
                # self.no_progress_counter = 0
                # s0, _ = self._patch_to_frenet()
                # self.prev_s = s0
            else:
                raise ValueError("Track has no centerline! Cannot use centerline navigation mode.")

        # For centerline mode, we already initialized f110 above
        if self.navigation_mode == "landmark":
            self.f110.ensure_initialized()
        
        # _, occ_map, resolution, origin = self.f110.get_track_data()
        track, occ_map, resolution, origin = self.f110.get_track_data()
        # Keep Frenet data initialized for all navigation modes when centerline exists.
        if track.centerline is not None and track.centerline.spline is not None:
            self.track_spline = track.centerline.spline
            self.track_length = float(self.track_spline.s[-1])
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

        # Initialize Frenet bookkeeping after patch state is available.
        if self.track_spline is not None:
            self.prev_steer = 0.0
            self.no_progress_counter = 0
            self.prev_s, _ = self._patch_to_frenet()

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
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        return obs, {}

    def step(self, patch_action):
        self.step_count += 1
        dt = self.cfg.control_dt
        lap_bonus = 0.0
        lap_time = None

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
            cx=0.0,
            cy=0.0,
            cos_t=self.patch.cos_t,
            sin_t=self.patch.sin_t,
        )

        positions_world = [
            [base_obs["poses_x"][i], base_obs["poses_y"][i]] for i in range(self.num_agents)
        ]
        positions_local = [[p[0] - self.patch.x, p[1] - self.patch.y] for p in positions_world]

        env_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        solve_mpc = True
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
            # Base env control_input is ("speed", "steering_angle"), so order is [speed, steering].
            env_actions[i] = [float(np.clip(steering, -0.4, 0.4)), v_new]

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

        lidar_distances = self.lidar_model.aggregate(base_obs, self.patch)
        min_dist = float(np.min(lidar_distances))
        min_dist = float(np.nan_to_num(min_dist, nan=0.0, posinf=10.0, neginf=0.0))
        lidar_safe = bool(min_dist >= self.cfg.collision_min_dist)
        lidar_info = {"is_safe": lidar_safe, "min_dist": min_dist}
        if min_dist < self.cfg.collision_min_dist:
            clearance = -(self.cfg.collision_min_dist - min_dist)
        else:
            clearance = (min_dist - self.cfg.collision_min_dist)
        if np.isfinite(clearance):
            self.patch_min_clearance = min(self.patch_min_clearance, float(clearance))

        # Update lap progress based on navigation mode
        if self.navigation_mode == "landmark":
            self.lap_progress, _, self._init_goal_dist = self.obs_builder.lap_progress(
                self.patch, self.goal_xy, self._init_goal_dist
            )
        elif self.navigation_mode == "centerline" and self.waypoints is not None:
            n_waypoints = len(self.waypoints)
            prev_idx = int(self._current_waypoint_idx)
            self.lap_progress, _, self._current_waypoint_idx = self.obs_builder.lap_progress_centerline(
                self.patch, self.waypoints, self._current_waypoint_idx, n_waypoints, self.search_window
            )
            if (
                prev_idx > int(0.8 * n_waypoints)
                and int(self._current_waypoint_idx) < int(0.2 * n_waypoints)
            ):
                self.lap_count += 1
                lap_time = (self.step_count - self.lap_start_step) * dt
                lap_time = max(float(lap_time), 1e-3)
                lap_bonus = self.cfg.lap_finish_bonus * float(
                    np.exp(-lap_time / max(self.cfg.lap_bonus_tau, 1e-6))
                )
                self.lap_start_step = self.step_count

            self.prev_waypoint_idx = int(self._current_waypoint_idx)
        
        reward = self._compute_reward_w_frenet(lidar_info, dt, lap_bonus=lap_bonus)
        reward = float(np.nan_to_num(reward, nan=-10.0, posinf=100.0, neginf=-100.0))
        self.episode_reward += reward

        terminated, truncated, reason = self._check_termination(lidar_info)
        if self.cfg.use_base_done_termination:
            terminated = bool(terminated or base_done)
            truncated = bool(truncated or base_truncated)
        else:
            terminated = bool(terminated)
            truncated = bool(truncated)
        if reason is None and self.cfg.use_base_done_termination and base_done:
            reason = "base_env_done"
        if reason is None and self.cfg.use_base_done_termination and base_truncated:
            reason = "base_env_truncated"

        obs = self._build_obs()
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        reward_terms = dict(self._last_reward_terms)
        info = {
            "episode_reward": self.episode_reward,
            "lap_progress": self.lap_progress,
            "Episode_steps": self.step_count,
            "patch_size": (self.patch.a, self.patch.b),
            "patch_velocity": self.patch.v,
            "step_reward": reward,
            "termination_reason": reason,
            "lidar_safe": lidar_safe,
            "lap_count": int(self.lap_count),
            "lap_bonus": float(lap_bonus),
            "lap_time": None if lap_time is None else float(lap_time),
            "mpc_feasibility_rate": 1.0 if self.mpc_attempts == 0 else float(self.mpc_successes / self.mpc_attempts),
            "safety_intervention_rate": float(self.safety_interventions / max(1, self.step_count * self.num_agents)),
            "min_lidar_dist": float(min_dist),
            "collision_min_dist_threshold": float(self.cfg.collision_min_dist),
            "base_done": bool(base_done),
            "base_truncated": bool(base_truncated),
            "base_reset_type": str(self.cfg.base_reset_type),
            "use_base_done_termination": bool(self.cfg.use_base_done_termination),
        }
        info.update(reward_terms)

        if self.cfg.debug_print_every_n_steps > 0 and (self.step_count % self.cfg.debug_print_every_n_steps == 0):
            print(
                f"[PATCH-DEBUG] step={self.step_count} reward={reward:.2f} "
                f"raw={reward_terms.get('reward_raw', 0.0):.2f} ds={reward_terms.get('ds', 0.0):.4f} "
                f"ey={reward_terms.get('ey', 0.0):.3f} min_dist={min_dist:.3f} "
                f"mpc_feas={info['mpc_feasibility_rate']:.3f} safety_rate={info['safety_intervention_rate']:.3f}"
            )

        if self.cfg.debug_print_episode_end and (terminated or truncated):
            print(
                f"[PATCH-END] steps={self.step_count} reason={reason} ep_reward={self.episode_reward:.2f} "
                f"last_r={reward:.2f} raw={reward_terms.get('reward_raw', 0.0):.2f} "
                f"ds={reward_terms.get('ds', 0.0):.4f} ey={reward_terms.get('ey', 0.0):.3f} "
                f"min_dist={min_dist:.3f} clipped={reward_terms.get('reward_was_clipped', False)} "
                f"mpc_feas={info['mpc_feasibility_rate']:.3f} safety_rate={info['safety_intervention_rate']:.3f}"
            )

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.f110.render()

    def _visualize(self):
        """Visualize patch and agents with wall collision info."""
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(12, 9))
            plt.show(block=False)
        
        self._ax.clear()
        self._ax.set_aspect('equal')
        self._ax.grid(True, alpha=0.3)
        
        # Draw track walls from occupancy map
        occ_map = None
        resolution = None
        origin = None
        if self.base_env is not None:
            try:
                # Get track data from F110EnvAdapter
                track, occ_map, resolution, origin = self.base_env.get_track_data()

                # Calculate view bounds around patch
                margin = max(self.patch.a, self.patch.b) + 10
                x_min = self.patch.x - margin
                x_max = self.patch.x + margin
                y_min = self.patch.y - margin
                y_max = self.patch.y + margin
                
                # Convert to pixel coordinates
                px_min = max(0, int((x_min - origin[0]) / resolution))
                px_max = min(occ_map.shape[1], int((x_max - origin[0]) / resolution))
                py_min = max(0, int((y_min - origin[1]) / resolution))
                py_max = min(occ_map.shape[0], int((y_max - origin[1]) / resolution))
                
                if px_max > px_min and py_max > py_min:
                    occ_region = occ_map[py_min:py_max, px_min:px_max]
                    
                    # Create world coordinate arrays
                    x_world = np.arange(px_min, px_max) * resolution + origin[0]
                    y_world = np.arange(py_min, py_max) * resolution + origin[1]
                    
                    # Draw walls using contourf for filled regions
                    # Note: occupancy map convention - 0.0 = free space, 1.0 = occupied/wall
                    wall_mask = occ_region <0.5  # Walls are values > 0.5
                    if np.any(wall_mask):
                        # Create meshgrid for contour
                        X, Y = np.meshgrid(x_world, y_world)

                        # Draw filled wall regions (more visible)
                        self._ax.contourf(
                            X, Y, occ_region,
                            levels=[0.5, 1.0],
                            colors=['#333333'],  # Dark gray for walls
                            alpha=0.8,
                            extend='neither',
                            zorder=0  # Draw walls behind everything else
                        )

                        # Draw wall boundaries (more prominent)
                        self._ax.contour(
                            X, Y, occ_region,
                            levels=[0.5],
                            colors='black',
                            linewidths=2.0,
                            alpha=0.9,
                            zorder=0
                        )
                
                # Draw centerline
                if hasattr(track, 'centerline') and track.centerline is not None:
                    centerline_x = track.centerline.xs
                    centerline_y = track.centerline.ys
                    self._ax.plot(centerline_x, centerline_y, 'g--', 
                                 linewidth=1.5, alpha=0.4, label='Track Centerline', zorder=1)
                
            except (AttributeError, KeyError, Exception) as e:
                # If track info not available, skip wall drawing
                print(f"Warning: Could not draw track walls: {type(e).__name__}: {e}")
                pass
        
        agents_inside = sum(1 for i in range(self.num_agents)
                          if self.patch.is_inside(self.agent_states[i][0],
                                                  self.agent_states[i][1]))
        
        # Compute wall collision status live so visualization matches runtime behavior.
        patch_wall_collision = False
        violated = []
        if occ_map is not None and resolution is not None and origin is not None:
            patch_wall_collision, violated = self.patch.check_patch_boundary_wall_collision(
                occ_map,
                resolution,
                origin,
                n_points=32,
                violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )

        # Visualization-friendly clearance estimate from latest lidar aggregation.
        clearance = float("nan")
        if self.current_base_obs is not None:
            lidar_distances = self.lidar_model.aggregate(self.current_base_obs, self.patch)
            min_dist = float(np.nanmin(lidar_distances))
            min_dist = float(np.nan_to_num(min_dist, nan=0.0, posinf=10.0, neginf=0.0))
            clearance = min_dist - float(self.cfg.collision_min_dist)
        elif np.isfinite(self.patch_min_clearance):
            clearance = float(self.patch_min_clearance)
        
        # Title with wall collision warning
        wall_status = (
            f"⚠️ WALL! ({len(violated)} pts)"
            if patch_wall_collision
            else f"Clear: {clearance:.2f}m"
        )
        self._ax.set_title(
            f'Patch Funnel V1 | Step {self.step_count} | Progress: {self.lap_progress:.1%}\n'
            f'Patch: v={self.patch.v:.1f}m/s, size=({self.patch.a:.1f}, {self.patch.b:.1f}) | '
            f'Inside: {agents_inside}/{self.num_agents} | {wall_status}',
            fontsize=11,
            color='red' if patch_wall_collision else 'black'
        )
        
        # Draw ellipsoid patch - RED if hitting wall
        if patch_wall_collision:
            face_color = 'red'
            edge_color = 'darkred'
            alpha = 0.4
        elif agents_inside == self.num_agents:
            face_color = 'cyan'
            edge_color = 'darkblue'
            alpha = 0.3
        else:
            face_color = 'yellow'
            edge_color = 'orange'
            alpha = 0.3
        
        ellipse = Ellipse(
            xy=(self.patch.x, self.patch.y),
            width=self.patch.a * 2,
            height=self.patch.b * 2,
            angle=np.degrees(self.patch.theta),
            facecolor=face_color,
            edgecolor=edge_color,
            alpha=alpha,
            linewidth=3
        )
        self._ax.add_patch(ellipse)
        
        # Draw patch boundary points to show wall collision check
        boundary_points = self.patch.get_boundary_points(16)
        for bx, by in boundary_points:
            self._ax.plot(bx, by, 'k.', markersize=4, alpha=0.5)
        
        # Draw patch center
        self._ax.plot(self.patch.x, self.patch.y, 'b+', markersize=15, markeredgewidth=2)
        
        # Draw patch velocity arrow
        vx, vy = self.patch.get_velocity_vector()
        self._ax.arrow(
            self.patch.x, self.patch.y,
            vx * 0.3, vy * 0.3,
            head_width=0.2, head_length=0.15,
            fc='blue', ec='blue', linewidth=2
        )
        
        # Draw agents
        colors = ['red', 'orange', 'purple', 'green']
        for i in range(self.num_agents):
            x, y, theta, v = self.agent_states[i]
            inside = self.patch.is_inside(x, y)
            collision = self.current_base_obs["collisions"][i] > 0.5
            
            color = colors[i % len(colors)]
            if collision:
                marker = 'X'
                size = 18
            elif inside:
                marker = 'o'
                size = 12
            else:
                marker = 's'  # Square if outside patch
                size = 14
            
            self._ax.plot(x, y, marker, color=color, markersize=size,
                         markeredgecolor='black', markeredgewidth=2)
            
            # Agent velocity arrow
            self._ax.arrow(
                x, y,
                v * np.cos(theta) * 0.2, v * np.sin(theta) * 0.2,
                head_width=0.1, head_length=0.05,
                fc=color, ec=color, alpha=0.7
            )
            
            # Label agents
            self._ax.text(x + 0.3, y + 0.3, f'R{i}', fontsize=8, fontweight='bold')
        
        # Draw next waypoint if available
        if self.waypoints is not None and len(self.waypoints) > 0:
            # Find nearest waypoint
            patch_pos = np.array([self.patch.x, self.patch.y])
            dists = np.linalg.norm(self.waypoints - patch_pos, axis=1)
            nearest_idx = np.argmin(dists)
            next_idx = (nearest_idx + 10) % len(self.waypoints)
            self._ax.plot(self.waypoints[next_idx, 0], self.waypoints[next_idx, 1],
                         'g*', markersize=20, markeredgecolor='black', markeredgewidth=1)
        
        # Info box
        info_text = (
            f'Reward: {self.episode_reward:.1f}\n'
            f'Patch collisions: {self.patch_wall_collisions}\n'
            f'Min clearance: {self.patch_min_clearance:.2f}m'
        )
        self._ax.text(0.02, 0.98, info_text, transform=self._ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set limits around patch
        margin = max(self.patch.a, self.patch.b) + 5
        self._ax.set_xlim(self.patch.x - margin, self.patch.x + margin)
        self._ax.set_ylim(self.patch.y - margin, self.patch.y + margin)
        
        plt.pause(0.001)

        try:
            self._ax.figure.canvas.draw()
            self._ax.figure.canvas.flush_events()
        except Exception as e:
            pass

    def close(self):
        if hasattr(self, '_fig') and self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        self.f110.close()


def make_patch_env(
    rank: int,
    seed: int = 0,
    domain_randomize: bool = False,
    navigation_mode: str = "landmark",
    debug_print_every_n_steps: int = 0,
    debug_print_episode_end: bool = True,
    base_reset_type: str = "rl_random_static",
    use_base_done_termination: bool = False,
):
    """
    Factory function to create patch environments for parallel training.
    
    Args:
        rank: Environment rank for seeding
        seed: Base seed
        domain_randomize: Whether to use domain randomization
        navigation_mode: "landmark" or "centerline" - determines navigation strategy
    
    Example:
        # Use landmark-based navigation (default)
        env = make_patch_env(0, seed=42, navigation_mode="landmark")
        
        # Use centerline-based navigation
        env = make_patch_env(0, seed=42, navigation_mode="centerline")
    """
    def _init():
        cfg = PatchEnvConfig(
            domain_randomize=domain_randomize,
            num_agents=2,
            render_mode=None,
            navigation_mode=navigation_mode,
            debug_print_every_n_steps=debug_print_every_n_steps,
            debug_print_episode_end=debug_print_episode_end,
            base_reset_type=base_reset_type,
            use_base_done_termination=use_base_done_termination,
        )
        env = PatchEnv(cfg)
        env.reset(seed=seed + rank)
        return env

    if SB3_AVAILABLE:
        set_random_seed(seed + rank)
    return _init




