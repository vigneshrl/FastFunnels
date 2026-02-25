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


# @dataclass
# class PatchEnvConfig:
#     num_agents: int = 0
#     control_dt = 0.05
#     # NOT USED: MPC horizon (patch_only_mode=True, num_agents=0)
#     # mpc_horizon_steps: int = 10
#     base_reset_type: str = "rl_random_static"
#     render_mode: Optional[str] = None
#     domain_randomize: bool = False
#     robot_radius: float = 0.15
#     wheelbase: float = 0.33
#     alpha: float = 0.1
#     max_steps: int = 2000
#     # NOT USED: goal-based navigation (using Frenet-only)
#     # goal_reached_radius: float = 1.5


#     # Frenet-style reward - which is very similar to the single-agent PPO 
#     # old: reward_progress_scale: float = 40.0
#     reward_progress_scale: float = 40.0
#     reward_crosstrack_weight: float = 4.0
#     reward_steer_bias_weight: float = 0.0
#     reward_steer_rate_weight: float = 0.0
#     spin_yawrate_threshold: float = 0.0
#     reward_spin_weight: float = 0.0
#     collision_penalty: float = 150.0 #High penalty make the policy more conservative
#     collision_min_dist: float = 0.25
#     enable_lidar_termination: bool = False
#     patch_boundary_violation_threshold: float = 0.05
#     offtrack_ey_termination: float = 8.0
#     use_frenet_proxy_lidar: bool = True
#     # old: frenet_edge_penalty_weight: float = 25.0
#     frenet_edge_penalty_weight: float = 80.0
#     trackwidth_violation_penalty: float = 500.0
#     # old: patch_wall_collision_penalty: float = 250.0
#     patch_wall_collision_penalty: float = 1500.0
#     edge_guard_enabled: bool = True
#     lidar_min_dist_termination = False
#     # old: edge_guard_start_ratio: float = 0.55
#     edge_guard_start_ratio: float = 0.35
#     # old: edge_guard_full_ratio: float = 0.90
#     edge_guard_full_ratio: float = 0.70
#     edge_guard_min_accel_scale: float = 0.35
#     # old: edge_guard_min_steer_scale: float = 0.70
#     edge_guard_min_steer_scale: float = 1.25
#     # old: edge_guard_centering_steer_gain: float = 0.12
#     edge_guard_centering_steer_gain: float = 0.22
#     edge_guard_min_max_a_scale: float = 0.75
#     edge_guard_min_max_b_scale: float = 0.65
#     # old: shape_aspect_ratio_cap: float = 3.0
#     shape_aspect_ratio_cap: float = 1.2
#     shape_aspect_ratio_penalty_weight: float = 8.0
#     shape_area_penalty_weight: float = 2.0
#     corner_kappa_ref: float = 0.22
#     corner_shrink_rate_boost: float = 3.0
#     corner_b_target_min_scale: float = 0.55
#     corner_a_target_min_scale: float = 0.75
#     corner_b_target_penalty_weight: float = 30.0
#     corner_a_target_penalty_weight: float = 12.0
#     patch_size_cap_enabled: bool = True
#     patch_size_cap_reference_agents: int = 6
#     # old: patch_size_cap_leeway: float = 1.25
#     patch_size_cap_leeway: float = 0.85
#     patch_size_cap_penalty_weight: float = 200.0
#     patch_size_softcap_start_ratio: float = 0.90
#     patch_size_softcap_penalty_weight: float = 40.0
#     # old: patch_only_min_b_scale: float = 0.60
#     patch_only_min_b_scale: float = 0.95
#     patch_only_speed_floor: float = 0.9 # decrease if you want to make the car to move at slow speeds at corners
#     patch_only_ey_termination_ratio: float = 0.75
#     patch_only_corner_speed_reduction_gain: float = 0.45
#     patch_only_corner_speed_min: float = 0.8
#     ey_termination_enabled: bool = False
#     # old: patch_only_centerline_assist_enabled: bool = True
#     # NOT USED: patch centerline assist (disabled)
#     # patch_only_centerline_assist_enabled: bool = False
#     # patch_only_assist_blend: float = 0.55
#     # patch_only_heading_gain: float = 1.0
#     # patch_only_ey_gain: float = 0.45
#     # patch_only_curvature_ff_gain: float = 1.0
#     # patch_only_corner_slowdown_gain: float = 0.8
#     # patch_only_corner_heading_thresh: float = 0.30
#     # NOT USED: agent-related parameters (patch_only_mode=True, num_agents=0)
#     # agent_lidar_brake_dist: float = 1.0
#     # use_agent_lidar_braking: bool = False
#     # agent_speed_boost: float = 1.0
#     # stable_agent_speed_cap: float = 8.0
#     # hard_speed_cap: float = 6.0
#     # agent_min_speed_cmd: float = 1.2
#     # agent_lag_speed_gain: float = 0.6
#     # agent_accel_clip: float = 3.0
#     # max_steer_delta_per_step: float = 0.12
#     # f1tenth_gym in this setup behaves as if action is [steering, speed].
#     # Keep this default True unless you have verified semantic channel ordering end-to-end.
#     force_legacy_steer_speed_order: bool = True
#     nonfinite_state_penalty: float = 500.0
#     # NOT USED: agent-patch coupling (patch_only_mode=True, num_agents=0)
#     # coupling_enabled: bool = True
#     # coupling_lag_threshold_m: float = 0.8
#     # coupling_patch_speed_scale: float = 0.35

#     stuck_no_progress_steps: int = 60 #It was 60, increased that the agent gets more steps to find the right steering 
#     stuck_progress_eps: float = 1e-3
#     stuck_penalty: float = 10.0
#     lap_finish_bonus: float = 4000.0
#     lap_bonus_tau = 200.0
#     time_penalty_per_sec: float = 0.0
#     compact_observation: bool = True
#     debug_print_every_n_steps: int = 0
#     debug_print_episode_end: bool = True
#     use_base_done_termination: bool = False
#     patch_only_mode: bool = False

#     # Navigation mode: "landmark" or "centerline"
#     navigation_mode: str = "centerline"
    
#     # NOT USED: Landmark-based setup (using centerline/Frenet-only)
#     # spawn_pose: tuple[float, float, float] = (-46.6, 27.0, 2.25)
#     # goal_xy: tuple[float, float] = (-52.8, 19.33)
    
#     # NOT USED: Centerline waypoint-based setup (using Frenet-only, no waypoints)
#     # look_ahead_waypoints: int = 20  # How many waypoints ahead to look for goal direction
#     # search_window: int = 150  # Window size for finding nearest waypoint


# class PatchEnv(gym.Env):
#     """
#     Patch-policy environment with dual navigation modes.
    
#     Navigation Modes:
#     - "landmark": Uses fixed spawn_pose and goal_xy for navigation
#     - "centerline": Uses track centerline waypoints for navigation
    
#     Quick Swap Example:
#         # Landmark mode (default)
#         cfg = PatchEnvConfig(navigation_mode="landmark")
#         env = PatchEnv(cfg)
        
#         # Centerline mode
#         cfg = PatchEnvConfig(navigation_mode="centerline")
#         env = PatchEnv(cfg)
#     """
#     metadata = {"render_modes": ["human", "rgb_array", None]}

#     def __init__(self, config: Optional[PatchEnvConfig] = None):
#         super().__init__()
#         self.cfg = config or PatchEnvConfig()
#         self.num_agents = self.cfg.num_agents
#         self.render_mode = self.cfg.render_mode
#         self.domain_randomize = self.cfg.domain_randomize

#         self.action_model = PatchAction(PatchActionConfig())
#         self.action_space = self.action_model.space

#         # Compact default: [s, ey, patch_v, patch_theta] + lidar sectors.
#         # Full observation remains available via config for ablations.
#         self.compact_observation = bool(self.cfg.compact_observation)
#         obs_dim = 20 if self.compact_observation else 48
#         self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

#         self.f110 = F110EnvAdapter(
#             F110Config(
#                 num_agents=self.num_agents,
#                 reset_type=self.cfg.base_reset_type,
#                 control_input=("speed", "steering_angle"),
#                 # timesteps = self.cfg.control_dt,  # old (invalid key)
#                 timestep=self.cfg.control_dt,  # new: F110Config expects `timestep`
#             ),
#             render_mode=self.render_mode,
#         )
#         self.base_env = self.f110  # Alias for visualization and other code
#         self.map_reset = MapResetHelper()
#         self.lidar_model = LidarModel()
#         self.obs_builder = PatchObservationBuilder()
#         self.patch = DynamicPatch()

#         if self.cfg.patch_only_mode:
#             # old: MPC/Safety objects were always created.
#             self.mpc_solvers = []
#             self.safety_layer = None
#         # else:  # NOT USED: patch_only_mode=True, MPC/Safety never created
#         #     self.mpc_solvers = [
#         #         SEMPCSolver(
#         #             MPCConfig(
#         #                 robot_radius=self.cfg.robot_radius,
#         #                 wheelbase=self.cfg.wheelbase,
#         #                 num_neighbors=self.num_agents - 1,
#         #                 containment_margin=self.cfg.robot_radius,
#         #                 min_agent_dist=2 * self.cfg.robot_radius + 0.1,
#         #                 horizon_steps=self.cfg.mpc_horizon_steps,
#         #                 horizon_seconds=self.cfg.control_dt * self.cfg.mpc_horizon_steps,
#         #             )
#         #         )
#         #         for _ in range(self.num_agents)
#         #     ]
#         #     self.safety_layer = SafetyLayer(
#         #         SafetyConfig(
#         #             robot_radius=self.cfg.robot_radius,
#         #             wheelbase=self.cfg.wheelbase,
#         #             min_agent_dist=2 * self.cfg.robot_radius + 0.12,
#         #             alpha_contain=1.0,
#         #             alpha_collision=1.0,
#         #         )
#         #     )

#         self.current_base_obs = None
#         self.agent_states = None
#         self.prev_v = None
#         self.prev_agent_steer = None
#         self._np_random = None

#         #For Frenet bookkepping 
#         self.track_spline = None
#         self.track_length = None
#         self.prev_s = None
#         self.prev_steer = 0.0
#         self.no_progress_counter = 0
#         self.lap_count = 0
#         self.prev_waypoint_idx = 0
#         self.lap_start_step = 0

#         # Navigation mode setup
#         self.navigation_mode = self.cfg.navigation_mode
#         # NOT USED: landmark navigation mode (using centerline/Frenet-only)
#         # if self.navigation_mode == "landmark":
#         #     self.spawn_pose = np.array(self.cfg.spawn_pose, dtype=np.float32)
#         #     self.goal_xy = np.array(self.cfg.goal_xy, dtype=np.float32)
#         #     self.start_xy = (self.spawn_pose[0], self.spawn_pose[1])
#         #     self.waypoints = None
#         #     self._current_waypoint_idx = 0
#         if self.navigation_mode == "centerline":
#             # NOT USED: waypoint-based navigation (using Frenet-only)
#             # self.waypoints = None
#             # self._current_waypoint_idx = 0
#             # self.look_ahead = self.cfg.look_ahead_waypoints
#             # self.search_window = self.cfg.search_window
#             # Will be initialized in reset() from track centerline
#             # self.spawn_pose = None
#             # self.goal_xy = None
#             self.start_xy = None  # Will be set in reset()
#         else:
#             raise ValueError(f"Unknown navigation_mode: {self.navigation_mode}. Must be 'landmark' or 'centerline'")

#         self._safe_init_a = 1.5
#         self._safe_init_b = 1.0
#         self._safe_min_a = 1.0
#         self._safe_min_b = 0.25
#         self.patch_a_range = (self._safe_min_a, self._safe_init_a)
#         self.patch_b_range = (self._safe_min_b, self._safe_init_b)

#         self.step_count = 0
#         self.episode_reward = 0.0
#         self.lap_progress = 0.0
#         # NOT USED: landmark/goal-based progress tracking (using Frenet-only)
#         # self._init_goal_dist = None
#         # self._prev_goal_dist = None
#         # self._prev_patch_pos = None
#         self.patch_min_clearance = float("inf")
#         self.patch_wall_collisions = 0

#         # Visualization attributes
#         self._fig = None
#         self._ax = None

#         self.mpc_successes = 0
#         self.mpc_attempts = 0
#         self.safety_interventions = 0
#         self.cached_controls = [None] * self.num_agents
#         self.cached_feasible = [False] * self.num_agents
#         self._last_reward_terms = {}
#         self._last_track_width = None
#         self._hard_cap_a = None
#         self._hard_cap_b = None
@dataclass
class PatchEnvConfig:
    """
    Simplified config matching single-agent PPO structure.
    Patch is treated as an inflated agent with fixed size.
    """
    # Environment basics
    num_agents: int = 0  # Patch-only, no agents
    control_dt: float = 0.05
    base_reset_type: str = "rl_random_static"
    render_mode: Optional[str] = None
    max_steps: int = 100_000
    domain_randomize: bool = False
    
    # Frenet reward (same as single-agent)
    reward_progress_scale: float = 40.0
    reward_crosstrack_weight: float = 2.0
    reward_steer_bias_weight: float = 0.0
    reward_steer_rate_weight: float = 0.0
    spin_yawrate_threshold: float = 1.5
    reward_spin_weight: float = 0.5 
    collision_penalty: float = 1500.0
    collision_min_dist: float = 0.25
    stuck_no_progress_steps: int = 60
    stuck_progress_eps: float = 1e-3
    stuck_penalty: float = 10.0
    lap_finish_bonus: float = 2000.0
    lap_bonus_tau: float = 200.0
    time_penalty_per_sec: float = 0.0
    
    # Lidar config (match single-agent)
    num_beams: int = 108  # Same as single-agent ENV_CONFIG
    
    # Vehicle dynamics
    wheelbase: float = 0.33
    robot_radius: float = 0.15
    
    # Patch size (FIXED - no control, like single-agent has fixed robot size)
    patch_a: float = 1.5  # Fixed size (semi-major axis)
    patch_b: float = 1.0  # Fixed size (semi-minor axis)
    
    # Patch boundary collision detection (using discretization)
    patch_boundary_violation_threshold: float = 0.05  # Fraction of boundary points that must be in wall to trigger collision
    
    # OLD COMPLEX CONFIG (commented out - not used in simplified version)
    # domain_randomize: bool = False
    # alpha: float = 0.1
    # enable_lidar_termination: bool = False
    # patch_boundary_violation_threshold: float = 0.05
    # offtrack_ey_termination: float = 8.0
    # use_frenet_proxy_lidar: bool = True
    # frenet_edge_penalty_weight: float = 80.0
    # trackwidth_violation_penalty: float = 500.0
    # patch_wall_collision_penalty: float = 1500.0
    # edge_guard_enabled: bool = True
    # lidar_min_dist_termination = False
    # edge_guard_start_ratio: float = 0.35
    # edge_guard_full_ratio: float = 0.70
    # edge_guard_min_accel_scale: float = 0.35
    # edge_guard_min_steer_scale: float = 1.25
    # edge_guard_centering_steer_gain: float = 0.22
    # edge_guard_min_max_a_scale: float = 0.75
    # edge_guard_min_max_b_scale: float = 0.65
    # shape_aspect_ratio_cap: float = 1.2
    # shape_aspect_ratio_penalty_weight: float = 8.0
    # shape_area_penalty_weight: float = 2.0
    # corner_kappa_ref: float = 0.22
    # corner_shrink_rate_boost: float = 3.0
    # corner_b_target_min_scale: float = 0.55
    # corner_a_target_min_scale: float = 0.75
    # corner_b_target_penalty_weight: float = 30.0
    # corner_a_target_penalty_weight: float = 12.0
    # patch_size_cap_enabled: bool = True
    # patch_size_cap_reference_agents: int = 6
    # patch_size_cap_leeway: float = 0.85
    # patch_size_cap_penalty_weight: float = 200.0
    # patch_size_softcap_start_ratio: float = 0.90
    # patch_size_softcap_penalty_weight: float = 40.0
    # patch_only_min_b_scale: float = 0.95
    # patch_only_speed_floor: float = 0.9
    # patch_only_ey_termination_ratio: float = 0.75
    # patch_only_corner_speed_reduction_gain: float = 0.45
    # patch_only_corner_speed_min: float = 0.8
    # ey_termination_enabled: bool = False
    # compact_observation: bool = True
    # debug_print_every_n_steps: int = 0
    # debug_print_episode_end: bool = True
    # use_base_done_termination: bool = False
    # patch_only_mode: bool = False
    # navigation_mode: str = "centerline"
    # force_legacy_steer_speed_order: bool = True
    # nonfinite_state_penalty: float = 500.0

class PatchEnv(gym.Env):
    """
    Simplified patch environment matching single-agent structure.
    Patch is treated as an inflated agent with fixed size.
    """
    metadata = {"render_modes": ["human", "rgb_array", None]}
    
    def __init__(self, config: Optional[PatchEnvConfig] = None):
        super().__init__()
        self.cfg = config or PatchEnvConfig()
        
        # Simple action space: [steering, speed] like single-agent
        self.action_space = spaces.Box(
            low=np.array([-0.4189, 0.5, 1.5, 1.0], dtype=np.float32),
            high=np.array([0.4189, 10.0, 2.0, 1.5], dtype=np.float32),
            dtype=np.float32,
        )
        
        # Observation space: [s, ey, v, yaw] + lidar (same as single-agent)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(4 + self.cfg.num_beams,),  # Match single-agent: 4 + num_beams
            dtype=np.float32
        )
        
        # Base environment (simplified - like single-agent)
        # num_agents=1 for real lidar (like single-agent code)
        # Agent follows patch to provide lidar, but agent is NOT trained (only patch is trained)
        self.f110 = F110EnvAdapter(
            F110Config(
                num_agents=1,  # Need 1 agent for real lidar (like single-agent), positioned at patch location
                reset_type=self.cfg.base_reset_type,
                control_input=("speed", "steering_angle"),
                timestep=self.cfg.control_dt,
            ),
            render_mode=self.cfg.render_mode,
        )
        # OLD: num_agents=0 with proxy lidar (commented out - using real lidar instead)
        # self.f110 = F110EnvAdapter(
        #     F110Config(
        #         num_agents=0,  # No agents in base env - agents controlled by MPC later
        #         reset_type=self.cfg.base_reset_type,
        #         control_input=("speed", "steering_angle"),
        #         timestep=self.cfg.control_dt,
        #     ),
        #     render_mode=self.cfg.render_mode,
        # )
        self.base_env = self.f110  # Alias for compatibility
        
        # Map reset helper (needed for track width estimation and other utilities)
        self.map_reset = MapResetHelper()
        # OLD: Comment said "for proxy lidar" but it's also used for other things
        
        # Patch with FIXED size (like single-agent has fixed robot size)
        self.patch = DynamicPatch()
        self.patch.a = self.cfg.patch_a  # Fixed size
        self.patch.b = self.cfg.patch_b  # Fixed size
        
        # Frenet bookkeeping (same as single-agent F1tenthWrapper)
        self.track_spline = None
        self.track_length = None
        self.prev_s = None
        self.prev_steer = 0.0
        self.no_progress_counter = 0
        self.lap_count = 0
        self.lap_start_step = 0
        
        # Episode tracking (same as single-agent)
        self.step_count = 0
        self.current_base_obs = None
        

        self.num_agents = 1
        # OLD COMPLEX INIT CODE (commented out - not used in simplified version)
        # self.action_model = PatchAction(PatchActionConfig())
        # self.map_reset = MapResetHelper()
        # self.lidar_model = LidarModel()
        # self.obs_builder = PatchObservationBuilder()
        # self.compact_observation = bool(self.cfg.compact_observation)
        # self.domain_randomize = self.cfg.domain_randomize
        # self.navigation_mode = self.cfg.navigation_mode
        # self._safe_init_a = 1.5
        # self._safe_init_b = 1.0
        # self._safe_min_a = 1.0
        # self._safe_min_b = 0.25
        # self.patch_a_range = (self._safe_min_a, self._safe_init_a)
        # self.patch_b_range = (self._safe_min_b, self._safe_init_b)
        # self.episode_reward = 0.0
        # self.lap_progress = 0.0
        # self.patch_min_clearance = float("inf")
        # self.patch_wall_collisions = 0
        self._fig = None
        self._ax = None
        # self.mpc_successes = 0
        # self.mpc_attempts = 0
        # self.safety_interventions = 0
        # self.cached_controls = []
        # self.cached_feasible = []
        # self._last_reward_terms = {}
        # self._last_track_width = None
        # self._hard_cap_a = None
        # self._hard_cap_b = None
        # self.agent_states = None
        # self.prev_v = None
        # self.prev_agent_steer = None
        # self._np_random = None

        
    # OLD ACTION HANDLING METHODS (commented out - simplified version uses direct [steering, speed] actions)
    # def _pack_base_action(self, steering_cmd: float, speed_cmd: float) -> np.ndarray:
    #     """Build one agent action according to configured base control_input semantics."""
    #     if self.cfg.force_legacy_steer_speed_order:
    #         return np.array([float(steering_cmd), float(speed_cmd)], dtype=np.float32)
    #     control_names = list(self.f110.config.control_input)
    #     action = np.zeros((2,), dtype=np.float32)
    #     for idx, name in enumerate(control_names):
    #         key = str(name).lower()
    #         if key == "speed":
    #             action[idx] = float(speed_cmd)
    #         elif key == "steering_angle":
    #             action[idx] = float(steering_cmd)
    #         else:
    #             action[idx] = float(steering_cmd if idx == 0 else speed_cmd)
    #     return action
    #
    # def _clip_base_actions(self, actions: np.ndarray) -> np.ndarray:
    #     """Clip actions by semantics."""
    #     clipped = np.nan_to_num(actions, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    #     if self.cfg.force_legacy_steer_speed_order:
    #         speed_cap = min(float(self.cfg.stable_agent_speed_cap), float(self.cfg.hard_speed_cap))
    #         clipped[:, 0] = np.clip(clipped[:, 0], -0.4189, 0.4189)  # steering
    #         clipped[:, 1] = np.clip(clipped[:, 1], 0.5, speed_cap)    # speed
    #         return clipped
    #     control_names = list(self.f110.config.control_input)
    #     for idx, name in enumerate(control_names):
    #         key = str(name).lower()
    #         if key == "speed":
    #             speed_cap = min(float(self.cfg.stable_agent_speed_cap), float(self.cfg.hard_speed_cap))
    #             clipped[:, idx] = np.clip(clipped[:, idx], 0.5, speed_cap)
    #         elif key == "steering_angle":
    #             clipped[:, idx] = np.clip(clipped[:, idx], -0.4189, 0.4189)
    #         else:
    #             clipped[:, idx] = clipped[:, idx]
    #     return clipped
    #
    # def _effective_action_indices(self) -> tuple[int, int]:
    #     """Returns (steer_idx, speed_idx) for the action array actually sent to base env."""
    #     if self.cfg.force_legacy_steer_speed_order:
    #         return 0, 1
    #     control_names = [str(n).lower() for n in self.f110.config.control_input]
    #     steer_idx = control_names.index("steering_angle") if "steering_angle" in control_names else 0
    #     speed_idx = control_names.index("speed") if "speed" in control_names else 1
    #     return steer_idx, speed_idx
    #
    # def _calculate_min_patch_size_for_agents(self):
    #     """Calculate minimum patch size to contain agents (not used in simplified version)."""
    #     if self.agent_states is None:
    #         return self.cfg.robot_radius * 2, self.cfg.robot_radius * 2
    #     positions = []
    #     for i in range(self.num_agents):
    #         x, y, _, _ = self.agent_states[i]
    #         x_p, y_p = self.patch.world_to_patch_frame(x, y)
    #         positions.append((x_p, y_p))
    #     if not positions:
    #         return self.cfg.robot_radius * 2, self.cfg.robot_radius * 2
    #     arr = np.asarray(positions)
    #     margin = self.cfg.robot_radius * 1.5
    #     min_a = max(np.max(np.abs(arr[:, 0])) + margin, 0.5)
    #     min_b = max(np.max(np.abs(arr[:, 1])) + margin, 0.5)
    #     return float(min_a), float(min_b)

    def _get_scan(self, obs, num_beams: int, i: int = 0) -> np.ndarray:
        """
        Get lidar scan and pad/trim to num_beams (same as single-agent F1tenthWrapper._get_scan).
        
        Args:
            obs: Observation dict from base env
            num_beams: Expected number of lidar beams
            i: Agent index (always 0 for patch-only)
        
        Returns:
            Scan array of shape (num_beams,) in meters
        """
        if "scans" in obs and len(obs["scans"]) > i:
            scan = np.asarray(obs["scans"][i], dtype=np.float32)
        else:
            scan = np.zeros((num_beams,), dtype=np.float32)
        
        # Handle unit conversion (same as single-agent - check if normalized)
        if np.nanmax(scan) <= 1.5:
            scan = scan * 10.0  # Convert from normalized [0,1] to meters
        
        # Pad or trim to expected num_beams (same as single-agent)
        if scan.shape[0] != num_beams:
            if scan.shape[0] > num_beams:
                scan = scan[:num_beams]
            else:
                scan = np.pad(scan, (0, num_beams - scan.shape[0]), constant_values=10.0)
        
        return scan.astype(np.float32)
    
    def _build_obs(self) -> np.ndarray:
        """
        Build observation exactly like single-agent: [s, ey, v, yaw] + lidar.
        Uses real lidar from agent (like single-agent code).
        NO normalization applied (raw values like single-agent).
        """
        if self.track_spline is None:
            raise RuntimeError("track_spline not initialized! Cannot compute Frenet coordinates.")
        
        # Get Frenet coordinates (same as single-agent)
        s, ey = self._patch_to_frenet()
        
        # Get patch state (equivalent to agent state in single-agent)
        speed = float(self.patch.v)
        yaw = float(self.patch.theta)
        
        # Get lidar scan from agent (same as single-agent - full scan, not aggregated sectors)
        scan = self._get_scan(self.current_base_obs, self.cfg.num_beams, i=0)
        
        # Concatenate WITHOUT normalization (match single-agent exactly)
        obs = np.concatenate([
            [s, ey, speed, yaw],  # Raw values, no division
            scan
        ], axis=0).astype(np.float32)
        
        return obs
        
        # OLD: Proxy lidar (Frenet-based) - commented out, using real lidar instead
        # # Use proxy lidar (Frenet-based) like single-agent code
        # track_width = self._estimate_current_track_width()
        # if track_width is not None:
        #     proxy_lidar = self._frenet_proxy_lidar(ey=ey, track_width=track_width)
        #     # Expand from n_sectors (16) to num_beams (108) by repeating/interpolating
        #     scan = np.repeat(proxy_lidar, self.cfg.num_beams // len(proxy_lidar))[:self.cfg.num_beams]
        #     if len(scan) < self.cfg.num_beams:
        #         scan = np.pad(scan, (0, self.cfg.num_beams - len(scan)), constant_values=scan[-1] if len(scan) > 0 else 10.0)
        # else:
        #     # Fallback: max range if no track width available
        #     scan = np.ones(self.cfg.num_beams, dtype=np.float32) * 10.0
        
        # OLD COMPLEX OBSERVATION CODE (commented out - not used in simplified version)
        # # track_width = self._estimate_current_track_width()
        # # self._last_track_width = track_width
        # # if self.cfg.use_frenet_proxy_lidar:
        # #     lidar_distances = self._frenet_proxy_lidar(ey=ey, track_width=track_width)
        # # else:
        # #     lidar_distances = self.lidar_model.aggregate(self.current_base_obs, self.patch)
        # # lidar_distances = np.nan_to_num(lidar_distances, nan=10.0, posinf=10.0, neginf=0.0).astype(np.float32)
        # # lidar_norm = np.clip(lidar_distances / 10.0, 0.0, 1.0).astype(np.float32)
        # # if self.compact_observation:
        # #     compact = np.concatenate(
        # #         [
        # #             np.array(
        # #                 [
        # #                     float(s) / 100.0,
        # #                     float(ey) / 10.0,
        # #                     float(np.clip(self.patch.v / 10.0, 0.0, 1.5)),
        # #                     float(self.patch.theta / np.pi),
        # #                 ],
        # #                 dtype=np.float32,
        # #             ),
        # #             lidar_norm,
        # #         ],
        # #         axis=0,
        # #     )
        # #     compact = np.nan_to_num(compact, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        # #     return np.clip(compact, -10.0, 10.0).astype(np.float32)

    # Freenet helper functions 
    def _wrap_ds(self, ds: float) -> float:
        if self.track_length is None or not np.isfinite(self.track_length):
            return ds  # Safe fallback
        L = float(self.track_length)
        if L <= 0:
            return ds
        return (ds + 0.5 * L) % L - 0.5 * L

    @staticmethod
    def _wrap_angle(angle_rad: float) -> float:
        return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

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

    def _estimate_current_track_width(self) -> float:
        """Estimate local track width at current patch pose."""
        try:
            _, occ_map, resolution, origin = self.f110.get_track_data()
            width = float(
                self.map_reset.estimate_track_width(
                    occ_map,
                    resolution,
                    origin,
                    float(self.patch.x),
                    float(self.patch.y),
                    float(self.patch.theta),
                )
            )
            if not np.isfinite(width) or width <= 0.0:
                return float(max(2.0 * self.patch.b, 1.0))
            return width
        except Exception:
            return float(max(2.0 * self.patch.b, 1.0))

    def _frenet_proxy_lidar(self, ey: float, track_width: float) -> np.ndarray:
        """Create a proxy lidar vector from Frenet lateral margin to track edges."""
        n_sectors = int(self.lidar_model.config.n_sectors)
        half_w = max(0.5 * float(track_width), 1e-3)
        left_margin = max(0.0, half_w - float(ey))
        right_margin = max(0.0, half_w + float(ey))
        edge_dist = float(np.clip(min(left_margin, right_margin), 0.0, self.lidar_model.config.max_range_m))
        proxy = np.ones(n_sectors, dtype=np.float32) * edge_dist
        return proxy

    # def _compute_reward_w_o_frenet(self, lidar_info: dict) -> float:
    #     reward = 0.0
    #     min_dist = float(lidar_info["min_dist"])
    #     if self.patch.v > 0.1:
    #         v_norm = min(self.patch.v / 5.0, 1.0)
    #         reward += 7.0 * v_norm
    #         if self.patch.v >= 3.0:
    #             reward += 5.0 * (self.patch.v - 3.0) / 2.0
    #     else:
    #         reward -= 5.0

    #     current_pos = np.array([self.patch.x, self.patch.y], dtype=np.float32)
    #     if self._prev_patch_pos is None:
    #         self._prev_patch_pos = current_pos.copy()
    #     moved = float(np.linalg.norm(current_pos - self._prev_patch_pos))
    #     if moved > 1e-3:
    #         heading_vec = np.array([np.cos(self.patch.theta), np.sin(self.patch.theta)])
    #         forward_component = float(np.dot(current_pos - self._prev_patch_pos, heading_vec))
    #         if forward_component > 0:
    #             reward += 5.0 * forward_component
    #         elif forward_component < -0.05:
    #             reward -= 3.0 * abs(forward_component)
    #     self._prev_patch_pos = current_pos.copy()

    #     reward += 5.0 * (min_dist / 2.0)
    #     if min_dist < 1.5:
    #         reward -= 8.0 * (1.5 - min_dist)

    #     # Goal distance calculation based on navigation mode
    #     if self.navigation_mode == "landmark":
    #         _, _, goal_dist = self.obs_builder.goal_direction_and_distance(self.patch, self.goal_xy)
    #         if self._prev_goal_dist is None:
    #             self._prev_goal_dist = goal_dist
    #         dist_improve = self._prev_goal_dist - goal_dist
    #         reward += 8.0 * dist_improve
    #         self._prev_goal_dist = goal_dist
    #     elif self.navigation_mode == "centerline" and self.waypoints is not None:
    #         _, _, goal_dist, self._current_waypoint_idx = self.obs_builder.goal_direction_and_distance_centerline(
    #             self.patch, self.waypoints, self._current_waypoint_idx, self.look_ahead, self.search_window
    #         )
    #         if self._prev_goal_dist is None:
    #             self._prev_goal_dist = goal_dist
    #         dist_improve = self._prev_goal_dist - goal_dist
    #         reward += 8.0 * dist_improve
    #         self._prev_goal_dist = goal_dist

    #     if self.mpc_attempts > 0:
    #         feas = self.mpc_successes / self.mpc_attempts
    #         reward += 10.0 * (feas - 0.5)
    #         if feas >= 0.95:
    #             reward += 3.0

    #     safety_rate = self.safety_interventions / max(1, self.step_count * self.num_agents)
    #     if safety_rate > 0.2:
    #         reward -= 5.0 * (safety_rate - 0.2)

    #     reward += 0.05
    #     return float(np.clip(reward, -50.0, 50.0))

    def _compute_reward_w_frenet(
        self,
        lidar_info: dict,
        dt: float,
        lap_bonus: float = 0.0,
    ) -> float:
        """
        Simplified reward matching single-agent PPO exactly.
        Only Frenet-based terms: progress, cross-track, steering penalties, collision, stuck, lap bonus, time penalty.
        """
        # Frenet features
        s, ey = self._patch_to_frenet()
        ds = self._wrap_ds(s - self.prev_s) if self.prev_s is not None else 0.0

        # Patch controls / yaw-rate proxy (same as single-agent uses agent controls)
        steer_cmd = float(self.patch.steering)
        steer_rate = abs(steer_cmd - self.prev_steer) / max(dt, 1e-6)
        yaw_rate = float((self.patch.v / max(self.cfg.wheelbase, 1e-6)) * np.tan(steer_cmd))
        spin_excess = max(0.0, abs(yaw_rate) - self.cfg.spin_yawrate_threshold)

        # Collision proxy from lidar (same as single-agent)
        min_dist = float(lidar_info["min_dist"])
        collision_lidar = (not np.isfinite(min_dist)) or (min_dist < self.cfg.collision_min_dist)
        
        # Collision from patch boundary discretization
        patch_boundary_collision = lidar_info.get("patch_boundary_collision", False)
        
        # Combined collision flag (either lidar or patch boundary collision)
        collision = collision_lidar or patch_boundary_collision

        # Stuck counter (same as single-agent)
        if abs(ds) < self.cfg.stuck_progress_eps:
            self.no_progress_counter += 1
        else:
            self.no_progress_counter = 0

        # Core reward (same as single-agent)
        # Apply heavy penalty (1500) for any collision (lidar or patch boundary)
        reward_raw = (
            self.cfg.reward_progress_scale * ds
            - self.cfg.reward_crosstrack_weight * abs(ey)
            - self.cfg.reward_steer_bias_weight * abs(steer_cmd)
            - self.cfg.reward_steer_rate_weight * steer_rate
            - self.cfg.reward_spin_weight * spin_excess
            - (self.cfg.collision_penalty if collision else 0.0)
        )

        # Per-step time penalty (same as single-agent)
        reward_raw -= self.cfg.time_penalty_per_sec * dt

        # Event-based lap bonus (same as single-agent)
        reward_raw += float(lap_bonus)

        # Stuck penalty (same as single-agent)
        if self.no_progress_counter >= self.cfg.stuck_no_progress_steps:
            reward_raw -= self.cfg.stuck_penalty

        # Update bookkeeping
        self.prev_s = s
        self.prev_steer = steer_cmd
        
        reward_clipped = float(np.clip(reward_raw, -100.0, 100.0))
        
        # Simplified reward terms (matching single-agent)
        self._last_reward_terms = {
            "s": float(s),
            "ey": float(ey),
            "ds": float(ds),
            "steer_cmd": float(steer_cmd),
            "steer_rate": float(steer_rate),
            "yaw_rate_proxy": float(yaw_rate),
            "spin_excess": float(spin_excess),
            "collision_proxy": bool(collision),
            "collision_lidar": bool(collision_lidar),
            "collision_patch_boundary": bool(patch_boundary_collision),
            "reward_raw": float(reward_raw),
            "reward_clipped": float(reward_clipped),
            "reward_was_clipped": bool(abs(reward_raw) > 100.0),
            "no_progress_counter": int(self.no_progress_counter),
        }
        return reward_clipped
        
        # OLD COMPLEX REWARD CODE (commented out - not used in simplified version)
        # # Stronger edge penalty from Frenet lateral distance L (ey).
        # if track_half_width is not None and track_half_width > 1e-3:
        #     edge_ratio = float(np.clip(abs(ey) / track_half_width, 0.0, 2.0))
        #     reward_raw -= self.cfg.frenet_edge_penalty_weight * (edge_ratio**2)
        # # Corner-aware size targets: in tighter turns / larger lateral error, shrink patch more.
        # corner_kappa_abs = 0.0
        # corner_factor = 0.0
        # corner_b_target = float(self._safe_init_b)
        # corner_a_target = float(self._safe_init_a)
        # corner_b_err = 0.0
        # corner_a_err = 0.0
        # if self.track_spline is not None and track_half_width is not None and track_half_width > 1e-3:
        #     try:
        #         corner_kappa_abs = abs(float(self.track_spline.calc_curvature(float(s))))
        #     except Exception:
        #         corner_kappa_abs = 0.0
        #     corner_from_kappa = float(np.clip(corner_kappa_abs / max(self.cfg.corner_kappa_ref, 1e-4), 0.0, 1.0))
        #     corner_from_edge = float(np.clip((edge_ratio - 0.35) / 0.65, 0.0, 1.0))
        #     corner_factor = max(corner_from_kappa, corner_from_edge)
        #     corner_b_target = float(self._safe_init_b) * (
        #         1.0 - corner_factor * (1.0 - float(self.cfg.corner_b_target_min_scale))
        #     )
        #     corner_a_target = float(self._safe_init_a) * (
        #         1.0 - corner_factor * (1.0 - float(self.cfg.corner_a_target_min_scale))
        #     )
        #     corner_b_err = max(0.0, (float(self.patch.b) - corner_b_target) / max(float(self._safe_init_b), 1e-3))
        #     corner_a_err = max(0.0, (float(self.patch.a) - corner_a_target) / max(float(self._safe_init_a), 1e-3))
        #     reward_raw -= float(self.cfg.corner_b_target_penalty_weight) * (corner_b_err**2)
        #     reward_raw -= float(self.cfg.corner_a_target_penalty_weight) * (corner_a_err**2)
        # # Soft cap penalty: discourage sitting near the hard size cap even without crossing it.
        # softcap_penalty = 0.0
        # softcap_a_ratio = 0.0
        # softcap_b_ratio = 0.0
        # if self.cfg.patch_size_cap_enabled and self._hard_cap_a is not None and self._hard_cap_b is not None:
        #     softcap_a_ratio = float(self.patch.a) / max(float(self._hard_cap_a), 1e-3)
        #     softcap_b_ratio = float(self.patch.b) / max(float(self._hard_cap_b), 1e-3)
        #     start = float(self.cfg.patch_size_softcap_start_ratio)
        #     excess_a = max(0.0, softcap_a_ratio - start)
        #     excess_b = max(0.0, softcap_b_ratio - start)
        #     softcap_penalty = float(self.cfg.patch_size_softcap_penalty_weight) * (excess_a**2 + excess_b**2)
        #     reward_raw -= softcap_penalty
        # # Shape regularization: discourage oversized / highly elongated patches
        # a_now = float(max(self.patch.a, 1e-3))
        # b_now = float(max(self.patch.b, 1e-3))
        # aspect_ratio = float(max(a_now, b_now) / max(min(a_now, b_now), 1e-3))
        # aspect_excess = max(0.0, aspect_ratio - float(self.cfg.shape_aspect_ratio_cap))
        # ref_area = max(float(self._safe_init_a) * float(self._safe_init_b), 1e-3)
        # area_ratio = float((a_now * b_now) / ref_area)
        # area_excess = max(0.0, area_ratio - 1.0)
        # shape_penalty_scale = 1.5 if self.cfg.patch_only_mode else 1.0
        # reward_raw -= shape_penalty_scale * float(self.cfg.shape_aspect_ratio_penalty_weight) * (aspect_excess**2)
        # reward_raw -= shape_penalty_scale * float(self.cfg.shape_area_penalty_weight) * (area_excess**2)

    def _check_termination(self, lidar_info: dict):
        """
        Simplified termination matching single-agent PPO exactly.
        Only: collision (lidar or patch boundary), stuck, max_steps.
        """
        terminated = False
        truncated = False
        reason = None

        # 1) Collision from lidar (same as single-agent)
        min_dist = float(lidar_info["min_dist"])
        if (not np.isfinite(min_dist)) or (min_dist < self.cfg.collision_min_dist):
            return True, False, "collision_lidar"

        # 2) Collision from patch boundary discretization
        patch_boundary_collision = lidar_info.get("patch_boundary_collision", False)
        if patch_boundary_collision:
            return True, False, "collision_patch_boundary"

        # 3) Stuck termination (same as single-agent)
        if self.no_progress_counter >= self.cfg.stuck_no_progress_steps:
            return True, False, "stuck_no_progress"

        # 4) Time limit (same as single-agent)
        if self.step_count >= self.cfg.max_steps:
            truncated = True
            reason = "max_steps"

        return terminated, truncated, reason
        
        # OLD COMPLEX TERMINATION CODE (commented out - not used in simplified version)
        # # 1) Hard collision from map boundary check
        # _, occ_map, resolution, origin = self.f110.get_track_data()
        # patch_collision, violated = self.patch.check_patch_boundary_wall_collision(
        #     occ_map,
        #     resolution,
        #     origin,
        #     n_points=32,
        #     violation_threshold=self.cfg.patch_boundary_violation_threshold,
        # )
        # if patch_collision:
        #     return True, False, f"patch_wall_collision ({len(violated)} points)"
        # # 2) Lidar safety collision proxy (ellipse-normalized metric)
        # if self.cfg.lidar_min_dist_termination:
        #     min_dist = float(lidar_info["min_dist"])
        #     if self.cfg.enable_lidar_termination and ((not np.isfinite(min_dist)) or (min_dist < self.cfg.collision_min_dist)):
        #         return True, False, "patch_wall_collision_lidar"
        # # 3) Optional shape-failure termination
        # if self.patch.a < self._safe_min_a or self.patch.b < self._safe_min_b:
        #     return True, False, "patch_too_small"
        # # 4) Off-track guard: cut hopeless episodes with very large lateral error.
        # _, ey = self._patch_to_frenet()
        # if self.cfg.ey_termination_enabled:
        #     if self.cfg.patch_only_mode and track_half_width is not None:
        #         ey_limit = float(self.cfg.patch_only_ey_termination_ratio) * float(track_half_width)
        #         if abs(float(ey)) > max(ey_limit, 1e-3):
        #             return True, False, "patch_only_centerline_ey_limit"
        #     if track_half_width is not None and abs(float(ey)) > track_half_width:
        #         return True, False, "patch_center_outside_trackwidth_half"
        #     if abs(float(ey)) > self.cfg.offtrack_ey_termination:
        #         return True, False, "offtrack_ey"
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
        """
        Simplified reset matching single-agent PPO exactly.
        Initialize Frenet spline, spawn patch at centerline start, reset base env.
        """
        super().reset(seed=seed)
        self._np_random = np.random.RandomState(seed if seed is not None else None)
        
        # Basic episode tracking (same as single-agent)
        self.step_count = 0
        self.episode_reward = 0.0
        self.lap_progress = 0.0
        self.lap_count = 0
        self.lap_start_step = 0
        
        # Initialize Frenet spline from track centerline (same as single-agent)
        self.f110.ensure_initialized()
        track, _, _, _ = self.f110.get_track_data()
        if track.centerline is None or track.centerline.spline is None:
            raise ValueError("Track centerline/spline missing for Frenet reward.")
        
        self.track_spline = track.centerline.spline
        self.track_length = float(self.track_spline.s[-1])
        
        # Spawn patch at first centerline waypoint (same as single-agent spawns at start)
        xs = np.asarray(track.centerline.xs, dtype=np.float32)
        ys = np.asarray(track.centerline.ys, dtype=np.float32)
        patch_x, patch_y = float(xs[0]), float(ys[0])
        if xs.shape[0] > 1:
            dx = float(xs[1] - xs[0])
            dy = float(ys[1] - ys[0])
            patch_theta = float(np.arctan2(dy, dx))
        else:
            patch_theta = 0.0
        self.start_xy = (patch_x, patch_y)
        
        # Create patch with fixed size (same as single-agent has fixed robot size)
        self.patch = DynamicPatch(
            x=patch_x, 
            y=patch_y, 
            theta=patch_theta, 
            v=0.5,  # Initial speed
            a=self.cfg.patch_a,  # Fixed size
            b=self.cfg.patch_b   # Fixed size
        )
        
        # Initialize Frenet bookkeeping (same as single-agent)
        self.prev_steer = 0.0
        self.no_progress_counter = 0
        self.prev_s, _ = self._patch_to_frenet()
        
        # Reset base env with agent at patch position (same as single-agent - 1 agent for lidar)
        # base_obs, _ = self.f110.reset(poses=[[patch_x, patch_y, patch_theta]])
        base_obs, _ = self.f110.reset(poses=np.array([[patch_x, patch_y, patch_theta]], dtype=np.float32))
        self.current_base_obs = base_obs
        
        # OLD: Reset with empty poses for proxy lidar (commented out - using real lidar instead)
        # # Reset base env (with num_agents=0, use empty poses)
        # # Agents will be controlled by MPC later, not base env
        # base_obs, _ = self.f110.reset(poses=[])
        # self.current_base_obs = base_obs  # May be None/empty, that's OK for proxy lidar
        
        # Build and return observation (same as single-agent)
        obs = self._build_obs()
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        return obs, {}
        
        # OLD COMPLEX RESET CODE (commented out - not used in simplified version)
        # self.patch_min_clearance = float("inf")
        # self.patch_wall_collisions = 0
        # self.mpc_successes = 0
        # self.mpc_attempts = 0
        # self.safety_interventions = 0
        # self.cached_controls = []
        # self.cached_feasible = []
        # self._last_reward_terms = {}
        # self.agent_motion_sum = 0.0
        # self.agent_motion_steps = 0
        # self.agent_move_event_steps = 0
        # self.agent_move_eps = 0.01
        # track_width = self.map_reset.estimate_track_width(...)
        # init_a, init_b = self._safe_init_a, self._safe_init_b
        # self._safe_min_a, self._safe_min_b = init_a * 0.5, init_b * 0.5
        # self.patch_a_range = (self._safe_min_a, self._safe_init_a)
        # self.patch_b_range = (self._safe_min_b, self._safe_init_b)
        # self.action_model.config.patch_a_range = self.patch_a_range
        # self.action_model.config.patch_b_range = self.patch_b_range
        # if self.domain_randomize:
        #     self.patch.randomize_dynamics(self._np_random)
        # self.agent_states = []
        # self.prev_v = []
        # self.prev_agent_steer = []
        # self._hard_cap_a = float(self._safe_init_a)
        # self._hard_cap_b = float(self._safe_init_b)

    def step(self, action):
        """
        Simplified step matching single-agent PPO exactly.
        Action: [steering, speed] directly (no denormalization wrapper).
        """
        self.step_count += 1
        dt = self.cfg.control_dt
        lap_bonus = 0.0
        lap_time = None

        # Direct action unpacking (same as single-agent: [steering, speed])
        steering_cmd = float(np.clip(action[0], -0.4189, 0.4189))
        speed_cmd = float(np.clip(action[1], 0.5, 10.0))
        a_cmd = float(np.clip(action[2], 1.0, ))
        
        # Update patch with bicycle model (same as single-agent updates agent)
        # Convert speed command to acceleration (simple: maintain speed)
        # current_speed = float(self.patch.v)
        # accel_cmd = (speed_cmd - current_speed) / max(dt, 1e-6)
        # accel_cmd = float(np.clip(accel_cmd, -3.0, 3.0))  # Reasonable accel limits

        steering_cmd = float(np.nan_to_num(steering_cmd, nan=0.0, posinf=0.4189, neginf=-0.4189))
        speed_cmd = float(np.nan_to_num(speed_cmd, nan=0.5, posinf=10.0, neginf=0.5))
        steering_cmd = float(np.clip(steering_cmd, -0.4189, 0.4189))
        speed_cmd = float(np.clip(speed_cmd, 0.5, 10.0))
        
        # Update patch state (fixed size, only position/velocity change)
        self.patch.steering = steering_cmd
        self.patch.step(speed_cmd, steering_cmd, dt)
        
        # Step base env with agent at patch position for lidar (like single-agent)
        # Agent follows patch with same action to provide lidar
        # Note: Agent is NOT trained - it's just a sensor that follows the patch
        agent_action = np.array([[speed_cmd, steering_cmd]], dtype=np.float32)
        agent_action = np.nan_to_num(agent_action, nan=0.0, posinf=10.0, neginf=0.0)
        agent_action = np.clip(agent_action, [[0.5, -0.4189]], [[10.0, 0.4189]])
        base_obs, _, base_done, base_truncated, _ = self.f110.step(agent_action)
        self.current_base_obs = base_obs
        
        # Get lidar info from agent (same as single-agent)
        scan = self._get_scan(base_obs, self.cfg.num_beams, i=0)
        min_dist = float(np.min(scan))
        min_dist = float(np.nan_to_num(min_dist, nan=0.0, posinf=10.0, neginf=0.0))
        
        # Check patch boundary collision using discretization
        patch_boundary_collision = False
        try:
            _, occ_map, resolution, origin = self.f110.get_track_data()
            patch_collision, violated = self.patch.check_patch_boundary_wall_collision(
                occ_map,
                resolution,
                origin,
                n_points=32,
                violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )
            patch_boundary_collision = patch_collision
        except Exception:
            # If track data unavailable, skip patch boundary check
            patch_boundary_collision = False
        
        lidar_info = {
            "is_safe": min_dist >= self.cfg.collision_min_dist,
            "min_dist": min_dist,
            "patch_boundary_collision": patch_boundary_collision,
        }
        
        # OLD: Proxy lidar (Frenet-based) - commented out, using real lidar instead
        # # Base env not stepped - agents will be controlled by MPC later
        # # For now, we don't need to step base env since num_agents=0
        # # When you add MPC agents later, you'll step them separately
        # base_done = False
        # base_truncated = False
        # # Get lidar info from proxy (Frenet-based) like single-agent code
        # s_now, ey_now = self._patch_to_frenet()
        # track_width_now = self._estimate_current_track_width()
        # if track_width_now is not None:
        #     proxy_lidar = self._frenet_proxy_lidar(ey=ey_now, track_width=track_width_now)
        #     min_dist = float(np.min(proxy_lidar))
        # else:
        #     min_dist = 10.0  # Max range if no track width
        # min_dist = float(np.nan_to_num(min_dist, nan=0.0, posinf=10.0, neginf=0.0))
        # lidar_info = {"is_safe": min_dist >= self.cfg.collision_min_dist, "min_dist": min_dist}
        
        # Lap detection (same as single-agent)
        s_now, ey_now = self._patch_to_frenet()
        if self.track_spline is not None and self.track_length is not None and self.track_length > 0.0:
            self.lap_progress = float(np.clip(s_now / self.track_length, 0.0, 1.0))
            if self.prev_s is not None:
                raw_ds = float(s_now - self.prev_s)
                wrapped_ds = float(self._wrap_ds(raw_ds))
                if raw_ds < -0.5 * float(self.track_length) and wrapped_ds > 0.0:
                    self.lap_count += 1
                    lap_time = (self.step_count - self.lap_start_step) * dt
                    lap_time = max(float(lap_time), 1e-3)
                    lap_bonus = self.cfg.lap_finish_bonus * float(
                        np.exp(-lap_time / max(self.cfg.lap_bonus_tau, 1e-6))
                    )
                    self.lap_start_step = self.step_count
        
        # Compute reward (simplified - no track_half_width needed)
        reward = self._compute_reward_w_frenet(lidar_info, dt, lap_bonus=lap_bonus)
        reward = float(np.nan_to_num(reward, nan=-10.0, posinf=100.0, neginf=-100.0))
        self.episode_reward += reward
        
        # Check termination (simplified - no track_half_width needed)
        terminated, truncated, reason = self._check_termination(lidar_info)
        
        # Build observation
        obs = self._build_obs()
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        
        # Simplified info dict (matching single-agent)
        info = {
            "episode_reward": self.episode_reward,
            "lap_progress": self.lap_progress,
            "Episode_steps": self.step_count,
            "step_reward": reward,
            "termination_reason": reason,
            "lap_count": int(self.lap_count),
            "lap_bonus": float(lap_bonus),
            "lap_time": None if lap_time is None else float(lap_time),
            "min_lidar_dist": float(min_dist),
            "frenet_s": float(s_now),
            "frenet_ey": float(ey_now),
        }
        info.update(self._last_reward_terms)
        
        return obs, reward, terminated, truncated, info
        
        # OLD COMPLEX STEP CODE (commented out - not used in simplified version)
        # a = float(self.patch.a)
        # b = float(self.patch.b)
	    # Patch-only: keep size fixed (no size action)
	    # a = float(self.patch.a)
	    # b = float(self.patch.b)
        # old smoothing path kept commented by request:
        # smooth_accel = self.action_model.smooth(self.patch.accel, raw_accel, self.cfg.alpha)
        # smooth_steer = self.action_model.smooth(self.patch.steering, raw_steer, self.cfg.alpha)
        # smooth_accel, smooth_steer = self.action_model.conservative_first_step(
        #     smooth_accel, smooth_steer, self.step_count
        # )
        # Disable smoothing to keep patch response sharp at corners.
        smooth_accel, smooth_steer = raw_accel, raw_steer

        # NOT USED: patch_only_mode=True, coupling never activates
        # if (not self.cfg.patch_only_mode) and self.cfg.coupling_enabled and self.agent_states is not None:
        #     local_x = []
        #     for i in range(self.num_agents):
        #         ax, ay, _, _ = self.agent_states[i]
        #         x_rel, _ = self.patch.world_to_patch_frame(ax, ay)
        #         local_x.append(float(x_rel))
        #     if local_x:
        #         mean_x = float(np.mean(local_x))
        #         lag = max(0.0, -mean_x - self.cfg.coupling_lag_threshold_m)
        #         if lag > 0.0:
        #             smooth_accel -= self.cfg.coupling_patch_speed_scale * lag

        # Edge guardrail: when patch center approaches track edge, reduce aggressiveness and bias steering inward.
        edge_ratio_pre = 0.0
        edge_guard_mix = 0.0
        track_half_pre = 1.0
        corner_factor_pre = 0.0
        corner_kappa_abs_pre = 0.0
        if self.cfg.edge_guard_enabled and self.track_spline is not None:
            _, ey_pre = self._patch_to_frenet()
            track_width_pre = self._estimate_current_track_width()
            track_half_pre = max(0.5 * track_width_pre, 1e-3)
            edge_ratio_pre = float(np.clip(abs(float(ey_pre)) / track_half_pre, 0.0, 2.0))
            start_r = float(self.cfg.edge_guard_start_ratio)
            full_r = max(float(self.cfg.edge_guard_full_ratio), start_r + 1e-3)
            edge_guard_mix = float(np.clip((edge_ratio_pre - start_r) / (full_r - start_r), 0.0, 1.0))
            if edge_guard_mix > 0.0:
                # old: no edge-aware scaling on patch commands.
                accel_scale = 1.0 - edge_guard_mix * (1.0 - float(self.cfg.edge_guard_min_accel_scale))
                steer_scale = 1.0 - edge_guard_mix * (1.0 - float(self.cfg.edge_guard_min_steer_scale))
                if smooth_accel > 0.0:
                    smooth_accel *= accel_scale
                smooth_steer *= steer_scale
                # Positive ey => patch is left of centerline in Frenet convention, steer right (negative) to recenter.
                smooth_steer += -np.sign(float(ey_pre)) * float(self.cfg.edge_guard_centering_steer_gain) * edge_guard_mix

        # Corner factor from Frenet curvature and lateral offset.
        # Used to encourage "shrink-in-corner" behavior instead of late wall contact.
        if self.track_spline is not None:
            try:
                s_corner, ey_corner = self._patch_to_frenet()
                corner_kappa_abs_pre = abs(float(self.track_spline.calc_curvature(float(s_corner))))
                corner_from_kappa = float(np.clip(corner_kappa_abs_pre / max(self.cfg.corner_kappa_ref, 1e-4), 0.0, 1.0))
                edge_ratio_corner = float(np.clip(abs(float(ey_corner)) / max(track_half_pre, 1e-3), 0.0, 2.0))
                corner_from_edge = float(np.clip((edge_ratio_corner - 0.35) / 0.65, 0.0, 1.0))
                corner_factor_pre = max(corner_from_kappa, corner_from_edge)
            except Exception:
                corner_factor_pre = 0.0
                corner_kappa_abs_pre = 0.0

        # NOT USED: patch_only_centerline_assist_enabled=False
        # if self.cfg.patch_only_mode and self.cfg.patch_only_centerline_assist_enabled and self.track_spline is not None:
        #     try:
        #         s_pre, ey_pre_assist = self._patch_to_frenet()
        #         yaw_ref = float(self.track_spline.calc_yaw(float(s_pre)))
        #         patch_curv_ref = float(self.track_spline.calc_curvature(float(s_pre)))
        #         patch_heading_err = self._wrap_angle(yaw_ref - float(self.patch.theta))
        #         delta_ff = float(np.arctan(float(self.cfg.wheelbase) * patch_curv_ref))
        #         steer_fb = (
        #             float(self.cfg.patch_only_heading_gain) * patch_heading_err
        #             - float(self.cfg.patch_only_ey_gain) * (float(ey_pre_assist) / max(track_half_pre, 1e-3))
        #         )
        #         steer_assist = float(self.cfg.patch_only_curvature_ff_gain) * delta_ff + steer_fb
        #         assist_blend = float(np.clip(self.cfg.patch_only_assist_blend + 0.35 * edge_guard_mix, 0.0, 1.0))
        #         smooth_steer = (1.0 - assist_blend) * smooth_steer + assist_blend * steer_assist
        #         if abs(patch_heading_err) > float(self.cfg.patch_only_corner_heading_thresh) and smooth_accel > 0.0:
        #             smooth_accel -= float(self.cfg.patch_only_corner_slowdown_gain) * abs(patch_heading_err)
        #     except Exception:
        #         pass
        patch_heading_err = 0.0
        patch_curv_ref = 0.0

        smooth_accel, smooth_steer = self.action_model.conservative_first_step(
            smooth_accel, smooth_steer, self.step_count
        )
        self.patch.accel = smooth_accel
        self.patch.steering = smooth_steer
        self.patch.step(smooth_accel, smooth_steer, dt)
        if self.cfg.patch_only_mode:
            # Keep enough forward speed in patch-only mode so heading can evolve through corners.
            # old: patch.v relied only on DynamicPatch v_min (0.5).
            self.patch.v = float(max(self.patch.v, self.cfg.patch_only_speed_floor))
            # Corner-speed control: reduce speed cap in high-corner factor regions.
            v_corner_cap = (
                self.patch.config.v_max
                * (1.0 - float(self.cfg.patch_only_corner_speed_reduction_gain) * float(corner_factor_pre))
            )
            v_corner_cap = float(np.clip(v_corner_cap, self.cfg.patch_only_corner_speed_min, self.patch.config.v_max))
            # old: no corner-dependent speed cap in patch-only mode.
            self.patch.v = float(min(self.patch.v, v_corner_cap))

        if self.cfg.patch_only_mode:
            # old: always constrained by agent positions.
            min_a_agents, min_b_agents = float(self._safe_min_a), float(self._safe_min_b)
        else:
            min_a_agents, min_b_agents = self._calculate_min_patch_size_for_agents()
        max_a_edge = float(self._safe_init_a)
        max_b_edge = float(self._safe_init_b)
        if self.cfg.edge_guard_enabled and edge_guard_mix > 0.0:
            # Shrink max admissible patch size near edges to avoid boundary intersection on curves.
            a_scale = 1.0 - edge_guard_mix * (1.0 - float(self.cfg.edge_guard_min_max_a_scale))
            b_scale = 1.0 - edge_guard_mix * (1.0 - float(self.cfg.edge_guard_min_max_b_scale))
            max_a_edge = float(self._safe_init_a) * float(np.clip(a_scale, 0.2, 1.0))
            max_b_edge = float(self._safe_init_b) * float(np.clip(b_scale, 0.2, 1.0))
        # old:
        # a = min(max(a, min_a_agents), self._safe_init_a)
        # b = min(max(b, min_b_agents), self._safe_init_b)
        a_cap = max(max_a_edge, float(min_a_agents))
        b_cap = max(max_b_edge, float(min_b_agents))
        # Keep patch from growing beyond spawn size.
        a_cap = min(a_cap, float(self._safe_init_a))
        b_cap = min(b_cap, float(self._safe_init_b))
        if self.cfg.patch_size_cap_enabled:
            hard_cap_a = float(self._hard_cap_a) if self._hard_cap_a is not None else float(self._safe_init_a)
            hard_cap_b = float(self._hard_cap_b) if self._hard_cap_b is not None else float(self._safe_init_b)
            # old: no explicit hard cap tied to equivalent 6-agent envelope.
            a_cap = min(a_cap, hard_cap_a)
            b_cap = min(b_cap, hard_cap_b)
        # Keep feasibility if hard cap is below min required bound.
        a_cap = max(a_cap, float(min_a_agents))
        b_cap = max(b_cap, float(min_b_agents))
        requested_a = float(a)
        requested_b = float(b)
        # old:
        # a = min(a, a_cap)
        # b = min(b, b_cap)
        # Correct clamp includes both lower and upper bounds.
        a = min(max(a, float(min_a_agents)), float(a_cap))
        b = min(max(b, float(min_b_agents)), float(b_cap))
        if self.cfg.patch_only_mode:
            # Prevent collapse to a needle-like ellipse in patch-only debugging.
            b_min_patch_only = max(float(self._safe_min_b), float(self.cfg.patch_only_min_b_scale) * float(self._safe_init_b))
            # old: no extra lower bound for b in patch-only mode.
            b = min(max(b, b_min_patch_only), float(b_cap))
        if self.cfg.shape_aspect_ratio_cap > 0.0:
            # Cap elongation a/b to avoid pathological "long capsule" corner failures.
            # old: no explicit aspect-ratio cap.
            a = min(a, float(self.cfg.shape_aspect_ratio_cap) * max(b, 1e-3))
        size_cap_penalty = 0.0
        size_cap_excess_ratio = 0.0
        size_cap_excess_a = 0.0
        size_cap_excess_b = 0.0
        # Temporarily increase shrink/expand rate in corners so policy can reduce size in time.
        orig_shape_rate = float(self.patch.config.size_change_rate)
        if self.cfg.patch_only_mode:
            # old: shape rate stayed constant regardless of corner difficulty.
            self.patch.config.size_change_rate = orig_shape_rate * (
                1.0 + float(self.cfg.corner_shrink_rate_boost) * float(corner_factor_pre)
            )
        if self.cfg.patch_size_cap_enabled:
            size_cap_excess_a = max(0.0, requested_a - float(a_cap))
            size_cap_excess_b = max(0.0, requested_b - float(b_cap))
            if size_cap_excess_a > 0.0 or size_cap_excess_b > 0.0:
                size_cap_excess_ratio = (
                    size_cap_excess_a / max(float(a_cap), 1e-3)
                    + size_cap_excess_b / max(float(b_cap), 1e-3)
                )
                # Penalize attempts to grow past hard cap, even though final size is clamped.
                # old: cap crossing was silently clamped with no direct reward penalty.
                size_cap_penalty = float(self.cfg.patch_size_cap_penalty_weight) * float(size_cap_excess_ratio**2)
        # old:
        # a = min(requested_a, float(a_cap))
        # b = min(requested_b, float(b_cap))
        # Keep clamped values computed above.
        self.patch.update_shape(a, b, dt, max_a=a_cap, max_b=b_cap)
        self.patch.config.size_change_rate = orig_shape_rate
        self.patch.save_state()

        base_obs = self.current_base_obs
        prev_base_obs = base_obs
        s_now, ey_now = self._patch_to_frenet()
        track_width_now = self._estimate_current_track_width()
        track_half_width = max(0.5 * track_width_now, 1e-3)
        self._last_track_width = track_width_now
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

        if self.cfg.patch_only_mode:
            # Patch-only debug mode: skip MPC/safety/agent stepping for faster iteration.
            mean_abs_steer_cmd = 0.0
            mean_speed_cmd = 0.0
            base_done = False
            base_truncated = False
            nonfinite_base_state = False
            mean_agent_step_disp = 0.0
            self.agent_motion_sum += mean_agent_step_disp
            self.agent_motion_steps += 1
            self.current_base_obs = base_obs
        # NOT USED: agent/MPC stepping code (patch_only_mode=True, num_agents=0)
        # else:
        #     positions_world = [
        #         [base_obs["poses_x"][i], base_obs["poses_y"][i]] for i in range(self.num_agents)
        #     ]
        #     positions_local = [[p[0] - self.patch.x, p[1] - self.patch.y] for p in positions_world]
        #
        #     env_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        #     solve_mpc = True
        #     for i in range(self.num_agents):
        #         x_local = positions_local[i][0]
        #         y_local = positions_local[i][1]
        #         theta_i = base_obs["poses_theta"][i]
        #         v_i = self.prev_v[i]
        #         x0_local = np.array([x_local, y_local, theta_i, v_i], dtype=np.float32)
        #         neighbors_local = [positions_local[j] for j in range(self.num_agents) if j != i]
        #
        #         if solve_mpc:
        #             self.mpc_attempts += 1
        #             u_opt, feasible = self.mpc_solvers[i].solve(x0_local, patch_for_mpc, neighbors_local)
        #             self.cached_controls[i] = u_opt
        #             self.cached_feasible[i] = feasible
        #             if feasible:
        #                 self.mpc_successes += 1
        #         else:
        #             u_opt = self.cached_controls[i]
        #             feasible = self.cached_feasible[i]
        #
        #         u_safe, intervention = self.safety_layer.filter_control(
        #             u_opt if feasible else None, x0_local, patch_for_mpc, neighbors_local, dt=dt
        #         )
        #         if intervention:
        #             self.safety_interventions += 1
        #         accel, steering = float(u_safe[0]), float(u_safe[1])
        #         if not np.isfinite(accel):
        #             accel = 0.0
        #         if not np.isfinite(steering):
        #             steering = 0.0
        #         accel = float(np.clip(accel, -self.cfg.agent_accel_clip, self.cfg.agent_accel_clip))
        #         prev_steer_i = float(self.prev_agent_steer[i]) if self.prev_agent_steer is not None else 0.0
        #         steering = float(np.clip(steering, prev_steer_i - self.cfg.max_steer_delta_per_step, prev_steer_i + self.cfg.max_steer_delta_per_step))
        #         steering = float(np.clip(steering, -0.4, 0.4))
        #         v_new = float(np.clip(v_i + accel * dt, 0.5, 10.0))
        #         if self.cfg.use_agent_lidar_braking and "scans" in base_obs and i < len(base_obs["scans"]):
        #             scan_i = np.asarray(base_obs["scans"][i], dtype=np.float32)
        #             scan_i = np.nan_to_num(scan_i, nan=10.0, posinf=10.0, neginf=0.0)
        #             min_scan_i = float(np.min(scan_i))
        #             if min_scan_i <= self.cfg.agent_lidar_brake_dist:
        #                 scale = float(np.clip(min_scan_i / max(self.cfg.agent_lidar_brake_dist, 1e-3), 0.2, 1.0))
        #                 v_new = max(0.5, v_new * scale)
        #
        #         v_new = float(np.clip(v_new * self.cfg.agent_speed_boost, 0.5, self.cfg.stable_agent_speed_cap))
        #         v_new = float(np.clip(v_new, 0.5, self.cfg.hard_speed_cap))
        #         min_speed_floor = float(self.cfg.agent_min_speed_cmd)
        #         if self.cfg.coupling_enabled:
        #             lag_i = max(0.0, -float(x_local) - float(self.cfg.coupling_lag_threshold_m))
        #             min_speed_floor += float(self.cfg.agent_lag_speed_gain) * lag_i
        #         min_speed_floor = float(np.clip(min_speed_floor, 0.5, self.cfg.hard_speed_cap))
        #         v_new = max(v_new, min_speed_floor)
        #         self.prev_v[i] = v_new
        #         self.prev_agent_steer[i] = steering
        #         env_actions[i] = self._pack_base_action(steering_cmd=steering, speed_cmd=v_new)
        #
        #     env_actions = self._clip_base_actions(env_actions)
        #     steer_idx, speed_idx = self._effective_action_indices()
        #     mean_abs_steer_cmd = float(np.mean(np.abs(env_actions[:, steer_idx])))
        #     mean_speed_cmd = float(np.mean(env_actions[:, speed_idx]))
        #
        #     base_obs, _, base_done, base_truncated, _ = self.f110.step(env_actions)
        #
        #     nonfinite_base_state = False
        #     try:
        #         nonfinite_base_state = (
        #             not np.all(np.isfinite(np.asarray(base_obs["poses_x"], dtype=np.float32)))
        #             or not np.all(np.isfinite(np.asarray(base_obs["poses_y"], dtype=np.float32)))
        #             or not np.all(np.isfinite(np.asarray(base_obs["poses_theta"], dtype=np.float32)))
        #             or not np.all(np.isfinite(np.asarray(base_obs["linear_vels_x"], dtype=np.float32)))
        #             or not np.all(np.isfinite(np.asarray(base_obs["linear_vels_y"], dtype=np.float32)))
        #         )
        #     except Exception:
        #         nonfinite_base_state = True
        #     if nonfinite_base_state:
        #         base_obs = prev_base_obs
        #     self.current_base_obs = base_obs
        #
        #     agent_step_disps = []
        #     for i in range(self.num_agents):
        #         prev_x, prev_y = float(positions_world[i][0]), float(positions_world[i][1])
        #         new_x, new_y = float(base_obs["poses_x"][i]), float(base_obs["poses_y"][i])
        #         d = float(np.hypot(new_x - prev_x, new_y - prev_y))
        #         agent_step_disps.append(d)
        #     mean_agent_step_disp = float(np.mean(agent_step_disps)) if agent_step_disps else 0.0
        #     self.agent_motion_sum += mean_agent_step_disp
        #     self.agent_motion_steps += 1
        #     if mean_agent_step_disp > self.agent_move_eps:
        #         self.agent_move_event_steps += 1
        #
        #     for i in range(self.num_agents):
        #         v_sim = float(
        #             np.sqrt(
        #                 base_obs["linear_vels_x"][i] ** 2
        #                 + base_obs["linear_vels_y"][i] ** 2
        #             )
        #         )
        #         self.agent_states[i] = [
        #             float(base_obs["poses_x"][i]),
        #             float(base_obs["poses_y"][i]),
        #             float(base_obs["poses_theta"][i]),
        #             max(v_sim, 0.5),
        #         ]

        # Old real lidar aggregation kept commented by request:
        # lidar_distances = self.lidar_model.aggregate(base_obs, self.patch)
        if self.cfg.use_frenet_proxy_lidar:
            lidar_distances = self._frenet_proxy_lidar(ey=ey_now, track_width=track_width_now)
        else:
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
        # NOT USED: landmark navigation mode
        # if self.navigation_mode == "landmark":
        #     self.lap_progress, _, self._init_goal_dist = self.obs_builder.lap_progress(
        #         self.patch, self.goal_xy, self._init_goal_dist
        #     )
        if self.navigation_mode == "centerline":
            # Waypoint-based lap/progress is disabled; use Frenet-only progress/lap detection.
            # old:
            # if self.navigation_mode == "centerline" and self.waypoints is not None:
            #     n_waypoints = len(self.waypoints)
            #     prev_idx = int(self._current_waypoint_idx)
            #     self.lap_progress, _, self._current_waypoint_idx = self.obs_builder.lap_progress_centerline(
            #         self.patch, self.waypoints, self._current_waypoint_idx, n_waypoints, self.search_window
            #     )
            #     if (
            #         prev_idx > int(0.8 * n_waypoints)
            #         and int(self._current_waypoint_idx) < int(0.2 * n_waypoints)
            #     ):
            #         self.lap_count += 1
            #         lap_time = (self.step_count - self.lap_start_step) * dt
            #         lap_time = max(float(lap_time), 1e-3)
            #         lap_bonus = self.cfg.lap_finish_bonus * float(
            #             np.exp(-lap_time / max(self.cfg.lap_bonus_tau, 1e-6))
            #         )
            #         self.lap_start_step = self.step_count
            #     self.prev_waypoint_idx = int(self._current_waypoint_idx)
            if self.track_spline is not None and self.track_length is not None and self.track_length > 0.0:
                s_now, _ey_now = self._patch_to_frenet()
                self.lap_progress = float(np.clip(s_now / self.track_length, 0.0, 1.0))
                if self.prev_s is not None:
                    raw_ds = float(s_now - self.prev_s)
                    wrapped_ds = float(self._wrap_ds(raw_ds))
                    if raw_ds < -0.5 * float(self.track_length) and wrapped_ds > 0.0:
                        self.lap_count += 1
                        lap_time = (self.step_count - self.lap_start_step) * dt
                        lap_time = max(float(lap_time), 1e-3)
                        lap_bonus = self.cfg.lap_finish_bonus * float(
                            np.exp(-lap_time / max(self.cfg.lap_bonus_tau, 1e-6))
                        )
                        self.lap_start_step = self.step_count
        
        reward = self._compute_reward_w_frenet(
            lidar_info,
            dt,
            lap_bonus=lap_bonus,
            track_half_width=track_half_width,
        )
        reward = float(np.nan_to_num(reward, nan=-10.0, posinf=100.0, neginf=-100.0))
        if size_cap_penalty > 0.0:
            reward -= float(size_cap_penalty)
        self.episode_reward += reward

        terminated, truncated, reason = self._check_termination(
            lidar_info,
            track_half_width=track_half_width,
        )
        if isinstance(reason, str) and reason.startswith("patch_wall_collision"):
            self.patch_wall_collisions += 1
            # old: no explicit wall-collision terminal penalty.
            reward -= float(self.cfg.patch_wall_collision_penalty)
            self.episode_reward -= float(self.cfg.patch_wall_collision_penalty)
        if reason == "patch_center_outside_trackwidth_half":
            reward -= float(self.cfg.trackwidth_violation_penalty)
            self.episode_reward -= float(self.cfg.trackwidth_violation_penalty)
        if nonfinite_base_state:
            terminated = True
            reason = "nonfinite_base_state"
            reward -= float(self.cfg.nonfinite_state_penalty)
            self.episode_reward -= float(self.cfg.nonfinite_state_penalty)
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
            "patch_size_cap": (
                None if self._hard_cap_a is None else float(self._hard_cap_a),
                None if self._hard_cap_b is None else float(self._hard_cap_b),
            ),
            "patch_size_cap_penalty": float(size_cap_penalty),
            "patch_size_cap_excess_ratio": float(size_cap_excess_ratio),
            "patch_size_cap_excess_a": float(size_cap_excess_a),
            "patch_size_cap_excess_b": float(size_cap_excess_b),
            "patch_velocity": self.patch.v,
            "patch_heading_err_ref": float(patch_heading_err),
            "patch_curvature_ref": float(patch_curv_ref),
            "step_reward": reward,
            "termination_reason": reason,
            "lidar_safe": lidar_safe,
            "lap_count": int(self.lap_count),
            "lap_bonus": float(lap_bonus),
            "lap_time": None if lap_time is None else float(lap_time),
            "mpc_feasibility_rate": 1.0 if self.mpc_attempts == 0 else float(self.mpc_successes / self.mpc_attempts),
            "safety_intervention_rate": float(self.safety_interventions / max(1, self.step_count * self.num_agents)),
            "min_lidar_dist": float(min_dist),
            "frenet_s": float(s_now),
            "frenet_ey": float(ey_now),
            "track_width": float(track_width_now),
            "track_half_width": float(track_half_width),
            "collision_min_dist_threshold": float(self.cfg.collision_min_dist),
            "base_done": bool(base_done),
            "base_truncated": bool(base_truncated),
            "nonfinite_base_state": bool(nonfinite_base_state),
            "base_reset_type": str(self.cfg.base_reset_type),
            "use_base_done_termination": bool(self.cfg.use_base_done_termination),
            "mean_agent_step_disp": mean_agent_step_disp,
            "episode_avg_agent_step_disp": float(self.agent_motion_sum / max(1, self.agent_motion_steps)),
            "episode_agent_move_step_ratio": float(self.agent_move_event_steps / max(1, self.agent_motion_steps)),
            "mean_abs_steer_cmd": mean_abs_steer_cmd,
            "mean_speed_cmd": mean_speed_cmd,
            "patch_only_mode": bool(self.cfg.patch_only_mode),
            "control_input_order": (
                "[steering_angle, speed] (forced_legacy)"
                if self.cfg.force_legacy_steer_speed_order
                else list(self.f110.config.control_input)
            ),
        }
        info.update(reward_terms)

        # if self.cfg.debug_print_every_n_steps > 0 and (self.step_count % self.cfg.debug_print_every_n_steps == 0):
            
            # print(
            #     f"[PATCH-DEBUG] step={self.step_count} reward={reward:.2f} "
            #     f"raw={reward_terms.get('reward_raw', 0.0):.2f} ds={reward_terms.get('ds', 0.0):.4f} "
            #     f"ey={reward_terms.get('ey', 0.0):.3f} min_dist={min_dist:.3f} "
            #     f"mpc_feas={info['mpc_feasibility_rate']:.3f} safety_rate={info['safety_intervention_rate']:.3f}"
            # )

        # if self.cfg.debug_print_episode_end and (terminated or truncated):
            
            # print(
            #     f"[PATCH-END] steps={self.step_count} reason={reason} ep_reward={self.episode_reward:.2f} "
            #     f"last_r={reward:.2f} raw={reward_terms.get('reward_raw', 0.0):.2f} "
            #     f"ds={reward_terms.get('ds', 0.0):.4f} ey={reward_terms.get('ey', 0.0):.3f} "
            #     f"min_dist={min_dist:.3f} clipped={reward_terms.get('reward_was_clipped', False)} "
            #     f"mpc_feas={info['mpc_feasibility_rate']:.3f} safety_rate={info['safety_intervention_rate']:.3f} "
            #     f"cmd(v={mean_speed_cmd:.2f},|st|={mean_abs_steer_cmd:.3f}) "
            #     f"sizecap_pen={size_cap_penalty:.2f} excess={size_cap_excess_ratio:.3f}"
            # )

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
                    wall_mask = occ_region < 0.5  # Walls are values > 0.5
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
        
        # agents_inside = sum(1 for i in range(self.num_agents)
        #                   if self.patch.is_inside(self.agent_states[i][0],
        #                                           self.agent_states[i][1]))
        
        # Compute wall collision status live so visualization matches runtime behavior.
        patch_wall_collision = False
        violated = []
        if occ_map is not None and resolution is not None and origin is not None:
            patch_wall_collision, violated = self.patch.check_patch_boundary_wall_collision(
                occ_map,
                resolution,
                origin,
                n_points=32,
                # violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )

        # Visualization-friendly clearance estimate from latest lidar aggregation.
        # clearance = float("nan")
        # if self.current_base_obs is not None:
        #     lidar_distances = self.lidar_model.aggregate(self.current_base_obs, self.patch)
        #     min_dist = float(np.nanmin(lidar_distances))
        #     min_dist = float(np.nan_to_num(min_dist, nan=0.0, posinf=10.0, neginf=0.0))
        #     clearance = min_dist - float(self.cfg.collision_min_dist)
        # elif np.isfinite(self.patch_min_clearance):
        #     clearance = float(self.patch_min_clearance)
        
        # Title with wall collision warning
        wall_status = (
            f"⚠️ WALL! ({len(violated)} pts)"
            # if patch_wall_collision
            # else f"Clear: {clearance:.2f}m"
        )
        self._ax.set_title(
            f'Patch Funnel V1 | Step {self.step_count} | Progress: {self.lap_progress:.1%}\n'
            f'Patch: v={self.patch.v:.1f}m/s, size=({self.patch.a:.1f}, {self.patch.b:.1f})' ,
            # f'Inside: {agents_inside}/{self.num_agents} | {wall_status}',
            fontsize=11,
            color='red' if patch_wall_collision else 'black'
        )
        
        # Draw ellipsoid patch - RED if hitting wall
        if patch_wall_collision:
            face_color = 'red'
            edge_color = 'darkred'
            alpha = 0.4
        # elif agents_inside == self.num_agents:
        #     face_color = 'cyan'
        #     edge_color = 'darkblue'
        #     alpha = 0.3
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
        # colors = ['red', 'orange', 'purple', 'green']
        # for i in range(self.num_agents):
        #     x, y, theta, v = self.agent_states[i]
        #     inside = self.patch.is_inside(x, y)
        #     collision = self.current_base_obs["collisions"][i] > 0.5
            
        #     color = colors[i % len(colors)]
        #     if collision:
        #         marker = 'X'
        #         size = 18
        #     elif inside:
        #         marker = 'o'
        #         size = 12
        #     else:
        #         marker = 's'  # Square if outside patch
        #         size = 14
            
        #     self._ax.plot(x, y, marker, color=color, markersize=size,
        #                  markeredgecolor='black', markeredgewidth=2)
            
        #     # Agent velocity arrow
        #     self._ax.arrow(
        #         x, y,
        #         v * np.cos(theta) * 0.2, v * np.sin(theta) * 0.2,
        #         head_width=0.1, head_length=0.05,
        #         fc=color, ec=color, alpha=0.7
        #     )
            
            # Label agents
            # self._ax.text(x + 0.3, y + 0.3, f'R{i}', fontsize=8, fontweight='bold')
        
        # Draw next waypoint if available
        # if self.waypoints is not None and len(self.waypoints) > 0:
        #     # Find nearest waypoint
        #     patch_pos = np.array([self.patch.x, self.patch.y])
        #     dists = np.linalg.norm(self.waypoints - patch_pos, axis=1)
        #     nearest_idx = np.argmin(dists)
        #     next_idx = (nearest_idx + 10) % len(self.waypoints)
        #     self._ax.plot(self.waypoints[next_idx, 0], self.waypoints[next_idx, 1],
        #                  'g*', markersize=20, markeredgecolor='black', markeredgewidth=1)
        
        # Info box
        info_text = (
            f'Reward: {self.episode_reward:.1f}\n'
            # f'Patch collisions: {self.patch_wall_collisions}\n'
            # f'Min clearance: {self.patch_min_clearance:.2f}m'
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
    # navigation_mode: str = "landmark",
    # debug_print_every_n_steps: int = 0,
    # debug_print_episode_end: bool = True,
    base_reset_type: str = "rl_random_static",
    # use_base_done_termination: bool = False,
    # patch_only_mode: bool = False,
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
            # navigation_mode=navigation_mode,
            # debug_print_every_n_steps=debug_print_every_n_steps,
            # debug_print_episode_end=debug_print_episode_end,
            base_reset_type=base_reset_type,
            # use_base_done_termination=use_base_done_termination,
            # patch_only_mode=patch_only_mode,
        )
        env = PatchEnv(cfg)
        env.reset(seed=seed + rank)
        return env

    if SB3_AVAILABLE:
        set_random_seed(seed + rank)
    return _init




