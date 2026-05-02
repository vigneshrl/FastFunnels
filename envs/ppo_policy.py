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
from typing import Any, Optional
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from scipy import ndimage

try:
    # from .action import PatchAction, PatchActionConfig
    from .f110_env import F110Config, F110EnvAdapter
    # from .laser_model import LidarModel
    # from .mpc import MPCConfig, SEMPCSolver, SafetyConfig, SafetyLayer
    # from .observation import PatchObservationBuilder
    from .patch import DynamicPatch
    from .reset.map_reset import MapResetHelper
except ImportError:
    # Allow running as a standalone script from repo root.
    # from envs.action import PatchAction, PatchActionConfig
    from envs.f110_env import F110Config, F110EnvAdapter
    # from envs.laser_model import LidarModel
    # from envs.mpc import MPCConfig, SEMPCSolver, SafetyConfig, SafetyLayer
    # from envs.observation import PatchObservationBuilder
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
    """
    Simplified config matching single-agent PPO structure.
    Patch is treated as an inflated agent with fixed size.
    """
    # Environment basics
    num_agents: int = 0  # Patch-only, no agents
    control_dt: float = 0.01
    
    base_reset_type: str = "rl_random_static"
    render_mode: Optional[str] = None
    max_steps: int = 10000 #was 100,000
    domain_randomize: bool = False
    
    # Frenet reward (same as single-agent)
    reward_progress_scale: float = 40.0
    reward_crosstrack_weight: float = 2.0
    reward_steer_bias_weight: float = 0.5    # penalise |steer|, breaks max-steer local optimum
    reward_steer_rate_weight: float = 0.0   # discourages jerky steering
    spin_yawrate_threshold: float = 3.0     # only penalize ω > 3 rad/s (spinning, not cornering)
    reward_spin_weight: float = 0.5         # penalty for spinning (yaw_rate > threshold)
    reward_speed_weight: float = 0.0        # disabled — speed reward caused fast-crash local optimum
    collision_penalty: float = 500.0
    collision_min_dist: float = 0.25
    stuck_no_progress_steps: int = 30
    stuck_progress_eps: float = 1e-3
    stuck_penalty: float = 200.0
    lap_finish_bonus: float = 2000.0
    lap_bonus_tau: float = 200.0
    time_penalty_per_sec: float = 2.0  # set >0 to prefer faster laps (subtracted per step as rate*dt)
    
    # Lidar config (match single-agent)
    # num_beams: int = 108  # Same as single-agent ENV_CONFIG
    #  obstacle_coords = 
    
    # Vehicle dynamics
    wheelbase: float = 0.33
    robot_radius: float = 0.15
    steering_max: float = 0.4189  # max steering angle in radians (~24 degrees)
    
    # Patch size at spawn
    patch_a: float = 2.0  # semi-major axis (length)
    patch_b: float = 1.5  # semi-minor axis (width)

    # b_cmd / a_cmd action bounds
    # b_cmd_min = 1.0 because 2 agents need at least b=1.0 to fit side-by-side in the patch
    b_cmd_min: float = 1.0
    b_cmd_max: float = 1.5
    a_cmd_min: float = 1.0
    a_cmd_max: float = 2.0

    shape_area_penalty_weight: float = 0.0
    random_spawn: bool = True

    # Patch boundary collision detection
    patch_boundary_violation_threshold: float = 0.05

    # Lookahead distances for observation
    # patch_lookahead_5m:  float = 5.0    # existing single lookahead made explicit
    # patch_lookahead_10m: float = 10.0   # new: half-width 10m ahead
    # patch_lookahead_15m: float = 15.0   # new: half-width 15m ahead

    # === SPLIT/MERGE MODE DISABLED FOR DEBUGGING ===
    # lookahead_dist: float = 5.0
    split_mode: bool = False
    # split_ahead_half_w: float = 2.0  # m — split when less than this half-width ahead
    # merge_ahead_half_w: float = 1.1   # m — merge when more than this half-width ahead
    # lane_centering_weight: float = 2.0
    

# @dataclass 
# class AgentEnvConfig:
#     control_dt: float = 0.01
#     num_agents: int = 2
#     patch_env: Optional[Any] = None
#     # base_reset_type: str = "rl_random_static"
#     render_mode: Optional[str] = None
#     random_spawn: bool = False
#     max_steps: int = 5000 # was 100,000
#     patch_a: float = 2.0   # must match PatchEnvConfig.patch_a — frozen policy was trained with this
#     patch_b: float = 1.5   # must match PatchEnvConfig.patch_b

#     wheelbase: float = 0.33
#     robot_radius: float = 0.15

#     out_of_patch_penalty: float = 200.0
#     inter_collision_penalty: float = 150.0
#     inside_patch_reward: float = 30.0
#     survival_reward_per_step: float = 2.0  # bonus per step alive inside patch

#     agents_random_spawn: bool = False
#     # lap_finish_bonus: float = 2000.0
#     # lap_bonus_tau: float = 200.0

#     inter_agent_collision_dist: float = 0.20
#     patch_boundary_violation_threshold: float = 0.10  # fraction of boundary pts in wall → collision


@dataclass
class JointEnvConfig:
    """Config for joint training: patch car + 1 learning agent trained simultaneously.

    The patch car is a real f110 car (car 1) providing physical inertia.
    The learning agent is car 0. DynamicPatch is used only for ellipse geometry.
    """
    control_dt: float = 0.01
    render_mode: Optional[str] = None
    random_spawn: bool = False
    max_steps: int = 3000 #100000
    patch_a: float = 2.0  # Match PatchEnvConfig for Phase 0 → Phase A consistency
    patch_b: float = 1.5  # Match PatchEnvConfig for Phase 0 → Phase A consistency
    # Agent reward — flat inside, warning ramp near boundary, penalty outside
    out_of_patch_penalty: float = 1500.0
    inter_collision_penalty: float = 1000.0
    inside_reward: float = 1.0               # flat reward per step while inside patch
    warning_zone_start: float = 0.85         # dn threshold where warning ramp begins (dn=1.0 = boundary)
    # Repulsion: agent vs patch car proximity avoidance
    repulsion_zone: float = 0.15             # metres — tight so agent can navigate narrow patches
    repulsion_weight: float = 30.0
    patch_car_collision_dist: float = 0.35   # metres — f1tenth body ~0.22m; 0.35 fires before physics pushes patch car
    # patch_boundary_violation_threshold: float = 0.02  # 1/64 points → immediate detection
    # === PATCH TRAINING REWARD CONFIG — disabled, patch driven by frozen policy ===
    # reward_progress_scale: float = 40.0
    # reward_crosstrack_weight: float = 2.0
    # reward_steer_bias_weight: float = 0.0
    # reward_steer_rate_weight: float = 0.0
    # spin_yawrate_threshold: float = 0.0
    # reward_spin_weight: float = 0.0
    # patch_wall_penalty: float = 1500.0
    # stuck_no_progress_steps: int = 60
    # stuck_progress_eps: float = 1e-3
    # stuck_penalty: float = 10.0
    # patch_lap_bonus: float = 2000.0
    # lap_bonus_tau: float = 200.0
    # wheelbase: float = 0.33
    # shape_area_penalty_weight: float = 0.0
    # time_penalty_per_sec: float = 0.0
    # agents_inside_bonus: float = 30.0
    # patch_min_speed_frac: float = 0.3
    # patch_lookahead_dist: float = 5.0
    # steering_max: float = 0.4189
    # ============================================================================
    patch_boundary_violation_threshold: float = 0.05
    # Patch action clipping bounds (still needed to clip frozen-policy outputs)
    patch_a_cmd_min: float = 1.0
    patch_a_cmd_max: float = 2.0
    patch_b_cmd_min: float = 1.0
    patch_b_cmd_max: float = 1.5
    agent_out_of_patch_threshold: float = 1.5


# class AgentEnv(gym.Env):
#     "Agent enviroment which is using the patch as the place to navigate"

#     metadata = { "render_modes" : ["human", "rgb_array", None]}

#     def __init__(self, config: Optional[AgentEnvConfig] = None):
#         super().__init__()
#         self.cfg = config or AgentEnvConfig()
#         self.patch_env = self.cfg.patch_env
#         # Action: [steer_delta, speed_delta]
#         # steer_delta — correction on top of patch co-driver steer (keeps agents
#         #               stable during early training; random deltas stay aligned)
#         # speed_delta — correction on top of frozen patch speed baseline
#         self.action_space = spaces.Box(
#             low=np.array([-0.4189, -1.0], dtype=np.float32),
#             high=np.array([0.4189,  1.0], dtype=np.float32),
#             dtype=np.float32,
#         )
#         # Obs: 24D — ego 12D concatenated with partner 12D (for CTDE centralized critic)
#         # obs[0:12]  = ego agent's 12D obs (actor only uses this slice)
#         # obs[12:24] = partner agent's 12D obs (critic uses full 24D)
#         # Layout of each 12D block:
#         # [x_norm, y_norm,            ← position in patch
#         #  ot_x_norm, ot_y_norm,      ← other agent position
#         #  dx_other,  dy_other,       ← separation vector
#         #  patch_yaw_rate, patch_v,   ← patch dynamics
#         #  a, b, speed, heading_rel]  ← shape + own kinematics
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32,
#         )
#         self._ego_idx = 0  # randomised each reset — breaks agent0/agent1 asymmetry

#         # 3-car simulation: car 0 = patch (frozen policy, real f110 inertia),
#         # cars 1 & 2 = learning agents.
#         # Using real f110 physics for the patch restores the inertia it had during
#         # PatchEnv training — the kinematic DynamicPatch had no inertia and sprinted
#         # ahead of the physical agent cars during the acceleration phase.
#         self.f110 = F110EnvAdapter(
#             F110Config(
#                 num_agents=3,
#                 control_input=("speed", "steering_angle"),
#                 timestep=self.cfg.control_dt,
#             ),
#             render_mode=self.cfg.render_mode,
#         )
#         self.base_env = self.f110
#         self.map_reset = MapResetHelper()

#         # DynamicPatch is synced from f110 car 0's physical position each step.
#         # Used only for geometry: world_to_patch_frame, check_boundary_collision,
#         # update_shape, and obs building. No longer stepped kinematically.
#         self.patch = DynamicPatch()
#         self.patch.a = self.cfg.patch_a
#         self.patch.b = self.cfg.patch_b

#         self.num_agents = 2  # learning agents (f110 cars 1 and 2)
#         self.track_spline = None
#         self.track_length = None
#         self._occ_map = None
#         self._resolution = None
#         self._origin = None
#         self.step_count = 0
#         self.episode_reward = 0.0
#         self.current_base_obs = None
#         self._fig = None
#         self._ax = None
#         self._term_counts = {"out_of_patch": 0, "inter_collision": 0, "max_steps": 0, "patch_wall": 0}
#         self.prev_patch_theta = 0.0
#         self.patch_yaw_rate = 0.0
#         self.patch_steer = 0.0
#         self._episode_count = 0
#         self.prev_dist_norms = [0.0, 0.0]  # tracked for shaped reward

#         # --- AgentView synchronization (parameter sharing) ---
#         # Two AgentView instances share this env. Each independently predicts
#         # a 2D action. Physics executes once BOTH actions are submitted.
#         self._pending_actions: list = [None, None]
#         self._step_obs:        list = [None, None]
#         self._step_rewards:    list = [0.0, 0.0]
#         self._step_terminated: bool = False
#         self._step_truncated:  bool = False
#         self._step_info:       dict = {}


#     #     # Frenet bookkeeping - per agent
#     #     self.track_spline = None
#     #     self.track_length = None
#     #     self.prev_rel = [None, None]
#     #     self.no_progress_counter = [0, 0]
#     #     self.step_count = 0
#     #     self.episode_reward = 0.0
#     #     self._last_reward_terms = {}
#     #     self.current_base_obs = None
#     #     self.num_agents = 2
#     #     self._fig = None
#     #     self._ax = None
    
#     # # def _wrap_ds(self, ds: float) -> float:
#     # #     if self.track_length is None or not np.isfinite(self.track_length):
#     # #         return ds  # Safe fallback
#     # #     L = float(self.track_length)
#     # #     if L <= 0:
#     # #         return ds
#     # #     return (ds + 0.5 * L) % L - 0.5 * L

#     # @staticmethod
#     # def _wrap_angle(angle_rad: float) -> float:
#     #     return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

#     # def _patch_to_frenet(self) -> tuple[float, float]:
#     #     if self.track_spline is None:
#     #         # Return safe fallback values (like single-car would never reach here)
#     #         return 0.0, 0.0
        
#     #     try:
#     #         s, ey = self.track_spline.calc_arclength_inaccurate(float(self.patch.x), float(self.patch.y))
#     #         # Check for NaN (single-car doesn't need this but multi-agent might)
#     #         if not np.isfinite(s) or not np.isfinite(ey):
#     #             return 0.0, 0.0
#     #         return float(s), float(ey)
#     #     except (AttributeError, TypeError):
#     #         return 0.0, 0.0

#     # def _estimate_current_track_width(self) -> float:
#     #     """Estimate local track width at current patch pose."""
#     #     try:
#     #         _, occ_map, resolution, origin = self.f110.get_track_data()
#     #         width = float(
#     #             self.map_reset.estimate_track_width(
#     #                 occ_map,
#     #                 resolution,
#     #                 origin,
#     #                 float(self.patch.x),
#     #                 float(self.patch.y),
#     #                 float(self.patch.theta),
#     #             )
#     #         )
#     #         if not np.isfinite(width) or width <= 0.0:
#     #             return float(max(2.0 * self.patch.b, 1.0))
#     #         return width
#     #     except Exception:
#     #         return float(max(2.0 * self.patch.b, 1.0))

#     # def _agent_to_frenet(self, x: float, y: float) -> tuple[float, float]:
#     #     """Convert agent world position to Frenet (s, ey)."""
#     #     if self.track_spline is None:
#     #         return 0.0, 0.0
#     #     try:
#     #         s, ey = self.track_spline.calc_arclength_inaccurate(float(x), float(y))
#     #         if not np.isfinite(s) or not np.isfinite(ey):
#     #             return 0.0, 0.0
#     #         return float(s), float(ey)
#     #     except (AttributeError, TypeError):
#     #         return 0.0, 0.0
    
#     def _sync_patch_from_car0(self, base_obs) -> None:
#         """Sync DynamicPatch geometry from f110 car 0 (the physical patch car).
#         Called after every f110.reset() and f110.step() so the patch ellipse
#         always reflects the real physical position of the patch car.
#         """
#         self.patch.x = float(base_obs["poses_x"][0])
#         self.patch.y = float(base_obs["poses_y"][0])
#         self.patch.theta = float(base_obs["poses_theta"][0])
#         vx = float(base_obs["linear_vels_x"][0]) if "linear_vels_x" in base_obs else 0.0
#         vy = float(base_obs["linear_vels_y"][0]) if "linear_vels_y" in base_obs else 0.0
#         self.patch.v = float(np.hypot(vx, vy))

#     def _build_agent_obs_for(self, ego: int, base_obs) -> np.ndarray:
#         """Build 12D ego-centric observation for logical agent `ego` (0 or 1).
#         Logical agent 0 → f110 car 1, logical agent 1 → f110 car 2.
#         (f110 car 0 is reserved for the patch car.)
#         """
#         other = 1 - ego
#         # Map logical agent index to f110 car index
#         f_ego  = ego  + 1   # f110 index for ego agent
#         f_other = other + 1  # f110 index for other agent
#         a = max(float(self.patch.a), 1e-3)
#         b = max(float(self.patch.b), 1e-3)

#         def _get_patch_pos(fi):
#             x = float(base_obs["poses_x"][fi])
#             y = float(base_obs["poses_y"][fi])
#             return self.patch.world_to_patch_frame(x, y)

#         def _get_speed(fi):
#             vx = float(base_obs["linear_vels_x"][fi]) if "linear_vels_x" in base_obs else 0.0
#             vy = float(base_obs["linear_vels_y"][fi]) if "linear_vels_y" in base_obs else 0.0
#             return float(np.hypot(vx, vy))

#         def _get_heading_rel(fi):
#             yaw = float(base_obs["poses_theta"][fi]) if "poses_theta" in base_obs else 0.0
#             return float((yaw - self.patch.theta + np.pi) % (2 * np.pi) - np.pi)

#         # Ego agent
#         ex, ey = _get_patch_pos(f_ego)
#         my_x_norm, my_y_norm = ex / b, ey / a

#         # Other agent
#         ox, oy = _get_patch_pos(f_other)
#         ot_x_norm, ot_y_norm = ox / b, oy / a

#         # Separation from ego's perspective
#         dx_other = ex - ox
#         dy_other = ey - oy

#         patch_yaw_rate = float(np.clip(self.patch_yaw_rate, -10.0, 10.0))
#         patch_v = float(self.patch.v)

#         return np.array([
#             my_x_norm, my_y_norm,
#             ot_x_norm, ot_y_norm,
#             dx_other, dy_other,
#             patch_yaw_rate, patch_v, a, b,
#             _get_speed(f_ego), _get_heading_rel(f_ego),
#         ], dtype=np.float32)

#     def _build_agent_obs(self, base_obs) -> np.ndarray:
#         """Backward-compat wrapper: builds obs for the current _ego_idx."""
#         return self._build_agent_obs_for(self._ego_idx, base_obs)

#     def _compute_agent_reward(self, dist_norm: float, prev_dist_norm: float,
#                               inter_collision: bool) -> float:
#         """Per-agent reward with position shaping + progress shaping.
#         dist_norm: 0=center, 1=patch edge, >1=outside.
#         prev_dist_norm: dist_norm from previous step — used for shaped reward.
#         """
#         if dist_norm <= 1.0:
#             # Inside: position reward highest at center + survival bonus
#             reward = self.cfg.inside_patch_reward * float(np.exp(-2.0 * dist_norm ** 2))
#             # reward += self.cfg.survival_reward_per_step
#             # Shaped: reward for moving toward center, penalise moving away
#             # prev - curr > 0 means closer than before → positive
#             reward += 5.0 * (prev_dist_norm - dist_norm)
#         elif dist_norm <= 1.5:
#             # Buffer zone: escalating penalty, still alive
#             reward = -self.cfg.out_of_patch_penalty * (dist_norm - 1.0) / 0.5
#         else:
#             # Hard outside: full penalty
#             reward = -self.cfg.out_of_patch_penalty
#         if inter_collision:
#             reward -= self.cfg.inter_collision_penalty
#         return float(np.clip(reward, -200.0, 200.0))

#     def _check_agent_termination(
#         self, agent_idx: int, dist_norm: float, inter_collision: bool
#     ) -> tuple[bool, bool, str]:
#         """Terminate only when dist_norm > 1.5 (buffer zone gives gradient before cliff)."""
#         if inter_collision:
#             return True, False, f"agent{agent_idx}_inter_collision"
#         if dist_norm > 1.5:
#             return True, False, f"agent{agent_idx}_out_of_patch"
#         if self.step_count >= self.cfg.max_steps:
#             return False, True, "max_steps"
#         return False, False, None
    
#     def reset(self, seed=None, options=None):
#         """Reset: spawn 2 agents inside patch, initialize frozen patch policy."""
#         super().reset(seed=seed)
#         self.step_count = 0
#         self.episode_reward = 0.0
#         self.prev_dist_norms = [0.0, 0.0]
#         # Randomise ego agent each episode — both agents train equally as "ego"
#         self._ego_idx = int(np.random.randint(0, 2))

#         self.f110.ensure_initialized()
#         track, self._occ_map, self._resolution, self._origin = self.f110.get_track_data()
#         if track.centerline is None or track.centerline.spline is None:
#             raise ValueError("Track centerline/spline missing.")

#         self.track_spline = track.centerline.spline
#         self.track_length = float(self.track_spline.s[-1])

#         # Sync frozen patch_env with track data so get_patch_action builds correct obs.
#         # PatchEnv.reset() is never called in AgentEnv — so we must seed its caches here.
#         # _precompute_track_widths is expensive (200 ray casts) — only do it once per env,
#         # not every episode reset.
#         if self.patch_env is not None:
#             self.patch_env._occ_map    = self._occ_map
#             self.patch_env._resolution = self._resolution
#             self.patch_env._origin     = self._origin
#             self.patch_env.track_spline  = self.track_spline
#             self.patch_env.track_length  = self.track_length
#             if self.patch_env._tw_s_vals is None:
#                 self.patch_env._precompute_track_widths(n_samples=200)

#         xs = np.asarray(track.centerline.xs, dtype=np.float32)
#         ys = np.asarray(track.centerline.ys, dtype=np.float32)
#         spawn_idx = int(np.random.randint(0, xs.shape[0])) if self.cfg.random_spawn else 0
#         spawn_idx = max(0, min(spawn_idx, xs.shape[0] - 1))
#         patch_x, patch_y = float(xs[spawn_idx]), float(ys[spawn_idx])
#         next_idx = (spawn_idx + 1) % xs.shape[0]
#         patch_theta = float(np.arctan2(
#             float(ys[next_idx] - ys[spawn_idx]),
#             float(xs[next_idx] - xs[spawn_idx])
#         ))

#         self.patch = DynamicPatch(
#             x=patch_x, y=patch_y, theta=patch_theta,
#             v=0.5, a=self.cfg.patch_a, b=self.cfg.patch_b
#         )
#         self.prev_patch_theta = patch_theta
#         self.patch_yaw_rate = 0.0
#         self.patch_steer = 0.0
#         if self.patch_env is not None:
#             self.patch_env.patch = self.patch
#         perp_dx = -np.sin(patch_theta)
#         perp_dy = np.cos(patch_theta)

#         if self.cfg.agents_random_spawn:
#             a = max(float(self.patch.a), 1e-3)
#             b = max(float(self.patch.b), 1e-3)
#             min_sep = 0.30
#             max_tries = 50
#             cos_t, sin_t = np.cos(patch_theta), np.sin(patch_theta)

#             def _sample_in_ellipse():
#                 for _ in range(200):
#                     xr = np.random.uniform(-b * 0.5, b * 0.5)
#                     yr = np.random.uniform(-a * 0.5, a * 0.5)
#                     if (xr / b) ** 2 + (yr / a) ** 2 < 0.25:
#                         wx = patch_x + cos_t * yr - sin_t * xr
#                         wy = patch_y + sin_t * yr + cos_t * xr
#                         return wx, wy
#                 return patch_x, patch_y

#             for _ in range(max_tries):
#                 w0x, w0y = _sample_in_ellipse()
#                 w1x, w1y = _sample_in_ellipse()
#                 if np.hypot(w0x - w1x, w0y - w1y) >= min_sep:
#                     break

#             poses = np.array([
#                 [patch_x, patch_y, patch_theta],  # car 0 = patch car, at center
#                 [w0x, w0y, patch_theta],           # car 1 = agent 0
#                 [w1x, w1y, patch_theta],           # car 2 = agent 1
#             ], dtype=np.float32)
#         else:
#             offset = 0.50
#             poses = np.array([
#                 [patch_x, patch_y, patch_theta],                                              # car 0 = patch car
#                 [patch_x + offset * perp_dx, patch_y + offset * perp_dy, patch_theta],       # car 1 = agent 0 (left)
#                 [patch_x - offset * perp_dx, patch_y - offset * perp_dy, patch_theta],       # car 2 = agent 1 (right)
#             ], dtype=np.float32)

#         base_obs, _ = self.f110.reset(poses=poses)
#         self.current_base_obs = base_obs

#         # Sync DynamicPatch geometry from the physical patch car (f110 car 0)
#         self._sync_patch_from_car0(base_obs)

#         # Pre-cache 24D joint obs for CTDE: [ego_12D | partner_12D]
#         ego0 = np.nan_to_num(self._build_agent_obs_for(0, base_obs),
#                               nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
#         ego1 = np.nan_to_num(self._build_agent_obs_for(1, base_obs),
#                               nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
#         self._step_obs[0] = np.concatenate([ego0, ego1])  # agent0: ego=0, partner=1
#         self._step_obs[1] = np.concatenate([ego1, ego0])  # agent1: ego=1, partner=0
#         self._pending_actions = [None, None]
#         self._step_rewards    = [0.0, 0.0]
#         self._step_terminated = False
#         self._step_truncated  = False
#         self._step_info       = {}

#         obs = self._step_obs[self._ego_idx]  # 24D joint obs already built above
#         return obs.copy(), {}

#     def _execute_combined_step(self, action0: np.ndarray, action1: np.ndarray):
#         """Execute one physics step with independent actions for each agent.

#         action0: [steer_delta, speed_delta] for physical agent 0
#         action1: [steer_delta, speed_delta] for physical agent 1

#         Updates _step_obs, _step_rewards, _step_terminated, _step_truncated,
#         _step_info in-place. Called by AgentView when both actions are ready.
#         """
#         self.step_count += 1
#         dt = self.cfg.control_dt

#         # --- Frozen patch policy acts on DynamicPatch state (synced from car 0) ---
#         if self.patch_env is not None:
#             self.patch_env.patch = self.patch
#             patch_action = self.patch_env.get_patch_action(self.patch, self.track_spline)
#             steer_p = float(np.clip(patch_action[0], -0.4189, 0.4189))
#             speed_p = float(np.clip(patch_action[1], 0.5, 10.0))
#             a_p = float(np.clip(patch_action[2], 1.0, self.cfg.patch_a))
#             b_p = float(np.clip(patch_action[3], 1.0, self.cfg.patch_b))
#         else:
#             steer_p, speed_p, a_p, b_p = 0.0, 2.0, self.cfg.patch_a, self.cfg.patch_b

#         # --- Decode agent actions (steer/speed deltas on top of patch baseline) ---
#         def _decode(raw):
#             sd = float(np.nan_to_num(raw[0], nan=0.0, posinf=0.4189, neginf=-0.4189))
#             vd = float(np.nan_to_num(raw[1], nan=0.0))
#             s  = float(np.clip(steer_p + sd, -0.4189, 0.4189))
#             v  = float(np.clip(speed_p + vd, 0.5, 10.0))
#             return s, v

#         s0, v0 = _decode(action0)
#         s1, v1 = _decode(action1)

#         # --- Step all 3 cars together ---
#         # car 0 = patch car (frozen policy), cars 1,2 = learning agents
#         base_obs, _, _, _, _ = self.f110.step(
#             np.array([[steer_p, speed_p],   # patch car: frozen policy command
#                       [s0, v0],             # agent 0
#                       [s1, v1]], dtype=np.float32)  # agent 1
#         )
#         self.current_base_obs = base_obs

#         # --- Sync patch geometry from car 0's physical position (with inertia) ---
#         prev_theta = self.patch.theta
#         self._sync_patch_from_car0(base_obs)
#         d_theta = (self.patch.theta - prev_theta + np.pi) % (2 * np.pi) - np.pi
#         self.patch_yaw_rate = d_theta / dt

#         # Update patch shape (a, b can grow/shrink per policy output)
#         self.patch.update_shape(a_p, b_p, dt, max_a=self.cfg.patch_a, max_b=self.cfg.patch_b)

#         # --- Patch wall collision (uses real physical patch position now) ---
#         patch_wall_hit = False
#         if self._occ_map is not None:
#             patch_wall_hit, _ = self.patch.check_patch_boundary_wall_collision(
#                 self._occ_map, self._resolution, self._origin,
#                 n_points=32,
#                 violation_threshold=self.cfg.patch_boundary_violation_threshold,
#             )

#         # --- Per-agent inside-patch check (logical agents 0,1 = f110 cars 1,2) ---
#         dist_norm_list = []
#         for i in range(2):
#             fi = i + 1  # f110 car index for logical agent i
#             x = float(base_obs["poses_x"][fi])
#             y = float(base_obs["poses_y"][fi])
#             x_rel, y_rel = self.patch.world_to_patch_frame(x, y)
#             dist_norm = float(np.sqrt(
#                 (x_rel / max(self.patch.a, 1e-3)) ** 2 +
#                 (y_rel / max(self.patch.b, 1e-3)) ** 2
#             ))
#             dist_norm_list.append(dist_norm)

#         # Inter-agent distance (cars 1 and 2)
#         x0w, y0w = float(base_obs["poses_x"][1]), float(base_obs["poses_y"][1])
#         x1w, y1w = float(base_obs["poses_x"][2]), float(base_obs["poses_y"][2])
#         inter_dist = float(np.hypot(x0w - x1w, y0w - y1w))
#         inter_collision = inter_dist < self.cfg.inter_agent_collision_dist

#         # --- Per-agent reward and termination ---
#         total_reward = 0.0
#         terminated_flags = []
#         reasons = []
#         for i in range(2):
#             r = self._compute_agent_reward(dist_norm_list[i], self.prev_dist_norms[i], inter_collision)
#             term, trunc, reason = self._check_agent_termination(i, dist_norm_list[i], inter_collision)
#             self._step_rewards[i] = float(np.clip(r, -200.0, 200.0))
#             total_reward += r
#             terminated_flags.append(term or trunc)
#             reasons.append(reason)
#         self.prev_dist_norms = list(dist_norm_list)

#         total_reward = float(np.clip(total_reward / 2.0, -200.0, 200.0))
#         self.episode_reward += total_reward
#         terminated = any(terminated_flags) or patch_wall_hit
#         truncated  = self.step_count >= self.cfg.max_steps
#         if patch_wall_hit:
#             reason = "patch_wall_collision"
#         else:
#             reason = next((r for r in reasons if r is not None), "max_steps" if truncated else None)

#         # --- Build 24D joint obs for CTDE: [ego_12D | partner_12D] ---
#         ego0 = np.nan_to_num(self._build_agent_obs_for(0, base_obs),
#                               nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
#         ego1 = np.nan_to_num(self._build_agent_obs_for(1, base_obs),
#                               nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
#         self._step_obs[0] = np.concatenate([ego0, ego1])
#         self._step_obs[1] = np.concatenate([ego1, ego0])

#         if terminated or truncated:
#             self._episode_count += 1
#             if reason == "patch_wall_collision":
#                 self._term_counts["patch_wall"] += 1
#             elif reason and "inter_collision" in reason:
#                 self._term_counts["inter_collision"] += 1
#             elif reason and "out_of_patch" in reason:
#                 self._term_counts["out_of_patch"] += 1
#             else:
#                 self._term_counts["max_steps"] += 1
#             if self._episode_count % 200 == 0:
#                 n = self._episode_count
#                 print(
#                     f"[AgentEnv ep={n}] out_of_patch={self._term_counts['out_of_patch']/n*100:.1f}%  "
#                     f"inter_collision={self._term_counts['inter_collision']/n*100:.1f}%  "
#                     f"patch_wall={self._term_counts['patch_wall']/n*100:.1f}%  "
#                     f"max_steps={self._term_counts['max_steps']/n*100:.1f}%  "
#                     f"| now: inter_dist={inter_dist:.3f}m  patch_v={self.patch.v:.2f}"
#                 )

#         self._step_terminated = terminated
#         self._step_truncated  = truncated
#         self._step_info = {
#             "episode_reward": self.episode_reward,
#             "Episode_steps": self.step_count,
#             "step_reward": total_reward,
#             "termination_reason": reason,
#             "inter_agent_dist": float(inter_dist),
#             "agent0_speed": float(v0),
#             "agent1_speed": float(v1),
#         }

#     def step(self, action):
#         """Standalone step (backward compat). Both agents use the same action.
#         For true parameter sharing use AgentView instead of this directly.
#         """
#         action = np.asarray(action, dtype=np.float32)
#         self._execute_combined_step(action, action)
#         # Return ego obs (randomised _ego_idx) + average reward
#         obs = self._step_obs[self._ego_idx]
#         avg_reward = float(np.mean(self._step_rewards))
#         return obs, avg_reward, self._step_terminated, self._step_truncated, self._step_info

#     def render(self):
#         return self.f110.render()

#     def _visualize(self):
#         """Visualize patch and agents with wall collision info."""
#         # --- First call: setup figure and per-visualisation cache ---
#         if self._fig is None:
#             plt.ion()
#             self._fig, self._ax = plt.subplots(figsize=(8, 6))
#             self._ax.set_aspect('equal')
#             self._ax.grid(True, alpha=0.3)
#             self._vis_bg_image = None
#             self._vis_bg_bounds = None
#             self._vis_dynamic_artists = []
#             self._vis_occ_cache = None
#             plt.show(block=False)

#         ax = self._ax

#         # --- Remove dynamic artists from the previous frame (no ax.clear()) ---
#         for artist in self._vis_dynamic_artists:
#             try:
#                 artist.remove()
#             except Exception:
#                 pass
#         self._vis_dynamic_artists = []

#         # --- Occupancy map: fetched once and cached (it never changes) ---
#         occ_map = resolution = origin = None
#         if self._vis_occ_cache is None and self.base_env is not None:
#             try:
#                 _, occ_map, resolution, origin = self.base_env.get_track_data()
#                 self._vis_occ_cache = (occ_map, resolution, origin)
#             except Exception as e:
#                 print(f"Warning: Could not get track data: {type(e).__name__}: {e}")
#         if self._vis_occ_cache is not None:
#             occ_map, resolution, origin = self._vis_occ_cache

#         # --- Redraw background imshow only when the view window shifts ---
#         cx, cy = self.patch.x, self.patch.y
#         margin = max(self.patch.a, self.patch.b) + 10
#         if occ_map is not None:
#             px_min = max(0, int((cx - margin - origin[0]) / resolution))
#             px_max = min(occ_map.shape[1], int((cx + margin - origin[0]) / resolution))
#             py_min = max(0, int((cy - margin - origin[1]) / resolution))
#             py_max = min(occ_map.shape[0], int((cy + margin - origin[1]) / resolution))
#             new_bounds = (px_min, px_max, py_min, py_max)

#             if self._vis_bg_bounds != new_bounds and px_max > px_min and py_max > py_min:
#                 if self._vis_bg_image is not None:
#                     try:
#                         self._vis_bg_image.remove()
#                     except Exception:
#                         pass
#                 region = occ_map[py_min:py_max, px_min:px_max]
#                 xw0 = px_min * resolution + origin[0]
#                 xw1 = px_max * resolution + origin[0]
#                 yw0 = py_min * resolution + origin[1]
#                 yw1 = py_max * resolution + origin[1]
#                 # Build RGBA image: one rasterised call instead of contourf vector ops
#                 rgba = np.zeros((*region.shape, 4), dtype=np.uint8)
#                 rgba[region < 0.5] = [51, 51, 51, 204]    # dark walls
#                 rgba[region >= 0.5] = [220, 220, 220, 60]  # light free space
#                 self._vis_bg_image = ax.imshow(
#                     rgba, extent=[xw0, xw1, yw0, yw1],
#                     origin='lower', aspect='auto', zorder=0, interpolation='nearest')
#                 self._vis_bg_bounds = new_bounds

#         # --- Agent positions ---
#         agent_positions = []
#         if self.current_base_obs is not None:
#             for i in range(self.num_agents):
#                 x = float(self.current_base_obs["poses_x"][i])
#                 y = float(self.current_base_obs["poses_y"][i])
#                 theta = float(self.current_base_obs["poses_theta"][i])
#                 v = float(self.current_base_obs["linear_vels_x"][i])
#                 agent_positions.append((x, y, theta, v))
#         else:
#             agent_positions = [(self.patch.x, self.patch.y, self.patch.theta, 0.0)] * self.num_agents

#         agents_inside = sum(1 for x, y, _, _ in agent_positions if self.patch.is_inside(x, y))

#         # --- Wall collision check ---
#         patch_wall_collision = False
#         violated = []
#         if occ_map is not None and resolution is not None and origin is not None:
#             patch_wall_collision, violated = self.patch.check_patch_boundary_wall_collision(
#                 occ_map, resolution, origin, n_points=32)

#         wall_status = f"⚠️ WALL! ({len(violated)} pts)" if patch_wall_collision else "Clear"
#         ax.set_title(
#             f'Patch Funnel V1 | Step {self.step_count}\n'
#             f'Patch: v={self.patch.v:.1f}m/s, size=({self.patch.a:.1f}, {self.patch.b:.1f}) | '
#             f'Inside: {agents_inside}/{self.num_agents} | {wall_status}',
#             fontsize=11, color='red' if patch_wall_collision else 'black'
#         )

#         # --- Patch ellipse ---
#         if patch_wall_collision:
#             face_color, edge_color, alpha = 'red', 'darkred', 0.4
#         elif agents_inside == self.num_agents:
#             face_color, edge_color, alpha = 'cyan', 'darkblue', 0.3
#         else:
#             face_color, edge_color, alpha = 'yellow', 'orange', 0.3

#         ellipse = Ellipse(
#             xy=(self.patch.x, self.patch.y),
#             width=self.patch.a * 2, height=self.patch.b * 2,
#             angle=np.degrees(self.patch.theta),
#             facecolor=face_color, edgecolor=edge_color, alpha=alpha, linewidth=3
#         )
#         ax.add_patch(ellipse)
#         self._vis_dynamic_artists.append(ellipse)

#         # --- Patch boundary points ---
#         for bx, by in self.patch.get_boundary_points(16):
#             ln, = ax.plot(bx, by, 'k.', markersize=4, alpha=0.5)
#             self._vis_dynamic_artists.append(ln)

#         # --- Patch center and velocity arrow ---
#         ln, = ax.plot(self.patch.x, self.patch.y, 'b+', markersize=15, markeredgewidth=2)
#         self._vis_dynamic_artists.append(ln)
#         vx, vy = self.patch.get_velocity_vector()
#         arr = ax.arrow(self.patch.x, self.patch.y, vx * 0.3, vy * 0.3,
#                        head_width=0.2, head_length=0.15, fc='blue', ec='blue', linewidth=2)
#         self._vis_dynamic_artists.append(arr)

#         # --- Agents ---
#         colors = ['red', 'orange']
#         collisions = self.current_base_obs.get("collisions", [0.0, 0.0]) if self.current_base_obs is not None else [0.0, 0.0]
#         for i, (x, y, theta, v) in enumerate(agent_positions):
#             inside = self.patch.is_inside(x, y)
#             collision = float(collisions[i]) > 0.5
#             color = colors[i % len(colors)]
#             marker, size = ('X', 18) if collision else ('o', 12) if inside else ('s', 14)
#             ln, = ax.plot(x, y, marker, color=color, markersize=size,
#                           markeredgecolor='black', markeredgewidth=2)
#             self._vis_dynamic_artists.append(ln)
#             arr = ax.arrow(x, y, v * np.cos(theta) * 0.2, v * np.sin(theta) * 0.2,
#                            head_width=0.1, head_length=0.05, fc=color, ec=color, alpha=0.7)
#             self._vis_dynamic_artists.append(arr)
#             txt = ax.text(x + 0.3, y + 0.3, f'R{i}', fontsize=8, fontweight='bold')
#             self._vis_dynamic_artists.append(txt)

#         # --- Info box ---
#         txt = ax.text(0.02, 0.98, f'Reward: {self.episode_reward:.1f}\n',
#                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
#                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
#         self._vis_dynamic_artists.append(txt)

#         # --- View limits ---
#         view_margin = max(self.patch.a, self.patch.b) + 5
#         ax.set_xlim(cx - view_margin, cx + view_margin)
#         ax.set_ylim(cy - view_margin, cy + view_margin)

#         try:
#             ax.figure.canvas.draw()
#             ax.figure.canvas.flush_events()
#         except Exception:
#             pass

#     def close(self):
#         if hasattr(self, '_fig') and self._fig is not None:
#             import matplotlib.pyplot as plt
#             plt.close(self._fig)
#             self._fig = None
#             self._ax = None
#         self.f110.close()


# class AgentView(gym.Env):
#     """Single-agent view into a shared AgentEnv for true parameter sharing.

#     Two AgentView instances (agent_idx=0 and agent_idx=1) wrap ONE AgentEnv.
#     Each independently predicts [steer_delta, speed_delta] for its own physical
#     agent. Physics executes once BOTH actions are submitted, so the two views
#     must be stepped in order (agent_idx=0 before agent_idx=1 within the same
#     DummyVecEnv call).

#     Training setup in training.py:
#         envs = []
#         for _ in range(N_AGENT_ENVS):
#             shared = AgentEnv(cfg)
#             envs.append(lambda env=shared: AgentView(env, 0))
#             envs.append(lambda env=shared: AgentView(env, 1))
#         vec_env = DummyVecEnv(envs)

#     This gives 2*N_AGENT_ENVS SB3 slots that all share the SAME network weights
#     (parameter sharing) while each agent independently controls its own vehicle.
#     """

#     metadata = {"render_modes": ["human", "rgb_array", None]}

#     def __init__(self, shared_env: "AgentEnv", agent_idx: int):
#         super().__init__()
#         assert agent_idx in (0, 1), "agent_idx must be 0 or 1"
#         self.env = shared_env
#         self.agent_idx = agent_idx

#         # Identical spaces to AgentEnv (CTDE: 24D obs, 2D action)
#         self.action_space = spaces.Box(
#             low=np.array([-0.4189, -1.0], dtype=np.float32),
#             high=np.array([0.4189,  1.0], dtype=np.float32),
#             dtype=np.float32,
#         )
#         # 24D = ego_12D + partner_12D — actor only uses [:12], critic uses [:24]
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32,
#         )

#     # ------------------------------------------------------------------
#     # reset
#     # ------------------------------------------------------------------
#     def reset(self, seed=None, options=None):
#         if self.agent_idx == 0:
#             # Agent 0 owns the reset — runs physics reset and seeds _step_obs
#             self.env.reset(seed=seed, options=options)
#             # _step_obs[0] and [1] are already set by AgentEnv.reset()
#         # Both views just return their own cached obs
#         obs = self.env._step_obs[self.agent_idx]
#         if obs is None:
#             # Fallback: env hasn't been reset yet by agent_idx=0 — trigger it now
#             self.env.reset(seed=seed, options=options)
#             obs = self.env._step_obs[self.agent_idx]
#         return obs.copy(), {}

#     # ------------------------------------------------------------------
#     # step
#     # ------------------------------------------------------------------
#     def step(self, action):
#         action = np.asarray(action, dtype=np.float32)
#         self.env._pending_actions[self.agent_idx] = action

#         if all(a is not None for a in self.env._pending_actions):
#             # Both actions ready — execute one physics step
#             self.env._execute_combined_step(
#                 self.env._pending_actions[0],
#                 self.env._pending_actions[1],
#             )
#             self.env._pending_actions = [None, None]

#         obs        = self.env._step_obs[self.agent_idx].copy()
#         reward     = self.env._step_rewards[self.agent_idx]
#         terminated = self.env._step_terminated
#         truncated  = self.env._step_truncated
#         info       = dict(self.env._step_info)
#         info["agent_idx"] = self.agent_idx
#         return obs, reward, terminated, truncated, info

#     # ------------------------------------------------------------------
#     # passthrough
#     # ------------------------------------------------------------------
#     def render(self):
#         return self.env.render()

#     def close(self):
#         if self.agent_idx == 0:
#             self.env.close()


# ===========================================================================
# Joint Training: patch + agents trained simultaneously
# ===========================================================================

class JointEnv:
    """Shared physics backend for joint patch car + single agent training.

    Not a gym.Env itself — wrapped by JointPatchView and AgentEnv (pz_env).

    Slot layout in _pending_actions / _step_obs / _step_rewards:
        slot 0 = patch
        slot 1 = agent 0

    f110 layout: car 0 = learning agent, car 1 = patch car (real f110 physics).
    DynamicPatch is used only for ellipse geometry tracking.

    Coordination between training phases:
        _patch_action_fn(patch_obs) → ndarray[4]
            Returns patch [steer, speed, a_cmd, b_cmd].
            If None: default action is used.
        _agent_action_fn(obs0) → ndarray[2]
            Used by JointPatchView to auto-fill agent slot.
            If None: zeros used for the agent.
    """

    # f110 layout: car 0 = learning agent; car 1 = patch car (ellipse center).
    PATCH_CAR_F110_IDX = 1
    AGENT_F110_IDX = (0,)

    PATCH_OBS_DIM = 273  # 17 scalars (7 state + 10 curvature lookahead) + 256 occupancy grid (16x16)
    AGENT_OBS_DIM = 10   # ex/a, ey/b, dn, agent_speed, heading_rel, patch_v, patch_steer, dist_patch_car, a, b

    def __init__(self, cfg: JointEnvConfig):
        self.cfg = cfg

        self.f110 = F110EnvAdapter(
            F110Config(
                num_agents=2,
                control_input=("speed", "steering_angle"),
                timestep=cfg.control_dt,
            ),
            render_mode=cfg.render_mode,
        )
        self.map_reset = MapResetHelper()
        self.patch = DynamicPatch()
        self.patch.a = cfg.patch_a
        self.patch.b = cfg.patch_b

        self.track_spline = None
        self.track_length = None
        self._occ_map = None
        self._resolution = None
        self._origin = None
        self._tw_s_vals = None
        self._tw_half_w = None

        self.step_count = 0
        self.episode_reward_agents = [0.0]
        self.patch_yaw_rate = 0.0
        # === PATCH TRAINING STATE — disabled, patch driven by frozen policy ===
        # self.episode_reward_patch = 0.0
        # self.prev_patch_s = 0.0
        # self.patch_lap_s = 0.0
        # self.patch_lap_count = 0
        # self.patch_prev_steer = 0.0
        # self.patch_prev_s = None
        # self.patch_no_progress = 0
        # =====================================================================
        self.current_base_obs = None
        self.prev_dist_norms = [0.0]
        self._episode_count = 0
        self._term_counts = {
            "out_of_patch": 0, "inter_collision": 0,
            "max_steps": 0, "patch_wall": 0,
        }

        self._pending_actions: list = [None, None]
        self._step_obs: list = [None, None]
        self._step_rewards: list = [0.0, 0.0]
        self._step_terminated: bool = False
        self._step_truncated: bool = False
        self._step_info: dict = {}

        # Frozen policy functions — set by training loop between phases
        self._patch_action_fn = None   # Callable(patch_obs) → ndarray[4]
        self._agent_action_fn = None   # Callable(obs0) → ndarray[2]
        # Set to True by pz_env when real agent actions are being passed directly
        # (vs Phase 1 where _agent_action_fn is None and dummy [0,0] actions are used)
        self._real_agents_active: bool = False

    # ------------------------------------------------------------------
    # Track width helpers (same logic as PatchEnv)
    # ------------------------------------------------------------------

    def _precompute_track_widths(self, n_samples: int = 200) -> None:
        if self._occ_map is None or self.track_spline is None:
            return
        s_vals = np.linspace(0.0, float(self.track_length), n_samples, endpoint=False)
        half_w = np.full(n_samples, 3.0, dtype=np.float32)
        for i, s in enumerate(s_vals):
            try:
                x, y = self.track_spline.calc_position(float(s))
                yaw  = self.track_spline.calc_yaw(float(s))
                w = float(self.map_reset.estimate_track_width(
                    self._occ_map, self._resolution, self._origin,
                    float(x), float(y), float(yaw),
                ))
                if np.isfinite(w) and w > 0.0:
                    half_w[i] = w / 2.0
            except Exception:
                pass
        self._tw_s_vals = s_vals.astype(np.float32)
        self._tw_half_w = half_w

    def _lookup_half_w(self, s: float) -> float:
        if self._tw_s_vals is None:
            return 1.5
        s_wrapped = float(s) % float(self.track_length)
        idx = int(np.searchsorted(self._tw_s_vals, s_wrapped))
        idx = min(idx, len(self._tw_half_w) - 1)
        return float(self._tw_half_w[idx])

    # === PATCH TRAINING HELPERS — disabled, patch driven by frozen policy ===
    # def _wrap_ds(self, ds: float) -> float:
    #     """Wrap distance increment to handle lap wraparound."""
    #     if self.track_length is None or not np.isfinite(self.track_length):
    #         return ds
    #     L = float(self.track_length)
    #     if L <= 0:
    #         return ds
    #     return (ds + 0.5 * L) % L - 0.5 * L

    # def _compute_max_steering_angle(self, b_cmd, ey, v, half_w):
    #     """Steering constraint by patch boundary — removed (patch is virtual/attached)."""
    #     pass
    # =========================================================================

    # ------------------------------------------------------------------
    # Obs builders
    # ------------------------------------------------------------------

    def _patch_to_frenet(self, patch=None) -> tuple:
        p = patch if patch is not None else self.patch
        if self.track is None or not hasattr(self.track, 'cartesian_to_frenet'):
            return 0.0, 0.0
        try:
            s, ey, _ = self.track.cartesian_to_frenet(
                float(p.x), float(p.y), float(p.theta), s_guess=0)
            return (float(s), float(ey)) if (np.isfinite(s) and np.isfinite(ey)) else (0.0, 0.0)
        except Exception:
            return 0.0, 0.0

    def _build_patch_obs(self, base_obs) -> np.ndarray:
        """263D patch obs: 7 scalars + 256-value occupancy grid (16x16).
        EXACT match to PatchCarEnv._build_obs_for() for Phase 0 transfer learning.
        Scalars: [s_norm, ey_signed, speed, psi_error, a, b, curvature]
        """
        # === 7D Scalar Features (MUST match PatchCarEnv) ===
        s, ey_signed = self._patch_to_frenet()
        s_norm = s / max(float(self.track_length), 1.0)

        # Track-relative heading (like PatchCarEnv does)
        yaw = float(self.patch.theta)
        try:
            track_yaw = float(self.track_spline.calc_yaw(float(s)))
        except Exception:
            track_yaw = yaw
        psi_error = float((yaw - track_yaw + np.pi) % (2.0 * np.pi) - np.pi)

        speed = float(self.patch.v)
        a = float(self.patch.a)
        b = float(self.patch.b)

        # COMMENTED OUT: Lookahead distances — not needed, CNN learns from occupancy grid
        # L = float(self.track_length) if self.track_length else 1.0
        # half_w_current = max(self._lookup_half_w(s), 1e-3)
        # half_w_5m = max(self._lookup_half_w((s + 5.0) % L), 1e-3)
        # half_w_10m = max(self._lookup_half_w((s + 10.0) % L), 1e-3)
        # half_w_15m = max(self._lookup_half_w((s + 15.0) % L), 1e-3)

        try:
            curvature = float(self.track_spline.calc_curvature(float(s)))
        except Exception:
            curvature = 0.0

        scalars = np.array(
            [
                s_norm,
                ey_signed,
                speed,
                psi_error,
                a,
                b,
                curvature,
                # COMMENTED OUT: Lookahead scalars (removed for PatchCarEnv parity)
                # half_w_current,
                # half_w_5m,
                # half_w_10m,
                # half_w_15m,
            ],
            dtype=np.float32,
        )

        # === 256D Occupancy Grid ===
        grid = self._build_occupancy_grid_16x16()

        return np.concatenate([scalars, grid.flatten()]).astype(np.float32)

    def _build_occupancy_grid_16x16(self) -> np.ndarray:
        """Build 16x16 local occupancy grid — EXACT same logic as PatchCarEnv._get_local_grid().

        8m × 8m region centered on patch in MAP FRAME (no rotation).
        scipy zoom to 16×16.  Returns flat (256,) array, 0.0=wall, 1.0=free.

        Must match PatchCarEnv exactly so the frozen patch policy sees the same
        visual representation it was trained on.
        """
        if self._occ_map is None:
            return np.ones(256, dtype=np.float32)

        occ = self._occ_map
        res = float(self._resolution)
        ox, oy = float(self._origin[0]), float(self._origin[1])

        px = (float(self.patch.x) - ox) / res
        py = (float(self.patch.y) - oy) / res

        half_size_pix = int(round(4.0 / res))  # 4m in each direction → 8m × 8m

        H, W = occ.shape
        r0 = max(0, int(round(py)) - half_size_pix)
        r1 = min(H, int(round(py)) + half_size_pix)
        c0 = max(0, int(round(px)) - half_size_pix)
        c1 = min(W, int(round(px)) + half_size_pix)
        crop = occ[r0:r1, c0:c1]

        if crop.shape[0] < 1 or crop.shape[1] < 1:
            return np.ones(256, dtype=np.float32)

        grid = ndimage.zoom(crop, (16.0 / crop.shape[0], 16.0 / crop.shape[1]), order=1)
        grid = grid[:16, :16]
        if grid.shape != (16, 16):
            grid = np.ones((16, 16), dtype=np.float32)

        return np.clip(grid, 0.0, 1.0).astype(np.float32).flatten()

    def _build_agent_obs_for(self, ego: int, base_obs) -> np.ndarray:
        """10D ego-centric observation.

        [0] ex / a         — forward offset from patch center, normalised by major axis
        [1] ey_e / b       — lateral offset from patch center, normalised by minor axis
        [2] dn             — normalised ellipse distance (1.0 = boundary, <1.0 inside)
        [3] agent_speed    — agent longitudinal speed (m/s)
        [4] heading_rel    — agent yaw minus patch yaw (rad), wrapped to [-π, π]
        [5] patch_v        — patch car speed (m/s)
        [6] patch_steer    — patch car steering angle (rad)
        [7] dist_patch_car — Euclidean distance agent-to-patch-car (m)
        [8] a              — patch major axis (forward half-length, m)
        [9] b              — patch minor axis (lateral half-width, m)

        Removed / commented-out features (dead signals):
        # pdx, pdy:       patch car in patch frame — ALWAYS ≈ (0,0) since patch IS patch car
        # ox, oy_o:       other-agent position     — ALWAYS 0.0, no second learning agent
        # inter_agent:    other-agent distance      — ALWAYS 999.0, no second agent
        # n_inside:       agents-inside count       — redundant with dn < 1.0
        # patch_yaw_rate: delta_theta/dt            — replaced by patch.steering (direct command)
        # split5[0-4]:    split zone context        — ALWAYS 0.0, split mode disabled
        """
        fe = self.AGENT_F110_IDX[ego]
        fp = self.PATCH_CAR_F110_IDX
        a = max(float(self.patch.a), 1e-3)
        b = max(float(self.patch.b), 1e-3)

        ex, ey_e = self.patch.world_to_patch_frame(
            float(base_obs["poses_x"][fe]), float(base_obs["poses_y"][fe])
        )
        dn = float(np.sqrt((ex / a) ** 2 + (ey_e / b) ** 2))

        vx = float(base_obs["linear_vels_x"][fe]) if "linear_vels_x" in base_obs else 0.0
        vy = float(base_obs["linear_vels_y"][fe]) if "linear_vels_y" in base_obs else 0.0
        agent_speed = float(np.hypot(vx, vy))

        yaw = float(base_obs["poses_theta"][fe]) if "poses_theta" in base_obs else 0.0
        heading_rel = float((yaw - self.patch.theta + np.pi) % (2 * np.pi) - np.pi)

        px_w = float(base_obs["poses_x"][fp])
        py_w = float(base_obs["poses_y"][fp])
        dist_patch_car = float(np.hypot(
            float(base_obs["poses_x"][fe]) - px_w,
            float(base_obs["poses_y"][fe]) - py_w,
        ))

        return np.array(
            [
                ex / a,
                ey_e / b,
                dn,
                agent_speed,
                heading_rel,
                float(self.patch.v),
                float(np.clip(self.patch.steering, -0.4189, 0.4189)),
                dist_patch_car,
                a,
                b,
            ],
            dtype=np.float32,
        )

    def _build_all_obs(self, base_obs) -> None:
        patch_obs = np.nan_to_num(
            self._build_patch_obs(base_obs), nan=0.0, posinf=10.0, neginf=-10.0
        ).astype(np.float32)
        ego0 = np.nan_to_num(
            self._build_agent_obs_for(0, base_obs), nan=0.0, posinf=10.0, neginf=-10.0
        ).astype(np.float32)
        self._step_obs[0] = patch_obs
        self._step_obs[1] = ego0

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _compute_agent_reward(self, dist_norm: float,
                              patch_car_collision: bool,
                              patch_car_dist: float = 999.0) -> float:
        """Flat inside / warning ramp / outside penalty + patch-car repulsion.

        Zones (by normalised distance to ellipse center):
            dn <= warning_zone_start (0.85): flat +inside_reward  — anywhere inside is fine
            warning_zone_start < dn <= 1.0:  linear ramp from +inside_reward → 0
            dn > 1.0:                        steep penalty proportional to overshoot
        """
        wz = self.cfg.warning_zone_start  # 0.85

        if dist_norm <= wz:
            # Comfortable interior — flat reward, no center bias
            reward = self.cfg.inside_reward
        elif dist_norm <= 1.0:
            # Warning zone — linear ramp from full reward to zero at boundary
            t = (dist_norm - wz) / (1.0 - wz)  # 0 at wz, 1 at boundary
            reward = self.cfg.inside_reward * (1.0 - t)
        else:
            # Outside — steep penalty, capped so gradient doesn't explode
            reward = -self.cfg.out_of_patch_penalty * min(dist_norm - 1.0, 1.0)

        # --- DISABLED: agent-vs-patch-car repulsion shaping ---
        # The agent isn't being trained to go to the center, so a smooth
        # pre-collision repulsion ramp isn't needed — the terminal collision
        # penalty alone provides the avoidance signal.
        # NOTE: this block also had a config-level dead-code bug:
        # repulsion_zone (0.15) < patch_car_collision_dist (0.18), so the
        # guard `dist < 0.15 AND not patch_car_collision (i.e. dist >= 0.18)`
        # was unsatisfiable and the penalty never fired anyway.
        # if patch_car_dist < self.cfg.repulsion_zone and not patch_car_collision:
        #     t = (self.cfg.repulsion_zone - patch_car_dist) / max(
        #         self.cfg.repulsion_zone - self.cfg.patch_car_collision_dist, 1e-6)
        #     reward -= self.cfg.repulsion_weight * float(np.clip(t, 0.0, 1.0))

        # Hard penalty on actual collision with patch car
        if patch_car_collision:
            reward -= self.cfg.inter_collision_penalty

        # return float(np.nan_to_num(np.clip(reward, -100.0, 100.0), nan=0.0))
        return float(np.nan_to_num(np.clip(reward, -2000.0, 2000.0), nan=0.0))

    def _check_agent_termination(self, agent_idx: int, dist_norm: float,
                                 inter_collision: bool) -> tuple:
        if inter_collision:
            return True, False, f"agent{agent_idx}_inter_collision"
        if dist_norm > self.cfg.agent_out_of_patch_threshold:
            return True, False, f"agent{agent_idx}_out_of_patch"
        if self.step_count >= self.cfg.max_steps:
            return False, True, "max_steps"
        return False, False, None

    # === JOINT TRAINING DISABLED — patch is driven by frozen policy, no reward needed ===
    # def _compute_patch_reward_for(
    #     self,
    #     patch: "DynamicPatch",
    #     lane_id: int,
    #     prev_s: float,
    #     prev_steer: float,
    #     no_progress: int,
    #     dt: float,
    #     lap_bonus: float = 0.0,
    #     cached_collision: bool = False,
    # ) -> tuple:
    #     """
    #     Compute patch reward using PatchEnv._compute_reward_for logic (reused from PatchCarEnv Phase 0).
    #     Returns (reward_clipped, new_prev_s, new_prev_steer, new_no_progress, terms_dict).
    #
    #     This eliminates duplication: both Phase 0 (PatchCarEnv) and Phase A (JointEnv) now use
    #     identical patch reward logic.
    #     """
    #     s, ey_track = self._patch_to_frenet()
    #     ds = self._wrap_ds(s - prev_s) if prev_s is not None else 0.0
    #
    #     # Lookup track half-width at current position (O(1) using precomputed array)
    #     half_w = max(self._lookup_half_w(s), 1e-3)
    #     if lane_id == 1:
    #         ey = ey_track - (half_w / 2.0)
    #     elif lane_id == 2:
    #         ey = ey_track + (half_w / 2.0)
    #     else:
    #         ey = ey_track
    #
    #     steer_cmd = float(getattr(patch, "steering", 0.0))
    #     steer_rate = abs(steer_cmd - prev_steer) / max(dt, 1e-6)
    #     yaw_rate = float((patch.v / max(self.cfg.wheelbase, 1e-6)) * np.tan(steer_cmd))
    #     spin_excess = max(0.0, abs(yaw_rate) - self.cfg.spin_yawrate_threshold)
    #
    #     # Use the pre-computed collision (occupancy map boundary check only)
    #     collision = cached_collision
    #
    #     # Stuck counter
    #     if abs(ds) < self.cfg.stuck_progress_eps:
    #         no_progress += 1
    #     else:
    #         no_progress = 0
    #
    #     # Lane-centering penalty in split mode
    #     # lane_center_penalty = 0.0
    #     # if lane_id != 0:
    #     #     lane_center_penalty = self.cfg.lane_centering_weight * abs(ey)
    #
    #     a_now = float(max(patch.a, 1e-3))
    #     b_now = float(max(patch.b, 1e-3))
    #     base_area = float(self.cfg.patch_a * self.cfg.patch_b)
    #     current_area = a_now * b_now
    #     area_ratio = current_area / max(base_area, 1e-3)
    #     area_excess = max(0.0, area_ratio - 1.0)
    #
    #     reward_raw = (
    #         self.cfg.reward_progress_scale * ds
    #         - self.cfg.reward_crosstrack_weight * abs(ey)
    #         # - lane_center_penalty
    #         - self.cfg.reward_steer_bias_weight * abs(steer_cmd)
    #         - self.cfg.reward_steer_rate_weight * steer_rate
    #         - self.cfg.reward_spin_weight * spin_excess
    #         - (self.cfg.patch_wall_penalty if collision else 0.0)
    #         - self.cfg.shape_area_penalty_weight * area_excess
    #         - self.cfg.time_penalty_per_sec * float(dt)
    #         + float(lap_bonus)
    #     )
    #     if no_progress >= self.cfg.stuck_no_progress_steps:
    #         reward_raw -= self.cfg.stuck_penalty
    #
    #     reward_clipped = float(np.clip(reward_raw, -100.0, 100.0))
    #
    #     terms = {
    #         "s": float(s), "ey": float(ey), "ds": float(ds),
    #         "collision": bool(collision), "no_progress": int(no_progress),
    #         "reward_raw": float(reward_raw), "reward_clipped": float(reward_clipped),
    #         "yaw_rate": float(yaw_rate),
    #         "spin_excess": float(spin_excess),
    #     }
    #     return reward_clipped, float(s), steer_cmd, no_progress, terms

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self.episode_reward_agents = [0.0]
        self.prev_dist_norms = [0.0]
        self._pending_actions = [None, None]
        # === PATCH TRAINING STATE — disabled, patch driven by frozen policy ===
        # self.episode_reward_patch = 0.0
        # self.patch_lap_s = 0.0
        # self.patch_lap_count = 0
        # self.patch_prev_steer = 0.0
        # self.patch_prev_s = None
        # self.patch_no_progress = 0
        # =====================================================================

        self.f110.ensure_initialized()
        track, occ_map, self._resolution, self._origin = self.f110.get_track_data()
        self._occ_map = occ_map / 255.0 if occ_map is not None else None  # Normalize to [0, 1]
        if track.centerline is None or track.centerline.spline is None:
            raise ValueError("Track centerline/spline missing.")
        self.track = track
        self.track_spline = track.centerline.spline
        self.track_length = float(self.track_spline.s[-1])

        if self._tw_s_vals is None:
            self._precompute_track_widths(n_samples=200)

        xs = np.asarray(track.centerline.xs, dtype=np.float32)
        ys = np.asarray(track.centerline.ys, dtype=np.float32)
        spawn_idx = int(np.random.randint(0, xs.shape[0])) if self.cfg.random_spawn else 0
        spawn_idx = max(0, min(spawn_idx, xs.shape[0] - 1))
        patch_x = float(xs[spawn_idx])
        patch_y = float(ys[spawn_idx])
        next_idx = (spawn_idx + 1) % xs.shape[0]
        patch_theta = float(np.arctan2(
            float(ys[next_idx] - ys[spawn_idx]),
            float(xs[next_idx] - xs[spawn_idx])
        ))

        self.patch = DynamicPatch(
            x=patch_x, y=patch_y, theta=patch_theta,
            v=0.5, a=self.cfg.patch_a, b=self.cfg.patch_b,
        )
        self.patch_yaw_rate = 0.0
        # self.prev_patch_s, _ = self._patch_to_frenet()  # patch training only

        perp_dx = -np.sin(patch_theta)
        perp_dy = np.cos(patch_theta)
        offset = 0.50
        # Car 0 = learning agent; car 1 = patch car (ellipse anchor)
        poses = np.array(
            [
                [patch_x + offset * perp_dx, patch_y + offset * perp_dy, patch_theta],
                [patch_x, patch_y, patch_theta],
            ],
            dtype=np.float32,
        )

        base_obs, _ = self.f110.reset(poses=poses)
        self.current_base_obs = base_obs
        pidx = self.PATCH_CAR_F110_IDX
        vx = float(base_obs["linear_vels_x"][pidx])
        vy = float(base_obs["linear_vels_y"][pidx])
        self.patch.sync_from_pose(
            float(base_obs["poses_x"][pidx]),
            float(base_obs["poses_y"][pidx]),
            float(base_obs["poses_theta"][pidx]),
            float(np.hypot(vx, vy)),
            0.0,
        )
        self._build_all_obs(base_obs)
        self._step_rewards = [0.0, 0.0]
        self._step_terminated = False
        self._step_truncated = False
        self._step_info = {}

        a = max(float(self.patch.a), 1e-3)
        b = max(float(self.patch.b), 1e-3)
        self.prev_dist_norms = []
        for i in self.AGENT_F110_IDX:
            xr, yr = self.patch.world_to_patch_frame(
                float(base_obs["poses_x"][i]), float(base_obs["poses_y"][i])
            )
            self.prev_dist_norms.append(float(np.sqrt((xr / a) ** 2 + (yr / b) ** 2)))

    # ------------------------------------------------------------------
    # Joint physics step
    # ------------------------------------------------------------------

    # === JOINT TRAINING DISABLED — patch respawn was a joint-training feature ===
    # def _respawn_patch_car(self, base_obs: dict) -> dict:
    #     """Teleport patch car (index 1) to a new centerline pose; keep agent 0 fixed."""
    #     track, _, _, _ = self.f110.get_track_data()
    #     xs = np.asarray(track.centerline.xs, dtype=np.float32)
    #     ys = np.asarray(track.centerline.ys, dtype=np.float32)
    #     spawn_idx = int(np.random.randint(0, xs.shape[0]))
    #     spawn_idx = max(0, min(spawn_idx, xs.shape[0] - 1))
    #     px = float(xs[spawn_idx])
    #     py = float(ys[spawn_idx])
    #     next_idx = (spawn_idx + 1) % xs.shape[0]
    #     pth = float(
    #         np.arctan2(
    #             float(ys[next_idx] - ys[spawn_idx]),
    #             float(xs[next_idx] - xs[spawn_idx]),
    #         )
    #     )
    #     poses = np.array(
    #         [
    #             [
    #                 float(base_obs["poses_x"][0]),
    #                 float(base_obs["poses_y"][0]),
    #                 float(base_obs["poses_theta"][0]),
    #             ],
    #             [px, py, pth],
    #         ],
    #         dtype=np.float32,
    #     )
    #     o, _ = self.f110.reset(poses=poses)
    #     return o

    def _step_with_frozen_policy(self, patch_action: np.ndarray,
                            action0: np.ndarray) -> None:
        """Execute one full physics step: f110 car 0 = agent, car 1 = patch car."""
        self.step_count += 1
        dt = self.cfg.control_dt
        pidx = self.PATCH_CAR_F110_IDX

        steer_p = float(np.clip(np.nan_to_num(patch_action[0]), -0.4189, 0.4189))
        speed_p = float(np.clip(np.nan_to_num(patch_action[1]), 1.5, 10.0))
        a_p = float(np.clip(
            np.nan_to_num(patch_action[2]),
            self.cfg.patch_a_cmd_min, self.cfg.patch_a_cmd_max,
        ))
        b_p = float(np.clip(
            np.nan_to_num(patch_action[3]),
            self.cfg.patch_b_cmd_min, self.cfg.patch_b_cmd_max,
        ))

        # === JOINT TRAINING DISABLED — agent-only with frozen patch policy ===
        # _agent_action_fn was used by JointPatchView (joint patch training) to
        # auto-fill the agent slot with a frozen agent snapshot. In agent-only
        # mode, pz_env passes the live agent action directly into action0.
        # has_learned_agents = (
        #     self._agent_action_fn is not None
        # ) or self._real_agents_active
        #
        # if self._agent_action_fn is not None:
        #     obs0 = self._step_obs[1]
        #     if obs0 is not None:
        #         action0 = self._agent_action_fn(obs0)

        s0 = float(np.clip(np.nan_to_num(action0[0]), -0.4189, 0.4189))
        v0 = float(np.clip(np.nan_to_num(action0[1]), 1.5, 10.0))
        self._last_agent_steers = [s0]

        # if not has_learned_agents:
        #     s0, v0 = 0.0, 0.5

        prev_theta = self.patch.theta
        prev_s, _ = self._patch_to_frenet()

        base_obs, _, _, _, _ = self.f110.step(
            np.array(
                [[s0, v0], [steer_p, speed_p]],
                dtype=np.float32,
            )
        )
        self.current_base_obs = base_obs

        vx = float(base_obs["linear_vels_x"][pidx])
        vy = float(base_obs["linear_vels_y"][pidx])
        self.patch.sync_from_pose(
            float(base_obs["poses_x"][pidx]),
            float(base_obs["poses_y"][pidx]),
            float(base_obs["poses_theta"][pidx]),
            float(np.hypot(vx, vy)),
            steer_p,
        )
        d_theta = (self.patch.theta - prev_theta + np.pi) % (2 * np.pi) - np.pi
        self.patch_yaw_rate = d_theta / dt

        s_mid, _ = self._patch_to_frenet()
        hw_mid = self._lookup_half_w(s_mid)
        max_b_dyn = min(self.cfg.patch_b, hw_mid * 0.90)
        self.patch.update_shape(
            a_p, b_p, dt, max_a=self.cfg.patch_a, max_b=max_b_dyn
        )

        # === JOINT TRAINING DISABLED — patch lap tracking unused (no patch reward) ===
        # curr_s, _ = self._patch_to_frenet()
        # L = float(self.track_length)
        # ds = float(np.nan_to_num(((curr_s - prev_s + L / 2.0) % L) - L / 2.0, nan=0.0))
        #
        # self.patch_lap_s += max(0.0, ds)
        # lap_bonus = 0.0
        # if self.patch_lap_s >= L:
        #     self.patch_lap_count += 1
        #     self.patch_lap_s -= L
        #     lap_bonus = self.cfg.patch_lap_bonus

        # --- BUGFIX: compute agent<->patch-car contact BEFORE the f110 wall
        # check, so we can disambiguate f110's collision flag (which fires for
        # ANY car-vs-car overlap, including agent-vs-patch-car) from a true
        # patch-vs-wall collision. Old block kept below for reference.
        px0 = float(base_obs["poses_x"][0])
        py0 = float(base_obs["poses_y"][0])
        pxc = float(base_obs["poses_x"][pidx])
        pyc = float(base_obs["poses_y"][pidx])
        # Agent-to-patch-car distance for repulsion and collision
        d0p = float(np.hypot(px0 - pxc, py0 - pyc))
        hit_patch_car = d0p < self.cfg.patch_car_collision_dist

        patch_wall_hit = False
        if self._occ_map is not None:
            patch_wall_hit, _ = self.patch.check_patch_boundary_wall_collision(
                self._occ_map, self._resolution, self._origin,
                n_points=32,
                violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )
        if "collisions" in base_obs and not hit_patch_car:
            # Only trust f110's per-body collision flag when the agent isn't
            # currently in contact with the patch car — otherwise inter-car
            # collisions get misreported as patch-vs-wall.
            c = np.asarray(base_obs["collisions"]).reshape(-1)
            if c.shape[0] > pidx and bool(c[pidx] > 0.5):
                patch_wall_hit = True

        dist_norm_list: list[float] = []
        for fi in self.AGENT_F110_IDX:
            px = float(np.nan_to_num(base_obs["poses_x"][fi], nan=self.patch.x))
            py = float(np.nan_to_num(base_obs["poses_y"][fi], nan=self.patch.y))
            xr, yr = self.patch.world_to_patch_frame(px, py)
            dn = float(
                np.nan_to_num(
                    np.sqrt(
                        (xr / max(self.patch.a, 1e-3)) ** 2
                        + (yr / max(self.patch.b, 1e-3)) ** 2
                    ),
                    nan=2.0,
                )
            )
            dist_norm_list.append(dn)

        # --- OLD (pre-bugfix) ordering, kept for reference ---
        # patch_wall_hit = False
        # if self._occ_map is not None:
        #     patch_wall_hit, _ = self.patch.check_patch_boundary_wall_collision(
        #         self._occ_map, self._resolution, self._origin,
        #         n_points=32,
        #         violation_threshold=self.cfg.patch_boundary_violation_threshold,
        #     )
        # if "collisions" in base_obs:
        #     c = np.asarray(base_obs["collisions"]).reshape(-1)
        #     if c.shape[0] > pidx and bool(c[pidx] > 0.5):
        #         patch_wall_hit = True
        #
        # px0 = float(base_obs["poses_x"][0])
        # py0 = float(base_obs["poses_y"][0])
        # pxc = float(base_obs["poses_x"][pidx])
        # pyc = float(base_obs["poses_y"][pidx])
        # d0p = float(np.hypot(px0 - pxc, py0 - pyc))
        # hit_patch_car = d0p < self.cfg.patch_car_collision_dist

        agents_inside = sum(1.0 for dn in dist_norm_list if dn <= 1.0)

        # === JOINT TRAINING DISABLED — patch reward computation removed ===
        # The patch is driven by a frozen policy; no reward signal is needed.
        # _, ey = self._patch_to_frenet()
        # steer_cmd = float(getattr(self.patch, "steering", 0.0))
        # steer_rate = abs(steer_cmd - self.patch_prev_steer) / max(dt, 1e-6)
        # yaw_rate = float(
        #     (self.patch.v / max(self.cfg.wheelbase, 1e-6)) * np.tan(steer_cmd)
        # )
        # spin_excess = max(0.0, abs(yaw_rate) - self.cfg.spin_yawrate_threshold)
        #
        # patch_reward, patch_s_now, _, self.patch_no_progress, _ = self._compute_patch_reward_for(
        #     self.patch, lane_id=0,  # patch is always in primary lane
        #     prev_s=self.patch_prev_s, prev_steer=self.patch_prev_steer,
        #     no_progress=self.patch_no_progress, dt=dt,
        #     lap_bonus=lap_bonus, cached_collision=patch_wall_hit,
        # )
        # self.patch_prev_s = patch_s_now  # Update for next step
        patch_reward = 0.0

        # DISABLED: No respawning during training
        # respawned_patch = False
        respawned_patch = False

        # Single agent reward/termination
        ic = hit_patch_car
        r = self._compute_agent_reward(
            dist_norm_list[0],
            patch_car_collision=ic,
            patch_car_dist=d0p,
        )
        r = float(np.nan_to_num(r, nan=0.0))
        term, trunc, reason = self._check_agent_termination(
            0, dist_norm_list[0], ic
        )
        self._step_rewards[1] = float(np.clip(r, -100.0, 100.0))
        self.episode_reward_agents[0] += r
        agent_terminated_flag = term or trunc
        self.prev_dist_norms = list(dist_norm_list)

        # === JOINT TRAINING DISABLED — no patch reward accumulation ===
        # self._step_rewards[0] = patch_reward
        # self.episode_reward_patch += patch_reward
        self._step_rewards[0] = 0.0

        # Termination logic:
        # - Agent inter-collision/out-of-patch terminates (always — JointEnv is
        #   only used when agents exist, so the _real_agents_active gate is
        #   redundant and was hiding out_of_patch terminations in agent-only
        #   training paths that never flipped the flag).
        # - Patch wall hit also terminates (agent should protect patch), but
        #   inter-collision wins the priority check so the reason is attributed
        #   correctly to the agent.
        # - max_steps truncates (doesn't terminate)
        agent_terminated = agent_terminated_flag
        patch_terminates = patch_wall_hit  # Always terminate on patch wall hit
        terminated = agent_terminated or patch_terminates
        truncated = self.step_count >= self.cfg.max_steps

        # Priority: agent inter/out-of-patch first, then patch_wall, then truncation.
        # This ensures hitting the patch car is reported as inter_collision, not
        # patch_wall (f110 flips the patch's collision bit on any car-vs-car
        # contact, which previously caused misattribution).
        if agent_terminated:
            pass  # reason already set by _check_agent_termination
        elif patch_terminates:
            reason = "patch_wall"
        elif truncated:
            reason = "max_steps"
        else:
            reason = None

        # --- OLD (pre-bugfix) termination logic, kept for reference ---
        # agent_terminated = self._real_agents_active and agent_terminated_flag
        # patch_terminates = patch_wall_hit  # Always terminate on patch wall hit
        # terminated = agent_terminated or patch_terminates
        # truncated = self.step_count >= self.cfg.max_steps
        #
        # if patch_terminates:
        #     reason = "patch_wall"
        # elif agent_terminated:
        #     pass  # reason already set by _check_agent_termination
        # elif truncated:
        #     reason = "max_steps"
        # else:
        #     reason = None

        self._build_all_obs(base_obs)

        if terminated or truncated:
            self._episode_count += 1
            key = (
                "inter_collision"
                if reason and "inter_collision" in (reason or "")
                else "out_of_patch"
                if reason and "out_of_patch" in (reason or "")
                else "patch_wall"
                if reason == "patch_wall"
                else "max_steps"
            )
            self._term_counts[key] += 1
            if self._episode_count % 200 == 0:
                n = self._episode_count
                print(
                    f"[JointEnv ep={n}] "
                    f"out_of_patch={self._term_counts['out_of_patch']/n*100:.1f}%  "
                    f"patch_wall={self._term_counts['patch_wall']/n*100:.1f}%  "
                    f"max_steps={self._term_counts['max_steps']/n*100:.1f}%  "
                    f"patch_v={self.patch.v:.2f}  d0p={d0p:.3f}"
                )

        self._step_terminated = terminated
        self._step_truncated = truncated
        agent_reward = float(self.episode_reward_agents[0])
        self._step_info = {
            "episode_reward": agent_reward,
            "Episode_steps": self.step_count,
            "termination_reason": reason,
            "episode_reward_agents": agent_reward,
            "agent_patch_dist": float(d0p),
            "agents_inside": int(agents_inside),
            # "episode_reward_patch": 0.0,  # patch not trained
            # "patch_reward": 0.0,          # patch not trained
            # "patch_respawned": False,      # patch not trained
        }

    def _visualize(self):
        """Live matplotlib overlay: occupancy map + patch ellipse + agent markers.
        Same style as AgentEnv._visualize — call each step for real-time display.
        """
        if not hasattr(self, '_fig') or self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(8, 6))
            self._ax.set_aspect('equal')
            self._ax.grid(True, alpha=0.3)
            self._vis_bg_image = None
            self._vis_bg_bounds = None
            self._vis_dynamic_artists = []
            self._vis_occ_cache = None
            plt.show(block=False)

        ax = self._ax

        # Remove dynamic artists from previous frame
        for artist in self._vis_dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._vis_dynamic_artists = []

        # Occupancy map — fetched once and cached
        if self._vis_occ_cache is None:
            try:
                _, occ_map, resolution, origin = self.f110.get_track_data()
                occ_map = occ_map / 255.0 if occ_map is not None else None  # Normalize to [0, 1]
                self._vis_occ_cache = (occ_map, resolution, origin)
            except Exception as e:
                print(f"Warning: could not get track data: {e}")
        occ_map = resolution = origin = None
        if self._vis_occ_cache is not None:
            occ_map, resolution, origin = self._vis_occ_cache

        # Background: redraw only when view window shifts
        cx, cy = self.patch.x, self.patch.y
        margin = max(self.patch.a, self.patch.b) + 10
        if occ_map is not None:
            px_min = max(0, int((cx - margin - origin[0]) / resolution))
            px_max = min(occ_map.shape[1], int((cx + margin - origin[0]) / resolution))
            py_min = max(0, int((cy - margin - origin[1]) / resolution))
            py_max = min(occ_map.shape[0], int((cy + margin - origin[1]) / resolution))
            new_bounds = (px_min, px_max, py_min, py_max)
            if self._vis_bg_bounds != new_bounds and px_max > px_min and py_max > py_min:
                if self._vis_bg_image is not None:
                    try:
                        self._vis_bg_image.remove()
                    except Exception:
                        pass
                region = occ_map[py_min:py_max, px_min:px_max]
                xw0 = px_min * resolution + origin[0]
                xw1 = px_max * resolution + origin[0]
                yw0 = py_min * resolution + origin[1]
                yw1 = py_max * resolution + origin[1]
                rgba = np.zeros((*region.shape, 4), dtype=np.uint8)
                rgba[region < 0.5]  = [51,  51,  51,  204]   # walls
                rgba[region >= 0.5] = [220, 220, 220, 60]    # free space
                self._vis_bg_image = ax.imshow(
                    rgba, extent=[xw0, xw1, yw0, yw1],
                    origin='lower', aspect='auto', zorder=0, interpolation='nearest')
                self._vis_bg_bounds = new_bounds

        # Car 0 = agent; car 1 = patch car
        agent_positions = []
        if self.current_base_obs is not None:
            bobs = self.current_base_obs
            n_c = len(bobs["poses_x"])
            for i in range(n_c):
                x = float(bobs["poses_x"][i])
                y = float(bobs["poses_y"][i])
                theta = float(bobs["poses_theta"][i])
                vx = float(bobs["linear_vels_x"][i])
                vy = float(bobs["linear_vels_y"][i])
                v = float(np.hypot(vx, vy))
                agent_positions.append((x, y, theta, v))
        agents_inside = sum(
            1
            for i, (x, y, _, _) in enumerate(agent_positions)
            if i < 1 and self.patch.is_inside(x, y)
        )

        # Wall collision check
        patch_wall_collision = False
        if occ_map is not None:
            patch_wall_collision, _ = self.patch.check_patch_boundary_wall_collision(
                occ_map,
                resolution,
                origin,
                n_points=32,
                violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )

        ax.set_title(
            f'Joint Funnel | Step {self.step_count}\n'
            f'Patch: v={self.patch.v:.1f}m/s  size=({self.patch.a:.1f},{self.patch.b:.1f}) | '
            f'Inside: {agents_inside}/1 | '
            f'{"⚠ WALL" if patch_wall_collision else "Clear"}',
            fontsize=11, color='red' if patch_wall_collision else 'black'
        )

        # Patch ellipse
        if patch_wall_collision:
            fc, ec, alpha = 'red', 'darkred', 0.4
        elif agents_inside == 1:
            fc, ec, alpha = 'cyan', 'darkblue', 0.3
        else:
            fc, ec, alpha = 'yellow', 'orange', 0.3
        from matplotlib.patches import Ellipse as _Ellipse
        ellipse = _Ellipse(
            xy=(self.patch.x, self.patch.y),
            width=self.patch.a * 2, height=self.patch.b * 2,
            angle=np.degrees(self.patch.theta),
            facecolor=fc, edgecolor=ec, alpha=alpha, linewidth=3,
        )
        ax.add_patch(ellipse)
        self._vis_dynamic_artists.append(ellipse)

        # Patch boundary dots
        for bx, by in self.patch.get_boundary_points(16):
            ln, = ax.plot(bx, by, 'k.', markersize=4, alpha=0.5)
            self._vis_dynamic_artists.append(ln)

        # Patch centre + velocity arrow
        ln, = ax.plot(self.patch.x, self.patch.y, 'b+', markersize=15, markeredgewidth=2)
        self._vis_dynamic_artists.append(ln)
        vx_p, vy_p = self.patch.get_velocity_vector()
        arr = ax.arrow(self.patch.x, self.patch.y, vx_p * 0.3, vy_p * 0.3,
                       head_width=0.2, head_length=0.15, fc='blue', ec='blue', linewidth=2)
        self._vis_dynamic_artists.append(arr)

        # Cars: A0 (agent), P (patch car)
        colors = ['red', 'blue']
        labels = ['A0', 'P']
        collisions = (self.current_base_obs.get("collisions", [0.0, 0.0])
                      if self.current_base_obs is not None else [0.0, 0.0])
        for i, (x, y, theta, v) in enumerate(agent_positions):
            inside = self.patch.is_inside(x, y) if i < 1 else True
            collision = float(collisions[i]) > 0.5 if i < len(collisions) else False
            color = colors[i] if i < len(colors) else 'gray'
            marker, size = ('X', 18) if collision else ('o', 12) if inside else ('s', 14)
            ln, = ax.plot(x, y, marker, color=color, markersize=size,
                          markeredgecolor='black', markeredgewidth=2)
            self._vis_dynamic_artists.append(ln)
            arr = ax.arrow(x, y, v * np.cos(theta) * 0.2, v * np.sin(theta) * 0.2,
                           head_width=0.1, head_length=0.05, fc=color, ec=color, alpha=0.7)
            self._vis_dynamic_artists.append(arr)
            lbl = labels[i] if i < len(labels) else str(i)
            txt = ax.text(x + 0.3, y + 0.3, lbl, fontsize=8, fontweight='bold')
            self._vis_dynamic_artists.append(txt)

        # Info box
        ep_rew = float(self.episode_reward_agents[0])
        # txt = ax.text(0.02, 0.98,
        #               f'Ep reward (agent): {ep_rew:.1f}\nPatch reward: {self.episode_reward_patch:.1f}',
        #               transform=ax.transAxes, fontsize=9, verticalalignment='top',
        #               bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        # self._vis_dynamic_artists.append(txt)

        # View limits centred on patch
        view_margin = max(self.patch.a, self.patch.b) + 5
        ax.set_xlim(cx - view_margin, cx + view_margin)
        ax.set_ylim(cy - view_margin, cy + view_margin)

        try:
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
        except Exception:
            pass

    def render(self):
        """Delegate to f110 renderer (requires render_mode='human' in JointEnvConfig)."""
        return self.f110.render()

    def close(self):
        if hasattr(self, '_fig') and self._fig is not None:
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        self.f110.close()


# === JOINT TRAINING DISABLED — JointPatchView wraps JointEnv for patch training ===
# class JointPatchView(gym.Env):
#     """Gym interface for the patch agent in joint training.
#
#     obs:    11D  [same layout as PatchEnv / PatchCarEnv: Frenet + track width + a,b + lane_id]
#     action: 4D   [steer, speed, a_cmd, b_cmd]
#
#     Physics executes immediately (does not wait for AgentViews).
#     Agent actions come from env._agent_action_fn if set, else zeros.
#     """
#
#     metadata = {"render_modes": [None]}
#
#     def __init__(self, shared_env: JointEnv):
#         super().__init__()
#         self.env = shared_env
#         cfg = shared_env.cfg
#         self.action_space = spaces.Box(
#             low=np.array([-0.4189, 1.5, cfg.patch_a_cmd_min, cfg.patch_b_cmd_min],
#                          dtype=np.float32),
#             high=np.array([0.4189, 10.0, cfg.patch_a_cmd_max, cfg.patch_b_cmd_max], dtype=np.float32),
#         )
#         # 263D: 7 scalars + 256 occupancy grid (exact match to Phase 0 PatchCarEnv)
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(263,), dtype=np.float32,
#         )
#
#     def reset(self, seed=None, options=None):
#         self.env.reset(seed=seed, options=options)
#         obs = self.env._step_obs[0]
#         if obs is None:
#             self.env.reset(seed=seed, options=options)
#             obs = self.env._step_obs[0]
#         return obs.copy(), {}
#
#     def step(self, patch_action):
#         patch_action = np.asarray(patch_action, dtype=np.float32)
#         obs0 = self.env._step_obs[1]
#         if self.env._agent_action_fn is not None and obs0 is not None:
#             action0 = self.env._agent_action_fn(obs0)
#         else:
#             action0 = np.zeros(2, dtype=np.float32)
#         self.env._execute_joint_step(
#             patch_action,
#             np.asarray(action0, dtype=np.float32),
#         )
#         obs = self.env._step_obs[0].copy()
#         info = dict(self.env._step_info)
#         info["agent_idx"] = "patch"
#         return obs, self.env._step_rewards[0], self.env._step_terminated, \
#                self.env._step_truncated, info
#
#     def set_agent_action_fn(self, fn):
#         """Inject frozen agent policy — callable from SubprocVecEnv.env_method."""
#         self.env._agent_action_fn = fn
#
#     def load_frozen_agent_npz(self, path):
#         """Load frozen agent weights from .npz and build a numpy-only forward pass.
#
#         Designed for SubprocVecEnv: only a file path (string) crosses the pipe,
#         so there are no pickle/cloudpickle issues with PyTorch objects.
#         """
#         if path is None:
#             self.env._agent_action_fn = None
#             return
#
#         data = dict(np.load(path, allow_pickle=False))
#         pi_weights, pi_biases = [], []
#         for i in range(0, 100, 2):
#             wk = f"pi_w_{i}"
#             bk = f"pi_b_{i}"
#             if wk not in data:
#                 break
#             pi_weights.append(data[wk])
#             pi_biases.append(data[bk])
#         act_w = data["act_w"]
#         act_b = data["act_b"]
#         use_tanh = bool(data["use_tanh"]) if "use_tanh" in data else True
#         has_vnorm = "vnorm_mean" in data
#         if has_vnorm:
#             vnorm_mean = data["vnorm_mean"]
#             vnorm_var = data["vnorm_var"]
#             vnorm_clip = float(data["vnorm_clip"])
#             vnorm_eps = float(data["vnorm_eps"])
#
#         def _forward(obs):
#             x = obs.astype(np.float32)
#             if has_vnorm:
#                 x = np.clip(
#                     (x - vnorm_mean) / np.sqrt(vnorm_var + vnorm_eps),
#                     -vnorm_clip, vnorm_clip,
#                 ).astype(np.float32)
#             for w, b in zip(pi_weights, pi_biases):
#                 x = x @ w.T + b
#                 x = np.tanh(x) if use_tanh else np.maximum(x, 0.0)
#             return (x @ act_w.T + act_b).astype(np.float32)
#
#         def _agent_fn(obs0):
#             return _forward(obs0)
#
#         self.env._agent_action_fn = _agent_fn
#
#     def render(self):
#         """Show live visualization if render_mode='human'."""
#         if self.env.cfg.render_mode == "human":
#             self.env._visualize()
#         return None
#
#     def close(self):
#         self.env.close()


# class JointAgentView(gym.Env):
#     """Gym interface for a learning agent in joint training.

#     obs:    24D CTDE  [ego 12D + partner 12D]
#     action: 2D        [steer, speed]  — absolute (not delta)

#     Two JointAgentViews (agent_idx=1 and agent_idx=2) share one JointEnv.
#     Physics executes once BOTH agent slots are filled.
#     Patch action comes from env._patch_action_fn if set, else a default action.
#     """

#     metadata = {"render_modes": [None]}

#     def __init__(self, shared_env: JointEnv, agent_idx: int):
#         super().__init__()
#         assert agent_idx in (1, 2), "agent_idx must be 1 or 2"
#         self.env = shared_env
#         self.agent_idx = agent_idx
#         self.action_space = spaces.Box(
#             low=np.array([-0.4189, 0.5], dtype=np.float32),
#             high=np.array([0.4189, 15.0], dtype=np.float32),
#         )
#         self.observation_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(JointEnv.AGENT_OBS_DIM,), dtype=np.float32,
#         )

#     def reset(self, seed=None, options=None):
#         if self.agent_idx == 1:
#             self.env.reset(seed=seed, options=options)
#         obs = self.env._step_obs[self.agent_idx]
#         if obs is None:
#             self.env.reset(seed=seed, options=options)
#             obs = self.env._step_obs[self.agent_idx]
#         return obs.copy(), {}

#     def step(self, action):
#         action = np.asarray(action, dtype=np.float32)
#         self.env._pending_actions[self.agent_idx] = action

#         if self.env._pending_actions[1] is not None and self.env._pending_actions[2] is not None:
#             patch_obs = self.env._step_obs[0]
#             if self.env._patch_action_fn is not None and patch_obs is not None:
#                 patch_action = np.asarray(
#                     self.env._patch_action_fn(patch_obs), dtype=np.float32)
#             else:
#                 cfg = self.env.cfg
#                 patch_action = np.array(
#                     [0.0, 2.0, cfg.patch_a, cfg.patch_b], dtype=np.float32)

#             self.env._execute_joint_step(
#                 patch_action,
#                 self.env._pending_actions[1],
#                 self.env._pending_actions[2],
#             )
#             self.env._pending_actions[1] = None
#             self.env._pending_actions[2] = None

#         obs = self.env._step_obs[self.agent_idx].copy()
#         info = dict(self.env._step_info)
#         info["agent_idx"] = self.agent_idx
#         return obs, self.env._step_rewards[self.agent_idx], \
#                self.env._step_terminated, self.env._step_truncated, info

#     def render(self):
#         return None

#     def close(self):
#         if self.agent_idx == 1:
#             self.env.close()


class PatchEnv(gym.Env):
    """
    Simplified patch environment matching single-agent structure.
    Patch is treated as an inflated agent with fixed size.
    """
    metadata = {"render_modes": ["human", "rgb_array", None]}
    
    def __init__(self, config: Optional[PatchEnvConfig] = None):
        super().__init__()
        self.cfg = config or PatchEnvConfig()
        
        # Action space: [steering, speed, a_cmd, b_cmd]
        #   b_cmd_min < patch_b so the policy CAN shrink the patch (required for split to trigger)
        self.action_space = spaces.Box(
            low=np.array([-0.4189, 1.5,
                          self.cfg.a_cmd_min, self.cfg.b_cmd_min], dtype=np.float32),
            high=np.array([0.4189, 10.0,
                           self.cfg.a_cmd_max, self.cfg.b_cmd_max], dtype=np.float32),
            dtype=np.float32,
        )
        # Lidar beams — same count as ppo_experiment.py ENV_CONFIG["num_beams"]
        # self.num_beams = self.cfg.num_beams
        # Holds the most recent f110 obs dict so _get_scan() can pull scan[0]
        self._current_base_obs = None

        # Observation space: 7 scalars + 16x16 local occupancy grid (flattened) = 263D
        # Scalars: [s_norm, ey_signed, speed, psi_error, a, b, curvature]
        # Grid:    16*16 = 256 values, occupancy 0.0=wall 1.0=free (map frame, no rotation)
        # REMOVED (COMMENTED OUT):
        # - Lookahead distances (half_w_5m, half_w_10m, half_w_15m)
        # - Grid rotation (cv2.warpAffine)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(17 + 256,),  # 273: 7 state + 10 curvature lookahead + 256 grid
            dtype=np.float32,
        )
        # No f110 agent needed — collision detection uses occupancy map directly
        # F110EnvAdapter is kept only to load the map and track data
        # num_agents=1 required: the simulator accesses agents[0] during init.
        # The dummy agent is never stepped — it's only there to satisfy the gym internals.
        self.f110 = F110EnvAdapter(
            F110Config(
                num_agents=1,
                reset_type=self.cfg.base_reset_type,
                control_input=("speed", "steering_angle"),
                timestep=self.cfg.control_dt,
                # extra_params={"num_beams": self.cfg.num_beams},
            ),
            render_mode=self.cfg.render_mode,
        )
        self.base_env = self.f110

        # Map reset helper — used for track width estimation and occupancy map
        self.map_reset = MapResetHelper()

        # Active patches — list of DynamicPatch objects
        # Normally 1 patch. Becomes 2+ after a split. Back to 1 after merge.
        self.active_patches = []
        self.lane_ids = []   # parallel list: lane_id for each active patch

        # Frenet bookkeeping — tracked per patch, indexed same as active_patches
        self.track_spline = None
        self.track_length = None
        self.prev_s_list = []
        self.prev_steer_list = []
        self.no_progress_list = []

        # --- Performance caches (rebuilt once per reset, reused every step) ---
        # Track data: never changes during an episode
        self._occ_map    = None
        self._resolution = None
        self._origin     = None
        # Track-width lookup: half_w sampled at regular s intervals along centerline
        # Shape: (N,) where _tw_s_vals[i] = s, _tw_half_w[i] = half track width
        self._tw_s_vals  = None
        self._tw_half_w  = None
        # Cached lookahead half-width for primary patch — set by _build_obs_for,
        # read by step() for split/merge decision (avoids calling _build_obs_for twice)
        self._last_ahead_half_w = 99.0

        # Episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.lap_progress = 0.0
        self.lap_count = 0
        self.lap_start_step = 0
        self._last_reward_terms = {}
        self._fig = None
        self._ax = None

    def _precompute_track_widths(self, n_samples: int = 200) -> None:
        """
        Sample track width at n_samples points along the centerline and store as a
        lookup array. Called once per reset — replaces per-step ray-casting calls.
        """
        if self._occ_map is None or self.track_spline is None:
            return
        s_vals = np.linspace(0.0, float(self.track_length), n_samples, endpoint=False)
        half_w = np.full(n_samples, 3.0, dtype=np.float32)   # default fallback
        for i, s in enumerate(s_vals):
            try:
                x, y = self.track_spline.calc_position(float(s))
                yaw  = self.track_spline.calc_yaw(float(s))
                w = float(self.map_reset.estimate_track_width(
                    self._occ_map, self._resolution, self._origin,
                    float(x), float(y), float(yaw),
                ))
                if np.isfinite(w) and w > 0.0:
                    half_w[i] = w / 2.0
            except Exception:
                pass
        self._tw_s_vals = s_vals.astype(np.float32)
        self._tw_half_w = half_w

    def _lookup_half_w(self, s: float) -> float:
        """Fast O(1) track half-width lookup using precomputed array."""
        if self._tw_s_vals is None:
            return 1.5   # fallback if not precomputed
        s_wrapped = float(s) % float(self.track_length)
        idx = int(np.searchsorted(self._tw_s_vals, s_wrapped))
        idx = min(idx, len(self._tw_half_w) - 1)
        return float(self._tw_half_w[idx])

    def _estimate_track_width_at_pos(self, x: float, y: float, theta: float) -> float:
        """Estimate track width — uses cached occ map (no repeated get_track_data calls)."""
        if self._occ_map is None:
            return 3.0
        try:
            width = float(
                self.map_reset.estimate_track_width(
                    self._occ_map, self._resolution, self._origin, x, y, theta,
                )
            )
            return width if (np.isfinite(width) and width > 0.0) else 3.0
        except Exception:
            return 3.0

    # def _get_scan(self) -> np.ndarray:
    #     """Extract scan for car 0 from the most recent f110 obs.
    #     Matches ppo_experiment.py F1tenthWrapper._get_scan — pad/trim to num_beams,
    #     replace NaN with 10.0 (max lidar range proxy).
    #     """
    #     base = self._current_base_obs
    #     if base is not None and "scans" in base:
    #         scan = np.asarray(base["scans"][0], dtype=np.float32)
    #     else:
    #         scan = np.zeros(self.num_beams, dtype=np.float32)

    #     if scan.shape[0] != self.num_beams:
    #         if scan.shape[0] > self.num_beams:
    #             scan = scan[:self.num_beams]
    #         else:
    #             scan = np.pad(scan, (0, self.num_beams - scan.shape[0]))

    #     return np.nan_to_num(scan, nan=10.0, posinf=10.0, neginf=0.0)

    def _get_local_grid(self, patch: "DynamicPatch") -> np.ndarray:
        """
        Extract a 16x16 local occupancy grid (8m x 8m) around the patch (map frame, no rotation).
        Returns float32 shape (256,), 0.0=wall, 1.0=free.

        SIMPLIFIED: Keep grid in map frame, no cv2.warpAffine rotation.
        CNN will handle rotation invariance.
        """
        # import cv2  # COMMENTED OUT: no rotation needed

        if self._occ_map is None:
            return np.ones(256, dtype=np.float32)  # fallback: all free

        occ = self._occ_map
        res = float(self._resolution)
        ox, oy = float(self._origin[0]), float(self._origin[1])

        # Convert patch center to pixel coords
        px = (float(patch.x) - ox) / res
        py = (float(patch.y) - oy) / res

        # Extract 8m x 8m region (4m in each direction from center)
        # map pixels per 4 meters
        half_size_pix = int(round(4.0 / res))

        H, W = occ.shape
        # Simple crop in map frame (no rotation)
        r0 = max(0, int(round(py)) - half_size_pix)
        r1 = min(H, int(round(py)) + half_size_pix)
        c0 = max(0, int(round(px)) - half_size_pix)
        c1 = min(W, int(round(px)) + half_size_pix)
        crop = occ[r0:r1, c0:c1]

        if crop.shape[0] < 1 or crop.shape[1] < 1:
            return np.ones(256, dtype=np.float32)

        # Resize to 16x16 using scipy (no cv2 needed)
        grid = ndimage.zoom(crop, (16.0 / crop.shape[0], 16.0 / crop.shape[1]), order=1)
        grid = grid[:16, :16]  # Ensure exactly 16x16
        if grid.shape != (16, 16):
            grid = np.ones((16, 16), dtype=np.float32)

        return np.clip(grid, 0.0, 1.0).astype(np.float32).flatten()  # (256,)

    def _build_obs_for(self, patch: "DynamicPatch", lane_id: int) -> np.ndarray:
        """
        Build simplified observation for patch car traversal.
        7 scalars + 256 grid values (16x16 local occupancy, map frame, no rotation).
        Scalars: [s_norm, ey_signed, speed, psi_error, a, b, curvature]

        REMOVED (COMMENTED OUT):
        - Lookahead distances (5m, 10m, 15m) — CNN will learn from occupancy grid
        - Grid rotation (cv2.warpAffine) — using map frame, CNN handles rotation invariance
        """
        if self.track_spline is None or self.track is None:
            raise RuntimeError("track_spline or track not initialized!")

        s, ey_signed = self._patch_to_frenet(patch)
        speed = float(patch.v)
        yaw   = float(patch.theta)
        a, b  = float(patch.a), float(patch.b)

        s_norm = float(s) / max(float(self.track_length), 1.0)

        # FIX: track-relative heading instead of world-frame yaw
        try:
            track_yaw = float(self.track_spline.calc_yaw(float(s)))
        except Exception:
            track_yaw = yaw
        psi_error = self._wrap_angle(yaw - track_yaw)  # in [-pi, pi]

        # COMMENTED OUT: lookahead distances — not needed
        # L = float(self.track_length)
        # half_w_5m  = max(self._lookup_half_w((s + self.cfg.patch_lookahead_5m)  % L), 1e-3)
        # half_w_10m = max(self._lookup_half_w((s + self.cfg.patch_lookahead_10m) % L), 1e-3)
        # half_w_15m = max(self._lookup_half_w((s + self.cfg.patch_lookahead_15m) % L), 1e-3)
        # self._last_ahead_half_w = half_w_5m  # keep for split/merge compat

        # Track curvature — current position + 10 lookahead samples every 2m (covers 20m ahead).
        # This is the car's "1D track map": sees upcoming corners and can brake proactively.
        L = float(self.track_length) if self.track_length else 1.0
        try:
            curvature = float(self.track_spline.calc_curvature(float(s)))
            curvature_ahead = [
                float(self.track_spline.calc_curvature(float((s + (i + 1) * 2.0) % L)))
                for i in range(10)  # s+2m, s+4m, ..., s+20m
            ]
        except Exception:
            curvature = 0.0
            curvature_ahead = [0.0] * 10

        # 17D scalars: 7 state features + 10 curvature lookahead values
        scalars = np.array(
            [s_norm, ey_signed, speed, psi_error, a, b, curvature] + curvature_ahead,
            dtype=np.float32,
        )  # (17,)

        grid_flat = self._get_local_grid(patch)  # (256,)

        return np.concatenate([scalars, grid_flat])  # (273,)

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

    def _patch_to_frenet(self, patch: "DynamicPatch") -> tuple[float, float]:
        if self.track is None or not hasattr(self.track, 'cartesian_to_frenet'):
            return 0.0, 0.0
        try:
            s, ey, _ = self.track.cartesian_to_frenet(
                float(patch.x), float(patch.y), float(patch.theta), s_guess=0)
            if not np.isfinite(s) or not np.isfinite(ey):
                return 0.0, 0.0
            return float(s), float(ey)
        except (AttributeError, TypeError):
            return 0.0, 0.0

    def _lane_center_pose(self, s: float, lane_id: int) -> tuple[float, float, float]:
        """World (x, y, psi) for lane center at arc length s. lane_id 1 = left, 2 = right."""
        if self.track_spline is None or self.track_length is None:
            return 0.0, 0.0, 0.0
        L = float(self.track_length)
        s_wrapped = float(s) % L if L > 0 else float(s)
        try:
            xc, yc = self.track_spline.calc_position(s_wrapped)
            psi = float(self.track_spline.calc_yaw(s_wrapped))
        except Exception:
            return 0.0, 0.0, 0.0
        half_w = max(self._lookup_half_w(s_wrapped), 1e-3)
        offset = half_w / 2.0
        lx = -float(np.sin(psi))
        ly = float(np.cos(psi))
        if lane_id == 1:
            x = float(xc + offset * lx)
            y = float(yc + offset * ly)
        elif lane_id == 2:
            x = float(xc - offset * lx)
            y = float(yc - offset * ly)
        else:
            x, y = float(xc), float(yc)
        return x, y, psi

    def _ego_and_other_lane_ids(self, x: float, y: float) -> tuple[int, int]:
        """Which lane (1=left, 2=right) is closest to (x,y); return (ego_lane, other_lane)."""
        if self.track_spline is None:
            return 1, 2
        try:
            s, _ = self.track_spline.calc_arclength_inaccurate(float(x), float(y))
        except Exception:
            return 1, 2
        x1, y1, _ = self._lane_center_pose(s, 1)
        x2, y2, _ = self._lane_center_pose(s, 2)
        d1 = float(np.hypot(x - x1, y - y1))
        d2 = float(np.hypot(x - x2, y - y2))
        if d1 <= d2:
            return 1, 2
        return 2, 1

    def _sync_follower_zone_from_spline(
        self,
        leader_patch: "DynamicPatch",
        zone_patch: "DynamicPatch",
        zone_lane_id: int,
        steering_cmd: float,
        dt: float,
        a_cmd: float,
        b_cmd: float,
    ) -> None:
        """Place zone_patch on the lane center at the same s as leader (map-based follower zone)."""
        s, _ = self._patch_to_frenet(leader_patch)
        x, y, psi = self._lane_center_pose(s, zone_lane_id)
        zone_patch.sync_from_pose(
            x, y, psi, float(leader_patch.v), float(steering_cmd),
        )
        s_i, _ = self._patch_to_frenet(zone_patch)
        hw_i = self._lookup_half_w(s_i)
        max_b_i = min(self.cfg.b_cmd_max, hw_i * 0.90)
        zone_patch.update_shape(
            a_cmd, b_cmd, dt,
            max_a=self.cfg.a_cmd_max, max_b=max_b_i,
        )

    def _estimate_track_width_at(self, patch: "DynamicPatch") -> float:
        """Estimate local track width at the given patch pose using occupancy map."""
        return self._estimate_track_width_at_pos(
            float(patch.x), float(patch.y), float(patch.theta)
        )

    # def _compute_reward_w_frenet(
    #     self,
    #     lidar_info: dict,
    #     dt: float,
    #     lap_bonus: float = 0.0,
    # ) -> float:
    #     """
    #     Legacy single-agent lidar reward path (not wired from step(); unused).

    #     Simplified reward matching single-agent PPO exactly.
    #     Only Frenet-based terms: progress, cross-track, steering penalties, collision, stuck, lap bonus, time penalty.
    #     """
    #     # Frenet features
    #     s, ey_track = self._patch_to_frenet()
    #     ds = self._wrap_ds(s - self.prev_s) if self.prev_s is not None else 0.0

    #     # In split mode, ey is relative to lane centerline, not track centerline
    #     track_width = self._estimate_current_track_width()
    #     half_w = max(track_width / 2.0, 1e-3)
    #     if self.lane_id == 1:
    #         ey = ey_track - (half_w / 2.0)
    #     elif self.lane_id == 2:
    #         ey = ey_track + (half_w / 2.0)
    #     else:
    #         ey = ey_track

    #     # Patch controls / yaw-rate proxy (same as single-agent uses agent controls)
    #     steer_cmd = float(self.patch.steering)
    #     steer_rate = abs(steer_cmd - self.prev_steer) / max(dt, 1e-6)
    #     yaw_rate = float((self.patch.v / max(self.cfg.wheelbase, 1e-6)) * np.tan(steer_cmd))
    #     spin_excess = max(0.0, abs(yaw_rate) - self.cfg.spin_yawrate_threshold)

    #     # Collision proxy from lidar (same as single-agent)
    #     min_dist = float(lidar_info["min_dist"])
    #     collision_lidar = (not np.isfinite(min_dist)) or (min_dist < self.cfg.collision_min_dist)
        
    #     # Collision from patch boundary discretization
    #     patch_boundary_collision = lidar_info.get("patch_boundary_collision", False)
        
    #     # Combined collision flag (either lidar or patch boundary collision)
    #     collision = collision_lidar or patch_boundary_collision

    #     # Stuck counter (same as single-agent)
    #     if abs(ds) < self.cfg.stuck_progress_eps:
    #         self.no_progress_counter += 1
    #     else:
    #         self.no_progress_counter = 0

    #     # Lane-centering penalty in split mode:
    #     # ey is already relative to lane centerline, so penalize deviation from it
    #     lane_center_penalty = 0.0
    #     if self.lane_id != 0:
    #         lane_center_penalty = self.cfg.lane_centering_weight * abs(ey)

    #     # Core reward
    #     reward_raw = (
    #         self.cfg.reward_progress_scale * ds
    #         - self.cfg.reward_crosstrack_weight * abs(ey)
    #         - lane_center_penalty                           # extra penalty in split mode
    #         - self.cfg.reward_steer_bias_weight * abs(steer_cmd)
    #         - self.cfg.reward_steer_rate_weight * steer_rate
    #         - self.cfg.reward_spin_weight * spin_excess
    #         - (self.cfg.collision_penalty if collision else 0.0)
    #     )


    #     a_now = float(max(self.patch.a, 1e-3))
    #     b_now = float(max(self.patch.b, 1e-3))
    #     base_area = float(self.cfg.patch_a * self.cfg.patch_b)  # area at spawn size
    #     current_area = a_now * b_now
    #     area_ratio = current_area / max(base_area, 1e-3)
    #     area_excess = max(0.0, area_ratio - 1.0)  # 0 if <= base area

    #     # Per-step time penalty (same as single-agent)
    #     reward_raw -= self.cfg.time_penalty_per_sec * dt
    #     reward_raw -= self.cfg.shape_area_penalty_weight * area_excess

    #     # Event-based lap bonus (same as single-agent)
    #     reward_raw += float(lap_bonus)

    #     # Stuck penalty (same as single-agent)
    #     if self.no_progress_counter >= self.cfg.stuck_no_progress_steps:
    #         reward_raw -= self.cfg.stuck_penalty

    #     # Update bookkeeping
    #     self.prev_s = s
    #     self.prev_steer = steer_cmd
        
    #     reward_clipped = float(np.clip(reward_raw, -100.0, 100.0))
        
    #     # Simplified reward terms (matching single-agent)
    #     self._last_reward_terms = {
    #         "s": float(s),
    #         "ey": float(ey),
    #         "ds": float(ds),
    #         "steer_cmd": float(steer_cmd),
    #         "steer_rate": float(steer_rate),
    #         "yaw_rate_proxy": float(yaw_rate),
    #         "spin_excess": float(spin_excess),
    #         "collision_proxy": bool(collision),
    #         "collision_lidar": bool(collision_lidar),
    #         "collision_patch_boundary": bool(patch_boundary_collision),
    #         "reward_raw": float(reward_raw),
    #         "reward_clipped": float(reward_clipped),
    #         "reward_was_clipped": bool(abs(reward_raw) > 100.0),
    #         "no_progress_counter": int(self.no_progress_counter),
    #     }
        # return reward_clipped
        
    # def _check_termination(self, lidar_info: dict):
    #     """
    #     Legacy lidar-based termination (not wired from step(); unused).

    #     Simplified termination matching single-agent PPO exactly.
    #     Only: collision (lidar or patch boundary), stuck, max_steps.
    #     """
    #     terminated = False
    #     truncated = False
    #     reason = None

    #     # 1) Collision from lidar (same as single-agent)
    #     min_dist = float(lidar_info["min_dist"])
    #     if (not np.isfinite(min_dist)) or (min_dist < self.cfg.collision_min_dist):
    #         return True, False, "collision_lidar"

    #     # 2) Collision from patch boundary discretization
    #     patch_boundary_collision = lidar_info.get("patch_boundary_collision", False)
    #     if patch_boundary_collision:
    #         return True, False, "collision_patch_boundary"

    #     # 3) Stuck termination (same as single-agent)
    #     if self.no_progress_counter >= self.cfg.stuck_no_progress_steps:
    #         return True, False, "stuck_no_progress"

    #     # 4) Time limit (same as single-agent)
    #     if self.step_count >= self.cfg.max_steps:
    #         truncated = True
    #         reason = "max_steps"

    #     return terminated, truncated, reason
        
    # ------------------------------------------------------------------
    # Per-patch helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Return 263D hybrid observation for the primary patch (active_patches[0]).
        Layout: [7 scalars | 256 grid values (16x16 local occupancy, flattened)]"""
        patch   = self.active_patches[0]
        lane_id = self.lane_ids[0]
        obs = self._build_obs_for(patch, lane_id)
        return obs

    def _collision_for(self, patch: "DynamicPatch") -> bool:
        """Check occupancy-map collision using cached map — no get_track_data() call."""
        if self._occ_map is None:
            return False
        try:
            patch_collision, _ = patch.check_patch_boundary_wall_collision(
                self._occ_map, self._resolution, self._origin,
                n_points=32,
                violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )
            return bool(patch_collision)
        except Exception:
            return False

    def _compute_reward_for(
        self,
        patch: "DynamicPatch",
        lane_id: int,
        prev_s: float,
        prev_steer: float,
        no_progress: int,
        dt: float,
        lap_bonus: float = 0.0,
        cached_collision: bool = False,
        sim_yaw_rate: Optional[float] = None,
    ) -> tuple:
        """
        Compute reward for one patch.
        Returns (reward_clipped, new_prev_s, new_prev_steer, new_no_progress, terms_dict).
        cached_collision: pre-computed occupancy-map collision result to avoid double call.
        """
        s, ey_track = self._patch_to_frenet(patch)
        ds = self._wrap_ds(s - prev_s) if prev_s is not None else 0.0

        half_w = max(self._lookup_half_w(s), 1e-3)   # O(1) lookup, no ray cast
        if lane_id == 1:
            ey = ey_track - (half_w / 2.0)
        elif lane_id == 2:
            ey = ey_track + (half_w / 2.0)
        else:
            ey = ey_track

        steer_cmd   = float(getattr(patch, "steering", 0.0))
        steer_rate  = abs(steer_cmd - prev_steer) / max(dt, 1e-6)
        if sim_yaw_rate is not None and np.isfinite(sim_yaw_rate):
            yaw_rate = float(sim_yaw_rate)
        else:
            yaw_rate = float((patch.v / max(self.cfg.wheelbase, 1e-6)) * np.tan(steer_cmd))
        spin_excess = max(0.0, abs(yaw_rate) - self.cfg.spin_yawrate_threshold)

        # Use the pre-computed collision (occupancy map boundary check only — no Frenet proxy)
        collision = cached_collision

        # Stuck counter
        if abs(ds) < self.cfg.stuck_progress_eps:
            no_progress += 1
        else:
            no_progress = 0

        # Lane-centering penalty in split mode
        # lane_center_penalty = 0.0
        # if lane_id != 0:
        #     lane_center_penalty = self.cfg.lane_centering_weight * abs(ey)

        a_now = float(max(patch.a, 1e-3))
        b_now = float(max(patch.b, 1e-3))
        base_area    = float(self.cfg.patch_a * self.cfg.patch_b)
        current_area = a_now * b_now
        area_ratio   = current_area / max(base_area, 1e-3)
        area_excess  = max(0.0, area_ratio - 1.0)

        speed_now = float(getattr(patch, "v", 0.0))
        reward_raw = (
            self.cfg.reward_progress_scale * ds
            - self.cfg.reward_crosstrack_weight * abs(ey)
            # - lane_center_penalty
            - self.cfg.reward_steer_bias_weight * abs(steer_cmd)
            - self.cfg.reward_steer_rate_weight * steer_rate
            - self.cfg.reward_spin_weight * spin_excess
            - (self.cfg.collision_penalty if collision else 0.0)
            - self.cfg.shape_area_penalty_weight * area_excess
            - self.cfg.time_penalty_per_sec * float(dt)
            + float(lap_bonus)
            + self.cfg.reward_speed_weight * (speed_now - 1.5)  # zero at min speed, ~1.28 at 10 m/s
        )
        if no_progress >= self.cfg.stuck_no_progress_steps:
            reward_raw -= self.cfg.stuck_penalty

        reward_clipped = float(np.clip(reward_raw, -100.0, 100.0))

        terms = {
            "s": float(s), "ey": float(ey), "ds": float(ds),
            "collision": bool(collision), "no_progress": int(no_progress),
            "reward_raw": float(reward_raw), "reward_clipped": float(reward_clipped),
            "yaw_rate": float(yaw_rate),
            "spin_excess": float(spin_excess),
        }
        return reward_clipped, float(s), steer_cmd, no_progress, terms

    def _check_termination_for(
        self,
        no_progress: int,
        patch_occ_collision: bool,
    ) -> tuple:
        """Check termination for a single patch. Returns (terminated, reason).

        PatchCarEnv uses only occupancy patch-boundary checks (not F110 mesh collisions):
        the ellipse is the safety envelope, so termination matches that criterion alone.
        """
        if patch_occ_collision:
            return True, "collision_patch_boundary"
        if no_progress >= self.cfg.stuck_no_progress_steps:
            return True, "stuck_no_progress"
        return False, None

    def _do_split(self) -> None:
        """
        Split into two zones: [ego, follower].

        - Index 0 (ego): stays tied to the leader vehicle pose (bicycle or f110 in step()).
        - Index 1 (follower): lane center on the spline at the same arc length as ego,
          opposite lane (map-based), matching deployment (one robot + virtual corridor).
        """
        parent = self.active_patches[0]
        parent_s, _ = self._patch_to_frenet(parent)
        parent_steer = float(getattr(parent, "steering", 0.0))

        lane_b = max(
            min(self._last_ahead_half_w * 0.85, parent.b / 2.0),
            0.3,
        )

        ego_lane, other_lane = self._ego_and_other_lane_ids(float(parent.x), float(parent.y))

        ego_patch = DynamicPatch(
            x=float(parent.x),
            y=float(parent.y),
            theta=float(parent.theta),
            v=float(parent.v),
            a=float(parent.a),
            b=float(lane_b),
        )
        ego_patch.steering = parent_steer

        ox, oy, opsi = self._lane_center_pose(parent_s, other_lane)
        other_patch = DynamicPatch(
            x=ox,
            y=oy,
            theta=float(opsi),
            v=float(parent.v),
            a=float(parent.a),
            b=float(lane_b),
        )
        other_patch.steering = parent_steer

        self.active_patches = [ego_patch, other_patch]
        self.lane_ids = [ego_lane, other_lane]
        self.prev_s_list = [float(parent_s), float(parent_s)]
        self.prev_steer_list = [parent_steer, parent_steer]
        self.no_progress_list = [0, 0]


    def _do_merge(self) -> None:
        """
        Merge two lane patches back into a single full-track patch at their centroid.
        """
        p0, p1 = self.active_patches[0], self.active_patches[1]
        cx = (p0.x + p1.x) / 2.0
        cy = (p0.y + p1.y) / 2.0
        # Average heading (handle wrap-around)
        dtheta = float(np.arctan2(
            np.sin(p1.theta - p0.theta),
            np.cos(p1.theta - p0.theta),
        ))
        merged_theta = p0.theta + dtheta / 2.0
        merged_v     = (p0.v + p1.v) / 2.0
        merged_b     = p0.b + p1.b  # restore width by summing lane halves

        merged = DynamicPatch(
            x=cx, y=cy, theta=float(merged_theta),
            v=float(merged_v), a=float(p0.a), b=float(merged_b),
        )
        merged.steering = (float(getattr(p0, "steering", 0.0)) +
                           float(getattr(p1, "steering", 0.0))) / 2.0

        merged_s = (self.prev_s_list[0] + self.prev_s_list[1]) / 2.0

        self.active_patches   = [merged]
        self.lane_ids         = [0]
        self.prev_s_list      = [merged_s]
        self.prev_steer_list  = [merged.steering]
        self.no_progress_list = [0]

    def reset(self, seed=None, options=None):
        """
        Simplified reset matching single-agent PPO exactly.
        Initialize Frenet spline, spawn patch at centerline start, reset base env.
        """
        super().reset(seed=seed)
        self._np_random = np.random.RandomState(seed if seed is not None else None)
        
        # Basic episode tracking
        self.step_count = 0
        self.episode_reward = 0.0
        self.lap_progress = 0.0
        self.lap_count = 0
        self.lap_start_step = 0
        self._last_sim_yaw_rate = 0.0

        # Initialize Frenet spline — one get_track_data() call, cache everything
        self.f110.ensure_initialized()
        track, occ_map, resolution, origin = self.f110.get_track_data()
        if track.centerline is None or track.centerline.spline is None:
            raise ValueError("Track centerline/spline missing for Frenet reward.")

        self.track = track
        self.track_spline  = track.centerline.spline
        self.track_length  = float(self.track_spline.s[-1])

        # Cache for the whole episode — reused by _collision_for and _lookup_half_w
        self._occ_map    = occ_map / 255.0 if occ_map is not None else None  # Normalize to [0, 1]
        self._resolution = resolution
        self._origin     = origin
        self._last_ahead_half_w = 99.0

        # Precompute half-widths along the centerline (200 ray casts at reset only)
        # All per-step width lookups become O(1) array index — zero ray casting in rollout
        self._precompute_track_widths(n_samples=max(200, int(self.track_length / 0.15)))

        # Spawn patch at centerline waypoint
        xs = np.asarray(track.centerline.xs, dtype=np.float32)
        ys = np.asarray(track.centerline.ys, dtype=np.float32)
        if self.cfg.random_spawn:
            spawn_idx = int(np.random.randint(0, xs.shape[0]))
        else:
            spawn_idx = 0
        spawn_idx = max(0, min(spawn_idx, xs.shape[0] - 1))
        patch_x, patch_y = float(xs[spawn_idx]), float(ys[spawn_idx])
        next_idx = (spawn_idx + 1) % xs.shape[0]
        dx = float(xs[next_idx] - xs[spawn_idx])
        dy = float(ys[next_idx] - ys[spawn_idx])
        patch_theta = float(np.arctan2(dy, dx))
        self.start_xy = (patch_x, patch_y)

        # Always spawn on centerline at full size.
        # Toll plaza narrows b mid-episode → triggers split via rule in step().
        init_patch = DynamicPatch(
            x=patch_x,
            y=patch_y,
            theta=patch_theta,
            v=0.5,
            a=self.cfg.patch_a,
            b=self.cfg.patch_b,
        )
        if self.cfg.split_mode:
            # Initialize multi-patch lists (start with single full-track patch)
            self.active_patches = [init_patch]
            self.lane_ids = [0]               # 0 = full track, 1 = left lane, 2 = right lane

            init_s, _ = self._patch_to_frenet(init_patch)
            self.prev_s_list     = [init_s]
            self.prev_steer_list = [0.0]
            self.no_progress_list = [0]

            # Reset base env with a dummy pose — 1 agent required by the simulator
            self._current_base_obs, _ = self.f110.reset(
                poses=np.array([[patch_x, patch_y, patch_theta]], dtype=np.float32))

            # Build and return observation for primary patch
            obs = self._build_obs()
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        else:
            # Single-patch traversal mode — same list structure as split_mode so
            # the rest of the code (step, reward, termination) works unchanged.
            # No split/merge will ever fire (guarded in step()).
            self.active_patches = [init_patch]
            self.lane_ids = [0]

            init_s, _ = self._patch_to_frenet(init_patch)
            self.prev_s_list      = [init_s]
            self.prev_steer_list  = [0.0]
            self.no_progress_list = [0]

            self._current_base_obs, _ = self.f110.reset(
                poses=np.array([[patch_x, patch_y, patch_theta]], dtype=np.float32))

            obs = self._build_obs()
            obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        return obs, {}

    def step(self, action):
        """
        Multi-patch step with rule-based split/merge.

        Action: [steering, speed, a_cmd, b_cmd] — always applied to primary patch (active_patches[0]).
        When in split mode (2 patches), patch_1 mirrors patch_0 with the same speed/size
        commands but opposite steering (symmetric lane-following).

        Reward is the cooperative sum across all active patches.
        """
        self.step_count += 1
        dt  = self.cfg.control_dt
        lap_bonus = 0.0
        lap_time  = None

        # --- Decode action (applied to primary patch) ---
        steering_cmd = float(np.clip(
            np.nan_to_num(action[0], nan=0.0, posinf=0.4189, neginf=-0.4189),
            -0.4189, 0.4189))
        speed_cmd = float(np.clip(
            np.nan_to_num(action[1], nan=1.5, posinf=10.0, neginf=1.5),
            1.5, 10.0))
        a_cmd = float(np.clip(
            np.nan_to_num(action[2], nan=self.cfg.a_cmd_min,
                          posinf=self.cfg.a_cmd_max, neginf=self.cfg.a_cmd_min),
            self.cfg.a_cmd_min, self.cfg.a_cmd_max))
        b_cmd = float(np.clip(
            np.nan_to_num(action[3], nan=self.cfg.b_cmd_min,
                          posinf=self.cfg.b_cmd_max, neginf=self.cfg.b_cmd_min),
            self.cfg.b_cmd_min, self.cfg.b_cmd_max))

        # --- Step each active patch ---
        # Split layout: [0]=ego (bicycle integration), [1]=follower zone (spline lane center).
        if len(self.active_patches) == 1:
            patch = self.active_patches[0]
            patch.steering = steering_cmd
            patch.step(speed_cmd, steering_cmd, dt)
            s_i, _ = self._patch_to_frenet(patch)
            hw_i = self._lookup_half_w(s_i)
            max_b_i = min(self.cfg.b_cmd_max, hw_i * 0.90)
            patch.update_shape(a_cmd, b_cmd, dt,
                               max_a=self.cfg.a_cmd_max, max_b=max_b_i)
        else:
            # ego = self.active_patches[0]
            # other = self.active_patches[1]
            # ego.steering = steering_cmd
            # ego.step(speed_cmd, steering_cmd, dt)
            # s_i, _ = self._patch_to_frenet(ego)
            # hw_i = self._lookup_half_w(s_i)
            # max_b_i = min(self.cfg.b_cmd_max, hw_i * 0.90)
            # ego.update_shape(a_cmd, b_cmd, dt,
            #                  max_a=self.cfg.a_cmd_max, max_b=max_b_i)
            # self._sync_follower_zone_from_spline(
            #     ego, other, self.lane_ids[1], steering_cmd, dt, a_cmd, b_cmd,
            # )
            pass

        # --- Lap detection (use primary patch) ---
        primary = self.active_patches[0]
        s_now, ey_now = self._patch_to_frenet(primary)
        if self.track_spline is not None and self.track_length is not None and self.track_length > 0.0:
            self.lap_progress = float(np.clip(s_now / self.track_length, 0.0, 1.0))
            prev_s_primary = self.prev_s_list[0]
            if prev_s_primary is not None:
                raw_ds = float(s_now - prev_s_primary)
                wrapped_ds = float(self._wrap_ds(raw_ds))
                if raw_ds < -0.5 * float(self.track_length) and wrapped_ds > 0.0:
                    self.lap_count += 1
                    lap_time = (self.step_count - self.lap_start_step) * dt
                    lap_time = max(float(lap_time), 1e-3)
                    lap_bonus = self.cfg.lap_finish_bonus * float(
                        np.exp(-lap_time / max(self.cfg.lap_bonus_tau, 1e-6))
                    )
                    self.lap_start_step = self.step_count

        # === SPLIT/MERGE DISABLED FOR DEBUGGING ===
        # if self.active_patches:
        #     self._build_obs_for(self.active_patches[0], self.lane_ids[0])
        # if self.cfg.split_mode:
        #     in_split = (len(self.active_patches) > 1)
        #     if not in_split and self._last_ahead_half_w < self.cfg.split_ahead_half_w:
        #         self._do_split()
        #     elif in_split and self._last_ahead_half_w > self.cfg.merge_ahead_half_w:
        #         self._do_merge()

        # --- Per-patch reward + termination ---
        total_reward = 0.0
        any_terminated = False
        termination_reason = None
        all_terms = {}

        for i, (patch, lane_id) in enumerate(zip(self.active_patches, self.lane_ids)):
            # Cache collision once per patch — reused in both reward and termination
            col_i = self._collision_for(patch)

            bonus_i = lap_bonus if i == 0 else 0.0
            r, new_s, new_steer, new_no_progress, terms = self._compute_reward_for(
                patch, lane_id,
                self.prev_s_list[i], self.prev_steer_list[i],
                self.no_progress_list[i],
                dt, lap_bonus=bonus_i,
                cached_collision=col_i,
            )
            self.prev_s_list[i]      = new_s
            self.prev_steer_list[i]  = new_steer
            self.no_progress_list[i] = new_no_progress

            total_reward += r

            terminated_i, reason_i = self._check_termination_for(
                new_no_progress, col_i
            )
            if terminated_i and not any_terminated:
                any_terminated = True
                termination_reason = reason_i

            for k, v in terms.items():
                all_terms[f"p{i}_{k}"] = v

        # Average reward across patches (cooperative but normalized so scale is stable)
        n = max(len(self.active_patches), 1)
        reward = float(np.nan_to_num(total_reward / n, nan=-10.0, posinf=100.0, neginf=-100.0))
        self.episode_reward += reward

        # --- Truncation check ---
        truncated = False
        if self.step_count >= self.cfg.max_steps:
            truncated = True
            termination_reason = termination_reason or "max_steps"

        # --- Build observation for primary patch ---
        obs = self._build_obs()
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

        info = {
            "episode_reward":     self.episode_reward,
            "lap_progress":       self.lap_progress,
            "Episode_steps":      self.step_count,
            "step_reward":        reward,
            "termination_reason": termination_reason,
            "lap_count":          int(self.lap_count),
            "lap_bonus":          float(lap_bonus),
            "lap_time":           None if lap_time is None else float(lap_time),
            "frenet_s":           float(s_now),
            "frenet_ey":          float(ey_now),
            "num_active_patches": len(self.active_patches),
            "lane_ids":           list(self.lane_ids),
            "primary_b":          float(self.active_patches[0].b),
        }
        info.update(all_terms)

        return obs, reward, any_terminated, truncated, info

    def render(self):
        """Matplotlib overlay + f1tenth pygame window (both need updates each step)."""
        rgb = None
        if self.cfg.render_mode == "human":
            self._visualize()
        # f1tenth_gym only draws the car/track in its window when base_env.render() runs;
        # without this call the pygame surface stays black even though physics steps.
        if self.cfg.render_mode in ("human", "rgb_array"):
            rgb = self.f110.render()
        return rgb

    def _visualize(self):
        """Visualize all active patches, track walls, and episode info."""
        # --- First call (or after window closed): setup figure and cache ---
        if self._fig is None or not plt.fignum_exists(self._fig.number):
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(9, 7))
            self._ax.set_aspect('equal')
            self._ax.grid(True, alpha=0.25)
            self._vis_bg_image = None
            self._vis_bg_bounds = None
            self._vis_dynamic_artists = []
            self._vis_occ_cache = None
            plt.show(block=False)

        ax = self._ax

        # --- Remove dynamic artists from the previous frame (no ax.clear()) ---
        for artist in self._vis_dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._vis_dynamic_artists = []

        if not self.active_patches:
            ax.set_title("No active patches")
            try:
                ax.figure.canvas.draw()
                ax.figure.canvas.flush_events()
            except Exception:
                pass
            return

        prim = self.active_patches[0]
        cx, cy = prim.x, prim.y

        # --- Occupancy map: fetched once and cached (never changes) ---
        occ_map = resolution = origin = None
        if self._vis_occ_cache is None:
            try:
                _, occ_map, resolution, origin = self.base_env.get_track_data()
                occ_map = occ_map / 255.0 if occ_map is not None else None  # Normalize to [0, 1]
                self._vis_occ_cache = (occ_map, resolution, origin)
            except Exception:
                pass
        if self._vis_occ_cache is not None:
            occ_map, resolution, origin = self._vis_occ_cache

        # --- Redraw background imshow only when the view window shifts ---
        if occ_map is not None:
            margin = max(prim.a, prim.b) + 15
            px_min = max(0, int((cx - margin - origin[0]) / resolution))
            px_max = min(occ_map.shape[1], int((cx + margin - origin[0]) / resolution))
            py_min = max(0, int((cy - margin - origin[1]) / resolution))
            py_max = min(occ_map.shape[0], int((cy + margin - origin[1]) / resolution))
            new_bounds = (px_min, px_max, py_min, py_max)

            if self._vis_bg_bounds != new_bounds and px_max > px_min and py_max > py_min:
                if self._vis_bg_image is not None:
                    try:
                        self._vis_bg_image.remove()
                    except Exception:
                        pass
                region = occ_map[py_min:py_max, px_min:px_max]
                xw0 = px_min * resolution + origin[0]
                xw1 = px_max * resolution + origin[0]
                yw0 = py_min * resolution + origin[1]
                yw1 = py_max * resolution + origin[1]
                # Single rasterised image instead of contourf/contour vector ops
                rgba = np.zeros((*region.shape, 4), dtype=np.uint8)
                rgba[region < 0.5] = [51, 51, 51, 204]    # dark walls
                rgba[region >= 0.5] = [220, 220, 220, 60]  # light free space
                self._vis_bg_image = ax.imshow(
                    rgba, extent=[xw0, xw1, yw0, yw1],
                    origin='lower', aspect='auto', zorder=0, interpolation='nearest')
                self._vis_bg_bounds = new_bounds

        # --- Draw each active patch ---
        patch_styles = [("gold", "darkorange"), ("cyan", "steelblue"), ("lime", "darkgreen")]
        any_collision = False
        for pi, p in enumerate(self.active_patches):
            p_col = False
            if occ_map is not None:
                try:
                    p_col, _ = p.check_patch_boundary_wall_collision(
                        occ_map,
                        resolution,
                        origin,
                        n_points=32,
                        violation_threshold=self.cfg.patch_boundary_violation_threshold,
                    )
                except Exception:
                    pass
            if p_col:
                any_collision = True

            fc, ec = ("red", "darkred") if p_col else patch_styles[pi % len(patch_styles)]
            ell = Ellipse(xy=(p.x, p.y), width=p.a * 2, height=p.b * 2,
                          angle=np.degrees(p.theta),
                          facecolor=fc, edgecolor=ec, alpha=0.35, linewidth=2.5, zorder=2)
            ax.add_patch(ell)
            self._vis_dynamic_artists.append(ell)

            for bx, by in p.get_boundary_points(20):
                ln, = ax.plot(bx, by, "k.", markersize=3, alpha=0.4, zorder=3)
                self._vis_dynamic_artists.append(ln)

            ln, = ax.plot(p.x, p.y, "b+", markersize=14, markeredgewidth=2, zorder=4)
            self._vis_dynamic_artists.append(ln)

            try:
                vx, vy = p.get_velocity_vector()
                ann = ax.annotate("", xy=(p.x + vx * 0.4, p.y + vy * 0.4), xytext=(p.x, p.y),
                                  arrowprops=dict(arrowstyle="->", color="blue", lw=2), zorder=5)
                self._vis_dynamic_artists.append(ann)
            except Exception:
                pass

            lane_id = self.lane_ids[pi] if pi < len(self.lane_ids) else "?"
            lane_str = {0: "Full", 1: "Left", 2: "Right"}.get(lane_id, str(lane_id))
            np_cnt = self.no_progress_list[pi] if pi < len(self.no_progress_list) else 0
            txt = ax.text(p.x + 0.3, p.y + 0.3,
                          f"P{pi} [{lane_str}]\nv={p.v:.1f}  b={p.b:.2f}\nnp={np_cnt}",
                          fontsize=8, color="navy", zorder=6,
                          bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
            self._vis_dynamic_artists.append(txt)

        # --- View limits ---
        view_margin = max(prim.a, prim.b) + 10
        ax.set_xlim(cx - view_margin, cx + view_margin)
        ax.set_ylim(cy - view_margin, cy + view_margin)

        # --- Title ---
        n_patches = len(self.active_patches)
        mode_str = "SPLIT" if n_patches > 1 else "FULL"
        ax.set_title(
            (f"PatchEnv | Step {self.step_count} | {mode_str} ({n_patches} patch)"
             f"  Progress {self.lap_progress:.1%}  Lap {self.lap_count}\n"
             f"v={prim.v:.2f}  a={prim.a:.2f}  b={prim.b:.2f}"),
            fontsize=10, color="red" if any_collision else "black",
        )

        # --- Info box ---
        info_lines = [
            f"Ep reward: {self.episode_reward:.1f}",
            f"Lap: {self.lap_count}",
            f"Step {self.step_count}/{self.cfg.max_steps}",
        ]
        txt = ax.text(0.01, 0.99, "\n".join(info_lines),
                      transform=ax.transAxes, fontsize=8, verticalalignment="top",
                      bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        self._vis_dynamic_artists.append(txt)

        try:
            ax.figure.canvas.draw()
            ax.figure.canvas.flush_events()
        except Exception:
            pass


    def close(self):
        if hasattr(self, '_fig') and self._fig is not None:
            import matplotlib.pyplot as plt
            plt.close(self._fig)
            self._fig = None
            self._ax = None
        self.f110.close()

    def get_patch_action(self, patch: "DynamicPatch", track_spline) -> np.ndarray:
        """
        Build an observation for `patch` and return the frozen policy's action.
        Called by AgentEnv.step() to drive the patch with the pre-trained policy.
        """
        # Sync track spline so _build_obs_for can compute Frenet coords
        if self.track_spline is None:
            self.track_spline = track_spline
            self.track_length = float(track_spline.s[-1])

        # Temporarily put the patch into active_patches so _build_obs_for works
        prev_patches = self.active_patches
        prev_lane_ids = self.lane_ids
        self.active_patches = [patch]
        self.lane_ids = [0]

        obs = self._build_obs_for(patch, lane_id=0)

        self.active_patches = prev_patches
        self.lane_ids = prev_lane_ids

        obs_input = obs[np.newaxis, :]  # (1, 11)
        if getattr(self, 'patch_vecnorm', None) is not None:
            obs_input = self.patch_vecnorm.normalize_obs(obs_input)

        action, _ = self.patch_policy.predict(obs_input, deterministic=True)
        return np.asarray(action).flatten()


class PatchCarEnv(PatchEnv):
    """
    Patch training with a real f110 car as the patch center (car 0).

    Same 11D observation and reward as PatchEnv, but pose/velocity come from
    `f110.step` instead of `DynamicPatch.step` bicycle integration. Ellipse
    shape (a, b) still uses `update_shape` with first-order lag.

    Kinematics for the primary patch match the single-car `ppo_experiment` wrapper:
    longitudinal speed from `linear_vels_x`, heading from `poses_theta`, and spin
    penalty from `ang_vels_z` (not finite-differenced heading).

    Episode termination for collisions uses only the patch boundary vs occupancy
    map (same as PatchEnv). F110 physics collision flags are exposed in info as
    `f110_collision` for debugging but do not end the episode or affect reward.
    """

    def _compute_max_steering_angle(self, b_cmd: float, ey: float, v: float, half_w: float) -> float:
        """
        Constrain steering angle based on patch boundary hitting track walls.

        The car IS the patch center. The patch extends ±b/2 from the car's position.
        The track extends ±half_w from the centerline.
        As the car steers, it drifts laterally, moving the patch boundaries with it.

        Args:
            b_cmd: current patch half-width (lateral extent from center)
            ey: lateral error (car position relative to centerline, signed: + = right, - = left)
            v: current longitudinal speed
            half_w: track half-width at current position

        Returns:
            max_steer: maximum safe steering angle in radians (prevents boundary hit)
        """
        wheelbase = self.cfg.wheelbase
        dt = self.cfg.control_dt
        safety_margin = 0.05  # 5cm buffer before patch boundary

        # Compute available margins
        # Patch left boundary at: ey + b/2, left wall at: +half_w
        # Patch right boundary at: ey - b/2, right wall at: -half_w
        margin_left = (half_w - ey) - (b_cmd / 2.0) - safety_margin
        margin_right = (ey + half_w) - (b_cmd / 2.0) - safety_margin

        # Ensure margins are non-negative
        margin_left = max(margin_left, 0.0)
        margin_right = max(margin_right, 0.0)

        # Stop speed case: no steering constraint
        if v < 0.01:
            return self.cfg.steering_max

        # For a bicycle model, steering angle relates to lateral drift via:
        # Δey ≈ v * sin(steer) * dt + 0.5 * (v²/wheelbase) * tan(steer) * dt²
        # For small angles: Δey ≈ v * steer * dt + 0.5 * (v²/wheelbase) * steer * dt²
        # Simplified: Δey ≈ (v * dt) * (1 + v*dt/(2*wheelbase)) * steer

        scale_factor = 1.0 + v * dt / (2.0 * wheelbase)
        drift_per_radian = v * dt * scale_factor

        # Max steering to stay within margins
        if drift_per_radian > 1e-3:
            max_steer_left = margin_left / drift_per_radian
            max_steer_right = margin_right / drift_per_radian
        else:
            return self.cfg.steering_max

        # Return the minimum constraint (most restrictive)
        max_steer = min(abs(max_steer_left), abs(max_steer_right))
        max_steer = np.clip(max_steer, 0.0, self.cfg.steering_max)

        return float(max_steer)

    def step(self, action):
        self.step_count += 1
        dt = self.cfg.control_dt
        lap_bonus = 0.0
        lap_time = None

        steering_cmd = float(np.clip(
            np.nan_to_num(action[0], nan=0.0, posinf=0.4189, neginf=-0.4189),
            -0.4189, 0.4189))
        speed_cmd = float(np.clip(
            np.nan_to_num(action[1], nan=1.5, posinf=10.0, neginf=1.5),
            1.5, 10.0))
        a_cmd = float(np.clip(
            np.nan_to_num(action[2], nan=self.cfg.a_cmd_min,
                          posinf=self.cfg.a_cmd_max, neginf=self.cfg.a_cmd_min),
            self.cfg.a_cmd_min, self.cfg.a_cmd_max))
        b_cmd = float(np.clip(
            np.nan_to_num(action[3], nan=self.cfg.b_cmd_min,
                          posinf=self.cfg.b_cmd_max, neginf=self.cfg.b_cmd_min),
            self.cfg.b_cmd_min, self.cfg.b_cmd_max))

        base_obs, _, _, _, _ = self.f110.step(
            np.array([[steering_cmd, speed_cmd]], dtype=np.float32)
        )
        # Store for _get_scan() — same pattern as ppo_experiment.py F1tenthWrapper
        self._current_base_obs = base_obs

        # Match test_PPO/.../ppo_experiment.py F1tenthWrapper: same scalars for pose/speed/spin.
        yaw = float(base_obs["poses_theta"][0]) if "poses_theta" in base_obs else 0.0
        longitudinal_speed = (
            float(base_obs["linear_vels_x"][0]) if "linear_vels_x" in base_obs else 0.0
        )
        yaw_rate_sim = (
            float(base_obs["ang_vels_z"][0]) if "ang_vels_z" in base_obs else 0.0
        )
        self._last_sim_yaw_rate = float(np.nan_to_num(yaw_rate_sim, nan=0.0))

        # === SINGLE PATCH ONLY (split/merge disabled) ===
        if len(self.active_patches) == 1:
            p0 = self.active_patches[0]
            p0.steering = steering_cmd
            p0.sync_from_pose(
                float(base_obs["poses_x"][0]),
                float(base_obs["poses_y"][0]),
                yaw,
                longitudinal_speed,
                steering_cmd,
            )
            s_i, _ = self._patch_to_frenet(p0)
            hw_i = self._lookup_half_w(s_i)
            max_b_i = min(self.cfg.b_cmd_max, hw_i * 0.90)
            p0.update_shape(a_cmd, b_cmd, dt,
                            max_a=self.cfg.a_cmd_max, max_b=max_b_i)
        else:
            # === MULTI-PATCH (SPLIT MODE) DISABLED FOR DEBUGGING ===
            # ego = self.active_patches[0]
            # other = self.active_patches[1]
            # ego.steering = steering_cmd
            # ego.sync_from_pose(
            #     float(base_obs["poses_x"][0]),
            #     float(base_obs["poses_y"][0]),
            #     yaw,
            #     longitudinal_speed,
            #     steering_cmd,
            # )
            # s_i, _ = self._patch_to_frenet(ego)
            # hw_i = self._lookup_half_w(s_i)
            # max_b_i = min(self.cfg.b_cmd_max, hw_i * 0.90)
            # ego.update_shape(a_cmd, b_cmd, dt,
            #                  max_a=self.cfg.a_cmd_max, max_b=max_b_i)
            # self._sync_follower_zone_from_spline(
            #     ego, other, self.lane_ids[1], steering_cmd, dt, a_cmd, b_cmd,
            # )
            pass  # Should never reach here since split/merge is disabled

        f110_collision = False
        if "collisions" in base_obs:
            c = base_obs["collisions"]
            f110_collision = bool(np.asarray(c).reshape(-1)[0] > 0.5)

        primary = self.active_patches[0]
        s_now, ey_now = self._patch_to_frenet(primary)
        if self.track_spline is not None and self.track_length is not None and self.track_length > 0.0:
            self.lap_progress = float(np.clip(s_now / self.track_length, 0.0, 1.0))
            prev_s_primary = self.prev_s_list[0]
            if prev_s_primary is not None:
                raw_ds = float(s_now - prev_s_primary)
                wrapped_ds = float(self._wrap_ds(raw_ds))
                if raw_ds < -0.5 * float(self.track_length) and wrapped_ds > 0.0:
                    self.lap_count += 1
                    lap_time = (self.step_count - self.lap_start_step) * dt
                    lap_time = max(float(lap_time), 1e-3)
                    lap_bonus = self.cfg.lap_finish_bonus * float(
                        np.exp(-lap_time / max(self.cfg.lap_bonus_tau, 1e-6))
                    )
                    self.lap_start_step = self.step_count

        # === SPLIT/MERGE DISABLED FOR DEBUGGING ===
        # if self.active_patches:
        #     self._build_obs_for(self.active_patches[0], self.lane_ids[0])
        # if self.cfg.split_mode:
        #     in_split = (len(self.active_patches) > 1)
        #     if not in_split and self._last_ahead_half_w < self.cfg.split_ahead_half_w:
        #         self._do_split()
        #     elif in_split and self._last_ahead_half_w > self.cfg.merge_ahead_half_w:
        #         self._do_merge()

        total_reward = 0.0
        any_terminated = False
        termination_reason = None
        all_terms = {}

        for i, (patch, lane_id) in enumerate(zip(self.active_patches, self.lane_ids)):
            patch_hit = self._collision_for(patch)
            bonus_i = lap_bonus if i == 0 else 0.0
            sim_yr = getattr(self, "_last_sim_yaw_rate", None) if i == 0 else None
            r, new_s, new_steer, new_no_progress, terms = self._compute_reward_for(
                patch, lane_id,
                self.prev_s_list[i], self.prev_steer_list[i],
                self.no_progress_list[i],
                dt, lap_bonus=bonus_i,
                cached_collision=patch_hit,
                sim_yaw_rate=sim_yr,
            )
            self.prev_s_list[i] = new_s
            self.prev_steer_list[i] = new_steer
            self.no_progress_list[i] = new_no_progress

            total_reward += r

            terminated_i, reason_i = self._check_termination_for(
                new_no_progress, patch_hit
            )
            if terminated_i and not any_terminated:
                any_terminated = True
                termination_reason = reason_i

            for k, v in terms.items():
                all_terms[f"p{i}_{k}"] = v

        n = max(len(self.active_patches), 1)
        reward = float(np.nan_to_num(total_reward / n, nan=-10.0, posinf=100.0, neginf=-100.0))
        self.episode_reward += reward

        truncated = False
        if self.step_count >= self.cfg.max_steps:
            truncated = True
            termination_reason = termination_reason or "max_steps"

        obs = self._build_obs()
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

        info = {
            "episode_reward":     self.episode_reward,
            "lap_progress":       self.lap_progress,
            "Episode_steps":      self.step_count,
            "step_reward":        reward,
            "termination_reason": termination_reason,
            "lap_count":          int(self.lap_count),
            "lap_bonus":          float(lap_bonus),
            "lap_time":           None if lap_time is None else float(lap_time),
            "frenet_s":           float(s_now),
            "frenet_ey":          float(ey_now),
            "num_active_patches": len(self.active_patches),
            "lane_ids":           list(self.lane_ids),
            "primary_b":          float(self.active_patches[0].b),
            "f110_collision":   bool(f110_collision),
        }
        info.update(all_terms)

        return obs, reward, any_terminated, truncated, info


# ---------------------------------------------------------------------------
# CTDE policy for multi-agent PPO (MAPPO)
# ---------------------------------------------------------------------------
# if SB3_AVAILABLE:
#     import torch as th
#     import torch.nn as nn
#     from stable_baselines3.common.policies import ActorCriticPolicy
#     from stable_baselines3.common.type_aliases import Schedule as _Schedule

#     class CTDEMlpExtractor(nn.Module):
#         """MLP extractor for Centralized Training, Decentralized Execution.

#         Obs layout (24D):
#             obs[:12]  = ego agent's 12D observation  → used by actor
#             obs[12:]  = partner agent's 12D obs      → concatenated for critic

#         Actor branch: processes only the ego slice (decentralized execution).
#         Critic branch: processes the full 24D joint state (centralized training).
#         """

#         EGO_DIM   = 12
#         JOINT_DIM = 24

#         def __init__(self, hidden: int = 256):
#             super().__init__()
#             self.latent_dim_pi = hidden
#             self.latent_dim_vf = hidden

#             # Decentralized actor — ego obs only
#             self.policy_net = nn.Sequential(
#                 nn.Linear(self.EGO_DIM, hidden), nn.Tanh(),
#                 nn.Linear(hidden, hidden),        nn.Tanh(),
#             )
#             # Centralized critic — full joint obs
#             self.value_net = nn.Sequential(
#                 nn.Linear(self.JOINT_DIM, hidden), nn.Tanh(),
#                 nn.Linear(hidden, hidden),          nn.Tanh(),
#             )

#         def forward(self, features: th.Tensor):
#             return self.forward_actor(features), self.forward_critic(features)

#         def forward_actor(self, features: th.Tensor) -> th.Tensor:
#             return self.policy_net(features[:, :self.EGO_DIM])

#         def forward_critic(self, features: th.Tensor) -> th.Tensor:
#             return self.value_net(features)

#     class MAPPOPolicy(ActorCriticPolicy):
#         """SB3-compatible PPO policy with CTDE.

#         Drop-in replacement for 'MlpPolicy' in PPO():
#             model = PPO(MAPPOPolicy, env, ...)
#         """

#         def _build_mlp_extractor(self) -> None:
#             self.mlp_extractor = CTDEMlpExtractor(hidden=256)


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
            random_spawn=True,  # full-track coverage for patch policy
            # split_mode=False,    # === SPLIT/MERGE DISABLED FOR DEBUGGING ===
            # navigation_mode=navigation_mode,
            # debug_print_every_n_steps=debug_print_every_n_steps,
            # debug_print_episode_end=debug_print_episode_end,
            base_reset_type=base_reset_type,
            # use_base_done_termination=use_base_done_termination,
            # patch_only_mode=patch_only_mode,
        )
        env = PatchCarEnv(cfg)
        env.reset(seed=seed + rank)
        return env

    if SB3_AVAILABLE:
        set_random_seed(seed + rank)
    return _init


# def make_agent_env(
#     rank: int,
#     seed: int = 0,
#     patch_env=None, ):
#     def _init():
#         cfg = AgentEnvConfig(
#             patch_env=patch_env,
#             render_mode=None,
#             random_spawn=False,
#         )
#         env = AgentEnv(cfg)
#         env.reset(seed=seed + rank)
#         return env

#     if SB3_AVAILABLE:
#         set_random_seed(seed + rank)
#     return _init


# def make_agent_views(
#     rank: int,
#     seed: int = 0,
#     patch_env=None, ) -> tuple:
#     """Return a pair of (thunk_for_agent0, thunk_for_agent1) that share one AgentEnv.

#     Usage in training.py:
#         env_fns = []
#         for i in range(NUM_ENVS):
#             fn0, fn1 = make_agent_views(i, seed=42, patch_env=patch_env)
#             env_fns.append(fn0)
#             env_fns.append(fn1)
#         vec_env = DummyVecEnv(env_fns)

#     This creates 2*NUM_ENVS slots. Slots come in consecutive pairs that share
#     physics — agent0's slot is always stepped before agent1's slot within each
#     DummyVecEnv.step(), so agent0 submits first and agent1 triggers the physics.
#     """
#     if SB3_AVAILABLE:
#         set_random_seed(seed + rank)

#     # Build the shared backend once; both thunks close over it.
#     cfg = AgentEnvConfig(
#         patch_env=patch_env,
#         render_mode=None,
#         random_spawn=False,
#     )
#     shared_env = AgentEnv(cfg)
#     shared_env.reset(seed=seed + rank)

#     def _view0():
#         return AgentView(shared_env, agent_idx=0)

#     def _view1():
#         return AgentView(shared_env, agent_idx=1)

#     return _view0, _view1


# def make_joint_views(rank: int, seed: int = 0, cfg: "JointEnvConfig" = None) -> tuple:
#     """Create one JointEnv and return (shared_env, patch_thunk, agent0_thunk, agent1_thunk).

#     Usage in training.py:
#         shared_envs, patch_fns, agent_fns = [], [], []
#         for i in range(NUM_ENVS):
#             env, pf, af0, af1 = make_joint_views(i, seed=42)
#             shared_envs.append(env)
#             patch_fns.append(pf)
#             agent_fns.extend([af0, af1])

#         patch_vec_env = DummyVecEnv(patch_fns)   # N slots — one per JointEnv
#         agent_vec_env = DummyVecEnv(agent_fns)   # 2N slots — two per JointEnv
#     """
#     if SB3_AVAILABLE:
#         set_random_seed(seed + rank)
#     if cfg is None:
#         cfg = JointEnvConfig()
#     shared_env = JointEnv(cfg)
#     shared_env.reset(seed=seed + rank)

#     def _patch():  return JointPatchView(shared_env)
#     def _agent0(): return JointAgentView(shared_env, agent_idx=1)
#     def _agent1(): return JointAgentView(shared_env, agent_idx=2)

#     return shared_env, _patch, _agent0, _agent1



