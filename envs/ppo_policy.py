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
    #  obstacle_coords = 
    
    # Vehicle dynamics
    wheelbase: float = 0.33
    robot_radius: float = 0.15
    
    # Patch size at spawn
    patch_a: float = 3.0  # semi-major axis (length)
    patch_b: float = 1.2  # semi-minor axis (width)

    # b_cmd / a_cmd action bounds
    # b_cmd_min = 1.0 because 2 agents need at least b=1.0 to fit side-by-side in the patch
    b_cmd_min: float = 1.0
    b_cmd_max: float = 1.5
    a_cmd_min: float = 1.5
    a_cmd_max: float = 3.0

    shape_area_penalty_weight: float = 0.1
    random_spawn: bool = False

    # Patch boundary collision detection
    patch_boundary_violation_threshold: float = 0.05

    # Lookahead distance (metres ahead along spline) — used for proactive split/merge
    lookahead_dist: float = 6.0
    split_mode: bool = False

    # Split/merge triggered by LOOKAHEAD track half-width, not by b value.
    # ahead_half_w < split threshold → narrow section coming → split now
    # ahead_half_w > merge threshold → wide section coming → merge now
    split_ahead_half_w: float = 0.8   # m — split when less than this half-width ahead
    merge_ahead_half_w: float = 1.1   # m — merge when more than this half-width ahead
    lane_centering_weight: float = 2.0
    

@dataclass 
class AgentEnvConfig:
    control_dt: float = 0.05
    num_agents: int = 2
    patch_env: Optional[Any] = None
    # base_reset_type: str = "rl_random_static"
    render_mode: Optional[str] = None
    random_spawn: bool = False
    max_steps: int = 100_000
    patch_a: float = 3.0   # must match PatchEnvConfig.patch_a — frozen policy was trained with this
    patch_b: float = 1.2   # must match PatchEnvConfig.patch_b

    wheelbase: float = 0.33
    robot_radius: float = 0.15

    out_of_patch_penalty: float = 200.0
    inter_collision_penalty: float = 150.0
    inside_patch_reward: float = 30.0
    survival_reward_per_step: float = 2.0  # bonus per step alive inside patch

    agents_random_spawn: bool = False
    # lap_finish_bonus: float = 2000.0
    # lap_bonus_tau: float = 200.0

    inter_agent_collision_dist: float = 0.20
    patch_boundary_violation_threshold: float = 0.10  # fraction of boundary pts in wall → collision


@dataclass
class JointEnvConfig:
    """Config for joint training: patch + 2 agents trained simultaneously."""
    control_dt: float = 0.05
    render_mode: Optional[str] = None
    random_spawn: bool = True
    max_steps: int = 100_000
    patch_a: float = 3.0
    patch_b: float = 1.5
    # Agent reward — large internal values, clipped to [-100, 100] (same style as PatchEnv)
    out_of_patch_penalty: float = 200.0
    inter_collision_penalty: float = 150.0
    inside_patch_reward: float = 15.0
    repulsion_zone: float = 2.0
    repulsion_weight: float = 30.0
    survival_reward_per_step: float = 2.0
    inter_agent_collision_dist: float = 0.50
    patch_boundary_violation_threshold: float = 0.02  # 1/64 points → immediate detection
    # Patch reward — mirrors PatchEnv._compute_reward_for exactly + JointEnv-specific terms
    reward_progress_scale: float = 40.0          # same as PatchEnv
    reward_crosstrack_weight: float = 2.0        # same as PatchEnv
    reward_steer_bias_weight: float = 0.0        # same as PatchEnv
    reward_steer_rate_weight: float = 0.0        # same as PatchEnv
    spin_yawrate_threshold: float = 1.5          # same as PatchEnv
    reward_spin_weight: float = 0.5              # same as PatchEnv
    patch_wall_penalty: float = 1500.0           # same as PatchEnv collision_penalty
    stuck_no_progress_steps: int = 60            # same as PatchEnv
    stuck_progress_eps: float = 1e-3             # same as PatchEnv
    stuck_penalty: float = 10.0                  # same as PatchEnv
    patch_lap_bonus: float = 2000.0              # same as PatchEnv lap_finish_bonus
    wheelbase: float = 0.33                      # same as PatchEnv
    # JointEnv-specific: agent coupling terms
    agents_inside_bonus: float = 30.0            # bonus per agent inside per step
    patch_min_speed_frac: float = 0.3
    patch_boundary_violation_threshold: float = 0.05
    # Patch obs / action bounds
    patch_lookahead_dist: float = 2.0
    patch_a_cmd_min: float = 1.5
    patch_b_cmd_min: float = 1.0


class AgentEnv(gym.Env):
    "Agent enviroment which is using the patch as the place to navigate"

    metadata = { "render_modes" : ["human", "rgb_array", None]}

    def __init__(self, config: Optional[AgentEnvConfig] = None):
        super().__init__()
        self.cfg = config or AgentEnvConfig()
        self.patch_env = self.cfg.patch_env
        # Action: [steer_delta, speed_delta]
        # steer_delta — correction on top of patch co-driver steer (keeps agents
        #               stable during early training; random deltas stay aligned)
        # speed_delta — correction on top of frozen patch speed baseline
        self.action_space = spaces.Box(
            low=np.array([-0.4189, -1.0], dtype=np.float32),
            high=np.array([0.4189,  1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # Obs: 24D — ego 12D concatenated with partner 12D (for CTDE centralized critic)
        # obs[0:12]  = ego agent's 12D obs (actor only uses this slice)
        # obs[12:24] = partner agent's 12D obs (critic uses full 24D)
        # Layout of each 12D block:
        # [x_norm, y_norm,            ← position in patch
        #  ot_x_norm, ot_y_norm,      ← other agent position
        #  dx_other,  dy_other,       ← separation vector
        #  patch_yaw_rate, patch_v,   ← patch dynamics
        #  a, b, speed, heading_rel]  ← shape + own kinematics
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32,
        )
        self._ego_idx = 0  # randomised each reset — breaks agent0/agent1 asymmetry

        # 3-car simulation: car 0 = patch (frozen policy, real f110 inertia),
        # cars 1 & 2 = learning agents.
        # Using real f110 physics for the patch restores the inertia it had during
        # PatchEnv training — the kinematic DynamicPatch had no inertia and sprinted
        # ahead of the physical agent cars during the acceleration phase.
        self.f110 = F110EnvAdapter(
            F110Config(
                num_agents=3,
                control_input=("speed", "steering_angle"),
                timestep=self.cfg.control_dt,
            ),
            render_mode=self.cfg.render_mode,
        )
        self.base_env = self.f110
        self.map_reset = MapResetHelper()

        # DynamicPatch is synced from f110 car 0's physical position each step.
        # Used only for geometry: world_to_patch_frame, check_boundary_collision,
        # update_shape, and obs building. No longer stepped kinematically.
        self.patch = DynamicPatch()
        self.patch.a = self.cfg.patch_a
        self.patch.b = self.cfg.patch_b

        self.num_agents = 2  # learning agents (f110 cars 1 and 2)
        self.track_spline = None
        self.track_length = None
        self._occ_map = None
        self._resolution = None
        self._origin = None
        self.step_count = 0
        self.episode_reward = 0.0
        self.current_base_obs = None
        self._fig = None
        self._ax = None
        self._term_counts = {"out_of_patch": 0, "inter_collision": 0, "max_steps": 0, "patch_wall": 0}
        self.prev_patch_theta = 0.0
        self.patch_yaw_rate = 0.0
        self.patch_steer = 0.0
        self._episode_count = 0
        self.prev_dist_norms = [0.0, 0.0]  # tracked for shaped reward

        # --- AgentView synchronization (parameter sharing) ---
        # Two AgentView instances share this env. Each independently predicts
        # a 2D action. Physics executes once BOTH actions are submitted.
        self._pending_actions: list = [None, None]
        self._step_obs:        list = [None, None]
        self._step_rewards:    list = [0.0, 0.0]
        self._step_terminated: bool = False
        self._step_truncated:  bool = False
        self._step_info:       dict = {}


    #     # Frenet bookkeeping - per agent
    #     self.track_spline = None
    #     self.track_length = None
    #     self.prev_rel = [None, None]
    #     self.no_progress_counter = [0, 0]
    #     self.step_count = 0
    #     self.episode_reward = 0.0
    #     self._last_reward_terms = {}
    #     self.current_base_obs = None
    #     self.num_agents = 2
    #     self._fig = None
    #     self._ax = None
    
    # # def _wrap_ds(self, ds: float) -> float:
    # #     if self.track_length is None or not np.isfinite(self.track_length):
    # #         return ds  # Safe fallback
    # #     L = float(self.track_length)
    # #     if L <= 0:
    # #         return ds
    # #     return (ds + 0.5 * L) % L - 0.5 * L

    # @staticmethod
    # def _wrap_angle(angle_rad: float) -> float:
    #     return float((angle_rad + np.pi) % (2.0 * np.pi) - np.pi)

    # def _patch_to_frenet(self) -> tuple[float, float]:
    #     if self.track_spline is None:
    #         # Return safe fallback values (like single-car would never reach here)
    #         return 0.0, 0.0
        
    #     try:
    #         s, ey = self.track_spline.calc_arclength_inaccurate(float(self.patch.x), float(self.patch.y))
    #         # Check for NaN (single-car doesn't need this but multi-agent might)
    #         if not np.isfinite(s) or not np.isfinite(ey):
    #             return 0.0, 0.0
    #         return float(s), float(ey)
    #     except (AttributeError, TypeError):
    #         return 0.0, 0.0

    # def _estimate_current_track_width(self) -> float:
    #     """Estimate local track width at current patch pose."""
    #     try:
    #         _, occ_map, resolution, origin = self.f110.get_track_data()
    #         width = float(
    #             self.map_reset.estimate_track_width(
    #                 occ_map,
    #                 resolution,
    #                 origin,
    #                 float(self.patch.x),
    #                 float(self.patch.y),
    #                 float(self.patch.theta),
    #             )
    #         )
    #         if not np.isfinite(width) or width <= 0.0:
    #             return float(max(2.0 * self.patch.b, 1.0))
    #         return width
    #     except Exception:
    #         return float(max(2.0 * self.patch.b, 1.0))

    # def _agent_to_frenet(self, x: float, y: float) -> tuple[float, float]:
    #     """Convert agent world position to Frenet (s, ey)."""
    #     if self.track_spline is None:
    #         return 0.0, 0.0
    #     try:
    #         s, ey = self.track_spline.calc_arclength_inaccurate(float(x), float(y))
    #         if not np.isfinite(s) or not np.isfinite(ey):
    #             return 0.0, 0.0
    #         return float(s), float(ey)
    #     except (AttributeError, TypeError):
    #         return 0.0, 0.0
    
    def _sync_patch_from_car0(self, base_obs) -> None:
        """Sync DynamicPatch geometry from f110 car 0 (the physical patch car).
        Called after every f110.reset() and f110.step() so the patch ellipse
        always reflects the real physical position of the patch car.
        """
        self.patch.x = float(base_obs["poses_x"][0])
        self.patch.y = float(base_obs["poses_y"][0])
        self.patch.theta = float(base_obs["poses_theta"][0])
        vx = float(base_obs["linear_vels_x"][0]) if "linear_vels_x" in base_obs else 0.0
        vy = float(base_obs["linear_vels_y"][0]) if "linear_vels_y" in base_obs else 0.0
        self.patch.v = float(np.hypot(vx, vy))

    def _build_agent_obs_for(self, ego: int, base_obs) -> np.ndarray:
        """Build 12D ego-centric observation for logical agent `ego` (0 or 1).
        Logical agent 0 → f110 car 1, logical agent 1 → f110 car 2.
        (f110 car 0 is reserved for the patch car.)
        """
        other = 1 - ego
        # Map logical agent index to f110 car index
        f_ego  = ego  + 1   # f110 index for ego agent
        f_other = other + 1  # f110 index for other agent
        a = max(float(self.patch.a), 1e-3)
        b = max(float(self.patch.b), 1e-3)

        def _get_patch_pos(fi):
            x = float(base_obs["poses_x"][fi])
            y = float(base_obs["poses_y"][fi])
            return self.patch.world_to_patch_frame(x, y)

        def _get_speed(fi):
            vx = float(base_obs["linear_vels_x"][fi]) if "linear_vels_x" in base_obs else 0.0
            vy = float(base_obs["linear_vels_y"][fi]) if "linear_vels_y" in base_obs else 0.0
            return float(np.hypot(vx, vy))

        def _get_heading_rel(fi):
            yaw = float(base_obs["poses_theta"][fi]) if "poses_theta" in base_obs else 0.0
            return float((yaw - self.patch.theta + np.pi) % (2 * np.pi) - np.pi)

        # Ego agent
        ex, ey = _get_patch_pos(f_ego)
        my_x_norm, my_y_norm = ex / b, ey / a

        # Other agent
        ox, oy = _get_patch_pos(f_other)
        ot_x_norm, ot_y_norm = ox / b, oy / a

        # Separation from ego's perspective
        dx_other = ex - ox
        dy_other = ey - oy

        patch_yaw_rate = float(np.clip(self.patch_yaw_rate, -10.0, 10.0))
        patch_v = float(self.patch.v)

        return np.array([
            my_x_norm, my_y_norm,
            ot_x_norm, ot_y_norm,
            dx_other, dy_other,
            patch_yaw_rate, patch_v, a, b,
            _get_speed(f_ego), _get_heading_rel(f_ego),
        ], dtype=np.float32)

    def _build_agent_obs(self, base_obs) -> np.ndarray:
        """Backward-compat wrapper: builds obs for the current _ego_idx."""
        return self._build_agent_obs_for(self._ego_idx, base_obs)

    def _compute_agent_reward(self, dist_norm: float, prev_dist_norm: float,
                              inter_collision: bool) -> float:
        """Per-agent reward with position shaping + progress shaping.
        dist_norm: 0=center, 1=patch edge, >1=outside.
        prev_dist_norm: dist_norm from previous step — used for shaped reward.
        """
        if dist_norm <= 1.0:
            # Inside: position reward highest at center + survival bonus
            reward = self.cfg.inside_patch_reward * float(np.exp(-2.0 * dist_norm ** 2))
            # reward += self.cfg.survival_reward_per_step
            # Shaped: reward for moving toward center, penalise moving away
            # prev - curr > 0 means closer than before → positive
            reward += 5.0 * (prev_dist_norm - dist_norm)
        elif dist_norm <= 1.5:
            # Buffer zone: escalating penalty, still alive
            reward = -self.cfg.out_of_patch_penalty * (dist_norm - 1.0) / 0.5
        else:
            # Hard outside: full penalty
            reward = -self.cfg.out_of_patch_penalty
        if inter_collision:
            reward -= self.cfg.inter_collision_penalty
        return float(np.clip(reward, -200.0, 200.0))

    def _check_agent_termination(
        self, agent_idx: int, dist_norm: float, inter_collision: bool
    ) -> tuple[bool, bool, str]:
        """Terminate only when dist_norm > 1.5 (buffer zone gives gradient before cliff)."""
        if inter_collision:
            return True, False, f"agent{agent_idx}_inter_collision"
        if dist_norm > 1.5:
            return True, False, f"agent{agent_idx}_out_of_patch"
        if self.step_count >= self.cfg.max_steps:
            return False, True, "max_steps"
        return False, False, None
    
    def reset(self, seed=None, options=None):
        """Reset: spawn 2 agents inside patch, initialize frozen patch policy."""
        super().reset(seed=seed)
        self.step_count = 0
        self.episode_reward = 0.0
        self.prev_dist_norms = [0.0, 0.0]
        # Randomise ego agent each episode — both agents train equally as "ego"
        self._ego_idx = int(np.random.randint(0, 2))

        self.f110.ensure_initialized()
        track, self._occ_map, self._resolution, self._origin = self.f110.get_track_data()
        if track.centerline is None or track.centerline.spline is None:
            raise ValueError("Track centerline/spline missing.")

        self.track_spline = track.centerline.spline
        self.track_length = float(self.track_spline.s[-1])

        # Sync frozen patch_env with track data so get_patch_action builds correct obs.
        # PatchEnv.reset() is never called in AgentEnv — so we must seed its caches here.
        # _precompute_track_widths is expensive (200 ray casts) — only do it once per env,
        # not every episode reset.
        if self.patch_env is not None:
            self.patch_env._occ_map    = self._occ_map
            self.patch_env._resolution = self._resolution
            self.patch_env._origin     = self._origin
            self.patch_env.track_spline  = self.track_spline
            self.patch_env.track_length  = self.track_length
            if self.patch_env._tw_s_vals is None:
                self.patch_env._precompute_track_widths(n_samples=200)

        xs = np.asarray(track.centerline.xs, dtype=np.float32)
        ys = np.asarray(track.centerline.ys, dtype=np.float32)
        spawn_idx = int(np.random.randint(0, xs.shape[0])) if self.cfg.random_spawn else 0
        spawn_idx = max(0, min(spawn_idx, xs.shape[0] - 1))
        patch_x, patch_y = float(xs[spawn_idx]), float(ys[spawn_idx])
        next_idx = (spawn_idx + 1) % xs.shape[0]
        patch_theta = float(np.arctan2(
            float(ys[next_idx] - ys[spawn_idx]),
            float(xs[next_idx] - xs[spawn_idx])
        ))

        self.patch = DynamicPatch(
            x=patch_x, y=patch_y, theta=patch_theta,
            v=0.5, a=self.cfg.patch_a, b=self.cfg.patch_b
        )
        self.prev_patch_theta = patch_theta
        self.patch_yaw_rate = 0.0
        self.patch_steer = 0.0
        if self.patch_env is not None:
            self.patch_env.patch = self.patch
        perp_dx = -np.sin(patch_theta)
        perp_dy = np.cos(patch_theta)

        if self.cfg.agents_random_spawn:
            a = max(float(self.patch.a), 1e-3)
            b = max(float(self.patch.b), 1e-3)
            min_sep = 0.30
            max_tries = 50
            cos_t, sin_t = np.cos(patch_theta), np.sin(patch_theta)

            def _sample_in_ellipse():
                for _ in range(200):
                    xr = np.random.uniform(-b * 0.5, b * 0.5)
                    yr = np.random.uniform(-a * 0.5, a * 0.5)
                    if (xr / b) ** 2 + (yr / a) ** 2 < 0.25:
                        wx = patch_x + cos_t * yr - sin_t * xr
                        wy = patch_y + sin_t * yr + cos_t * xr
                        return wx, wy
                return patch_x, patch_y

            for _ in range(max_tries):
                w0x, w0y = _sample_in_ellipse()
                w1x, w1y = _sample_in_ellipse()
                if np.hypot(w0x - w1x, w0y - w1y) >= min_sep:
                    break

            poses = np.array([
                [patch_x, patch_y, patch_theta],  # car 0 = patch car, at center
                [w0x, w0y, patch_theta],           # car 1 = agent 0
                [w1x, w1y, patch_theta],           # car 2 = agent 1
            ], dtype=np.float32)
        else:
            offset = 0.50
            poses = np.array([
                [patch_x, patch_y, patch_theta],                                              # car 0 = patch car
                [patch_x + offset * perp_dx, patch_y + offset * perp_dy, patch_theta],       # car 1 = agent 0 (left)
                [patch_x - offset * perp_dx, patch_y - offset * perp_dy, patch_theta],       # car 2 = agent 1 (right)
            ], dtype=np.float32)

        base_obs, _ = self.f110.reset(poses=poses)
        self.current_base_obs = base_obs

        # Sync DynamicPatch geometry from the physical patch car (f110 car 0)
        self._sync_patch_from_car0(base_obs)

        # Pre-cache 24D joint obs for CTDE: [ego_12D | partner_12D]
        ego0 = np.nan_to_num(self._build_agent_obs_for(0, base_obs),
                              nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        ego1 = np.nan_to_num(self._build_agent_obs_for(1, base_obs),
                              nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        self._step_obs[0] = np.concatenate([ego0, ego1])  # agent0: ego=0, partner=1
        self._step_obs[1] = np.concatenate([ego1, ego0])  # agent1: ego=1, partner=0
        self._pending_actions = [None, None]
        self._step_rewards    = [0.0, 0.0]
        self._step_terminated = False
        self._step_truncated  = False
        self._step_info       = {}

        obs = self._step_obs[self._ego_idx]  # 24D joint obs already built above
        return obs.copy(), {}

    def _execute_combined_step(self, action0: np.ndarray, action1: np.ndarray):
        """Execute one physics step with independent actions for each agent.

        action0: [steer_delta, speed_delta] for physical agent 0
        action1: [steer_delta, speed_delta] for physical agent 1

        Updates _step_obs, _step_rewards, _step_terminated, _step_truncated,
        _step_info in-place. Called by AgentView when both actions are ready.
        """
        self.step_count += 1
        dt = self.cfg.control_dt

        # --- Frozen patch policy acts on DynamicPatch state (synced from car 0) ---
        if self.patch_env is not None:
            self.patch_env.patch = self.patch
            patch_action = self.patch_env.get_patch_action(self.patch, self.track_spline)
            steer_p = float(np.clip(patch_action[0], -0.4189, 0.4189))
            speed_p = float(np.clip(patch_action[1], 0.5, 10.0))
            a_p = float(np.clip(patch_action[2], 1.0, self.cfg.patch_a))
            b_p = float(np.clip(patch_action[3], 1.0, self.cfg.patch_b))
        else:
            steer_p, speed_p, a_p, b_p = 0.0, 2.0, self.cfg.patch_a, self.cfg.patch_b

        # --- Decode agent actions (steer/speed deltas on top of patch baseline) ---
        def _decode(raw):
            sd = float(np.nan_to_num(raw[0], nan=0.0, posinf=0.4189, neginf=-0.4189))
            vd = float(np.nan_to_num(raw[1], nan=0.0))
            s  = float(np.clip(steer_p + sd, -0.4189, 0.4189))
            v  = float(np.clip(speed_p + vd, 0.5, 10.0))
            return s, v

        s0, v0 = _decode(action0)
        s1, v1 = _decode(action1)

        # --- Step all 3 cars together ---
        # car 0 = patch car (frozen policy), cars 1,2 = learning agents
        base_obs, _, _, _, _ = self.f110.step(
            np.array([[steer_p, speed_p],   # patch car: frozen policy command
                      [s0, v0],             # agent 0
                      [s1, v1]], dtype=np.float32)  # agent 1
        )
        self.current_base_obs = base_obs

        # --- Sync patch geometry from car 0's physical position (with inertia) ---
        prev_theta = self.patch.theta
        self._sync_patch_from_car0(base_obs)
        d_theta = (self.patch.theta - prev_theta + np.pi) % (2 * np.pi) - np.pi
        self.patch_yaw_rate = d_theta / dt

        # Update patch shape (a, b can grow/shrink per policy output)
        self.patch.update_shape(a_p, b_p, dt, max_a=self.cfg.patch_a, max_b=self.cfg.patch_b)

        # --- Patch wall collision (uses real physical patch position now) ---
        patch_wall_hit = False
        if self._occ_map is not None:
            patch_wall_hit, _ = self.patch.check_patch_boundary_wall_collision(
                self._occ_map, self._resolution, self._origin,
                n_points=32,
                violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )

        # --- Per-agent inside-patch check (logical agents 0,1 = f110 cars 1,2) ---
        dist_norm_list = []
        for i in range(2):
            fi = i + 1  # f110 car index for logical agent i
            x = float(base_obs["poses_x"][fi])
            y = float(base_obs["poses_y"][fi])
            x_rel, y_rel = self.patch.world_to_patch_frame(x, y)
            dist_norm = float(np.sqrt(
                (x_rel / max(self.patch.a, 1e-3)) ** 2 +
                (y_rel / max(self.patch.b, 1e-3)) ** 2
            ))
            dist_norm_list.append(dist_norm)

        # Inter-agent distance (cars 1 and 2)
        x0w, y0w = float(base_obs["poses_x"][1]), float(base_obs["poses_y"][1])
        x1w, y1w = float(base_obs["poses_x"][2]), float(base_obs["poses_y"][2])
        inter_dist = float(np.hypot(x0w - x1w, y0w - y1w))
        inter_collision = inter_dist < self.cfg.inter_agent_collision_dist

        # --- Per-agent reward and termination ---
        total_reward = 0.0
        terminated_flags = []
        reasons = []
        for i in range(2):
            r = self._compute_agent_reward(dist_norm_list[i], self.prev_dist_norms[i], inter_collision)
            term, trunc, reason = self._check_agent_termination(i, dist_norm_list[i], inter_collision)
            self._step_rewards[i] = float(np.clip(r, -200.0, 200.0))
            total_reward += r
            terminated_flags.append(term or trunc)
            reasons.append(reason)
        self.prev_dist_norms = list(dist_norm_list)

        total_reward = float(np.clip(total_reward / 2.0, -200.0, 200.0))
        self.episode_reward += total_reward
        terminated = any(terminated_flags) or patch_wall_hit
        truncated  = self.step_count >= self.cfg.max_steps
        if patch_wall_hit:
            reason = "patch_wall_collision"
        else:
            reason = next((r for r in reasons if r is not None), "max_steps" if truncated else None)

        # --- Build 24D joint obs for CTDE: [ego_12D | partner_12D] ---
        ego0 = np.nan_to_num(self._build_agent_obs_for(0, base_obs),
                              nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        ego1 = np.nan_to_num(self._build_agent_obs_for(1, base_obs),
                              nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        self._step_obs[0] = np.concatenate([ego0, ego1])
        self._step_obs[1] = np.concatenate([ego1, ego0])

        if terminated or truncated:
            self._episode_count += 1
            if reason == "patch_wall_collision":
                self._term_counts["patch_wall"] += 1
            elif reason and "inter_collision" in reason:
                self._term_counts["inter_collision"] += 1
            elif reason and "out_of_patch" in reason:
                self._term_counts["out_of_patch"] += 1
            else:
                self._term_counts["max_steps"] += 1
            if self._episode_count % 200 == 0:
                n = self._episode_count
                print(
                    f"[AgentEnv ep={n}] out_of_patch={self._term_counts['out_of_patch']/n*100:.1f}%  "
                    f"inter_collision={self._term_counts['inter_collision']/n*100:.1f}%  "
                    f"patch_wall={self._term_counts['patch_wall']/n*100:.1f}%  "
                    f"max_steps={self._term_counts['max_steps']/n*100:.1f}%  "
                    f"| now: inter_dist={inter_dist:.3f}m  patch_v={self.patch.v:.2f}"
                )

        self._step_terminated = terminated
        self._step_truncated  = truncated
        self._step_info = {
            "episode_reward": self.episode_reward,
            "Episode_steps": self.step_count,
            "step_reward": total_reward,
            "termination_reason": reason,
            "inter_agent_dist": float(inter_dist),
            "agent0_speed": float(v0),
            "agent1_speed": float(v1),
        }

    def step(self, action):
        """Standalone step (backward compat). Both agents use the same action.
        For true parameter sharing use AgentView instead of this directly.
        """
        action = np.asarray(action, dtype=np.float32)
        self._execute_combined_step(action, action)
        # Return ego obs (randomised _ego_idx) + average reward
        obs = self._step_obs[self._ego_idx]
        avg_reward = float(np.mean(self._step_rewards))
        return obs, avg_reward, self._step_terminated, self._step_truncated, self._step_info

    def render(self):
        return self.f110.render()

    def _visualize(self):
        """Visualize patch and agents with wall collision info."""
        # --- First call: setup figure and per-visualisation cache ---
        if self._fig is None:
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

        # --- Remove dynamic artists from the previous frame (no ax.clear()) ---
        for artist in self._vis_dynamic_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self._vis_dynamic_artists = []

        # --- Occupancy map: fetched once and cached (it never changes) ---
        occ_map = resolution = origin = None
        if self._vis_occ_cache is None and self.base_env is not None:
            try:
                _, occ_map, resolution, origin = self.base_env.get_track_data()
                self._vis_occ_cache = (occ_map, resolution, origin)
            except Exception as e:
                print(f"Warning: Could not get track data: {type(e).__name__}: {e}")
        if self._vis_occ_cache is not None:
            occ_map, resolution, origin = self._vis_occ_cache

        # --- Redraw background imshow only when the view window shifts ---
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
                # Build RGBA image: one rasterised call instead of contourf vector ops
                rgba = np.zeros((*region.shape, 4), dtype=np.uint8)
                rgba[region < 0.5] = [51, 51, 51, 204]    # dark walls
                rgba[region >= 0.5] = [220, 220, 220, 60]  # light free space
                self._vis_bg_image = ax.imshow(
                    rgba, extent=[xw0, xw1, yw0, yw1],
                    origin='lower', aspect='auto', zorder=0, interpolation='nearest')
                self._vis_bg_bounds = new_bounds

        # --- Agent positions ---
        agent_positions = []
        if self.current_base_obs is not None:
            for i in range(self.num_agents):
                x = float(self.current_base_obs["poses_x"][i])
                y = float(self.current_base_obs["poses_y"][i])
                theta = float(self.current_base_obs["poses_theta"][i])
                v = float(self.current_base_obs["linear_vels_x"][i])
                agent_positions.append((x, y, theta, v))
        else:
            agent_positions = [(self.patch.x, self.patch.y, self.patch.theta, 0.0)] * self.num_agents

        agents_inside = sum(1 for x, y, _, _ in agent_positions if self.patch.is_inside(x, y))

        # --- Wall collision check ---
        patch_wall_collision = False
        violated = []
        if occ_map is not None and resolution is not None and origin is not None:
            patch_wall_collision, violated = self.patch.check_patch_boundary_wall_collision(
                occ_map, resolution, origin, n_points=32)

        wall_status = f"⚠️ WALL! ({len(violated)} pts)" if patch_wall_collision else "Clear"
        ax.set_title(
            f'Patch Funnel V1 | Step {self.step_count}\n'
            f'Patch: v={self.patch.v:.1f}m/s, size=({self.patch.a:.1f}, {self.patch.b:.1f}) | '
            f'Inside: {agents_inside}/{self.num_agents} | {wall_status}',
            fontsize=11, color='red' if patch_wall_collision else 'black'
        )

        # --- Patch ellipse ---
        if patch_wall_collision:
            face_color, edge_color, alpha = 'red', 'darkred', 0.4
        elif agents_inside == self.num_agents:
            face_color, edge_color, alpha = 'cyan', 'darkblue', 0.3
        else:
            face_color, edge_color, alpha = 'yellow', 'orange', 0.3

        ellipse = Ellipse(
            xy=(self.patch.x, self.patch.y),
            width=self.patch.a * 2, height=self.patch.b * 2,
            angle=np.degrees(self.patch.theta),
            facecolor=face_color, edgecolor=edge_color, alpha=alpha, linewidth=3
        )
        ax.add_patch(ellipse)
        self._vis_dynamic_artists.append(ellipse)

        # --- Patch boundary points ---
        for bx, by in self.patch.get_boundary_points(16):
            ln, = ax.plot(bx, by, 'k.', markersize=4, alpha=0.5)
            self._vis_dynamic_artists.append(ln)

        # --- Patch center and velocity arrow ---
        ln, = ax.plot(self.patch.x, self.patch.y, 'b+', markersize=15, markeredgewidth=2)
        self._vis_dynamic_artists.append(ln)
        vx, vy = self.patch.get_velocity_vector()
        arr = ax.arrow(self.patch.x, self.patch.y, vx * 0.3, vy * 0.3,
                       head_width=0.2, head_length=0.15, fc='blue', ec='blue', linewidth=2)
        self._vis_dynamic_artists.append(arr)

        # --- Agents ---
        colors = ['red', 'orange']
        collisions = self.current_base_obs.get("collisions", [0.0, 0.0]) if self.current_base_obs is not None else [0.0, 0.0]
        for i, (x, y, theta, v) in enumerate(agent_positions):
            inside = self.patch.is_inside(x, y)
            collision = float(collisions[i]) > 0.5
            color = colors[i % len(colors)]
            marker, size = ('X', 18) if collision else ('o', 12) if inside else ('s', 14)
            ln, = ax.plot(x, y, marker, color=color, markersize=size,
                          markeredgecolor='black', markeredgewidth=2)
            self._vis_dynamic_artists.append(ln)
            arr = ax.arrow(x, y, v * np.cos(theta) * 0.2, v * np.sin(theta) * 0.2,
                           head_width=0.1, head_length=0.05, fc=color, ec=color, alpha=0.7)
            self._vis_dynamic_artists.append(arr)
            txt = ax.text(x + 0.3, y + 0.3, f'R{i}', fontsize=8, fontweight='bold')
            self._vis_dynamic_artists.append(txt)

        # --- Info box ---
        txt = ax.text(0.02, 0.98, f'Reward: {self.episode_reward:.1f}\n',
                      transform=ax.transAxes, fontsize=10, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        self._vis_dynamic_artists.append(txt)

        # --- View limits ---
        view_margin = max(self.patch.a, self.patch.b) + 5
        ax.set_xlim(cx - view_margin, cx + view_margin)
        ax.set_ylim(cy - view_margin, cy + view_margin)

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


class AgentView(gym.Env):
    """Single-agent view into a shared AgentEnv for true parameter sharing.

    Two AgentView instances (agent_idx=0 and agent_idx=1) wrap ONE AgentEnv.
    Each independently predicts [steer_delta, speed_delta] for its own physical
    agent. Physics executes once BOTH actions are submitted, so the two views
    must be stepped in order (agent_idx=0 before agent_idx=1 within the same
    DummyVecEnv call).

    Training setup in training.py:
        envs = []
        for _ in range(N_AGENT_ENVS):
            shared = AgentEnv(cfg)
            envs.append(lambda env=shared: AgentView(env, 0))
            envs.append(lambda env=shared: AgentView(env, 1))
        vec_env = DummyVecEnv(envs)

    This gives 2*N_AGENT_ENVS SB3 slots that all share the SAME network weights
    (parameter sharing) while each agent independently controls its own vehicle.
    """

    metadata = {"render_modes": ["human", "rgb_array", None]}

    def __init__(self, shared_env: "AgentEnv", agent_idx: int):
        super().__init__()
        assert agent_idx in (0, 1), "agent_idx must be 0 or 1"
        self.env = shared_env
        self.agent_idx = agent_idx

        # Identical spaces to AgentEnv (CTDE: 24D obs, 2D action)
        self.action_space = spaces.Box(
            low=np.array([-0.4189, -1.0], dtype=np.float32),
            high=np.array([0.4189,  1.0], dtype=np.float32),
            dtype=np.float32,
        )
        # 24D = ego_12D + partner_12D — actor only uses [:12], critic uses [:24]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # reset
    # ------------------------------------------------------------------
    def reset(self, seed=None, options=None):
        if self.agent_idx == 0:
            # Agent 0 owns the reset — runs physics reset and seeds _step_obs
            self.env.reset(seed=seed, options=options)
            # _step_obs[0] and [1] are already set by AgentEnv.reset()
        # Both views just return their own cached obs
        obs = self.env._step_obs[self.agent_idx]
        if obs is None:
            # Fallback: env hasn't been reset yet by agent_idx=0 — trigger it now
            self.env.reset(seed=seed, options=options)
            obs = self.env._step_obs[self.agent_idx]
        return obs.copy(), {}

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------
    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.env._pending_actions[self.agent_idx] = action

        if all(a is not None for a in self.env._pending_actions):
            # Both actions ready — execute one physics step
            self.env._execute_combined_step(
                self.env._pending_actions[0],
                self.env._pending_actions[1],
            )
            self.env._pending_actions = [None, None]

        obs        = self.env._step_obs[self.agent_idx].copy()
        reward     = self.env._step_rewards[self.agent_idx]
        terminated = self.env._step_terminated
        truncated  = self.env._step_truncated
        info       = dict(self.env._step_info)
        info["agent_idx"] = self.agent_idx
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # passthrough
    # ------------------------------------------------------------------
    def render(self):
        return self.env.render()

    def close(self):
        if self.agent_idx == 0:
            self.env.close()


# ===========================================================================
# Joint Training: patch + agents trained simultaneously
# ===========================================================================

class JointEnv:
    """Shared physics backend for joint patch + agent training.

    Not a gym.Env itself — wrapped by JointPatchView and JointAgentView.

    Slot layout in _pending_actions / _step_obs / _step_rewards:
        slot 0 = patch
        slot 1 = agent 0
        slot 2 = agent 1

    Coordination between training phases:
        _patch_action_fn(patch_obs) → ndarray[4]
            Used by JointAgentView when both agent slots are ready.
            Returns patch [steer, speed, a_cmd, b_cmd].
            If None: default action is used.
        _agent_action_fn(obs0, obs1) → (ndarray[2], ndarray[2])
            Used by JointPatchView to auto-fill agent slots.
            If None: zeros used for both agents.
    """

    PATCH_OBS_DIM = 15
    AGENT_OBS_DIM = 24  # CTDE: ego 12D + partner 12D

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
        self.episode_reward_patch = 0.0
        self.episode_reward_agents = [0.0, 0.0]
        self.prev_patch_s = 0.0
        self.patch_lap_s = 0.0    # cumulative Frenet distance for lap detection
        self.patch_lap_count = 0
        self.patch_prev_steer = 0.0   # for steer_rate penalty (PatchEnv parity)
        self.patch_no_progress = 0    # stuck counter (PatchEnv parity)
        self.patch_yaw_rate = 0.0
        self.current_base_obs = None
        self.prev_dist_norms = [0.0, 0.0]
        self._episode_count = 0
        self._term_counts = {
            "out_of_patch": 0, "inter_collision": 0,
            "max_steps": 0, "patch_wall": 0,
        }

        self._pending_actions: list = [None, None, None]
        self._step_obs: list = [None, None, None]
        self._step_rewards: list = [0.0, 0.0, 0.0]
        self._step_terminated: bool = False
        self._step_truncated: bool = False
        self._step_info: dict = {}

        # Frozen policy functions — set by training loop between phases
        self._patch_action_fn = None   # Callable(patch_obs) → ndarray[4]
        self._agent_action_fn = None   # Callable(obs0, obs1) → (ndarray[2], ndarray[2])
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

    # ------------------------------------------------------------------
    # Obs builders
    # ------------------------------------------------------------------

    def _patch_to_frenet(self) -> tuple:
        if self.track_spline is None:
            return 0.0, 0.0
        try:
            s, ey = self.track_spline.calc_arclength_inaccurate(
                float(self.patch.x), float(self.patch.y))
            return (float(s), float(ey)) if (np.isfinite(s) and np.isfinite(ey)) else (0.0, 0.0)
        except Exception:
            return 0.0, 0.0

    def _build_patch_obs(self, base_obs) -> np.ndarray:
        """15D patch obs — agent-agnostic: uses aggregated agent info, not per-agent slots.

        [s_norm, ey, v, theta, wall_left, wall_right, ahead_half_w,
         a, b, curvature, patch_yaw_rate,
         centroid_x_norm, centroid_y_norm,   ← mean position of ALL assigned agents
         n_inside_norm,                       ← fraction of agents inside patch [0,1]
         nearest_dist_norm]                   ← closest agent's dist_norm
        """
        s, ey = self._patch_to_frenet()
        s_norm = s / max(float(self.track_length), 1.0)
        half_w = max(self._lookup_half_w(s), 1e-3)
        ahead_s = (s + self.cfg.patch_lookahead_dist) % float(self.track_length)
        ahead_half_w = max(self._lookup_half_w(ahead_s), 1e-3)
        wall_left  = float(np.clip(half_w - ey, 0.0, 20.0))
        wall_right = float(np.clip(half_w + ey, 0.0, 20.0))
        try:
            curvature = float(self.track_spline.calc_curvature(s))
        except Exception:
            curvature = 0.0

        a = max(float(self.patch.a), 1e-3)
        b = max(float(self.patch.b), 1e-3)

        # Aggregate over all agents — works for any N agents
        n_agents = len(base_obs["poses_x"])
        patch_positions = [
            self.patch.world_to_patch_frame(
                float(base_obs["poses_x"][i]), float(base_obs["poses_y"][i]))
            for i in range(n_agents)
        ]
        dist_norms = [
            float(np.sqrt((px / a) ** 2 + (py / b) ** 2))
            for px, py in patch_positions
        ]
        centroid_x = float(np.mean([px for px, _ in patch_positions]))
        centroid_y = float(np.mean([py for _, py in patch_positions]))
        n_inside   = sum(1 for d in dist_norms if d <= 1.0) / max(n_agents, 1)
        nearest_d  = float(min(dist_norms))

        return np.array([
            s_norm, ey, float(self.patch.v), float(self.patch.theta),
            wall_left, wall_right, ahead_half_w, a, b, curvature,
            float(np.clip(self.patch_yaw_rate, -10.0, 10.0)),
            float(np.clip(centroid_x / b, -5.0, 5.0)),
            float(np.clip(centroid_y / a, -5.0, 5.0)),
            float(n_inside),
            float(np.clip(nearest_d, 0.0, 5.0)),
        ], dtype=np.float32)

    def _build_agent_obs_for(self, ego: int, base_obs) -> np.ndarray:
        """12D ego-centric obs for agent `ego` (0 or 1). f110 cars are 0 and 1."""
        other = 1 - ego
        a = max(float(self.patch.a), 1e-3)
        b = max(float(self.patch.b), 1e-3)

        def _patch_pos(fi):
            return self.patch.world_to_patch_frame(
                float(base_obs["poses_x"][fi]), float(base_obs["poses_y"][fi]))

        def _speed(fi):
            vx = float(base_obs["linear_vels_x"][fi]) if "linear_vels_x" in base_obs else 0.0
            vy = float(base_obs["linear_vels_y"][fi]) if "linear_vels_y" in base_obs else 0.0
            return float(np.hypot(vx, vy))

        def _heading_rel(fi):
            yaw = float(base_obs["poses_theta"][fi]) if "poses_theta" in base_obs else 0.0
            return float((yaw - self.patch.theta + np.pi) % (2 * np.pi) - np.pi)

        ex, ey_e = _patch_pos(ego)
        ox, oy_o = _patch_pos(other)
        return np.array([
            ex / b, ey_e / a,
            ox / b, oy_o / a,
            ex - ox, ey_e - oy_o,
            float(np.clip(self.patch_yaw_rate, -10.0, 10.0)),
            float(self.patch.v), a, b,
            _speed(ego), _heading_rel(ego),
        ], dtype=np.float32)

    def _build_all_obs(self, base_obs) -> None:
        patch_obs = np.nan_to_num(
            self._build_patch_obs(base_obs), nan=0.0, posinf=10.0, neginf=-10.0
        ).astype(np.float32)
        ego0 = np.nan_to_num(
            self._build_agent_obs_for(0, base_obs), nan=0.0, posinf=10.0, neginf=-10.0
        ).astype(np.float32)
        ego1 = np.nan_to_num(
            self._build_agent_obs_for(1, base_obs), nan=0.0, posinf=10.0, neginf=-10.0
        ).astype(np.float32)
        self._step_obs[0] = patch_obs
        self._step_obs[1] = np.concatenate([ego0, ego1])  # agent0: ego=0, partner=1
        self._step_obs[2] = np.concatenate([ego1, ego0])  # agent1: ego=1, partner=0

    # ------------------------------------------------------------------
    # Reward / termination
    # ------------------------------------------------------------------

    def _compute_agent_reward(self, dist_norm: float, prev_dist_norm: float,
                              inter_collision: bool, agent_speed: float,
                              inter_dist: float = 999.0) -> float:
        # Continuous distance-based reward — NO hard boundary, smooth gradient everywhere.
        # Agents always get signal to move toward patch center, even when far outside.
        # Speed-matching fires everywhere — agents must match patch speed whether inside or outside
        speed_err = abs(agent_speed - self.patch.v) / max(self.patch.v, 0.5)
        speed_match = 20.0 * max(0.0, 1.0 - speed_err)

        if dist_norm <= 1.0:
            # Inside patch: position reward (Gaussian, max at center) + speed match
            reward = self.cfg.inside_patch_reward * float(np.exp(-2.0 * dist_norm ** 2))
            reward += 20.0 * (prev_dist_norm - dist_norm)  # pull toward center
            reward += speed_match
        else:
            # Outside: penalty grows with distance + strong recovery gradient + speed match
            # At dist_norm=1.0: 0 penalty; at dist_norm=2.0: full penalty
            reward = -self.cfg.out_of_patch_penalty * min(dist_norm - 1.0, 1.0)
            reward += 50.0 * (prev_dist_norm - dist_norm)  # 50 not 5: match inside pull strength
            reward += speed_match  # still reward catching up to patch speed when outside
        # Gradient repulsion: smooth penalty that ramps up before hard collision threshold.
        # Fires when agents are within repulsion_zone but haven't fully collided yet.
        if inter_dist < self.cfg.repulsion_zone and not inter_collision:
            t = (self.cfg.repulsion_zone - inter_dist) / max(
                self.cfg.repulsion_zone - self.cfg.inter_agent_collision_dist, 1e-6)
            reward -= self.cfg.repulsion_weight * float(np.clip(t, 0.0, 1.0))
        if inter_collision:
            reward -= self.cfg.inter_collision_penalty
        return float(np.nan_to_num(np.clip(reward, -100.0, 100.0), nan=0.0))

    def _check_agent_termination(self, agent_idx: int, dist_norm: float,
                                 inter_collision: bool) -> tuple:
        if inter_collision:
            return True, False, f"agent{agent_idx}_inter_collision"
        if dist_norm > 1.35:
            return True, False, f"agent{agent_idx}_out_of_patch"
        if self.step_count >= self.cfg.max_steps:
            return False, True, "max_steps"
        return False, False, None

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        self.step_count = 0
        self.episode_reward_patch = 0.0
        self.episode_reward_agents = [0.0, 0.0]
        self.prev_dist_norms = [0.0, 0.0]
        self.patch_lap_s = 0.0
        self.patch_lap_count = 0
        self.patch_prev_steer = 0.0
        self.patch_no_progress = 0
        self._pending_actions = [None, None, None]

        self.f110.ensure_initialized()
        track, self._occ_map, self._resolution, self._origin = self.f110.get_track_data()
        if track.centerline is None or track.centerline.spline is None:
            raise ValueError("Track centerline/spline missing.")
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
        self.prev_patch_s, _ = self._patch_to_frenet()

        perp_dx = -np.sin(patch_theta)
        perp_dy = np.cos(patch_theta)
        offset = 0.50
        poses = np.array([
            [patch_x + offset * perp_dx, patch_y + offset * perp_dy, patch_theta],
            [patch_x - offset * perp_dx, patch_y - offset * perp_dy, patch_theta],
        ], dtype=np.float32)

        base_obs, _ = self.f110.reset(poses=poses)
        self.current_base_obs = base_obs
        self._build_all_obs(base_obs)
        self._step_rewards    = [0.0, 0.0, 0.0]
        self._step_terminated = False
        self._step_truncated  = False
        self._step_info       = {}

        # Initialise prev_dist_norms from actual spawn positions so first-step
        # center-pull shaping is zero (not a spurious penalty from 0.0 baseline).
        a = max(float(self.patch.a), 1e-3)
        b = max(float(self.patch.b), 1e-3)
        self.prev_dist_norms = []
        for i in range(2):
            xr, yr = self.patch.world_to_patch_frame(
                float(base_obs["poses_x"][i]), float(base_obs["poses_y"][i]))
            self.prev_dist_norms.append(
                float(np.sqrt((xr / a) ** 2 + (yr / b) ** 2))
            )

    # ------------------------------------------------------------------
    # Joint physics step
    # ------------------------------------------------------------------

    def _execute_joint_step(self, patch_action: np.ndarray,
                            action0: np.ndarray, action1: np.ndarray) -> None:
        """Execute one full physics step for patch + 2 agents."""
        self.step_count += 1
        dt = self.cfg.control_dt

        steer_p = float(np.clip(np.nan_to_num(patch_action[0]), -0.4189, 0.4189))
        speed_p = float(np.clip(np.nan_to_num(patch_action[1]), 0.5, 10.0))
        a_p = float(np.clip(np.nan_to_num(patch_action[2]),
                            self.cfg.patch_a_cmd_min, self.cfg.patch_a))
        b_p = float(np.clip(np.nan_to_num(patch_action[3]),
                            self.cfg.patch_b_cmd_min, self.cfg.patch_b))

        prev_theta = self.patch.theta
        prev_s, _ = self._patch_to_frenet()
        self.patch.step(speed_p, steer_p, dt)
        d_theta = (self.patch.theta - prev_theta + np.pi) % (2 * np.pi) - np.pi
        self.patch_yaw_rate = d_theta / dt
        self.patch.update_shape(a_p, b_p, dt,
                                max_a=self.cfg.patch_a, max_b=self.cfg.patch_b)

        # True when real agent actions are running (Phase 2 via pz_env, or Phase 1 with
        # a trained agent snapshot set as _agent_action_fn).
        has_real_agents = (self._agent_action_fn is not None) or self._real_agents_active

        if has_real_agents:
            # If _agent_action_fn is set (Phase 1 iteration 2+), override passed actions
            # with the frozen agent policy.
            if self._agent_action_fn is not None:
                obs0 = self._step_obs[1]
                obs1 = self._step_obs[2]
                if obs0 is not None and obs1 is not None:
                    action0, action1 = self._agent_action_fn(obs0, obs1)
            s0 = float(np.clip(np.nan_to_num(action0[0]), -0.4189, 0.4189))
            v0 = float(np.clip(np.nan_to_num(action0[1]), 0.5, 12.0))
            s1 = float(np.clip(np.nan_to_num(action1[0]), -0.4189, 0.4189))
            v1 = float(np.clip(np.nan_to_num(action1[1]), 0.5, 12.0))
            base_obs, _, _, _, _ = self.f110.step(
                np.array([[s0, v0], [s1, v1]], dtype=np.float32)
            )
            self.current_base_obs = base_obs
        else:
            # Phase 1 iteration 1: patch trains alone — no f110 simulation needed.
            # Agents are at patch center in the obs (well-defined, and ignored for rewards).
            px, py, pth = self.patch.x, self.patch.y, self.patch.theta
            base_obs = {
                "poses_x":       np.array([px, px], dtype=np.float32),
                "poses_y":       np.array([py, py], dtype=np.float32),
                "poses_theta":   np.array([pth, pth], dtype=np.float32),
                "linear_vels_x": np.zeros(2, dtype=np.float32),
                "linear_vels_y": np.zeros(2, dtype=np.float32),
            }
            self.current_base_obs = base_obs

        # Patch progress reward + lap detection
        curr_s, _ = self._patch_to_frenet()
        L = float(self.track_length)
        ds = float(np.nan_to_num(((curr_s - prev_s + L / 2.0) % L) - L / 2.0, nan=0.0))

        # Lap bonus: accumulate ds and fire bonus each time a full lap is completed.
        # This is the primary incentive for the patch to go fast — same as PatchEnv.
        self.patch_lap_s += max(0.0, ds)   # only forward progress counts
        lap_bonus = 0.0
        if self.patch_lap_s >= L:
            self.patch_lap_count += 1
            self.patch_lap_s -= L
            lap_bonus = self.cfg.patch_lap_bonus

        patch_wall_hit = False
        if self._occ_map is not None:
            patch_wall_hit, _ = self.patch.check_patch_boundary_wall_collision(
                self._occ_map, self._resolution, self._origin,
                n_points=32,
                violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )

        dist_norm_list = []
        for i in range(2):
            px = float(np.nan_to_num(base_obs["poses_x"][i], nan=self.patch.x))
            py = float(np.nan_to_num(base_obs["poses_y"][i], nan=self.patch.y))
            xr, yr = self.patch.world_to_patch_frame(px, py)
            dn = float(np.nan_to_num(np.sqrt(
                (xr / max(self.patch.a, 1e-3)) ** 2 +
                (yr / max(self.patch.b, 1e-3)) ** 2
            ), nan=2.0))  # default to "outside" if NaN
            dist_norm_list.append(dn)

        px0 = float(np.nan_to_num(base_obs["poses_x"][0], nan=self.patch.x))
        py0 = float(np.nan_to_num(base_obs["poses_y"][0], nan=self.patch.y))
        px1 = float(np.nan_to_num(base_obs["poses_x"][1], nan=self.patch.x))
        py1 = float(np.nan_to_num(base_obs["poses_y"][1], nan=self.patch.y))
        inter_dist = float(np.hypot(px0 - px1, py0 - py1))
        inter_collision = inter_dist < self.cfg.inter_agent_collision_dist


        agents_inside = sum(1.0 for dn in dist_norm_list if dn <= 1.0)

        # has_real_agents already computed above (before the f110 branch).
        effective_inside = agents_inside if self._real_agents_active else 0.0

        # --- Patch reward: identical to PatchEnv._compute_reward_for + agents_inside_bonus ---
        # Speed is fully decoupled from agent presence: patch always gets full progress reward
        # regardless of whether agents are inside. agents_inside_bonus is a separate additive
        # bonus — it tells the patch "having passengers is good" without capping its speed.
        # Agents must chase the patch independently; the patch optimises purely for fast laps.
        curr_s_for_ey, ey = self._patch_to_frenet()   # ey = crosstrack error (m)

        steer_cmd  = float(getattr(self.patch, "steering", 0.0))
        steer_rate = abs(steer_cmd - self.patch_prev_steer) / max(dt, 1e-6)
        yaw_rate   = float((self.patch.v / max(self.cfg.wheelbase, 1e-6)) * np.tan(steer_cmd))
        spin_excess = max(0.0, abs(yaw_rate) - self.cfg.spin_yawrate_threshold)

        if abs(ds) < self.cfg.stuck_progress_eps:
            self.patch_no_progress += 1
        else:
            self.patch_no_progress = 0

        reward_raw = (
            self.cfg.reward_progress_scale * ds                  # full progress — not gated by agents
            - self.cfg.reward_crosstrack_weight * abs(ey)
            - self.cfg.reward_steer_bias_weight * abs(steer_cmd)
            - self.cfg.reward_steer_rate_weight * steer_rate
            - self.cfg.reward_spin_weight * spin_excess
            # + self.cfg.agents_inside_bonus * effective_inside    # additive bonus, doesn't gate speed
            + lap_bonus
            - (self.cfg.patch_wall_penalty if patch_wall_hit else 0.0)
        )
        if self.patch_no_progress >= self.cfg.stuck_no_progress_steps:
            reward_raw -= self.cfg.stuck_penalty

        self.patch_prev_steer = steer_cmd

        patch_reward = float(np.nan_to_num(np.clip(reward_raw, -100.0, 100.0), nan=0.0))

        terminated_flags = []
        reasons = []
        for i in range(2):
            vx = float(np.nan_to_num(base_obs.get("linear_vels_x", [0.0, 0.0])[i], nan=0.0))
            vy = float(np.nan_to_num(base_obs.get("linear_vels_y", [0.0, 0.0])[i], nan=0.0))
            agent_speed = float(np.hypot(vx, vy))
            r = self._compute_agent_reward(
                dist_norm_list[i], self.prev_dist_norms[i], inter_collision, agent_speed,
                inter_dist=inter_dist)
            r = float(np.nan_to_num(r, nan=0.0))
            term, trunc, reason = self._check_agent_termination(
                i, dist_norm_list[i], inter_collision)
            self._step_rewards[i + 1] = float(np.clip(r, -100.0, 100.0))
            self.episode_reward_agents[i] += r
            terminated_flags.append(term or trunc)
            reasons.append(reason)
        self.prev_dist_norms = list(dist_norm_list)

        # Inertia (τ=0.5s in patch.py) now prevents the patch from outrunning agents.
        # The -300 penalty was a workaround for instant velocity — no longer needed.
        # agents_frac=0 → zero progress reward already handles the alignment signal.
        self._step_rewards[0] = patch_reward
        self.episode_reward_patch += patch_reward

        # During Phase 1 (no real agents), stationary dummy agents immediately leave the patch
        # and would terminate the episode in ~10 steps.  Only count agent termination when a
        # real agent policy is running.
        agent_terminated = self._real_agents_active and any(terminated_flags)
        terminated = agent_terminated or patch_wall_hit
        truncated  = self.step_count >= self.cfg.max_steps
        reason = next((r for r in reasons if r is not None and has_real_agents),
                      "patch_wall" if patch_wall_hit else
                      "max_steps" if truncated else None)

        self._build_all_obs(base_obs)

        if terminated or truncated:
            self._episode_count += 1
            key = ("inter_collision" if reason and "inter_collision" in (reason or "")
                   else "out_of_patch" if reason and "out_of_patch" in (reason or "")
                   else "patch_wall" if reason == "patch_wall"
                   else "max_steps")
            self._term_counts[key] += 1
            if self._episode_count % 200 == 0:
                n = self._episode_count
                print(
                    f"[JointEnv ep={n}] "
                    f"out_of_patch={self._term_counts['out_of_patch']/n*100:.1f}%  "
                    f"inter_collision={self._term_counts['inter_collision']/n*100:.1f}%  "
                    f"patch_wall={self._term_counts['patch_wall']/n*100:.1f}%  "
                    f"max_steps={self._term_counts['max_steps']/n*100:.1f}%  "
                    f"patch_v={self.patch.v:.2f}  inter_dist={inter_dist:.3f}"
                )

        self._step_terminated = terminated
        self._step_truncated  = truncated
        mean_agent_reward = float(np.mean(self.episode_reward_agents))
        self._step_info = {
            # keys expected by TrainingCallback
            "episode_reward": mean_agent_reward,
            "Episode_steps": self.step_count,
            "termination_reason": reason,
            # joint-specific extras
            "episode_reward_patch": self.episode_reward_patch,
            "episode_reward_agents": mean_agent_reward,
            "patch_reward": patch_reward,
            "inter_agent_dist": float(inter_dist),
            "agents_inside": int(agents_inside),
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

        # Agent positions from current_base_obs
        agent_positions = []
        if self.current_base_obs is not None:
            bobs = self.current_base_obs
            for i in range(2):
                x = float(bobs["poses_x"][i])
                y = float(bobs["poses_y"][i])
                theta = float(bobs["poses_theta"][i])
                v = float(bobs["linear_vels_x"][i])
                agent_positions.append((x, y, theta, v))
        agents_inside = sum(1 for x, y, _, _ in agent_positions if self.patch.is_inside(x, y))

        # Wall collision check
        patch_wall_collision = False
        if occ_map is not None:
            patch_wall_collision, _ = self.patch.check_patch_boundary_wall_collision(
                occ_map, resolution, origin, n_points=32)

        ax.set_title(
            f'Joint Funnel | Step {self.step_count}\n'
            f'Patch: v={self.patch.v:.1f}m/s  size=({self.patch.a:.1f},{self.patch.b:.1f}) | '
            f'Inside: {agents_inside}/2 | '
            f'{"⚠ WALL" if patch_wall_collision else "Clear"}',
            fontsize=11, color='red' if patch_wall_collision else 'black'
        )

        # Patch ellipse
        if patch_wall_collision:
            fc, ec, alpha = 'red', 'darkred', 0.4
        elif agents_inside == 2:
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

        # Agents
        colors = ['red', 'orange']
        collisions = (self.current_base_obs.get("collisions", [0.0, 0.0])
                      if self.current_base_obs is not None else [0.0, 0.0])
        for i, (x, y, theta, v) in enumerate(agent_positions):
            inside = self.patch.is_inside(x, y)
            collision = float(collisions[i]) > 0.5
            color = colors[i]
            marker, size = ('X', 18) if collision else ('o', 12) if inside else ('s', 14)
            ln, = ax.plot(x, y, marker, color=color, markersize=size,
                          markeredgecolor='black', markeredgewidth=2)
            self._vis_dynamic_artists.append(ln)
            arr = ax.arrow(x, y, v * np.cos(theta) * 0.2, v * np.sin(theta) * 0.2,
                           head_width=0.1, head_length=0.05, fc=color, ec=color, alpha=0.7)
            self._vis_dynamic_artists.append(arr)
            txt = ax.text(x + 0.3, y + 0.3, f'A{i}', fontsize=8, fontweight='bold')
            self._vis_dynamic_artists.append(txt)

        # Info box
        ep_rew = float(np.mean(self.episode_reward_agents))
        txt = ax.text(0.02, 0.98,
                      f'Ep reward (agents): {ep_rew:.1f}\nPatch reward: {self.episode_reward_patch:.1f}',
                      transform=ax.transAxes, fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        self._vis_dynamic_artists.append(txt)

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


class JointPatchView(gym.Env):
    """Gym interface for the patch agent in joint training.

    obs:    15D  [Frenet state + track width + agent positions in patch frame]
    action: 4D   [steer, speed, a_cmd, b_cmd]

    Physics executes immediately (does not wait for AgentViews).
    Agent actions come from env._agent_action_fn if set, else zeros.
    """

    metadata = {"render_modes": [None]}

    def __init__(self, shared_env: JointEnv):
        super().__init__()
        self.env = shared_env
        cfg = shared_env.cfg
        self.action_space = spaces.Box(
            low=np.array([-0.4189, 0.5, cfg.patch_a_cmd_min, cfg.patch_b_cmd_min],
                         dtype=np.float32),
            high=np.array([0.4189, 10.0, cfg.patch_a, cfg.patch_b], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(JointEnv.PATCH_OBS_DIM,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        self.env.reset(seed=seed, options=options)
        obs = self.env._step_obs[0]
        if obs is None:
            self.env.reset(seed=seed, options=options)
            obs = self.env._step_obs[0]
        return obs.copy(), {}

    def step(self, patch_action):
        patch_action = np.asarray(patch_action, dtype=np.float32)
        obs0 = self.env._step_obs[1]
        obs1 = self.env._step_obs[2]
        if self.env._agent_action_fn is not None and obs0 is not None and obs1 is not None:
            action0, action1 = self.env._agent_action_fn(obs0, obs1)
        else:
            action0 = action1 = np.zeros(2, dtype=np.float32)
        self.env._execute_joint_step(
            patch_action,
            np.asarray(action0, dtype=np.float32),
            np.asarray(action1, dtype=np.float32),
        )
        obs = self.env._step_obs[0].copy()
        info = dict(self.env._step_info)
        info["agent_idx"] = "patch"
        return obs, self.env._step_rewards[0], self.env._step_terminated, \
               self.env._step_truncated, info

    def render(self):
        return None

    def close(self):
        self.env.close()


class JointAgentView(gym.Env):
    """Gym interface for a learning agent in joint training.

    obs:    24D CTDE  [ego 12D + partner 12D]
    action: 2D        [steer, speed]  — absolute (not delta)

    Two JointAgentViews (agent_idx=1 and agent_idx=2) share one JointEnv.
    Physics executes once BOTH agent slots are filled.
    Patch action comes from env._patch_action_fn if set, else a default action.
    """

    metadata = {"render_modes": [None]}

    def __init__(self, shared_env: JointEnv, agent_idx: int):
        super().__init__()
        assert agent_idx in (1, 2), "agent_idx must be 1 or 2"
        self.env = shared_env
        self.agent_idx = agent_idx
        self.action_space = spaces.Box(
            low=np.array([-0.4189, 0.5], dtype=np.float32),
            high=np.array([0.4189, 15.0], dtype=np.float32),
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(JointEnv.AGENT_OBS_DIM,), dtype=np.float32,
        )

    def reset(self, seed=None, options=None):
        if self.agent_idx == 1:
            self.env.reset(seed=seed, options=options)
        obs = self.env._step_obs[self.agent_idx]
        if obs is None:
            self.env.reset(seed=seed, options=options)
            obs = self.env._step_obs[self.agent_idx]
        return obs.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        self.env._pending_actions[self.agent_idx] = action

        if self.env._pending_actions[1] is not None and self.env._pending_actions[2] is not None:
            patch_obs = self.env._step_obs[0]
            if self.env._patch_action_fn is not None and patch_obs is not None:
                patch_action = np.asarray(
                    self.env._patch_action_fn(patch_obs), dtype=np.float32)
            else:
                cfg = self.env.cfg
                patch_action = np.array(
                    [0.0, 2.0, cfg.patch_a, cfg.patch_b], dtype=np.float32)

            self.env._execute_joint_step(
                patch_action,
                self.env._pending_actions[1],
                self.env._pending_actions[2],
            )
            self.env._pending_actions[1] = None
            self.env._pending_actions[2] = None

        obs = self.env._step_obs[self.agent_idx].copy()
        info = dict(self.env._step_info)
        info["agent_idx"] = self.agent_idx
        return obs, self.env._step_rewards[self.agent_idx], \
               self.env._step_terminated, self.env._step_truncated, info

    def render(self):
        return None

    def close(self):
        if self.agent_idx == 1:
            self.env.close()


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
            low=np.array([-0.4189, 0.5,
                          self.cfg.a_cmd_min, self.cfg.b_cmd_min], dtype=np.float32),
            high=np.array([0.4189, 10.0,
                           self.cfg.a_cmd_max, self.cfg.b_cmd_max], dtype=np.float32),
            dtype=np.float32,
        )

        # Observation space: 11D
        #   [s, ey, v, theta, wall_left, wall_right, ahead_half_w, a, b, curvature, lane_id]
        #   ahead_half_w : half track-width at lookahead_dist metres ahead — proactive split signal
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(11,),
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

    def _build_obs_for(self, patch: "DynamicPatch", lane_id: int) -> np.ndarray:
        """
        Build 11-dim observation for a single patch:
        [s, ey, v, theta, wall_left, wall_right, ahead_half_w, a, b, curvature, lane_id]

        ahead_half_w = half track-width at (s + lookahead_dist) ahead — the proactive split signal.
        """
        if self.track_spline is None:
            raise RuntimeError("track_spline not initialized!")

        s, ey_track = self.track_spline.calc_arclength_inaccurate(
            float(patch.x), float(patch.y)
        )
        speed = float(patch.v)
        yaw   = float(patch.theta)
        a     = float(patch.a)
        b     = float(patch.b)

        s_norm = float(s) / max(float(self.track_length), 1.0)
        # Use precomputed lookup — O(1), no ray casting during rollout
        half_w      = max(self._lookup_half_w(s), 1e-3)
        ahead_s     = float(s + self.cfg.lookahead_dist) % float(self.track_length)
        ahead_half_w = max(self._lookup_half_w(ahead_s), 1e-3)

        # Cache for split/merge decision in step() — always update (primary patch calls this first)
        self._last_ahead_half_w = ahead_half_w

        try:
            curvature = float(self.track_spline.calc_curvature(float(s)))
        except Exception:
            curvature = 0.0

        if lane_id == 1:
            ey = ey_track - (half_w / 2.0)
            lane_half_w = half_w / 2.0
        elif lane_id == 2:
            ey = ey_track + (half_w / 2.0)
            lane_half_w = half_w / 2.0
        else:
            ey = ey_track
            lane_half_w = half_w

        wall_left  = float(np.clip(lane_half_w - ey, 0.0, 20.0))
        wall_right = float(np.clip(lane_half_w + ey, 0.0, 20.0))

        return np.array(
            [s_norm, ey, speed, yaw, wall_left, wall_right, ahead_half_w,
             a, b, curvature, float(lane_id)],
            dtype=np.float32,
        )

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
        if self.track_spline is None:
            return 0.0, 0.0
        try:
            s, ey = self.track_spline.calc_arclength_inaccurate(float(patch.x), float(patch.y))
            if not np.isfinite(s) or not np.isfinite(ey):
                return 0.0, 0.0
            return float(s), float(ey)
        except (AttributeError, TypeError):
            return 0.0, 0.0

    def _estimate_track_width_at(self, patch: "DynamicPatch") -> float:
        """Estimate local track width at the given patch pose using occupancy map."""
        return self._estimate_track_width_at_pos(
            float(patch.x), float(patch.y), float(patch.theta)
        )

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
        s, ey_track = self._patch_to_frenet()
        ds = self._wrap_ds(s - self.prev_s) if self.prev_s is not None else 0.0

        # In split mode, ey is relative to lane centerline, not track centerline
        track_width = self._estimate_current_track_width()
        half_w = max(track_width / 2.0, 1e-3)
        if self.lane_id == 1:
            ey = ey_track - (half_w / 2.0)
        elif self.lane_id == 2:
            ey = ey_track + (half_w / 2.0)
        else:
            ey = ey_track

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

        # Lane-centering penalty in split mode:
        # ey is already relative to lane centerline, so penalize deviation from it
        lane_center_penalty = 0.0
        if self.lane_id != 0:
            lane_center_penalty = self.cfg.lane_centering_weight * abs(ey)

        # Core reward
        reward_raw = (
            self.cfg.reward_progress_scale * ds
            - self.cfg.reward_crosstrack_weight * abs(ey)
            - lane_center_penalty                           # extra penalty in split mode
            - self.cfg.reward_steer_bias_weight * abs(steer_cmd)
            - self.cfg.reward_steer_rate_weight * steer_rate
            - self.cfg.reward_spin_weight * spin_excess
            - (self.cfg.collision_penalty if collision else 0.0)
        )


        a_now = float(max(self.patch.a, 1e-3))
        b_now = float(max(self.patch.b, 1e-3))
        base_area = float(self.cfg.patch_a * self.cfg.patch_b)  # area at spawn size
        current_area = a_now * b_now
        area_ratio = current_area / max(base_area, 1e-3)
        area_excess = max(0.0, area_ratio - 1.0)  # 0 if <= base area

        # Per-step time penalty (same as single-agent)
        # reward_raw -= self.cfg.time_penalty_per_sec * dt
        reward_raw -= self.cfg.shape_area_penalty_weight * area_excess

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
        
    # ------------------------------------------------------------------
    # Per-patch helpers
    # ------------------------------------------------------------------

    def _build_obs(self) -> np.ndarray:
        """Return observation for the primary patch (active_patches[0])."""
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
        yaw_rate    = float((patch.v / max(self.cfg.wheelbase, 1e-6)) * np.tan(steer_cmd))
        spin_excess = max(0.0, abs(yaw_rate) - self.cfg.spin_yawrate_threshold)

        # Use the pre-computed collision (occupancy map boundary check only — no Frenet proxy)
        collision = cached_collision

        # Stuck counter
        if abs(ds) < self.cfg.stuck_progress_eps:
            no_progress += 1
        else:
            no_progress = 0

        # Lane-centering penalty in split mode
        lane_center_penalty = 0.0
        if lane_id != 0:
            lane_center_penalty = self.cfg.lane_centering_weight * abs(ey)

        a_now = float(max(patch.a, 1e-3))
        b_now = float(max(patch.b, 1e-3))
        base_area    = float(self.cfg.patch_a * self.cfg.patch_b)
        current_area = a_now * b_now
        area_ratio   = current_area / max(base_area, 1e-3)
        area_excess  = max(0.0, area_ratio - 1.0)

        reward_raw = (
            self.cfg.reward_progress_scale * ds
            - self.cfg.reward_crosstrack_weight * abs(ey)
            - lane_center_penalty
            - self.cfg.reward_steer_bias_weight * abs(steer_cmd)
            - self.cfg.reward_steer_rate_weight * steer_rate
            - self.cfg.reward_spin_weight * spin_excess
            - (self.cfg.collision_penalty if collision else 0.0)
            - self.cfg.shape_area_penalty_weight * area_excess
            + float(lap_bonus)
        )
        if no_progress >= self.cfg.stuck_no_progress_steps:
            reward_raw -= self.cfg.stuck_penalty

        reward_clipped = float(np.clip(reward_raw, -100.0, 100.0))

        terms = {
            "s": float(s), "ey": float(ey), "ds": float(ds),
            "collision": bool(collision), "no_progress": int(no_progress),
            "reward_raw": float(reward_raw), "reward_clipped": float(reward_clipped),
        }
        return reward_clipped, float(s), steer_cmd, no_progress, terms

    def _check_termination_for(
        self,
        patch: "DynamicPatch",
        no_progress: int,
        cached_collision: bool,
    ) -> tuple:
        """Check termination for a single patch. Returns (terminated, reason).
        Uses pre-computed occupancy-map collision — no duplicate expensive call.
        """
        if cached_collision:
            return True, "collision_patch_boundary"
        if no_progress >= self.cfg.stuck_no_progress_steps:
            return True, "stuck_no_progress"
        return False, None

    def _do_split(self) -> None:
        """
        Split the single full-track patch into two lane patches (left / right).

        Math
        ----
        offset  = current half_w / 2
                  Places each child at its lane centreline in the CURRENT wide section.
                  As the track narrows ahead the walls guide each child into its toll lane.

        lane_b  = ahead_half_w * 0.85
                  Sizes each child to fit the UPCOMING narrow lane, not parent.b/2.
                  0.85 gives a 15% safety margin inside the lane.
                  Also capped at parent.b/2 so children are never wider than the parent.

        Both children share the parent heading and steering.
        Steering is NOT mirrored — both lanes face the same road curvature so
        both patches need to turn the same direction.
        """
        parent      = self.active_patches[0]
        parent_s, _ = self._patch_to_frenet(parent)

        # Current half-width determines lateral spawn offset
        half_w_now = max(self._lookup_half_w(parent_s), 1e-3)
        offset     = half_w_now / 2.0

        # Size each child to fit the AHEAD (narrow) lane
        lane_b = max(
            min(self._last_ahead_half_w * 0.85, parent.b / 2.0),
            0.3,
        )

        # Lateral unit vector perpendicular to heading (world frame)
        # Rotate heading +90 degrees: lat = (-sin(theta), cos(theta))
        lat_x = -float(np.sin(parent.theta))
        lat_y  =  float(np.cos(parent.theta))

        parent_steer = float(getattr(parent, "steering", 0.0))

        # Left-lane patch (lane_id=1) — offset left of centreline
        left_patch = DynamicPatch(
            x=parent.x + offset * lat_x,
            y=parent.y + offset * lat_y,
            theta=parent.theta,
            v=parent.v,
            a=parent.a,
            b=lane_b,
        )
        left_patch.steering = parent_steer

        # Right-lane patch (lane_id=2) — offset right of centreline
        right_patch = DynamicPatch(
            x=parent.x - offset * lat_x,
            y=parent.y - offset * lat_y,
            theta=parent.theta,
            v=parent.v,
            a=parent.a,
            b=lane_b,
        )
        right_patch.steering = parent_steer   # same steer — same curvature

        self.active_patches   = [left_patch, right_patch]
        self.lane_ids         = [1, 2]
        self.prev_s_list      = [parent_s, parent_s]
        self.prev_steer_list  = [parent_steer, parent_steer]
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

        # Initialize Frenet spline — one get_track_data() call, cache everything
        self.f110.ensure_initialized()
        track, occ_map, resolution, origin = self.f110.get_track_data()
        if track.centerline is None or track.centerline.spline is None:
            raise ValueError("Track centerline/spline missing for Frenet reward.")

        self.track_spline  = track.centerline.spline
        self.track_length  = float(self.track_spline.s[-1])

        # Cache for the whole episode — reused by _collision_for and _lookup_half_w
        self._occ_map    = occ_map
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
            self.f110.reset(poses=np.array([[patch_x, patch_y, patch_theta]], dtype=np.float32))

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

            self.f110.reset(poses=np.array([[patch_x, patch_y, patch_theta]], dtype=np.float32))

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
            np.nan_to_num(action[1], nan=0.5, posinf=10.0, neginf=0.5),
            0.5, 10.0))
        a_cmd = float(np.clip(
            np.nan_to_num(action[2], nan=self.cfg.a_cmd_min,
                          posinf=self.cfg.a_cmd_max, neginf=self.cfg.a_cmd_min),
            self.cfg.a_cmd_min, self.cfg.a_cmd_max))
        b_cmd = float(np.clip(
            np.nan_to_num(action[3], nan=self.cfg.b_cmd_min,
                          posinf=self.cfg.b_cmd_max, neginf=self.cfg.b_cmd_min),
            self.cfg.b_cmd_min, self.cfg.b_cmd_max))

        # --- Step each active patch ---
        for i, patch in enumerate(self.active_patches):
            # Both patches get the SAME steering — they face identical road curvature.
            # Mirroring would turn patch_1 the wrong way on every curve.
            patch.steering = steering_cmd
            patch.step(speed_cmd, steering_cmd, dt)

            # Cap max_b to 90% of the current lane half-width so the patch physically
            # cannot grow wider than the lane it is in, even if b_cmd is large.
            s_i, _ = self._patch_to_frenet(patch)
            hw_i   = self._lookup_half_w(s_i)
            max_b_i = min(self.cfg.b_cmd_max, hw_i * 0.90)
            patch.update_shape(a_cmd, b_cmd, dt,
                               max_a=self.cfg.a_cmd_max, max_b=max_b_i)

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
                patch, new_no_progress, col_i
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

        # --- Split / merge transitions (only when split_mode=True) ---
        # _last_ahead_half_w was set by _build_obs_for inside _build_obs() above —
        # no second obs build needed here.
        if self.cfg.split_mode:
            in_split = (len(self.active_patches) > 1)
            if not in_split and self._last_ahead_half_w < self.cfg.split_ahead_half_w:
                self._do_split()
            elif in_split and self._last_ahead_half_w > self.cfg.merge_ahead_half_w:
                self._do_merge()

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
        if self.cfg.render_mode == "human":
            self._visualize()
        return None

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
                        occ_map, resolution, origin, n_points=32)
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


# ---------------------------------------------------------------------------
# CTDE policy for multi-agent PPO (MAPPO)
# ---------------------------------------------------------------------------
if SB3_AVAILABLE:
    import torch as th
    import torch.nn as nn
    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3.common.type_aliases import Schedule as _Schedule

    class CTDEMlpExtractor(nn.Module):
        """MLP extractor for Centralized Training, Decentralized Execution.

        Obs layout (24D):
            obs[:12]  = ego agent's 12D observation  → used by actor
            obs[12:]  = partner agent's 12D obs      → concatenated for critic

        Actor branch: processes only the ego slice (decentralized execution).
        Critic branch: processes the full 24D joint state (centralized training).
        """

        EGO_DIM   = 12
        JOINT_DIM = 24

        def __init__(self, hidden: int = 256):
            super().__init__()
            self.latent_dim_pi = hidden
            self.latent_dim_vf = hidden

            # Decentralized actor — ego obs only
            self.policy_net = nn.Sequential(
                nn.Linear(self.EGO_DIM, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden),        nn.Tanh(),
            )
            # Centralized critic — full joint obs
            self.value_net = nn.Sequential(
                nn.Linear(self.JOINT_DIM, hidden), nn.Tanh(),
                nn.Linear(hidden, hidden),          nn.Tanh(),
            )

        def forward(self, features: th.Tensor):
            return self.forward_actor(features), self.forward_critic(features)

        def forward_actor(self, features: th.Tensor) -> th.Tensor:
            return self.policy_net(features[:, :self.EGO_DIM])

        def forward_critic(self, features: th.Tensor) -> th.Tensor:
            return self.value_net(features)

    class MAPPOPolicy(ActorCriticPolicy):
        """SB3-compatible PPO policy with CTDE.

        Drop-in replacement for 'MlpPolicy' in PPO():
            model = PPO(MAPPOPolicy, env, ...)
        """

        def _build_mlp_extractor(self) -> None:
            self.mlp_extractor = CTDEMlpExtractor(hidden=256)


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
            random_spawn=True, #change this to true so the patch agent learns well the entire track 
            split_mode=False,
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


def make_agent_env(
    rank: int,
    seed: int = 0,
    patch_env=None,
):
    def _init():
        cfg = AgentEnvConfig(
            patch_env=patch_env,
            render_mode=None,
            random_spawn=False,
        )
        env = AgentEnv(cfg)
        env.reset(seed=seed + rank)
        return env

    if SB3_AVAILABLE:
        set_random_seed(seed + rank)
    return _init


def make_agent_views(
    rank: int,
    seed: int = 0,
    patch_env=None,
) -> tuple:
    """Return a pair of (thunk_for_agent0, thunk_for_agent1) that share one AgentEnv.

    Usage in training.py:
        env_fns = []
        for i in range(NUM_ENVS):
            fn0, fn1 = make_agent_views(i, seed=42, patch_env=patch_env)
            env_fns.append(fn0)
            env_fns.append(fn1)
        vec_env = DummyVecEnv(env_fns)

    This creates 2*NUM_ENVS slots. Slots come in consecutive pairs that share
    physics — agent0's slot is always stepped before agent1's slot within each
    DummyVecEnv.step(), so agent0 submits first and agent1 triggers the physics.
    """
    if SB3_AVAILABLE:
        set_random_seed(seed + rank)

    # Build the shared backend once; both thunks close over it.
    cfg = AgentEnvConfig(
        patch_env=patch_env,
        render_mode=None,
        random_spawn=False,
    )
    shared_env = AgentEnv(cfg)
    shared_env.reset(seed=seed + rank)

    def _view0():
        return AgentView(shared_env, agent_idx=0)

    def _view1():
        return AgentView(shared_env, agent_idx=1)

    return _view0, _view1


def make_joint_views(rank: int, seed: int = 0, cfg: "JointEnvConfig" = None) -> tuple:
    """Create one JointEnv and return (shared_env, patch_thunk, agent0_thunk, agent1_thunk).

    Usage in training.py:
        shared_envs, patch_fns, agent_fns = [], [], []
        for i in range(NUM_ENVS):
            env, pf, af0, af1 = make_joint_views(i, seed=42)
            shared_envs.append(env)
            patch_fns.append(pf)
            agent_fns.extend([af0, af1])

        patch_vec_env = DummyVecEnv(patch_fns)   # N slots — one per JointEnv
        agent_vec_env = DummyVecEnv(agent_fns)   # 2N slots — two per JointEnv
    """
    if SB3_AVAILABLE:
        set_random_seed(seed + rank)
    if cfg is None:
        cfg = JointEnvConfig()
    shared_env = JointEnv(cfg)
    shared_env.reset(seed=seed + rank)

    def _patch():  return JointPatchView(shared_env)
    def _agent0(): return JointAgentView(shared_env, agent_idx=1)
    def _agent1(): return JointAgentView(shared_env, agent_idx=2)

    return shared_env, _patch, _agent0, _agent1



