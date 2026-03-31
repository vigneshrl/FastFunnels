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
    patch_a: float = 2.0
    patch_b: float = 1.5

    wheelbase: float = 0.33
    robot_radius: float = 0.15

    out_of_patch_penalty: float = 200.0
    inter_collision_penalty: float = 150.0
    inside_patch_reward: float = 15.0
    # lap_finish_bonus: float = 2000.0
    # lap_bonus_tau: float = 200.0

    inter_agent_collision_dist: float = 0.30
    patch_boundary_violation_threshold: float = 0.10  # fraction of boundary pts in wall → collision

class AgentEnv(gym.Env):
    "Agent enviroment which is using the patch as the place to navigate"

    metadata = { "render_modes" : ["human", "rgb_array", None]}

    def __init__(self, config: Optional[AgentEnvConfig] = None):
        super().__init__()
        self.cfg = config or AgentEnvConfig()
        self.patch_env = self.cfg.patch_env
        # Action: [Δsteer1, Δsteer2]
        # Passengers lean left/right to stay inside the bus.
        # Base speed and steer come directly from the frozen patch (bus) policy.
        self.action_space = spaces.Box(
            low=np.array([-0.2, -0.2], dtype=np.float32),
            high=np.array([ 0.2,  0.2], dtype=np.float32),
            dtype=np.float32,
        )
        # Obs: 10 numbers — bus-passenger framing.
        # [x_rel0/b, y_rel0/a,   ← passenger 0: normalized position in bus (1=at door)
        #  x_rel1/b, y_rel1/a,   ← passenger 1: normalized position in bus
        #  dx_other,  dy_other,  ← separation between passengers (collision signal)
        #  patch_yaw_rate,        ← is the bus turning? (passengers need to lean)
        #  patch_v,               ← how fast is the bus going?
        #  a, b]                  ← bus size (how much room do passengers have?)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32,
        )

        # f110 with 2 real trained agents
        self.f110 = F110EnvAdapter(
            F110Config(
                num_agents=2,
                # reset_type=self.cfg.base_reset_type,
                control_input=("speed", "steering_angle"),
                timestep=self.cfg.control_dt,
            ),
            render_mode=self.cfg.render_mode,
        )
        self.base_env = self.f110
        self.map_reset = MapResetHelper()

        # Patch - driven by frozen policy
        self.patch = DynamicPatch()
        self.patch.a = self.cfg.patch_a
        self.patch.b = self.cfg.patch_b

        self.num_agents = 2
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
    
    def _build_agent_obs(self, base_obs) -> np.ndarray:
        """Build 10-dim bus-passenger observation.
        Passengers know: where they are in the bus (normalized), how far apart they are,
        what the bus is doing (turning speed, velocity), and how big the bus is.
        """
        a = max(float(self.patch.a), 1e-3)
        b = max(float(self.patch.b), 1e-3)
        patch_v = float(self.patch.v)
        patch_yaw_rate = float(np.clip(self.patch_yaw_rate, -10.0, 10.0))

        positions = []
        for i in range(2):
            x = float(base_obs["poses_x"][i])
            y = float(base_obs["poses_y"][i])
            x_rel, y_rel = self.patch.world_to_patch_frame(x, y)
            positions.append((x_rel, y_rel))

        # Normalized positions: 0=bus center, 1=at the door, >1=fell out
        x_norm0, y_norm0 = positions[0][0] / b, positions[0][1] / a
        x_norm1, y_norm1 = positions[1][0] / b, positions[1][1] / a

        # Separation between the two passengers in patch frame
        dx_other = positions[0][0] - positions[1][0]
        dy_other = positions[0][1] - positions[1][1]

        return np.array(
            [x_norm0, y_norm0, x_norm1, y_norm1, dx_other, dy_other, patch_yaw_rate, patch_v, a, b],
            dtype=np.float32,
        )

    def _compute_agent_reward(self, dist_norm: float, inter_collision: bool) -> float:
        """Per-agent reward with windowing: gradient slope before hard boundary.
        dist_norm = sqrt((x_rel/a)^2 + (y_rel/b)^2): 0=center, 1=patch edge, >1=outside.
        """
        if dist_norm <= 1.0:
            # Inside: shaped reward — highest at center, decreasing toward edge
            reward = self.cfg.inside_patch_reward * float(np.exp(-2.0 * dist_norm ** 2))
        elif dist_norm <= 1.5:
            # Buffer zone: episode continues but escalating penalty (slope before cliff)
            reward = -self.cfg.out_of_patch_penalty * (dist_norm - 1.0) / 0.5
        else:
            # Hard outside: full penalty (triggers termination)
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

        self.f110.ensure_initialized()
        track, self._occ_map, self._resolution, self._origin = self.f110.get_track_data()
        if track.centerline is None or track.centerline.spline is None:
            raise ValueError("Track centerline/spline missing.")

        self.track_spline = track.centerline.spline
        self.track_length = float(self.track_spline.s[-1])

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

        # Spawn 2 agents laterally offset inside patch
        perp_dx = -np.sin(patch_theta)
        perp_dy = np.cos(patch_theta)
        offset = 0.25  # Lateral offset from patch center
        poses = np.array([
            [patch_x + offset * perp_dx, patch_y + offset * perp_dy, patch_theta], #left
            [patch_x - offset * perp_dx, patch_y - offset * perp_dy, patch_theta], #right
        ], dtype=np.float32)
        base_obs, _ = self.f110.reset(poses=poses)
        self.current_base_obs = base_obs

        obs = self._build_agent_obs(base_obs)
        return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32), {}

    def step(self, action):
        """Step: frozen patch policy drives patch, action drives 2 agents."""
        self.step_count += 1
        dt = self.cfg.control_dt
        # --- Frozen patch policy ---
        if self.patch_env is not None:
            self.patch_env.patch = self.patch  # sync patch state
            patch_action = self.patch_env.get_patch_action(self.patch, self.track_spline)
            steer_p = float(np.clip(patch_action[0], -0.4189, 0.4189))
            speed_p = float(np.clip(patch_action[1], 0.5, 10.0))
            a_p = float(np.clip(patch_action[2], 1.0, self.cfg.patch_a))
            b_p = float(np.clip(patch_action[3], 1.0, self.cfg.patch_b))
            # print(speed_full)
            # Ramp from slow/still to full frozen-policy speed over ramp_steps.
            # Agents learn low-speed stability first, then high-speed — no abrupt jump.
            ramp_steps = 1000
            ramp_t = float(min(self.step_count / ramp_steps, 1.0))
            speed_pr = 0.5 + ramp_t * (speed_p - 0.5)
            steer_pr = ramp_t * steer_p
        else:
            steer_p, speed_p, a_p, b_p = 0.0, 2.0, self.cfg.patch_a, self.cfg.patch_b

        # if self.patch_policy is not None:
        #     s_p, ey_p = self._patch_to_frenet()
        #     patch_obs = np.array(
        #         [s_p, ey_p, float(self.patch.v), float(self.patch.theta)],
        #         dtype=np.float32,
        #     )
        #     if self.patch_vecnorm is not None:
        #         patch_obs_norm = self.patch_vecnorm.normalize_obs(patch_obs.reshape(1, -1))
        #     else:
        #         patch_obs_norm = patch_obs.reshape(1, -1)
        #     patch_action, _ = self.patch_policy.predict(patch_obs_norm, deterministic=True)
        #     patch_action = patch_action[0]
        #     steer_p = float(np.clip(patch_action[0], -0.4189, 0.4189))
        #     speed_p = float(np.clip(patch_action[1], 0.5, 10.0))
        #     a_p = float(np.clip(patch_action[2], 0.5, 2.5))
        #     b_p = float(np.clip(patch_action[3], 0.5, 1.5))
        # else:
        #     steer_p, speed_p, a_p, b_p = 0.0, 1.0, 1.5, 1.0
        self.patch.steering = steer_p
        self.patch.step(speed_p, steer_p, dt)
        self.patch.update_shape(a_p, b_p, dt, max_a=self.cfg.patch_a, max_b=self.cfg.patch_b)
        d_theta = (self.patch.theta - self.prev_patch_theta + np.pi) % (2 * np.pi) - np.pi
        self.patch_yaw_rate = d_theta / dt
        self.prev_patch_theta = self.patch.theta

        # --- Patch wall collision (mirrors frozen PatchEnv termination condition) ---
        patch_wall_hit = False
        if self._occ_map is not None:
            patch_wall_hit, _ = self.patch.check_patch_boundary_wall_collision(
                self._occ_map, self._resolution, self._origin,
                n_points=32,
                violation_threshold=self.cfg.patch_boundary_violation_threshold,
            )

        # --- Step 2 agents (bus-passenger) ---
        # Bus (patch) drives the track — passengers just follow the same commands.
        # PPO adds a tiny steer delta per passenger to stay inside and avoid each other.
        steer1 = float(np.clip(steer_p + np.nan_to_num(action[0], nan=0.0), -0.4189, 0.4189))
        steer2 = float(np.clip(steer_p + np.nan_to_num(action[1], nan=0.0), -0.4189, 0.4189))
        speed1 = speed_p   # passengers match bus speed exactly
        speed2 = speed_p
        base_obs, _, _, _, _ = self.f110.step(
            np.array([[steer1, speed1], [steer2, speed2]], dtype=np.float32)
        )
        # print ("f{base_obs["linear_vel_x"]}")
        self.current_base_obs = base_obs

        # --- Per-agent inside-patch check ---
        agent_patch = []
        dist_norm_list = []
        for i in range(2):
            x = float(base_obs["poses_x"][i])
            y = float(base_obs["poses_y"][i])
            x_rel, y_rel = self.patch.world_to_patch_frame(x, y)
            agent_patch.append((x_rel, y_rel))
            dist_norm = float(np.sqrt((x_rel / max(self.patch.a, 1e-3)) ** 2 + (y_rel / max(self.patch.b, 1e-3)) ** 2))
            dist_norm_list.append(dist_norm)

        x0, y0 = float(base_obs["poses_x"][0]), float(base_obs["poses_y"][0])
        x1, y1 = float(base_obs["poses_x"][1]), float(base_obs["poses_y"][1])
        inter_dist = float(np.hypot(x0 - x1, y0 - y1))
        inter_collision = inter_dist < self.cfg.inter_agent_collision_dist

        # --- Per-agent reward and termination ---
        total_reward = 0.0
        terminated_flags = []
        reasons = []
        for i in range(2):
            r = self._compute_agent_reward(dist_norm_list[i], inter_collision)
            term, trunc, reason = self._check_agent_termination(
                i, dist_norm_list[i], inter_collision
            )
            total_reward += r
            terminated_flags.append(term or trunc)
            reasons.append(reason)

        total_reward = float(np.clip(total_reward / 2.0, -200.0, 200.0))
        self.episode_reward += total_reward
        terminated = any(t for t in terminated_flags) or patch_wall_hit
        truncated = self.step_count >= self.cfg.max_steps
        if patch_wall_hit:
            reason = "patch_wall_collision"
        else:
            reason = next((r for r in reasons if r is not None), "max_steps" if truncated else None)

        obs = self._build_agent_obs(base_obs)
        obs = np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)

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
                    f"| now: inter_dist={inter_dist:.3f}m  patch_v={self.patch.v:.2f}  "
                    f"patch_a={self.patch.a:.2f}  patch_b={self.patch.b:.2f}"
                )

        info = {
            "episode_reward": self.episode_reward,
            "Episode_steps": self.step_count,
            "step_reward": total_reward,
            "termination_reason": reason,
            "inter_agent_dist": float(inter_dist),
            "agent0_speed": float(speed1),
            "agent0_pose" : float(x0),
            # "agent0_inside_patch": bool(is_inside_list[0]),
            # "agent1_inside_patch": bool(is_inside_list[1]),
        }
        return obs, total_reward, terminated, truncated, info

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
            random_spawn=True,
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
            random_spawn=True,
        )
        env = AgentEnv(cfg)
        env.reset(seed=seed + rank)
        return env

    if SB3_AVAILABLE:
        set_random_seed(seed + rank)
    return _init




