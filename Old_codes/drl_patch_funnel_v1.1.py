#!/usr/bin/env python3
"""
DRL-GUIDED CONSTRAINT FUNNELING SYSTEM V1
==========================================

Major Changes from V0:
1. Patch has CAR-LIKE DYNAMICS (Ackermann model)
2. DRL outputs: a, b, acceleration, steering (patch theta is STATE)
3. Agents use NEURO-CONTROLLER instead of SE-MPC
4. Agents see RELATIVE positions to patch only
5. Domain randomization for robust learning
6. 8 parallel environments for faster training

Architecture:
┌──────────────────────────────────────────────────────────────────────────┐
│                    LAYER I: PATCH POLICY (DRL)                           │
│                         Frequency: ~10 Hz                                │
├──────────────────────────────────────────────────────────────────────────┤
│  Inputs:                           │  Outputs:                           │
│  • Patch state (x, y, θ, v)        │  • Shape: a, b                      │
│  • Aggregated LIDAR (all agents)   │  • Control: accel, steering         │
│  • Agent relative positions        │                                     │
│  • Goal direction/distance         │  Patch θ comes from dynamics!       │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 LAYER II: AGENT NEURO-CONTROLLERS                        │
│                         Frequency: ≥20 Hz                                │
├──────────────────────────────────────────────────────────────────────────┤
│  Per-Agent Neural Network Policy:                                        │
│                                                                          │
│  Inputs (RELATIVE to patch):                                             │
│  ├─ Relative position: (x-cx, y-cy) in patch frame                      │
│  ├─ Relative velocity                                                    │
│  ├─ Neighbors' relative positions                                        │
│  ├─ Patch shape (a, b)                                                   │
│  └─ Patch velocity                                                       │
│                                                                          │
│  Outputs: [acceleration, steering]                                       │
│                                                                          │
│  Learns:                                                                 │
│  ├─ Stay inside ellipsoid (containment)                                  │
│  └─ Avoid collisions with neighbors                                      │
└──────────────────────────────────────────────────────────────────────────┘

KEY CHANGES:
- Patch moves like a CAR (not teleporting to centroid)
- Agents are "blind" to the world - only see relative to patch
- Patch aggregates all LIDAR data from agents
- Domain randomization for robust policies
"""

import time
import os
import json
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from collections import deque
import copy

# Matplotlib
import matplotlib
matplotlib.use('TkAgg')  # Interactive for visualization
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    SB3_AVAILABLE = True
except ImportError:
    print("WARNING: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


# ============================================================================
# 1. PATCH WITH CAR-LIKE DYNAMICS
# ============================================================================

class DynamicPatch:
    """
    Deformable ellipsoid patch with CAR-LIKE (Ackermann) dynamics.
    
    State: [x, y, theta, v] - position, heading, velocity
    Control: [acceleration, steering] - from DRL
    Shape: [a, b] - semi-axes, from DRL
    
    Unlike V0 where patch center = centroid, here patch moves independently
    using bicycle model dynamics.
    """
    
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=2.0, a=3.0, b=2.5, wheelbase=0.5):
        # State
        self.x = x
        self.y = y
        self.theta = theta  # Heading - THIS IS NOW STATE, not action!
        self.v = v  # Forward velocity
        
        # Shape (from DRL)
        self.a = max(a, 2.0)  # Semi-major axis
        self.b = max(b, 2.0)  # Semi-minor axis
        
        # Dynamics parameters
        self.wheelbase = wheelbase  # Virtual wheelbase for Ackermann
        self.v_min = 0.5
        self.v_max = 8.0
        self.accel_max = 4.0
        self.steering_max = 0.5
        
        # Domain randomization parameters (set during reset)
        self.size_change_rate = 1.0  # How fast a,b can change
        
        # Precompute rotation for containment checks
        self._update_rotation()
        
        # History for visualization
        self.history = deque(maxlen=100)
    
    def _update_rotation(self):
        """Update rotation matrix components."""
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
    
    def step(self, accel, steering, dt=0.05):
        """
        Integrate patch dynamics using bicycle model.
        
        Args:
            accel: Acceleration command
            steering: Steering angle command
            dt: Time step
        """
        # Clip controls
        accel = np.clip(accel, -self.accel_max, self.accel_max)
        steering = np.clip(steering, -self.steering_max, self.steering_max)
        
        # Bicycle model dynamics
        self.x += self.v * np.cos(self.theta) * dt
        self.y += self.v * np.sin(self.theta) * dt
        self.theta += (self.v / self.wheelbase) * np.tan(steering) * dt
        self.v += accel * dt
        
        # Velocity limits
        self.v = np.clip(self.v, self.v_min, self.v_max)
        
        # Normalize theta to [-pi, pi]
        while self.theta > np.pi:
            self.theta -= 2 * np.pi
        while self.theta < -np.pi:
            self.theta += 2 * np.pi
        
        self._update_rotation()
    
    def update_shape(self, a, b, dt=0.05):
        """
        Update shape with rate limiting (domain randomized).
        
        Args:
            a, b: Target semi-axes
            dt: Time step
        """
        # Rate-limited shape change
        max_change = self.size_change_rate * dt
        
        target_a = max(a, 2.0)
        target_b = max(b, 2.0)
        
        # Smooth transition
        self.a += np.clip(target_a - self.a, -max_change, max_change)
        self.b += np.clip(target_b - self.b, -max_change, max_change)
    
    def is_inside(self, x, y, margin=0.0):
        """Check if point is inside ellipsoid (in world frame)."""
        dx = x - self.x
        dy = y - self.y
        
        # Rotate to patch frame
        x_rot = dx * self.cos_t + dy * self.sin_t
        y_rot = -dx * self.sin_t + dy * self.cos_t
        
        a_eff = max(self.a - margin, 0.5)
        b_eff = max(self.b - margin, 0.5)
        
        return (x_rot / a_eff)**2 + (y_rot / b_eff)**2 <= 1.0
    
    def world_to_patch_frame(self, x, y):
        """Convert world coordinates to patch-relative frame."""
        dx = x - self.x
        dy = y - self.y
        
        # Rotate to patch frame (patch heading = x-axis)
        x_rel = dx * self.cos_t + dy * self.sin_t
        y_rel = -dx * self.sin_t + dy * self.cos_t
        
        return x_rel, y_rel
    
    def patch_to_world_frame(self, x_rel, y_rel):
        """Convert patch-relative coordinates to world frame."""
        # Rotate from patch frame to world
        x = x_rel * self.cos_t - y_rel * self.sin_t + self.x
        y = x_rel * self.sin_t + y_rel * self.cos_t + self.y
        
        return x, y
    
    def signed_distance(self, x, y):
        """Approximate signed distance (negative = inside)."""
        dx = x - self.x
        dy = y - self.y
        
        x_rot = dx * self.cos_t + dy * self.sin_t
        y_rot = -dx * self.sin_t + dy * self.cos_t
        
        normalized = np.sqrt((x_rot / self.a)**2 + (y_rot / self.b)**2)
        return (normalized - 1.0) * (self.a + self.b) / 2
    
    def get_state(self):
        """Get patch state vector."""
        return np.array([self.x, self.y, self.theta, self.v])
    
    def get_velocity_vector(self):
        """Get velocity as (vx, vy) in world frame."""
        return self.v * self.cos_t, self.v * self.sin_t
    
    def save_state(self):
        """Save to history."""
        self.history.append({
            'x': self.x, 'y': self.y, 'theta': self.theta, 'v': self.v,
            'a': self.a, 'b': self.b
        })
    
    def randomize_dynamics(self, np_random):
        """Domain randomization for patch dynamics."""
        # Randomize shape change rate (how quickly patch can adapt)
        self.size_change_rate = np_random.uniform(0.5, 2.0)
        
        # Randomize velocity limits slightly
        self.v_max = np_random.uniform(6.0, 10.0)
        
        # Randomize wheelbase (affects turning)
        self.wheelbase = np_random.uniform(0.4, 0.7)
    
    def get_boundary_points(self, n_points=16):
        """
        Get points on the ellipsoid boundary in world frame.
        Used for wall collision checking.
        """
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = []
        for angle in angles:
            # Point on ellipse in patch frame
            x_local = self.a * np.cos(angle)
            y_local = self.b * np.sin(angle)
            # Transform to world frame
            x_world, y_world = self.patch_to_world_frame(x_local, y_local)
            points.append((x_world, y_world))
        return points
    
    def check_wall_collision_fast(self, aggregated_lidar, margin=0.5):
        """
        FAST check if patch boundary is too close to walls.
        Uses pre-aggregated LIDAR (minimum across all agents per sector).
        
        Args:
            aggregated_lidar: Array of min distances per sector (already computed)
            margin: Safety margin
        
        Returns:
            min_clearance: Minimum clearance from patch to walls
            collision: True if patch is too close to walls
        """
        # Simple check: compare patch size to LIDAR readings
        # The patch extends by max(a, b) from center
        max_radius = max(self.a, self.b)
        
        # Check front, left, right sectors (most important)
        n = len(aggregated_lidar)
        front = aggregated_lidar[n // 2] * 10.0  # Denormalize (was /10)
        left = aggregated_lidar[n // 4] * 10.0
        right = aggregated_lidar[3 * n // 4] * 10.0
        
        # Minimum wall distance in movement direction
        min_wall_dist = min(front, left, right)
        
        # Clearance = wall distance - patch radius
        clearance = min_wall_dist - max_radius
        
        return clearance, clearance < margin


# ============================================================================
# 2. AGENT NEURO-CONTROLLER
# ============================================================================

class AgentNeuroController:
    """
    Neural network controller for individual agents.
    
    KEY DESIGN:
    - Agents see ONLY relative positions to patch
    - Agents don't know world coordinates
    - Learns to stay inside ellipsoid + avoid collisions
    
    This is trained jointly with the patch policy.
    """
    
    def __init__(self, agent_id, robot_radius=0.15, wheelbase=0.33):
        self.agent_id = agent_id
        self.robot_radius = robot_radius
        self.wheelbase = wheelbase
        
        # Control limits
        self.v_min = 0.5
        self.v_max = 10.0
        self.accel_max = 6.0
        self.steering_max = 0.4
        
        # Observation dimension for agent (relative observations)
        # - Relative position to patch center (2)
        # - Relative velocity to patch (2)
        # - Normalized position in ellipse (2) - how close to boundary
        # - Patch shape (2) - a, b normalized
        # - Patch velocity (2)
        # - Neighbors' relative positions (3 neighbors × 2) = 6
        # Total: 16
        self.obs_dim = 16
        
        # Action dimension: [acceleration, steering]
        self.action_dim = 2
    
    def get_observation(self, agent_state, patch, neighbor_states):
        """
        Build observation for this agent (all RELATIVE to patch).
        
        Args:
            agent_state: [x, y, theta, v] in world frame
            patch: DynamicPatch instance
            neighbor_states: List of [x, y, theta, v] for other agents
        
        Returns:
            obs: Relative observation vector
        """
        x, y, theta, v = agent_state
        
        # 1. Relative position to patch center (in patch frame)
        x_rel, y_rel = patch.world_to_patch_frame(x, y)
        
        # 2. Relative velocity to patch
        vx_agent = v * np.cos(theta)
        vy_agent = v * np.sin(theta)
        vx_patch, vy_patch = patch.get_velocity_vector()
        
        # Transform velocity to patch frame
        dvx = vx_agent - vx_patch
        dvy = vy_agent - vy_patch
        vx_rel = dvx * patch.cos_t + dvy * patch.sin_t
        vy_rel = -dvx * patch.sin_t + dvy * patch.cos_t
        
        # 3. Normalized position in ellipse (how close to boundary)
        # Values > 1 mean outside
        norm_x = x_rel / patch.a
        norm_y = y_rel / patch.b
        
        # 4. Patch shape (normalized)
        a_norm = patch.a / 5.0
        b_norm = patch.b / 5.0
        
        # 5. Patch velocity (normalized)
        v_patch_norm = patch.v / 10.0
        # Heading relative to agent (always 0 in patch frame, but include for completeness)
        heading_rel = 0.0
        
        # 6. Neighbors' relative positions (in patch frame)
        neighbor_obs = []
        for ns in neighbor_states[:3]:  # Max 3 neighbors
            nx, ny, _, _ = ns
            nx_rel, ny_rel = patch.world_to_patch_frame(nx, ny)
            # Relative to this agent (in patch frame)
            neighbor_obs.extend([
                (nx_rel - x_rel) / 3.0,  # Normalized
                (ny_rel - y_rel) / 3.0
            ])
        
        # Pad if fewer than 3 neighbors
        while len(neighbor_obs) < 6:
            neighbor_obs.extend([0.0, 0.0])
        
        obs = np.array([
            x_rel / patch.a,  # Normalized relative position
            y_rel / patch.b,
            vx_rel / 5.0,     # Normalized relative velocity
            vy_rel / 5.0,
            norm_x,           # Position in ellipse
            norm_y,
            a_norm,           # Patch shape
            b_norm,
            v_patch_norm,     # Patch velocity
            heading_rel,
            *neighbor_obs     # Neighbors (6 values)
        ], dtype=np.float32)
        
        return obs
    
    def compute_reward(self, agent_state, patch, neighbor_states, collision):
        """
        Compute reward for this agent.
        
        Rewards:
        - Staying inside ellipsoid
        - Avoiding collisions with neighbors
        - Matching patch velocity (roughly)
        """
        x, y, theta, v = agent_state
        reward = 0.0
        
        # 1. Containment reward
        x_rel, y_rel = patch.world_to_patch_frame(x, y)
        ellipse_dist = (x_rel / patch.a)**2 + (y_rel / patch.b)**2
        
        if ellipse_dist <= 1.0:
            # Inside - reward for being well-centered
            reward += 1.0 - 0.5 * ellipse_dist
        else:
            # Outside - penalty
            reward -= 2.0 * (ellipse_dist - 1.0)
        
        # 2. Collision penalty
        if collision:
            reward -= 5.0
        
        # 3. Neighbor distance (soft collision avoidance)
        min_dist = float('inf')
        for ns in neighbor_states:
            nx, ny, _, _ = ns
            dist = np.sqrt((x - nx)**2 + (y - ny)**2)
            min_dist = min(min_dist, dist)
        
        safe_dist = 2 * self.robot_radius + 0.3
        if min_dist < safe_dist:
            reward -= 1.0 * (safe_dist - min_dist) / safe_dist
        
        # 4. Velocity matching (encourage following patch)
        vx_agent = v * np.cos(theta)
        vy_agent = v * np.sin(theta)
        vx_patch, vy_patch = patch.get_velocity_vector()
        vel_error = np.sqrt((vx_agent - vx_patch)**2 + (vy_agent - vy_patch)**2)
        reward += 0.3 * np.exp(-vel_error / 3.0)
        
        return reward


# ============================================================================
# 3. GYMNASIUM ENVIRONMENT (Hierarchical: Patch DRL + Agent DRL)
# ============================================================================

class PatchFunnelEnvV1(gym.Env):
    """
    DRL-Guided Constraint Funneling Environment V1.
    
    HIERARCHICAL ARCHITECTURE:
    - Outer loop: Patch policy (this env's action space)
    - Inner loop: Agent neuro-controllers (learned alongside)
    
    For simplicity, we train both jointly:
    - Action space includes patch actions + agent actions
    - Or we can use a shared policy with different observation subsets
    
    In this version, we use a SINGLE policy that outputs:
    - Patch: a, b, accel, steering (4)
    - Agents: accel, steering per agent (4 agents × 2 = 8)
    - Total: 12 actions
    """
    
    metadata = {"render_modes": ["human", "rgb_array", None]}
    
    def __init__(self, num_agents=4, render_mode=None, domain_randomize=True):
        super().__init__()
        
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.domain_randomize = domain_randomize
        
        # Robot parameters
        self.robot_radius = 0.15
        self.wheelbase = 0.33
        
        # Create dynamic patch
        self.patch = DynamicPatch()
        
        # Create agent controllers
        self.agent_controllers = [
            AgentNeuroController(i, self.robot_radius, self.wheelbase)
            for i in range(num_agents)
        ]
        
        # Base F1TENTH environment
        self.base_env = None
        
        # Lap tracking
        self.waypoints = None
        self.track_length = 100.0
        self.lap_progress = 0.0
        
        # ===== ACTION SPACE =====
        # Patch: a, b, accel, steering (4)
        # Agents: accel, steering per agent (4 × 2 = 8)
        # Total: 12
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4 + num_agents * 2,), dtype=np.float32
        )
        
        # Action bounds
        self.patch_a_range = (2.0, 5.0)
        self.patch_b_range = (2.0, 5.0)
        self.patch_accel_max = 4.0
        self.patch_steering_max = 0.5
        self.agent_accel_max = 6.0
        self.agent_steering_max = 0.4
        
        # ===== OBSERVATION SPACE =====
        # Patch state: x, y, theta, v (4)
        # Patch shape: a, b (2)
        # Aggregated LIDAR: 16 sectors (from all agents) (16)
        # Best heading + clearance (2)
        # Goal: direction (2) + distance (1) = 3
        # Agent relative positions: 4 agents × 2 = 8
        # Agent relative velocities: 4 agents × 2 = 8
        # Agents inside count (1)
        # Domain randomization params: size_change_rate (1)
        # Total: 4 + 2 + 16 + 2 + 3 + 8 + 8 + 1 + 1 = 45
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(45,), dtype=np.float32
        )
        
        # Tracking
        self.step_count = 0
        self.max_steps = 500
        self.episode_reward = 0.0
        
        # Agent states [x, y, theta, v] per agent
        self.agent_states = None
        
        # Patch wall collision tracking
        self.patch_wall_collisions = 0
        self.patch_min_clearance = float('inf')
        
        # Visualization
        self._fig = None
        self._ax = None
        
        # Random generator
        self._np_random = None
        
        # Starting position (will be loaded from track)
        self.start_x = None
        self.start_y = None
        self.start_theta = None
    
    def _denormalize_patch_action(self, action):
        """Convert normalized action to patch parameters."""
        # action[0:4] = patch actions
        a = (action[0] + 1) / 2 * (self.patch_a_range[1] - self.patch_a_range[0]) + self.patch_a_range[0]
        b = (action[1] + 1) / 2 * (self.patch_b_range[1] - self.patch_b_range[0]) + self.patch_b_range[0]
        accel = action[2] * self.patch_accel_max
        steering = action[3] * self.patch_steering_max
        
        return a, b, accel, steering
    
    def _denormalize_agent_action(self, action, agent_idx):
        """Convert normalized action to agent control."""
        # action[4 + agent_idx*2 : 4 + agent_idx*2 + 2]
        base_idx = 4 + agent_idx * 2
        accel = action[base_idx] * self.agent_accel_max
        steering = action[base_idx + 1] * self.agent_steering_max
        
        return accel, steering
    
    def _aggregate_lidar(self, obs):
        """
        Aggregate LIDAR data from all agents into unified view.
        
        Strategy: For each sector, take minimum distance across all agents
        (most conservative / safest).
        """
        n_sectors = 16
        aggregated = np.ones(n_sectors) * 30.0  # Max range
        
        for i in range(self.num_agents):
            scan = obs["scans"][i]
            n = len(scan)
            sector_size = n // n_sectors
            
            for s in range(n_sectors):
                start = s * sector_size
                end = (s + 1) * sector_size if s < n_sectors - 1 else n
                sector_min = np.min(scan[start:end])
                
                # Transform to patch frame (approximate - use agent heading)
                # For now, just aggregate raw (could improve with proper transform)
                aggregated[s] = min(aggregated[s], sector_min)
        
        # Normalize
        return np.clip(aggregated / 10.0, 0, 1.0)
    
    def _get_best_heading(self, obs):
        """Find direction with most clearance from aggregated LIDAR."""
        aggregated = self._aggregate_lidar(obs) * 10.0  # Denormalize
        
        # Smooth
        smoothed = np.convolve(aggregated, np.ones(3)/3, mode='same')
        
        best_idx = np.argmax(smoothed)
        best_clearance = smoothed[best_idx]
        
        # Convert to angle offset (-1 to 1)
        n_sectors = len(smoothed)
        angle_offset = (best_idx - n_sectors // 2) / (n_sectors // 2)
        
        return angle_offset, min(best_clearance / 10.0, 1.0)
    
    def _get_observation(self, base_obs):
        """Build observation for policy."""
        # 1. Patch state (4)
        patch_state = self.patch.get_state()
        patch_x, patch_y, patch_theta, patch_v = patch_state
        
        # Normalize
        patch_obs = np.array([
            patch_x / 50.0,
            patch_y / 20.0,
            patch_theta / np.pi,
            patch_v / 10.0
        ])
        
        # 2. Patch shape (2)
        shape_obs = np.array([
            self.patch.a / 5.0,
            self.patch.b / 5.0
        ])
        
        # 3. Aggregated LIDAR (16)
        lidar_obs = self._aggregate_lidar(base_obs)
        
        # 4. Best heading (2)
        best_heading, best_clearance = self._get_best_heading(base_obs)
        heading_obs = np.array([best_heading, best_clearance])
        
        # 5. Goal info (3)
        progress, goal_dir, dist_to_center = self._get_lap_progress(
            np.array([patch_x, patch_y])
        )
        goal_dist = (1.0 - progress) * self.track_length
        goal_obs = np.array([
            goal_dir[0], goal_dir[1],
            min(goal_dist / 50.0, 1.0)
        ])
        
        # 6. Agent relative positions (8)
        agent_pos_obs = []
        for i in range(self.num_agents):
            x, y, _, _ = self.agent_states[i]
            x_rel, y_rel = self.patch.world_to_patch_frame(x, y)
            agent_pos_obs.extend([x_rel / self.patch.a, y_rel / self.patch.b])
        agent_pos_obs = np.array(agent_pos_obs)
        
        # 7. Agent relative velocities (8)
        agent_vel_obs = []
        vx_patch, vy_patch = self.patch.get_velocity_vector()
        for i in range(self.num_agents):
            _, _, theta, v = self.agent_states[i]
            vx = v * np.cos(theta)
            vy = v * np.sin(theta)
            dvx = (vx - vx_patch) / 5.0
            dvy = (vy - vy_patch) / 5.0
            agent_vel_obs.extend([dvx, dvy])
        agent_vel_obs = np.array(agent_vel_obs)
        
        # 8. Agents inside count (1)
        agents_inside = sum(1 for i in range(self.num_agents)
                          if self.patch.is_inside(self.agent_states[i][0], 
                                                  self.agent_states[i][1]))
        inside_obs = np.array([agents_inside / self.num_agents])
        
        # 9. Domain randomization params (1)
        domain_obs = np.array([self.patch.size_change_rate / 2.0])
        
        obs = np.concatenate([
            patch_obs,      # 4
            shape_obs,      # 2
            lidar_obs,      # 16
            heading_obs,    # 2
            goal_obs,       # 3
            agent_pos_obs,  # 8
            agent_vel_obs,  # 8
            inside_obs,     # 1
            domain_obs      # 1
        ]).astype(np.float32)
        
        return obs
    
    def _get_lap_progress(self, position):
        """Compute progress along track."""
        if self.waypoints is None or len(self.waypoints) < 2:
            return 0.0, np.array([1.0, 0.0]), 0.0
        
        dists = np.linalg.norm(self.waypoints - position, axis=1)
        nearest_idx = np.argmin(dists)
        dist_to_centerline = dists[nearest_idx]
        
        progress = nearest_idx / len(self.waypoints)
        
        next_idx = (nearest_idx + 5) % len(self.waypoints)
        next_wp = self.waypoints[next_idx]
        direction = next_wp - position
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        return progress, direction, dist_to_centerline
    
    def _compute_reward(self, base_obs):
        """Compute reward for the hierarchical system."""
        reward = 0.0
        
        # 1. Progress reward (patch moving forward)
        patch_pos = np.array([self.patch.x, self.patch.y])
        progress, goal_dir, dist_to_center = self._get_lap_progress(patch_pos)
        
        progress_delta = progress - self.lap_progress
        if progress_delta > 0:
            reward += 3.0 * progress_delta * 100
        elif progress_delta < -0.5:
            progress_delta = progress + (1.0 - self.lap_progress)
            reward += 3.0 * progress_delta * 100
        
        self.lap_progress = progress
        
        # Penalty for being far from centerline
        reward -= 0.2 * min(dist_to_center / 3.0, 1.0)
        
        # 2. Containment reward
        agents_inside = 0
        total_outside_dist = 0.0
        
        for i in range(self.num_agents):
            x, y, _, _ = self.agent_states[i]
            if self.patch.is_inside(x, y, margin=self.robot_radius):
                agents_inside += 1
            else:
                dist_out = max(0, self.patch.signed_distance(x, y))
                total_outside_dist += dist_out
        
        containment_rate = agents_inside / self.num_agents
        reward += 1.5 * containment_rate
        reward -= 1.0 * min(total_outside_dist / 5.0, 2.0)
        
        # 3. Collision penalty
        num_collisions = sum(1 for i in range(self.num_agents)
                            if base_obs["collisions"][i] > 0.5)
        reward -= 3.0 * (num_collisions / self.num_agents)
        
        # 4. Inter-agent collision penalty
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                xi, yi, _, _ = self.agent_states[i]
                xj, yj, _, _ = self.agent_states[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                if dist < 2 * self.robot_radius + 0.2:
                    reward -= 1.0
        
        # 5. Efficiency bonus
        if containment_rate == 1.0:
            avg_size = (self.patch.a + self.patch.b) / 2.0
            compactness = np.clip((5.0 - avg_size) / 3.0, 0, 1)
            reward += 0.3 * compactness
        
        # 6. Velocity reward (patch moving at good speed)
        if self.patch.v > 2.0:
            reward += 0.2 * min(self.patch.v / 5.0, 1.0)
        
        # 7. Front clearance (don't crash into walls)
        lidar = self._aggregate_lidar(base_obs)
        front_clear = lidar[len(lidar)//2]  # Front sector
        if front_clear < 0.2:
            reward -= 0.5 * (0.2 - front_clear)
        
        # 8. PATCH WALL COLLISION PENALTY (KEY for patch not hitting walls!)
        # Use FAST check with pre-aggregated LIDAR (already computed above)
        clearance, patch_collision = self.patch.check_wall_collision_fast(lidar, margin=0.5)
        self.patch_min_clearance = min(self.patch_min_clearance, clearance)
        self._cached_clearance = clearance  # Cache for info/visualization
        self._cached_patch_collision = patch_collision
        
        if patch_collision:
            self.patch_wall_collisions += 1
            reward -= 2.0  # Heavy penalty for patch touching walls!
        elif clearance < 1.0:
            # Soft penalty for getting close to walls
            reward -= 0.5 * (1.0 - clearance)
        
        return np.clip(reward, -5.0, 5.0)
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        else:
            self._np_random = np.random.RandomState()
        
        # Create base environment ONLY ONCE, then just reset it
        if self.base_env is None:
            # Use rl_grid_static for fixed starting positions at track start
            # Format: <refline>_<resetfn>_<shuffle>
            # - rl = raceline, cl = centerline
            # - grid = grid formation, random = random positions
            # - static = same order, random = shuffle order
            self.base_env = gym.make(
                "f1tenth_gym:f1tenth-v0",
                config={
                    "map": "Spielberg",
                    "num_agents": self.num_agents,
                    "timestep": 0.01,
                    "integrator": "rk4",
                    "control_input": ["speed", "steering_angle"],
                    "model": "st",
                    "observation_config": {"type": "original"},
                    "params": {"mu": 1.0},
                    "reset_config": {
                        "type": "rl_grid_static",  # raceline, grid formation, static order
                    },
                },
                render_mode="human" if self.render_mode == "human" else None,
            )
        
        # Just reset the existing environment (FAST!)
        base_obs, info = self.base_env.reset()
        
        # Load track waypoints and starting position
        if self.waypoints is None:
            track = self.base_env.unwrapped.track
            if track.centerline is not None:
                self.waypoints = np.column_stack([
                    track.centerline.xs,
                    track.centerline.ys
                ])
                self.track_length = track.centerline.length
                # Get starting position from track (first waypoint)
                self.start_x = track.centerline.xs[0]
                self.start_y = track.centerline.ys[0]
                # Compute starting heading from first two waypoints
                if len(track.centerline.xs) > 1:
                    dx = track.centerline.xs[1] - track.centerline.xs[0]
                    dy = track.centerline.ys[1] - track.centerline.ys[0]
                    self.start_theta = np.arctan2(dy, dx)
                else:
                    self.start_theta = 0.0
                # Only print once (not every reset)
                # print(f"Track start: ({self.start_x:.1f}, {self.start_y:.1f}), θ={np.degrees(self.start_theta):.0f}°")
        
        # Initialize patch at team centroid
        centroid_x = np.mean(base_obs["poses_x"][:self.num_agents])
        centroid_y = np.mean(base_obs["poses_y"][:self.num_agents])
        mean_theta = np.mean(base_obs["poses_theta"][:self.num_agents])
        
        # Domain randomization for initial patch size
        if self.domain_randomize:
            init_a = self._np_random.uniform(2.5, 4.0)
            init_b = self._np_random.uniform(2.0, 3.5)
        else:
            init_a, init_b = 3.0, 2.5
        
        self.patch = DynamicPatch(
            x=centroid_x, y=centroid_y,
            theta=mean_theta, v=2.0,
            a=init_a, b=init_b
        )
        
        # Domain randomization for patch dynamics
        if self.domain_randomize:
            self.patch.randomize_dynamics(self._np_random)
        
        # Initialize agent states
        self.agent_states = []
        for i in range(self.num_agents):
            v = np.sqrt(base_obs["linear_vels_x"][i]**2 + 
                       base_obs["linear_vels_y"][i]**2)
            self.agent_states.append([
                base_obs["poses_x"][i],
                base_obs["poses_y"][i],
                base_obs["poses_theta"][i],
                max(v, 0.5)
            ])
        
        # Reset tracking
        self.step_count = 0
        self.lap_progress = 0.0
        self.episode_reward = 0.0
        self.current_base_obs = base_obs
        self.patch_wall_collisions = 0
        self.patch_min_clearance = float('inf')
        
        return self._get_observation(base_obs), {}
    
    def step(self, action):
        """Execute one step with hierarchical control."""
        self.step_count += 1
        dt = 0.05  # 20 Hz control
        
        # ===== 1. PATCH CONTROL =====
        a, b, patch_accel, patch_steering = self._denormalize_patch_action(action)
        
        # Update patch dynamics
        self.patch.step(patch_accel, patch_steering, dt)
        self.patch.update_shape(a, b, dt)
        self.patch.save_state()
        
        # ===== 2. AGENT CONTROL =====
        env_actions = np.zeros((self.num_agents, 2))
        
        for i in range(self.num_agents):
            agent_accel, agent_steering = self._denormalize_agent_action(action, i)
            
            # Update agent state
            x, y, theta, v = self.agent_states[i]
            
            # Bicycle model for agent
            x += v * np.cos(theta) * dt
            y += v * np.sin(theta) * dt
            theta += (v / self.wheelbase) * np.tan(agent_steering) * dt
            v += agent_accel * dt
            v = np.clip(v, 0.5, 10.0)
            
            # Normalize theta
            while theta > np.pi:
                theta -= 2 * np.pi
            while theta < -np.pi:
                theta += 2 * np.pi
            
            self.agent_states[i] = [x, y, theta, v]
            
            # Action for base environment
            env_actions[i] = [np.clip(agent_steering, -0.4, 0.4), v]
        
        # ===== 3. STEP BASE ENVIRONMENT =====
        base_obs, _, base_done, base_truncated, _ = self.base_env.step(env_actions)
        self.current_base_obs = base_obs
        
        # Sync agent states with simulation
        for i in range(self.num_agents):
            v = np.sqrt(base_obs["linear_vels_x"][i]**2 + 
                       base_obs["linear_vels_y"][i]**2)
            self.agent_states[i] = [
                base_obs["poses_x"][i],
                base_obs["poses_y"][i],
                base_obs["poses_theta"][i],
                max(v, 0.5)
            ]
        
        # ===== 4. COMPUTE REWARD =====
        reward = self._compute_reward(base_obs)
        self.episode_reward += reward
        
        # ===== 5. CHECK TERMINATION =====
        terminated = False
        truncated = False
        termination_reason = None
        
        # 1. Collision with wall
        if any(base_obs["collisions"][i] > 0.5 for i in range(self.num_agents)):
            terminated = True
            termination_reason = "wall_collision"
        
        # 2. Agent outside patch → HARD FAILURE (agents must stay inside!)
        agents_inside = sum(1 for i in range(self.num_agents)
                          if self.patch.is_inside(self.agent_states[i][0], 
                                                  self.agent_states[i][1],
                                                  margin=self.robot_radius))
        if agents_inside < self.num_agents:
            terminated = True
            termination_reason = "agent_outside_patch"
        
        # 3. Inter-agent collision → HARD FAILURE
        min_inter_dist = float('inf')
        for i in range(self.num_agents):
            for j in range(i + 1, self.num_agents):
                xi, yi, _, _ = self.agent_states[i]
                xj, yj, _, _ = self.agent_states[j]
                dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                min_inter_dist = min(min_inter_dist, dist)
                if dist < 2 * self.robot_radius + 0.1:  # Collision threshold
                    terminated = True
                    termination_reason = "inter_agent_collision"
        
        # 4. Lap complete (success!)
        patch_pos = np.array([self.patch.x, self.patch.y])
        progress, _, _ = self._get_lap_progress(patch_pos)
        if progress > 0.95 and self.lap_progress > 0.9:
            reward += 10.0
            terminated = True
            termination_reason = "lap_complete"
        
        # 5. Max steps
        if self.step_count >= self.max_steps:
            truncated = True
        
        # ===== 6. VISUALIZE =====
        if self.render_mode == "human" and self.step_count % 5 == 0:
            self._visualize()
            self.base_env.render()
        
        obs = self._get_observation(base_obs)
        
        # Use cached clearance from reward computation (no recomputation!)
        clearance = getattr(self, '_cached_clearance', 0.0)
        
        info = {
            "episode_reward": self.episode_reward,
            "lap_progress": self.lap_progress,
            "agents_inside": agents_inside,
            "patch_wall_collisions": self.patch_wall_collisions,
            "patch_min_clearance": self.patch_min_clearance,
            "patch_current_clearance": clearance,
            "patch_size": (self.patch.a, self.patch.b),
            "patch_velocity": self.patch.v,
            "step_reward": reward,
            "termination_reason": termination_reason,
            "min_inter_agent_dist": min_inter_dist,
        }
        
        return obs, reward, terminated, truncated, info
    
    def _visualize(self):
        """Visualize patch and agents with wall collision info."""
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(12, 9))
        
        self._ax.clear()
        self._ax.set_aspect('equal')
        self._ax.grid(True, alpha=0.3)
        
        agents_inside = sum(1 for i in range(self.num_agents)
                          if self.patch.is_inside(self.agent_states[i][0],
                                                  self.agent_states[i][1]))
        
        # Use cached wall collision result (no recomputation!)
        clearance = getattr(self, '_cached_clearance', 0.0)
        patch_wall_collision = getattr(self, '_cached_patch_collision', False)
        
        # Title with wall collision warning
        wall_status = "⚠️ WALL!" if patch_wall_collision else f"Clear: {clearance:.1f}m"
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
    
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
        if self.base_env is not None:
            self.base_env.close()


# ============================================================================
# 4. TRAINING WITH MULTIPLE ENVIRONMENTS
# ============================================================================

def make_env(rank, seed=0, render_mode=None, domain_randomize=True):
    """Create a single environment instance for parallel training."""
    def _init():
        env = PatchFunnelEnvV1(
            num_agents=4,
            render_mode=render_mode,
            domain_randomize=domain_randomize
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


def train_patch_funnel_v1(
    total_timesteps=500000,
    save_path="patch_funnel_models_v1",
    checkpoint_freq=10000,
    num_envs=8,
    resume_from=None
):
    """Train the hierarchical Patch Funnel system with multiple environments."""
    
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return
    
    os.makedirs(save_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_path, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 70)
    print("DRL-GUIDED CONSTRAINT FUNNELING V1 - TRAINING")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of parallel environments: {num_envs}")
    print(f"Checkpoint frequency: {checkpoint_freq}")
    print(f"Save directory: {run_dir}")
    print()
    print("V1 Architecture:")
    print("  [Patch Policy] → a, b, accel, steering (car-like dynamics!)")
    print("  [Agent Policy] → accel, steering (neuro-controller)")
    print("  Agents see ONLY relative positions to patch")
    print("  Domain randomization: size, adaptation rate")
    print("=" * 70)
    
    # Create parallel environments
    if num_envs > 1:
        env = SubprocVecEnv([
            make_env(i, seed=42, render_mode=None, domain_randomize=True)
            for i in range(num_envs)
        ])
    else:
        env = DummyVecEnv([
            make_env(0, seed=42, render_mode=None, domain_randomize=True)
        ])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=0.99
    )
    
    # Callback for logging with detailed metrics
    class FunnelCallbackV1(BaseCallback):
        def __init__(self, save_dir, save_freq, num_envs, verbose=1):
            super().__init__(verbose)
            self.save_dir = save_dir
            self.save_freq = save_freq
            self.num_envs = num_envs
            self.episode_rewards = []
            self.episode_progresses = []
            self.episode_lengths = []
            self.patch_wall_collisions = []
            self.termination_reasons = []
            self.episode_count = 0
            self.best_reward = -np.inf
            
            # Timing
            self.start_time = time.time()
            self.last_log_time = time.time()
            self.last_log_timesteps = 0
            
            # Training log file
            self.log_file = os.path.join(save_dir, "training_log.csv")
            with open(self.log_file, 'w') as f:
                f.write("timesteps,episodes,avg_reward,avg_progress,avg_length,"
                       "patch_collisions,time_elapsed,timesteps_per_sec\n")
        
        def _on_step(self):
            # Check for episode completion
            for i, done in enumerate(self.locals.get("dones", [])):
                if done:
                    self.episode_count += 1
                    info = self.locals.get("infos", [{}])[i]
                    reward = info.get("episode_reward", 0)
                    progress = info.get("lap_progress", 0)
                    inside = info.get("agents_inside", 0)
                    patch_collisions = info.get("patch_wall_collisions", 0)
                    reason = info.get("termination_reason", "truncated")
                    
                    self.episode_rewards.append(reward)
                    self.episode_progresses.append(progress)
                    self.patch_wall_collisions.append(patch_collisions)
                    self.termination_reasons.append(reason)
                    
                    # Every 50 episodes, log detailed stats
                    if self.episode_count % 50 == 0:
                        current_time = time.time()
                        elapsed = current_time - self.start_time
                        dt = current_time - self.last_log_time
                        timesteps_delta = self.num_timesteps - self.last_log_timesteps
                        
                        avg_r = np.mean(self.episode_rewards[-50:])
                        avg_prog = np.mean(self.episode_progresses[-50:])
                        avg_collisions = np.mean(self.patch_wall_collisions[-50:])
                        timesteps_per_sec = timesteps_delta / dt if dt > 0 else 0
                        
                        # Count termination reasons in last 50
                        recent_reasons = self.termination_reasons[-50:]
                        reason_counts = {}
                        for r in recent_reasons:
                            reason_counts[r] = reason_counts.get(r, 0) + 1
                        
                        print(f"\n{'='*60}")
                        print(f"[Episode {self.episode_count}] Timesteps: {self.num_timesteps}")
                        print(f"  Avg Reward: {avg_r:.1f} | Avg Progress: {avg_prog:.1%}")
                        print(f"  Agents Inside: {inside}/4 | Patch Wall Hits: {avg_collisions:.1f}")
                        print(f"  Time: {elapsed/60:.1f} min | Speed: {timesteps_per_sec:.1f} steps/sec")
                        print(f"  Time per 1000 steps: {1000/timesteps_per_sec:.1f} sec" if timesteps_per_sec > 0 else "")
                        print(f"  Terminations (last 50): {reason_counts}")
                        print(f"{'='*60}")
                        
                        # Update timing
                        self.last_log_time = current_time
                        self.last_log_timesteps = self.num_timesteps
                        
                        # Log to CSV
                        with open(self.log_file, 'a') as f:
                            f.write(f"{self.num_timesteps},{self.episode_count},{avg_r:.2f},"
                                   f"{avg_prog:.4f},{0},{avg_collisions:.2f},"
                                   f"{elapsed:.1f},{timesteps_per_sec:.2f}\n")
                        
                        # Save best model
                        if avg_r > self.best_reward:
                            self.best_reward = avg_r
                            self.model.save(os.path.join(self.save_dir, "best_model"))
                            if hasattr(self.training_env, 'save'):
                                self.training_env.save(
                                    os.path.join(self.save_dir, "best_vecnormalize.pkl")
                                )
                            print(f"   🏆 New best model! (reward: {avg_r:.1f})")
            
            # Checkpoint saving
            if self.num_timesteps % self.save_freq == 0:
                path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}")
                self.model.save(path)
                if hasattr(self.training_env, 'save'):
                    self.training_env.save(path + "_vecnormalize.pkl")
                
                # Save metadata
                metadata = {
                    "timesteps": self.num_timesteps,
                    "episodes": self.episode_count,
                    "best_reward": self.best_reward,
                    "avg_reward_last_50": np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards),
                    "avg_progress_last_50": np.mean(self.episode_progresses[-50:]) if len(self.episode_progresses) >= 50 else np.mean(self.episode_progresses),
                    "time_elapsed_min": (time.time() - self.start_time) / 60,
                    "timestamp": datetime.now().isoformat()
                }
                with open(path + "_metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                print(f"\n💾 Checkpoint saved: {path}")
            
            return True
    
    callback = FunnelCallbackV1(run_dir, checkpoint_freq, num_envs)
    
    # Adjust n_steps for multiple environments
    n_steps_per_env = max(256, 2048 // num_envs)
    
    # Create/load model
    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"\n📂 Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=n_steps_per_env,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(save_path, "tensorboard"),
            policy_kwargs={
                "net_arch": dict(pi=[256, 256], vf=[256, 256])
            }
        )
    
    print(f"\n🚀 Starting training with {num_envs} parallel environments...")
    print(f"   n_steps per env: {n_steps_per_env}")
    print(f"   Effective batch: {n_steps_per_env * num_envs} timesteps/update")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        model.save(os.path.join(run_dir, "final_model"))
        env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
        print(f"\n✅ Training complete! Model saved to: {run_dir}/final_model")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        model.save(os.path.join(run_dir, "interrupted_model"))
        env.save(os.path.join(run_dir, "interrupted_vecnormalize.pkl"))
        print(f"💾 Interrupted model saved")
    
    env.close()


# ============================================================================
# 5. EVALUATION
# ============================================================================

def run_eval_v1(model_path, num_episodes=40):
    """Evaluate V1 model with visualization."""
    print("=" * 70)
    print("DRL-GUIDED CONSTRAINT FUNNELING V1 - EVALUATION")
    print(f"Loading model: {model_path}")
    print("=" * 70)
    
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return
    
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = PatchFunnelEnvV1(num_agents=4, render_mode="human", domain_randomize=False)
    
    # Load normalization stats
    vecnorm_path = model_path.replace(".zip", "").replace("best_model", "best_vecnormalize.pkl")
    if not vecnorm_path.endswith(".pkl"):
        vecnorm_path = model_path + "_vecnormalize.pkl"
    
    vec_env = DummyVecEnv([lambda: env])
    
    if os.path.exists(vecnorm_path):
        print(f"Loading normalization stats: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    print("\nV1 Features:")
    print("  - Patch has CAR-LIKE dynamics (watch the cyan/red ellipse!)")
    print("  - RED ellipse = Patch too close to walls")
    print("  - Agents marked with X = wall collision")
    print("  - Agents marked with square = outside patch")
    print("=" * 70)
    
    # Aggregate stats
    all_rewards = []
    all_progress = []
    all_patch_collisions = []
    termination_reasons = {
        "wall_collision": 0,
        "agent_outside_patch": 0,
        "inter_agent_collision": 0,
        "lap_complete": 0,
        "truncated": 0,
    }
    
    for ep in range(num_episodes):
        obs = vec_env.reset()
        total_reward = 0
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            step += 1
            
            # Print periodic updates during episode
            if step % 100 == 0:
                clearance = info[0].get("patch_current_clearance", 0)
                inside = info[0].get("agents_inside", 0)
                min_dist = info[0].get("min_inter_agent_dist", 0)
                print(f"  Step {step}: Clearance={clearance:.2f}m, Inside={inside}/4, MinDist={min_dist:.2f}m")
            
            if done[0]:
                progress = info[0].get("lap_progress", 0)
                inside = info[0].get("agents_inside", 0)
                patch_collisions = info[0].get("patch_wall_collisions", 0)
                min_clearance = info[0].get("patch_min_clearance", 0)
                reason = info[0].get("termination_reason", "truncated")
                min_dist = info[0].get("min_inter_agent_dist", 0)
                
                all_rewards.append(total_reward)
                all_progress.append(progress)
                all_patch_collisions.append(patch_collisions)
                if reason:
                    termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
                else:
                    termination_reasons["truncated"] += 1
                
                # Emoji based on termination reason
                emoji = {
                    "wall_collision": "💥",
                    "agent_outside_patch": "🚫",
                    "inter_agent_collision": "💢",
                    "lap_complete": "🏆",
                    "truncated": "⏱️",
                }.get(reason, "❓")
                
                print(f"\n{'='*50}")
                print(f"{emoji} Episode {ep+1}/{num_episodes} finished at step {step}")
                print(f"   Reason: {reason}")
                print(f"   Total reward: {total_reward:.1f}")
                print(f"   Lap progress: {progress:.1%}")
                print(f"   Agents inside: {inside}/4")
                print(f"   Min inter-agent dist: {min_dist:.2f}m")
                print(f"   Patch wall collisions: {patch_collisions}")
                print(f"   Patch min clearance: {min_clearance:.2f}m")
                print(f"{'='*50}")
                break
    
    # Final summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Episodes: {num_episodes}")
    print(f"Avg Reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"Avg Progress: {np.mean(all_progress):.1%}")
    print(f"Avg Patch Wall Collisions: {np.mean(all_patch_collisions):.1f}")
    print(f"\nTermination Reasons:")
    for reason, count in termination_reasons.items():
        if count > 0:
            pct = count / num_episodes * 100
            print(f"   {reason}: {count} ({pct:.0f}%)")
    print(f"{'='*70}")
    
    vec_env.close()
    print("\n✅ Evaluation complete!")


def run_demo_v1(num_steps=1000):
    """Run demo with heuristic policy."""
    print("=" * 70)
    print("DRL-GUIDED CONSTRAINT FUNNELING V1 - DEMO")
    print("=" * 70)
    print()
    print("V1 Architecture:")
    print("  Patch: Car-like dynamics (accel + steering)")
    print("  Agents: Neuro-controller outputs")
    print()
    print("Watch:")
    print("  - Cyan ellipse = Patch with car dynamics")
    print("  - Blue arrow = Patch velocity direction")
    print("  - Agents learn to stay inside!")
    print("=" * 70)
    
    env = PatchFunnelEnvV1(num_agents=4, render_mode="human", domain_randomize=True)
    obs, _ = env.reset()
    
    total_reward = 0
    try:
        for step in range(num_steps):
            # Heuristic: move patch forward, agents follow
            # Patch actions (4): a, b, accel, steering
            patch_accel = 0.3  # Gentle forward
            patch_steering = obs[20] * 0.3  # Steer toward best heading
            
            # Agent actions (8): each agent tries to match patch
            agent_actions = []
            for i in range(4):
                # Simple: accelerate if behind, steer toward center
                rel_x = obs[17 + i * 2]
                rel_y = obs[18 + i * 2]
                
                # If behind patch center, accelerate
                accel = 0.3 if rel_x < 0 else -0.1
                # Steer toward center
                steering = -rel_y * 0.5
                
                agent_actions.extend([accel, steering])
            
            action = np.array([
                0.0,  # a (neutral)
                0.0,  # b (neutral)
                patch_accel,
                patch_steering,
                *agent_actions
            ], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 100 == 0:
                progress = info.get("lap_progress", 0)
                inside = info.get("agents_inside", 0)
                print(f"Step {step}: Reward={total_reward:.0f}, "
                      f"Progress={progress:.1%}, Inside={inside}/4")
            
            if terminated or truncated:
                print(f"\n🏁 Episode ended at step {step}")
                print(f"   Total reward: {total_reward:.0f}")
                obs, _ = env.reset()
                total_reward = 0
                
    except KeyboardInterrupt:
        print("\n⚠️ Demo stopped")
    
    env.close()


# ============================================================================
# 6. MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DRL-Guided Constraint Funneling V1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
V1 Changes:
  - Patch has CAR-LIKE dynamics (not teleporting to centroid)
  - DRL outputs: a, b, acceleration, steering
  - Patch theta comes from dynamics (state, not action)
  - Agents use neuro-controller (not SE-MPC)
  - Agents see ONLY relative positions to patch
  - Domain randomization for robust learning
  - 8 parallel environments by default

Examples:
  python3 drl_patch_funnel_v1.py --mode demo
  python3 drl_patch_funnel_v1.py --mode train --timesteps 500000 --num-envs 8
  python3 drl_patch_funnel_v1.py --mode eval --model path/to/model
        """
    )
    parser.add_argument("--mode", choices=["demo", "train", "eval"], default="demo")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model", type=str, default=None,
                       help="Path to model for eval mode")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes for eval mode")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo_v1()
    elif args.mode == "eval":
        if args.model is None:
            print("ERROR: --model path required for eval mode!")
            print("Example: --mode eval --model patch_funnel_models_v1/run_xxx/best_model")
        else:
            run_eval_v1(args.model, num_episodes=args.episodes)
    else:
        train_patch_funnel_v1(
            total_timesteps=args.timesteps,
            checkpoint_freq=args.checkpoint_freq,
            num_envs=args.num_envs,
            resume_from=args.resume
        )
