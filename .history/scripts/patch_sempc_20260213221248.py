#!/usr/bin/env python3
"""
NeuroPatch - Patch Environment and Dynamics
===========================================

Contains:
- DynamicPatch: Ellipsoid patch with car-like dynamics
- PatchEnv: Environment for training patch policy (leader)
"""

from mimetypes import init
import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Import SE-MPC and SafetyLayer from FastFunnels (parent scripts directory)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from SEMPC import SEMPCSolver, SafetyLayer

# Import agent controller for heuristic agents
try:
    from neurocontrollers import AgentNeuroController
except ImportError:
    AgentNeuroController = None


class DynamicPatch:
    """
    Deformable ellipsoid patch with CAR-LIKE (Ackermann) dynamics.
    
    State: [x, y, theta, v] - position, heading, velocity
    Control: [acceleration, steering] - from DRL
    Shape: [a, b] - semi-axes, from DRL
    """
    
    def __init__(self, x=0.0, y=0.0, theta=0.0, v=2.0, a=3.0, b=2.5, wheelbase=0.5):
        # State
        self.x = x
        self.y = y
        self.theta = theta
        self.v = v
        
        # Shape (from DRL)
        self.a = a
        self.b = b

        self.accel = 0.0
        self.steering = 0.0
        
        # Dynamics parameters
        self.wheelbase = wheelbase
        self.v_min = 0.5
        self.v_max = 5.0
        self.accel_max = 4.0
        self.steering_max = 0.5
        
        # Domain randomization parameters
        self.size_change_rate = 1.0
        
        # Precompute rotation
        self._update_rotation()
        
        # History for visualization
        self.history = deque(maxlen=100)
    
    def _update_rotation(self):
        """Update rotation matrix components."""
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
    
    def step(self, accel, steering, dt=0.05):
        """Integrate patch dynamics using bicycle model."""
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
    
    def update_shape(self, a, b, dt=0.05, max_a=None, max_b=None):
        """Update shape with rate limiting."""
        max_change = self.size_change_rate * dt
        
        if max_a is not None:
            a = min(a, max_a)
        if max_b is not None:
            b = min(b, max_b)
        target_a = a
        target_b = b 
        
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
        
        x_rel = dx * self.cos_t + dy * self.sin_t
        y_rel = -dx * self.sin_t + dy * self.cos_t
        
        return x_rel, y_rel
    
    def patch_to_world_frame(self, x_rel, y_rel):
        """Convert patch-relative coordinates to world frame."""
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
        self.size_change_rate = np_random.uniform(0.5, 2.0)
        self.v_max = np_random.uniform(4.0, 5.5)
        self.wheelbase = np_random.uniform(0.4, 0.7)
    
    def get_boundary_points(self, n_points=32):
        """Get points on the ellipsoid boundary in world frame."""
        angles = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
        points = []
        for angle in angles:
            x_local = self.a * np.cos(angle)
            y_local = self.b * np.sin(angle)
            x_world, y_world = self.patch_to_world_frame(x_local, y_local)
            points.append((x_world, y_world))
        return points
    
    def check_patch_boundary_wall_collision(self, base_env, n_points=32, violation_threshold=0.15):
        """
        Check if any point on the discretized ellipsoid boundary is on or outside walls.
        
        Args:
            base_env: Base F1TENTH environment (for access to occupancy map)
            n_points: Number of points to sample on ellipsoid boundary
        
        Returns:
            collision: True if any boundary point is in wall or outside track
            violated_points: List of (x, y) points that violated constraints
        """
        if base_env is None:
            return False, []
        
        # Get discretized boundary points
        boundary_points = self.get_boundary_points(n_points=n_points)
        
        # Access occupancy map from track
        try:
            track = base_env.unwrapped.track
            occ_map = track.occupancy_map
            resolution = track.spec.resolution
            origin = track.spec.origin
        except (AttributeError, KeyError):
            return False, []
        
        violated_points = []
        
        for x_world, y_world in boundary_points:
            # Convert world coordinates to pixel coordinates
            px_pixel = int((x_world - origin[0]) / resolution)
            py_pixel = int((y_world - origin[1]) / resolution)
            
            # Check if point is outside map bounds
            if (px_pixel < 0 or px_pixel >= occ_map.shape[1] or 
                py_pixel < 0 or py_pixel >= occ_map.shape[0]):
                violated_points.append((x_world, y_world))
                continue
            
            # Check if point is in wall (occupancy < 0.5 means wall/obstacle)
            if occ_map[py_pixel, px_pixel] < 0.5:
                violated_points.append((x_world, y_world))
        
        violation_fraction = len(violated_points) / n_points
        collision = violation_fraction >= violation_threshold
        return collision, violated_points


class PatchEnv(gym.Env):
    """
    Environment for training the PATCH POLICY (Leader).
    
    The patch policy controls:
    - Patch shape: a, b
    - Patch movement: acceleration, steering
    
    Agent actions come from a separate (frozen or heuristic) agent policy.
    """
    
    metadata = {"render_modes": ["human", "rgb_array", None]}
    
    def __init__(self, num_agents=2, render_mode=None, domain_randomize=True, 
                 agent_policy=None):
        super().__init__()
        
        self.num_agents = num_agents
        self.render_mode = render_mode
        self.domain_randomize = domain_randomize
        self.agent_policy = agent_policy  # Frozen agent policy or None for heuristic
        
        # Robot parameters
        self.robot_radius = 0.15
        self.wheelbase = 0.33

        #Plot patch and agents
        self._fig = None
        self._ax = None

        # self.current_base_obs = None
        # self._current_waypoint_idx = 0
        
        # ---- Landmark navigation config (no centerline dependency) ----
        self.spawn_pose = np.array([-46.6, 27.0, 2.25], dtype=np.float32)  # x, y, theta
        self.goal_xy = np.array([-52.8, 19.33], dtype=np.float32)            # global landmark
        self.goal_reached_radius = 1.5

        # Goal tracking for dense reward
        self._prev_goal_dist = None
        self._init_goal_dist = None
        
        # Add these for visualization tracking
        self.patch_wall_collisions = 0
        self.patch_min_clearance = float('inf')
        self._cached_clearance = 0.0
        self._cached_patch_collision = False
        self._track_vis_cache = None #Track visualization cache

        #Lidar safety instead of termination
        self._lidar_safety_violation = False
        self._lidar_safety_info = None
        
        # Create dynamic patch
        self.patch = DynamicPatch()
        self.alpha = 0.1

        # === SE-MPC + Safety layer for agents ===
        self.mpc_solvers = [
            SEMPCSolver(self.robot_radius, self.wheelbase, num_neighbors=self.num_agents - 1)
            for _ in range(self.num_agents)
        ]
        self.safety_layer = SafetyLayer(self.robot_radius, self.wheelbase)

        # Tracking for MPC & safety, used in obs and reward
        self.mpc_successes = 0
        self.mpc_attempts = 0
        self.safety_interventions = 0
        self.cached_controls = [None] * self.num_agents
        self.cached_feasible = [False] * self.num_agents
        self.prev_v = None  # per-agent previous speeds
        
        # Create agent controllers (for observation building)
        if AgentNeuroController:
            self.agent_controllers = [
                AgentNeuroController(i, self.robot_radius, self.wheelbase)
                for i in range(num_agents)
            ]
        else:
            self.agent_controllers = None
        
        # Base F1TENTH environment
        self.base_env = None
        self.waypoints = None
        self.track_length = 100.0
        self.lap_progress = 0.0
        
        # ===== ACTION SPACE (PATCH ONLY) =====
        # Patch: a, b, accel, steering (4)
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )
        
        # Action bounds (will be updated dynamically in reset based on spawn size)
        self.patch_a_range = (1.5, 2.0)
        self.patch_b_range = (1.5, 2.0)
        self.patch_accel_max = 4.0
        self.patch_steering_max = 0.5
        
        # ===== OBSERVATION SPACE (PATCH PERSPECTIVE) =====
        # Patch state: x, y, theta, v (4)
        # Patch shape: a, b (2)
        # Aggregated LIDAR: 16 sectors (16)
        # Best heading + clearance (2)
        # Goal: direction (2) + distance (1) = 3
        # Domain randomization params (1)
        # Alpha (1)
        # SE-MPC feedback: feasibility_rate (1) + safety_rate (1)
        obs_dim = 4 + 2 + 16 + 2 + 3 + 16 + 1 + 1 + 2
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        
        # Tracking
        self.step_count = 0
        self.max_steps = 3000
        self.episode_reward = 0.0
        self.agent_states = None
        self._np_random = None
        self.start_x = None
        self.start_y = None
        self.start_theta = None
        self._fig = None
        self._ax = None
        self.current_base_obs = None
        self._current_waypoint_idx = 0
    
    def _denormalize_patch_action(self, action):
        """Convert normalized action to patch parameters."""
        a = (action[0] + 1) / 2 * (self.patch_a_range[1] - self.patch_a_range[0]) + self.patch_a_range[0]
        b = (action[1] + 1) / 2 * (self.patch_b_range[1] - self.patch_b_range[0]) + self.patch_b_range[0]
        accel = action[2] * self.patch_accel_max
        steering = action[3] * self.patch_steering_max
        return a, b, accel, steering
    
    def _calculate_min_patch_size_for_agents(self):
        """
        Calculate minimum patch size (a, b) needed to contain all agents.
        
        Returns:
            min_a, min_b: Minimum semi-axes needed to fit all agents
        """
        if not hasattr(self, 'agent_states') or self.agent_states is None:
            # Fallback: use robot radius * 2 for minimum
            min_size = self.robot_radius * 2
            return min_size, min_size
        
        # Get all agent positions in patch frame
        agent_positions_patch_frame = []
        for i in range(self.num_agents):
            x_world, y_world, _, _ = self.agent_states[i]
            x_patch, y_patch = self.patch.world_to_patch_frame(x_world, y_world)
            agent_positions_patch_frame.append((x_patch, y_patch))
        
        if len(agent_positions_patch_frame) == 0:
            min_size = self.robot_radius * 2
            return min_size, min_size
        
        agent_positions_patch_frame = np.array(agent_positions_patch_frame)
        
        # Calculate required semi-axes to contain all agents (with margin)
        # For ellipse: (x/a)^2 + (y/b)^2 <= 1
        # We need: a >= |x| and b >= |y| for all agents (with margin)
        margin = self.robot_radius * 1.5  # Safety margin (1.5x robot radius)
        
        # Find maximum absolute x and y in patch frame
        max_abs_x = np.max(np.abs(agent_positions_patch_frame[:, 0]))
        max_abs_y = np.max(np.abs(agent_positions_patch_frame[:, 1]))
        
        # Required semi-axes: must be at least the distance to furthest agent + margin
        min_a = max(max_abs_x + margin, self.robot_radius * 1.5)
        min_b = max(max_abs_y + margin, self.robot_radius * 1.5)
        
        # Ensure minimum reasonable size
        min_a = max(min_a, 0.5)
        min_b = max(min_b, 0.5)
        
        return min_a, min_b
    
    def _position_agents_as_sensors(self, randomize=True):
        """
        Position agents as static sensors relative to patch center.
        Agents stay INSIDE the patch ellipse.
        
        Returns:
            agent_poses: Array of [x, y, theta] for each agent (in world frame)
        """
        agent_poses = np.zeros((self.num_agents, 3))
        
        if randomize and self._np_random is not None:
            for i in range(self.num_agents):
                # Generate random position INSIDE ellipse
                max_attempts = 10
                for attempt in range(max_attempts):
                    # Step 1: Generate random angle and radius
                    angle = self._np_random.uniform(0, 2 * np.pi)
                    r = np.sqrt(self._np_random.uniform(0, 0.9))  # Use 0.9 to stay well inside
                    
                    # Step 2: Convert to ellipse coordinates in patch frame
                    # FIX: Use patch.a for x and patch.b for y
                    x_local = r * self.patch.a * np.cos(angle)
                    y_local = r * self.patch.b * np.sin(angle)  # FIX: was self.patch.a
                    
                    # Step 3: Validate point is inside ellipse (safety check)
                    ellipse_dist = (x_local / self.patch.a)**2 + (y_local / self.patch.b)**2
                    if ellipse_dist <= 1.0:
                        break  # Valid point found
                    # If not valid, try again (shouldn't happen with r < 0.9, but safety check)
                
                # Step 4: Add small random offset to orientation
                theta_offset = self._np_random.uniform(-0.2, 0.2)
                theta_local = self.patch.theta + theta_offset
                
                # Step 5: Transform to world frame
                cos_t = np.cos(self.patch.theta)
                sin_t = np.sin(self.patch.theta)
                
                x_world = self.patch.x + x_local * cos_t - y_local * sin_t
                y_world = self.patch.y + x_local * sin_t + y_local * cos_t
                theta_world = theta_local
                
                # Final validation: ensure agent is inside patch
                if not self.patch.is_inside(x_world, y_world, margin=0.1):
                    # If somehow outside, place at center as fallback
                    x_world = self.patch.x
                    y_world = self.patch.y
                    theta_world = self.patch.theta
                
                agent_poses[i] = [x_world, y_world, theta_world]
                
        else:
            # Fixed positions (non-randomized)
            if self.num_agents == 2:
                offsets = [
                    (0.0, 0.0),      # Agent 0 at center
                    (0.0, 0.0),      # Agent 1 at center
                ]
            else:
                offsets = []
                for i in range(self.num_agents):
                    angle = (i / self.num_agents) * 2 * np.pi
                    radius = 0.3  # Small radius from center
                    offsets.append((radius * np.cos(angle), radius * np.sin(angle)))
            
            cos_t = np.cos(self.patch.theta)
            sin_t = np.sin(self.patch.theta)
            
            for i, (x_offset, y_offset) in enumerate(offsets):
                x_local = x_offset
                y_local = y_offset
                
                x_world = self.patch.x + x_local * cos_t - y_local * sin_t
                y_world = self.patch.y + x_local * sin_t + y_local * cos_t
                theta_world = self.patch.theta
                
                agent_poses[i] = [x_world, y_world, theta_world]
        
        return agent_poses
    
    def _aggregate_lidar(self, obs):
        """Aggregate LIDAR data from all agents."""
        # this is because the lidar is divided into 16 equal sectors as 22.5 degrees each
        n_sectors = 16
        sector_angle = 2 * np.pi /n_sectors # 22.5 degress per sector 
        # Get agent positions - use desired poses if available (for static sensors)
        if hasattr(self, '_desired_agent_poses') and self._desired_agent_poses is not None:
            # Use desired positions (where agents should be)
            agent_x = np.array([pose[0] for pose in self._desired_agent_poses])
            agent_y = np.array([pose[1] for pose in self._desired_agent_poses])
            agent_theta = np.array([pose[2] for pose in self._desired_agent_poses])
        else:
            # Fallback to simulator positions
            agent_x = np.array(obs["poses_x"])
            agent_y = np.array(obs["poses_y"])
            agent_theta = np.array(obs["poses_theta"])

        scans = np.array(obs["scans"]) * 10.0 #[num_agents, num_rays] in meters 
        num_rays = scans.shape[1]
        scans = np.clip(scans, 0.0, 10.0)  # Clamp to max range
        scans = np.where(np.isfinite(scans), scans, 10.0)  # Replace NaN/inf
        ray_angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)

        x_loc = scans * np.cos(ray_angles)
        y_loc = scans * np.sin(ray_angles)

        ax = agent_x[:, None]
        ay = agent_y[:, None]
        at = agent_theta[:, None]

            # Rotation matrix: [cos(θ) -sin(θ)]  [x]
    #                  [sin(θ)  cos(θ)]  [y]
        x_world = ax + x_loc * np.cos(at) - y_loc * np.sin(at)
        y_world = ay + x_loc * np.sin(at) + y_loc * np.cos(at)
        
        # 5. Flatten to single list of world points
        all_x_world = x_world.flatten()  # [num_agents * num_rays]
        all_y_world = y_world.flatten()  # [num_agents * num_rays]
        
        # 6. Compute direction from patch center to each world point
        dx = all_x_world - self.patch.x
        dy = all_y_world - self.patch.y
        
        # 7. Compute angle of each point relative to patch center
        # atan2 gives angle in range [-π, π], we want [0, 2π]
        point_angles = np.arctan2(dy, dx)
        point_angles = np.where(point_angles < 0, point_angles + 2 * np.pi, point_angles)
        
        # 8. Compute distance from patch center to each point
        point_distances = np.sqrt(dx**2 + dy**2)
        valid_mask = np.isfinite(point_distances) & (point_distances > 0.0)
        # 9. For each sector, find the closest point
        aggregated = np.ones(n_sectors) * np.inf  # Initialize with infinity
        
        for sector_idx in range(n_sectors):
            # Sector angle range: [sector_center - sector_angle/2, sector_center + sector_angle/2]
            sector_center = sector_idx * sector_angle
            sector_start = sector_center - sector_angle / 2
            sector_end = sector_center + sector_angle / 2
            
            # Handle wrap-around (sector 0 might include angles near 2π)
            if sector_start < 0:
                # Sector spans across 0°, so check two ranges
                mask = (point_angles >= (sector_start + 2 * np.pi)) | (point_angles <= sector_end)
            elif sector_end > 2 * np.pi:
                # Sector spans across 2π, so check two ranges
                mask = (point_angles >= sector_start) | (point_angles <= (sector_end - 2 * np.pi))
            else:
                # Normal case
                mask = (point_angles >= sector_start) & (point_angles <= sector_end)

            mask = mask & valid_mask
            # Find minimum distance in this sector
            if np.any(mask):
                aggregated[sector_idx] = np.min(point_distances[mask])
        
        # Normalize to [0, 1] range (assuming max distance is 10 meters)
        # Return in meters for now (we'll use it for clearance calculation)
        return aggregated  # Return in meters, not normalized
        # aggregated = np.ones(n_sectors) * np.inf        
        # for i in range(self.num_agents):
        #     scan = obs["scans"][i]
        #     n = len(scan)
        #     sector_size = n // n_sectors
            
        #     for s in range(n_sectors):
        #         start = s * sector_size
        #         end = (s + 1) * sector_size if s < n_sectors - 1 else n
        #         sector_min = np.min(scan[start:end])
        #         # The patch safety is governed by the CLOSEST wall seen by ANY agent
        #         aggregated[s] = min(aggregated[s], sector_min)
        
        # return np.clip(aggregated / 10.0, 0, 1.0)

        #THis is to see if the patch is safe based on Lidar data at the first step 
    # def _check_lidar_safety(self, base_obs, safety_margin=0.5):
    #     """
    #     Check if patch dimensions exceed LIDAR distances (safety rule).
        
    #     Safety rule: Patch should not extend beyond LIDAR readings.
    #     This adds "inflation" to walls - patch must stay within LIDAR clearance.
        
    #     Args:
    #         base_obs: Base environment observation (contains LIDAR scans)
    #         safety_margin: Additional safety margin (meters)
        
    #     Returns:
    #         is_safe: True if patch respects LIDAR safety
    #         max_safe_a: Maximum safe semi-axis a (forward/back)
    #         max_safe_b: Maximum safe semi-axis b (left/right)
    #         violation_info: Dict with violation details
    #     """
    #     # Aggregate LIDAR from all agents
    #     lidar = self._aggregate_lidar(base_obs) * 10.0  # Convert to meters
        
    #     # Get LIDAR in key directions relative to patch heading
    #     n_sectors = len(lidar)
    #     patch_heading_idx = n_sectors // 2  # Forward direction
        
    #     # Forward/backward clearance (semi-axis a)
    #     forward_dist = lidar[patch_heading_idx]  # Forward
    #     backward_idx = (patch_heading_idx + n_sectors // 2) % n_sectors
    #     backward_dist = lidar[backward_idx]  # Backward
    #     max_safe_a = min(forward_dist, backward_dist) - safety_margin
        
    #     # Left/right clearance (semi-axis b)
    #     left_idx = (patch_heading_idx + n_sectors // 4) % n_sectors
    #     right_idx = (patch_heading_idx - n_sectors // 4) % n_sectors
    #     left_dist = lidar[left_idx]
    #     right_dist = lidar[right_idx]
    #     max_safe_b = min(left_dist, right_dist) - safety_margin
        
    #     # Check if current patch violates LIDAR safety
    #     a_violation = self.patch.a > max_safe_a
    #     b_violation = self.patch.b > max_safe_b
        
    #     is_safe = not (a_violation or b_violation)
        
    #     violation_info = {
    #         "a_violation": a_violation,
    #         "b_violation": b_violation,
    #         "current_a": self.patch.a,
    #         "current_b": self.patch.b,
    #         "max_safe_a": max_safe_a,
    #         "max_safe_b": max_safe_b,
    #         "forward_lidar": forward_dist,
    #         "backward_lidar": backward_dist,
    #         "left_lidar": left_dist,
    #         "right_lidar": right_dist,
    #     }
        
    #     return is_safe, max_safe_a, max_safe_b, violation_info
    # Instead of checkin 4 points doing a radial sweep is much better
    def _check_lidar_safety(self, base_obs, safety_margin=0.5):
        # 1. Collect all lidar points from all agents into one array
        scans = np.array(base_obs["scans"]) * 10.0 # [num_agents, num_rays]
        num_rays = scans.shape[1]
        
        scans = np.clip(scans, 0.0, 10.0)  # Clamp to reasonable range
        scans = np.where(np.isfinite(scans), scans, 10.0)  
        # 2. Pre-calculate angles for all rays
        angles = np.linspace(0, 2 * np.pi, num_rays, endpoint=False)
        
        # 3. Vectorized Polar -> Cartesian (Local Agent Frame)
        x_loc = scans * np.cos(angles)
        y_loc = scans * np.sin(angles)
        
        # 4. Transform all points to World Frame (Vectorized)
        # Using the agent poses from base_obs
        ax = base_obs["poses_x"][:, None]
        ay = base_obs["poses_y"][:, None]
        at = base_obs["poses_theta"][:, None]
        
        x_world = ax + x_loc * np.cos(at) - y_loc * np.sin(at)
        y_world = ay + x_loc * np.sin(at) + y_loc * np.cos(at)
        
        # 5. Transform to Patch Frame
        dx = x_world - self.patch.x
        dy = y_world - self.patch.y
        x_p = dx * self.patch.cos_t + dy * self.patch.sin_t
        y_p = -dx * self.patch.sin_t + dy * self.patch.cos_t
        
        # 6. The "Magic" Ellipse Equation
        ellipse_dists = (x_p / self.patch.a)**2 + (y_p / self.patch.b)**2
        valid_mask = np.isfinite(ellipse_dists) & (ellipse_dists > 0.0)
        valid_dists = ellipse_dists[valid_mask]
        if len(valid_dists) == 0:
            # No valid LIDAR points - assume unsafe (very conservative)
            min_dist = 0.5  # Assume close to wall
        else:
            min_dist = np.min(valid_dists)
            # FIX: Clamp min_dist to reasonable range (0 to 10)
            # Values < 1.0 mean inside ellipse, > 1.0 means outside
            min_dist = np.clip(min_dist, 0.0, 10.0)
            if not np.isfinite(min_dist) or min_dist > 100.0:
                min_dist = 0.5  
        is_safe = min_dist > 1.0

        violation_info = {
            # "min_ellipse_dist": min_ellipse_dist,
            # "clearance_score": clearance_score,
            # "num_points_inside": num_points_inside,
            # "total_points_checked": total_points_checked,
            # "closest_point_patch_frame": closest_point_patch_frame,
            "is_safe": is_safe,
            "min_dist": float(min_dist)
            # "max_safe_a_est": max_safe_a_est,
            # "max_safe_b_est": max_safe_b_est,
        }
        
        # Return format: (is_safe, max_safe_a, max_safe_b, violation_info)
        # For backward compatibility, but max_safe_a/b are now estimates
        return is_safe, min_dist, violation_info

    def _get_best_heading(self, obs):
        """Find direction with most clearance from aggregated LIDAR."""
        aggregated = self._aggregate_lidar(obs) * 10.0
        smoothed = np.convolve(aggregated, np.ones(3)/3, mode='same')
        best_idx = np.argmax(smoothed)
        best_clearance = smoothed[best_idx]
        n_sectors = len(smoothed)
        angle_offset = (best_idx - n_sectors // 2) / (n_sectors // 2)
        return angle_offset, min(best_clearance / 10.0, 1.0)
    
    def _get_goal_direction_and_distance(self, patch_x, patch_y):
        """Get goal direction and distance for landmark-based navigation."""
        goal_vec_world = np.array([self.goal_xy[0] - patch_x, self.goal_xy[1] - patch_y], dtype=np.float32)
        goal_dist = float(np.linalg.norm(goal_vec_world))

        if goal_dist > 1e-6:
            goal_dir_world = goal_vec_world / goal_dist
        else:
            goal_dir_world = np.array([1.0, 0.0], dtype=np.float32)

        # Convert 1m point along world goal direction into patch-local direction
        gx = patch_x + goal_dir_world[0]
        gy = patch_y + goal_dir_world[1]
        lx, ly = self.patch.world_to_patch_frame(gx, gy)
        local_norm = np.linalg.norm([lx, ly]) + 1e-6
        local_goal_dir = np.array([lx, ly], dtype=np.float32) / local_norm

        return goal_dir_world, local_goal_dir, goal_dist
    
    def _get_lap_progress(self, position):
        """
        Legacy API kept for compatibility.
        Returns normalized progress-to-goal and global goal direction, without centerline use.
        """
        goal_dir_world, _, goal_dist = self._get_goal_direction_and_distance(position[0], position[1])
        if self._init_goal_dist is None:
            self._init_goal_dist = max(goal_dist, 1e-6)
        progress = float(np.clip(1.0 - (goal_dist / max(self._init_goal_dist, 1e-6)), 0.0, 1.0))
        return progress, goal_dir_world
    
    def _get_patch_observation(self, base_obs):
        """Build observation for patch policy (leader)."""
        patch_state = self.patch.get_state()
        patch_x, patch_y, patch_theta, patch_v = patch_state
        patch_v = max(patch_v, 0.1)

        if not hasattr(self, 'start_x') or self.start_x is None:
            self.start_x = patch_x
            self.start_y = patch_y
        
        rel_x = patch_x - self.start_x
        rel_y = patch_y - self.start_y

        track_scale = 100.0
        # Normalize patch state
        patch_obs = np.array([
            rel_x / track_scale,
            rel_y / track_scale,
            patch_theta / np.pi,
            patch_v / 10.0
        ])
        
        # Patch shape
        shape_obs = np.array([
            self.patch.a / 5.0,
            self.patch.b / 5.0
        ])
        
        # Aggregated LIDAR
        # lidar_obs = self._aggregate_lidar(base_obs)
        lidar_distances = self._aggregate_lidar(base_obs)
        lidar_obs = np.clip(lidar_distances / 10.0, 0.0, 1.0)
        clearance_obs = self._compute_clearance_observation(lidar_distances)
        # Best heading
        best_heading, best_clearance = self._get_best_heading(base_obs)
        heading_obs = np.array([best_heading, best_clearance])
        
        # Goal info (landmark-based, no centerline/waypoint dependency)
        _, local_goal_dir, goal_dist = self._get_goal_direction_and_distance(patch_x, patch_y)
        goal_obs = np.array([
            local_goal_dir[0],  # Goal direction in patch local frame (x)
            local_goal_dir[1],  # Goal direction in patch local frame (y)
            min(goal_dist / 50.0, 1.0),  # Normalized distance to goal
        ])
    
        # # Agent relative positions
        # agent_pos_obs = []
        # for i in range(self.num_agents):
        #     x, y, _, _ = self.agent_states[i]
        #     x_rel, y_rel = self.patch.world_to_patch_frame(x, y)
        #     agent_pos_obs.extend([x_rel / self.patch.a, y_rel / self.patch.b])
        # agent_pos_obs = np.array(agent_pos_obs)
        
        # # Agent relative velocities
        # agent_vel_obs = []
        # vx_patch, vy_patch = self.patch.get_velocity_vector()
        # for i in range(self.num_agents):
        #     _, _, theta, v = self.agent_states[i]
        #     vx = v * np.cos(theta)
        #     vy = v * np.sin(theta)
        #     dvx = (vx - vx_patch) / 5.0
        #     dvy = (vy - vy_patch) / 5.0
        #     agent_vel_obs.extend([dvx, dvy])
        # agent_vel_obs = np.array(agent_vel_obs)
        
        # # Agents inside count
        # agents_inside = sum(1 for i in range(self.num_agents)
        #                   if self.patch.is_inside(self.agent_states[i][0], 
        #                                           self.agent_states[i][1]))
        # inside_obs = np.array([agents_inside / self.num_agents])
        
        # Domain randomization
        domain_obs = np.array([self.patch.size_change_rate / 2.0])
        alpha_obs = np.array([self.alpha])

        # SE-MPC feedback: feasibility and safety rates
        if self.mpc_attempts > 0:
            feasibility_rate = self.mpc_successes / self.mpc_attempts
        else:
            feasibility_rate = 1.0
        safety_rate = self.safety_interventions / max(1, self.step_count * self.num_agents)
        mpc_obs = np.array([feasibility_rate, safety_rate])

        obs = np.concatenate([
            patch_obs,      # 4
            shape_obs,      # 2
            lidar_obs,      # 16
            heading_obs,    # 2
            goal_obs,       # 3
            clearance_obs,  # 16 (clearance per sector)
            domain_obs,     # 1
            alpha_obs,      # 1
            mpc_obs         # 2 (feasibility, safety)
        ]).astype(np.float32)
        
        return obs
    
    # def _compute_reward(self, base_obs, lidar_info):
    #     reward = 0.0
    #     is_safe = lidar_info['is_safe']
    #     min_dist = lidar_info['min_dist'] # The radial sweep result

    #     # --- 1. THE CRITICAL SAFETY GRADIENT ---
    #     if min_dist < 1.5:
    #         # Get clearance in forward/backward vs left/right directions
    #         lidar_distances = self._aggregate_lidar(base_obs)
    #         forward_clearance = min(lidar_distances[0], lidar_distances[8])
    #         left_right_clearance = min(lidar_distances[4], lidar_distances[12])

    #         safety_penalty = 10.0 * (1.5 - min_dist)

    #         reward -= safety_penalty
    #         # Forward/backward clearance (affects semi-axis a)
    #         # forward_idx = 0  # Sector 0 is forward
    #         # backward_idx = 8  # Sector 8 is backward
    #         # forward_clearance = lidar_distances[forward_idx]
    #         # backward_clearance = lidar_distances[backward_idx]
    #         # min_forward_back = min(forward_clearance, backward_clearance)
            
    #         # # Left/right clearance (affects semi-axis b)
    #         # left_idx = 4  # Sector 4 is left
    #         # right_idx = 12  # Sector 12 is right
    #         # left_clearance = lidar_distances[left_idx]
    #         # right_clearance = lidar_distances[right_idx]
    #         # min_left_right = min(left_clearance, right_clearance)
            
    #         # Reward shrinking a if forward/back clearance is small
    #         if forward_clearance < 4.0:  # Tight forward/back
    #             size_ratio_a = (self.patch.a - 1.5) / (5.0 - 1.5)
    #             reward += 1.0 * (1.0 - size_ratio_a)  # Strong incentive to shrink a
            
    #         # Reward shrinking b if left/right clearance is small
    #         if left_right_clearance < 4.0:  # Tight left/right
    #             size_ratio_b = (self.patch.b - 1.5) / (5.0 - 1.5)
    #             reward += 1.0 * (1.0 - size_ratio_b)  # Strong incentive to shrink b

    #         # Exponential penalty as we get closer to 1.0 (the edge)
    #         # and massive penalty if we go below 1.0 (inside)
    #         # safety_penalty = 10.0 * (1.5 - min_dist)
    #         # reward -= safety_penalty
    #         # assymetric normalisation in shriking reward 
    #         size_penalty_a = (self.patch.a - 1.5) / (5.0 - 1.5)
    #         size_penalty_b = (self.patch.b - 1.5) / (5.0 - 1.5) # this way we normalise to [0,1]
    #         # size_ratio_a = self.patch.a / 2.5  # Initial a was 2.5
    #         # size_ratio_b = self.patch.b / 1.8  
    #         # Reward smaller size when near walls
    #         reward += 0.5 * (1.0 - size_penalty_a)  # Positive reward for smaller a
    #         reward += 0.5 * (1.0 - size_penalty_b)  # Positive reward for smaller b
    #     #massive penalty
    #     if min_dist < 1.0:
    #             reward -= 50.0 * (1.0 - min_dist)

    #     # --- 2. PROGRESS (Gated by Safety) ---
    #     patch_pos = np.array([self.patch.x, self.patch.y])
    #     progress, goal_dir = self._get_lap_progress(patch_pos)
    #     progress_delta = progress - self.lap_progress
        
    #     # Handle lap wrap-around
    #     if self.lap_progress > 0.9 and progress < 0.1:
    #         progress_delta = progress + (1.0 - self.lap_progress)
        
    #     # Only reward progress if we aren't currently "crashing" the patch
    #     if is_safe:
    #         reward += progress_delta * 100.0
    #         if self.patch.v > 1.0:
    #             reward += 0.5 * min(self.patch.v / 5.0, 1.0)
    #     else:
    #         reward += progress_delta * 0.5 # Way less reward if unsafe

    #     # FIX: Penalty for not moving forward (stuck)
    #     if abs(progress_delta) < 0.001 and self.patch.v < 0.5:
    #         reward -= 2.0 
    #     # --- 3. ALIGNMENT & CENTERING ---
    #     patch_direction = np.array([np.cos(self.patch.theta), np.sin(self.patch.theta)])
    #     goal_alignment = np.dot(patch_direction, goal_dir)
        
    #     # Reward alignment only if moving forward
    #     if self.patch.v > 0.5:
    #         reward += 2.0 * goal_alignment

    #         if goal_alignment > 0.8:
    #             reward += 1.0
    #     #     # Centering reward (High weight to keep it away from walls)
    #     #     reward += 0.3 * np.exp(-dist_to_center)
    #     # min_a, min_b = 2.5, 1.8
    #     # if self.patch.a < min_a:
    #     #     reward -= 2.0 * (min_a - self.patch.a)
    #     # if self.patch.b < min_b:
    #     #     reward -= 2.0 * (min_b - self.patch.b)
    #     if self.patch.a > 3.0:
    #         reward -= 0.1 * (self.patch.a - 3.0)
    #     if self.patch.b > 2.5:
    #         reward -= 0.1 * (self.patch.b - 2.5)
    #         # reward -= 0.05 * (self.patch.a + self.patch.b)
    #         # # --- 4. STABILITY PENALTIES ---
    #     # if self.patch.v < 0.1:
    #     #     reward -= 1.0 # Don't stop!

    #     reward += 0.1

    #     reward -= 0.01
            
    #     return np.clip(reward, -20.0, 20.0)
        # return reward
        

    def _compute_reward(self, base_obs, lidar_info):
        reward = 0.0
        is_safe = lidar_info['is_safe']
        min_dist = lidar_info['min_dist']  # The radial sweep result
        
        # Get safe minimums for normalization
        safe_min_a = getattr(self, '_safe_min_a', 0.5)
        safe_min_b = getattr(self, '_safe_min_b', 0.35)
        max_a = 5.0
        max_b = 3.0
        
        # ===== 1. DENSE VELOCITY REWARD (Always Active) - MAXIMUM SPEED =====
        # Reward forward velocity continuously - encourage maximum speed!
        if self.patch.v > 0.1:  # Only reward if moving
            # Linear reward for speed (no cap - encourage going as fast as possible)
            # Normalize by max possible speed (5.0 m/s from patch dynamics)
            v_normalized = min(self.patch.v / 5.0, 1.0)  # Normalize to [0, 1]
            reward += 10.0 * v_normalized  # Strong reward for speed (was 5.0)
            
            # Bonus for high speed (encourage pushing limits)
            if self.patch.v >= 3.0:
                reward += 5.0 * (self.patch.v - 3.0) / 2.0  # Extra bonus for 3-5 m/s
        else:
            # Penalty for being stuck
            reward -= 5.0  # Stronger penalty (was 3.0)
        
        # ===== 2. DENSE DISTANCE TRAVELED REWARD =====
        # Track previous position to compute distance moved
        if not hasattr(self, '_prev_patch_pos'):
            self._prev_patch_pos = np.array([self.patch.x, self.patch.y])
        
        current_pos = np.array([self.patch.x, self.patch.y])
        distance_moved = np.linalg.norm(current_pos - self._prev_patch_pos)
        
        # Reward any forward movement (dense signal)
        # Project movement onto forward direction
        if distance_moved > 0.001:
            patch_direction = np.array([np.cos(self.patch.theta), np.sin(self.patch.theta)])
            movement_vector = current_pos - self._prev_patch_pos
            forward_component = np.dot(movement_vector, patch_direction)
            
            if forward_component > 0:  # Moving forward
                reward += 5.0 * forward_component  # Dense reward for forward distance
            elif forward_component < -0.05:  # Moving backward significantly
                reward -= 3.0 * abs(forward_component)  # Penalty for backward movement
        
        # Update previous position
        self._prev_patch_pos = current_pos.copy()
        
        # ===== 3. DENSE CLEARANCE REWARD (Always Active) =====
        # Continuous reward for maintaining safe clearance
        # Higher clearance = safer = better reward
        if min_dist > 0.3:  # Only reward if we have some clearance
            # Normalize clearance: optimal around 2.0m
            clearance_reward = min(min_dist / 2.0, 1.0)  # Normalize to [0, 1]
            reward += 1.0 * clearance_reward  # Dense reward for safety margin
        else:
            # Penalty for being too close
            reward -= 4.0 * (0.5 - min_dist)  # Exponential penalty as we get closer
        
        # Additional penalty for very close calls
        if min_dist < 0.8:
            reward -= 10.0 * (0.8 - min_dist)  # Large penalty for dangerous proximity
        
        # ===== 4. LANDMARK PROGRESS + ALIGNMENT REWARD =====
        goal_dir_world, _, goal_dist = self._get_goal_direction_and_distance(self.patch.x, self.patch.y)
        if self._prev_goal_dist is None:
            self._prev_goal_dist = goal_dist
        dist_improve = self._prev_goal_dist - goal_dist  # positive when moving toward goal
        reward += 8.0 * dist_improve  # Reward for approaching goal
        if dist_improve < -0.01:  # Moving away from goal
            reward -= 2.0 * abs(dist_improve)  # Penalty for moving away
        self._prev_goal_dist = goal_dist

        patch_direction = np.array([np.cos(self.patch.theta), np.sin(self.patch.theta)])
        goal_alignment = np.dot(patch_direction, goal_dir_world)  # [-1, 1]
        if self.patch.v > 0.1:  # Only when moving
            reward += 2.0 * max(goal_alignment, 0.0)  # Reward positive alignment
            if goal_alignment < -0.3:  # Going wrong way
                reward -= 2.0 * abs(goal_alignment)  # Penalty for wrong direction
        
        # ===== 6. DENSE SIZE EFFICIENCY REWARD (Always Active) =====
        # Get LIDAR distances for directional shrinking
        lidar_distances = self._aggregate_lidar(base_obs)
        forward_clearance = min(lidar_distances[0], lidar_distances[8])
        left_right_clearance = min(lidar_distances[4], lidar_distances[12])
        
        # Get spawn size for normalization
        spawn_a = getattr(self, '_safe_init_a', 2.0)
        spawn_b = getattr(self, '_safe_init_b', 1.5)
        min_a = getattr(self, '_safe_min_a', 0.5)
        min_b = getattr(self, '_safe_min_b', 0.35)
        
        # Normalize patch sizes relative to spawn size (0.0 = min, 1.0 = spawn size)
        size_ratio_a = (self.patch.a - min_a) / (spawn_a - min_a) if spawn_a > min_a else 0.0
        size_ratio_b = (self.patch.b - min_b) / (spawn_b - min_b) if spawn_b > min_b else 0.0
        size_ratio_a = np.clip(size_ratio_a, 0.0, 1.0)
        size_ratio_b = np.clip(size_ratio_b, 0.0, 1.0)
        
        # Base efficiency reward: smaller is generally better (but not too small)
        # Optimal size ratio is around 0.6-0.7 of spawn size for safety margin
        optimal_ratio = 0.65  # 65% of spawn size is optimal
        efficiency_a = 1.0 - abs(size_ratio_a - optimal_ratio) / optimal_ratio
        efficiency_b = 1.0 - abs(size_ratio_b - optimal_ratio) / optimal_ratio

        efficiency_a = np.clip(efficiency_a, 0.0, 1.0)
        efficiency_b = np.clip(efficiency_b, 0.0, 1.0)
        reward += 0.5 * (efficiency_a + efficiency_b)  # Dense reward for efficient size
        
        # Directional shrinking when clearance is tight
        if forward_clearance < 4.0:  # Tight forward/back
            shrink_bonus_a = (1.0 - size_ratio_a)  # Reward for smaller a
            reward += 3.0 * shrink_bonus_a * (4.0 - forward_clearance) / 4.0  # More reward when tighter
        
        if left_right_clearance < 4.0:  # Tight left/right
            shrink_bonus_b = (1.0 - size_ratio_b)  # Reward for smaller b
            reward += 3.0 * shrink_bonus_b * (4.0 - left_right_clearance) / 4.0  # More reward when tighter
        
        # ===== STRONG PENALTY FOR EXPANDING BEYOND SPAWN SIZE =====
        spawn_a = getattr(self, '_safe_init_a', 2.0)
        spawn_b = getattr(self, '_safe_init_b', 1.5)
        
        # Heavy penalty for exceeding spawn size (should never happen with constraints, but safety check)
        if self.patch.a > spawn_a:
            reward -= 100.0 * (self.patch.a - spawn_a)  # Very strong penalty
        if self.patch.b > spawn_b:
            reward -= 100.0 * (self.patch.b - spawn_b)  # Very strong penalty
        
        # Penalty for being too close to spawn size (encourage staying smaller for safety margin)
        if self.patch.a > spawn_a * 0.9:  # If > 90% of spawn size
            reward -= 5.0 * (self.patch.a - spawn_a * 0.9)  # Encourage staying smaller
        if self.patch.b > spawn_b * 0.9:  # If > 90% of spawn size
            reward -= 5.0 * (self.patch.b - spawn_b * 0.9)  # Encourage staying smaller
        
        # ===== CHECK IF PATCH IS TOO SMALL FOR AGENTS =====
        min_a_for_agents, min_b_for_agents = self._calculate_min_patch_size_for_agents()
        
        if self.patch.a < min_a_for_agents:
            reward -= 20.0 * (min_a_for_agents - self.patch.a)  # Penalty for being too small
        if self.patch.b < min_b_for_agents:
            reward -= 20.0 * (min_b_for_agents - self.patch.b)  # Penalty for being too small
        
        # ===== 7. DENSE SMOOTHNESS REWARD =====
        # Reward smooth control (reduce jerky movements)
        if hasattr(self, '_prev_accel') and hasattr(self, '_prev_steering'):
            accel_change = abs(self.patch.accel - self._prev_accel)
            steering_change = abs(self.patch.steering - self._prev_steering)
            
            # Reward smooth actions
            reward += 0.2 * (1.0 - min(accel_change / 2.0, 1.0))  # Less change = more reward
            reward += 0.2 * (1.0 - min(steering_change / 0.5, 1.0))  # Less change = more reward
        
        # Store for next step
        self._prev_accel = self.patch.accel
        self._prev_steering = self.patch.steering
        
        # ===== 8. SURVIVAL BONUS =====
        reward += 0.05  # Small bonus for every step survived
        
        # ===== 9. STUCK PENALTY =====
        # Penalty for not approaching goal and not moving
        if abs(dist_improve) < 1e-3 and self.patch.v < 0.3:
            reward -= 2.0

        # ===== 10. SE-MPC FEEDBACK (Patch learns MAXIMUM speed agents can achieve) =====
        # CRITICAL: This is how patch learns to go as fast as possible while agents can keep up
        if self.mpc_attempts > 0:
            feasibility_rate = self.mpc_successes / self.mpc_attempts
            # Strong reward for high MPC feasibility (agents can keep up = can go faster!)
            # Scale: 0.0 feasibility = -5.0, 1.0 feasibility = +5.0
            reward += 10.0 * (feasibility_rate - 0.5)  # Stronger feedback (was 0.5)
            
            # Bonus for perfect feasibility (encourage pushing speed limits)
            if feasibility_rate >= 0.95:
                reward += 3.0  # Extra bonus when agents can easily keep up

        safety_rate = self.safety_interventions / max(1, self.step_count * self.num_agents)
        if safety_rate > 0.2:  # Lower threshold (was 0.3) - be more sensitive
            # Strong penalty for frequent safety interventions (patch going too fast)
            reward -= 5.0 * (safety_rate - 0.2)  # Stronger penalty (was 0.5)
        
        return np.clip(reward, -50.0, 50.0)  # Wider clipping range for dense rewards
    
    def _check_termination(self, base_obs):
        """
        Check all termination conditions - ALL HARD TERMINATIONS.
        """
        terminated = False
        truncated = False
        termination_reason = None
        # patch_collision_rate = 0

        # if self.step_count < 10:
        #     return terminated, truncated, termination_reason
        
        # # HARD TERMINATION 1: Agent wall collision
        # if any(base_obs["collisions"][i] > 0.5 for i in range(self.num_agents)):
        #     terminated = True
        #     termination_reason = "agent_wall_collision"
        #     return terminated, truncated, termination_reason
        patch_collision, violated_points = self.patch.check_patch_boundary_wall_collision(
        self.base_env, n_points=32, violation_threshold=0.05  # Stricter threshold
        )
        if patch_collision:
            terminated = True
            termination_reason = f"patch_wall_collision ({len(violated_points)} points)"
            return terminated, truncated, termination_reason

        # HARD TERMINATION 2: Patch violates LIDAR safety
        lidar_safe, min_dist, lidar_info = self._check_lidar_safety(
            base_obs, safety_margin=0.05
        )
        # if not lidar_safe:
        #     terminated = True
        #     termination_reason = "lidar_safety_violation"
        #     return terminated, truncated, termination_reason

            # FIX: More robust check - ensure min_dist is valid
        if not np.isfinite(min_dist):
            min_dist = 0.5 

        if min_dist < 0.95:
            terminated = True
            termination_reason = "patch wall collision"
            return terminated, truncated, termination_reason

        # if self.patch.a < 1.0 or self.patch.b < 0.8:
        #     terminated = True 
        #     termination_reason = "patch too small"
        #     return terminated, truncated, termination_reason

        if hasattr(self, '_safe_min_a') and hasattr(self, '_safe_min_b'):
            if self.patch.a < self._safe_min_a or self.patch.b < self._safe_min_b:
                terminated = True 
                termination_reason = "patch too small"
                return terminated, truncated, termination_reason
        # if not lidar_safe:
        #     # a_violation = max(0, self.patch.a - max_safe_a)
        #     # b_violation = max(0, self.patch.b - max_safe_b)
        #     total_violation = min_dist 
            
        #     # Only terminate if violation is severe (> 1.0m total)
        #     if total_violation > 1.0:
        #         terminated = True
        #         # termination_reason = f"lidar_safety_violation (a={a_violation:.2f}, b={b_violation:.2f})"
        #         return terminated, truncated, termination_reason
        # # HARD TERMINATION 2: Patch boundary on/outside wall
        # patch_collision, violated_points = self.patch.check_patch_boundary_wall_collision(
        #     self.base_env, n_points=32, violation_threshold=0.15
        # )
        # if patch_collision:
        #     patch_collision_rate += 1
        # if patch_collision_rate > 20:
        #     terminated = True
        #     termination_reason = f"patch_boundary_wall_collision ({len(violated_points)} points)"

        
        patch_pos = np.array([self.patch.x, self.patch.y])
        progress, _= self._get_lap_progress(patch_pos)

        progress_delta = progress - self.lap_progress
        if progress_delta <0.01 and self.lap_progress > 0.95:
            terminated = True
            termination_reason = "lap_complete"
            return terminated, truncated, termination_reason
                        # Debug: print patch info
            # print(f"DEBUG [PatchEnv]: Patch boundary collision at step {self.step_count}")
            # print(f"  Patch center: ({self.patch.x:.2f}, {self.patch.y:.2f})")
            # print(f"  Patch size: a={self.patch.a:.2f}, b={self.patch.b:.2f}")
            # print(f"  Patch heading: {self.patch.theta:.2f} rad")
            # print(f"  Violated points: {len(violated_points)}")
            # return terminated, truncated, termination_reason
        
        # HARD TERMINATION 3: Agent outside patch
        # agents_inside = sum(1 for i in range(self.num_agents)
        #                   if self.patch.is_inside(
        #                       self.agent_states[i][0], 
        #                       self.agent_states[i][1],
        #                       margin=self.robot_radius
        #                   ))
        # if agents_inside < self.num_agents:
        #     terminated = True
        #     termination_reason = f"agent_outside_patch ({self.num_agents - agents_inside} agents)"
        #     return terminated, truncated, termination_reason
        
        # # HARD TERMINATION 4: Inter-agent collision
        # for i in range(self.num_agents):
        #     for j in range(i + 1, self.num_agents):
        #         xi, yi, _, _ = self.agent_states[i]
        #         xj, yj, _, _ = self.agent_states[j]
        #         dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
        #         if dist < 2 * self.robot_radius:
        #             terminated = True
        #             termination_reason = f"inter_agent_collision (agents {i}-{j})"
        #             Colliding_agents = [i for i in range(self.num_agents) if base_obs["collisions"][i] > 0.5]
        #             print(f"DEBUG [PatchEnv]: Agents {Colliding_agents} collided with walls at step {self.step_count}")
        #             return terminated, truncated, termination_reason
        
        # # SUCCESS TERMINATION: Lap complete
        # patch_pos = np.array([self.patch.x, self.patch.y])
        # progress, _, _ = self._get_lap_progress(patch_pos)
        # if progress > 0.95 and self.lap_progress > 0.9:
        #     terminated = True
        #     termination_reason = "lap_complete"
        #     return terminated, truncated, termination_reason
        
        # TRUNCATION: Max steps
        if self.step_count >= self.max_steps:
            truncated = True
            termination_reason = "max_steps"
        
        return terminated, truncated, termination_reason
    
    def _estimate_track_width(self, x, y, theta):
        """Estimate track width at a given position."""
        if self.base_env is None:
            return 4.0
        
        track = self.base_env.unwrapped.track
        occ_map = track.occupancy_map
        resolution = track.spec.resolution
        origin = track.spec.origin
        
        perp_angle_left = theta + np.pi / 2
        perp_angle_right = theta - np.pi / 2
        max_dist = 5.0
        step = 0.1
        
        dist_left = 0.0
        dist_right = 0.0
        
        for d in np.arange(0, max_dist, step):
            px = x + d * np.cos(perp_angle_left)
            py = y + d * np.sin(perp_angle_left)
            px_pixel = int((px - origin[0]) / resolution)
            py_pixel = int((py - origin[1]) / resolution)
            
            if (0 <= px_pixel < occ_map.shape[1] and 
                0 <= py_pixel < occ_map.shape[0]):
                if occ_map[py_pixel, px_pixel] < 0.5:
                    dist_left = d
                    break
            else:
                dist_left = d
                break
        else:
            dist_left = max_dist
        
        for d in np.arange(0, max_dist, step):
            px = x + d * np.cos(perp_angle_right)
            py = y + d * np.sin(perp_angle_right)
            px_pixel = int((px - origin[0]) / resolution)
            py_pixel = int((py - origin[1]) / resolution)
            
            if (0 <= px_pixel < occ_map.shape[1] and 
                0 <= py_pixel < occ_map.shape[0]):
                if occ_map[py_pixel, px_pixel] < 0.5:
                    dist_right = d
                    break
            else:
                dist_right = d
                break
        else:
            dist_right = max_dist
        
        return dist_left + dist_right
    
    def _generate_safe_agent_poses(self, patch_x, patch_y, patch_theta, patch_a, patch_b):
        """Generate agent poses INSIDE the patch with safe inter-agent spacing."""
        poses = np.zeros((self.num_agents, 3))
        
        edge_margin = self.robot_radius + 0.3
        min_inter_dist = 2 * self.robot_radius + 0.3
        
        eff_a = max(patch_a - edge_margin, 1.0)
        eff_b = max(patch_b - edge_margin, 0.8)
        
        if self.num_agents == 2:
            x_spread = 0.4
            y_spread = 0.0
            spacing = max(min_inter_dist * 0.6, x_spread / 2)
            local_positions = [
                (-0.5 * spacing, 0.0),
                (1.0 * spacing, 0.0),
            ]
        else:
            spacing = max(min_inter_dist, 2 * eff_a / (self.num_agents + 1))
            local_positions = []
            for i in range(self.num_agents):
                x_local = (i - (self.num_agents - 1) / 2) * spacing
                y_local = 0.0
                local_positions.append((x_local, y_local))
        
        cos_t = np.cos(patch_theta)
        sin_t = np.sin(patch_theta)
        
        for i, (x_local, y_local) in enumerate(local_positions):
            x_world = patch_x + x_local * cos_t - y_local * sin_t
            y_world = patch_y + x_local * sin_t + y_local * cos_t
            poses[i] = [x_world, y_world, patch_theta]
        
        if self.domain_randomize:
            for i in range(self.num_agents):
                poses[i, 0] += self._np_random.uniform(-0.1, 0.1)
                poses[i, 1] += self._np_random.uniform(-0.1, 0.1)
                poses[i, 2] += self._np_random.uniform(-0.05, 0.05)
        
        return poses


    def _compute_clearance_observation(self, lidar_distances):
        """
        Compute clearance = LIDAR distance - ellipse radius for each sector.
        
        Args:
            lidar_distances: Array of 16 LIDAR distances (one per sector) in meters
        
        Returns:
            clearance_obs: Array of 16 clearance values (normalized)
        """
        n_sectors = 16
        sector_angle = 2 * np.pi / n_sectors
        
        clearance_values = np.zeros(n_sectors)
        
        for sector_idx in range(n_sectors):
            # Angle of this sector (relative to patch heading)
            # Sector 0 is forward (patch heading), sector 8 is backward
            angle = sector_idx * sector_angle
            
            # Compute ellipse radius in this direction
            # Formula: r(θ) = (a * b) / sqrt((b * cos(θ))² + (a * sin(θ))²)
            cos_theta = np.cos(angle)
            sin_theta = np.sin(angle)
            
            denominator = np.sqrt((self.patch.b * cos_theta)**2 + (self.patch.a * sin_theta)**2)
            if denominator > 1e-6:
                r_ellipse = (self.patch.a * self.patch.b) / denominator
            else:
                r_ellipse = 0.0
            
            # Get LIDAR distance for this sector
            d_lidar = lidar_distances[sector_idx]
            
            # Compute clearance
            clearance = d_lidar - r_ellipse
            
            clearance_values[sector_idx] = clearance
        
        # Normalize clearance values
        # Positive = safe, negative = unsafe
        # Normalize to [-1, 1] range (assuming max clearance is ±5 meters)
        normalized_clearance = np.clip(clearance_values / 5.0, -1.0, 1.0)
        
        return normalized_clearance
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.patch_min_clearance = float('inf')
        if seed is not None:
            self._np_random = np.random.RandomState(seed)
        else:
            self._np_random = np.random.RandomState()
        
        # Create base environment (once)
        if self.base_env is None:
            # import f1tenth_gym  # Register env in subprocess
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
                    "reset_config": {"type": "rl_grid_static"},
                },
                render_mode="human" if self.render_mode == "human" else None,
            )
            self.base_env.reset()       
       
        # Get track info
        if self.waypoints is None:
            track = self.base_env.unwrapped.track
            if track.centerline is not None:
                self.waypoints = np.column_stack([
                    track.centerline.xs,
                    track.centerline.ys
                ])
                self.track_length = track.centerline.length
                self.start_x = track.centerline.xs[300]
                self.start_y = track.centerline.ys[300]
                if len(track.centerline.xs) > 1:
                    dx = track.centerline.xs[1] - track.centerline.xs[0]
                    dy = track.centerline.ys[1] - track.centerline.ys[0]
                    self.start_theta = np.arctan2(dy, dx)
                else:
                    self.start_theta = 0.0

        # Determine patch position and size
        patch_x = self.start_x
        patch_y = self.start_y
        patch_theta = self.start_theta
        
        #Estimate track width and set max patch size based on it
        track_width = self._estimate_track_width(patch_x, patch_y, patch_theta)
        # wall_margin = 1.0 #Increased for safety
        safety_margin = 0.5
        max_patch_width = track_width - (2 * safety_margin)
        # max_b = (track_width / 2) - wall_margin
        max_b = max_patch_width / 2.0

        min_b = 0.5 
        max_b_allowed = 1.5 
        max_b = np.clip(max_b, min_b, max_b_allowed)
        # max_a = 3.5
        
        # if self.domain_randomize:
        #     init_a = self._np_random.uniform(2.0, max_a)
        #     init_b = self._np_random.uniform(1.5, max_b)
        # else:
        #     #Conservative initial patch size
        init_b = max_b
        init_a = min(2.0, max_b * 1.5)
        # init_b = 1.8

        # Store safe initial size for validation checks
        self._safe_init_a = init_a
        self._safe_init_b = init_b
        min_a = init_a * 0.5
        min_b = init_b * 0.5
        self.patch_a_range = (min_a, init_a)  # Can shrink to 50%, max is spawn size
        self.patch_b_range = (min_b, init_b)

        self._safe_min_a = self._safe_init_a * 0.5
        self._safe_min_b = self._safe_init_b * 0.5
        
        # Create patch
        self.patch = DynamicPatch(
            x=patch_x, y=patch_y,
            theta=patch_theta, v=0.5,
            a=init_a, b=init_b
        )
        self.patch.a= init_a
        self.patch.b= init_b
        
        if self.domain_randomize:
            self.patch.randomize_dynamics(self._np_random)
            self.patch.a = init_a
            self.patch.b = init_b
        

        self._prev_patch_pos = np.array([self.patch.x, self.patch.y])
        self._prev_accel = 0.0
        self._prev_steering = 0.0
        # Generate agent poses
        # agent_poses = self._generate_safe_agent_poses(
        #     patch_x, patch_y, patch_theta, init_a, init_b
        # )
        agent_poses = self._position_agents_as_sensors(randomize=True)
        # Reset base env
        base_obs, info = self.base_env.reset(options={"poses": agent_poses})
        
        # Initialize agent states
        self.agent_states = []
        for i in range(self.num_agents):
            v = np.sqrt(base_obs["linear_vels_x"][i]**2 + 
                       base_obs["linear_vels_y"][i]**2)
            self.agent_states.append([
                base_obs["poses_x"][i],
                base_obs["poses_y"][i],
                base_obs["poses_theta"][i],
                0.5
            ])

        # === SE-MPC tracking state ===
        self.prev_v = [
            np.sqrt(base_obs["linear_vels_x"][i]**2 +
                    base_obs["linear_vels_y"][i]**2)
            for i in range(self.num_agents)
        ]
        self.mpc_successes = 0
        self.mpc_attempts = 0
        self.safety_interventions = 0
        self.cached_controls = [None] * self.num_agents
        self.cached_feasible = [False] * self.num_agents

        # CRITICAL: Update action range minimum based on agent positions
        # Patch must be at least large enough to contain all agents
        min_a_for_agents, min_b_for_agents = self._calculate_min_patch_size_for_agents()
        
        # Update action range: minimum is max(50% spawn, agent requirement)
        min_a = max(self._safe_min_a, min_a_for_agents)
        min_b = max(self._safe_min_b, min_b_for_agents)
        
        # Ensure minimum doesn't exceed spawn size
        min_a = min(min_a, self._safe_init_a)
        min_b = min(min_b, self._safe_init_b)
        
        self.patch_a_range = (min_a, self._safe_init_a)
        self.patch_b_range = (min_b, self._safe_init_b)

        # ===== VALIDATION: Check LIDAR safety FIRST (before discretization) =====
        # Get LIDAR data and check if patch respects LIDAR safety rule
        lidar_safe, min_dist, lidar_info = self._check_lidar_safety(
            base_obs, safety_margin=0.5
        )
        
        if not lidar_safe:
        #     print(f"WARNING [PatchEnv]: Patch violates LIDAR safety!")
        #     print(f"  Current: a={self.patch.a:.2f}, b={self.patch.b:.2f}")
        #     print(f"  Max safe: a={max_safe_a:.2f}, b={max_safe_b:.2f}")
        #     print(f"  LIDAR: forward={lidar_info['forward_lidar']:.2f}, "
        #           f"backward={lidar_info['backward_lidar']:.2f}, "
        #           f"left={lidar_info['left_lidar']:.2f}, "
        #           f"right={lidar_info['right_lidar']:.2f}")
            
            # Adjust patch to respect LIDAR safety
            safe_min_a = self._safe_min_a
            safe_min_b = self._safe_min_b
            self.patch.a = max( min(self.patch.a, min_dist - 0.5), safe_min_a)  # Don't go below 2.0
            self.patch.b = max( min(self.patch.b, min_dist - 0.5), safe_min_b)  # Don't go below 1.5
            # self.patch.a = min(self.patch.a, max(max_safe_a, 2.0))  # Don't go below 2.0
            # self.patch.b = min(self.patch.b, max(max_safe_b, 1.5))  # Don't go below 1.5
            
            # print(f"  Adjusted to: a={self.patch.a:.2f}, b={self.patch.b:.2f}")
        
        # Then check discretization (geometric check)
        patch_collision, violated_points = self.patch.check_patch_boundary_wall_collision(
            self.base_env, n_points=32, violation_threshold=0.15
        )
        if patch_collision:
            # If still colliding after LIDAR adjustment, reduce further
            safe_min_a = self._safe_min_a  # Use stored value
            safe_min_b = self._safe_min_b  
            self.patch.a = max(self.patch.a * 0.8, safe_min_a)
            self.patch.b = max(self.patch.b * 0.8, safe_min_b)
            # print(f"WARNING [PatchEnv]: Patch boundary still in walls after LIDAR adjustment!")

        # ===== VALIDATION: Check if agents/patch are in valid positions =====
        # Check if agents spawned in walls
        # if any(base_obs["collisions"][i] > 0.5 for i in range(self.num_agents)):
        #     print(f"WARNING [AgentEnv]: Agent spawned in wall! Retrying reset...")
            # Retry with different seed
            # return self.reset(seed=(seed + 1) if seed is not None else None, options=options)
        
        # Check if patch boundary is in walls
        patch_collision, violated_points = self.patch.check_patch_boundary_wall_collision(
            self.base_env, n_points=32
        )
        if patch_collision:
            # print(f"DEBUG [PatchEnv]: Patch boundary in walls ({len(violated_points)} points)! Adjusting patch size...")
            # Reduce patch size
            safe_min_a = self._safe_min_a  # Use stored value
            safe_min_b = self._safe_min_b
            self.patch.a = max(self.patch.a * 0.8, safe_min_a)
            self.patch.b = max(self.patch.b * 0.8, safe_min_b)
            # Regenerate agent poses with smaller patch
            agent_poses = self._generate_safe_agent_poses(
                self.patch.x, self.patch.y, self.patch.theta, 
                self.patch.a, self.patch.b
            )
            # Reset again with new poses
            base_obs, info = self.base_env.reset(options={"poses": agent_poses})
            # Update agent states
            self.agent_states = []
            for i in range(self.num_agents):
                v = np.sqrt(base_obs["linear_vels_x"][i]**2 + 
                           base_obs["linear_vels_y"][i]**2)
                self.agent_states.append([
                    base_obs["poses_x"][i],
                    base_obs["poses_y"][i],
                    base_obs["poses_theta"][i],
                    0.5 #Initial velocity
                ])
        
        # self.patch.a = 2.5
        # self.patch.b = 1.8
        # Reset tracking
        self.step_count = 0
        self.lap_progress = 0.0
        self.episode_reward = 0.0
        self.current_base_obs = base_obs
        self._current_waypoint_idx = 0
        
        return self._get_patch_observation(base_obs), {}
    
    def step(self, patch_action):
        """Execute one step: patch policy controls patch, agent policy controls agents."""
        self.step_count += 1
        dt = 0.05
        
        # 1. PATCH CONTROL (from patch policy)
        a, b, patch_accel, patch_steering = self._denormalize_patch_action(patch_action)

        #action used  = (1 -alpha) * action_previous + alpha * action_new
        #alpha = 0.1
        # alpha = 0.1
        smoothed_patch_accel = (1 - self.alpha) * self.patch.accel + self.alpha * patch_accel
        smoothed_patch_steering = (1 - self.alpha) * self.patch.steering + self.alpha * patch_steering

        self.patch.accel = smoothed_patch_accel
        self.patch.steering = smoothed_patch_steering

        # Make first step more conservative for patch
        if self.step_count == 1:
            patch_accel = np.clip(smoothed_patch_accel, -1.0, 1.0)
            patch_steering = np.clip(smoothed_patch_steering, -0.2, 0.2)
        self.patch.step(smoothed_patch_accel, smoothed_patch_steering, dt)

        prev_x, prev_y = self.patch.x, self.patch.y
        
        # Calculate minimum patch size needed to contain all agents
        min_a_for_agents, min_b_for_agents = self._calculate_min_patch_size_for_agents()
        
        # Constrain target size: must be >= agent requirement, <= spawn size
        spawn_a = getattr(self, '_safe_init_a', None)
        spawn_b = getattr(self, '_safe_init_b', None)
        
        # Ensure patch is large enough for agents
        a = max(a, min_a_for_agents)
        b = max(b, min_b_for_agents)
        
        # Ensure patch doesn't exceed spawn size
        if spawn_a is not None:
            a = min(a, spawn_a)
        if spawn_b is not None:
            b = min(b, spawn_b)
        
        # Pass spawn size as maximum to prevent expansion
        self.patch.update_shape(a, b, dt, max_a=spawn_a, max_b=spawn_b)
        self.patch.save_state()

        lidar_check_obs = self.current_base_obs if hasattr(self, 'current_base_obs') and self.current_base_obs is not None else None

        #Add a check if the previous point of the patch was outside the walls 
        if self.base_env is not None:
            try:
                track = self.base_env.unwrapped.track
                occ_map = track.occupancy_map
                resolution = track.spec.resolution
                origin = track.spec.origin
                
                #Convert to pixel coordinates
                px_prev = int((self.patch.x - origin[0]) / resolution)
                py_prev = int((self.patch.y - origin[1]) / resolution)
                
                center_outside  = (
                    px_prev < 0 or px_prev >= occ_map.shape[1] or 
                    py_prev < 0 or py_prev >= occ_map.shape[0] or 
                    occ_map[py_prev, px_prev] < 0.5
                )
                if center_outside:
                    self.patch.x = prev_x
                    self.patch.y = prev_y
                    self.patch.v = max(self.patch.v * 0.5, 0.5)
            except (AttributeError, KeyError):
                pass
        
        #Update visualization cache
        if self._track_vis_cache is not None:
            self._track_vis_cache.append((prev_x, prev_y))
            if len(self._track_vis_cache) > 100:
                self._track_vis_cache.popleft()
        
        #Update patch wall collision tracking

        # === 2. AGENT CONTROL VIA SE-MPC + SAFETY LAYER ===

        # Use last base observation for current agent poses
        base_obs = self.current_base_obs

        # Patch proxy for MPC & safety layer: use patch-local velocities and world center
        vx_patch, vy_patch = self.patch.get_velocity_vector()
        from types import SimpleNamespace
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

        # Agent positions in world & patch-local coordinates
        positions_world = [
            [base_obs["poses_x"][i], base_obs["poses_y"][i]]
            for i in range(self.num_agents)
        ]
        patch_cx_world = self.patch.x
        patch_cy_world = self.patch.y
        positions_local = [
            [p[0] - patch_cx_world, p[1] - patch_cy_world]
            for p in positions_world
        ]

        env_actions = np.zeros((self.num_agents, 2), dtype=np.float32)
        mpc_feasible = [False] * self.num_agents

        # Solve MPC every few steps and cache controls for efficiency
        solve_mpc = (self.step_count == 1) or (self.step_count % 5 == 0)

        for i in range(self.num_agents):
            x_local = positions_local[i][0]
            y_local = positions_local[i][1]
            theta_i = base_obs["poses_theta"][i]
            v_i = self.prev_v[i]

            x0_local = np.array([x_local, y_local, theta_i, v_i], dtype=np.float32)
            neighbors_local = [positions_local[j] for j in range(self.num_agents) if j != i]

            # 2a. MPC solve with soft containment
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

            mpc_feasible[i] = feasible

            # 2b. Safety layer hardens constraints
            u_safe, intervention = self.safety_layer.filter_control(
                u_opt if feasible else None,
                x0_local,
                patch_for_mpc,
                neighbors_local,
                dt=dt,
            )
            if intervention:
                self.safety_interventions += 1

            accel, steering = u_safe

            # Integrate agent speed (env expects [steering, speed])
            v_new = v_i + accel * dt
            v_new = np.clip(v_new, 0.5, 10.0)
            self.prev_v[i] = v_new

            env_actions[i] = [np.clip(steering, -0.4, 0.4), v_new]

        # 3. STEP BASE ENVIRONMENT ONCE with MPC actions
        base_obs, _, base_done, base_truncated, _ = self.base_env.step(env_actions)
        self.current_base_obs = base_obs

        # Sync agent states from simulator
        for i in range(self.num_agents):
            v_sim = np.sqrt(base_obs["linear_vels_x"][i]**2 +
                            base_obs["linear_vels_y"][i]**2)
            self.agent_states[i] = [
                base_obs["poses_x"][i],
                base_obs["poses_y"][i],
                base_obs["poses_theta"][i],
                max(v_sim, 0.5),
            ]
        
        # Check LIDAR safety with radial sweep
        lidar_safe, min_dist, lidar_info = self._check_lidar_safety(
            base_obs, safety_margin=0.2
        )
        self._lidar_safety_violation = not lidar_safe
        # self._lidar_safety_info = (max_safe_a, max_safe_b, lidar_info)  # Store for reward
        if min_dist < 1.0:
            # Inside ellipse - negative clearance (unsafe)
            # Estimate actual distance: if min_dist = 0.8, we're 20% inside
            clearance = -(1.0 - min_dist) * max(self.patch.a, self.patch.b)
        else:
            # Outside ellipse - positive clearance (safe)
            # Estimate actual distance: if min_dist = 1.2, we're 20% outside
            clearance = (min_dist - 1.0) * max(self.patch.a, self.patch.b)

        # Update for display (keep minimum over episode)
        if np.isfinite(clearance):
            self.patch_min_clearance = min(self.patch_min_clearance, clearance)
        else:
            # If clearance is invalid, set to a default value
            self.patch_min_clearance = -1.0  # Indicates invalid/unknown

        # 4. COMPUTE REWARD
        reward = self._compute_reward(base_obs, lidar_info)
        self.episode_reward += reward
        
        # this ensures progress_delta is calculated correctly next step 
        patch_pos = np.array([self.patch.x, self.patch.y])
        self.lap_progress, _ = self._get_lap_progress(patch_pos)

        # 5. CHECK TERMINATION
        terminated, truncated, termination_reason = self._check_termination(base_obs)
        
        obs = self._get_patch_observation(base_obs)
        
        info = {
            "episode_reward": self.episode_reward,
            "lap_progress": self.lap_progress,
            # "avg_reward": self.episode_reward / self.step_count,
            "Episode_steps": self.step_count,
            # "agents_inside": sum(1 for i in range(self.num_agents)
            #                    if self.patch.is_inside(self.agent_states[i][0], 
            #                                            self.agent_states[i][1])),
            "patch_size": (self.patch.a, self.patch.b),
            "patch_velocity": self.patch.v,
            "step_reward": reward,
            "termination_reason": termination_reason,
        }
        
        return obs, reward, terminated, truncated, info
    
    #Render the base environment
    def render(self):
        if self.base_env is not None:
            if self.render_mode == "human":
                self.base_env.render()
            elif self.render_mode == "rgb_array":
                return self.base_env.render()
        return None

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
        if self.base_env is not None:
            try:
                track = self.base_env.unwrapped.track
                occ_map = track.occupancy_map
                resolution = track.spec.resolution
                origin = track.spec.origin
                
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
                    wall_mask = occ_region < 0.5
                    if np.any(wall_mask):
                        # Create meshgrid for contour
                        X, Y = np.meshgrid(x_world, y_world)
                        
                        # Draw filled wall regions
                        self._ax.contourf(
                            X, Y, occ_region,
                            levels=[0, 0.5, 1.0],
                            colors=['darkgray', 'lightgray'],
                            alpha=0.5,
                            extend='neither'
                        )
                        
                        # Draw wall boundaries
                        self._ax.contour(
                            X, Y, occ_region,
                            levels=[0.5],
                            colors='black',
                            linewidths=1.5,
                            alpha=0.7
                        )
                
                # Draw centerline
                if hasattr(track, 'centerline') and track.centerline is not None:
                    centerline_x = track.centerline.xs
                    centerline_y = track.centerline.ys
                    self._ax.plot(centerline_x, centerline_y, 'g--', 
                                 linewidth=1.5, alpha=0.4, label='Track Centerline', zorder=1)
                
            except (AttributeError, KeyError, Exception) as e:
                # If track info not available, skip wall drawing
                pass
        
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

        try:
            self._ax.figure.canvas.draw()
            self._ax.figure.canvas.flush_events()
        except Exception as e:
            pass
        
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
        if self.base_env is not None:
            self.base_env.close()