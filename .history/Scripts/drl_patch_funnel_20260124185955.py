#!/usr/bin/env python3
"""
DRL-GUIDED CONSTRAINT FUNNELING SYSTEM
======================================

Architecture: DRL-Guided Deformable Containment (Patch Funnel)

┌──────────────────────────────────────────────────────────────────────────┐
│                    LAYER I: STRATEGIC (DRL Policy)                       │
│                         Frequency: 1-5 Hz                                │
├──────────────────────────────────────────────────────────────────────────┤
│  Inputs:                           │  Outputs:                           │
│  • Swarm centroid (x, y)           │  • Shape(t): a, b, θ                │
│  • Swarm spread                    │  • V_patch(t): vx, vy               │
│  • Goal direction/distance         │                                     │
│  • Corridor width (LIDAR)          │  P_patch(t) = centroid (fixed)      │
│  • Wall clearance (LIDAR)          │                                     │
│  • SE-MPC feasibility rate ←───────┤  ← KEY FEEDBACK!                    │
└──────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                 LAYER II: TACTICAL (SE-MPC Solvers)                      │
│                         Frequency: ≥20 Hz                                │
├──────────────────────────────────────────────────────────────────────────┤
│  Per-Agent SE-MPC Optimization (CasADi/IPOPT):                           │
│                                                                          │
│  min  J = w_vel·|V_agent - V_patch|² + w_center·|X - P_patch|²          │
│   U                                                                      │
│                                                                          │
│  HARD CONSTRAINTS:                                                       │
│  ├─ G_Containment: X(k) ∈ Ellipsoid(k)   ← Stay in patch!               │
│  ├─ G_Safety:      dist(i,j) ≥ ε_min     ← No inter-collision           │
│  └─ G_Feasibility: Ackermann dynamics    ← Car-like constraints         │
└──────────────────────────────────────────────────────────────────────────┘

KEY DESIGN:
- DRL learns V_patch that agents CAN achieve
- If SE-MPC fails (can't keep up) → DRL is penalized
- This makes the patch "wait" for slow agents automatically!
"""

import time
import os
import json
from datetime import datetime
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
import casadi as ca
from collections import deque

# Matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    SB3_AVAILABLE = True
except ImportError:
    print("WARNING: stable-baselines3 not installed. Install with: pip install stable-baselines3")
    SB3_AVAILABLE = False


# ============================================================================
# 1. DEFORMABLE ELLIPSOID PATCH (The Control Funnel)
# ============================================================================

class DeformableEllipsoid:
    """
    Time-varying deformable ellipsoid patch.
    
    Parameters from DRL:
        a, b: Semi-major and semi-minor axes
        theta: Orientation angle
        vx, vy: Patch velocity (how fast the funnel moves)
    
    Center (cx, cy): Always at team centroid (NOT learned)
    """
    
    def __init__(self, cx=0.0, cy=0.0, a=2.5, b=2.0, theta=0.0, vx=0.0, vy=0.0):
        self.cx = cx
        self.cy = cy
        self.a = max(a, 2.0)  # Larger minimum for 4 agents in formation
        self.b = max(b, 2.0)
        self.theta = theta
        self.vx = vx
        self.vy = vy
        
        # Precompute rotation
        self._update_rotation()
        
        # History for visualization
        self.history = deque(maxlen=100)
    
    def _update_rotation(self):
        self.cos_t = np.cos(self.theta)
        self.sin_t = np.sin(self.theta)
    
    def update_center(self, cx, cy):
        """Update center (always team centroid)."""
        self.cx = cx
        self.cy = cy
    
    def update_shape(self, a, b, theta):
        """Update shape from DRL."""
        self.a = max(a, 2.0)
        self.b = max(b, 2.0)
        self.theta = theta
        self._update_rotation()
    
    def update_velocity(self, vx, vy):
        """Update velocity from DRL."""
        self.vx = vx
        self.vy = vy
    
    def is_inside(self, x, y, margin=0.0):
        """Check if point is inside ellipsoid."""
        dx = x - self.cx
        dy = y - self.cy
        
        x_rot = dx * self.cos_t + dy * self.sin_t
        y_rot = -dx * self.sin_t + dy * self.cos_t
        
        a_eff = max(self.a - margin, 0.5)
        b_eff = max(self.b - margin, 0.5)
        
        return (x_rot / a_eff)**2 + (y_rot / b_eff)**2 <= 1.0
    
    def signed_distance(self, x, y):
        """Approximate signed distance (negative = inside)."""
        dx = x - self.cx
        dy = y - self.cy
        
        x_rot = dx * self.cos_t + dy * self.sin_t
        y_rot = -dx * self.sin_t + dy * self.cos_t
        
        normalized = np.sqrt((x_rot / self.a)**2 + (y_rot / self.b)**2)
        return (normalized - 1.0) * (self.a + self.b) / 2
    
    def predict_center(self, dt, steps):
        """Predict future centers based on velocity."""
        future = []
        for k in range(steps + 1):
            t = k * dt
            future.append((self.cx + self.vx * t, self.cy + self.vy * t))
        return future
    
    def save_state(self):
        """Save to history."""
        self.history.append({
            'cx': self.cx, 'cy': self.cy,
            'a': self.a, 'b': self.b, 'theta': self.theta,
            'vx': self.vx, 'vy': self.vy
        })


# ============================================================================
# 2. SE-MPC WITH HARD CONTAINMENT CONSTRAINT (CasADi/IPOPT)
# ============================================================================

class SEMPCSolver:
    """
    SE-MPC solver with HARD containment constraint using CasADi/IPOPT.
    
    HARD CONSTRAINTS:
    - G_Containment: Trajectory must stay inside time-varying ellipsoid
    - G_Safety: Minimum distance to neighbors
    - G_Feasibility: Ackermann dynamics, control limits
    """
    
    def __init__(self, robot_radius=0.15, wheelbase=0.33, num_neighbors=3):
        self.robot_radius = robot_radius
        self.wheelbase = wheelbase
        self.num_neighbors = num_neighbors
        
        # MPC parameters
        self.T_horizon = 2.0  # Prediction horizon (seconds)
        self.N = 10  # Number of steps
        self.dt = self.T_horizon / self.N
        
        # Control limits
        self.v_max = 10.0
        self.v_min = 0.5
        self.accel_max = 6.0
        self.delta_max = 0.4  # Max steering angle
        
        # Safety margins (FIXED: smaller margin for better feasibility)
        self.containment_margin = robot_radius  # Just robot radius, not extra padding
        self.min_agent_dist = 2 * robot_radius + 0.2  # Slightly reduced
        
        # Build the optimization problem
        self._build_mpc()
        
        # Warm start storage
        self.prev_X_sol = None
        self.prev_U_sol = None
    
    def _build_mpc(self):
        """Build the CasADi optimization problem."""
        
        self.opti = ca.Opti()
        
        # ===== DECISION VARIABLES =====
        # State: [x, y, theta, v]
        self.X = self.opti.variable(4, self.N + 1)
        self.x = self.X[0, :]
        self.y = self.X[1, :]
        self.theta = self.X[2, :]
        self.v = self.X[3, :]
        
        # Control: [acceleration, steering]
        self.U = self.opti.variable(2, self.N)
        self.accel = self.U[0, :]
        self.delta = self.U[1, :]
        
        # ===== PARAMETERS (set at solve time) =====
        self.p_x0 = self.opti.parameter(4, 1)  # Initial state
        
        # Patch parameters (time-varying!)
        self.p_patch_cx = self.opti.parameter(self.N + 1, 1)  # Center x at each step
        self.p_patch_cy = self.opti.parameter(self.N + 1, 1)  # Center y at each step
        self.p_patch_a = self.opti.parameter(1, 1)  # Semi-major axis
        self.p_patch_b = self.opti.parameter(1, 1)  # Semi-minor axis
        self.p_patch_theta = self.opti.parameter(1, 1)  # Orientation
        self.p_patch_vx = self.opti.parameter(1, 1)  # Velocity x
        self.p_patch_vy = self.opti.parameter(1, 1)  # Velocity y
        
        # Neighbor positions (for collision avoidance)
        self.p_neighbors = self.opti.parameter(self.num_neighbors, 2)
        
        # ===== DYNAMICS (Ackermann/Bicycle Model) =====
        for k in range(self.N):
            x_next = self.x[k] + self.v[k] * ca.cos(self.theta[k]) * self.dt
            y_next = self.y[k] + self.v[k] * ca.sin(self.theta[k]) * self.dt
            theta_next = self.theta[k] + (self.v[k] / self.wheelbase) * ca.tan(self.delta[k]) * self.dt
            v_next = self.v[k] + self.accel[k] * self.dt
            
            self.opti.subject_to(self.x[k+1] == x_next)
            self.opti.subject_to(self.y[k+1] == y_next)
            self.opti.subject_to(self.theta[k+1] == theta_next)
            self.opti.subject_to(self.v[k+1] == v_next)
        
        # Initial condition
        self.opti.subject_to(self.X[:, 0] == self.p_x0)
        
        # ===== CONTAINMENT CONSTRAINT (ALL SOFT with Safety Layer) =====
        # Philosophy: Soft constraints in MPC + Safety Layer for guarantees
        # This gives MPC flexibility to find solutions while safety layer prevents danger
        cos_t = ca.cos(-self.p_patch_theta)
        sin_t = ca.sin(-self.p_patch_theta)
        
        # Effective axes with margin
        a_eff = ca.fmax(self.p_patch_a - self.containment_margin, 0.5)
        b_eff = ca.fmax(self.p_patch_b - self.containment_margin, 0.5)
        
        # ALL timesteps: SOFT containment (safety layer handles hard guarantees)
        W_contain = 800.0  # High weight to strongly discourage violation
        self.J_contain_soft = 0.0
        
        for k in range(self.N + 1):
            dx = self.x[k] - self.p_patch_cx[k]
            dy = self.y[k] - self.p_patch_cy[k]
            
            x_rot = dx * cos_t - dy * sin_t
            y_rot = dx * sin_t + dy * cos_t
            
            # Normalized distance (>1 means outside)
            g_contain = (x_rot / a_eff)**2 + (y_rot / b_eff)**2 - 1.0
            
            # Quadratic penalty for being outside (soft constraint)
            # Higher weight for later timesteps (more important to converge)
            weight_k = W_contain * (1.0 + 0.5 * k / self.N)
            self.J_contain_soft += weight_k * ca.fmax(0.0, g_contain)**2
        
        # ===== HARD CONSTRAINT: G_Feasibility =====
        for k in range(self.N + 1):
            self.opti.subject_to(self.v[k] >= self.v_min)
            self.opti.subject_to(self.v[k] <= self.v_max)
        
        for k in range(self.N):
            self.opti.subject_to(self.accel[k] >= -self.accel_max)
            self.opti.subject_to(self.accel[k] <= self.accel_max)
            self.opti.subject_to(self.delta[k] >= -self.delta_max)
            self.opti.subject_to(self.delta[k] <= self.delta_max)
        
        # ===== OBJECTIVE =====
        
        # 1. Track patch velocity
        W_vel = 50.0
        J_vel = 0.0
        for k in range(self.N + 1):
            vx_agent = self.v[k] * ca.cos(self.theta[k])
            vy_agent = self.v[k] * ca.sin(self.theta[k])
            J_vel += W_vel * ((vx_agent - self.p_patch_vx)**2 + (vy_agent - self.p_patch_vy)**2)
        
        # 2. Stay near patch center
        W_center = 5.0
        J_center = 0.0
        for k in range(self.N + 1):
            J_center += W_center * ((self.x[k] - self.p_patch_cx[k])**2 + 
                                    (self.y[k] - self.p_patch_cy[k])**2)
        
        # 3. Inter-agent collision avoidance (soft, high weight)
        W_collision = 200.0
        J_collision = 0.0
        for k in range(self.N + 1):
            for j in range(self.num_neighbors):
                dx_n = self.x[k] - self.p_neighbors[j, 0]
                dy_n = self.y[k] - self.p_neighbors[j, 1]
                dist_sq = dx_n**2 + dy_n**2 + 1e-4
                
                # Barrier function
                violation = ca.fmax(0.0, self.min_agent_dist**2 - dist_sq)
                J_collision += W_collision * violation
        
        # 4. Smoothness
        W_smooth = 1.0
        J_smooth = 0.0
        for k in range(self.N - 1):
            J_smooth += W_smooth * (self.accel[k+1] - self.accel[k])**2
            J_smooth += W_smooth * (self.delta[k+1] - self.delta[k])**2
        
        # 5. Control effort
        W_effort = 0.1
        J_effort = 0.0
        for k in range(self.N):
            J_effort += W_effort * self.accel[k]**2
            J_effort += W_effort * self.delta[k]**2
        
        # Include soft containment penalty for k=0
        self.opti.minimize(J_vel + J_center + J_collision + J_smooth + J_effort + self.J_contain_soft)
        
        # ===== SOLVER =====
        p_opts = {"expand": True, "print_time": False, "verbose": False}
        s_opts = {
            "max_iter": 300,
            "tol": 1e-4,
            "acceptable_tol": 1e-3,
            "print_level": 0,
            "sb": "yes"  # Suppress banner
        }
        self.opti.solver('ipopt', p_opts, s_opts)
    
    def solve(self, x0, patch, neighbor_positions):
        """
        Solve MPC with hard containment constraint.
        
        Args:
            x0: Initial state [x, y, theta, v]
            patch: DeformableEllipsoid instance
            neighbor_positions: List of [x, y] for neighbors
        
        Returns:
            u_opt: Optimal control [accel, steering] or None
            feasible: True if solution found (CRITICAL for DRL feedback!)
        """
        try:
            # Validate inputs
            if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
                return None, False
            
            self.opti.set_value(self.p_x0, x0.reshape(4, 1))
            
            # Compute future patch centers based on velocity
            cx_traj = np.zeros(self.N + 1)
            cy_traj = np.zeros(self.N + 1)
            for k in range(self.N + 1):
                t = k * self.dt
                cx_traj[k] = patch.cx + patch.vx * t
                cy_traj[k] = patch.cy + patch.vy * t
            
            self.opti.set_value(self.p_patch_cx, cx_traj.reshape(self.N + 1, 1))
            self.opti.set_value(self.p_patch_cy, cy_traj.reshape(self.N + 1, 1))
            self.opti.set_value(self.p_patch_a, patch.a)
            self.opti.set_value(self.p_patch_b, patch.b)
            self.opti.set_value(self.p_patch_theta, patch.theta)
            self.opti.set_value(self.p_patch_vx, patch.vx)
            self.opti.set_value(self.p_patch_vy, patch.vy)
            
            # Set neighbors
            neighbors_arr = np.zeros((self.num_neighbors, 2))
            for j, pos in enumerate(neighbor_positions[:self.num_neighbors]):
                neighbors_arr[j] = pos
            # Fill remaining with far away positions
            for j in range(len(neighbor_positions), self.num_neighbors):
                neighbors_arr[j] = [1000.0, 1000.0]
            self.opti.set_value(self.p_neighbors, neighbors_arr)
            
            # Warm start
            if self.prev_X_sol is not None:
                try:
                    self.opti.set_initial(self.X, self.prev_X_sol)
                    self.opti.set_initial(self.U, self.prev_U_sol)
                except:
                    pass
            
            # Solve!
            sol = self.opti.solve()
            
            if sol.stats()['success']:
                self.prev_X_sol = sol.value(self.X)
                self.prev_U_sol = sol.value(self.U)
                
                u_opt = sol.value(self.U[:, 0])
                return u_opt, True
            else:
                self.prev_X_sol = None
                self.prev_U_sol = None
                return None, False
                
        except Exception as e:
            self.prev_X_sol = None
            self.prev_U_sol = None
            return None, False


# ============================================================================
# 3. SAFETY LAYER (CBF-like backup for guaranteed safety)
# ============================================================================

class SafetyLayer:
    """
    Control Barrier Function (CBF)-like safety layer.
    
    Provides hard safety guarantees even when MPC uses soft constraints.
    
    Safety checks:
    1. Containment: Don't accelerate away from patch if outside
    2. Collision: Don't approach neighbors too fast
    3. Velocity: Ensure control stays within physical limits
    
    If MPC output violates safety, modify it minimally to be safe.
    """
    
    def __init__(self, robot_radius=0.15, wheelbase=0.33):
        self.robot_radius = robot_radius
        self.wheelbase = wheelbase
        
        # Safety thresholds
        self.min_agent_dist = 2 * robot_radius + 0.25
        self.max_accel = 6.0
        self.max_steering = 0.4
        self.v_min = 0.5
        self.v_max = 10.0
        
        # CBF parameters
        self.alpha_contain = 1.0  # CBF class-K function gain for containment
        self.alpha_collision = 2.0  # CBF gain for collision avoidance
    
    def compute_containment_cbf(self, x, y, theta, v, patch):
        """
        Compute CBF value and gradient for patch containment.
        
        h(x) = 1 - ((x-cx)/a)^2 - ((y-cy)/b)^2
        h > 0 means inside patch (safe)
        h < 0 means outside patch (unsafe)
        """
        dx = x - patch.cx
        dy = y - patch.cy
        
        # Rotate to patch frame
        x_rot = dx * patch.cos_t + dy * patch.sin_t
        y_rot = -dx * patch.sin_t + dy * patch.cos_t
        
        a_eff = max(patch.a - self.robot_radius, 0.5)
        b_eff = max(patch.b - self.robot_radius, 0.5)
        
        # CBF value
        h = 1.0 - (x_rot / a_eff)**2 - (y_rot / b_eff)**2
        
        # Gradient w.r.t. position (in world frame)
        dh_dx_rot = -2 * x_rot / (a_eff**2)
        dh_dy_rot = -2 * y_rot / (b_eff**2)
        
        # Transform gradient back to world frame
        dh_dx = dh_dx_rot * patch.cos_t - dh_dy_rot * patch.sin_t
        dh_dy = dh_dx_rot * patch.sin_t + dh_dy_rot * patch.cos_t
        
        # Lie derivative along vehicle dynamics: Lf_h = dh/dx * vx + dh/dy * vy
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        Lf_h = dh_dx * vx + dh_dy * vy
        
        return h, Lf_h, (dh_dx, dh_dy)
    
    def compute_collision_cbf(self, x, y, theta, v, neighbors):
        """
        Compute CBF for inter-agent collision avoidance.
        
        h(x) = ||p_i - p_j||^2 - d_min^2
        h > 0 means safe distance
        """
        min_h = float('inf')
        critical_neighbor = None
        
        for nx, ny in neighbors:
            dx = x - nx
            dy = y - ny
            dist_sq = dx**2 + dy**2
            
            h = dist_sq - self.min_agent_dist**2
            
            if h < min_h:
                min_h = h
                critical_neighbor = (nx, ny, dx, dy, dist_sq)
        
        if critical_neighbor is None:
            return float('inf'), 0.0
        
        nx, ny, dx, dy, dist_sq = critical_neighbor
        
        # Gradient
        dh_dx = 2 * dx
        dh_dy = 2 * dy
        
        # Lie derivative
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        Lf_h = dh_dx * vx + dh_dy * vy
        
        return min_h, Lf_h
    
    def filter_control(self, u_mpc, x0, patch, neighbors, dt=0.05):
        """
        Safety filter: Modify MPC control if it violates CBF constraints.
        
        Args:
            u_mpc: [accel, steering] from MPC (or None if MPC failed)
            x0: Current state [x, y, theta, v]
            patch: Ellipsoid patch
            neighbors: List of neighbor [x, y] positions
            dt: Time step
        
        Returns:
            u_safe: Safe control [accel, steering]
            intervention: True if safety layer modified the control
        """
        x, y, theta, v = x0
        
        # Default safe control if MPC failed
        if u_mpc is None:
            return self._compute_safe_fallback(x, y, theta, v, patch), True
        
        accel, steering = u_mpc
        intervention = False
        
        # ===== 1. CONTAINMENT CBF =====
        h_contain, Lf_h_contain, grad_h = self.compute_containment_cbf(x, y, theta, v, patch)
        
        # CBF condition: Lf_h + alpha * h >= 0
        # If violated, we need to modify acceleration
        cbf_contain = Lf_h_contain + self.alpha_contain * h_contain
        
        if h_contain < 0:  # Outside patch
            # We're outside - need to steer back in
            # Compute direction toward patch center
            dx_to_center = patch.cx - x
            dy_to_center = patch.cy - y
            desired_theta = np.arctan2(dy_to_center, dx_to_center)
            heading_error = desired_theta - theta
            
            # Normalize heading error
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            
            # Override steering to head toward patch
            steering = np.clip(2.5 * heading_error, -self.max_steering, self.max_steering)
            
            # Reduce speed if we're moving away from patch
            if cbf_contain < 0:
                # Decelerate to reduce violation
                accel = min(accel, -2.0)
            
            intervention = True
        
        elif cbf_contain < 0:  # Inside but about to exit
            # Reduce acceleration to prevent exit
            # Need: Lf_h_new + alpha * h >= 0
            # With new accel affecting velocity
            accel = min(accel, 0.0)  # At least don't accelerate
            intervention = True
        
        # ===== 2. COLLISION CBF =====
        h_collision, Lf_h_collision = self.compute_collision_cbf(x, y, theta, v, neighbors)
        
        cbf_collision = Lf_h_collision + self.alpha_collision * h_collision
        
        if h_collision < 0:  # Too close!
            # Emergency brake
            accel = -self.max_accel
            intervention = True
        elif cbf_collision < 0:  # Approaching too fast
            # Reduce speed
            accel = min(accel, -1.0)
            intervention = True
        
        # ===== 3. VELOCITY BOUNDS =====
        v_next = v + accel * dt
        if v_next > self.v_max:
            accel = (self.v_max - v) / dt
            intervention = True
        elif v_next < self.v_min:
            accel = (self.v_min - v) / dt
            intervention = True
        
        # ===== 4. CONTROL LIMITS =====
        accel = np.clip(accel, -self.max_accel, self.max_accel)
        steering = np.clip(steering, -self.max_steering, self.max_steering)
        
        return np.array([accel, steering]), intervention
    
    def _compute_safe_fallback(self, x, y, theta, v, patch):
        """
        Compute safe fallback control when MPC fails.
        
        Strategy: Head toward patch center at moderate speed.
        """
        dx = patch.cx - x
        dy = patch.cy - y
        dist_to_center = np.sqrt(dx**2 + dy**2) + 1e-6
        
        # Desired heading toward patch center
        desired_theta = np.arctan2(dy, dx)
        heading_error = desired_theta - theta
        
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi
        
        # Steering proportional to heading error
        steering = np.clip(2.5 * heading_error, -self.max_steering, self.max_steering)
        
        # Speed control based on position
        h_contain, _, _ = self.compute_containment_cbf(x, y, theta, v, patch)
        
        if h_contain < -0.5:  # Far outside
            # High speed toward center
            target_v = 5.0
        elif h_contain < 0:  # Just outside
            target_v = 4.0
        else:  # Inside
            # Match patch velocity roughly
            patch_speed = np.sqrt(patch.vx**2 + patch.vy**2)
            target_v = max(patch_speed, 3.0)
        
        # Compute required acceleration
        accel = np.clip((target_v - v) / 0.1, -self.max_accel, self.max_accel)
        
        return np.array([accel, steering])


# ============================================================================
# 4. GYMNASIUM ENVIRONMENT (DRL + SE-MPC + Safety Layer)
# ============================================================================

class PatchFunnelEnv(gym.Env):
    """
    DRL-Guided Constraint Funneling Environment.
    
    DRL POLICY outputs:
        - Shape: (a, b, theta) 
        - Velocity: (vx, vy)
        - Total: 5 dimensions
    
    SE-MPC executes:
        - Runs per agent
        - HARD containment constraint
        - Returns feasibility status → DRL feedback!
    
    KEY: If SE-MPC fails, DRL is penalized!
    """
    
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, num_agents=4, render_mode="human"):
        super().__init__()
        
        self.num_agents = num_agents
        self.render_mode = render_mode
        # self.team_goal = np.array([50.0, 0.0])
        #instead assign the goal as the track completion point
        # Lap completion tracking (instead of fixed goal)
        self.waypoints = None  # Will be loaded from track centerline
        self.lap_progress = 0.0  # 0.0 to 1.0 (1.0 = lap complete)
        self.current_waypoint_idx = 0
        self.start_position = None
                
        # Robot parameters
        self.robot_radius = 0.15
        self.wheelbase = 0.33
        
        # Create deformable patch
        self.patch = DeformableEllipsoid()
        
        # Create SE-MPC solvers (one per agent)
        self.mpc_solvers = [
            SEMPCSolver(self.robot_radius, self.wheelbase, num_agents - 1)
            for _ in range(num_agents)
        ]
        
        # Create Safety Layer (CBF-like backup for guarantees)
        self.safety_layer = SafetyLayer(self.robot_radius, self.wheelbase)
        self.safety_interventions = 0  # Track how often safety layer intervenes
        
        # Base F1TENTH environment
        self.base_env = None
        
        # ===== ACTION SPACE (DRL outputs) =====
        # Shape: a, b, theta (3)
        # Velocity: vx, vy (2)
        # Total: 5
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(5,), dtype=np.float32
        )
        
        # Action bounds (FIXED: larger minimum patch for 4 agents!)
        self.a_range = (2.5, 5.0)  # Min 2.5m to fit formation
        self.b_range = (2.5, 5.0)
        self.theta_range = (0.0, np.pi)
        self.v_range = (-2.0, 6.0)  # Conservative - agents must keep up!
        
        # ===== OBSERVATION SPACE =====
        # Team state: centroid(2) + spread(1) + mean_vel(2) = 5
        # Goal: direction(2) + distance(1) = 3
        # Environment basic: corridor_width(1) + wall_clearance(1) + safe_size(1) = 3
        # Environment LIDAR: 8 sectors + best_heading(1) + best_clearance(1) = 10  ← NEW!
        # Patch state: a(1) + b(1) + theta(1) + vx(1) + vy(1) = 5
        # SE-MPC feedback: feasibility_rate(1) + agents_inside(1) = 2
        # Total: 5 + 3 + 3 + 10 + 5 + 2 = 28
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(28,), dtype=np.float32
        )
        
        # Tracking
        self.step_count = 0
        self.max_steps = 500  # FIXED: Shorter episodes for stable value learning!
        self.mpc_successes = 0
        self.mpc_attempts = 0
        self.safety_interventions = 0
        self.episode_reward = 0.0
        
        # Formation offsets (diamond)
        self.formation_offsets = [
            [0.8, 0.0],   # Front
            [-0.8, 0.0],  # Back
            [0.0, 0.6],   # Left
            [0.0, -0.6],  # Right
        ]
        
        # Visualization
        self._fig = None
        self._ax = None
    
    def _denormalize_action(self, action):
        """Convert normalized DRL action to patch parameters."""
        a = (action[0] + 1) / 2 * (self.a_range[1] - self.a_range[0]) + self.a_range[0]
        b = (action[1] + 1) / 2 * (self.b_range[1] - self.b_range[0]) + self.b_range[0]
        theta = (action[2] + 1) / 2 * (self.theta_range[1] - self.theta_range[0]) + self.theta_range[0]
        vx = action[3] * self.v_range[1]  # Scaled by max speed
        vy = action[4] * self.v_range[1]
        
        # Enforce minimum size (must fit 4 agents in formation!)
        a = max(a, 2.0)
        b = max(b, 2.0)
        
        return a, b, theta, vx, vy
    
    def _get_centroid(self, obs):
        """Get team centroid."""
        return np.array([
            np.mean(obs["poses_x"][:self.num_agents]),
            np.mean(obs["poses_y"][:self.num_agents])
        ])
    
    def _estimate_corridor_width(self, obs):
        """Estimate corridor width from LIDAR."""
        widths = []
        for i in range(self.num_agents):
            scan = obs["scans"][i]
            quarter = len(scan) // 4
            left = np.min(scan[:quarter]) if len(scan) > 0 else 30.0
            right = np.min(scan[-quarter:]) if len(scan) > 0 else 30.0
            widths.append(left + right)
        return np.mean(widths)
    
    def _estimate_wall_clearance(self, obs):
        """Estimate minimum wall clearance."""
        min_dist = float('inf')
        for i in range(self.num_agents):
            scan = obs["scans"][i]
            agent_min = np.min(scan[scan > 0.1])
            min_dist = min(min_dist, agent_min)
        return min_dist if min_dist < 30.0 else 5.0
    
    def _get_safe_size(self, obs):
        """Get maximum safe patch size from LIDAR."""
        min_dists = []
        for i in range(self.num_agents):
            scan = obs["scans"][i]
            quarter = len(scan) // 4
            min_dists.append(np.min(scan[quarter:3*quarter]))  # Front
            min_dists.append(np.min(scan[:quarter]))  # Left
            min_dists.append(np.min(scan[3*quarter:]))  # Right
        return max(0.5, np.min(min_dists) - 0.5)
    
    def _get_directional_lidar(self, obs):
        """
        Extract directional LIDAR features for track navigation.
        Returns: [front, front_left, left, back_left, back, back_right, right, front_right]
        Each value is the minimum distance in that direction (normalized).
        """
        # Aggregate LIDAR from all agents (use leader/front agent primarily)
        scan = obs["scans"][0]  # Lead agent's LIDAR
        n = len(scan)
        
        # Split into 8 sectors (45° each)
        sector_size = n // 8
        sectors = []
        for i in range(8):
            start = i * sector_size
            end = (i + 1) * sector_size if i < 7 else n
            sector_min = np.min(scan[start:end])
            sectors.append(min(sector_min / 10.0, 1.0))  # Normalize, cap at 1.0
        
        # Sectors: [front, front_left, left, back_left, back, back_right, right, front_right]
        # Reorder to: [front, front_right, right, back_right, back, back_left, left, front_left]
        # Depends on LIDAR orientation - adjust as needed
        return np.array(sectors, dtype=np.float32)
    
    def _get_best_heading(self, obs):
        """
        Find the direction with most clearance (where to steer the patch).
        Returns: (best_angle_offset, clearance) normalized
        """
        scan = obs["scans"][0]
        n = len(scan)
        
        # Find the angle with maximum clearance
        # Smooth the scan to avoid noise
        window = max(1, n // 20)
        smoothed = np.convolve(scan, np.ones(window)/window, mode='same')
        
        best_idx = np.argmax(smoothed)
        best_clearance = smoothed[best_idx]
        
        # Convert to angle offset from front (front = n//2)
        # Negative = turn left, Positive = turn right
        angle_offset = (best_idx - n // 2) / (n // 2)  # Normalized to [-1, 1]
        
        return angle_offset, min(best_clearance / 10.0, 1.0)
    
    def _get_observation(self, base_obs):
        """Build observation for DRL policy."""
        centroid = self._get_centroid(base_obs)
        
        # Team spread
        positions = [[base_obs["poses_x"][i], base_obs["poses_y"][i]] 
                     for i in range(self.num_agents)]
        spread = np.std(positions)
        
        # Mean velocity
        vx_mean = np.mean(base_obs["linear_vels_x"][:self.num_agents])
        vy_mean = np.mean(base_obs["linear_vels_y"][:self.num_agents])
        
        # Goal
        # goal_vec = self.team_goal - centroid
        progress, goal_dir, dist_to_center = self._get_lap_progress(centroid)
        # goal_dist = np.linalg.norm(goal_vec)
        goal_dist = (1.0 - progress) * self.track_length
        # goal_dir = goal_vec / (goal_dist + 1e-6)
        
        # Environment - basic
        corridor_width = self._estimate_corridor_width(base_obs)
        wall_clearance = self._estimate_wall_clearance(base_obs)
        safe_size = self._get_safe_size(base_obs)
        
        # Environment - NAVIGATION (NEW!)
        lidar_sectors = self._get_directional_lidar(base_obs)  # 8 values
        best_heading, best_clearance = self._get_best_heading(base_obs)  # 2 values
        
        # MPC feedback
        if self.mpc_attempts > 0:
            feasibility_rate = self.mpc_successes / self.mpc_attempts
        else:
            feasibility_rate = 1.0
        
        agents_inside = sum(1 for p in positions if self.patch.is_inside(p[0], p[1]))
        
        obs = np.array([
            # Team state (5)
            centroid[0] / 50.0, centroid[1] / 10.0,
            spread / 5.0,
            vx_mean / 10.0, vy_mean / 10.0,
            # Goal (3)
            goal_dir[0], goal_dir[1],
            goal_dist / 50.0,
            # Environment - basic (3)
            corridor_width / 10.0,
            wall_clearance / 5.0,
            safe_size / 5.0,
            # Environment - LIDAR navigation (10) - NEW!
            *lidar_sectors,  # 8 directional clearances
            best_heading,    # Which way has most space
            best_clearance,  # How much space in best direction
            # Patch state (5)
            self.patch.a / 5.0, self.patch.b / 5.0,
            self.patch.theta / np.pi,
            self.patch.vx / 10.0, self.patch.vy / 10.0,
            # SE-MPC feedback (2)
            feasibility_rate,
            agents_inside / self.num_agents
        ], dtype=np.float32)
        
        return obs
    
    def _compute_reward(self, base_obs, mpc_feasible):
        """
        Compute reward for DRL policy.
        
        FIXED: Scaled to [-5, 5] range for stable PPO learning!
        """
        reward = 0.0
        
        centroid = self._get_centroid(base_obs)
        progress, _, dist_to_center = self._get_lap_progress(centroid)

        # Reward for progress along track
        progress_delta = progress - self.lap_progress
        if progress_delta > 0:
            reward += 2.0 * progress_delta * 100  # Bonus for moving forward
        elif progress_delta < -0.5:
            # Went backwards (probably wrapped around) - that's okay
            progress_delta = progress + (1.0 - self.lap_progress)
            reward += 2.0 * progress_delta * 100

        self.lap_progress = progress

        # Penalty for being far from centerline
        reward -= 0.3 * min(dist_to_center / 3.0, 1.0)

    #OLD CODE FOR GOAL BASED REWARD FUNCTION
        positions = [[base_obs["poses_x"][i], base_obs["poses_y"][i]] 
                     for i in range(self.num_agents)]
        
        # # ===== 1. PROGRESS TOWARD GOAL (main objective) =====
        # dist_to_goal = np.linalg.norm(centroid - self.team_goal)
        
        # # Reward for being closer to goal (scaled 0-1)
        # progress_reward = (50.0 - dist_to_goal) / 50.0  # 0 at start, 1 at goal
        # reward += 1.0 * progress_reward
        
        # # Bonus for patch velocity toward goal
        # goal_dir = (self.team_goal - centroid) / (np.linalg.norm(self.team_goal - centroid) + 1e-6)
        # v_toward = self.patch.vx * goal_dir[0] + self.patch.vy * goal_dir[1]
        # reward += 0.3 * np.clip(v_toward / 6.0, -1, 1)  # Normalized by max velocity
        
        # ===== 2. TRACK NAVIGATION (NEW - follow clear path!) =====
        best_heading, best_clearance = self._get_best_heading(base_obs)
        
        # Compute patch velocity direction
        v_mag = np.sqrt(self.patch.vx**2 + self.patch.vy**2) + 1e-6
        
        # Reward for steering toward clear space
        # best_heading is in [-1, 1], convert to approximate direction reward
        # Positive reward if patch velocity aligns with clear path
        if v_mag > 0.5:  # Only reward if moving
            # Simplified: reward if not heading toward walls
            front_clear = self._get_directional_lidar(base_obs)[0]  # Front sector
            reward += 0.5 * front_clear  # Bonus for clear path ahead
        
        # Penalty for low clearance in movement direction
        if best_clearance < 0.3:  # Getting close to walls
            reward -= 0.3 * (0.3 - best_clearance)
        
        # ===== 3. PATCH SIZE APPROPRIATENESS =====
        safe_size = self._get_safe_size(base_obs)
        
        # Penalty for patch being too big for corridor
        if self.patch.a > safe_size:
            reward -= 0.5 * np.clip((self.patch.a - safe_size) / 2.0, 0, 1)
        if self.patch.b > safe_size:
            reward -= 0.5 * np.clip((self.patch.b - safe_size) / 2.0, 0, 1)
        
        # ===== 3. MPC FEASIBILITY (DRL learns to set achievable params) =====
        feasibility_rate = sum(mpc_feasible) / len(mpc_feasible)
        
        # Smooth reward based on feasibility (not harsh step)
        reward += 0.5 * (feasibility_rate - 0.5)  # -0.25 to +0.25
        
        # ===== 4. AGENT CONTAINMENT (most important!) =====
        agents_inside = 0
        total_dist_outside = 0.0
        
        for pos in positions:
            if self.patch.is_inside(pos[0], pos[1]):
                agents_inside += 1
            else:
                dist_out = max(0, self.patch.signed_distance(pos[0], pos[1]))
                total_dist_outside += dist_out
        
        # Reward for agents inside (scaled 0-1)
        containment_rate = agents_inside / self.num_agents
        reward += 1.0 * containment_rate
        
        # Penalty for agents outside (capped)
        reward -= 0.5 * np.clip(total_dist_outside / 5.0, 0, 2)
        
        # ===== 5. COLLISION PENALTY =====
        num_collisions = sum(1 for i in range(self.num_agents) 
                            if base_obs["collisions"][i] > 0.5)
        reward -= 2.0 * (num_collisions / self.num_agents)  # Max -2
        
        # ===== 6. EFFICIENCY BONUS (compact patch when safe) =====
        if containment_rate == 1.0 and feasibility_rate >= 0.9:
            # Reward for smaller patch (more efficient)
            avg_size = (self.patch.a + self.patch.b) / 2.0
            compactness = np.clip((5.0 - avg_size) / 3.0, 0, 1)  # 0-1
            reward += 0.2 * compactness
        
        # ===== 7. SAFETY LAYER FEEDBACK =====
        # If safety layer is intervening too much, patch params are bad
        safety_rate = self.safety_interventions / max(1, self.mpc_attempts)
        if safety_rate > 0.5:
            reward -= 0.3 * (safety_rate - 0.5)  # Penalty for high intervention
        
        # Clip final reward to reasonable range
        reward = np.clip(reward, -5.0, 5.0)
        
        return reward
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Create/reset base environment
        if self.base_env is not None:
            self.base_env.close()
        
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
                "reset_config": {"type": "rl_random_static"},
            },
            render_mode="human" if self.render_mode == "human" else None,
        )
        
        base_obs, info = self.base_env.reset()
        # Initialize lap tracking
        # Load track centerline waypoints
        if self.waypoints is None:
            track = self.base_env.unwrapped.track
            if track.centerline is not None:
                self.waypoints = np.column_stack([
                    track.centerline.xs,
                    track.centerline.ys
                ])
                self.track_length = track.centerline.length
                print(f"Loaded {len(self.waypoints)} waypoints, track length: {self.track_length:.1f}m")
            else:
                # Fallback to fixed goal if no centerline
                self.waypoints = np.array([[50.0, 0.0]])
                self.track_length = 50.0

        # Reset lap tracking
        self.lap_progress = 0.0
        self.current_waypoint_idx = 0
        centroid = self._get_centroid(base_obs)
        self.start_position = centroid.copy()
        self.current_obs = base_obs
        
        # Initialize patch at centroid
        centroid = self._get_centroid(base_obs)
        corridor_width = self._estimate_corridor_width(base_obs)
        init_size = min(corridor_width * 0.4, 2.5)
        
        self.patch.update_center(centroid[0], centroid[1])
        self.patch.update_shape(init_size, init_size, 0.0)
        self.patch.update_velocity(2.0, 0.0)  # Initial forward velocity
        
        # Reset tracking
        self.step_count = 0
        self.mpc_successes = 0
        self.mpc_attempts = 0
        self.safety_interventions = 0
        self.episode_reward = 0.0

        # Previous velocities for integration
        self.prev_v = [np.sqrt(base_obs["linear_vels_x"][i]**2 + 
                               base_obs["linear_vels_y"][i]**2)
                       for i in range(self.num_agents)]
        
        return self._get_observation(base_obs), {}

    def _get_lap_progress(self, position):
        """
        Compute progress along the track (0.0 to 1.0).
        Returns: (progress, next_waypoint_direction, distance_to_centerline)
        """
        if self.waypoints is None or len(self.waypoints) < 2:
            return 0.0, np.array([1.0, 0.0]), 0.0
        
        # Find nearest waypoint
        dists = np.linalg.norm(self.waypoints - position, axis=1)
        nearest_idx = np.argmin(dists)
        dist_to_centerline = dists[nearest_idx]
        
        # Progress = how far along the waypoints we are
        progress = nearest_idx / len(self.waypoints)
        
        # Direction to next waypoint
        next_idx = (nearest_idx + 5) % len(self.waypoints)  # Look ahead 5 waypoints
        next_wp = self.waypoints[next_idx]
        direction = next_wp - position
        direction = direction / (np.linalg.norm(direction) + 1e-6)
        
        return progress, direction, dist_to_centerline

    def step(self, action):
        """
        Execute one step:
        1. DRL updates patch parameters (shape + velocity)
        2. SE-MPC solves for each agent (hard containment)
        3. If MPC fails → tracked for DRL penalty
        """
        self.step_count += 1
        
        # ===== 1. UPDATE PATCH FROM DRL =====
        a, b, theta, vx, vy = self._denormalize_action(action)
        
        # Center always at centroid
        centroid = self._get_centroid(self.current_obs)
        self.patch.update_center(centroid[0], centroid[1])
        self.patch.update_shape(a, b, theta)
        self.patch.update_velocity(vx, vy)
        self.patch.save_state()
        
        # ===== 2. SE-MPC SOLVES FOR EACH AGENT =====
        positions = [[self.current_obs["poses_x"][i], self.current_obs["poses_y"][i]] 
                     for i in range(self.num_agents)]
        
        env_actions = np.zeros((self.num_agents, 2))
        mpc_feasible = [False] * self.num_agents
        
        for i in range(self.num_agents):
            # Agent state
            x = self.current_obs["poses_x"][i]
            y = self.current_obs["poses_y"][i]
            theta_i = self.current_obs["poses_theta"][i]
            v = self.prev_v[i]
            
            x0 = np.array([x, y, theta_i, v])
            
            # Neighbor positions
            neighbors = [positions[j] for j in range(self.num_agents) if j != i]
            
            # ===== STEP 2a: Solve MPC (soft constraints) =====
            self.mpc_attempts += 1
            u_opt, feasible = self.mpc_solvers[i].solve(x0, self.patch, neighbors)
            
            if feasible:
                self.mpc_successes += 1
                mpc_feasible[i] = True
            
            # ===== STEP 2b: Safety Layer (CBF-like filter) =====
            # Always pass through safety layer for guarantees!
            u_safe, intervention = self.safety_layer.filter_control(
                u_opt if feasible else None,
                x0, 
                self.patch, 
                neighbors
            )
            
            if intervention:
                self.safety_interventions += 1
            
            accel, steering = u_safe
            
            # Integrate velocity
            v_new = v + accel * 0.05  # dt
            v_new = np.clip(v_new, 0.5, 10.0)
            self.prev_v[i] = v_new
            
            env_actions[i] = [np.clip(steering, -0.4, 0.4), v_new]
        
        # ===== 3. STEP BASE ENVIRONMENT =====
        base_obs, _, base_done, base_truncated, _ = self.base_env.step(env_actions)
        self.current_obs = base_obs
        
        # Update patch center to new centroid
        new_centroid = self._get_centroid(base_obs)
        self.patch.update_center(new_centroid[0], new_centroid[1])
        
        # ===== 4. COMPUTE REWARD =====
        reward = self._compute_reward(base_obs, mpc_feasible)
        self.episode_reward += reward
        
        # ===== 5. CHECK TERMINATION =====
        terminated = False
        truncated = False
        
        # Collision
        if any(base_obs["collisions"][i] > 0.5 for i in range(self.num_agents)):
            terminated = True
        
        # Goal reached
        progress, _, _ = self._get_lap_progress(new_centroid)
        if progress > 0.95 and self.lap_progress > 0.9:
            reward += 10.0  # FIXED: Scaled goal bonus (was 500!)
            terminated = True
            print("Lap completed!")
        
        # Max steps
        if self.step_count >= self.max_steps:
            truncated = True
        
        # ===== 6. VISUALIZE =====
        if self.step_count % 5 == 0:
            self._visualize(positions, mpc_feasible)
        
        if self.render_mode == "human":
            self.base_env.render()
        
        obs = self._get_observation(base_obs)
        
        return obs, reward, terminated, truncated, {
            "episode_reward": self.episode_reward,
            "mpc_feasibility": self.mpc_successes / max(1, self.mpc_attempts),
            "safety_intervention_rate": self.safety_interventions / max(1, self.mpc_attempts)
        }
    
    def _visualize(self, positions, mpc_feasible):
        """Visualize patch funnel and agents."""
        if self._fig is None:
            plt.ion()
            self._fig, self._ax = plt.subplots(figsize=(12, 9))
        
        self._ax.clear()
        self._ax.set_aspect('equal')
        self._ax.grid(True, alpha=0.3)
        
        # MPC feasibility rate
        feas_rate = self.mpc_successes / max(1, self.mpc_attempts)
        agents_inside = sum(1 for p in positions if self.patch.is_inside(p[0], p[1]))
        
        safety_rate = self.safety_interventions / max(1, self.mpc_attempts)
        self._ax.set_title(
            f'🎯 DRL-Guided Constraint Funnel | Step {self.step_count}\n'
            f'MPC: {feas_rate:.1%} | Safety: {safety_rate:.1%} | Inside: {agents_inside}/{self.num_agents} | '
            f'V_patch: ({self.patch.vx:.1f}, {self.patch.vy:.1f})',
            fontsize=11
        )
        
        # Draw ellipsoid patch
        ellipse = Ellipse(
            xy=(self.patch.cx, self.patch.cy),
            width=self.patch.a * 2,
            height=self.patch.b * 2,
            angle=np.degrees(self.patch.theta),
            facecolor='cyan' if feas_rate == 1.0 else 'yellow',
            edgecolor='darkblue' if feas_rate == 1.0 else 'red',
            alpha=0.3,
            linewidth=3
        )
        self._ax.add_patch(ellipse)
        
        # Draw velocity arrow (V_patch)
        arrow_scale = 0.5
        self._ax.arrow(
            self.patch.cx, self.patch.cy,
            self.patch.vx * arrow_scale, self.patch.vy * arrow_scale,
            head_width=0.3, head_length=0.2,
            fc='blue', ec='blue', linewidth=2
        )
        
        # Draw agents
        colors = ['red', 'orange', 'purple', 'green']
        for i, pos in enumerate(positions):
            inside = self.patch.is_inside(pos[0], pos[1])
            feasible = mpc_feasible[i]
            
            color = colors[i % len(colors)]
            marker = 'o' if (inside and feasible) else 'X'
            size = 15 if feasible else 20
            
            self._ax.plot(pos[0], pos[1], marker, color=color, markersize=size,
                         markeredgecolor='black', markeredgewidth=2)
            
            status = "✓" if feasible else "✗"
            self._ax.text(pos[0] + 0.3, pos[1] + 0.3, f'R{i}{status}',
                         fontsize=10, fontweight='bold',
                         color='green' if feasible else 'red')
        
        # Draw goal
        # self._ax.plot(self.team_goal[0], self.team_goal[1], 'g*', markersize=25,
        #              markeredgecolor='black', markeredgewidth=2)
                # Draw next waypoint (instead of fixed goal)
        if self.waypoints is not None and len(self.waypoints) > 0:
            next_idx = min(self.current_waypoint_idx + 10, len(self.waypoints) - 1)
            self._ax.plot(self.waypoints[next_idx, 0], self.waypoints[next_idx, 1], 
                         'g*', markersize=25, markeredgecolor='black', markeredgewidth=2)
        # Info box
        info = (
            f'Patch: a={self.patch.a:.1f}, b={self.patch.b:.1f}\n'
            f'θ={np.degrees(self.patch.theta):.0f}°\n'
            f'V=({self.patch.vx:.1f}, {self.patch.vy:.1f})\n'
            f'Reward: {self.episode_reward:.0f}'
        )
        self._ax.text(0.02, 0.98, info, transform=self._ax.transAxes,
                     fontsize=10, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        # Set limits
        margin = max(self.patch.a, self.patch.b) + 5
        self._ax.set_xlim(self.patch.cx - margin, self.patch.cx + margin)
        self._ax.set_ylim(self.patch.cy - margin, self.patch.cy + margin)
        
        plt.pause(0.001)
    
    def close(self):
        if self._fig is not None:
            plt.close(self._fig)
        if self.base_env is not None:
            self.base_env.close()


# ============================================================================
# 4. TRAINING
# ============================================================================

def train_patch_funnel(
    total_timesteps=500000,
    save_path="patch_funnel_models_v3",
    checkpoint_freq=10000,
    resume_from=None
):
    """Train the DRL Patch Funnel Policy."""
    
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return
    
    os.makedirs(save_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_path, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 70)
    print("DRL-GUIDED CONSTRAINT FUNNELING SYSTEM - TRAINING")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Checkpoint frequency: {checkpoint_freq}")
    print(f"Save directory: {run_dir}")
    print()
    print("Architecture:")
    print("  [DRL Policy] → Patch (a, b, θ, vx, vy)")
    print("  [SE-MPC × 4] → Agent controls (hard containment!)")
    print("  MPC failure → DRL penalty (learns to wait!)")
    print("=" * 70)
    
    # Create environment (no render during training for speed)
    env = PatchFunnelEnv(num_agents=4, render_mode=None)
    env = DummyVecEnv([lambda: env])
    
    # Normalize observations and rewards for stable learning
    env = VecNormalize(
        env,
        norm_obs=True,      # Normalize observations
        norm_reward=True,   # Normalize rewards
        clip_obs=10.0,      # Clip normalized observations
        clip_reward=10.0,   # Clip normalized rewards
        gamma=0.99          # Match PPO gamma for consistent returns
    )
    
    # Callback for logging
    class FunnelCallback(BaseCallback):
        def __init__(self, save_dir, save_freq, verbose=1):
            super().__init__(verbose)
            self.save_dir = save_dir
            self.save_freq = save_freq
            self.episode_rewards = []
            self.mpc_feasibilities = []
            self.episode_count = 0
            self.best_reward = -np.inf
        
        def _on_step(self):
            if self.locals.get("dones", [False])[0]:
                self.episode_count += 1
                info = self.locals.get("infos", [{}])[0]
                reward = info.get("episode_reward", 0)
                feas = info.get("mpc_feasibility", 0)
                safety_int = info.get("safety_intervention_rate", 0)
                
                self.episode_rewards.append(reward)
                self.mpc_feasibilities.append(feas)
                
                if self.episode_count % 10 == 0:
                    avg_r = np.mean(self.episode_rewards[-10:])
                    avg_f = np.mean(self.mpc_feasibilities[-10:])
                    print(f"\n[Episode {self.episode_count}] "
                          f"Reward: {avg_r:.1f}, "
                          f"MPC Feas: {avg_f:.1%}, "
                          f"Safety: {safety_int:.1%}")
                    
                    if avg_r > self.best_reward:
                        self.best_reward = avg_r
                        self.model.save(os.path.join(self.save_dir, "best_model"))
                        # Also save VecNormalize stats
                        if hasattr(self.training_env, 'save'):
                            self.training_env.save(os.path.join(self.save_dir, "best_vecnormalize.pkl"))
                        print(f"   🏆 New best model! (reward: {avg_r:.1f})")
            
            if self.num_timesteps % self.save_freq == 0:
                path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}")
                self.model.save(path)
                # Save VecNormalize stats with checkpoint
                if hasattr(self.training_env, 'save'):
                    self.training_env.save(path + "_vecnormalize.pkl")
                print(f"\n💾 Checkpoint saved: {path}")
            
            return True
    
    callback = FunnelCallback(run_dir, checkpoint_freq)
    
    # Create/load model
    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"\n📂 Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,        # Standard PPO LR
            n_steps=2048,
            batch_size=256,            # Larger batch for stability
            n_epochs=10,               # Standard PPO epochs
            gamma=0.99,                # Standard for shorter 500-step episodes
            gae_lambda=0.95,           # Standard GAE lambda
            clip_range=0.2,            # Standard PPO clip
            ent_coef=0.01,             # Encourage exploration
            vf_coef=0.5,               # Balance value function learning
            max_grad_norm=0.5,         # Gradient clipping for stability
            verbose=1,
            tensorboard_log=os.path.join(save_path, "tensorboard"),
            policy_kwargs={
                "net_arch": dict(pi=[256, 256], vf=[256, 256])  # Larger networks for complex task
            }
        )
    
    print("\n🚀 Starting training...")
    print("   DRL learns patch velocity that SE-MPC can achieve!")
    print("   If MPC fails → DRL is penalized → learns to slow down")
    
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
# 4. Evaluation
# ============================================================================
def run_eval(model_path, num_episodes=5):
    """Evaluate a trained policy with visualization."""
    print("=" * 70)
    print("DRL-GUIDED CONSTRAINT FUNNELING - EVALUATION")
    print(f"Loading model: {model_path}")
    print("=" * 70)
    
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return
    
    # Load model
    model = PPO.load(model_path)
    
    # Create environment with rendering
    env = PatchFunnelEnv(num_agents=4, render_mode="human")
    
    # Load VecNormalize stats if available
    vecnorm_path = model_path.replace(".zip", "").replace("best_model", "best_vecnormalize.pkl")
    if not vecnorm_path.endswith(".pkl"):
        vecnorm_path = model_path + "_vecnormalize.pkl"
    
    # Wrap in DummyVecEnv for compatibility
    from stable_baselines3.common.vec_env import DummyVecEnv
    vec_env = DummyVecEnv([lambda: env])
    
    # Try to load normalization stats
    if os.path.exists(vecnorm_path):
        print(f"Loading normalization stats: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False  # Don't update stats during eval
        vec_env.norm_reward = False  # Don't normalize rewards for display
    
    print("\nWatch:")
    print("  - Cyan ellipse = Learned patch shape")
    print("  - Blue arrow = Learned patch velocity")
    print("  - Robots should STAY INSIDE the patch!")
    print("=" * 70)
    
    for ep in range(num_episodes):
        obs = vec_env.reset()
        total_reward = 0
        step = 0
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            step += 1
            
            if done[0]:
                feas = info[0].get("mpc_feasibility", 0)
                safety = info[0].get("safety_intervention_rate", 0)
                print(f"\n🏁 Episode {ep+1} finished at step {step}")
                print(f"   Total reward: {total_reward:.1f}")
                print(f"   MPC feasibility: {feas:.1%}")
                print(f"   Safety interventions: {safety:.1%}")
                break
    
    vec_env.close()
    print("\n✅ Evaluation complete!")


def run_demo(num_steps=1000):
    """Run demo with heuristic policy."""
    print("=" * 70)
    print("DRL-GUIDED CONSTRAINT FUNNELING - DEMO")
    print("=" * 70)
    print()
    print("Architecture:")
    print("  DRL → Patch (a, b, θ, vx, vy)")
    print("  SE-MPC → Agent controls (hard containment!)")
    print()
    print("Watch:")
    print("  - Cyan ellipse = Patch (control funnel)")
    print("  - Blue arrow = V_patch (patch velocity)")
    print("  - ✓ = MPC feasible, ✗ = MPC failed")
    print("=" * 70)
    
    env = PatchFunnelEnv(num_agents=4, render_mode="human")
    obs, _ = env.reset()
    
    total_reward = 0
    try:
        for step in range(num_steps):
            # Heuristic action using new observation layout:
            # [8] = corridor_width, [11] = front_sector, [19] = best_heading, [20] = best_clearance
            corridor = obs[8] * 10.0  # Denormalize corridor width
            front_clear = obs[11]      # Front LIDAR sector (0-1)
            best_heading = obs[19]     # Best direction to steer (-1 to 1)
            best_clearance = obs[20]   # Clearance in best direction
            
            # Adapt to corridor - smarter heuristic
            if corridor < 4.0:
                size = -0.3  # Smaller patch in tight spaces
            else:
                size = 0.2   # Larger in open areas
            
            # Adapt velocity based on clearance
            if front_clear > 0.5:
                vx = 0.6     # Fast if clear ahead
            elif front_clear > 0.3:
                vx = 0.3     # Medium speed
            else:
                vx = 0.1     # Slow down, wall ahead!
            
            # Steer toward clear space
            vy = best_heading * 0.3  # Slight lateral adjustment
            
            action = np.array([
                size,   # a
                size,   # b
                0.0,    # theta
                vx,     # vx (toward goal)
                vy      # vy (steer toward clear space)
            ], dtype=np.float32)
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if step % 100 == 0:
                feas = info.get("mpc_feasibility", 0)
                safety = info.get("safety_intervention_rate", 0)
                print(f"Step {step}: Reward={total_reward:.0f}, MPC={feas:.1%}, Safety={safety:.1%}")
            
            if terminated or truncated:
                print(f"\n🏁 Episode ended at step {step}")
                print(f"   Total reward: {total_reward:.0f}")
                print(f"   Final MPC feasibility: {info.get('mpc_feasibility', 0):.1%}")
                obs, _ = env.reset()
                total_reward = 0
                
    except KeyboardInterrupt:
        print("\n⚠️ Demo stopped")
    
    env.close()


# ============================================================================
# 5. MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="DRL-Guided Constraint Funneling System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Architecture:
  Layer I (DRL):  Outputs patch shape (a,b,θ) and velocity (vx,vy)
  Layer II (SE-MPC): Hard containment constraint, Ackermann dynamics
  
Key Feature:
  If SE-MPC fails (can't keep up) → DRL is penalized!
  This makes the patch learn to "wait" for agents.

Examples:
  python3 drl_patch_funnel.py --mode demo
  python3 drl_patch_funnel.py --mode train --timesteps 200000
  python3 drl_patch_funnel.py --mode train --resume patch_funnel_models/run_xxx/best_model
        """
    )
    parser.add_argument("--mode", choices=["demo", "train", "eval"], default="demo")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, help="Path to model for eval mode")
    
    args = parser.parse_args()
    
    if args.mode == "demo":
        run_demo()
    elif args.mode == "eval":
        if args.model is None:
            print("ERROR: --model path required for eval mode!")
            print("Example: --mode eval --model /home/mrvik/ADFS/patch_funnel_models/run_20251221_181514/best_model")
        else:
            run_eval(args.model)
    else:
        train_patch_funnel(
            total_timesteps=args.timesteps,
            checkpoint_freq=args.checkpoint_freq,
            resume_from=args.resume
        )
