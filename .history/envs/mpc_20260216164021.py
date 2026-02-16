"""
SE-MPC and safety layer module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import casadi as ca
import numpy as np


@dataclass
class MPCConfig:
    robot_radius: float = 0.15
    wheelbase: float = 0.33
    num_neighbors: int = 1
    horizon_seconds: float = 1.0
    horizon_steps: int = 5
    v_min: float = 0.5
    v_max: float = 10.0
    accel_max: float = 6.0
    steering_max: float = 0.4
    containment_margin: float = 0.15
    min_agent_dist: float = 0.5


class SEMPCSolver:
    """CasADi-based soft-containment MPC."""

    def __init__(self, config: MPCConfig | None = None) -> None:
        self.config = config or MPCConfig()
        self.dt = self.config.horizon_seconds / self.config.horizon_steps
        self.prev_X_sol = None
        self.prev_U_sol = None
        self._build_problem()

    def _build_problem(self) -> None:
        N = self.config.horizon_steps
        self.opti = ca.Opti()
        self.X = self.opti.variable(4, N + 1)
        self.U = self.opti.variable(2, N)

        self.x = self.X[0, :]
        self.y = self.X[1, :]
        self.theta = self.X[2, :]
        self.v = self.X[3, :]
        self.accel = self.U[0, :]
        self.delta = self.U[1, :]

        self.p_x0 = self.opti.parameter(4, 1)
        self.p_patch_cx = self.opti.parameter(N + 1, 1)
        self.p_patch_cy = self.opti.parameter(N + 1, 1)
        self.p_patch_a = self.opti.parameter(1, 1)
        self.p_patch_b = self.opti.parameter(1, 1)
        self.p_patch_theta = self.opti.parameter(1, 1)
        self.p_patch_vx = self.opti.parameter(1, 1)
        self.p_patch_vy = self.opti.parameter(1, 1)
        self.p_neighbors = self.opti.parameter(self.config.num_neighbors, 2)

        for k in range(N):
            x_next = self.x[k] + self.v[k] * ca.cos(self.theta[k]) * self.dt
            y_next = self.y[k] + self.v[k] * ca.sin(self.theta[k]) * self.dt
            theta_next = self.theta[k] + (self.v[k] / self.config.wheelbase) * ca.tan(self.delta[k]) * self.dt
            v_next = self.v[k] + self.accel[k] * self.dt
            self.opti.subject_to(self.x[k + 1] == x_next)
            self.opti.subject_to(self.y[k + 1] == y_next)
            self.opti.subject_to(self.theta[k + 1] == theta_next)
            self.opti.subject_to(self.v[k + 1] == v_next)

        self.opti.subject_to(self.X[:, 0] == self.p_x0)

        cos_t = ca.cos(-self.p_patch_theta)
        sin_t = ca.sin(-self.p_patch_theta)
        a_eff = ca.fmax(self.p_patch_a - self.config.containment_margin, 0.5)
        b_eff = ca.fmax(self.p_patch_b - self.config.containment_margin, 0.5)

        J_contain = 0.0
        for k in range(N + 1):
            dx = self.x[k] - self.p_patch_cx[k]
            dy = self.y[k] - self.p_patch_cy[k]
            x_rot = dx * cos_t - dy * sin_t
            y_rot = dx * sin_t + dy * cos_t
            g_contain = (x_rot / a_eff) ** 2 + (y_rot / b_eff) ** 2 - 1.0
            J_contain += 2500.0 * (1.0 + 0.8 * k / N) * ca.fmax(0.0, g_contain) ** 2

        for k in range(N + 1):
            self.opti.subject_to(self.v[k] >= self.config.v_min)
            self.opti.subject_to(self.v[k] <= self.config.v_max)
        for k in range(N):
            self.opti.subject_to(self.accel[k] >= -self.config.accel_max)
            self.opti.subject_to(self.accel[k] <= self.config.accel_max)
            self.opti.subject_to(self.delta[k] >= -self.config.steering_max)
            self.opti.subject_to(self.delta[k] <= self.config.steering_max)

        J_vel = 0.0
        for k in range(N + 1):
            vx_agent = self.v[k] * ca.cos(self.theta[k])
            vy_agent = self.v[k] * ca.sin(self.theta[k])
            J_vel += 50.0 * ((vx_agent - self.p_patch_vx) ** 2 + (vy_agent - self.p_patch_vy) ** 2)

        J_center = 0.0
        for k in range(N + 1):
            J_center += 5.0 * ((self.x[k] - self.p_patch_cx[k]) ** 2 + (self.y[k] - self.p_patch_cy[k]) ** 2)

        J_collision = 0.0
        for k in range(N + 1):
            for j in range(self.config.num_neighbors):
                dx_n = self.x[k] - self.p_neighbors[j, 0]
                dy_n = self.y[k] - self.p_neighbors[j, 1]
                dist_sq = dx_n**2 + dy_n**2 + 1e-4
                violation = ca.fmax(0.0, self.config.min_agent_dist**2 - dist_sq)
                J_collision += 200.0 * violation

        J_smooth = 0.0
        for k in range(N - 1):
            J_smooth += (self.accel[k + 1] - self.accel[k]) ** 2
            J_smooth += (self.delta[k + 1] - self.delta[k]) ** 2

        J_effort = 0.0
        for k in range(N):
            J_effort += 0.1 * self.accel[k] ** 2
            J_effort += 0.1 * self.delta[k] ** 2

        self.opti.minimize(J_vel + J_center + J_collision + J_smooth + J_effort + J_contain)
        self.opti.solver(
            "ipopt",
            {"expand": True, "print_time": False, "verbose": False},
            {
                "max_iter": 50,
                "tol": 1e-3,
                "acceptable_tol": 5e-3,
                "print_level": 0,
                "sb": "yes",
                "mu_strategy": "adaptive",
            },
        )

    def solve(self, x0: np.ndarray, patch, neighbor_positions: List[List[float]]) -> Tuple[np.ndarray | None, bool]:
        try:
            if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
                return None, False

            N = self.config.horizon_steps
            self.opti.set_value(self.p_x0, x0.reshape(4, 1))

            cx_traj = np.array([patch.vx * (k * self.dt) for k in range(N + 1)], dtype=np.float32)
            cy_traj = np.array([patch.vy * (k * self.dt) for k in range(N + 1)], dtype=np.float32)
            self.opti.set_value(self.p_patch_cx, cx_traj.reshape(N + 1, 1))
            self.opti.set_value(self.p_patch_cy, cy_traj.reshape(N + 1, 1))
            self.opti.set_value(self.p_patch_a, patch.a)
            self.opti.set_value(self.p_patch_b, patch.b)
            self.opti.set_value(self.p_patch_theta, patch.theta)
            self.opti.set_value(self.p_patch_vx, patch.vx)
            self.opti.set_value(self.p_patch_vy, patch.vy)

            neighbors_arr = np.zeros((self.config.num_neighbors, 2), dtype=np.float32)
            for j, pos in enumerate(neighbor_positions[: self.config.num_neighbors]):
                neighbors_arr[j] = pos
            for j in range(len(neighbor_positions), self.config.num_neighbors):
                neighbors_arr[j] = [1000.0, 1000.0]
            self.opti.set_value(self.p_neighbors, neighbors_arr)

            if self.prev_X_sol is not None:
                try:
                    self.opti.set_initial(self.X, self.prev_X_sol)
                    self.opti.set_initial(self.U, self.prev_U_sol)
                except Exception:
                    pass

            sol = self.opti.solve()
            if not sol.stats().get("success", False):
                self.prev_X_sol = None
                self.prev_U_sol = None
                return None, False

            self.prev_X_sol = sol.value(self.X)
            self.prev_U_sol = sol.value(self.U)
            return sol.value(self.U[:, 0]), True
        except Exception:
            self.prev_X_sol = None
            self.prev_U_sol = None
            return None, False


@dataclass
class SafetyConfig:
    robot_radius: float = 0.15
    wheelbase: float = 0.33
    min_agent_dist: float = 0.55
    max_accel: float = 6.0
    max_steering: float = 0.4
    v_min: float = 0.5
    v_max: float = 10.0
    alpha_contain: float = .0
    alpha_collision: float = 2.0


class SafetyLayer:
    """CBF-like safety filtering around MPC output."""

    def __init__(self, config: SafetyConfig | None = None) -> None:
        self.config = config or SafetyConfig()

    def compute_containment_cbf(self, x: float, y: float, theta: float, v: float, patch):
        dx = x - patch.cx
        dy = y - patch.cy
        x_rot = dx * patch.cos_t + dy * patch.sin_t
        y_rot = -dx * patch.sin_t + dy * patch.cos_t
        a_eff = max(patch.a - self.config.robot_radius, 0.5)
        b_eff = max(patch.b - self.config.robot_radius, 0.5)
        h = 1.0 - (x_rot / a_eff) ** 2 - (y_rot / b_eff) ** 2

        dh_dx_rot = -2.0 * x_rot / (a_eff**2)
        dh_dy_rot = -2.0 * y_rot / (b_eff**2)
        dh_dx = dh_dx_rot * patch.cos_t - dh_dy_rot * patch.sin_t
        dh_dy = dh_dx_rot * patch.sin_t + dh_dy_rot * patch.cos_t
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        lf_h = dh_dx * vx + dh_dy * vy
        return h, lf_h, (dh_dx, dh_dy)

    def compute_collision_cbf(self, x: float, y: float, theta: float, v: float, neighbors):
        min_h = float("inf")
        critical = None
        for nx, ny in neighbors:
            dx = x - nx
            dy = y - ny
            dist_sq = dx**2 + dy**2
            h = dist_sq - self.config.min_agent_dist**2
            if h < min_h:
                min_h = h
                critical = (dx, dy)
        if critical is None:
            return float("inf"), 0.0

        dx, dy = critical
        dh_dx = 2.0 * dx
        dh_dy = 2.0 * dy
        vx = v * np.cos(theta)
        vy = v * np.sin(theta)
        lf_h = dh_dx * vx + dh_dy * vy
        return min_h, lf_h

    def filter_control(self, u_mpc, x0: np.ndarray, patch, neighbors, dt: float = 0.05):
        x, y, theta, v = x0
        if u_mpc is None:
            return self._compute_safe_fallback(x, y, theta, v, patch), True

        accel, steering = float(u_mpc[0]), float(u_mpc[1])
        intervention = False

        h_contain, lf_h_contain, _ = self.compute_containment_cbf(x, y, theta, v, patch)
        cbf_contain = lf_h_contain + self.config.alpha_contain * h_contain
        if h_contain < 0.0:
            dx_to_center = patch.cx - x
            dy_to_center = patch.cy - y
            desired_theta = np.arctan2(dy_to_center, dx_to_center)
            heading_error = desired_theta - theta
            while heading_error > np.pi:
                heading_error -= 2 * np.pi
            while heading_error < -np.pi:
                heading_error += 2 * np.pi
            steering = np.clip(2.5 * heading_error, -self.config.max_steering, self.config.max_steering)
            if cbf_contain < 0.0:
                accel = min(accel, -2.0)
            intervention = True
        elif cbf_contain < 0.0:
            accel = min(accel, 0.0)
            intervention = True

        h_collision, lf_h_collision = self.compute_collision_cbf(x, y, theta, v, neighbors)
        cbf_collision = lf_h_collision + self.config.alpha_collision * h_collision
        if h_collision < 0.0:
            accel = -self.config.max_accel
            intervention = True
        elif cbf_collision < 0.0:
            accel = min(accel, -1.0)
            intervention = True

        v_next = v + accel * dt
        if v_next > self.config.v_max:
            accel = (self.config.v_max - v) / dt
            intervention = True
        elif v_next < self.config.v_min:
            accel = (self.config.v_min - v) / dt
            intervention = True

        accel = float(np.clip(accel, -self.config.max_accel, self.config.max_accel))
        steering = float(np.clip(steering, -self.config.max_steering, self.config.max_steering))
        return np.array([accel, steering], dtype=np.float32), intervention

    def _compute_safe_fallback(self, x: float, y: float, theta: float, v: float, patch):
        dx = patch.cx - x
        dy = patch.cy - y
        desired_theta = np.arctan2(dy, dx)
        heading_error = desired_theta - theta
        while heading_error > np.pi:
            heading_error -= 2 * np.pi
        while heading_error < -np.pi:
            heading_error += 2 * np.pi

        steering = np.clip(2.5 * heading_error, -self.config.max_steering, self.config.max_steering)
        h_contain, _, _ = self.compute_containment_cbf(x, y, theta, v, patch)
        if h_contain < -0.5:
            target_v = 5.0
        elif h_contain < 0.0:
            target_v = 4.0
        else:
            patch_speed = np.sqrt(patch.vx**2 + patch.vy**2)
            target_v = max(patch_speed, 3.0)
        accel = np.clip((target_v - v) / 0.1, -self.config.max_accel, self.config.max_accel)
        return np.array([accel, steering], dtype=np.float32)

