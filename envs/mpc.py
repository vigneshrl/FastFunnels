"""
MPC and safety layer module.
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
    # cost weights (defaults preserve historical behaviour)
    w_vel: float = 50.0
    w_center: float = 5.0
    w_contain: float = 800.0
    max_iter: int = 120
    # inter-agent collision term
    w_collision: float = 200.0          # weight on the soft square-distance penalty
    collision_hard: bool = False        # add a slacked HARD min-distance constraint
    collision_hard_patch_only: bool = False  # hard constraint only vs neighbour 0 (patch car)
    collision_radius: float = 0.0       # hard keep-out radius (0 -> use min_agent_dist)
    w_collision_slack: float = 4000.0   # penalty on the hard-constraint slack
    # prediction model: "kinematic" (default, 4-state bicycle) or
    # "st" (7-state single-track dynamic model, matches f1tenth_gym)
    model: str = "kinematic"
    st_substeps: int = 3            # RK4 sub-steps per horizon step (st only)
    # single-track params (f1tenth_gym defaults)
    st_mu: float = 1.0489
    st_C_Sf: float = 4.718
    st_C_Sr: float = 5.4562
    st_lf: float = 0.15875
    st_lr: float = 0.17145
    st_h: float = 0.074
    st_m: float = 3.74
    st_Iz: float = 0.04712
    st_sv_max: float = 3.2          # steering-rate limit (rad/s)
    st_v_eps: float = 0.5          # floor on v in the 1/v tyre terms


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
        self.st = str(self.config.model).lower() == "st"
        self.nx = 7 if self.st else 4
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, N + 1)
        self.U = self.opti.variable(2, N)

        self.x = self.X[0, :]
        self.y = self.X[1, :]
        self.accel = self.U[0, :]
        if self.st:
            # state = [x, y, delta, v, psi, psi_dot, beta];  u = [accel, steer_vel]
            self.delta = self.X[2, :]           # steering ANGLE is a state now
            self.v = self.X[3, :]
            self.psi = self.X[4, :]
            self.beta = self.X[6, :]
            self.steer_vel = self.U[1, :]
            self.theta = self.psi + self.beta   # course angle (for the velocity cost)
        else:
            self.theta = self.X[2, :]
            self.v = self.X[3, :]
            self.delta = self.U[1, :]           # kinematic: steering angle is the input

        self.p_x0 = self.opti.parameter(self.nx, 1)
        self.p_patch_cx = self.opti.parameter(N + 1, 1)
        self.p_patch_cy = self.opti.parameter(N + 1, 1)
        self.p_patch_a = self.opti.parameter(1, 1)
        self.p_patch_b = self.opti.parameter(1, 1)
        self.p_patch_theta = self.opti.parameter(1, 1)
        self.p_patch_vx = self.opti.parameter(1, 1)
        self.p_patch_vy = self.opti.parameter(1, 1)
        self.p_neighbors = self.opti.parameter(self.config.num_neighbors, 2)
        self.p_neighbor_vel = self.opti.parameter(self.config.num_neighbors, 2)

        if self.st:
            for k in range(N):
                xk = self.X[:, k]
                uk = self.U[:, k]
                h = self.dt / self.config.st_substeps
                for _ in range(self.config.st_substeps):
                    k1 = self._st_rhs(xk, uk)
                    k2 = self._st_rhs(xk + 0.5 * h * k1, uk)
                    k3 = self._st_rhs(xk + 0.5 * h * k2, uk)
                    k4 = self._st_rhs(xk + h * k3, uk)
                    xk = xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                self.opti.subject_to(self.X[:, k + 1] == xk)
        else:
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
            J_contain += self.config.w_contain * (1.0 + 0.8 * k / N) * ca.fmax(0.0, g_contain) ** 2 #increased the containment penalty weight from 800 --> 2500
            # to make containment more stricter without tire model 

        # v_min only from k>=1: X[:,0]==p_x0 is a hard equality, so if the
        # measured speed is below v_min (car just spawned / nearly stopped) a
        # k=0 bound makes the whole NLP infeasible. k>=1 still forces the plan
        # up to v_min.
        self.opti.subject_to(self.v[0] <= self.config.v_max)
        for k in range(1, N + 1):
            self.opti.subject_to(self.v[k] >= self.config.v_min)
            self.opti.subject_to(self.v[k] <= self.config.v_max)
        for k in range(N):
            self.opti.subject_to(self.accel[k] >= -self.config.accel_max)
            self.opti.subject_to(self.accel[k] <= self.config.accel_max)
        if self.st:
            for k in range(N):
                self.opti.subject_to(self.steer_vel[k] >= -self.config.st_sv_max)
                self.opti.subject_to(self.steer_vel[k] <= self.config.st_sv_max)
            for k in range(1, N + 1):        # k=0 pinned by p_x0
                self.opti.subject_to(self.delta[k] >= -self.config.steering_max)
                self.opti.subject_to(self.delta[k] <= self.config.steering_max)
        else:
            for k in range(N):
                self.opti.subject_to(self.delta[k] >= -self.config.steering_max)
                self.opti.subject_to(self.delta[k] <= self.config.steering_max)

        J_vel = 0.0
        for k in range(N + 1):
            vx_agent = self.v[k] * ca.cos(self.theta[k])
            vy_agent = self.v[k] * ca.sin(self.theta[k])
            J_vel += self.config.w_vel * ((vx_agent - self.p_patch_vx) ** 2 + (vy_agent - self.p_patch_vy) ** 2)

        J_center = 0.0
        for k in range(N + 1):
            J_center += self.config.w_center * ((self.x[k] - self.p_patch_cx[k]) ** 2 + (self.y[k] - self.p_patch_cy[k]) ** 2)

        # --- inter-agent collision ---------------------------------------------
        # neighbours are propagated as constant-velocity over the horizon (they
        # ride roughly with the funnel) so the keep-out is not needlessly
        # conservative; a soft square-distance penalty plus, optionally, a
        # slacked HARD min-distance constraint that cannot be traded away.
        r_soft = self.config.min_agent_dist
        r_hard = self.config.collision_radius or self.config.min_agent_dist
        self.S_coll = None
        if self.config.collision_hard and self.config.num_neighbors > 0:
            self.S_coll = self.opti.variable(self.config.num_neighbors, N + 1)
        J_collision = 0.0
        for k in range(N + 1):
            for j in range(self.config.num_neighbors):
                nx_k = self.p_neighbors[j, 0] + self.p_neighbor_vel[j, 0] * (k * self.dt)
                ny_k = self.p_neighbors[j, 1] + self.p_neighbor_vel[j, 1] * (k * self.dt)
                dx_n = self.x[k] - nx_k
                dy_n = self.y[k] - ny_k
                dist_sq = dx_n**2 + dy_n**2 + 1e-4
                J_collision += self.config.w_collision * ca.fmax(0.0, r_soft**2 - dist_sq)
                hard_j = self.S_coll is not None and (
                    j == 0 or not self.config.collision_hard_patch_only)
                if hard_j:
                    s = self.S_coll[j, k]
                    self.opti.subject_to(dist_sq >= r_hard**2 - s)
                    self.opti.subject_to(s >= 0.0)
                    J_collision += self.config.w_collision_slack * (s + 10.0 * s**2)

        J_smooth = 0.0
        for k in range(N - 1):
            J_smooth += (self.accel[k + 1] - self.accel[k]) ** 2
            J_smooth += (self.delta[k + 1] - self.delta[k]) ** 2

        J_effort = 0.0
        for k in range(N):
            J_effort += 0.1 * self.accel[k] ** 2
            J_effort += 0.1 * self.delta[k] ** 2

        self.opti.minimize(J_vel + J_center + J_collision + J_smooth + J_effort + J_contain)
        ipopt_opts = {
            "max_iter": int(self.config.max_iter),
            "tol": 1e-3,
            # declare "acceptable" (return a usable feasible point) well before
            # grinding to max_iter -- a swervy funnel reference makes the cost
            # landscape nasty and full optimality is not worth a dropped step
            "acceptable_tol": 2e-2,
            "acceptable_iter": 12,
            "acceptable_constr_viol_tol": 1e-2,
            "print_level": 0,
            "sb": "yes",
            "mu_strategy": "adaptive",
        }
        if self.st:
            # the RK4-of-single-track equality constraints are stiff -> more
            # iterations + monotone barrier, but keep the feasibility tol tight
            # so an "acceptable" exit is still a dynamically-consistent plan
            ipopt_opts.update({
                "max_iter": max(int(self.config.max_iter), 200),
                "tol": 2e-3,
                "acceptable_tol": 1e-2,
                "acceptable_iter": 8,
                "acceptable_constr_viol_tol": 5e-3,
                "mu_strategy": "monotone",
                "nlp_scaling_method": "gradient-based",
            })
        self.opti.solver("ipopt", {"expand": True, "print_time": False, "verbose": False},
                         ipopt_opts)

    def _st_rhs(self, x, u):
        """Continuous-time single-track dynamics (f1tenth_gym / CommonRoad §7),
        high-speed branch, as a CasADi expression.  x = [X,Y,delta,v,psi,psi_dot,
        beta];  u = [accel, steer_vel]."""
        c = self.config
        g = 9.81
        DELTA, V, PSI_DOT, BETA = x[2], x[3], x[5], x[6]
        ACCL, STEER_VEL = u[0], u[1]
        Vs = ca.fmax(V, c.st_v_eps)               # keep the 1/v tyre terms finite
        lf, lr, hcg, m, Iz = c.st_lf, c.st_lr, c.st_h, c.st_m, c.st_Iz
        Csf, Csr, mu = c.st_C_Sf, c.st_C_Sr, c.st_mu
        gf = g * lr - ACCL * hcg                  # front load term
        gr = g * lf + ACCL * hcg                  # rear load term
        psi_ddot = (mu * m / (Iz * (lf + lr))) * (
            lf * Csf * gf * DELTA
            + (lr * Csr * gr - lf * Csf * gf) * BETA
            - (lf * lf * Csf * gf + lr * lr * Csr * gr) * (PSI_DOT / Vs)
        )
        beta_dot = (mu / (Vs * (lr + lf))) * (
            Csf * gf * DELTA
            - (Csr * gr + Csf * gf) * BETA
            + (Csr * gr * lr - Csf * gf * lf) * (PSI_DOT / Vs)
        ) - PSI_DOT
        return ca.vertcat(
            V * ca.cos(x[4] + BETA),   # X_dot
            V * ca.sin(x[4] + BETA),   # Y_dot
            STEER_VEL,                 # delta_dot
            ACCL,                      # v_dot
            PSI_DOT,                   # psi_dot
            psi_ddot,                  # psi_ddot
            beta_dot,                  # beta_dot
        )

    def solve(self, x0: np.ndarray, patch, neighbor_positions: List[List[float]],
              center_traj: np.ndarray | None = None,
              neighbor_vels: List[List[float]] | None = None) -> Tuple[np.ndarray | None, bool]:
        try:
            if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
                return None, False

            N = self.config.horizon_steps
            x0 = np.asarray(x0, np.float64).reshape(-1)
            if x0.shape[0] != self.nx:            # kinematic x0 handed to an st solver (or vice versa)
                if self.st and x0.shape[0] == 4:  # [x,y,theta,v] -> [x,y,delta,v,psi,psi_dot,beta]
                    x0 = np.array([x0[0], x0[1], 0.0, x0[3], x0[2], 0.0, 0.0])
                else:
                    return None, False
            self.opti.set_value(self.p_x0, x0.reshape(self.nx, 1))

            if center_traj is not None:
                # caller-supplied predicted funnel-centre path, same frame as x0
                ct = np.asarray(center_traj, np.float32).reshape(N + 1, 2)
                cx_traj = ct[:, 0]
                cy_traj = ct[:, 1]
            else:
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

            nvel_arr = np.zeros((self.config.num_neighbors, 2), dtype=np.float32)
            if neighbor_vels is not None:
                for j, vv in enumerate(neighbor_vels[: self.config.num_neighbors]):
                    nvel_arr[j] = vv
            self.opti.set_value(self.p_neighbor_vel, nvel_arr)

            warm = (self.prev_X_sol is not None
                    and getattr(self.prev_X_sol, "shape", (0,))[0] == self.nx)
            if warm:
                try:
                    self.opti.set_initial(self.X, self.prev_X_sol)
                    self.opti.set_initial(self.U, self.prev_U_sol)
                except Exception:
                    warm = False
            if not warm and self.st:
                # cold start: the RK4-of-ST equality constraints are stiff and
                # IPOPT thrashes from the all-zero default.  Seed a constant-speed
                # straight-line rollout so the guess is dynamically consistent.
                Xg = np.zeros((self.nx, N + 1))
                v0 = float(np.clip(x0[3], self.config.v_min, self.config.v_max))
                psi0 = float(x0[4])
                for k in range(N + 1):
                    Xg[0, k] = x0[0] + v0 * np.cos(psi0) * k * self.dt
                    Xg[1, k] = x0[1] + v0 * np.sin(psi0) * k * self.dt
                    Xg[2, k] = x0[2]
                    Xg[3, k] = v0
                    Xg[4, k] = psi0
                try:
                    self.opti.set_initial(self.X, Xg)
                    self.opti.set_initial(self.U, np.zeros((2, N)))
                except Exception:
                    pass

            sol = self.opti.solve()
            st = sol.stats()
            ok = st.get("success", False) or st.get("return_status", "") in (
                "Solve_Succeeded", "Solved_To_Acceptable_Level")
            if not ok:
                return None, False

            self.prev_X_sol = sol.value(self.X)
            self.prev_U_sol = sol.value(self.U)
            return sol.value(self.U[:, 0]), True
        except Exception:
            # opti.solve() raises on any non-optimal IPOPT exit. If the last
            # iterate is still near-feasible (small constraint violation) it is
            # a perfectly usable control -- accept it and keep the warm-start
            # rather than dropping the whole step.
            try:
                xs = self.opti.debug.value(self.X)
                us = self.opti.debug.value(self.U)
                g = float(np.max(np.abs(self.opti.debug.value(self.opti.g))))
                if np.all(np.isfinite(xs)) and np.all(np.isfinite(us)) and g < 1.5e-1:
                    self.prev_X_sol, self.prev_U_sol = xs, us
                    return us[:, 0], True
            except Exception:
                pass
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
    alpha_contain: float = 1.0 #bumped up from 1.0 to increase the safety aggressiveness
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

