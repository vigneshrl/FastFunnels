"""
Distributed MPC (DMPC) for the funnel followers.

WHY THIS EXISTS
---------------
`envs/mpc.py`'s SEMPCSolver is a *myopic, decoupled* scheme: every follower
solves its own OCP and treats the other cars as obstacles moving at CONSTANT
VELOCITY over the whole horizon (see the `p_neighbors + p_neighbor_vel * k*dt`
term).  Nobody tells anybody what they actually intend to do, and nobody
re-solves after learning it.  That is a one-shot game played with a bad model
of the opponent, so at N>=5 two followers routinely pick mirror-image dodges,
both assume the other will hold course, and they drive into each other.

DMPC fixes the information problem rather than the geometry.  Two changes:

1. TRAJECTORY EXCHANGE.  Neighbours are no longer extrapolated -- each agent is
   handed the other agents' actual PREDICTED TRAJECTORIES over the horizon, so
   it plans against what they intend to do, not against a straight line.

2. RECIPROCAL SEPARATING HYPERPLANES (buffered Voronoi cells).  For each pair
   (i, j) and each horizon step k we take the two predicted positions, drop a
   plane between them, and constrain each agent to stay on its own side with
   half the required clearance.  Because BOTH agents are given the same plane
   with opposite normals, each one yielding half the gap is sufficient -- the
   avoidance effort is split instead of duplicated or skipped.  This is the
   "golden rule" of DMPC coordination: every agent concedes exactly the share
   it expects the other to concede.  The constraint is linear in the decision
   variables, so it costs almost nothing to add.

   Against the PATCH CAR the split is switched off (`coop=False`): the patch is
   an RL policy that has never heard of the followers and will not yield, so
   the follower takes the entire clearance burden.

The solve is iterated (Gauss-Seidel by default) so agents converge on a
consistent joint plan within a control tick instead of reacting a tick late.

Everything else -- single-track dynamics, funnel containment, slot tracking,
velocity matching, IPOPT settings -- is deliberately identical to SEMPCSolver
so that a DMPC-vs-NMPC comparison isolates the coordination change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple

import casadi as ca
import numpy as np


@dataclass
class DMPCConfig:
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
    w_vel: float = 50.0
    w_center: float = 5.0
    w_contain: float = 800.0
    max_iter: int = 120
    w_collision: float = 200.0
    collision_radius: float = 0.0
    # --- DMPC-specific ---------------------------------------------------
    w_hp_slack: float = 6000.0      # penalty on separating-hyperplane slack
    hp_from_step: int = 1           # k=0 is pinned by x0; a plane there can only be infeasible
    model: str = "kinematic"
    st_substeps: int = 3
    st_mu: float = 1.0489
    st_C_Sf: float = 4.718
    st_C_Sr: float = 5.4562
    st_lf: float = 0.15875
    st_lr: float = 0.17145
    st_h: float = 0.074
    st_m: float = 3.74
    st_Iz: float = 0.04712
    st_sv_max: float = 3.2
    st_v_eps: float = 0.5


def separating_hyperplanes(ego_traj: np.ndarray,
                           nbr_trajs: List[np.ndarray],
                           radius: float,
                           coop: List[bool]) -> Tuple[np.ndarray, np.ndarray]:
    """Build the per-neighbour, per-step half-space that keeps ego clear.

    ego_traj : (K, 2) ego's currently-believed plan
    nbr_trajs: list of (K, 2), one per neighbour, their currently-believed plans
    returns  : normals (J, K, 2) and offsets (J, K), defining  n . p >= off

    For a COOPERATIVE neighbour the plane sits at the midpoint pushed out by
    radius/2, so ego concedes half the gap and the neighbour -- solving the same
    problem with the normal negated -- concedes the other half.  For a
    NON-cooperative neighbour the plane sits a full `radius` off the neighbour's
    own predicted position, so ego concedes everything.
    """
    K = ego_traj.shape[0]
    J = len(nbr_trajs)
    normals = np.zeros((J, K, 2), np.float64)
    offsets = np.zeros((J, K), np.float64)
    for j, q in enumerate(nbr_trajs):
        d = ego_traj[:K] - q[:K]                       # (K, 2) ego relative to neighbour
        nrm = np.linalg.norm(d, axis=1)
        # Degenerate case: the two plans coincide, so there is no meaningful
        # direction to separate along.  Fall back to the ego's own direction of
        # travel rotated 90 deg, which at least splits them sideways.
        bad = nrm < 1e-6
        if np.any(bad):
            fwd = np.zeros_like(d)
            fwd[1:] = ego_traj[1:K] - ego_traj[:K - 1]
            fn = np.linalg.norm(fwd, axis=1)
            side = np.stack([-fwd[:, 1], fwd[:, 0]], axis=1)
            with np.errstate(invalid="ignore", divide="ignore"):
                side = np.where(fn[:, None] > 1e-9, side / np.maximum(fn[:, None], 1e-9),
                                np.array([1.0, 0.0]))
            d = np.where(bad[:, None], side, d)
            nrm = np.maximum(np.linalg.norm(d, axis=1), 1e-9)
        n_hat = d / nrm[:, None]
        if coop[j]:
            point = 0.5 * (ego_traj[:K] + q[:K]) + 0.5 * radius * n_hat
        else:
            point = q[:K] + radius * n_hat
        normals[j] = n_hat
        offsets[j] = np.einsum("ij,ij->i", n_hat, point)
    return normals, offsets


class DMPCSolver:
    """Single-agent OCP with trajectory-exchange + reciprocal-hyperplane coupling."""

    def __init__(self, config: DMPCConfig | None = None) -> None:
        self.config = config or DMPCConfig()
        self.dt = self.config.horizon_seconds / self.config.horizon_steps
        self.prev_X_sol = None
        self.prev_U_sol = None
        self._build_problem()

    # ------------------------------------------------------------------ build
    def _build_problem(self) -> None:
        c = self.config
        N = c.horizon_steps
        J = max(1, c.num_neighbors)
        self.st = str(c.model).lower() == "st"
        self.nx = 7 if self.st else 4
        self.opti = ca.Opti()
        self.X = self.opti.variable(self.nx, N + 1)
        self.U = self.opti.variable(2, N)

        self.x = self.X[0, :]
        self.y = self.X[1, :]
        self.accel = self.U[0, :]
        if self.st:
            self.delta = self.X[2, :]
            self.v = self.X[3, :]
            self.psi = self.X[4, :]
            self.beta = self.X[6, :]
            self.steer_vel = self.U[1, :]
            self.theta = self.psi + self.beta
        else:
            self.theta = self.X[2, :]
            self.v = self.X[3, :]
            self.delta = self.U[1, :]

        self.p_x0 = self.opti.parameter(self.nx, 1)
        self.p_patch_cx = self.opti.parameter(N + 1, 1)
        self.p_patch_cy = self.opti.parameter(N + 1, 1)
        self.p_patch_a = self.opti.parameter(1, 1)
        self.p_patch_b = self.opti.parameter(1, 1)
        self.p_patch_theta = self.opti.parameter(1, 1)
        self.p_patch_vx = self.opti.parameter(1, 1)
        self.p_patch_vy = self.opti.parameter(1, 1)
        # DMPC: full predicted neighbour paths, and the reciprocal half-spaces
        self.p_nbr_x = self.opti.parameter(J, N + 1)
        self.p_nbr_y = self.opti.parameter(J, N + 1)
        self.p_hp_nx = self.opti.parameter(J, N + 1)
        self.p_hp_ny = self.opti.parameter(J, N + 1)
        self.p_hp_off = self.opti.parameter(J, N + 1)

        # ---- dynamics
        if self.st:
            for k in range(N):
                xk = self.X[:, k]
                uk = self.U[:, k]
                h = self.dt / c.st_substeps
                for _ in range(c.st_substeps):
                    k1 = self._st_rhs(xk, uk)
                    k2 = self._st_rhs(xk + 0.5 * h * k1, uk)
                    k3 = self._st_rhs(xk + 0.5 * h * k2, uk)
                    k4 = self._st_rhs(xk + h * k3, uk)
                    xk = xk + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                self.opti.subject_to(self.X[:, k + 1] == xk)
        else:
            for k in range(N):
                self.opti.subject_to(self.x[k + 1] == self.x[k] + self.v[k] * ca.cos(self.theta[k]) * self.dt)
                self.opti.subject_to(self.y[k + 1] == self.y[k] + self.v[k] * ca.sin(self.theta[k]) * self.dt)
                self.opti.subject_to(self.theta[k + 1] == self.theta[k]
                                     + (self.v[k] / c.wheelbase) * ca.tan(self.delta[k]) * self.dt)
                self.opti.subject_to(self.v[k + 1] == self.v[k] + self.accel[k] * self.dt)

        self.opti.subject_to(self.X[:, 0] == self.p_x0)

        # ---- funnel containment (identical to SEMPCSolver)
        cos_t = ca.cos(-self.p_patch_theta)
        sin_t = ca.sin(-self.p_patch_theta)
        a_eff = ca.fmax(self.p_patch_a - c.containment_margin, 0.5)
        b_eff = ca.fmax(self.p_patch_b - c.containment_margin, 0.5)
        J_contain = 0.0
        for k in range(N + 1):
            dx = self.x[k] - self.p_patch_cx[k]
            dy = self.y[k] - self.p_patch_cy[k]
            x_rot = dx * cos_t - dy * sin_t
            y_rot = dx * sin_t + dy * cos_t
            g = (x_rot / a_eff) ** 2 + (y_rot / b_eff) ** 2 - 1.0
            J_contain += c.w_contain * (1.0 + 0.8 * k / N) * ca.fmax(0.0, g) ** 2

        # ---- bounds
        self.opti.subject_to(self.v[0] <= c.v_max)
        for k in range(1, N + 1):
            self.opti.subject_to(self.v[k] >= c.v_min)
            self.opti.subject_to(self.v[k] <= c.v_max)
        for k in range(N):
            self.opti.subject_to(self.accel[k] >= -c.accel_max)
            self.opti.subject_to(self.accel[k] <= c.accel_max)
        if self.st:
            for k in range(N):
                self.opti.subject_to(self.steer_vel[k] >= -c.st_sv_max)
                self.opti.subject_to(self.steer_vel[k] <= c.st_sv_max)
            for k in range(1, N + 1):
                self.opti.subject_to(self.delta[k] >= -c.steering_max)
                self.opti.subject_to(self.delta[k] <= c.steering_max)
        else:
            for k in range(N):
                self.opti.subject_to(self.delta[k] >= -c.steering_max)
                self.opti.subject_to(self.delta[k] <= c.steering_max)

        # ---- tracking costs (identical to SEMPCSolver)
        J_vel = 0.0
        for k in range(N + 1):
            vx_a = self.v[k] * ca.cos(self.theta[k])
            vy_a = self.v[k] * ca.sin(self.theta[k])
            J_vel += c.w_vel * ((vx_a - self.p_patch_vx) ** 2 + (vy_a - self.p_patch_vy) ** 2)

        J_center = 0.0
        for k in range(N + 1):
            J_center += c.w_center * ((self.x[k] - self.p_patch_cx[k]) ** 2
                                      + (self.y[k] - self.p_patch_cy[k]) ** 2)

        # ---- DMPC coupling ---------------------------------------------------
        # (a) soft square-distance penalty, but now against the neighbour's REAL
        #     predicted path rather than a constant-velocity ray;
        # (b) the reciprocal separating hyperplane, slacked so a transient
        #     violation degrades the solution instead of killing the solve.
        r_soft = c.min_agent_dist
        self.S_hp = self.opti.variable(J, N + 1)
        J_coupling = 0.0
        for k in range(N + 1):
            for j in range(J):
                dxn = self.x[k] - self.p_nbr_x[j, k]
                dyn = self.y[k] - self.p_nbr_y[j, k]
                J_coupling += c.w_collision * ca.fmax(0.0, r_soft ** 2 - (dxn ** 2 + dyn ** 2 + 1e-4))
                s = self.S_hp[j, k]
                self.opti.subject_to(s >= 0.0)
                if k >= c.hp_from_step:
                    lhs = self.p_hp_nx[j, k] * self.x[k] + self.p_hp_ny[j, k] * self.y[k]
                    self.opti.subject_to(lhs >= self.p_hp_off[j, k] - s)
                    J_coupling += c.w_hp_slack * (s + 10.0 * s ** 2)
                else:
                    self.opti.subject_to(s == 0.0)

        J_smooth = 0.0
        for k in range(N - 1):
            J_smooth += (self.accel[k + 1] - self.accel[k]) ** 2
            J_smooth += (self.delta[k + 1] - self.delta[k]) ** 2

        J_effort = 0.0
        for k in range(N):
            J_effort += 0.1 * self.accel[k] ** 2
            J_effort += 0.1 * self.delta[k] ** 2

        self.opti.minimize(J_vel + J_center + J_coupling + J_smooth + J_effort + J_contain)

        ipopt_opts = {
            "max_iter": int(c.max_iter),
            "tol": 1e-3,
            "acceptable_tol": 2e-2,
            "acceptable_iter": 12,
            "acceptable_constr_viol_tol": 1e-2,
            "print_level": 0,
            "sb": "yes",
            "mu_strategy": "adaptive",
        }
        if self.st:
            ipopt_opts.update({
                "max_iter": max(int(c.max_iter), 200),
                "tol": 2e-3,
                "acceptable_tol": 1e-2,
                "acceptable_iter": 8,
                "acceptable_constr_viol_tol": 5e-3,
                "mu_strategy": "monotone",
                "nlp_scaling_method": "gradient-based",
            })
        self.opti.solver("ipopt", {"expand": True, "print_time": False, "verbose": False},
                         ipopt_opts)

    # ------------------------------------------------------------------ model
    def _st_rhs(self, x, u):
        c = self.config
        g = 9.81
        DELTA, V, PSI_DOT, BETA = x[2], x[3], x[5], x[6]
        ACCL, STEER_VEL = u[0], u[1]
        Vs = ca.fmax(V, c.st_v_eps)
        lf, lr, hcg, m, Iz = c.st_lf, c.st_lr, c.st_h, c.st_m, c.st_Iz
        Csf, Csr, mu = c.st_C_Sf, c.st_C_Sr, c.st_mu
        gf = g * lr - ACCL * hcg
        gr = g * lf + ACCL * hcg
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
            V * ca.cos(x[4] + BETA),
            V * ca.sin(x[4] + BETA),
            STEER_VEL,
            ACCL,
            PSI_DOT,
            psi_ddot,
            beta_dot,
        )

    # ------------------------------------------------------------------ solve
    def solve(self, x0: np.ndarray, patch,
              nbr_traj: np.ndarray,
              hp_normals: np.ndarray,
              hp_offsets: np.ndarray,
              center_traj: np.ndarray | None = None):
        """One agent's OCP given the neighbours' exchanged plans.

        nbr_traj   : (J, N+1, 2) neighbour predicted paths, patch-at-origin frame
        hp_normals : (J, N+1, 2) unit normals of the reciprocal half-spaces
        hp_offsets : (J, N+1)    offsets, constraint is  n . p >= off

        Returns (u0, ok, X_pred) where X_pred is (N+1, 2) -- the plan this agent
        now intends, to be broadcast to the others for the next sweep.
        """
        c = self.config
        N = c.horizon_steps
        J = max(1, c.num_neighbors)
        try:
            if np.any(np.isnan(x0)) or np.any(np.isinf(x0)):
                return None, False, None
            x0 = np.asarray(x0, np.float64).reshape(-1)
            if x0.shape[0] != self.nx:
                if self.st and x0.shape[0] == 4:
                    x0 = np.array([x0[0], x0[1], 0.0, x0[3], x0[2], 0.0, 0.0])
                else:
                    return None, False, None
            self.opti.set_value(self.p_x0, x0.reshape(self.nx, 1))

            if center_traj is not None:
                ct = np.asarray(center_traj, np.float64).reshape(N + 1, 2)
            else:
                ct = np.stack([patch.vx * np.arange(N + 1) * self.dt,
                               patch.vy * np.arange(N + 1) * self.dt], axis=1)
            self.opti.set_value(self.p_patch_cx, ct[:, 0].reshape(N + 1, 1))
            self.opti.set_value(self.p_patch_cy, ct[:, 1].reshape(N + 1, 1))
            self.opti.set_value(self.p_patch_a, patch.a)
            self.opti.set_value(self.p_patch_b, patch.b)
            self.opti.set_value(self.p_patch_theta, patch.theta)
            self.opti.set_value(self.p_patch_vx, patch.vx)
            self.opti.set_value(self.p_patch_vy, patch.vy)

            nt = np.asarray(nbr_traj, np.float64).reshape(J, N + 1, 2)
            self.opti.set_value(self.p_nbr_x, nt[:, :, 0])
            self.opti.set_value(self.p_nbr_y, nt[:, :, 1])
            hn = np.asarray(hp_normals, np.float64).reshape(J, N + 1, 2)
            self.opti.set_value(self.p_hp_nx, hn[:, :, 0])
            self.opti.set_value(self.p_hp_ny, hn[:, :, 1])
            self.opti.set_value(self.p_hp_off, np.asarray(hp_offsets, np.float64).reshape(J, N + 1))

            warm = (self.prev_X_sol is not None
                    and getattr(self.prev_X_sol, "shape", (0,))[0] == self.nx)
            if warm:
                try:
                    self.opti.set_initial(self.X, self.prev_X_sol)
                    self.opti.set_initial(self.U, self.prev_U_sol)
                except Exception:
                    warm = False
            if not warm and self.st:
                Xg = np.zeros((self.nx, N + 1))
                v0 = float(np.clip(x0[3], c.v_min, c.v_max))
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
            stt = sol.stats()
            ok = stt.get("success", False) or stt.get("return_status", "") in (
                "Solve_Succeeded", "Solved_To_Acceptable_Level")
            if not ok:
                return None, False, None
            self.prev_X_sol = sol.value(self.X)
            self.prev_U_sol = sol.value(self.U)
            xy = np.stack([self.prev_X_sol[0, :], self.prev_X_sol[1, :]], axis=1)
            return sol.value(self.U[:, 0]), True, xy
        except Exception:
            # Same rescue as SEMPCSolver: a near-feasible last iterate is a
            # perfectly usable plan; dropping the step is strictly worse.
            try:
                xs = self.opti.debug.value(self.X)
                us = self.opti.debug.value(self.U)
                gmax = float(np.max(np.abs(self.opti.debug.value(self.opti.g))))
                if np.all(np.isfinite(xs)) and np.all(np.isfinite(us)) and gmax < 1.5e-1:
                    self.prev_X_sol, self.prev_U_sol = xs, us
                    xy = np.stack([xs[0, :], xs[1, :]], axis=1)
                    return us[:, 0], True, xy
            except Exception:
                pass
            return None, False, None
