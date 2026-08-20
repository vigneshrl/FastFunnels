"""f1tenth_gym's single-track car, usable outside f1tenth_gym.

The cross-over study needs the *same* vehicle model on both sides of the
simulator swap: ORCA moved into PyRoboSim has to keep the single-track
dynamics it uses by default, and GCBF+ moved into f1tenth_gym has to give up
its own Dubins integrator for that model.  Rather than reimplement it, this
wraps ``f1tenth_gym``'s own dynamics, PID layer and integrator, so a car
stepped here is stepped by exactly the code ``RaceCar.update_pose`` runs --
minus the scan simulator and the gym's collision handling, which PyRoboSim
supplies instead.

State is f1tenth's 7-vector: ``[x, y, delta, v, theta, theta_dot, beta]``.
"""

from __future__ import annotations

import numpy as np

from f1tenth_gym.envs.action import SpeedAction, SteeringAngleAction
from f1tenth_gym.envs.dynamic_models import DynamicModel
from f1tenth_gym.envs.integrator import RK4Integrator

__all__ = ["F110_PARAMS", "SingleTrackFleet"]

# f110_env's documented defaults, i.e. what the funnel and ORCA already run on.
F110_PARAMS = {
    "mu": 1.0489, "C_Sf": 4.718, "C_Sr": 5.4562,
    "lf": 0.15875, "lr": 0.17145, "h": 0.074, "m": 3.74, "I": 0.04712,
    "s_min": -0.4189, "s_max": 0.4189, "sv_min": -3.2, "sv_max": 3.2,
    "v_switch": 7.319, "a_max": 9.51, "v_min": -5.0, "v_max": 20.0,
    "width": 0.31, "length": 0.58,
}


class SingleTrackFleet:
    """N single-track cars advanced by desired (steering angle, speed).

    Mirrors ``RaceCar.update_pose``: the same PID layer maps the desired
    steering angle and speed onto steering velocity and acceleration, the same
    RK4 integrator advances the same ``f_dynamics``, and the same one-step
    steering delay buffer is applied.
    """

    def __init__(self, n_agents: int, dt: float = 0.01,
                 params: dict | None = None, steer_buffer_size: int = 2):
        self.n = n_agents
        self.dt = float(dt)
        self.params = dict(params or F110_PARAMS)
        self.model = DynamicModel.ST
        self.integrator = RK4Integrator()
        self.long_action = SpeedAction(self.params)
        self.steer_action = SteeringAngleAction(self.params)
        self.steer_buffer_size = steer_buffer_size
        self.reset(np.zeros((n_agents, 3)))

    # -- state -------------------------------------------------------------

    def reset(self, poses: np.ndarray) -> None:
        """``poses`` is ``(n, 3)`` of x, y, yaw."""
        self.state = np.stack(
            [self.model.get_initial_state(pose=np.asarray(p, dtype=float))
             for p in poses]
        )
        self.steer_buffer = [np.empty(0) for _ in range(self.n)]

    @property
    def xy(self) -> np.ndarray:
        return self.state[:, :2]

    @property
    def yaw(self) -> np.ndarray:
        return self.state[:, 4]

    @property
    def speed(self) -> np.ndarray:
        return self.state[:, 3]

    def poses(self) -> np.ndarray:
        """``(n, 3)`` of x, y, yaw."""
        return np.stack([self.state[:, 0], self.state[:, 1], self.state[:, 4]],
                        axis=1)

    def obs(self) -> dict:
        """The subset of the gym observation the metric collectors read."""
        return {
            "poses_x": self.state[:, 0].copy(),
            "poses_y": self.state[:, 1].copy(),
            "poses_theta": self.state[:, 4].copy(),
            "linear_vels_x": self.state[:, 3].copy(),
        }

    # -- stepping ----------------------------------------------------------

    def step(self, steer: np.ndarray, speed: np.ndarray) -> None:
        """Advance every car one ``dt`` under desired steering angle and speed."""
        steer = np.atleast_1d(np.asarray(steer, dtype=float))
        speed = np.atleast_1d(np.asarray(speed, dtype=float))

        for i in range(self.n):
            # The gym delays steering commands by a small buffer; keep it, or
            # the car turns in noticeably faster than it does in f1tenth_gym.
            buf = self.steer_buffer[i]
            if buf.shape[0] < self.steer_buffer_size:
                applied = 0.0
                self.steer_buffer[i] = np.append(steer[i], buf)
            else:
                applied = buf[-1]
                self.steer_buffer[i] = np.append(steer[i], buf[:-1])

            accl = self.long_action.act(speed[i], self.state[i], self.params)
            sv = self.steer_action.act(applied, self.state[i], self.params)
            self.state[i] = self.integrator.integrate(
                f=self.model.f_dynamics,
                x=self.state[i],
                u=np.array([sv, accl]),
                dt=self.dt,
                params=self.params,
            )
            self.state[i][4] %= 2 * np.pi
