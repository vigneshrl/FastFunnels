"""GCBF+ side of the corridor baseline.

GCBF+ was trained on square arenas of scattered rectangles, in its own length
units (``car_radius = 0.05``, ``comm_radius = 0.5``, ``area_size = 4``).  To run
the pretrained DubinsCar policy on a FastFunnels corridor without retraining or
rescaling the network, the *map* is converted into those units instead:

    scale = car_radius_gcbf / robot_radius_m

so an agent is exactly the disc GCBF+ expects, and every distance the policy
sees -- gap width, obstacle size, lidar range -- keeps its true ratio to the
robot.  Time is not rescaled: ``dt = 0.03 s`` stays real seconds, which puts the
policy's 0.8 unit/s speed cap at ``0.8 / scale`` m/s.

Nothing in ``gcbfplus`` is modified.  ``CorridorDubinsCar`` only replaces the
random ``reset`` with a deterministic one driven by the map.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import Callable

import jax
import jax.numpy as jnp
import numpy as np

_GCBF_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "gcbfplus")
if _GCBF_ROOT not in sys.path:
    sys.path.insert(0, _GCBF_ROOT)

from gcbfplus.env.dubins_car import DubinsCar  # noqa: E402
from gcbfplus.env.obstacle import Rectangle  # noqa: E402
from gcbfplus.utils.graph import GraphsTuple  # noqa: E402

__all__ = ["Frame", "CorridorDubinsCar", "build_obstacles", "make_policy_step",
           "load_gcbf_policy"]


@dataclass(frozen=True)
class Frame:
    """Affine map between map metres and GCBF+ units."""

    scale: float
    offset: np.ndarray  # metres; the origin of the GCBF+ frame

    def to_gcbf(self, xy_m: np.ndarray) -> np.ndarray:
        return (np.asarray(xy_m, dtype=float) - self.offset) * self.scale

    def to_metres(self, xy_g: np.ndarray) -> np.ndarray:
        return np.asarray(xy_g, dtype=float) / self.scale + self.offset

    def speed_to_metres(self, v_g: np.ndarray) -> np.ndarray:
        return np.asarray(v_g, dtype=float) / self.scale


class CorridorDubinsCar(DubinsCar):
    """DubinsCar whose obstacles and start/goal come from a corridor map.

    ``reset`` is never called -- states are built explicitly by the runner -- but
    the class still has to exist so that ``get_graph``, ``u_ref`` and the action
    limits stay exactly the ones the pretrained policy was trained against.
    """

    def __init__(self, num_agents: int, area_size: float, max_step: int,
                 params: dict, dt: float = 0.03):
        super().__init__(num_agents=num_agents, area_size=area_size,
                         max_step=max_step, dt=dt, params=params)

    def reset(self, key):  # pragma: no cover - the runner supplies the state
        raise NotImplementedError(
            "CorridorDubinsCar states come from the map; use the runner instead"
        )


def build_obstacles(rects_m: np.ndarray, frame: Frame, n_rect: int) -> Rectangle:
    """Convert metre-space wall rectangles into a padded GCBF+ obstacle batch.

    A fixed ``n_rect`` keeps the traced shapes identical from map to map, so the
    policy step is compiled once for a whole split rather than once per map.
    Padding rectangles are placed far outside the corridor and are too small for
    any ray to reach.
    """
    if len(rects_m) > n_rect:
        raise ValueError(
            f"map needs {len(rects_m)} rectangles but n_rect is {n_rect}"
        )

    centres = frame.to_gcbf(rects_m[:, :2])
    widths = rects_m[:, 2] * frame.scale
    heights = rects_m[:, 3] * frame.scale
    thetas = rects_m[:, 4]

    pad = n_rect - len(rects_m)
    if pad:
        far = np.full((pad, 2), 1.0e4)
        centres = np.concatenate([centres, far], axis=0)
        widths = np.concatenate([widths, np.full(pad, 1.0e-3)])
        heights = np.concatenate([heights, np.full(pad, 1.0e-3)])
        thetas = np.concatenate([thetas, np.zeros(pad)])

    return jax.vmap(Rectangle.create)(
        jnp.asarray(centres, dtype=jnp.float32),
        jnp.asarray(widths, dtype=jnp.float32),
        jnp.asarray(heights, dtype=jnp.float32),
        jnp.asarray(thetas, dtype=jnp.float32),
    )


def load_gcbf_policy(path: str, step: int | None = None):
    """Load a pretrained GCBF+ checkpoint and return ``(act_fn, env_factory)``.

    ``env_factory(num_agents, area_size, max_step, params)`` builds the matching
    :class:`CorridorDubinsCar`; the algorithm is constructed against it so the
    network sees the same node/edge dimensions it was trained with.
    """
    import yaml

    with open(os.path.join(path, "config.yaml")) as f:
        config = yaml.load(f, Loader=yaml.UnsafeLoader)
    if config.env != "DubinsCar":
        raise ValueError(
            f"this baseline drives the DubinsCar policy; {path} holds {config.env}"
        )

    model_path = os.path.join(path, "models")
    if step is None:
        step = max(int(d) for d in os.listdir(model_path) if d.isdigit())

    def build(num_agents: int, area_size: float, max_step: int, params: dict):
        from gcbfplus.algo import make_algo

        env = CorridorDubinsCar(num_agents, area_size, max_step, params)
        algo = make_algo(
            algo=config.algo,
            env=env,
            node_dim=env.node_dim,
            edge_dim=env.edge_dim,
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            n_agents=env.num_agents,
            gnn_layers=config.gnn_layers,
            batch_size=config.batch_size,
            buffer_size=config.buffer_size,
            horizon=config.horizon,
            lr_actor=config.lr_actor,
            lr_cbf=config.lr_cbf,
            alpha=config.alpha,
            eps=0.02,
            inner_epoch=8,
            loss_action_coef=config.loss_action_coef,
            loss_unsafe_coef=config.loss_unsafe_coef,
            loss_safe_coef=config.loss_safe_coef,
            loss_h_dot_coef=config.loss_h_dot_coef,
            max_grad_norm=2.0,
            seed=config.seed,
        )
        algo.load(model_path, step)
        return env, algo

    return build, config, step


def make_policy_step(env: CorridorDubinsCar, algo, policy: str = "gcbf") -> Callable:
    """One jitted control step.

    ``policy="gcbf"`` uses the learned actor exactly as ``gcbfplus/test.py``
    does; ``policy="u_ref"`` runs the environment's nominal goal-tracking PID
    with no safety filter, which is the reference point that says how much of
    any failure is GCBF+ and how much is the corridor itself.

    Returns ``(next_agent_state, action)`` for agent states, goal states and a
    padded obstacle batch, all in GCBF+ units.  Compiled once per (agent count,
    rectangle count) pair.
    """
    if policy == "gcbf":
        act = algo.act
    elif policy == "u_ref":
        act = env.u_ref
    else:
        raise ValueError(f"unknown policy {policy!r}")

    @jax.jit
    def step(agent: jnp.ndarray, goal: jnp.ndarray, obstacle: Rectangle):
        graph: GraphsTuple = env.get_graph(env.EnvState(agent, goal, obstacle))
        action = env.clip_action(act(graph))
        stop = env.stop_mask(graph)
        nxt = env.agent_step_euler(agent, action, stop)
        return nxt, action

    return step
