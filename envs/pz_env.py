"""
PettingZoo ParallelEnv wrapper for agent-only training.

Exposes agent_0 as a PettingZoo parallel agent backed by JointEnv.
The patch runs a fixed/heuristic policy internally — the agent learns to stay inside.

Usage with SuperSuit:
    import supersuit as ss
    from envs.pz_env import AgentEnv

    env = AgentEnv()
    env = ss.pettingzoo_env_to_vec_env_v1(env)   # gym VecEnv, 1 env (one agent)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=4)  # 8 total

Notes:
- Single agent with obs/action spaces (21D / 2D).
- patch_policy: callable(patch_obs[11D]) → action[4], or None for a simple heuristic.
- At each step the patch acts first (heuristic or frozen policy), then agent acts.
"""

from __future__ import annotations

import functools
from typing import Callable, Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    from pettingzoo import ParallelEnv
except ImportError as e:
    raise ImportError("pettingzoo is required: pip install pettingzoo") from e

try:
    from .ppo_policy import JointEnv, JointEnvConfig
except ImportError:
    from envs.ppo_policy import JointEnv, JointEnvConfig

# Sentinel used when an agent has already terminated and no obs is available
_ZERO_OBS = None  # filled lazily per env instance

# ---------------------------------------------------------------------------
# Default patch heuristic: maintain mid-range speed, zero steering
# ---------------------------------------------------------------------------

def _default_patch_policy(patch_obs: np.ndarray) -> np.ndarray:
    """Simple heuristic patch action: straight at moderate speed, mid-size."""
    # action = [steer, speed, a_cmd, b_cmd]
    return np.array([0.0, 3.0, 2.0, 1.5], dtype=np.float32)


# ---------------------------------------------------------------------------
# PettingZoo ParallelEnv
# ---------------------------------------------------------------------------

class AgentEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for N learning agents inside a moving patch.

    The patch is controlled by `patch_policy` (default: heuristic).
    The patch car is a real f110 car providing physical inertia;
    all N agents learn to stay inside the patch ellipse.

    Parameters
    ----------
    env_config : dict
        Passed to JointEnvConfig (only recognised fields are forwarded).
        Use {"num_agents": N} to set the number of learning agents (1–4).
    patch_policy : callable, optional
        (patch_obs: np.ndarray[86]) -> np.ndarray[4]
        If None, uses the built-in heuristic.
    """

    metadata = {"render_modes": [], "name": "agent_env_v0"}
    render_mode = None

    def __init__(
        self,
        env_config: dict | None = None,
        patch_policy: Callable[[np.ndarray], np.ndarray] | None = None,
        **kwargs,  # absorbs obs_space/act_space injected by SuperSuit subprocess path
    ) -> None:
        super().__init__()

        cfg_kwargs = {k: v for k, v in (env_config or {}).items()
                      if hasattr(JointEnvConfig, k)}
        self.joint_env = JointEnv(JointEnvConfig(**cfg_kwargs))
        n = self.joint_env.num_agents
        self.patch_policy: Callable[[np.ndarray], np.ndarray] = (
            patch_policy if patch_policy is not None else _default_patch_policy
        )

        # PettingZoo required attribute — dynamic based on num_agents
        self.possible_agents: list[str] = [f"agent_{i}" for i in range(n)]

        obs_dim = JointEnv.agent_obs_dim(
            n,
            use_lidar=self.joint_env.cfg.agent_use_lidar,
            n_lidar_beams=self.joint_env.cfg.num_agent_lidar_beams,
        )
        obs_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        # Action = [steering, speed]. BOTH dims are centered on 0 so a freshly
        # initialised policy (raw output ≈ 0) starts at a sane action:
        #   steering ∈ [-0.4189, 0.4189]  (rad, 0 = straight)
        #   speed    ∈ [-1, 1]            (normalised; env maps to 2..10 m/s,
        #                                  0 → 6 m/s, the middle of the range)
        # The raw [2,10] speed range was a bug: 0 clipped up to the 2 m/s floor,
        # pinning every fresh agent at minimum throttle.
        act_space = spaces.Box(
            low=np.array([-0.4189, -1.0], dtype=np.float32),
            high=np.array([0.4189, 1.0], dtype=np.float32),
        )
        self._obs_space = obs_space
        self._act_space = act_space
        self._zero_obs = np.zeros(obs_dim, dtype=np.float32)

        self.agents: list[str] = []      # populated on reset()

    # ------------------------------------------------------------------
    # Space accessors (PettingZoo API)
    # ------------------------------------------------------------------

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        return self._obs_space

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
        return self._act_space

    
    def set_patch_policy(self, fn):
        self.patch_policy = fn
    # ------------------------------------------------------------------
    # Core PettingZoo interface
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
        self.joint_env.reset(seed=seed, options=options)
        self.agents = list(self.possible_agents)

        obs = {
            f"agent_{i}": self.joint_env._step_obs[1 + i].copy()
            for i in range(self.joint_env.num_agents)
        }
        infos: Dict[str, dict] = {f"agent_{i}": {} for i in range(self.joint_env.num_agents)}
        return obs, infos

    def step(
        self,
        actions: Dict[str, np.ndarray],
    ) -> Tuple[
        Dict[str, np.ndarray],
        Dict[str, float],
        Dict[str, bool],
        Dict[str, bool],
        Dict[str, dict],
    ]:
        n = self.joint_env.num_agents

        # Patch acts via its policy (heuristic or frozen trained policy)
        patch_obs = self.joint_env._step_obs[0]
        patch_action = self.patch_policy(patch_obs)

        # Build (N, 2) agent action array — fall back to zeros for missing agents
        agent_actions = np.array(
            [
                actions.get(f"agent_{i}", self._zero_obs[:2])
                for i in range(n)
            ],
            dtype=np.float32,
        )

        self.joint_env._real_agents_active = True
        self.joint_env._step_with_frozen_policy(
            np.asarray(patch_action, dtype=np.float32),
            agent_actions,
        )

        done = self.joint_env._step_terminated or self.joint_env._step_truncated
        base_info = dict(self.joint_env._step_info)

        obs, rewards, terminated, truncated, infos = {}, {}, {}, {}, {}
        for i in range(n):
            aid = f"agent_{i}"
            obs[aid]        = self.joint_env._step_obs[1 + i].copy()
            rewards[aid]    = float(self.joint_env._step_rewards[1 + i])
            terminated[aid] = self.joint_env._step_terminated
            truncated[aid]  = self.joint_env._step_truncated
            infos[aid]      = base_info

        if done:
            self.agents = []   # PettingZoo: clear agents list on episode end

        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        self.joint_env.close()

#old code 
# """
# PettingZoo ParallelEnv wrapper for agent-only training.

# Exposes agent_0 as a PettingZoo parallel agent backed by JointEnv.
# The patch runs a fixed/heuristic policy internally — the agent learns to stay inside.

# Usage with SuperSuit:
#     import supersuit as ss
#     from envs.pz_env import AgentEnv

#     env = AgentEnv()
#     env = ss.pettingzoo_env_to_vec_env_v1(env)   # gym VecEnv, 1 env (one agent)
#     env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=4)  # 8 total

# Notes:
# - Single agent with obs/action spaces (21D / 2D).
# - patch_policy: callable(patch_obs[11D]) → action[4], or None for a simple heuristic.
# - At each step the patch acts first (heuristic or frozen policy), then agent acts.
# """

# from __future__ import annotations

# import functools
# from typing import Callable, Dict, Optional, Tuple

# import numpy as np
# import gymnasium as gym
# from gymnasium import spaces

# try:
#     from pettingzoo import ParallelEnv
# except ImportError as e:
#     raise ImportError("pettingzoo is required: pip install pettingzoo") from e

# try:
#     from .ppo_policy import JointEnv, JointEnvConfig
# except ImportError:
#     from envs.ppo_policy import JointEnv, JointEnvConfig

# # ---------------------------------------------------------------------------
# # Default patch heuristic: maintain mid-range speed, zero steering
# # ---------------------------------------------------------------------------

# def _default_patch_policy(patch_obs: np.ndarray) -> np.ndarray:
#     """Simple heuristic patch action: straight at moderate speed, mid-size."""
#     # action = [steer, speed, a_cmd, b_cmd]
#     return np.array([0.0, 3.0, 2.0, 1.5], dtype=np.float32)


# # ---------------------------------------------------------------------------
# # PettingZoo ParallelEnv
# # ---------------------------------------------------------------------------

# class AgentEnv(ParallelEnv):
#     """
#     PettingZoo ParallelEnv for a single learning agent.

#     The patch is controlled by `patch_policy` (default: heuristic).
#     The patch car is a real f110 car providing physical inertia;
#     the single agent learns to stay inside the patch ellipse.

#     Parameters
#     ----------
#     env_config : dict
#         Passed to JointEnvConfig (only recognised fields are forwarded).
#     patch_policy : callable, optional
#         (patch_obs: np.ndarray[11]) -> np.ndarray[4]
#         If None, uses the built-in heuristic.
#     """

#     metadata = {"render_modes": [], "name": "agent_env_v0"}
#     render_mode = None

#     # PettingZoo required attributes
#     possible_agents = ["agent_0"]

#     def __init__(
#         self,
#         env_config: dict | None = None,
#         patch_policy: Callable[[np.ndarray], np.ndarray] | None = None,
#         **kwargs,  # absorbs obs_space/act_space injected by SuperSuit subprocess path
#     ) -> None:
#         super().__init__()

#         cfg_kwargs = {k: v for k, v in (env_config or {}).items()
#                       if hasattr(JointEnvConfig, k)}
#         self.joint_env = JointEnv(JointEnvConfig(**cfg_kwargs))
#         self.patch_policy: Callable[[np.ndarray], np.ndarray] = (
#             patch_policy if patch_policy is not None else _default_patch_policy
#         )

#         # Observation / action spaces (identical for both agents)
#         obs_space = spaces.Box(
#             low=-np.inf, high=np.inf, shape=(JointEnv.AGENT_OBS_DIM,), dtype=np.float32
#         )
#         act_space = spaces.Box(
#             low=np.array([-0.4189, 0.5], dtype=np.float32),
#             high=np.array([0.4189, 10.0], dtype=np.float32),
#         )
#         self._obs_space = obs_space
#         self._act_space = act_space

#         self.agents: list[str] = []      # populated on reset()

#     # ------------------------------------------------------------------
#     # Space accessors (PettingZoo API)
#     # ------------------------------------------------------------------

#     @functools.lru_cache(maxsize=None)
#     def observation_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
#         return self._obs_space

#     @functools.lru_cache(maxsize=None)
#     def action_space(self, agent: str) -> spaces.Space:  # type: ignore[override]
#         return self._act_space

    
#     def set_patch_policy(self, fn):
#         self.patch_policy = fn
#     # ------------------------------------------------------------------
#     # Core PettingZoo interface
#     # ------------------------------------------------------------------

#     def reset(
#         self,
#         seed: int | None = None,
#         options: dict | None = None,
#     ) -> Tuple[Dict[str, np.ndarray], Dict[str, dict]]:
#         self.joint_env.reset(seed=seed, options=options)
#         self.agents = list(self.possible_agents)

#         obs = {
#             "agent_0": self.joint_env._step_obs[1].copy(),
#         }
#         infos: Dict[str, dict] = {"agent_0": {}}
#         return obs, infos

#     def step(
#         self,
#         actions: Dict[str, np.ndarray],
#     ) -> Tuple[
#         Dict[str, np.ndarray],
#         Dict[str, float],
#         Dict[str, bool],
#         Dict[str, bool],
#         Dict[str, dict],
#     ]:
#         # Patch acts via its policy (heuristic or frozen trained policy)
#         patch_obs = self.joint_env._step_obs[0]
#         patch_action = self.patch_policy(patch_obs)

#         # Signal that real agent actions are being passed (not Phase 1 dummy zeros)
#         self.joint_env._real_agents_active = True
#         self.joint_env._step_with_frozen_policy(
#             np.asarray(patch_action,          dtype=np.float32),
#             np.asarray(actions["agent_0"],    dtype=np.float32),
#         )

#         obs = {
#             "agent_0": self.joint_env._step_obs[1].copy(),
#         }
#         rewards = {
#             "agent_0": float(self.joint_env._step_rewards[1]),
#         }

#         done = self.joint_env._step_terminated or self.joint_env._step_truncated
#         terminated = {"agent_0": self.joint_env._step_terminated}
#         truncated  = {"agent_0": self.joint_env._step_truncated}

#         base_info = dict(self.joint_env._step_info)
#         infos = {"agent_0": base_info}

#         if done:
#             self.agents = []   # PettingZoo: clear agents list on episode end

#         return obs, rewards, terminated, truncated, infos

#     def close(self) -> None:
#         self.joint_env.close()