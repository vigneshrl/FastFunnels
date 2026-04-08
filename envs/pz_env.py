"""
PettingZoo ParallelEnv wrapper for agent-only training.

Exposes agent_0 and agent_1 as PettingZoo parallel agents backed by JointEnv.
The patch runs a fixed/heuristic policy internally — agents learn to stay inside.

Usage with SuperSuit:
    import supersuit as ss
    from envs.pz_env import AgentEnv

    env = AgentEnv()
    env = ss.pettingzoo_env_to_vec_env_v1(env)   # gym VecEnv, 2 envs (one per agent)
    env = ss.concat_vec_envs_v1(env, num_vec_envs=8, num_cpus=4)  # 16 total

Notes:
- Both agents have identical obs/action spaces (24D / 2D) → no padding needed.
- patch_policy: callable(patch_obs) → action[4], or None for a simple heuristic.
- At each step the patch acts first (heuristic or frozen policy), then agents act.
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

# ---------------------------------------------------------------------------
# Default patch heuristic: maintain mid-range speed, zero steering
# ---------------------------------------------------------------------------

def _default_patch_policy(patch_obs: np.ndarray) -> np.ndarray:
    """Simple heuristic patch action: straight at moderate speed, mid-size."""
    # action = [steer, speed, a_cmd, b_cmd]
    return np.array([0.0, 3.0, 3.0, 2.5], dtype=np.float32)


# ---------------------------------------------------------------------------
# PettingZoo ParallelEnv
# ---------------------------------------------------------------------------

class AgentEnv(ParallelEnv):
    """
    PettingZoo ParallelEnv for the two learning agents.

    The patch is controlled by `patch_policy` (default: heuristic).
    agent_0 and agent_1 share the same obs/action space so SuperSuit's
    pettingzoo_env_to_vec_env_v1 stacks them without padding.

    Parameters
    ----------
    env_config : dict
        Passed to JointEnvConfig (only recognised fields are forwarded).
    patch_policy : callable, optional
        (patch_obs: np.ndarray[15]) -> np.ndarray[4]
        If None, uses the built-in heuristic.
    """

    metadata = {"render_modes": [], "name": "agent_env_v0"}
    render_mode = None

    # PettingZoo required attributes
    possible_agents = ["agent_0", "agent_1"]

    def __init__(
        self,
        env_config: dict | None = None,
        patch_policy: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        super().__init__()

        cfg_kwargs = {k: v for k, v in (env_config or {}).items()
                      if hasattr(JointEnvConfig, k)}
        self.joint_env = JointEnv(JointEnvConfig(**cfg_kwargs))
        self.patch_policy: Callable[[np.ndarray], np.ndarray] = (
            patch_policy if patch_policy is not None else _default_patch_policy
        )

        # Observation / action spaces (identical for both agents)
        obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)
        act_space = spaces.Box(
            low=np.array([-0.4189, 0.5], dtype=np.float32),
            high=np.array([0.4189, 12.0], dtype=np.float32),
        )
        self._obs_space = obs_space
        self._act_space = act_space

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
            "agent_0": self.joint_env._step_obs[1].copy(),
            "agent_1": self.joint_env._step_obs[2].copy(),
        }
        infos: Dict[str, dict] = {"agent_0": {}, "agent_1": {}}
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
        # Patch acts via its policy (heuristic or frozen trained policy)
        patch_obs = self.joint_env._step_obs[0]
        patch_action = self.patch_policy(patch_obs)

        # Signal that real agent actions are being passed (not Phase 1 dummy zeros)
        self.joint_env._real_agents_active = True
        self.joint_env._execute_joint_step(
            np.asarray(patch_action,          dtype=np.float32),
            np.asarray(actions["agent_0"],    dtype=np.float32),
            np.asarray(actions["agent_1"],    dtype=np.float32),
        )

        obs = {
            "agent_0": self.joint_env._step_obs[1].copy(),
            "agent_1": self.joint_env._step_obs[2].copy(),
        }
        rewards = {
            "agent_0": float(self.joint_env._step_rewards[1]),
            "agent_1": float(self.joint_env._step_rewards[2]),
        }

        done = self.joint_env._step_terminated or self.joint_env._step_truncated
        terminated = {"agent_0": self.joint_env._step_terminated,
                      "agent_1": self.joint_env._step_terminated}
        truncated  = {"agent_0": self.joint_env._step_truncated,
                      "agent_1": self.joint_env._step_truncated}

        base_info = dict(self.joint_env._step_info)
        infos = {"agent_0": base_info, "agent_1": base_info}

        if done:
            self.agents = []   # PettingZoo: clear agents list on episode end

        return obs, rewards, terminated, truncated, infos

    def close(self) -> None:
        self.joint_env.close()
