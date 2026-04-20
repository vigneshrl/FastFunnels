#!/usr/bin/env python3
"""
Visualization and Evaluation Scripts
====================================

Contains:
- Evaluation functions for both policies
- Visualization tools
- Performance metrics
"""

import os
import numpy as np
import matplotlib
# Prefer TkAgg for interactive runs; fall back to Agg for headless environments.
# try:
#     matplotlib.use("TkAgg")
# except Exception:
#     matplotlib.use("Agg")
matplotlib.use('TkAgg')  # headless: no display/X11 needed
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    print("ERROR: stable-baselines3 required!")
    SB3_AVAILABLE = False

try:
    from ..ppo_policy import PatchEnv, PatchCarEnv, PatchEnvConfig
    # AgentEnv, AgentEnvConfig
    from ..ppo_policy import JointEnv, JointEnvConfig
except ImportError:
    from envs.ppo_policy import PatchEnv, PatchCarEnv, PatchEnvConfig
    # AgentEnv, AgentEnvConfig
    from envs.ppo_policy import JointEnv, JointEnvConfig


def _resolve_vecnorm_path(policy_path):
    """Resolve VecNormalize stats file path for best/final/checkpoint model names."""
    base_path = policy_path[:-4] if policy_path.endswith(".zip") else policy_path
    model_name = os.path.basename(base_path)
    model_dir = os.path.dirname(base_path) or "."

    candidates = []

    if model_name in {"best_model", "final_model", "interrupted_model"}:
        candidates.append(os.path.join(model_dir, model_name.replace("_model", "_vecnormalize.pkl")))
    elif model_name.startswith("checkpoint_"):
        candidates.append(base_path + "_vecnormalize.pkl")

    # joint_sb3 naming: patch_sb3_final → patch_sb3_vecnorm_final.pkl
    #                   agents_sb3_final → agents_sb3_vecnorm_final.pkl
    if model_name.endswith("_final"):
        prefix = model_name[: -len("_final")]
        candidates.append(os.path.join(model_dir, f"{prefix}_vecnorm_final.pkl"))

    candidates.extend(
        [
            base_path + "_vecnormalize.pkl",
            os.path.join(model_dir, "best_vecnormalize.pkl"),
            os.path.join(model_dir, "final_vecnormalize.pkl"),
            os.path.join(model_dir, "interrupted_vecnormalize.pkl"),
        ]
    )

    # Return first existing candidate while preserving order and uniqueness.
    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        if os.path.exists(candidate):
            return candidate
    return None


def evaluate_patch_policy(
    patch_policy_path,
    agent_policy_path=None,
    num_episodes=10,
    render=False,
    visualise=False,
    # patch_only_mode=patch_only_mode,
    plot=True
):
    """
    Evaluate patch policy.
    
    Args:
        patch_policy_path: Path to patch policy model
        agent_policy_path: Path to agent policy (or None for heuristic)
        num_episodes: Number of episodes to run
        render: Whether to render episodes
        visualise: Whether to run environment-specific visualisation
        plot: Whether to generate plots
    """
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return
    
    print("=" * 70)
    print("EVALUATING PATCH POLICY")
    print("=" * 70)
    print(f"Patch policy: {patch_policy_path}")
    if agent_policy_path:
        print(f"Agent policy: {agent_policy_path}")
    else:
        print("Using heuristic agents")
    print(f"Episodes: {num_episodes}")
    print("=" * 70)
    
    # Load policies
    patch_policy = PPO.load(patch_policy_path)
    agent_policy = None
    if agent_policy_path and os.path.exists(agent_policy_path + ".zip"):
        agent_policy = PPO.load(agent_policy_path)
    
    # PatchCarEnv steps the real f110 car each step (same as --mode patch training).
    # Plain PatchEnv only integrates the bicycle DynamicPatch and never calls f110.step(),
    # so the pygame window would show a frozen car at spawn.
    env = PatchCarEnv(
        PatchEnvConfig(
            num_agents=2,
            render_mode="human" if render else None,
            domain_randomize=False,
            random_spawn=False,
            split_mode=True,
        )
    )
        
    vec_env = DummyVecEnv([lambda: env])
    
    vecnorm_path = _resolve_vecnorm_path(patch_policy_path)
    if vecnorm_path:
        print(f"Loading normalization stats: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("WARNING: VecNormalize stats not found. Continuing without normalization stats.")
    
    # Run episodes and collect metrics
    all_rewards = []
    all_progress = []
    termination_reasons = defaultdict(int)
    all_episodes = []
    
    # for ep in range(num_episodes):
    obs = vec_env.reset()
    total_reward = 0
    step = 0
        
    # Collect metrics for this episode
    metrics = {
        'timestep': [],
        'patch_velocity': [],
        'patch_x': [],
        'patch_y': [],
        'patch_a': [],
        'patch_b': [],
        'lap_progress': [],
        'cumulative_reward': [],
    }
        
    inner_env = vec_env.envs[0]
    if hasattr(inner_env, 'env'):
        inner_env = inner_env.env
    
    while True:
        action, _ = patch_policy.predict(obs, deterministic=False)
        obs, reward, done, info = vec_env.step(action)
        total_reward += reward[0]
        step += 1
        
        if render and hasattr(inner_env, 'render'):
            inner_env.render()

        if visualise and hasattr(inner_env, '_visualize'):
            inner_env._visualize()

        # # Collect metrics
        # metrics['timestep'].append(step)
        # if hasattr(inner_env, 'patch'):
        #     metrics['patch_velocity'].append(inner_env.patch.v)
        #     metrics['patch_x'].append(inner_env.patch.x)
        #     metrics['patch_y'].append(inner_env.patch.y)
        #     patch_size = info[0].get("patch_size", (inner_env.patch.a, inner_env.patch.b))
        #     metrics['patch_a'].append(patch_size[0] if isinstance(patch_size, (list, tuple)) else inner_env.patch.a)
        #     metrics['patch_b'].append(patch_size[1] if isinstance(patch_size, (list, tuple)) else inner_env.patch.b)
        # else:
        #     metrics['patch_velocity'].append(info[0].get("patch_velocity", 0))
        #     metrics['patch_x'].append(0)
        #     metrics['patch_y'].append(0)
        #     patch_size = info[0].get("patch_size", (3.0, 2.5))
        #     metrics['patch_a'].append(patch_size[0] if isinstance(patch_size, (list, tuple)) else 3.0)
        #     metrics['patch_b'].append(patch_size[1] if isinstance(patch_size, (list, tuple)) else 2.5)
        
        # metrics['lap_progress'].append(info[0].get("lap_progress", 0) * 100)  # As percentage
        # metrics['cumulative_reward'].append(total_reward)
        
        if done[0]:
        #     progress = info[0].get("lap_progress", 0)
        #     reason = info[0].get("termination_reason", "truncated")
            
        #     all_rewards.append(total_reward)
        #     all_progress.append(progress)
        #     termination_reasons[reason] += 1
        #     all_episodes.append(metrics)
            
            # print(f"\nEpisode {ep+1}/{num_episodes} finished at step {step}")
            # print(f"  Reward: {total_reward:.1f} | Progress: {progress:.1%} | Reason: {reason}")
            progress = info[0].get("lap_progress", 0)
            reason = info[0].get("termination_reason", "truncated")
            print(
                f"  Reward: {total_reward:.1f} | Progress: {progress:.1%} | "
                f"Reason: {reason} | Steps: {step}"
            )
            total_reward = 0
            step = 0
            obs = vec_env.reset()
    
    # Summary
    print(f"\n{'='*70}")
    print("EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Avg Reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"Avg Progress: {np.mean(all_progress):.1%}")
    print(f"\nTermination Reasons:")
    for reason, count in termination_reasons.items():
        print(f"  {reason}: {count} ({count/num_episodes*100:.0f}%)")
    print(f"{'='*70}")
    
    # Generate plots
    if plot and all_episodes:
        _plot_patch_evaluation(all_episodes, num_episodes, patch_policy_path)
    
    vec_env.close()
    return all_rewards, all_progress


def _plot_patch_evaluation(all_episodes, num_episodes, model_path):
    """Generate plots for patch policy evaluation."""
    print("\nGenerating plots...")
    
    # Find max length for padding
    max_len = max(len(ep['timestep']) for ep in all_episodes)
    
    # Pad all episodes to same length (for mean/std calculation)
    def pad_to_length(arr, length):
        if len(arr) >= length:
            return arr[:length]
        return arr + [arr[-1]] * (length - len(arr))
    
    # Create arrays for mean/std
    timesteps = np.arange(1, max_len + 1)
    
    velocity_arr = np.array([pad_to_length(ep['patch_velocity'], max_len) for ep in all_episodes])
    progress_arr = np.array([pad_to_length(ep['lap_progress'], max_len) for ep in all_episodes])
    reward_arr = np.array([pad_to_length(ep['cumulative_reward'], max_len) for ep in all_episodes])
    patch_a_arr = np.array([pad_to_length(ep['patch_a'], max_len) for ep in all_episodes])
    patch_b_arr = np.array([pad_to_length(ep['patch_b'], max_len) for ep in all_episodes])
    
    # Calculate mean and std
    vel_mean, vel_std = velocity_arr.mean(axis=0), velocity_arr.std(axis=0)
    prog_mean, prog_std = progress_arr.mean(axis=0), progress_arr.std(axis=0)
    rew_mean, rew_std = reward_arr.mean(axis=0), reward_arr.std(axis=0)
    a_mean, a_std = patch_a_arr.mean(axis=0), patch_a_arr.std(axis=0)
    b_mean, b_std = patch_b_arr.mean(axis=0), patch_b_arr.std(axis=0)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(6, 5))
    fig.suptitle(f'Patch Policy Evaluation (n={num_episodes} episodes)', fontsize=14, fontweight='bold')
    
    # 1. Patch Velocity
    ax = axes[0, 0]
    ax.plot(timesteps, vel_mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(timesteps, vel_mean - vel_std, vel_mean + vel_std, alpha=0.3, color='blue')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Patch Velocity')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Lap Progress
    ax = axes[0, 1]
    ax.plot(timesteps, prog_mean, 'purple', linewidth=2)
    ax.fill_between(timesteps, prog_mean - prog_std, prog_mean + prog_std, alpha=0.3, color='purple')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Progress (%)')
    ax.set_title('Lap Progress')
    ax.grid(True, alpha=0.3)
    
    # 3. Cumulative Reward
    ax = axes[0, 2]
    ax.plot(timesteps, rew_mean, 'r-', linewidth=2)
    ax.fill_between(timesteps, rew_mean - rew_std, rew_mean + rew_std, alpha=0.3, color='red')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Reward Over Time')
    ax.grid(True, alpha=0.3)
    
    # 4. Patch Size A
    ax = axes[1, 0]
    ax.plot(timesteps, a_mean, 'orange', linewidth=2)
    ax.fill_between(timesteps, a_mean - a_std, a_mean + a_std, alpha=0.3, color='orange')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Semi-axis A (m)')
    ax.set_title('Patch Size (A)')
    ax.grid(True, alpha=0.3)
    
    # 5. Patch Size B
    ax = axes[1, 1]
    ax.plot(timesteps, b_mean, 'green', linewidth=2)
    ax.fill_between(timesteps, b_mean - b_std, b_mean + b_std, alpha=0.3, color='green')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Semi-axis B (m)')
    ax.set_title('Patch Size (B)')
    ax.grid(True, alpha=0.3)
    
    # 6. Patch Trajectory (show up to 5 episodes)
    ax = axes[1, 2]
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    for i, ep in enumerate(all_episodes[:5]):
        if len(ep['patch_x']) > 0 and len(ep['patch_y']) > 0:
            ax.plot(ep['patch_x'], ep['patch_y'], color=colors[i % len(colors)], alpha=0.7, 
                    linewidth=1.5, label=f'Ep {i+1}')
            if len(ep['patch_x']) > 0:
                ax.scatter(ep['patch_x'][0], ep['patch_y'][0], color=colors[i % len(colors)], 
                          marker='o', s=60, zorder=5)
                ax.scatter(ep['patch_x'][-1], ep['patch_y'][-1], color=colors[i % len(colors)], 
                          marker='x', s=60, zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Patch Trajectory (o=start, x=end)')
    ax.axis('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
    png_path = os.path.join(output_dir, 'patch_evaluation_plots.png')
    pdf_path = os.path.join(output_dir, 'patch_evaluation_plots.pdf')
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")
    
    plt.show()


def evaluate_agent_policy(
    agent_policy_path,
    patch_checkpoint_path,
    patch_vecnorm_path=None,
    num_episodes=10,
    num_agents= 2,
    render=False,
    visualise=False,):
    """
    Evaluate agent policy inside a frozen patch corridor.

    Args:
        agent_policy_path: Path to trained agent policy model (.zip)
        patch_checkpoint_path: Path to frozen patch policy model (.zip)
        patch_vecnorm_path: Path to patch VecNormalize stats (.pkl), or None
        num_episodes: Number of episodes to run
        render: Whether to call env.render() each step (f110 camera view)
        visualise: Whether to call env._visualize() each step (custom overlay)
    """
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return

    print("=" * 70)
    print("EVALUATING AGENT POLICY")
    print("=" * 70)
    print(f"Agent policy:  {agent_policy_path}")
    print(f"Patch policy:  {patch_checkpoint_path}")
    print(f"Patch vecnorm: {patch_vecnorm_path}")
    print(f"Episodes:      {num_episodes}")
    print("=" * 70)

    # Resolve patch vecnorm automatically if not provided
    if patch_vecnorm_path is None:
        patch_vecnorm_path = _resolve_vecnorm_path(patch_checkpoint_path)
        if patch_vecnorm_path:
            print(f"Auto-resolved patch vecnorm: {patch_vecnorm_path}")

    # Load agent policy
    agent_policy = PPO.load(agent_policy_path)
    agent_policy.policy.set_training_mode(False)

    # Load agent vecnorm if available
    agent_vecnorm_path = _resolve_vecnorm_path(agent_policy_path)
    agent_vecnorm = None

    # Build frozen patch_env
    patch_env = PatchEnv(PatchEnvConfig())
    patch_env.patch_policy = PPO.load(patch_checkpoint_path)
    patch_env.patch_policy.policy.set_training_mode(False)
    if patch_vecnorm_path and os.path.exists(patch_vecnorm_path):
        import gymnasium as gym
        from gymnasium import spaces
        class _DummyPatchEnv(gym.Env):
            observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
            action_space = spaces.Box(
                low=np.array([-0.4189, 0.5, 0.325, 0.25], dtype=np.float32),
                high=np.array([0.4189, 10.0, 2.5, 1.5], dtype=np.float32),
            )
            def reset(self, **kw): return np.zeros(4, dtype=np.float32), {}
            def step(self, a): return np.zeros(4, dtype=np.float32), 0.0, False, False, {}
        from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv as _DVE
        patch_env.patch_vecnorm = VecNormalize.load(patch_vecnorm_path, _DVE([_DummyPatchEnv]))
        patch_env.patch_vecnorm.training = False
        patch_env.patch_vecnorm.norm_reward = False
    else:
        patch_env.patch_vecnorm = None

    # Create environment
    env = AgentEnv(
        AgentEnvConfig(
            patch_env=patch_env,
            render_mode="human" if render else None,
            random_spawn=False,
        )
    )
    vec_env = DummyVecEnv([lambda: env])

    if agent_vecnorm_path:
        print(f"Loading agent normalization stats: {agent_vecnorm_path}")
        vec_env = VecNormalize.load(agent_vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    else:
        print("WARNING: Agent VecNormalize stats not found. Continuing without.")

    # Unwrap to inner env for visualisation calls
    inner_env = vec_env.envs[0]
    if hasattr(inner_env, 'env'):
        inner_env = inner_env.env

    all_rewards = []
    all_steps = []
    termination_reasons = defaultdict(int)
    _obs_keys_printed = False  # print obs keys once

    for ep in range(num_episodes):
        obs = vec_env.reset()
        total_reward = 0.0
        step = 0
        prev_px, prev_py = None, None  # for position-delta check

        while True:
            get_patch_action, _ = agent_policy.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(get_patch_action)
            total_reward += float(reward[0])
            step += 1

            # ── per-step diagnostics (first 5 steps of first 2 episodes) ──
            if ep < 2 and step <= 5:
                b = getattr(inner_env, 'current_base_obs', None)
                if b is not None:
                    if not _obs_keys_printed:
                        print(f"\n[DIAG] base_obs keys: {list(b.keys())}")
                        _obs_keys_printed = True
                    px = list(b.get("poses_x", [0, 0]))
                    py = list(b.get("poses_y", [0, 0]))
                    vx = list(b.get("linear_vels_x", [0, 0]))
                    vy = list(b.get("linear_vels_y", [0, 0]))  # may be empty list
                    spd0 = float(np.hypot(vx[0] if vx else 0, vy[0] if vy else 0))
                    spd1 = float(np.hypot(vx[1] if len(vx) > 1 else 0, vy[1] if len(vy) > 1 else 0))
                    # position delta vs previous step
                    if prev_px is not None:
                        d0 = float(np.hypot(px[0]-prev_px[0], py[0]-prev_py[0]))
                        d1 = float(np.hypot(px[1]-prev_px[1], py[1]-prev_py[1]))
                        delta_str = f"  pos_delta A0:{d0:.4f}m A1:{d1:.4f}m"
                    else:
                        delta_str = ""
                    print(
                        f"[Ep{ep+1} Step{step}] "
                        # f"cmd_speed=[{action[0][1]:.3f},{action[0][3]:.3f}] "
                        f"vx={[f'{v:.3f}' for v in vx]}  vy={[f'{v:.3f}' for v in vy]} "
                        f"speed_mag A0:{spd0:.3f} A1:{spd1:.3f}"
                        + delta_str
                    )
                    prev_px, prev_py = px, py
                else:
                    print(f"[Ep{ep+1} Step{step}] current_base_obs is None")
            # ── end diagnostics ──

            if render and hasattr(inner_env, 'render'):
                inner_env.render()

            if visualise and hasattr(inner_env, '_visualize'):
                inner_env._visualize()

            if done[0]:
                reason = info[0].get("termination_reason", "truncated")
                termination_reasons[reason] += 1
                all_rewards.append(total_reward)
                all_steps.append(step)
                base_obs = getattr(inner_env, 'current_base_obs', None)
                if base_obs is not None and "linear_vels_x" in base_obs:
                    vx = base_obs["linear_vels_x"]
                    vy = base_obs.get("linear_vels_y", [0.0, 0.0])
                    # handle empty list from f110
                    vy = vy if len(vy) >= 2 else [0.0, 0.0]
                    print(f"vx :{vx}, vy:{vy}")
                    spd0 = float(np.hypot(vx[0], vy[0]))
                    spd1 = float(np.hypot(vx[1], vy[1]))
                    speed_str = f"A0:{spd0:.2f} m/s  A1:{spd1:.2f} m/s"
                    # speed_str = f"A0:{vx:.2f} m/s A1:{vy:.2f} m/s"
                else:
                    speed_str = "N/A"
                if base_obs is not None and "poses_x" in base_obs:
                    pos_str = (f"X0:{base_obs['poses_x'][0]:.2f} Y0:{base_obs['poses_y'][0]:.2f} | "
                               f"X1:{base_obs['poses_x'][1]:.2f} Y1:{base_obs['poses_y'][1]:.2f}")
                else:
                    pos_str = "N/A"
                print(
                    f"Episode {ep+1}/{num_episodes} | "
                    f"Reward: {total_reward:.1f} | Steps: {step} | Reason: {reason} | "
                    f"Speed: {speed_str} | Poses: {pos_str}"
                )
                break

    print(f"\n{'='*70}")
    print("AGENT EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Avg Reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
    print(f"Avg Steps:  {np.mean(all_steps):.0f} ± {np.std(all_steps):.0f}")
    print(f"\nTermination Reasons:")
    for reason, count in termination_reasons.items():
        print(f"  {reason}: {count} ({count/num_episodes*100:.0f}%)")
    print(f"{'='*70}")

    vec_env.close()
    return all_rewards, all_steps


def evaluate_joint_policy(
    patch_model_path: str,
    agent_model_path: str,
    patch_vecnorm_path: str = None,
    agent_vecnorm_path: str = None,
    num_episodes: int = 10,
    render: bool = False,
    plot: bool = True,
    live_viz: bool = False,
    random_spawn: bool = True,
):
    """Evaluate the jointly-trained patch + agent policies on a fresh JointEnv.

    Both models are loaded from their SB3 .zip files. VecNormalize stats are
    applied to observations before passing to each policy (identical to training).

    Parameters
    ----------
    patch_model_path : path to patch_sb3_final.zip (or similar)
    agent_model_path : path to agents_sb3_final.zip (or similar)
    patch_vecnorm_path : path to patch VecNormalize .pkl  (auto-resolved if None)
    agent_vecnorm_path : path to agent VecNormalize .pkl  (auto-resolved if None)
    num_episodes : how many episodes to run
    render : call env._visualize() for a live matplotlib view each step
    plot : generate summary plots after evaluation
    live_viz : show a real-time matplotlib window of patch + agent positions
    """
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return

    print("=" * 70)
    print("EVALUATING JOINT POLICY (patch + agents)")
    print("=" * 70)
    print(f"Patch model : {patch_model_path}")
    print(f"Agent model : {agent_model_path}")
    print(f"Episodes    : {num_episodes}")
    print("=" * 70)

    # --- resolve vecnorm paths ---
    if patch_vecnorm_path is None:
        patch_vecnorm_path = _resolve_vecnorm_path(patch_model_path)
        if patch_vecnorm_path:
            print(f"Auto-resolved patch vecnorm : {patch_vecnorm_path}")
    if agent_vecnorm_path is None:
        agent_vecnorm_path = _resolve_vecnorm_path(agent_model_path)
        if agent_vecnorm_path:
            print(f"Auto-resolved agent vecnorm : {agent_vecnorm_path}")

    # --- load models ---
    patch_ppo = PPO.load(patch_model_path)
    patch_ppo.policy.set_training_mode(False)
    agent_ppo = PPO.load(agent_model_path)
    agent_ppo.policy.set_training_mode(False)

    # --- build dummy VecNormalize wrappers for obs normalisation only ---
    # Load VecNormalize stats via pickle to avoid obs-shape mismatch in set_venv.
    # We only need normalize_obs() — no real env needs to be attached.
    import pickle

    patch_vn = None
    if patch_vecnorm_path and os.path.exists(patch_vecnorm_path):
        with open(patch_vecnorm_path, "rb") as f:
            patch_vn = pickle.load(f)
        patch_vn.training = False
        patch_vn.norm_reward = False

    agent_vn = None
    if agent_vecnorm_path and os.path.exists(agent_vecnorm_path):
        with open(agent_vecnorm_path, "rb") as f:
            agent_vn = pickle.load(f)
        agent_vn.training = False
        agent_vn.norm_reward = False

    def _norm_patch(obs):
        if patch_vn is None:
            return obs
        if patch_vn.obs_rms.mean.shape != obs.shape:
            print(f"WARNING: patch vecnorm shape {patch_vn.obs_rms.mean.shape} != obs {obs.shape}; skipping normalisation")
            return obs
        return patch_vn.normalize_obs(obs[np.newaxis])[0]

    def _norm_agent(obs):
        if agent_vn is None:
            return obs
        if agent_vn.obs_rms.mean.shape != obs.shape:
            print(f"WARNING: agent vecnorm shape {agent_vn.obs_rms.mean.shape} != obs {obs.shape}; skipping normalisation")
            return obs
        return agent_vn.normalize_obs(obs[np.newaxis])[0]

    # render_mode="human" enables both f110 top-down view and matplotlib overlay
    env = JointEnv(JointEnvConfig(
        render_mode="human" if (render or live_viz) else None,
        random_spawn=random_spawn,
    ))
    env._real_agents_active = True   # real agent policies are running — enable f110 simulation

    # --- episode loop ---
    all_rewards_agents = []
    all_rewards_patch  = []
    all_ep_lens        = []
    termination_reasons = defaultdict(int)

    # per-episode trajectory storage
    all_patch_xs, all_patch_ys = [], []
    all_agent0_xs, all_agent0_ys = [], []
    all_agent1_xs, all_agent1_ys = [], []
    all_agents_inside = []
    all_patch_speeds  = []

    for ep in range(num_episodes):
        env.reset(seed=ep)
        ep_reward_agents = 0.0
        ep_reward_patch  = 0.0
        patch_xs, patch_ys = [], []
        agent0_xs, agent0_ys = [], []
        agent1_xs, agent1_ys = [], []
        agents_inside_ts = []
        patch_speed_ts   = []

        while True:
            # --- get observations ---
            patch_obs = env._step_obs[0]
            agent0_obs = env._step_obs[1]
            # agent1_obs = env._step_obs[2]

            # if patch_obs is None or agent0_obs is None or agent1_obs is None:
            if patch_obs is None or agent0_obs is None is None:
                break  # env not yet initialised (shouldn't happen after reset)

            # --- predict actions (with obs normalisation) ---
            patch_action, _ = patch_ppo.predict(_norm_patch(patch_obs), deterministic=True)
            agent0_action, _ = agent_ppo.predict(_norm_agent(agent0_obs), deterministic=True)
            # agent1_action, _ = agent_ppo.predict(_norm_agent(agent1_obs), deterministic=True)

            # --- step ---
            env._execute_joint_step(
                np.asarray(patch_action, dtype=np.float32),
                np.asarray(agent0_action, dtype=np.float32),
                # np.asarray(agent1_action, dtype=np.float32),
            )

            ep_reward_agents += float(np.mean([env._step_rewards[1]]))
            ep_reward_patch  += float(env._step_rewards[0])

            # record positions
            patch_xs.append(env.patch.x)
            patch_ys.append(env.patch.y)
            if env.current_base_obs is not None:
                bobs = env.current_base_obs
                agent0_xs.append(float(bobs["poses_x"][0]))
                agent0_ys.append(float(bobs["poses_y"][0]))
                agent1_xs.append(float(bobs["poses_x"][1]))
                agent1_ys.append(float(bobs["poses_y"][1]))
            agents_inside_ts.append(env._step_info.get("agents_inside", 0))
            patch_speed_ts.append(env.patch.v)

            # --- f110 gym window (top-down car view) ---
            if render or live_viz:
                env.render()

            # --- matplotlib overlay (patch ellipse + agent markers) ---
            if live_viz:
                env._visualize()

            if env._step_terminated or env._step_truncated:
                reason = env._step_info.get("termination_reason", "unknown")
                termination_reasons[reason] += 1
                all_rewards_agents.append(ep_reward_agents)
                all_rewards_patch.append(ep_reward_patch)
                all_ep_lens.append(env.step_count)
                all_patch_xs.append(patch_xs)
                all_patch_ys.append(patch_ys)
                all_agent0_xs.append(agent0_xs)
                all_agent0_ys.append(agent0_ys)
                all_agent1_xs.append(agent1_xs)
                all_agent1_ys.append(agent1_ys)
                all_agents_inside.append(agents_inside_ts)
                all_patch_speeds.append(patch_speed_ts)
                print(
                    f"Ep {ep+1:3d}/{num_episodes} | "
                    f"len={env.step_count:4d} | "
                    f"rew_agents={ep_reward_agents:8.1f} | "
                    f"rew_patch={ep_reward_patch:8.1f} | "
                    f"reason={reason}"
                )
                break

    env.close()  # also closes the matplotlib figure via JointEnv.close()

    # --- summary ---
    print(f"\n{'='*70}")
    print("JOINT EVALUATION SUMMARY")
    print(f"{'='*70}")
    print(f"Avg Episode Length : {np.mean(all_ep_lens):.0f} ± {np.std(all_ep_lens):.0f} steps")
    print(f"Avg Agent Reward   : {np.mean(all_rewards_agents):.1f} ± {np.std(all_rewards_agents):.1f}")
    print(f"Avg Patch Reward   : {np.mean(all_rewards_patch):.1f} ± {np.std(all_rewards_patch):.1f}")
    print(f"\nTermination Reasons:")
    for reason, count in sorted(termination_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count} ({count/num_episodes*100:.0f}%)")
    print(f"{'='*70}")

    if plot and all_patch_xs:
        _plot_joint_evaluation(
            all_patch_xs, all_patch_ys,
            all_agent0_xs, all_agent0_ys,
            all_agent1_xs, all_agent1_ys,
            all_agents_inside, all_patch_speeds,
            all_ep_lens, all_rewards_agents,
            num_episodes, patch_model_path,
        )

    return all_rewards_agents, all_ep_lens


def _plot_joint_evaluation(
    patch_xs, patch_ys,
    agent0_xs, agent0_ys,
    agent1_xs, agent1_ys,
    agents_inside_ts, patch_speed_ts,
    ep_lens, rewards_agents,
    num_episodes, model_path,
):
    """Generate summary plots for joint policy evaluation."""
    print("\nGenerating joint evaluation plots...")

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Joint Policy Evaluation  (n={num_episodes} episodes)", fontsize=14, fontweight="bold")
    colors = plt.cm.tab10.colors

    # 1. Patch + agent trajectories (up to 5 episodes)
    ax = axes[0, 0]
    for i in range(min(5, len(patch_xs))):
        c = colors[i % len(colors)]
        ax.plot(patch_xs[i], patch_ys[i], color=c, linewidth=1.5,
                alpha=0.8, label=f"Ep{i+1} patch")
        if agent0_xs[i]:
            ax.plot(agent0_xs[i], agent0_ys[i], color=c, linewidth=1,
                    linestyle="--", alpha=0.5)
            ax.plot(agent1_xs[i], agent1_ys[i], color=c, linewidth=1,
                    linestyle=":", alpha=0.5)
        ax.scatter(patch_xs[i][0], patch_ys[i][0], color=c, marker="o", s=40, zorder=5)
        ax.scatter(patch_xs[i][-1], patch_ys[i][-1], color=c, marker="x", s=40, zorder=5)
    ax.set_title("Trajectories (solid=patch, --=A0, :=A1)")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.axis("equal")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

    # 2. Agents-inside fraction over time (mean ± std across episodes)
    ax = axes[0, 1]
    max_len = max(len(ts) for ts in agents_inside_ts)
    def _pad(arr, length, fill=None):
        fill = arr[-1] if fill is None and arr else 0
        return arr + [fill] * (length - len(arr))
    inside_arr = np.array([_pad(ts, max_len) for ts in agents_inside_ts], dtype=float)
    t = np.arange(max_len)
    m, s = inside_arr.mean(0), inside_arr.std(0)
    ax.plot(t, m, color="blue", linewidth=2)
    ax.fill_between(t, np.clip(m - s, 0, 2), np.clip(m + s, 0, 2), alpha=0.25, color="blue")
    ax.set_ylim(-0.1, 2.5)
    ax.set_title("Agents Inside Patch (mean ± std)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Count")
    ax.axhline(2, color="green", linestyle="--", alpha=0.5, label="Both inside")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # 3. Patch speed over time
    ax = axes[0, 2]
    speed_arr = np.array([_pad(ts, max_len) for ts in patch_speed_ts], dtype=float)
    m, s = speed_arr.mean(0), speed_arr.std(0)
    ax.plot(t, m, color="orange", linewidth=2)
    ax.fill_between(t, m - s, m + s, alpha=0.25, color="orange")
    ax.set_title("Patch Speed (m/s)")
    ax.set_xlabel("Step")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)

    # 4. Episode lengths
    ax = axes[1, 0]
    ax.bar(range(1, len(ep_lens) + 1), ep_lens, color="steelblue", alpha=0.8)
    ax.axhline(np.mean(ep_lens), color="red", linestyle="--", label=f"Mean={np.mean(ep_lens):.0f}")
    ax.set_title("Episode Length per Episode")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Steps")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 5. Agent rewards per episode
    ax = axes[1, 1]
    ax.bar(range(1, len(rewards_agents) + 1), rewards_agents, color="salmon", alpha=0.8)
    ax.axhline(np.mean(rewards_agents), color="red", linestyle="--",
               label=f"Mean={np.mean(rewards_agents):.0f}")
    ax.set_title("Agent Episode Reward")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Cumulative Reward")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # 6. Agents-inside fraction per episode (summary bar)
    ax = axes[1, 2]
    inside_fracs = [np.mean(ts) / 2.0 for ts in agents_inside_ts]  # 0-1 fraction of max
    ax.bar(range(1, len(inside_fracs) + 1), inside_fracs, color="mediumseagreen", alpha=0.8)
    ax.axhline(np.mean(inside_fracs), color="red", linestyle="--",
               label=f"Mean={np.mean(inside_fracs):.2f}")
    ax.set_ylim(0, 1.05)
    ax.set_title("Avg Fraction of Steps Both Agents Inside")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Fraction (0–1)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else "."
    png_path = os.path.join(output_dir, "joint_evaluation_plots.png")
    pdf_path = os.path.join(output_dir, "joint_evaluation_plots.pdf")
    plt.savefig(png_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.savefig(pdf_path, bbox_inches="tight")
    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Patch or Agent Policy"
    )
    
    parser.add_argument("--policy", type=str, choices=["patch", "agent", "joint"], required=True,
                       help="Which policy to evaluate")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to policy model (.zip)")
    parser.add_argument("--patch-checkpoint", type=str, default=None,
                       help="Path to frozen patch policy (required for --policy agent)")
    parser.add_argument("--patch-vecnorm", type=str, default=None,
                       help="Path to patch VecNormalize stats (auto-resolved if omitted)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes (f110 camera view)")
    parser.add_argument("--visualise", action="store_true",
                       help="Visualise episodes (custom patch+agent overlay)")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable plotting")
    # joint-mode args
    parser.add_argument("--agent-model", type=str, default=None,
                       help="Path to agent PPO model (required for --policy joint)")
    parser.add_argument("--agent-vecnorm", type=str, default=None,
                       help="Path to agent VecNormalize .pkl (auto-resolved if omitted)")
    parser.add_argument("--live-viz", action="store_true",
                       help="Show real-time matplotlib window (joint mode)")
    parser.add_argument("--no-random-spawn", action="store_true",
                       help="Disable random spawn (start from fixed position)")

    args = parser.parse_args()

    if args.policy == "joint":
        if not args.agent_model:
            parser.error("--agent-model is required when --policy joint")
        evaluate_joint_policy(
            patch_model_path=args.model,
            agent_model_path=args.agent_model,
            patch_vecnorm_path=args.patch_vecnorm,
            agent_vecnorm_path=args.agent_vecnorm,
            num_episodes=args.episodes,
            render=args.render,
            plot=not args.no_plot,
            live_viz=args.live_viz,
            random_spawn=not args.no_random_spawn,
        )
    elif args.policy == "patch":
        evaluate_patch_policy(
            args.model,
            num_episodes=args.episodes,
            render=args.render,
            visualise=args.visualise,
            plot=not args.no_plot,
        )
    elif args.policy == "agent":
        if not args.patch_checkpoint:
            parser.error("--patch-checkpoint is required when --policy agent")
        evaluate_agent_policy(
            args.model,
            patch_checkpoint_path=args.patch_checkpoint,
            patch_vecnorm_path=args.patch_vecnorm,
            num_episodes=args.episodes,
            render=args.render,
            visualise=args.visualise,
        )


