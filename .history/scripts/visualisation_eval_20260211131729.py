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
try:
    matplotlib.use("TkAgg")
except Exception:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    SB3_AVAILABLE = True
except ImportError:
    print("ERROR: stable-baselines3 required!")
    SB3_AVAILABLE = False

from patch_sempc import PatchEnv
# from neurocontrollers import AgentEnv


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
    
    # Create environment
    env = PatchEnv(
        num_agents=2,
        render_mode="human" if (render or visualise) else None,
        domain_randomize=False,
        agent_policy=agent_policy
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
    
    for ep in range(num_episodes):
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
            action, _ = patch_policy.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            total_reward += reward[0]
            step += 1
            
            if render and hasattr(inner_env, 'render'):
                inner_env.render()

            if visualise and hasattr(inner_env, '_visualize'):
                inner_env._visualize()

            # Collect metrics
            metrics['timestep'].append(step)
            if hasattr(inner_env, 'patch'):
                metrics['patch_velocity'].append(inner_env.patch.v)
                metrics['patch_x'].append(inner_env.patch.x)
                metrics['patch_y'].append(inner_env.patch.y)
                patch_size = info[0].get("patch_size", (inner_env.patch.a, inner_env.patch.b))
                metrics['patch_a'].append(patch_size[0] if isinstance(patch_size, (list, tuple)) else inner_env.patch.a)
                metrics['patch_b'].append(patch_size[1] if isinstance(patch_size, (list, tuple)) else inner_env.patch.b)
            else:
                metrics['patch_velocity'].append(info[0].get("patch_velocity", 0))
                metrics['patch_x'].append(0)
                metrics['patch_y'].append(0)
                patch_size = info[0].get("patch_size", (3.0, 2.5))
                metrics['patch_a'].append(patch_size[0] if isinstance(patch_size, (list, tuple)) else 3.0)
                metrics['patch_b'].append(patch_size[1] if isinstance(patch_size, (list, tuple)) else 2.5)
            
            metrics['lap_progress'].append(info[0].get("lap_progress", 0) * 100)  # As percentage
            metrics['cumulative_reward'].append(total_reward)
            
            if done[0]:
                progress = info[0].get("lap_progress", 0)
                reason = info[0].get("termination_reason", "truncated")
                
                all_rewards.append(total_reward)
                all_progress.append(progress)
                termination_reasons[reason] += 1
                all_episodes.append(metrics)
                
                print(f"\nEpisode {ep+1}/{num_episodes} finished at step {step}")
                print(f"  Reward: {total_reward:.1f} | Progress: {progress:.1%} | Reason: {reason}")
                break
    
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


# def evaluate_agent_policy(
#     agent_policy_path,
#     patch_policy_path=None,
#     num_episodes=10,
#     render=False,
#     plot=True
# ):
#     """
#     Evaluate agent policy.
    
#     Args:
#         agent_policy_path: Path to agent policy model
#         patch_policy_path: Path to patch policy (or None for heuristic)
#         num_episodes: Number of episodes to run
#         render: Whether to render episodes
#         plot: Whether to generate plots
#     """
#     if not SB3_AVAILABLE:
#         print("ERROR: stable-baselines3 required!")
#         return
    
#     print("=" * 70)
#     print("EVALUATING AGENT POLICY")
#     print("=" * 70)
#     print(f"Agent policy: {agent_policy_path}")
#     if patch_policy_path:
#         print(f"Patch policy: {patch_policy_path}")
#     else:
#         print("Using heuristic patch")
#     print(f"Episodes: {num_episodes}")
#     print("=" * 70)
    
#     # Load policies
#     agent_policy = PPO.load(agent_policy_path)
#     patch_policy = None
#     if patch_policy_path and os.path.exists(patch_policy_path + ".zip"):
#         patch_policy = PPO.load(patch_policy_path)
    
#     # Create environment
#     # env = AgentEnv(
#     #     num_agents=2,
#     #     render_mode="human" if render else None,
#     #     domain_randomize=False,
#     #     patch_policy=patch_policy
#     # )
    
#     vec_env = DummyVecEnv([lambda: env])
    
#     vecnorm_path = _resolve_vecnorm_path(agent_policy_path)
#     if vecnorm_path:
#         print(f"Loading normalization stats: {vecnorm_path}")
#         vec_env = VecNormalize.load(vecnorm_path, vec_env)
#         vec_env.training = False
#         vec_env.norm_reward = False
#     else:
#         print("WARNING: VecNormalize stats not found. Continuing without normalization stats.")
    
#     # Run episodes and collect metrics
#     all_rewards = []
#     all_containment = []
#     termination_reasons = defaultdict(int)
#     all_episodes = []
    
#     for ep in range(num_episodes):
#         obs = vec_env.reset()
#         total_reward = 0
#         step = 0
        
#         # Collect metrics for this episode
#         metrics = {
#             'timestep': [],
#             'containment_rate': [],
#             'cumulative_reward': [],
#         }
        
#         inner_env = vec_env.envs[0]
#         if hasattr(inner_env, 'env'):
#             inner_env = inner_env.env
        
#         while True:
#             action, _ = agent_policy.predict(obs, deterministic=True)
#             obs, reward, done, info = vec_env.step(action)
#             total_reward += reward[0]
#             step += 1

#             if render and hasattr(inner_env, 'render'):
#                 inner_env.render()
            
#             # Collect metrics
#             metrics['timestep'].append(step)
#             inside = info[0].get("agents_inside", 0)
#             metrics['containment_rate'].append(inside / 2.0)  # 2 agents
#             metrics['cumulative_reward'].append(total_reward)
            
#             if done[0]:
#                 reason = info[0].get("termination_reason", "truncated")
                
#                 all_rewards.append(total_reward)
#                 all_containment.append(inside / 2.0)
#                 termination_reasons[reason] += 1
#                 all_episodes.append(metrics)
                
#                 print(f"\nEpisode {ep+1}/{num_episodes} finished at step {step}")
#                 print(f"  Reward: {total_reward:.1f} | Containment: {inside}/2 | Reason: {reason}")
#                 break
    
#     # Summary
#     print(f"\n{'='*70}")
#     print("EVALUATION SUMMARY")
#     print(f"{'='*70}")
#     print(f"Avg Reward: {np.mean(all_rewards):.1f} ± {np.std(all_rewards):.1f}")
#     print(f"Avg Containment: {np.mean(all_containment):.1%}")
#     print(f"\nTermination Reasons:")
#     for reason, count in termination_reasons.items():
#         print(f"  {reason}: {count} ({count/num_episodes*100:.0f}%)")
#     print(f"{'='*70}")
    
#     # Generate plots
#     if plot and all_episodes:
#         _plot_agent_evaluation(all_episodes, num_episodes, agent_policy_path)
    
#     vec_env.close()
#     return all_rewards, all_containment


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
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
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


def _plot_agent_evaluation(all_episodes, num_episodes, model_path):
    """Generate plots for agent policy evaluation."""
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
    
    containment_arr = np.array([pad_to_length(ep['containment_rate'], max_len) for ep in all_episodes])
    reward_arr = np.array([pad_to_length(ep['cumulative_reward'], max_len) for ep in all_episodes])
    
    # Calculate mean and std
    cont_mean, cont_std = containment_arr.mean(axis=0), containment_arr.std(axis=0)
    rew_mean, rew_std = reward_arr.mean(axis=0), reward_arr.std(axis=0)
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Agent Policy Evaluation (n={num_episodes} episodes)', fontsize=14, fontweight='bold')
    
    # 1. Containment Rate
    ax = axes[0]
    ax.plot(timesteps, cont_mean, 'g-', linewidth=2)
    ax.fill_between(timesteps, cont_mean - cont_std, cont_mean + cont_std, alpha=0.3, color='green')
    ax.axhline(y=1.0, color='darkgreen', linestyle='--', alpha=0.7, label='Target (100%)')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Containment Rate')
    ax.set_title('Agents Inside Patch')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # 2. Cumulative Reward
    ax = axes[1]
    ax.plot(timesteps, rew_mean, 'r-', linewidth=2)
    ax.fill_between(timesteps, rew_mean - rew_std, rew_mean + rew_std, alpha=0.3, color='red')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Reward Over Time')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plots
    output_dir = os.path.dirname(model_path) if os.path.dirname(model_path) else '.'
    png_path = os.path.join(output_dir, 'agent_evaluation_plots.png')
    pdf_path = os.path.join(output_dir, 'agent_evaluation_plots.pdf')
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")
    
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Evaluate Patch or Agent Policy"
    )
    
    parser.add_argument("--policy", type=str, choices=["patch", "agent"], required=True,
                       help="Which policy to evaluate")
    parser.add_argument("--model", type=str, required=True,
                       help="Path to policy model")
    parser.add_argument("--frozen-policy", type=str, default=None,
                       help="Path to frozen policy (for evaluation)")
    parser.add_argument("--episodes", type=int, default=10,
                       help="Number of episodes")
    parser.add_argument("--render", action="store_true",
                       help="Render episodes")
    parser.add_argument("--visualise", action="store_true",
                       help="Visualise episodes")
    parser.add_argument("--no-plot", action="store_true",
                       help="Disable plotting (default: plots are shown)")
    
    args = parser.parse_args()
    
    plot = not args.no_plot
    
    if args.policy == "patch":
        evaluate_patch_policy(
            args.model,
            agent_policy_path=args.frozen_policy,
            num_episodes=args.episodes,
            render=args.render,
            visualise=args.visualise,
            plot=plot
        )
    # elif args.policy == "agent":
    #     evaluate_agent_policy(
    #         args.model,
    #         patch_policy_path=args.frozen_policy,
    #         num_episodes=args.episodes,
    #         render=args.render,
    #         plot=plot
    #     )