#!/usr/bin/env python3
"""Clean evaluation plots for DRL Patch Funnel."""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drl_patch_funnel_v1_2_1 import PatchFunnelEnvV1

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def evaluate_and_plot(model_path, num_episodes=10, output_dir=None):
    """Run model and generate CLEAN plots."""
    
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    
    print("=" * 70)
    print("LOADING MODEL...")
    print("=" * 70)
    
    model = PPO.load(model_path)
    
    env = PatchFunnelEnvV1(
        num_agents=2, 
        render_mode=None,
        domain_randomize=False
    )
    
    vec_env = DummyVecEnv([lambda: env])
    
    vecnorm_path = model_path.replace(".zip", "").replace("best_model", "best_vecnormalize.pkl")
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize: {vecnorm_path}")
        vec_env = VecNormalize.load(vecnorm_path, vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
    
    all_episodes = []
    
    print(f"\nRunning {num_episodes} episodes...")
    
    for ep in range(num_episodes):
        metrics = {
            'timestep': [],
            'patch_velocity': [],
            'patch_x': [],
            'patch_y': [],
            'containment_rate': [],
            'lap_progress': [],
            'cumulative_reward': [],
            'agent_dist': [],  # Average of both agents
        }
        
        obs = vec_env.reset()
        cumulative_reward = 0
        step = 0
        
        inner_env = vec_env.envs[0]
        if hasattr(inner_env, 'env'):
            inner_env = inner_env.env
        
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, info = vec_env.step(action)
            
            cumulative_reward += reward[0]
            step += 1
            
            metrics['timestep'].append(step)
            metrics['patch_velocity'].append(inner_env.patch.v)
            metrics['patch_x'].append(inner_env.patch.x)
            metrics['patch_y'].append(inner_env.patch.y)
            metrics['lap_progress'].append(info[0].get('lap_progress', 0) * 100)  # As percentage
            metrics['cumulative_reward'].append(cumulative_reward)
            metrics['containment_rate'].append(info[0].get('agents_inside', 0) / 2.0)
            
            # Average agent distance
            dists = []
            for state in inner_env.agent_states:
                d = np.sqrt((state[0] - inner_env.patch.x)**2 + (state[1] - inner_env.patch.y)**2)
                dists.append(d)
            metrics['agent_dist'].append(np.mean(dists))
            
            if done[0]:
                reason = info[0].get('termination_reason', 'unknown')
                print(f"  Ep {ep+1}: steps={step}, reward={cumulative_reward:.0f}, "
                      f"progress={metrics['lap_progress'][-1]:.1f}%, reason={reason}")
                break
        
        all_episodes.append(metrics)
    
    vec_env.close()
    
    # ============ CLEAN PLOTS ============
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
    containment_arr = np.array([pad_to_length(ep['containment_rate'], max_len) for ep in all_episodes])
    progress_arr = np.array([pad_to_length(ep['lap_progress'], max_len) for ep in all_episodes])
    reward_arr = np.array([pad_to_length(ep['cumulative_reward'], max_len) for ep in all_episodes])
    agent_dist_arr = np.array([pad_to_length(ep['agent_dist'], max_len) for ep in all_episodes])
    
    # Calculate mean and std
    vel_mean, vel_std = velocity_arr.mean(axis=0), velocity_arr.std(axis=0)
    cont_mean, cont_std = containment_arr.mean(axis=0), containment_arr.std(axis=0)
    prog_mean, prog_std = progress_arr.mean(axis=0), progress_arr.std(axis=0)
    rew_mean, rew_std = reward_arr.mean(axis=0), reward_arr.std(axis=0)
    dist_mean, dist_std = agent_dist_arr.mean(axis=0), agent_dist_arr.std(axis=0)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    fig.suptitle(f'DRL Patch Funnel Evaluation (n={num_episodes} episodes)', fontsize=14, fontweight='bold')
    
    # 1. Patch Velocity
    ax = axes[0, 0]
    ax.plot(timesteps, vel_mean, 'b-', linewidth=2, label='Mean')
    ax.fill_between(timesteps, vel_mean - vel_std, vel_mean + vel_std, alpha=0.3, color='blue')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Velocity (m/s)')
    ax.set_title('Patch Velocity')
    ax.grid(True, alpha=0.3)
    
    # 2. Containment Rate
    ax = axes[0, 1]
    ax.plot(timesteps, cont_mean, 'g-', linewidth=2)
    ax.fill_between(timesteps, cont_mean - cont_std, cont_mean + cont_std, alpha=0.3, color='green')
    ax.axhline(y=1.0, color='darkgreen', linestyle='--', alpha=0.7, label='Target')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Containment Rate')
    ax.set_title('Agents Inside Patch')
    ax.set_ylim(-0.05, 1.1)
    ax.grid(True, alpha=0.3)
    
    # 3. Lap Progress
    ax = axes[0, 2]
    ax.plot(timesteps, prog_mean, 'purple', linewidth=2)
    ax.fill_between(timesteps, prog_mean - prog_std, prog_mean + prog_std, alpha=0.3, color='purple')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Progress (%)')
    ax.set_title('Lap Progress')
    ax.grid(True, alpha=0.3)
    
    # 4. Cumulative Reward
    ax = axes[1, 0]
    ax.plot(timesteps, rew_mean, 'r-', linewidth=2)
    ax.fill_between(timesteps, rew_mean - rew_std, rew_mean + rew_std, alpha=0.3, color='red')
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Cumulative Reward')
    ax.set_title('Reward Over Time')
    ax.grid(True, alpha=0.3)
    
    # 5. Agent Distance from Patch Center
    ax = axes[1, 1]
    ax.plot(timesteps, dist_mean, 'orange', linewidth=2)
    ax.fill_between(timesteps, dist_mean - dist_std, dist_mean + dist_std, alpha=0.3, color='orange')
    ax.set_xlabel('Timestep')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Avg Agent Distance from Patch')
    ax.grid(True, alpha=0.3)
    
    # 6. Patch Trajectory (just show 3 episodes max)
    ax = axes[1, 2]
    colors = ['blue', 'red', 'green']
    for i, ep in enumerate(all_episodes[:3]):  # Only 3 episodes
        ax.plot(ep['patch_x'], ep['patch_y'], color=colors[i % 3], alpha=0.7, 
                linewidth=1.5, label=f'Ep {i+1}')
        ax.scatter(ep['patch_x'][0], ep['patch_y'][0], color=colors[i % 3], marker='o', s=60, zorder=5)
        ax.scatter(ep['patch_x'][-1], ep['patch_y'][-1], color=colors[i % 3], marker='x', s=60, zorder=5)
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title('Patch Trajectory (o=start, x=end)')
    ax.axis('equal')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    png_path = os.path.join(output_dir, 'evaluation_plots.png')
    pdf_path = os.path.join(output_dir, 'evaluation_plots.pdf')
    plt.savefig(png_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.savefig(pdf_path, bbox_inches='tight')
    
    print(f"\n✅ Saved: {png_path}")
    print(f"✅ Saved: {pdf_path}")
    
    plt.show()
    
    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)
    final_progress = [ep['lap_progress'][-1] for ep in all_episodes]
    final_rewards = [ep['cumulative_reward'][-1] for ep in all_episodes]
    avg_vel = [np.mean(ep['patch_velocity']) for ep in all_episodes]
    avg_cont = [np.mean(ep['containment_rate']) for ep in all_episodes]
    
    print(f"Progress:     {np.mean(final_progress):.2f}% ± {np.std(final_progress):.2f}%")
    print(f"Reward:       {np.mean(final_rewards):.1f} ± {np.std(final_rewards):.1f}")
    print(f"Avg Velocity: {np.mean(avg_vel):.2f} ± {np.std(avg_vel):.2f} m/s")
    print(f"Containment:  {np.mean(avg_cont)*100:.1f}% ± {np.std(avg_cont)*100:.1f}%")
    print("=" * 70)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--episodes", type=int, default=10)  # Use 10, not 100!
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    
    evaluate_and_plot(args.model, args.episodes, args.output)
