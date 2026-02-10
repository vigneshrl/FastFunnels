#!/usr/bin/env python3
"""
Training Scripts for Hierarchical Patch-Agent System
====================================================

Supports:
- Training patch policy separately
- Training agent policy separately
- Command-line interface for flexible training
"""

import os
import time
import json
import argparse
from datetime import datetime
import numpy as np
#import f1tenth_gym
#import gymnaisum as gp

# Stable-Baselines3
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.utils import set_random_seed
    SB3_AVAILABLE = True
except ImportError:
    print("ERROR: stable-baselines3 not installed!")
    SB3_AVAILABLE = False

# Import environments
from scripts.NeuroPatch import PatchEnv
from neurocontrollers import AgentEnv


def make_patch_env(rank, seed=0, agent_policy_path=None, domain_randomize=False):
    """Create a patch environment instance for parallel training."""
    def _init():
        import f1tenth_gym
        # Explicitly add this because when running the subprocvecenv, 
        # it spawns seperate python processes and sometimes these child 
        # processes dont inherit the environment imports 
        # dont forget to add this in the agent make_env
        # Load agent policy if provided
        agent_policy = None
        if agent_policy_path and os.path.exists(agent_policy_path + ".zip"):
            agent_policy = PPO.load(agent_policy_path)
        
        env = PatchEnv(
            num_agents=2,
            render_mode=None,
            domain_randomize=domain_randomize,
            agent_policy=agent_policy
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


def make_agent_env(rank, seed=0, patch_policy_path=None, domain_randomize=False):
    """Create an agent environment instance for parallel training."""
    def _init():
        # Load patch policy if provided
        patch_policy = None
        if patch_policy_path and os.path.exists(patch_policy_path + ".zip"):
            patch_policy = PPO.load(patch_policy_path)
        
        env = AgentEnv(
            num_agents=2,
            render_mode=None,
            domain_randomize=domain_randomize,
            patch_policy=patch_policy
        )
        env.reset(seed=seed + rank)
        return env
    set_random_seed(seed + rank)
    return _init


class TrainingCallback(BaseCallback):
    """Callback for logging and checkpointing during training."""
    
    def __init__(self, save_dir, save_freq, num_envs, policy_name, verbose=1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.num_envs = num_envs
        self.policy_name = policy_name
        self.episode_rewards = []
        self.episode_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_timesteps = 0
        self.best_reward = -np.inf
        
        # Training log file
        self.log_file = os.path.join(save_dir, "training_log.csv")
        with open(self.log_file, 'w') as f:
            f.write("timesteps,episodes,avg_reward,time_elapsed,timesteps_per_sec\n")
    
    def _on_step(self):
        # Check for episode completion
        for i, done in enumerate(self.locals.get("dones", [])):
            if done:
                self.episode_count += 1
                info = self.locals.get("infos", [{}])[i]
                reward = info.get("episode_reward", 0)
                self.episode_rewards.append(reward)
                
                # Log every 50 episodes
                if self.episode_count % 50 == 0:
                    current_time = time.time()
                    elapsed = current_time - self.start_time
                    dt = current_time - self.last_log_time
                    timesteps_delta = self.num_timesteps - self.last_log_timesteps
                    
                    #convert NumPy types to Python types for JSON serialization
                    avg_r = float(np.mean(self.episode_rewards[-50:]))
                    timesteps_per_sec = timesteps_delta / dt if dt > 0 else 0
                    
                    print(f"\n{'='*60}")
                    print(f"[{self.policy_name}] Episode {self.episode_count} | Timesteps: {self.num_timesteps}")
                    print(f"  Avg Reward: {avg_r:.1f} | Time: {elapsed/60:.1f} min | Speed: {timesteps_per_sec:.1f} steps/sec")
                    print(f"{'='*60}")
                    
                    # Log to CSV
                    with open(self.log_file, 'a') as f:
                        f.write(f"{self.num_timesteps},{self.episode_count},{avg_r:.2f},"
                               f"{elapsed:.1f},{timesteps_per_sec:.2f}\n")
                    
                    # Save best model
                    if avg_r > self.best_reward:
                        self.best_reward = float(avg_r)
                        self.model.save(os.path.join(self.save_dir, "best_model"))
                        if hasattr(self.training_env, 'save'):
                            self.training_env.save(
                                os.path.join(self.save_dir, "best_vecnormalize.pkl")
                            )
                        print(f"   🏆 New best model! (reward: {avg_r:.1f})")
                    
                    self.last_log_time = current_time
                    self.last_log_timesteps = self.num_timesteps
        
        # Checkpoint saving
        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}")
            self.model.save(path)
            if hasattr(self.training_env, 'save'):
                self.training_env.save(path + "_vecnormalize.pkl")
            
            #convert Numpy types to native Python types for JSON serialization
            avg_reward = np.mean(self.episode_rewards[-50:]) if len(self.episode_rewards) >= 50 else np.mean(self.episode_rewards)
            metadata = {
                "timesteps": int(self.num_timesteps),
                "episodes": int(self.episode_count),
                "best_reward": float(self.best_reward),
                "avg_reward_last_50": float(avg_reward),
                "time_elapsed_min": float((time.time() - self.start_time) / 60),
                "timestamp": datetime.now().isoformat()
            }
            with open(path + "_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n💾 Checkpoint saved: {path}")
        
        return True


def train_patch_policy(
    total_timesteps=500000,
    save_path="patch_policy_models",
    checkpoint_freq=10000,
    num_envs=8,
    agent_policy_path=None,
    resume_from=None,
    domain_randomize=False
):
    """
    Train the patch policy (leader).
    
    Args:
        total_timesteps: Total training timesteps
        save_path: Directory to save models
        checkpoint_freq: Frequency of checkpoint saving
        num_envs: Number of parallel environments
        agent_policy_path: Path to agent policy (frozen during training)
        resume_from: Path to resume training from
        domain_randomize: Whether to use domain randomization
    """
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return None
    
    os.makedirs(save_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_path, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 70)
    print("TRAINING PATCH POLICY (Leader)")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of parallel environments: {num_envs}")
    print(f"Checkpoint frequency: {checkpoint_freq}")
    print(f"Save directory: {run_dir}")
    if agent_policy_path:
        print(f"Using frozen agent policy: {agent_policy_path}")
    else:
        print("Using Random/Static agents")
    print("=" * 70)
    
    # Create parallel environments
    if num_envs > 1:
        env = SubprocVecEnv([
            make_patch_env(i, seed=42, agent_policy_path=agent_policy_path, 
                          domain_randomize=domain_randomize)
            for i in range(num_envs)
        ])
    else:
        env = DummyVecEnv([
            make_patch_env(0, seed=42, agent_policy_path=agent_policy_path,
                          domain_randomize=domain_randomize)
        ])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=5.0,
        gamma=0.99
    )
    
    # Create callback
    callback = TrainingCallback(run_dir, checkpoint_freq, num_envs, "PATCH")
    
    # Create/load model
    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"\n📂 Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env)
        vecnorm_path = resume_from.replace(".zip", "") + "_vecnormalize.pkl"
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env.venv)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=5e-5,
            n_steps=1024,
            batch_size=256,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2, 
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(save_path, "tensorboard"),
            use_sde=True,
            policy_kwargs={
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "squash_output": False,
                "log_std_init": -1.0
            }
        )
    
    print(f"\n🚀 Starting training...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save final model
        model.save(os.path.join(run_dir, "final_model"))
        env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"   Model saved to: {run_dir}/final_model")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        model.save(os.path.join(run_dir, "interrupted_model"))
        env.save(os.path.join(run_dir, "interrupted_vecnormalize.pkl"))
    
    env.close()
    return model


def train_agent_policy(
    total_timesteps=500000,
    save_path="agent_policy_models",
    checkpoint_freq=10000,
    num_envs=8,
    patch_policy_path=None,
    resume_from=None,
    domain_randomize=False
):
    """
    Train the agent policy (follower).
    
    Args:
        total_timesteps: Total training timesteps
        save_path: Directory to save models
        checkpoint_freq: Frequency of checkpoint saving
        num_envs: Number of parallel environments
        patch_policy_path: Path to patch policy (frozen during training)
        resume_from: Path to resume training from
        domain_randomize: Whether to use domain randomization
    """
    if not SB3_AVAILABLE:
        print("ERROR: stable-baselines3 required!")
        return None
    
    os.makedirs(save_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_path, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)
    
    print("=" * 70)
    print("TRAINING AGENT POLICY (Follower)")
    print("=" * 70)
    print(f"Total timesteps: {total_timesteps}")
    print(f"Number of parallel environments: {num_envs}")
    print(f"Checkpoint frequency: {checkpoint_freq}")
    print(f"Save directory: {run_dir}")
    if patch_policy_path:
        print(f"Using frozen patch policy: {patch_policy_path}")
    else:
        print("Using Trained patch")
    print("=" * 70)
    
    # Create parallel environments
    if num_envs > 1:
        env = SubprocVecEnv([
            make_agent_env(i, seed=42, patch_policy_path=patch_policy_path,
                          domain_randomize=domain_randomize)
            for i in range(num_envs)
        ])
    else:
        env = DummyVecEnv([
            make_agent_env(0, seed=42, patch_policy_path=patch_policy_path,
                          domain_randomize=domain_randomize)
        ])
    
    # Normalize observations and rewards
    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=5.0,
        gamma=0.99
    )
    
    # Create callback
    callback = TrainingCallback(run_dir, checkpoint_freq, num_envs, "AGENT")
    
    # Create/load model
    if resume_from and os.path.exists(resume_from + ".zip"):
        print(f"\n📂 Resuming from: {resume_from}")
        model = PPO.load(resume_from, env=env)
        vecnorm_path = resume_from.replace(".zip", "") + "_vecnormalize.pkl"
        if os.path.exists(vecnorm_path):
            env = VecNormalize.load(vecnorm_path, env.venv)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=256,
            batch_size=256,
            n_epochs=5,
            gamma=0.98,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=os.path.join(save_path, "tensorboard"),
            use_sde=True,
            policy_kwargs={
                "net_arch": dict(pi=[256, 256], vf=[256, 256]),
                "squash_output": True,
                "log_std_init": -2.0
            }
        )
    
    print(f"\n🚀 Starting training...")
    
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=callback,
            progress_bar=True
        )
        
        # Save final model
        model.save(os.path.join(run_dir, "final_model"))
        env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
        
        print(f"\n{'='*70}")
        print(f"✅ TRAINING COMPLETE!")
        print(f"   Model saved to: {run_dir}/final_model")
        print(f"{'='*70}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted")
        model.save(os.path.join(run_dir, "interrupted_model"))
        env.save(os.path.join(run_dir, "interrupted_vecnormalize.pkl"))
    
    env.close()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Patch or Agent Policy",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--policy", type=str, choices=["patch", "agent"], required=True,
                       help="Which policy to train: 'patch' or 'agent'")
    parser.add_argument("--timesteps", type=int, default=500000,
                       help="Total training timesteps")
    parser.add_argument("--checkpoint-freq", type=int, default=10000,
                       help="Checkpoint frequency")
    parser.add_argument("--num-envs", type=int, default=8,
                       help="Number of parallel environments")
    parser.add_argument("--save-path", type=str, default=None,
                       help="Save directory (default: {policy}_policy_models)")
    parser.add_argument("--frozen-policy", type=str, default=None,
                       help="Path to frozen policy (agent policy for patch training, patch policy for agent training)")
    parser.add_argument("--resume", type=str, default=None,
                       help="Path to resume training from")
    parser.add_argument("--domain-randomize", action="store_true",
                       help="Enable domain randomization")
    
    args = parser.parse_args()
    
    # Set default save path
    if args.save_path is None:
        args.save_path = f"{args.policy}_policy_models"
    
    if args.policy == "patch_sempc":
        train_patch_policy(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            checkpoint_freq=args.checkpoint_freq,
            num_envs=args.num_envs,
            agent_policy_path=args.frozen_policy,
            resume_from=args.resume,
            domain_randomize=args.domain_randomize
        )
    # elif args.policy == "agent":
    #     train_agent_policy(
    #         total_timesteps=args.timesteps,
    #         save_path=args.save_path,
    #         checkpoint_freq=args.checkpoint_freq,
    #         num_envs=args.num_envs,
    #         patch_policy_path=args.frozen_policy,
    #         resume_from=args.resume,
    #         domain_randomize=args.domain_randomize
    #     )
