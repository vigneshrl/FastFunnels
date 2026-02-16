import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

import numpy as np

try:
    from .ppo_policy import make_patch_env
except ImportError:
    from envs.ppo_policy import make_patch_env

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class TrainingCallback(BaseCallback):
    def __init__(self, save_dir: str, save_freq: int, policy_name: str, use_wandb: bool = True, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.policy_name = policy_name
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.episode_rewards = []
        self.episode_count = 0
        self.start_time = time.time()
        self.last_log_time = time.time()
        self.last_log_timesteps = 0
        self.best_reward = -np.inf
        self.log_file = os.path.join(save_dir, "training_log.csv")
        with open(self.log_file, "w", encoding="utf-8") as f:
            f.write("timesteps,episodes,avg_reward,time_elapsed,timesteps_per_sec\n")

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue
            self.episode_count += 1
            info = self.locals.get("infos", [{}])[i]
            reward = float(info.get("episode_reward", 0.0))
            self.episode_rewards.append(reward)

            if self.episode_count % 50 == 0 and self.episode_rewards:
                now = time.time()
                elapsed = now - self.start_time
                dt = now - self.last_log_time
                ts_delta = self.num_timesteps - self.last_log_timesteps
                tps = ts_delta / dt if dt > 0 else 0.0
                avg_r = float(np.mean(self.episode_rewards[-50:]))
                print(f"\n[{self.policy_name}] Episode {self.episode_count} | Timesteps: {self.num_timesteps}")
                print(f"  Avg Reward: {avg_r:.2f} | Time: {elapsed/60:.1f} min | Speed: {tps:.1f} steps/sec")
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(f"{self.num_timesteps},{self.episode_count},{avg_r:.2f},{elapsed:.1f},{tps:.2f}\n")

                if avg_r > self.best_reward:
                    self.best_reward = avg_r
                    self.model.save(os.path.join(self.save_dir, "best_model"))
                    if hasattr(self.training_env, "save"):
                        self.training_env.save(os.path.join(self.save_dir, "best_vecnormalize.pkl"))
                    print(f"  New best model: {avg_r:.2f}")
                self.last_log_time = now
                self.last_log_timesteps = self.num_timesteps

        if self.num_timesteps % self.save_freq == 0:
            path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}")
            self.model.save(path)
            if hasattr(self.training_env, "save"):
                self.training_env.save(path + "_vecnormalize.pkl")
            with open(path + "_metadata.json", "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "timesteps": int(self.num_timesteps),
                        "episodes": int(self.episode_count),
                        "best_reward": float(self.best_reward),
                        "timestamp": datetime.now().isoformat(),
                    },
                    f,
                    indent=2,
                )
            print(f"Checkpoint saved: {path}")
        return True


def train_patch_policy(
    total_timesteps: int = 500000,
    save_path: str = "patch_policy_models",
    checkpoint_freq: int = 10000,
    num_envs: int = 8,
    resume_from: Optional[str] = None,
    domain_randomize: bool = False,
    wandb_project: str = "patch_sempc_training",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
):
    if not SB3_AVAILABLE:
        raise RuntimeError("stable-baselines3 is required for training.")

    os.makedirs(save_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_path, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    if WANDB_AVAILABLE:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"patch_envs_{run_id}",
            config={
                "total_timesteps": total_timesteps,
                "num_envs": num_envs,
                "checkpoint_freq": checkpoint_freq,
                "domain_randomize": domain_randomize,
                "run_dir": run_dir,
            },
            sync_tensorboard=False,
            monitor_gym=False,
        )

    if num_envs > 1:
        env = SubprocVecEnv([make_patch_env(i, seed=42, domain_randomize=domain_randomize) for i in range(num_envs)])
    else:
        env = DummyVecEnv([make_patch_env(0, seed=42, domain_randomize=domain_randomize)])

    env = VecNormalize(
        env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=5.0,
        gamma=0.99,
    )

    callback = TrainingCallback(
        run_dir, checkpoint_freq, "PATCH", use_wandb=WANDB_AVAILABLE
    )

    if resume_from and os.path.exists(resume_from + ".zip"):
        model = PPO.load(resume_from, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=512,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            use_sde=True,
            policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256]), "squash_output": False},
        )

    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    model.save(os.path.join(run_dir, "final_model"))
    env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
    env.close()
    if WANDB_AVAILABLE:
        wandb.finish()
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train patch policy with modular envs stack.")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--save-path", type=str, default="patch_policy_models")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="patch_sempc_training")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()

    train_patch_policy(
        total_timesteps=args.timesteps,
        save_path=args.save_path,
        checkpoint_freq=args.checkpoint_freq,
        num_envs=args.num_envs,
        resume_from=args.resume,
        domain_randomize=args.domain_randomize,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )