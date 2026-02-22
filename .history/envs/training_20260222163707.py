import argparse
import json
import os
import time
from collections import Counter
from datetime import datetime
from typing import Optional

import numpy as np
import torch 
import torch.nn as nn 

try:
    from .ppo_policy import make_patch_env
except ImportError:
    from envs.ppo_policy import make_patch_env

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

try:
    import wandb
    from wandb.integration.sb3 import WandbCallback

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
        self.episode_lengths = []
        self.termination_reasons = {}
        self.recent_reasons = []
        self.recent_min_dists = []
        self.recent_last_ds = []
        self.recent_last_ey = []
        self.recent_mpc_feas = []
        self.recent_safety_rates = []
        self.recent_reward_clipped = []
        self.recent_agent_step_disp = []
        self.recent_agent_move_ratio = []
        self.recent_mean_speed_cmd = []
        self.recent_mean_abs_steer_cmd = []
        self.recent_control_order = []
        self.log_file = os.path.join(save_dir, "training_log.csv")
        with open(self.log_file, "w", encoding="utf-8") as f:
            # old csv header kept for reference:
            # f.write("timesteps,episodes,avg_reward,avg_len,time_elapsed,timesteps_per_sec,top_reason,avg_min_dist,avg_last_ds,avg_last_ey,avg_mpc_feas,avg_safety_rate,clip_frac\n")
            f.write(
                "timesteps,episodes,avg_reward,avg_len,time_elapsed,timesteps_per_sec,"
                "top_reason,avg_min_dist,avg_last_ds,avg_last_ey,avg_mpc_feas,avg_safety_rate,clip_frac,"
                "avg_agent_step_disp,avg_agent_move_ratio,avg_cmd_speed,avg_cmd_abs_steer,control_order_sample\n"
            )

    def _on_step(self):
        for i, done in enumerate(self.locals.get("dones", [])):
            if not done:
                continue
            self.episode_count += 1
            info = self.locals.get("infos", [{}])[i]
            reward = float(info.get("episode_reward", 0.0))
            ep_len = int(info.get("Episode_steps", 0))
            reason = str(info.get("termination_reason", "unknown"))
            self.episode_rewards.append(reward)
            self.episode_lengths.append(ep_len)
            self.termination_reasons[reason] = self.termination_reasons.get(reason, 0) + 1
            self.recent_reasons.append(reason)
            self.recent_min_dists.append(float(info.get("min_lidar_dist", np.nan)))
            self.recent_last_ds.append(float(info.get("ds", np.nan)))
            self.recent_last_ey.append(float(info.get("ey", np.nan)))
            self.recent_mpc_feas.append(float(info.get("mpc_feasibility_rate", np.nan)))
            self.recent_safety_rates.append(float(info.get("safety_intervention_rate", np.nan)))
            self.recent_reward_clipped.append(bool(info.get("reward_was_clipped", False)))
            self.recent_agent_step_disp.append(float(info.get("episode_avg_agent_step_disp", np.nan)))
            self.recent_agent_move_ratio.append(float(info.get("episode_agent_move_step_ratio", np.nan)))
            self.recent_mean_speed_cmd.append(float(info.get("mean_speed_cmd", np.nan)))
            self.recent_mean_abs_steer_cmd.append(float(info.get("mean_abs_steer_cmd", np.nan)))
            self.recent_control_order.append(str(info.get("control_input_order", "unknown")))

            if self.episode_count % 50 == 0 and self.episode_rewards:
                now = time.time()
                elapsed = now - self.start_time
                dt = now - self.last_log_time
                ts_delta = self.num_timesteps - self.last_log_timesteps
                tps = ts_delta / dt if dt > 0 else 0.0
                avg_r = float(np.mean(self.episode_rewards[-50:]))
                avg_len = float(np.mean(self.episode_lengths[-50:])) if self.episode_lengths else 0.0
                top_reason = max(self.termination_reasons.items(), key=lambda kv: kv[1])[0] if self.termination_reasons else "unknown"
                recent_reason_counts = Counter(self.recent_reasons[-50:])
                reason_breakdown = ", ".join(f"{k}:{v}" for k, v in recent_reason_counts.most_common(4))
                avg_min_dist = float(np.nanmean(self.recent_min_dists[-50:])) if self.recent_min_dists else float("nan")
                avg_last_ds = float(np.nanmean(self.recent_last_ds[-50:])) if self.recent_last_ds else float("nan")
                avg_last_ey = float(np.nanmean(self.recent_last_ey[-50:])) if self.recent_last_ey else float("nan")
                avg_mpc_feas = float(np.nanmean(self.recent_mpc_feas[-50:])) if self.recent_mpc_feas else float("nan")
                avg_safety = float(np.nanmean(self.recent_safety_rates[-50:])) if self.recent_safety_rates else float("nan")
                clip_frac = float(np.mean(self.recent_reward_clipped[-50:])) if self.recent_reward_clipped else 0.0
                avg_agent_step_disp = float(np.nanmean(self.recent_agent_step_disp[-50:])) if self.recent_agent_step_disp else float("nan")
                avg_agent_move_ratio = float(np.nanmean(self.recent_agent_move_ratio[-50:])) if self.recent_agent_move_ratio else float("nan")
                avg_cmd_speed = float(np.nanmean(self.recent_mean_speed_cmd[-50:])) if self.recent_mean_speed_cmd else float("nan")
                avg_cmd_abs_steer = float(np.nanmean(self.recent_mean_abs_steer_cmd[-50:])) if self.recent_mean_abs_steer_cmd else float("nan")
                control_order_sample = self.recent_control_order[-1] if self.recent_control_order else "unknown"
                print(f"\n[{self.policy_name}] Episode {self.episode_count} | Timesteps: {self.num_timesteps}")
                print(
                    f"  Avg Reward: {avg_r:.2f} | Avg Len: {avg_len:.1f} | "
                    f"Top End: {top_reason} | Time: {elapsed/60:.1f} min | Speed: {tps:.1f} steps/sec"
                )
                print(
                    f"  Debug(last50): reasons=[{reason_breakdown}] "
                    f"min_dist={avg_min_dist:.3f} ds={avg_last_ds:.4f} ey={avg_last_ey:.3f} "
                    f"mpc_feas={avg_mpc_feas:.3f} safety_rate={avg_safety:.3f} clip_frac={clip_frac:.2f}"
                )
                print(
                    f"  AgentMotion(last50): avg_step_disp={avg_agent_step_disp:.4f}m "
                    f"move_step_ratio={avg_agent_move_ratio:.3f}"
                )
                print(
                    f"  Cmd(last50): mean_speed={avg_cmd_speed:.3f} "
                    f"mean_abs_steer={avg_cmd_abs_steer:.3f} "
                    f"control_order={control_order_sample}"
                )
                if self.use_wandb:
                    # old:
                    # wandb.log({...}, step=self.num_timesteps)
                    # With sync_tensorboard=True, explicit step causes a warning.
                    wandb.log(
                        {
                            "train/avg_reward_50": avg_r,
                            "train/avg_len_50": avg_len,
                            "train/avg_min_dist_50": avg_min_dist,
                            "train/avg_last_ds_50": avg_last_ds,
                            "train/avg_last_ey_50": avg_last_ey,
                            "train/avg_mpc_feas_50": avg_mpc_feas,
                            "train/avg_safety_rate_50": avg_safety,
                            "train/reward_clip_frac_50": clip_frac,
                            "train/avg_agent_step_disp_50": avg_agent_step_disp,
                            "train/avg_agent_move_ratio_50": avg_agent_move_ratio,
                            "train/avg_cmd_speed_50": avg_cmd_speed,
                            "train/avg_cmd_abs_steer_50": avg_cmd_abs_steer,
                            "train/episodes": self.episode_count,
                            "train/timesteps": self.num_timesteps,
                            "train/timesteps_per_sec": tps,
                        }
                    )
                if avg_len < 5.0:
                    print("  WARNING: Very short episodes. Focus on termination reasons and min_dist.")
                with open(self.log_file, "a", encoding="utf-8") as f:
                    # old csv row kept for reference:
                    # f.write(f"{self.num_timesteps},{self.episode_count},{avg_r:.2f},{avg_len:.2f},{elapsed:.1f},{tps:.2f},{top_reason},{avg_min_dist:.4f},{avg_last_ds:.6f},{avg_last_ey:.4f},{avg_mpc_feas:.4f},{avg_safety:.4f},{clip_frac:.4f}\n")
                    f.write(
                        f"{self.num_timesteps},{self.episode_count},{avg_r:.2f},"
                        f"{avg_len:.2f},{elapsed:.1f},{tps:.2f},{top_reason},"
                        f"{avg_min_dist:.4f},{avg_last_ds:.6f},{avg_last_ey:.4f},"
                        f"{avg_mpc_feas:.4f},{avg_safety:.4f},{clip_frac:.4f},"
                        f"{avg_agent_step_disp:.6f},{avg_agent_move_ratio:.4f},"
                        f"{avg_cmd_speed:.4f},{avg_cmd_abs_steer:.4f},\"{control_order_sample}\"\n"
                    )

                if avg_r > self.best_reward:
                    self.best_reward = avg_r
                    self.model.save(os.path.join(self.save_dir, "best_model"))
                    if hasattr(self.training_env, "save"):
                        self.training_env.save(os.path.join(self.save_dir, "best_vecnormalize.pkl"))
                    print(f"  New best model: {avg_r:.2f}")
                    if self.use_wandb:
                        # old: wandb.log(..., step=self.num_timesteps)
                        wandb.log(
                            {
                                "train/best_avg_reward_50": self.best_reward,
                            }
                        )
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
    checkpoint_freq: int = 2000,
    NUM_ENVS: int = 4,
    resume_from: Optional[str] = None,
    domain_randomize: bool = False,
    norm_reward: bool = True,
    base_reset_type: str = "rl_random_static",
    use_base_done_termination: bool = False,
    patch_only_mode: bool = False,
    debug_print_every_n_steps: int = 0,
    debug_print_episode_end: bool = True,
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
    tb_log_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)

    wandb_run = None
    if WANDB_AVAILABLE:
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"patch_envs_{run_id}",
            config={
                "total_timesteps": total_timesteps,
                "num_envs": NUM_ENVS,
                "checkpoint_freq": checkpoint_freq,
                "domain_randomize": domain_randomize,
                "patch_only_mode": patch_only_mode,
                "run_dir": run_dir,
            },
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )

    if NUM_ENVS > 1:
        env = SubprocVecEnv(
            [
                make_patch_env(
                    i,
                    seed=42,
                    domain_randomize=domain_randomize,
                    navigation_mode="centerline",
                    debug_print_every_n_steps=debug_print_every_n_steps,
                    debug_print_episode_end=debug_print_episode_end,
                    base_reset_type=base_reset_type,
                    use_base_done_termination=use_base_done_termination,
                    patch_only_mode=patch_only_mode,
                )
                for i in range(NUM_ENVS)
            ]
        )
    else:
        env = DummyVecEnv(
            [
                make_patch_env(
                    0,
                    seed=42,
                    domain_randomize=domain_randomize,
                    navigation_mode="centerline",
                    debug_print_every_n_steps=debug_print_every_n_steps,
                    debug_print_episode_end=debug_print_episode_end,
                    base_reset_type=base_reset_type,
                    use_base_done_termination=use_base_done_termination,
                    patch_only_mode=patch_only_mode,
                )
            ]
        )

    env = VecCheckNan(env, raise_exception=True)
    env = VecNormalize(
        env,
        norm_obs=True,
        norm_reward=norm_reward,
        clip_obs=10.0,
        clip_reward=5.0,
        gamma=0.99,
    )

    callback = TrainingCallback(
        run_dir, checkpoint_freq, "PATCH", use_wandb=WANDB_AVAILABLE
    )

    rollout_steps = 2048
    total_rollout = rollout_steps * max(1, NUM_ENVS)
    batch_size = NUM_ENVS *  rollout_steps //4 #max(64, total_rollout // 4)
    # if total_rollout % batch_size != 0:
    #     # Keep PPO minibatches exact to avoid partial batches.
    #     batch_size = 256 if total_rollout % 256 == 0 else 128

    if resume_from and os.path.exists(resume_from + ".zip"):
        model = PPO.load(resume_from, env=env)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=tb_log_dir,
            learning_rate=3e-4,
            seed = 42,
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            use_sde=False,
            policy_kwargs={"net_arch": dict(pi=[256, 256], vf=[256, 256]), "squash_output": False,
            "activation_fn": nn.ReLU,
            }
        )

    callbacks = [callback]
    if WANDB_AVAILABLE:
        callbacks.append(
            WandbCallback(
                gradient_save_freq=0,
                model_save_path=run_dir,
                model_save_freq=checkpoint_freq,
                verbose=0,
            )
        )

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(run_dir, "final_model"))
    env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
    env.close()
    if WANDB_AVAILABLE and wandb_run is not None:
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
    parser.add_argument("--no-norm-reward", action="store_true")
    parser.add_argument("--base-reset-type", type=str, default="rl_random_static")
    parser.add_argument("--use-base-done-termination", action="store_true")
    parser.add_argument("--patch-only-mode", action="store_true")
    parser.add_argument("--debug-step-print-every", type=int, default=0)
    parser.add_argument("--no-debug-episode-end", action="store_true")
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
        norm_reward=not args.no_norm_reward,
        base_reset_type=args.base_reset_type,
        use_base_done_termination=args.use_base_done_termination,
        patch_only_mode=args.patch_only_mode,
        debug_print_every_n_steps=args.debug_step_print_every,
        debug_print_episode_end=not args.no_debug_episode_end,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
    )