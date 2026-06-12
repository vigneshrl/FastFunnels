import argparse
import dataclasses
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
    from .ppo_policy import (
        make_patch_env, PatchEnv, PatchEnvConfig, JointEnvConfig,
    )
except ImportError:
    from envs.ppo_policy import (
        make_patch_env, PatchEnv, PatchEnvConfig, JointEnvConfig,
    )

try:
    from .cnn_extractor import HybridCNNExtractor
except ImportError:
    from envs.cnn_extractor import HybridCNNExtractor

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False
    BaseCallback = object

try:
    import supersuit as ss
    SUPERSUIT_AVAILABLE = True
except ImportError:
    SUPERSUIT_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

######################################################
# Save best model + VecNormalize stats
######################################################
class SaveBestWithVecNormalize(BaseCallback):
    """
    Periodically checks recent episode rewards and saves:
      - the best PPO model checkpoint
      - VecNormalize statistics (critical for consistent eval)
    """

    def __init__(self, save_dir: str, check_freq: int = 2048, verbose: int = 1,
                 model_name: str = "best_model"):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.check_freq = int(check_freq)
        self.best_model_path = os.path.join(save_dir, model_name)
        self.best_vecnorm_path = os.path.join(save_dir, model_name + "_vecnorm.pkl")
        self.best_mean_ep_len = -1e18

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        if len(self.model.ep_info_buffer) == 0:
            return True

        mean_ep_len = float(np.mean([ep["l"] for ep in self.model.ep_info_buffer]))
        mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
        print(f"Num timesteps: {self.num_timesteps} | Mean ep_len: {mean_ep_len:.0f} | Mean reward: {mean_reward:.2f}")

        # Track by episode LENGTH not reward — longer episodes = better navigation.
        # Under penalty-heavy rewards, short episodes can have higher reward than long ones.
        if mean_ep_len > self.best_mean_ep_len:
            self.best_mean_ep_len = mean_ep_len
            print(f"Saving new best (ep_len={mean_ep_len:.0f})")
            self.model.save(self.best_model_path)

            if isinstance(self.training_env, VecNormalize):
                self.training_env.save(self.best_vecnorm_path)

        return True

class TrainingCallback(BaseCallback):
    def __init__(self, save_dir: str, save_freq: int, policy_name: str, verbose: int = 1):
        super().__init__(verbose)
        self.save_dir = save_dir
        self.save_freq = save_freq
        self.policy_name = policy_name
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
        self.recent_lap_progress = []
        self.recent_terminal_frenet_s = []
        self.log_file = os.path.join(save_dir, "training_log.csv")
        with open(self.log_file, "w", encoding="utf-8") as f:
            # old csv header kept for reference:
            # f.write("timesteps,episodes,avg_reward,avg_len,time_elapsed,timesteps_per_sec,top_reason,avg_min_dist,avg_last_ds,avg_last_ey,avg_mpc_feas,avg_safety_rate,clip_frac\n")
            f.write(
                "timesteps,episodes,avg_reward,avg_len,time_elapsed,timesteps_per_sec,"
                "top_reason,avg_min_dist,avg_last_ds,avg_last_ey,avg_mpc_feas,avg_safety_rate,clip_frac,"
                "avg_agent_step_disp,avg_agent_move_ratio,avg_cmd_speed,avg_cmd_abs_steer,control_order_sample\n"
            )

    _LOG_WINDOW = 200

    def _on_step(self):
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [{}])
        for i, done in enumerate(dones):
            if not done:
                continue
            self.episode_count += 1
            info = infos[i]
            reward = float(info.get("episode_reward", 0.0))
            ep_len = int(info.get("Episode_steps", 0))
            reason = str(info.get("termination_reason", "unknown"))
            self.episode_rewards.append(reward)
            self.episode_lengths.append(ep_len)
            if len(self.episode_rewards) > self._LOG_WINDOW * 2:
                self.episode_rewards = self.episode_rewards[-self._LOG_WINDOW:]
                self.episode_lengths = self.episode_lengths[-self._LOG_WINDOW:]
            self.termination_reasons[reason] = self.termination_reasons.get(reason, 0) + 1
            self.recent_reasons.append(reason)
            self.recent_min_dists.append(float(info.get("min_lidar_dist", np.nan)))
            self.recent_last_ds.append(float(info.get("ds", np.nan)))
            self.recent_last_ey.append(float(info.get("ey", np.nan)))
            # self.recent_mpc_feas.append(float(info.get("mpc_feasibility_rate", np.nan)))
            self.recent_safety_rates.append(float(info.get("safety_intervention_rate", np.nan)))
            self.recent_reward_clipped.append(bool(info.get("reward_was_clipped", False)))
            self.recent_agent_step_disp.append(float(info.get("episode_avg_agent_step_disp", np.nan)))
            self.recent_agent_move_ratio.append(float(info.get("episode_agent_move_step_ratio", np.nan)))
            self.recent_mean_speed_cmd.append(float(info.get("mean_speed_cmd", np.nan)))
            self.recent_mean_abs_steer_cmd.append(float(info.get("mean_abs_steer_cmd", np.nan)))
            self.recent_control_order.append(str(info.get("control_input_order", "unknown")))
            self.recent_lap_progress.append(float(info.get("lap_progress", np.nan)))
            self.recent_terminal_frenet_s.append(float(info.get("frenet_s", np.nan)))

            if len(self.recent_reasons) > self._LOG_WINDOW * 2:
                w = self._LOG_WINDOW
                self.recent_reasons = self.recent_reasons[-w:]
                self.recent_min_dists = self.recent_min_dists[-w:]
                self.recent_last_ds = self.recent_last_ds[-w:]
                self.recent_last_ey = self.recent_last_ey[-w:]
                self.recent_mpc_feas = self.recent_mpc_feas[-w:]
                self.recent_safety_rates = self.recent_safety_rates[-w:]
                self.recent_reward_clipped = self.recent_reward_clipped[-w:]
                self.recent_agent_step_disp = self.recent_agent_step_disp[-w:]
                self.recent_agent_move_ratio = self.recent_agent_move_ratio[-w:]
                self.recent_mean_speed_cmd = self.recent_mean_speed_cmd[-w:]
                self.recent_mean_abs_steer_cmd = self.recent_mean_abs_steer_cmd[-w:]
                self.recent_control_order = self.recent_control_order[-w:]
                self.recent_lap_progress = self.recent_lap_progress[-w:]
                self.recent_terminal_frenet_s = self.recent_terminal_frenet_s[-w:]

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
                avg_lap_progress = float(np.nanmean(self.recent_lap_progress[-50:])) if self.recent_lap_progress else float("nan")
                avg_terminal_s = float(np.nanmean(self.recent_terminal_frenet_s[-50:])) if self.recent_terminal_frenet_s else float("nan")
                control_order_sample = self.recent_control_order[-1] if self.recent_control_order else "unknown"
                print(f"\n[{self.policy_name}] Episode {self.episode_count} | Timesteps: {self.num_timesteps}")
                print(
                    f"  Avg Reward: {avg_r:.2f} | Avg Len: {avg_len:.1f} | "
                    f"Top End: {top_reason} | Time: {elapsed/60:.1f} min | Speed: {tps:.1f} steps/sec"
                )
                print(f"  Reasons(last50): [{reason_breakdown}]")
                # Fields below only exist in old single-agent env; skip if all NaN
                if not (np.isnan(avg_min_dist) and np.isnan(avg_last_ds)):
                    print(
                        f"  Debug(last50): min_dist={avg_min_dist:.3f} "
                        f"ds={avg_last_ds:.4f} ey={avg_last_ey:.3f}"
                    )
                if not np.isnan(avg_agent_step_disp):
                    print(
                        f"  AgentMotion(last50): avg_step_disp={avg_agent_step_disp:.4f}m "
                        f"move_step_ratio={avg_agent_move_ratio:.3f}"
                    )
                if not np.isnan(avg_cmd_speed):
                    print(
                        f"  Cmd(last50): mean_speed={avg_cmd_speed:.3f} "
                        f"mean_abs_steer={avg_cmd_abs_steer:.3f} "
                        f"control_order={control_order_sample}"
                    )
                if avg_len < 5.0:
                    print("  WARNING: Very short episodes. Focus on termination reasons.")
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

                # Push custom metrics to WandB (in addition to the SB3 TB sync)
                if WANDB_AVAILABLE:
                    try:
                        import wandb as _wandb
                        if _wandb.run is not None:
                            log = {
                                f"{self.policy_name}/avg_reward": avg_r,
                                f"{self.policy_name}/avg_ep_len": avg_len,
                                f"{self.policy_name}/steps_per_sec": tps,
                            }
                            for reason, cnt in recent_reason_counts.items():
                                log[f"{self.policy_name}/reason_{reason}"] = cnt / max(len(self.recent_reasons[-50:]), 1)
                            if not np.isnan(avg_cmd_speed):
                                log[f"{self.policy_name}/cmd_speed"] = avg_cmd_speed
                                log[f"{self.policy_name}/cmd_abs_steer"] = avg_cmd_abs_steer
                            if not np.isnan(avg_lap_progress):
                                log[f"{self.policy_name}/lap_progress"] = avg_lap_progress
                            if not np.isnan(avg_terminal_s):
                                log[f"{self.policy_name}/terminal_frenet_s"] = avg_terminal_s
                            _wandb.log(log, step=self.num_timesteps)
                    except Exception:
                        pass

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


def _linear_lr_schedule(initial_lr: float, final_lr: float):
    """Linear decay from initial_lr (step 0) to final_lr (step total_timesteps)."""
    def schedule(progress_remaining: float) -> float:
        return final_lr + (initial_lr - final_lr) * progress_remaining
    return schedule


def train_patch_policy(
    total_timesteps: int = 500000,
    save_path: str = "patch_policy_models",
    checkpoint_freq: int = 2000,
    NUM_ENVS: int = 8,
    resume_from: Optional[str] = None,
    domain_randomize: bool = False,
    norm_reward: bool = False,
    base_reset_type: str = "rl_random_static",
    use_base_done_termination: bool = False,
    patch_only_mode: bool = True,
    debug_print_every_n_steps: int = 0,
    debug_print_episode_end: bool = True,
    wandb_project: str = "patch_sempc_training",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    obs_mode: str = "lidar", #grid
    num_lidar_beams: int = 108,
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
        _cfg_dict = dataclasses.asdict(PatchEnvConfig())
        _tr = 65536  # TARGET_ROLLOUT — must match the value set below at line ~390
        wandb_run = wandb.init(
            project=wandb_project,
            config={
                "mode": "patch_solo",
                "num_envs": NUM_ENVS,
                "rollout_steps": max(512, _tr // max(1, NUM_ENVS)),
                "batch_size": max(256, _tr // 8),
                "reward_steer_bias_weight": _cfg_dict["reward_steer_bias_weight"],
                "reward_spin_weight": _cfg_dict["reward_spin_weight"],
                "spin_yawrate_threshold": _cfg_dict["spin_yawrate_threshold"],
                # "reward_speed_weight": _cfg_dict["reward_speed_weight"],
                "reward_size_weight": _cfg_dict["reward_size_weight"],
                "collision_penalty": _cfg_dict["collision_penalty"],
                "stuck_no_progress_steps": _cfg_dict["stuck_no_progress_steps"],
                "stuck_penalty": _cfg_dict["stuck_penalty"],
                "reward_progress_scale": _cfg_dict["reward_progress_scale"],
                "reward_crosstrack_weight": _cfg_dict["reward_crosstrack_weight"],
                "time_penalty_per_sec": _cfg_dict["time_penalty_per_sec"],
                "max_steps": _cfg_dict["max_steps"],
                "obs_dim": 86,
                "lr_initial": 3e-4,
                "lr_final": 1e-5,
                "lr_schedule": "linear",
                "patch_env_full": _cfg_dict,
            },
            sync_tensorboard=True,
            save_code=True,
        )

    def _monitored(thunk):
        return lambda: Monitor(thunk())

    if NUM_ENVS > 1:
        env = SubprocVecEnv(
            [
                _monitored(make_patch_env(
                    i,
                    seed=42,
                    domain_randomize=domain_randomize,
                    base_reset_type=base_reset_type,
                    obs_mode=obs_mode,
                    num_lidar_beams=num_lidar_beams,
                ))
                for i in range(NUM_ENVS)
            ],
            start_method="fork",  # avoids PyCapsule datetime error from spawn/forkserver on Linux
        )
    else:
        env = DummyVecEnv(
            [
                _monitored(make_patch_env(
                    0,
                    seed=42,
                    domain_randomize=domain_randomize,
                    base_reset_type=base_reset_type,
                    obs_mode=obs_mode,
                    num_lidar_beams=num_lidar_beams,
                ))
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

    callback = TrainingCallback(run_dir, checkpoint_freq, "PATCH")

    _TARGET_ROLLOUT = 65536
    rollout_steps = max(512, _TARGET_ROLLOUT // max(1, NUM_ENVS))  # keeps total GPU work fixed
    batch_size = max(256, _TARGET_ROLLOUT // 8)                    # constant regardless of num_envs
    # if total_rollout % batch_size != 0:
    #     # Keep PPO minibatches exact to avoid partial batches.
    #     batch_size = 256 if total_rollout % 256 == 0 else 128

    # Detect GPU automatically, fall back to CPU if not available
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[train_patch_policy] Using device: {_device}")

    if resume_from and os.path.exists(resume_from + ".zip"):
        model = PPO.load(resume_from, env=env, device=_device)
        # Restore VecNormalize stats so the policy sees the same obs scale it was trained on.
        # Convention: vecnorm lives next to the model file as best_vecnormalize.pkl
        import pickle as _pkl
        _vn_path = os.path.join(os.path.dirname(resume_from), "best_vecnormalize.pkl")
        if os.path.exists(_vn_path) and isinstance(env, VecNormalize):
            with open(_vn_path, "rb") as _f:
                _saved_vn = _pkl.load(_f)
            env.obs_rms = _saved_vn.obs_rms
            env.ret_rms = _saved_vn.ret_rms
            print(f"[resume] Loaded VecNormalize stats from {_vn_path}")
        else:
            print(f"[resume] WARNING: VecNormalize not found at {_vn_path} — obs stats reset")
    else:
        _lr_initial = 3e-4
        _lr_final   = 1e-5
        model = PPO(
            "MlpPolicy",
            env,
            tensorboard_log=tb_log_dir,
            learning_rate=_linear_lr_schedule(_lr_initial, _lr_final),
            # learning_rate = 3e-4,
            seed = 42,
            n_steps=rollout_steps,
            batch_size=batch_size,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.001,  # 0.01 caused entropy explosion (std→32); spinning fixed by reward penalties now
            # ent_coef=lambda progress: 0.008 * progress + 0.001,  # decays 0.009→0.001 over training
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            use_sde=False,
            device=_device,
            policy_kwargs=dict(
                **(
                    dict(
                        features_extractor_class=HybridCNNExtractor,
                        features_extractor_kwargs=dict(features_dim=256),
                    ) if obs_mode == "grid" else {}
                ),
                net_arch=dict(pi=[64, 64], vf=[64, 64]) if obs_mode == "grid" else dict(pi=[256, 256], vf=[256, 256]),
            )
        )

    save_best_cb = SaveBestWithVecNormalize(save_dir=run_dir, check_freq=rollout_steps)
    callbacks = [callback, save_best_cb]

    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
    model.save(os.path.join(run_dir, "final_model"))
    env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
    env.close()
    if WANDB_AVAILABLE and wandb_run is not None:
        wandb.finish()
    return model


# def train_agent_policy(
#     total_timesteps: int = 1000000,
#     save_path: str = "agent_policy_models",
#     checkpoint_freq: int = 2000,
#     NUM_ENVS: int = 4,
#     resume_from: Optional[str] = None,
#     patch_checkpoint_path: str = "",
#     patch_vecnorm_path: str = "",
#     norm_reward: bool = True,
#     patch_env=None,
#     # base_reset_type: str = "rl_random_static",
#     wandb_project: str = "agent_sempc_training",
#     wandb_entity: Optional[str] = None,
#     wandb_run_name: Optional[str] = None, ):
#     if not SB3_AVAILABLE:
#         raise RuntimeError("stable-baselines3 is required for training.")

#     os.makedirs(save_path, exist_ok=True)
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = os.path.join(save_path, f"run_{run_id}")
#     os.makedirs(run_dir, exist_ok=True)
#     tb_log_dir = os.path.join(run_dir, "tb")
#     os.makedirs(tb_log_dir, exist_ok=True)

#     wandb_run = None
#     if WANDB_AVAILABLE:
#         wandb_run = wandb.init(
#             project=wandb_project,
#             sync_tensorboard=True,
#             save_code=True,
#         )

#     def _monitored(thunk):
#         return lambda: Monitor(thunk())

#     if patch_env is None and patch_checkpoint_path:
#         patch_env = PatchEnv(PatchEnvConfig())
#         patch_env.patch_policy = PPO.load(patch_checkpoint_path)
#         patch_env.patch_policy.policy.set_training_mode(False)
#         if patch_vecnorm_path and os.path.exists(patch_vecnorm_path):
#             from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
#             import gymnasium as gym
#             from gymnasium import spaces
#             class _DummyPatchEnv(gym.Env):
#                 observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
#                 action_space = spaces.Box(
#                     low=np.array([-0.4189, 0.5, 0.325, 0.25], dtype=np.float32),
#                     high=np.array([0.4189, 10.0, 2.5, 1.5], dtype=np.float32),
#                 )
#                 def reset(self, **kw): return np.zeros(11, dtype=np.float32), {}
#                 def step(self, a): return np.zeros(11, dtype=np.float32), 0.0, False, False, {}
#             patch_env.patch_vecnorm = VecNormalize.load(patch_vecnorm_path, DummyVecEnv([_DummyPatchEnv]))
#             patch_env.patch_vecnorm.training = False
#             patch_env.patch_vecnorm.norm_reward = False
#         else:
#             patch_env.patch_vecnorm = None

#     # True parameter sharing: each AgentEnv is shared by two AgentView slots.
#     # Slots come in consecutive pairs [view0, view1, view0, view1, ...] so that
#     # view0 always submits its action first and view1 triggers the physics step.
#     env_fns = []
#     for i in range(NUM_ENVS):
#         fn0, fn1 = make_agent_views(i, seed=42, patch_env=patch_env)
#         env_fns.append(_monitored(fn0))
#         env_fns.append(_monitored(fn1))

#     # DummyVecEnv required: AgentView pairs share state — SubprocVecEnv would
#     # fork the process and break the shared reference between view0 and view1.
#     env = DummyVecEnv(env_fns)

#     env = VecCheckNan(env, raise_exception=True)
#     env = VecNormalize(
#         env,
#         norm_obs=True,
#         norm_reward=norm_reward,
#         clip_obs=10.0,
#         clip_reward=5.0,
#         gamma=0.99,
#     )

#     callback = TrainingCallback(run_dir, checkpoint_freq, "AGENT")

#     rollout_steps = 2048
#     # 2*NUM_ENVS slots (two AgentView per shared env)
#     n_slots = 2 * NUM_ENVS
#     batch_size = n_slots * rollout_steps // 4

#     if resume_from and os.path.exists(resume_from + ".zip"):
#         model = PPO.load(resume_from, env=env)
#     else:
#         model = PPO(
#             MAPPOPolicy,   # CTDE: decentralized actor (obs[:12]), centralized critic (obs[:24])
#             env,
#             tensorboard_log=tb_log_dir,
#             learning_rate=3e-4,
#             seed=42,
#             n_steps=rollout_steps,
#             batch_size=batch_size,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#             ent_coef=0.01,
#             vf_coef=0.5,
#             max_grad_norm=0.5,
#             verbose=1,
#             use_sde=False,
#             # net_arch is handled inside MAPPOPolicy._build_mlp_extractor
#         )

#     save_best_cb = SaveBestWithVecNormalize(save_dir=run_dir, check_freq=rollout_steps)
#     callbacks = [callback, save_best_cb]

#     model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
#     model.save(os.path.join(run_dir, "final_model"))
#     env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
#     env.close()
#     if WANDB_AVAILABLE and wandb_run is not None:
#         wandb.finish()
#     return model


# def train_joint_policy(
#     total_timesteps: int = 2_000_000,
#     save_path: str = "joint_policy_models",
#     checkpoint_freq: int = 10000,
#     NUM_ENVS: int = 4,
#     phase_steps: int = 4096,
#     resume_patch: str = "",
#     resume_agents: str = "",
#     norm_reward: bool = True,
#     wandb_project: str = "joint_sempc_training", ):
#     """Train patch + agent policies jointly with alternating MAPPO.

#     Each iteration alternates between:
#       1. Patch training (phase_steps): agents use frozen snapshot of agent policy.
#       2. Agent training (phase_steps): patch uses frozen snapshot of patch policy.

#     Both policies co-evolve — the patch learns to keep agents inside;
#     agents learn to follow whatever pace the patch sets.
#     """
#     if not SB3_AVAILABLE:
#         raise RuntimeError("stable-baselines3 is required.")

#     os.makedirs(save_path, exist_ok=True)
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = os.path.join(save_path, f"run_{run_id}")
#     os.makedirs(run_dir, exist_ok=True)

#     wandb_run = None
#     if WANDB_AVAILABLE:
#         wandb_run = wandb.init(
#             project=wandb_project,
#             sync_tensorboard=True,
#             save_code=True,
#         )

#     def _mon(thunk):
#         return lambda: Monitor(thunk())

#     # Build separate JointEnv sets for each training phase to avoid state collision.
#     # patch_shared_envs[i] is used for patch training only.
#     # agent_shared_envs[i] is used for agent training only.
#     patch_shared_envs, patch_fns = [], []
#     agent_shared_envs, agent_fns = [], []

#     for i in range(NUM_ENVS):
#         env_p, pf, _, _ = make_joint_views(i, seed=42)
#         patch_shared_envs.append(env_p)
#         patch_fns.append(_mon(pf))

#         env_a, _, af0, af1 = make_joint_views(i + NUM_ENVS, seed=42)
#         agent_shared_envs.append(env_a)
#         agent_fns.extend([_mon(af0), _mon(af1)])

#     # DummyVecEnv required: shared env state cannot cross process boundaries.
#     patch_vec = DummyVecEnv(patch_fns)
#     agent_vec = DummyVecEnv(agent_fns)

#     patch_vec = VecCheckNan(patch_vec, raise_exception=True)
#     agent_vec = VecCheckNan(agent_vec, raise_exception=True)
#     patch_vec = VecNormalize(patch_vec, norm_obs=True, norm_reward=norm_reward,
#                               clip_obs=10.0, clip_reward=10.0, gamma=0.99)
#     agent_vec = VecNormalize(agent_vec, norm_obs=True, norm_reward=norm_reward,
#                               clip_obs=10.0, clip_reward=5.0, gamma=0.99)

#     rollout_steps = 2048

#     # --- Patch PPO: standard MlpPolicy (15D obs → 4D action) ---
#     if resume_patch and os.path.exists(resume_patch + ".zip"):
#         patch_ppo = PPO.load(resume_patch, env=patch_vec)
#     else:
#         patch_ppo = PPO(
#             "MlpPolicy", patch_vec,
#             tensorboard_log=os.path.join(run_dir, "tb_patch"),
#             learning_rate=3e-4, seed=42,
#             n_steps=rollout_steps,
#             batch_size=NUM_ENVS * rollout_steps // 4,
#             n_epochs=10, gamma=0.99, gae_lambda=0.95,
#             clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
#             max_grad_norm=0.5, verbose=1,
#             policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
#         )

#     # --- Agent PPO: CTDE MAPPOPolicy (24D obs → 2D action, parameter shared) ---
#     n_agent_slots = 2 * NUM_ENVS
#     if resume_agents and os.path.exists(resume_agents + ".zip"):
#         agent_ppo = PPO.load(resume_agents, env=agent_vec)
#     else:
#         agent_ppo = PPO(
#             MAPPOPolicy, agent_vec,
#             tensorboard_log=os.path.join(run_dir, "tb_agents"),
#             learning_rate=3e-4, seed=42,
#             n_steps=rollout_steps,
#             batch_size=n_agent_slots * rollout_steps // 4,
#             n_epochs=10, gamma=0.99, gae_lambda=0.95,
#             clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
#             max_grad_norm=0.5, verbose=1,
#         )

#     patch_cb = TrainingCallback(run_dir, checkpoint_freq, "JOINT-PATCH")
#     agent_cb = TrainingCallback(run_dir, checkpoint_freq, "JOINT-AGENT")

#     # ------------------------------------------------------------------
#     # Alternating training loop
#     # ------------------------------------------------------------------
#     total_steps = 0
#     iteration = 0

#     while total_steps < total_timesteps:
#         iteration += 1
#         print(f"\n[Joint iter {iteration}] steps={total_steps}/{total_timesteps}")

#         # --- Phase 1: Train patch — agents use frozen snapshot ---
#         _agent_snap = agent_ppo.policy
#         _agent_vec_norm = agent_vec  # VecNormalize for un-normalising agent obs

#         for env in patch_shared_envs:
#             def _agent_fn(obs0, obs1, _pol=_agent_snap, _vn=_agent_vec_norm):
#                 # obs0/obs1 are raw (un-normalised) — normalise before prediction
#                 obs0_n = _vn.normalize_obs(obs0[np.newaxis])[0]
#                 obs1_n = _vn.normalize_obs(obs1[np.newaxis])[0]
#                 a0, _ = _pol.predict(obs0_n[np.newaxis], deterministic=True)
#                 a1, _ = _pol.predict(obs1_n[np.newaxis], deterministic=True)
#                 return a0[0], a1[0]
#             env._agent_action_fn = _agent_fn

#         patch_ppo.learn(
#             phase_steps, callback=patch_cb,
#             reset_num_timesteps=(iteration == 1), progress_bar=False,
#         )
#         total_steps += phase_steps

#         # --- Phase 2: Train agents — patch uses frozen snapshot ---
#         _patch_snap = patch_ppo.policy
#         _patch_vec_norm = patch_vec  # VecNormalize for patch obs

#         for env in agent_shared_envs:
#             def _patch_fn(obs, _pol=_patch_snap, _vn=_patch_vec_norm):
#                 obs_n = _vn.normalize_obs(obs[np.newaxis])[0]
#                 action, _ = _pol.predict(obs_n[np.newaxis], deterministic=True)
#                 return action[0]
#             env._patch_action_fn = _patch_fn

#         agent_ppo.learn(
#             phase_steps, callback=agent_cb,
#             reset_num_timesteps=(iteration == 1), progress_bar=False,
#         )
#         total_steps += phase_steps

#         # Log combined metrics to wandb once per iteration
#         if WANDB_AVAILABLE and wandb_run is not None:
#             patch_buf = patch_ppo.ep_info_buffer
#             agent_buf = agent_ppo.ep_info_buffer
#             log = {"joint/total_steps": total_steps, "joint/iteration": iteration}
#             if patch_buf:
#                 log["joint/patch_ep_rew_mean"] = float(np.mean([e["r"] for e in patch_buf]))
#                 log["joint/patch_ep_len_mean"] = float(np.mean([e["l"] for e in patch_buf]))
#             if agent_buf:
#                 log["joint/agent_ep_rew_mean"] = float(np.mean([e["r"] for e in agent_buf]))
#                 log["joint/agent_ep_len_mean"] = float(np.mean([e["l"] for e in agent_buf]))
#             wandb.log(log, step=total_steps)

#     # --- Save ---
#     patch_ppo.save(os.path.join(run_dir, "patch_final"))
#     agent_ppo.save(os.path.join(run_dir, "agents_final"))
#     patch_vec.save(os.path.join(run_dir, "patch_vecnorm_final.pkl"))
#     agent_vec.save(os.path.join(run_dir, "agents_vecnorm_final.pkl"))
#     patch_vec.close()
#     agent_vec.close()
#     if WANDB_AVAILABLE and wandb_run is not None:
#         wandb.finish()
#     print(f"Joint training complete. Models saved to {run_dir}")
#     return patch_ppo, agent_ppo


def _save_frozen_agent_npz(policy, vec_norm, path):
    """Extract agent policy weights + VecNormalize stats as numpy .npz.

    Only plain numpy arrays cross the process boundary (via filesystem),
    so SubprocVecEnv workers never need to pickle PyTorch objects.
    """
    sd = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}

    save_kw = {}
    for i in range(0, 100, 2):
        wk = f"mlp_extractor.policy_net.{i}.weight"
        bk = f"mlp_extractor.policy_net.{i}.bias"
        if wk not in sd:
            break
        save_kw[f"pi_w_{i}"] = sd[wk]
        save_kw[f"pi_b_{i}"] = sd[bk]
    save_kw["act_w"] = sd["action_net.weight"]
    save_kw["act_b"] = sd["action_net.bias"]

    act_fn = getattr(policy, "activation_fn", None)
    save_kw["use_tanh"] = np.array(
        act_fn is nn.Tanh if act_fn is not None else True
    )

    if vec_norm is not None and hasattr(vec_norm, "obs_rms"):
        save_kw["vnorm_mean"] = vec_norm.obs_rms.mean.copy()
        save_kw["vnorm_var"] = vec_norm.obs_rms.var.copy()
        save_kw["vnorm_clip"] = np.array(vec_norm.clip_obs)
        save_kw["vnorm_eps"] = np.array(vec_norm.epsilon)

    np.savez(path, **save_kw)


def train_joint_sb3(
    total_timesteps: int = 2_000_000,
    save_path: str = "joint_sb3_models",
    checkpoint_freq: int = 2000,
    num_envs: int = 8,
    num_cpus: int = 4,
    num_agents: int = 1,
    phase_steps: int = 8192,
    patch_solo_timesteps: int = 10_000_000,
    resume_patch: str = "",
    resume_agents: str = "",
    norm_reward: bool = False,
    agent_only: bool = False,
    pretrained_patch_path: str = "",
    pretrained_patch_vecnorm: str = "",
    wandb_project: str = "joint_sb3_training",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    dummy_vec: bool = False,):
    """Alternating co-evolution using PettingZoo + SuperSuit for agent parallelization.

    Phase 0 (optional): Patch-car solo on f110 (PatchCarEnv, same 11D obs as PatchEnv) for
    ``patch_solo_timesteps`` — set to 0 to skip and start directly with joint patch training.

    Phase A (each iteration): Train patch policy on JointPatchView (2 f110 cars; frozen
    agent policy from .npz when available). SubprocVecEnv for true parallelism.

    Phase B: Train single agent policy via SuperSuit (1 PettingZoo agent × num_envs parallel).
    Patch inside uses a frozen snapshot of the patch policy.
    """
    if not SB3_AVAILABLE:
        raise RuntimeError("stable-baselines3 is required.")

    # === JOINT TRAINING DISABLED — only agent_only path is supported ===
    # Use train_patch_policy() to produce the frozen patch checkpoint, then
    # call this function with agent_only=True and pretrained_patch_path=...
    if not agent_only:
        raise NotImplementedError(
            "Joint training (Phase A/B alternation) is disabled. "
            "Run with --agent_only=True and a pretrained_patch_path."
        )

    from stable_baselines3.common.vec_env import VecMonitor

    os.makedirs(save_path, exist_ok=True)
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(save_path, f"run_{run_id}")
    os.makedirs(run_dir, exist_ok=True)

    tb_log_dir = os.path.join(run_dir, "tb")
    os.makedirs(tb_log_dir, exist_ok=True)

    wandb_run = None
    if WANDB_AVAILABLE:
        _patch_cfg = dataclasses.asdict(PatchEnvConfig())
        wandb_run = wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=wandb_run_name or f"joint_sb3_{run_id}",
            config={
                "mode": "joint_sb3",
                "total_timesteps": total_timesteps,
                "patch_solo_timesteps": patch_solo_timesteps,
                "num_envs": num_envs,
                "num_cpus": num_cpus,
                "phase_steps": phase_steps,
                # key reward knobs — visible as flat fields in wandb run table
                "reward_steer_bias_weight": _patch_cfg["reward_steer_bias_weight"],
                "reward_spin_weight": _patch_cfg["reward_spin_weight"],
                "spin_yawrate_threshold": _patch_cfg["spin_yawrate_threshold"],
                "collision_penalty": _patch_cfg["collision_penalty"],
                "stuck_no_progress_steps": _patch_cfg["stuck_no_progress_steps"],
                "stuck_penalty": _patch_cfg["stuck_penalty"],
                "reward_progress_scale": _patch_cfg["reward_progress_scale"],
                "time_penalty_per_sec": _patch_cfg["time_penalty_per_sec"],
                "max_steps": _patch_cfg["max_steps"],
                # full config nested for completeness
                "patch_env_full": _patch_cfg,
            },
            sync_tensorboard=True,
            save_code=True,
            reinit=True,
        )

    _device = "cuda"

    TARGET_ROLLOUT = 65536
    patch_n_steps = max(512, TARGET_ROLLOUT // max(1, num_envs))
    agent_n_steps = max(512, TARGET_ROLLOUT // max(1, num_envs))
    patch_batch   = max(256, patch_n_steps * num_envs // 8)
    agent_batch   = max(256, agent_n_steps * num_envs // 8)

    min_phase = max(patch_n_steps * num_envs, agent_n_steps * num_envs)
    if phase_steps < min_phase:
        print(f"[WARNING] phase_steps={phase_steps} < {min_phase} "
              f"(min to guarantee 1 PPO update per phase). Rounding up.")
        phase_steps = min_phase
    print(f"[joint_sb3] patch n_steps={patch_n_steps} batch={patch_batch} | "
          f"agent n_steps={agent_n_steps} batch={agent_batch} | "
          f"phase_steps={phase_steps} | device={_device}")

    if agent_only:
        # phase_steps is for alternating joint training; agent_only needs a compact rollout.
        # With phase_steps=131072 and num_envs=16 the first PPO update requires 2M steps —
        # the policy never updates within a typical 20M-step budget. Use 4096 instead.
        agent_n_steps = 4096
        agent_batch   = max(256, agent_n_steps * num_envs // 8)  # → 8192
        print(f"[joint_sb3 agent_only] agent n_steps={agent_n_steps} batch={agent_batch}")

    patch_dir = os.path.join(run_dir, "patch")
    agent_dir = os.path.join(run_dir, "agents")
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Patch loading: agent_only mode loads the frozen patch from a
    # pretrained checkpoint produced by train_patch_policy. Joint patch
    # training (Phase 0 solo + Phase A on JointPatchView) is disabled.
    # ------------------------------------------------------------------
    # === JOINT TRAINING DISABLED — JointPatchView/JointEnv imports removed ===
    # try:
    #     from .ppo_policy import (
    #         JointEnv, JointEnvConfig, JointPatchView, PatchCarEnv, PatchEnvConfig,
    #     )
    # except ImportError:
    #     from envs.ppo_policy import (
    #         JointEnv, JointEnvConfig, JointPatchView, PatchCarEnv, PatchEnvConfig,
    #     )

    # === JOINT TRAINING DISABLED — patch_cb/save_patch_best unused ===
    # patch_cb = TrainingCallback(
    #     save_dir=patch_dir,
    #     save_freq=max(1, checkpoint_freq // num_envs),
    #     policy_name="patch_sb3",
    # )
    # save_patch_best = SaveBestWithVecNormalize(save_dir=patch_dir, check_freq=patch_n_steps,
    #                                            model_name="best_model")

    # === JOINT TRAINING DISABLED — Phase 0 PatchCarEnv builder unused here ===
    # def _make_patch_car():
    #     cfg = PatchEnvConfig(split_mode=True, random_spawn=True)
    #     env = PatchCarEnv(cfg)
    #     env.reset()
    #     return Monitor(env)

    # === JOINT TRAINING DISABLED — JointPatchView builder ===
    # def _make_joint_patch_env():
    #     try:
    #         print("[DEBUG] Creating JointEnv...")
    #         shared = JointEnv(JointEnvConfig(render_mode=None))
    #         print("[DEBUG] JointEnv created, calling reset...")
    #         shared.reset()
    #         print("[DEBUG] JointEnv reset successful")
    #         print("[DEBUG] Creating JointPatchView...")
    #         view = JointPatchView(shared)
    #         print("[DEBUG] JointPatchView created")
    #         return Monitor(view)
    #     except Exception as e:
    #         print(f"[ERROR] Failed to create JointEnv: {type(e).__name__}: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         raise

    patch_ppo = None
    # patch_car_vec = None
    # _solo_obs_rms = None

    # === JOINT TRAINING DISABLED — Phase 0 solo + Phase A patch training removed ===
    # if not agent_only:
    #     if patch_solo_timesteps > 0 and not resume_patch:
    #         if num_envs > 1:
    #             patch_car_vec = SubprocVecEnv(
    #                 [_make_patch_car for _ in range(num_envs)],
    #                 start_method="fork",
    #             )
    #         else:
    #             patch_car_vec = DummyVecEnv([_make_patch_car])
    #         patch_car_vec = VecMonitor(patch_car_vec)
    #         if norm_reward:
    #             patch_car_vec = VecNormalize(
    #                 patch_car_vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
    #             )
    #         patch_ppo = PPO(
    #             "MlpPolicy", patch_car_vec,
    #             tensorboard_log=os.path.join(tb_log_dir, "patch_solo"),
    #             n_steps=patch_n_steps, batch_size=patch_batch, n_epochs=10,
    #             gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    #             ent_coef=0.05,
    #             vf_coef=0.5, max_grad_norm=0.5,
    #             device=_device, verbose=1,
    #         )
    #         print(f"[joint_sb3] Phase 0: PatchCarEnv solo f110 for {patch_solo_timesteps} steps")
    #         patch_ppo.learn(
    #             patch_solo_timesteps,
    #             callback=[patch_cb, save_patch_best],
    #             reset_num_timesteps=True,
    #             progress_bar=False,
    #         )
    #         patch_ppo.save(os.path.join(patch_dir, "patch_after_solo"))
    #         _solo_obs_rms = patch_car_vec.obs_rms if isinstance(patch_car_vec, VecNormalize) else None
    #         if isinstance(patch_car_vec, VecNormalize):
    #             patch_car_vec.save(os.path.join(patch_dir, "patch_after_solo_vecnormalize.pkl"))
    #         patch_car_vec.close()
    #         patch_car_vec = None
    #
    #     # TEMPORARY: Use DummyVecEnv for Phase A to see actual worker errors
    #     # TODO: Switch back to SubprocVecEnv once JointEnv is verified stable
    #     print(f"[DEBUG] Creating patch_vec with DummyVecEnv (num_envs={num_envs})")
    #     # patch_vec = DummyVecEnv([_make_joint_patch_env for _ in range(num_envs)])
    #     if num_envs > 1:
    #         patch_vec = SubprocVecEnv(
    #             [_make_joint_patch_env for _ in range(num_envs)],
    #             start_method="fork",
    #         )
    #     else:
    #         patch_vec = DummyVecEnv([_make_joint_patch_env])
    #     patch_vec = VecMonitor(patch_vec)
    #     if norm_reward:
    #         patch_vec = VecNormalize(patch_vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
    #
    #     if resume_patch:
    #         patch_ppo = PPO.load(resume_patch, env=patch_vec)
    #     elif patch_ppo is not None:
    #         # Phase 0 trained on 267D (scalars + occupancy grid)
    #         # Phase A now also provides 267D (same format) so obs spaces match
    #         patch_ppo.set_env(patch_vec)
    #         # Transfer obs normalization stats from Phase 0
    #         if _solo_obs_rms is not None and isinstance(patch_vec, VecNormalize):
    #             patch_vec.obs_rms = _solo_obs_rms
    #     else:
    #         patch_ppo = PPO(
    #             "MlpPolicy", patch_vec,
    #             tensorboard_log=os.path.join(tb_log_dir, "patch"),
    #             n_steps=patch_n_steps, batch_size=patch_batch, n_epochs=10,
    #             gamma=0.99, gae_lambda=0.95, clip_range=0.2,
    #             ent_coef=0.05,
    #             vf_coef=0.5, max_grad_norm=0.5,
    #             device=_device, verbose=1,
    #         )
    # else:
    # agent_only: load frozen patch from checkpoint, no patch_vec
    if not pretrained_patch_path:
        raise ValueError("agent_only=True requires pretrained_patch_path")

    _frozen_patch_ppo = PPO.load(pretrained_patch_path, device="cpu")
    _frozen_patch_vn = None
    if pretrained_patch_vecnorm and os.path.exists(pretrained_patch_vecnorm):
        import pickle
        with open(pretrained_patch_vecnorm, "rb") as f:
            _frozen_patch_vn = pickle.load(f)
        _frozen_patch_vn.training = False
        _frozen_patch_vn.norm_reward = False

    print(f"[joint_sb3 agent_only] Loaded frozen patch from {pretrained_patch_path}")
    patch_vec = None   # signal for final cleanup
    patch_ppo = None

    # ------------------------------------------------------------------
    # Frozen patch policy → pure numpy.
    #
    # The patch policy must run inside every env worker. Running PyTorch in a
    # forked worker segfaults (fork copies PyTorch's thread pools/mutexes).
    # In lidar mode the patch policy is a plain MlpPolicy (FlattenExtractor +
    # [256,256] tanh MLP — see train_patch_policy net_arch), so we extract its
    # weights to numpy and do the forward pass by hand. The resulting closure
    # captures ONLY numpy arrays → picklable, and fork/spawn-safe (no PyTorch
    # in workers at all).
    # ------------------------------------------------------------------
    _sd = _frozen_patch_ppo.policy.state_dict()
    _patch_mlp = []                       # [(W, b), ...] for the policy MLP
    _li = 0
    while f"mlp_extractor.policy_net.{_li}.weight" in _sd:
        _W = _sd[f"mlp_extractor.policy_net.{_li}.weight"].detach().cpu().numpy().astype(np.float32)
        _b = _sd[f"mlp_extractor.policy_net.{_li}.bias"].detach().cpu().numpy().astype(np.float32)
        _patch_mlp.append((_W, _b))
        _li += 2                          # skip Tanh layers (odd indices)
    _patch_act_w = _sd["action_net.weight"].detach().cpu().numpy().astype(np.float32)
    _patch_act_b = _sd["action_net.bias"].detach().cpu().numpy().astype(np.float32)
    _patch_act_low  = np.asarray(_frozen_patch_ppo.action_space.low,  dtype=np.float32)
    _patch_act_high = np.asarray(_frozen_patch_ppo.action_space.high, dtype=np.float32)

    if _frozen_patch_vn is not None and hasattr(_frozen_patch_vn, "obs_rms"):
        _pnorm_mean = _frozen_patch_vn.obs_rms.mean.copy().astype(np.float32)
        _pnorm_var  = _frozen_patch_vn.obs_rms.var.copy().astype(np.float32)
        _pnorm_clip = float(_frozen_patch_vn.clip_obs)
        _pnorm_eps  = float(getattr(_frozen_patch_vn, "epsilon", 1e-8))
    else:
        _pnorm_mean = None
        _pnorm_var  = None
        _pnorm_clip = 10.0
        _pnorm_eps  = 1e-8

    if len(_patch_mlp) == 0:
        raise RuntimeError(
            "Frozen patch policy has no mlp_extractor.policy_net layers — "
            "expected a plain MlpPolicy (lidar mode). A CNN/grid-mode patch "
            "checkpoint is not supported by the numpy patch runner."
        )

    def _frozen_patch_fn(obs,
                         _mlp=_patch_mlp, _aw=_patch_act_w, _ab=_patch_act_b,
                         _alow=_patch_act_low, _ahigh=_patch_act_high,
                         _mean=_pnorm_mean, _var=_pnorm_var,
                         _clip=_pnorm_clip, _eps=_pnorm_eps):
        h = np.asarray(obs, dtype=np.float32)
        if _mean is not None:
            h = np.clip((h - _mean) / np.sqrt(_var + _eps), -_clip, _clip).astype(np.float32)
        for _W, _b in _mlp:                       # tanh MLP (SB3 MlpPolicy default)
            h = np.tanh(h @ _W.T + _b)
        action = h @ _aw.T + _ab                  # action head, no activation
        return np.clip(action, _alow, _ahigh).astype(np.float32)

    # ------------------------------------------------------------------
    # Agent vec env — SuperSuit multi-agent (Plan B).
    #
    # pettingzoo_env_to_vec_env_v1 turns the N-agent AgentEnv into an N-slot
    # vector env: each agent gets its OWN observation and the shared policy
    # produces a SEPARATE action per agent — true parameter sharing. (The old
    # _AgentGymEnv broadcast agent_0's action to every agent, so agents 1..N-1
    # were uncontrolled passengers that always fell out of the patch.)
    #
    # We then stack num_envs copies → N * num_envs agent slots for SB3,
    # parallelised across num_cpus worker processes.
    #
    # NOTE: ss.concat_vec_envs_v1 is broken for num_cpus > 1 in the installed
    # SuperSuit — its internal vec_env_args() returns only (env_fns,) while the
    # async constructor (MakeCPUAsyncConstructor) requires
    # (env_fns, obs_space, act_space). We call the constructor directly and
    # supply the spaces from the MarkovVectorEnv ourselves.
    # ------------------------------------------------------------------
    if not SUPERSUIT_AVAILABLE:
        raise ImportError("supersuit is required for joint_sb3 agent training")

    from supersuit.vector import MakeCPUAsyncConstructor
    from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

    _agent_env_config = {"num_agents": num_agents}

    def _make_agent_pz_env():
        try:
            from .pz_env import AgentEnv
        except ImportError:
            from envs.pz_env import AgentEnv
        env = AgentEnv(patch_policy=_frozen_patch_fn, env_config=_agent_env_config)
        return ss.pettingzoo_env_to_vec_env_v1(env)

    # dummy_vec → num_cpus=0 (serial, single process) so exceptions surface
    # with full tracebacks instead of being swallowed by worker processes.
    _concat_cpus = 0 if dummy_vec else num_cpus

    # One template just to read the per-agent obs/action spaces (no f110 init).
    _markov_template = _make_agent_pz_env()
    _agent_obs_space = _markov_template.observation_space
    _agent_act_space = _markov_template.action_space
    _markov_template.close()

    agent_vec = SB3VecEnvWrapper(
        MakeCPUAsyncConstructor(_concat_cpus)(
            [_make_agent_pz_env] * num_envs,
            _agent_obs_space,
            _agent_act_space,
        )
    )
    agent_vec = VecMonitor(agent_vec)
    # Always apply obs normalisation (zero-mean, unit-var → helps policy gradient).
    # NEVER normalise rewards: clip_reward=10 would compress +0.1 inside_reward to ≈ 0
    # and make repulsion (-100) and collision (-3000) indistinguishable at the clip wall.
    # The reward structure is intentional — PPO + max_grad_norm=0.5 handles raw magnitudes.
    agent_vec = VecNormalize(agent_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

    agent_cb = TrainingCallback(
        save_dir=agent_dir,
        save_freq=max(1, checkpoint_freq // num_envs),
        policy_name="agent_sb3",
    )
    save_agent_best = SaveBestWithVecNormalize(save_dir=agent_dir, check_freq=agent_n_steps,
                                               model_name="best_model")

    _agent_tb_log = os.path.join(tb_log_dir, "agent")
    if resume_agents:
        agent_ppo = PPO.load(resume_agents, env=agent_vec,
                             tensorboard_log=_agent_tb_log)
    else:
        agent_ppo = PPO(
            "MlpPolicy", agent_vec,
            tensorboard_log=_agent_tb_log,
            n_steps=agent_n_steps, batch_size=agent_batch, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
            device=_device, verbose=1,
        )

    # === JOINT TRAINING DISABLED — frozen-agent .npz handoff was used by Phase A ===
    # _frozen_agent_npz = os.path.join(run_dir, "_frozen_agent.npz")

    print(f"[joint_sb3 agent_only] {num_agents} agent(s) × {num_envs} envs "
          f"= {num_agents * num_envs} SuperSuit slots. Training for {total_timesteps} steps.")

    agent_ppo.learn(
        total_timesteps,
        callback=[agent_cb, save_agent_best],
        reset_num_timesteps=not bool(resume_agents),
        progress_bar=False,
    )

    # Save final agent only
    agent_ppo.save(os.path.join(agent_dir, "final_model"))
    if isinstance(agent_vec, VecNormalize):
        agent_vec.save(os.path.join(agent_dir, "final_vecnormalize.pkl"))
    agent_vec.close()
    if WANDB_AVAILABLE and wandb_run is not None:
        wandb.finish()
    print(f"agent_only training complete. Models saved to {run_dir}")
    return None, agent_ppo

    # === JOINT TRAINING DISABLED — Phase A/B alternating loop ===
    # else:
    #     for iteration in range(1, iterations + 1):
    #         print(f"\n=== Iteration {iteration}/{iterations} ===")
    #
    #         # ------------------------------------------------------------------
    #         # Inject frozen agent policy into patch SubprocVecEnv workers.
    #         # Workers load a .npz file (only a string crosses the pipe).
    #         # ------------------------------------------------------------------
    #         if _has_trained_agents:
    #             patch_vec.env_method("load_frozen_agent_npz", _frozen_agent_npz)
    #         else:
    #             patch_vec.env_method("load_frozen_agent_npz", None)
    #
    #         # Phase 1: train patch (do not reset SB3 timestep schedule on iter 1 if solo warmup ran)
    #         patch_ppo.learn(
    #             phase_steps,
    #             callback=[patch_cb, save_patch_best],
    #             reset_num_timesteps=(iteration == 1 and not solo_done),
    #             progress_bar=False,
    #         )
    #         total_steps     += phase_steps
    #         patch_ppo_steps += phase_steps
    #
    #         # Phase 2: train agent with frozen patch policy snapshot.
    #         # Just swap the policy inside the mutable holder — no env rebuild needed.
    #         _patch_snap = patch_ppo.policy
    #         _patch_vn = patch_vec if isinstance(patch_vec, VecNormalize) else None
    #
    #         def _frozen_patch(obs: np.ndarray, _pol=_patch_snap, _vn=_patch_vn) -> np.ndarray:
    #             obs_in = obs[np.newaxis]
    #             if _vn is not None:
    #                 obs_in = _vn.normalize_obs(obs_in)
    #             action, _ = _pol.predict(obs_in, deterministic=True)
    #             return action[0]
    #
    #         _patch_holder.update(_frozen_patch)
    #
    #         agent_ppo.learn(
    #             phase_steps,
    #             callback=[agent_cb, save_agent_best],
    #             reset_num_timesteps=(iteration == 1),
    #             progress_bar=False,
    #         )
    #         total_steps     += phase_steps
    #         agent_ppo_steps += phase_steps
    #
    #         # Save frozen agent weights for next iteration's patch phase
    #         _agent_vn = agent_vec if isinstance(agent_vec, VecNormalize) else None
    #         _save_frozen_agent_npz(agent_ppo.policy, _agent_vn, _frozen_agent_npz)
    #         _has_trained_agents = True
    #
    #         if WANDB_AVAILABLE and wandb_run is not None:
    #             patch_buf = patch_ppo.ep_info_buffer
    #             agent_buf = agent_ppo.ep_info_buffer
    #             log = {
    #                 "joint_sb3/total_steps":     total_steps,
    #                 "joint_sb3/patch_ppo_steps": patch_ppo_steps,
    #                 "joint_sb3/agent_ppo_steps": agent_ppo_steps,
    #                 "joint_sb3/iteration":       iteration,
    #             }
    #             if patch_buf:
    #                 log["joint_sb3/patch_ep_rew_mean"] = float(np.mean([e["r"] for e in patch_buf]))
    #                 log["joint_sb3/patch_ep_len_mean"] = float(np.mean([e["l"] for e in patch_buf]))
    #             if agent_buf:
    #                 log["joint_sb3/agent_ep_rew_mean"] = float(np.mean([e["r"] for e in agent_buf]))
    #                 log["joint_sb3/agent_ep_len_mean"] = float(np.mean([e["l"] for e in agent_buf]))
    #             wandb.log(log, step=total_steps)
    #
    #     # --- Save final ---
    #     patch_ppo.save(os.path.join(patch_dir, "final_model"))
    #     agent_ppo.save(os.path.join(agent_dir, "final_model"))
    #     if isinstance(patch_vec, VecNormalize):
    #         patch_vec.save(os.path.join(patch_dir, "final_vecnormalize.pkl"))
    #     if isinstance(agent_vec, VecNormalize):
    #         agent_vec.save(os.path.join(agent_dir, "final_vecnormalize.pkl"))
    #     patch_vec.close()
    #     agent_vec.close()
    #     if os.path.exists(_frozen_agent_npz):
    #         os.remove(_frozen_agent_npz)
    #     if WANDB_AVAILABLE and wandb_run is not None:
    #         wandb.finish()
    #     print(f"joint_sb3 training complete. Models saved to {run_dir}")
    #     return patch_ppo, agent_ppo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train patch or agent policy with modular envs stack.")
    parser.add_argument("--mode", type=str, default="patch",
                        choices=["patch", "agent", "joint", "joint_sb3"],
                        help="Which policy to train")
    parser.add_argument("--timesteps", type=int, default=500000)
    parser.add_argument("--checkpoint-freq", type=int, default=10000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--save-path", type=str, default="patch_policy_models")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--domain-randomize", action="store_true")
    parser.add_argument("--no-norm-reward", action="store_true")
    parser.add_argument("--base-reset-type", type=str, default="rl_random_static")
    # Agent-mode args
    parser.add_argument("--patch-checkpoint", type=str, default="",
                        help="Path to frozen patch PPO checkpoint (without .zip), required for --mode agent")
    parser.add_argument("--patch-vecnorm", type=str, default="",
                        help="Path to VecNormalize .pkl for the patch policy, required for --mode agent")
    # Joint-mode args
    parser.add_argument("--phase-steps", type=int, default=131072,
                        help="Steps per training phase per policy (joint mode)")
    parser.add_argument(
        "--patch-solo-timesteps",
        type=int,
        default=10_000_000,
        help="joint_sb3: f110 PatchCarEnv solo warmup before alternating (0 = skip)",
    )
    parser.add_argument("--resume-patch", type=str, default="",
                        help="Resume patch policy from checkpoint (joint mode)")
    parser.add_argument("--resume-agents", type=str, default="",
                        help="Resume agent policy from checkpoint (joint mode)")
    parser.add_argument("--wandb-project", type=str, default="patch_sempc_training")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    # joint_sb3-specific
    parser.add_argument("--num-cpus", type=int, default=4,
                        help="CPU workers for SuperSuit concat_vec_envs_v1 (joint_sb3 mode)")
    parser.add_argument("--num-agents", type=int, default=2, choices=[1, 2, 3, 4],
                        help="Number of learning agents in each JointEnv (1–4)")
    parser.add_argument("--agent-only", action="store_true",
                        help="Train agents only against a frozen patch (no Phase A)")
    parser.add_argument("--pretrained-patch-path", type=str, default="",
                        help="Path to frozen patch .zip checkpoint (agent_only mode)")
    parser.add_argument("--pretrained-patch-vecnorm", type=str, default="",
                        help="Path to patch VecNormalize .pkl (agent_only mode, optional)")
    parser.add_argument("--dummy-vec", action="store_true",
                        help="Run agent envs serially in-process (num_cpus=0) — surfaces full tracebacks for debugging")
    args = parser.parse_args()

    if args.mode == "joint_sb3":
        train_joint_sb3(
            total_timesteps=args.timesteps,
            save_path=args.save_path if args.save_path != "patch_policy_models" else "joint_sb3_models",
            checkpoint_freq=args.checkpoint_freq,
            num_envs=1 if args.dummy_vec else args.num_envs,
            num_cpus=args.num_cpus,
            num_agents=args.num_agents,
            phase_steps=args.phase_steps,
            patch_solo_timesteps=args.patch_solo_timesteps,
            resume_patch=args.resume_patch,
            resume_agents=args.resume_agents,
            norm_reward=not args.no_norm_reward,
            agent_only=args.agent_only,
            pretrained_patch_path=args.pretrained_patch_path,
            pretrained_patch_vecnorm=args.pretrained_patch_vecnorm,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            dummy_vec=args.dummy_vec,
        )
    # === JOINT TRAINING DISABLED — joint and agent modes routed through commented-out functions ===
    # elif args.mode == "joint":
    #     train_joint_policy(
    #         total_timesteps=args.timesteps,
    #         save_path=args.save_path if args.save_path != "patch_policy_models" else "joint_policy_models",
    #         checkpoint_freq=args.checkpoint_freq,
    #         NUM_ENVS=args.num_envs,
    #         phase_steps=args.phase_steps,
    #         resume_patch=args.resume_patch,
    #         resume_agents=args.resume_agents,
    #         norm_reward=not args.no_norm_reward,
    #         wandb_project=args.wandb_project,
    #     )
    # elif args.mode == "agent":
    #     train_agent_policy(
    #         total_timesteps=args.timesteps,
    #         save_path=args.save_path if args.save_path != "patch_policy_models" else "agent_policy_models",
    #         checkpoint_freq=args.checkpoint_freq,
    #         NUM_ENVS=args.num_envs,
    #         resume_from=args.resume,
    #         patch_checkpoint_path=args.patch_checkpoint,
    #         patch_vecnorm_path=args.patch_vecnorm,
    #         norm_reward=not args.no_norm_reward,
    #         wandb_project=args.wandb_project,
    #         wandb_entity=args.wandb_entity,
    #         wandb_run_name=args.wandb_run_name,
    #     )
    elif args.mode in ("joint", "agent"):
        raise SystemExit(
            f"--mode {args.mode} is disabled. "
            "Use --mode patch to train the patch, then --mode joint_sb3 --agent-only "
            "with --pretrained-patch-path to train agents against a frozen patch."
        )
    else:
        train_patch_policy(
            total_timesteps=args.timesteps,
            save_path=args.save_path,
            checkpoint_freq=args.checkpoint_freq,
            NUM_ENVS=args.num_envs,
            resume_from=args.resume,
            domain_randomize=args.domain_randomize,
            norm_reward=not args.no_norm_reward,
            base_reset_type=args.base_reset_type,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
        )



# for training the agents with the patch run this command 

# python -m envs.training \
#   --mode agent \
#   --patch-checkpoint /path/to/patch_model \
#   --patch-vecnorm /path/to/vecnormalize.pkl \
#   --timesteps 1000000 \
#   --num-envs 4


#old code 
# import argparse
# import dataclasses
# import json
# import os
# import time
# from collections import Counter
# from datetime import datetime
# from typing import Optional

# import numpy as np
# import torch 
# import torch.nn as nn 

# try:
#     from .ppo_policy import (
#         make_patch_env, PatchEnv, PatchEnvConfig, JointEnvConfig,
#     )
# except ImportError:
#     from envs.ppo_policy import (
#         make_patch_env, PatchEnv, PatchEnvConfig, JointEnvConfig,
#     )

# try:
#     from .cnn_extractor import HybridCNNExtractor
# except ImportError:
#     from envs.cnn_extractor import HybridCNNExtractor

# try:
#     from stable_baselines3 import PPO
#     from stable_baselines3.common.callbacks import BaseCallback
#     from stable_baselines3.common.monitor import Monitor
#     from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize

#     SB3_AVAILABLE = True
# except ImportError:
#     SB3_AVAILABLE = False
#     BaseCallback = object

# try:
#     import supersuit as ss
#     SUPERSUIT_AVAILABLE = True
# except ImportError:
#     SUPERSUIT_AVAILABLE = False

# try:
#     import wandb

#     WANDB_AVAILABLE = True
# except ImportError:
#     WANDB_AVAILABLE = False

# ######################################################
# # Save best model + VecNormalize stats
# ######################################################
# class SaveBestWithVecNormalize(BaseCallback):
#     """
#     Periodically checks recent episode rewards and saves:
#       - the best PPO model checkpoint
#       - VecNormalize statistics (critical for consistent eval)
#     """

#     def __init__(self, save_dir: str, check_freq: int = 2048, verbose: int = 1,
#                  model_name: str = "best_model"):
#         super().__init__(verbose)
#         self.save_dir = save_dir
#         self.check_freq = int(check_freq)
#         self.best_model_path = os.path.join(save_dir, model_name)
#         self.best_vecnorm_path = os.path.join(save_dir, model_name + "_vecnorm.pkl")
#         self.best_mean_ep_len = -1e18

#     def _on_step(self) -> bool:
#         if self.n_calls % self.check_freq != 0:
#             return True

#         if len(self.model.ep_info_buffer) == 0:
#             return True

#         mean_ep_len = float(np.mean([ep["l"] for ep in self.model.ep_info_buffer]))
#         mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
#         print(f"Num timesteps: {self.num_timesteps} | Mean ep_len: {mean_ep_len:.0f} | Mean reward: {mean_reward:.2f}")

#         # Track by episode LENGTH not reward — longer episodes = better navigation.
#         # Under penalty-heavy rewards, short episodes can have higher reward than long ones.
#         if mean_ep_len > self.best_mean_ep_len:
#             self.best_mean_ep_len = mean_ep_len
#             print(f"Saving new best (ep_len={mean_ep_len:.0f})")
#             self.model.save(self.best_model_path)

#             if isinstance(self.training_env, VecNormalize):
#                 self.training_env.save(self.best_vecnorm_path)

#         return True

# class TrainingCallback(BaseCallback):
#     def __init__(self, save_dir: str, save_freq: int, policy_name: str, verbose: int = 1):
#         super().__init__(verbose)
#         self.save_dir = save_dir
#         self.save_freq = save_freq
#         self.policy_name = policy_name
#         self.episode_rewards = []
#         self.episode_count = 0
#         self.start_time = time.time()
#         self.last_log_time = time.time()
#         self.last_log_timesteps = 0
#         self.best_reward = -np.inf
#         self.episode_lengths = []
#         self.termination_reasons = {}
#         self.recent_reasons = []
#         self.recent_min_dists = []
#         self.recent_last_ds = []
#         self.recent_last_ey = []
#         self.recent_mpc_feas = []
#         self.recent_safety_rates = []
#         self.recent_reward_clipped = []
#         self.recent_agent_step_disp = []
#         self.recent_agent_move_ratio = []
#         self.recent_mean_speed_cmd = []
#         self.recent_mean_abs_steer_cmd = []
#         self.recent_control_order = []
#         self.recent_lap_progress = []
#         self.recent_terminal_frenet_s = []
#         self.log_file = os.path.join(save_dir, "training_log.csv")
#         with open(self.log_file, "w", encoding="utf-8") as f:
#             # old csv header kept for reference:
#             # f.write("timesteps,episodes,avg_reward,avg_len,time_elapsed,timesteps_per_sec,top_reason,avg_min_dist,avg_last_ds,avg_last_ey,avg_mpc_feas,avg_safety_rate,clip_frac\n")
#             f.write(
#                 "timesteps,episodes,avg_reward,avg_len,time_elapsed,timesteps_per_sec,"
#                 "top_reason,avg_min_dist,avg_last_ds,avg_last_ey,avg_mpc_feas,avg_safety_rate,clip_frac,"
#                 "avg_agent_step_disp,avg_agent_move_ratio,avg_cmd_speed,avg_cmd_abs_steer,control_order_sample\n"
#             )

#     _LOG_WINDOW = 200

#     def _on_step(self):
#         dones = self.locals.get("dones", [])
#         infos = self.locals.get("infos", [{}])
#         for i, done in enumerate(dones):
#             if not done:
#                 continue
#             self.episode_count += 1
#             info = infos[i]
#             reward = float(info.get("episode_reward", 0.0))
#             ep_len = int(info.get("Episode_steps", 0))
#             reason = str(info.get("termination_reason", "unknown"))
#             self.episode_rewards.append(reward)
#             self.episode_lengths.append(ep_len)
#             if len(self.episode_rewards) > self._LOG_WINDOW * 2:
#                 self.episode_rewards = self.episode_rewards[-self._LOG_WINDOW:]
#                 self.episode_lengths = self.episode_lengths[-self._LOG_WINDOW:]
#             self.termination_reasons[reason] = self.termination_reasons.get(reason, 0) + 1
#             self.recent_reasons.append(reason)
#             self.recent_min_dists.append(float(info.get("min_lidar_dist", np.nan)))
#             self.recent_last_ds.append(float(info.get("ds", np.nan)))
#             self.recent_last_ey.append(float(info.get("ey", np.nan)))
#             # self.recent_mpc_feas.append(float(info.get("mpc_feasibility_rate", np.nan)))
#             self.recent_safety_rates.append(float(info.get("safety_intervention_rate", np.nan)))
#             self.recent_reward_clipped.append(bool(info.get("reward_was_clipped", False)))
#             self.recent_agent_step_disp.append(float(info.get("episode_avg_agent_step_disp", np.nan)))
#             self.recent_agent_move_ratio.append(float(info.get("episode_agent_move_step_ratio", np.nan)))
#             self.recent_mean_speed_cmd.append(float(info.get("mean_speed_cmd", np.nan)))
#             self.recent_mean_abs_steer_cmd.append(float(info.get("mean_abs_steer_cmd", np.nan)))
#             self.recent_control_order.append(str(info.get("control_input_order", "unknown")))
#             self.recent_lap_progress.append(float(info.get("lap_progress", np.nan)))
#             self.recent_terminal_frenet_s.append(float(info.get("frenet_s", np.nan)))

#             if len(self.recent_reasons) > self._LOG_WINDOW * 2:
#                 w = self._LOG_WINDOW
#                 self.recent_reasons = self.recent_reasons[-w:]
#                 self.recent_min_dists = self.recent_min_dists[-w:]
#                 self.recent_last_ds = self.recent_last_ds[-w:]
#                 self.recent_last_ey = self.recent_last_ey[-w:]
#                 self.recent_mpc_feas = self.recent_mpc_feas[-w:]
#                 self.recent_safety_rates = self.recent_safety_rates[-w:]
#                 self.recent_reward_clipped = self.recent_reward_clipped[-w:]
#                 self.recent_agent_step_disp = self.recent_agent_step_disp[-w:]
#                 self.recent_agent_move_ratio = self.recent_agent_move_ratio[-w:]
#                 self.recent_mean_speed_cmd = self.recent_mean_speed_cmd[-w:]
#                 self.recent_mean_abs_steer_cmd = self.recent_mean_abs_steer_cmd[-w:]
#                 self.recent_control_order = self.recent_control_order[-w:]
#                 self.recent_lap_progress = self.recent_lap_progress[-w:]
#                 self.recent_terminal_frenet_s = self.recent_terminal_frenet_s[-w:]

#             if self.episode_count % 50 == 0 and self.episode_rewards:
#                 now = time.time()
#                 elapsed = now - self.start_time
#                 dt = now - self.last_log_time
#                 ts_delta = self.num_timesteps - self.last_log_timesteps
#                 tps = ts_delta / dt if dt > 0 else 0.0
#                 avg_r = float(np.mean(self.episode_rewards[-50:]))
#                 avg_len = float(np.mean(self.episode_lengths[-50:])) if self.episode_lengths else 0.0
#                 top_reason = max(self.termination_reasons.items(), key=lambda kv: kv[1])[0] if self.termination_reasons else "unknown"
#                 recent_reason_counts = Counter(self.recent_reasons[-50:])
#                 reason_breakdown = ", ".join(f"{k}:{v}" for k, v in recent_reason_counts.most_common(4))
#                 avg_min_dist = float(np.nanmean(self.recent_min_dists[-50:])) if self.recent_min_dists else float("nan")
#                 avg_last_ds = float(np.nanmean(self.recent_last_ds[-50:])) if self.recent_last_ds else float("nan")
#                 avg_last_ey = float(np.nanmean(self.recent_last_ey[-50:])) if self.recent_last_ey else float("nan")
#                 avg_mpc_feas = float(np.nanmean(self.recent_mpc_feas[-50:])) if self.recent_mpc_feas else float("nan")
#                 avg_safety = float(np.nanmean(self.recent_safety_rates[-50:])) if self.recent_safety_rates else float("nan")
#                 clip_frac = float(np.mean(self.recent_reward_clipped[-50:])) if self.recent_reward_clipped else 0.0
#                 avg_agent_step_disp = float(np.nanmean(self.recent_agent_step_disp[-50:])) if self.recent_agent_step_disp else float("nan")
#                 avg_agent_move_ratio = float(np.nanmean(self.recent_agent_move_ratio[-50:])) if self.recent_agent_move_ratio else float("nan")
#                 avg_cmd_speed = float(np.nanmean(self.recent_mean_speed_cmd[-50:])) if self.recent_mean_speed_cmd else float("nan")
#                 avg_cmd_abs_steer = float(np.nanmean(self.recent_mean_abs_steer_cmd[-50:])) if self.recent_mean_abs_steer_cmd else float("nan")
#                 avg_lap_progress = float(np.nanmean(self.recent_lap_progress[-50:])) if self.recent_lap_progress else float("nan")
#                 avg_terminal_s = float(np.nanmean(self.recent_terminal_frenet_s[-50:])) if self.recent_terminal_frenet_s else float("nan")
#                 control_order_sample = self.recent_control_order[-1] if self.recent_control_order else "unknown"
#                 print(f"\n[{self.policy_name}] Episode {self.episode_count} | Timesteps: {self.num_timesteps}")
#                 print(
#                     f"  Avg Reward: {avg_r:.2f} | Avg Len: {avg_len:.1f} | "
#                     f"Top End: {top_reason} | Time: {elapsed/60:.1f} min | Speed: {tps:.1f} steps/sec"
#                 )
#                 print(f"  Reasons(last50): [{reason_breakdown}]")
#                 # Fields below only exist in old single-agent env; skip if all NaN
#                 if not (np.isnan(avg_min_dist) and np.isnan(avg_last_ds)):
#                     print(
#                         f"  Debug(last50): min_dist={avg_min_dist:.3f} "
#                         f"ds={avg_last_ds:.4f} ey={avg_last_ey:.3f}"
#                     )
#                 if not np.isnan(avg_agent_step_disp):
#                     print(
#                         f"  AgentMotion(last50): avg_step_disp={avg_agent_step_disp:.4f}m "
#                         f"move_step_ratio={avg_agent_move_ratio:.3f}"
#                     )
#                 if not np.isnan(avg_cmd_speed):
#                     print(
#                         f"  Cmd(last50): mean_speed={avg_cmd_speed:.3f} "
#                         f"mean_abs_steer={avg_cmd_abs_steer:.3f} "
#                         f"control_order={control_order_sample}"
#                     )
#                 if avg_len < 5.0:
#                     print("  WARNING: Very short episodes. Focus on termination reasons.")
#                 with open(self.log_file, "a", encoding="utf-8") as f:
#                     # old csv row kept for reference:
#                     # f.write(f"{self.num_timesteps},{self.episode_count},{avg_r:.2f},{avg_len:.2f},{elapsed:.1f},{tps:.2f},{top_reason},{avg_min_dist:.4f},{avg_last_ds:.6f},{avg_last_ey:.4f},{avg_mpc_feas:.4f},{avg_safety:.4f},{clip_frac:.4f}\n")
#                     f.write(
#                         f"{self.num_timesteps},{self.episode_count},{avg_r:.2f},"
#                         f"{avg_len:.2f},{elapsed:.1f},{tps:.2f},{top_reason},"
#                         f"{avg_min_dist:.4f},{avg_last_ds:.6f},{avg_last_ey:.4f},"
#                         f"{avg_mpc_feas:.4f},{avg_safety:.4f},{clip_frac:.4f},"
#                         f"{avg_agent_step_disp:.6f},{avg_agent_move_ratio:.4f},"
#                         f"{avg_cmd_speed:.4f},{avg_cmd_abs_steer:.4f},\"{control_order_sample}\"\n"
#                     )

#                 if avg_r > self.best_reward:
#                     self.best_reward = avg_r
#                     self.model.save(os.path.join(self.save_dir, "best_model"))
#                     if hasattr(self.training_env, "save"):
#                         self.training_env.save(os.path.join(self.save_dir, "best_vecnormalize.pkl"))
#                     print(f"  New best model: {avg_r:.2f}")

#                 # Push custom metrics to WandB (in addition to the SB3 TB sync)
#                 if WANDB_AVAILABLE:
#                     try:
#                         import wandb as _wandb
#                         if _wandb.run is not None:
#                             log = {
#                                 f"{self.policy_name}/avg_reward": avg_r,
#                                 f"{self.policy_name}/avg_ep_len": avg_len,
#                                 f"{self.policy_name}/steps_per_sec": tps,
#                             }
#                             for reason, cnt in recent_reason_counts.items():
#                                 log[f"{self.policy_name}/reason_{reason}"] = cnt / max(len(self.recent_reasons[-50:]), 1)
#                             if not np.isnan(avg_cmd_speed):
#                                 log[f"{self.policy_name}/cmd_speed"] = avg_cmd_speed
#                                 log[f"{self.policy_name}/cmd_abs_steer"] = avg_cmd_abs_steer
#                             if not np.isnan(avg_lap_progress):
#                                 log[f"{self.policy_name}/lap_progress"] = avg_lap_progress
#                             if not np.isnan(avg_terminal_s):
#                                 log[f"{self.policy_name}/terminal_frenet_s"] = avg_terminal_s
#                             _wandb.log(log, step=self.num_timesteps)
#                     except Exception:
#                         pass

#                 self.last_log_time = now
#                 self.last_log_timesteps = self.num_timesteps

#         if self.num_timesteps % self.save_freq == 0:
#             path = os.path.join(self.save_dir, f"checkpoint_{self.num_timesteps}")
#             self.model.save(path)
#             if hasattr(self.training_env, "save"):
#                 self.training_env.save(path + "_vecnormalize.pkl")
#             with open(path + "_metadata.json", "w", encoding="utf-8") as f:
#                 json.dump(
#                     {
#                         "timesteps": int(self.num_timesteps),
#                         "episodes": int(self.episode_count),
#                         "best_reward": float(self.best_reward),
#                         "timestamp": datetime.now().isoformat(),
#                     },
#                     f,
#                     indent=2,
#                 )
#             print(f"Checkpoint saved: {path}")
#         return True


# def _linear_lr_schedule(initial_lr: float, final_lr: float):
#     """Linear decay from initial_lr (step 0) to final_lr (step total_timesteps)."""
#     def schedule(progress_remaining: float) -> float:
#         return final_lr + (initial_lr - final_lr) * progress_remaining
#     return schedule


# def train_patch_policy(
#     total_timesteps: int = 500000,
#     save_path: str = "patch_policy_models",
#     checkpoint_freq: int = 2000,
#     NUM_ENVS: int = 8,
#     resume_from: Optional[str] = None,
#     domain_randomize: bool = False,
#     norm_reward: bool = False,
#     base_reset_type: str = "rl_random_static",
#     use_base_done_termination: bool = False,
#     patch_only_mode: bool = True,
#     debug_print_every_n_steps: int = 0,
#     debug_print_episode_end: bool = True,
#     wandb_project: str = "patch_sempc_training",
#     wandb_entity: Optional[str] = None,
#     wandb_run_name: Optional[str] = None,
#     obs_mode: str = "grid", #grid
#     num_lidar_beams: int = 108,
#     ):
#     if not SB3_AVAILABLE:
#         raise RuntimeError("stable-baselines3 is required for training.")

#     os.makedirs(save_path, exist_ok=True)
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = os.path.join(save_path, f"run_{run_id}")
#     os.makedirs(run_dir, exist_ok=True)
#     tb_log_dir = os.path.join(run_dir, "tb")
#     os.makedirs(tb_log_dir, exist_ok=True)

#     wandb_run = None
#     if WANDB_AVAILABLE:
#         _cfg_dict = dataclasses.asdict(PatchEnvConfig())
#         _tr = 65536  # TARGET_ROLLOUT — must match the value set below at line ~390
#         wandb_run = wandb.init(
#             project=wandb_project,
#             config={
#                 "mode": "patch_solo",
#                 "num_envs": NUM_ENVS,
#                 "rollout_steps": max(512, _tr // max(1, NUM_ENVS)),
#                 "batch_size": max(256, _tr // 8),
#                 "reward_steer_bias_weight": _cfg_dict["reward_steer_bias_weight"],
#                 "reward_spin_weight": _cfg_dict["reward_spin_weight"],
#                 "spin_yawrate_threshold": _cfg_dict["spin_yawrate_threshold"],
#                 # "reward_speed_weight": _cfg_dict["reward_speed_weight"],
#                 "reward_size_weight": _cfg_dict["reward_size_weight"],
#                 "collision_penalty": _cfg_dict["collision_penalty"],
#                 "stuck_no_progress_steps": _cfg_dict["stuck_no_progress_steps"],
#                 "stuck_penalty": _cfg_dict["stuck_penalty"],
#                 "reward_progress_scale": _cfg_dict["reward_progress_scale"],
#                 "reward_crosstrack_weight": _cfg_dict["reward_crosstrack_weight"],
#                 "time_penalty_per_sec": _cfg_dict["time_penalty_per_sec"],
#                 "max_steps": _cfg_dict["max_steps"],
#                 "obs_dim": 86,
#                 "lr_initial": 3e-4,
#                 "lr_final": 1e-5,
#                 "lr_schedule": "linear",
#                 "patch_env_full": _cfg_dict,
#             },
#             sync_tensorboard=True,
#             save_code=True,
#         )

#     def _monitored(thunk):
#         return lambda: Monitor(thunk())

#     if NUM_ENVS > 1:
#         env = SubprocVecEnv(
#             [
#                 _monitored(make_patch_env(
#                     i,
#                     seed=42,
#                     domain_randomize=domain_randomize,
#                     base_reset_type=base_reset_type,
#                     obs_mode=obs_mode,
#                     num_lidar_beams=num_lidar_beams,
#                 ))
#                 for i in range(NUM_ENVS)
#             ],
#             start_method="fork",  # avoids PyCapsule datetime error from spawn/forkserver on Linux
#         )
#     else:
#         env = DummyVecEnv(
#             [
#                 _monitored(make_patch_env(
#                     0,
#                     seed=42,
#                     domain_randomize=domain_randomize,
#                     base_reset_type=base_reset_type,
#                     obs_mode=obs_mode,
#                     num_lidar_beams=num_lidar_beams,
#                 ))
#             ]
#         )

#     env = VecCheckNan(env, raise_exception=True)
#     env = VecNormalize(
#         env,
#         norm_obs=True,
#         norm_reward=norm_reward,
#         clip_obs=10.0,
#         clip_reward=5.0,
#         gamma=0.99,
#     )

#     callback = TrainingCallback(run_dir, checkpoint_freq, "PATCH")

#     _TARGET_ROLLOUT = 65536
#     rollout_steps = max(512, _TARGET_ROLLOUT // max(1, NUM_ENVS))  # keeps total GPU work fixed
#     batch_size = max(256, _TARGET_ROLLOUT // 8)                    # constant regardless of num_envs
#     # if total_rollout % batch_size != 0:
#     #     # Keep PPO minibatches exact to avoid partial batches.
#     #     batch_size = 256 if total_rollout % 256 == 0 else 128

#     # Detect GPU automatically, fall back to CPU if not available
#     _device = "cuda" if torch.cuda.is_available() else "cpu"
#     print(f"[train_patch_policy] Using device: {_device}")

#     if resume_from and os.path.exists(resume_from + ".zip"):
#         model = PPO.load(resume_from, env=env, device=_device)
#         # Restore VecNormalize stats so the policy sees the same obs scale it was trained on.
#         # Convention: vecnorm lives next to the model file as best_vecnormalize.pkl
#         import pickle as _pkl
#         _vn_path = os.path.join(os.path.dirname(resume_from), "best_vecnormalize.pkl")
#         if os.path.exists(_vn_path) and isinstance(env, VecNormalize):
#             with open(_vn_path, "rb") as _f:
#                 _saved_vn = _pkl.load(_f)
#             env.obs_rms = _saved_vn.obs_rms
#             env.ret_rms = _saved_vn.ret_rms
#             print(f"[resume] Loaded VecNormalize stats from {_vn_path}")
#         else:
#             print(f"[resume] WARNING: VecNormalize not found at {_vn_path} — obs stats reset")
#     else:
#         _lr_initial = 3e-4
#         _lr_final   = 1e-5
#         model = PPO(
#             "MlpPolicy",
#             env,
#             tensorboard_log=tb_log_dir,
#             learning_rate=_linear_lr_schedule(_lr_initial, _lr_final),
#             # learning_rate = 3e-4,
#             seed = 42,
#             n_steps=rollout_steps,
#             batch_size=batch_size,
#             n_epochs=10,
#             gamma=0.99,
#             gae_lambda=0.95,
#             clip_range=0.2,
#             ent_coef=0.001,  # 0.01 caused entropy explosion (std→32); spinning fixed by reward penalties now
#             # ent_coef=lambda progress: 0.008 * progress + 0.001,  # decays 0.009→0.001 over training
#             vf_coef=0.5,
#             max_grad_norm=0.5,
#             verbose=1,
#             use_sde=False,
#             device=_device,
#             policy_kwargs=dict(
#                 **(
#                     dict(
#                         features_extractor_class=HybridCNNExtractor,
#                         features_extractor_kwargs=dict(features_dim=256),
#                     ) if obs_mode == "grid" else {}
#                 ),
#                 net_arch=dict(pi=[64, 64], vf=[64, 64]) if obs_mode == "grid" else dict(pi=[256, 256], vf=[256, 256]),
#             )
#         )

#     save_best_cb = SaveBestWithVecNormalize(save_dir=run_dir, check_freq=rollout_steps)
#     callbacks = [callback, save_best_cb]

#     model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
#     model.save(os.path.join(run_dir, "final_model"))
#     env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
#     env.close()
#     if WANDB_AVAILABLE and wandb_run is not None:
#         wandb.finish()
#     return model


# # def train_agent_policy(
# #     total_timesteps: int = 1000000,
# #     save_path: str = "agent_policy_models",
# #     checkpoint_freq: int = 2000,
# #     NUM_ENVS: int = 4,
# #     resume_from: Optional[str] = None,
# #     patch_checkpoint_path: str = "",
# #     patch_vecnorm_path: str = "",
# #     norm_reward: bool = True,
# #     patch_env=None,
# #     # base_reset_type: str = "rl_random_static",
# #     wandb_project: str = "agent_sempc_training",
# #     wandb_entity: Optional[str] = None,
# #     wandb_run_name: Optional[str] = None, ):
# #     if not SB3_AVAILABLE:
# #         raise RuntimeError("stable-baselines3 is required for training.")

# #     os.makedirs(save_path, exist_ok=True)
# #     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     run_dir = os.path.join(save_path, f"run_{run_id}")
# #     os.makedirs(run_dir, exist_ok=True)
# #     tb_log_dir = os.path.join(run_dir, "tb")
# #     os.makedirs(tb_log_dir, exist_ok=True)

# #     wandb_run = None
# #     if WANDB_AVAILABLE:
# #         wandb_run = wandb.init(
# #             project=wandb_project,
# #             sync_tensorboard=True,
# #             save_code=True,
# #         )

# #     def _monitored(thunk):
# #         return lambda: Monitor(thunk())

# #     if patch_env is None and patch_checkpoint_path:
# #         patch_env = PatchEnv(PatchEnvConfig())
# #         patch_env.patch_policy = PPO.load(patch_checkpoint_path)
# #         patch_env.patch_policy.policy.set_training_mode(False)
# #         if patch_vecnorm_path and os.path.exists(patch_vecnorm_path):
# #             from stable_baselines3.common.vec_env import VecNormalize, DummyVecEnv
# #             import gymnasium as gym
# #             from gymnasium import spaces
# #             class _DummyPatchEnv(gym.Env):
# #                 observation_space = spaces.Box(-np.inf, np.inf, shape=(11,), dtype=np.float32)
# #                 action_space = spaces.Box(
# #                     low=np.array([-0.4189, 0.5, 0.325, 0.25], dtype=np.float32),
# #                     high=np.array([0.4189, 10.0, 2.5, 1.5], dtype=np.float32),
# #                 )
# #                 def reset(self, **kw): return np.zeros(11, dtype=np.float32), {}
# #                 def step(self, a): return np.zeros(11, dtype=np.float32), 0.0, False, False, {}
# #             patch_env.patch_vecnorm = VecNormalize.load(patch_vecnorm_path, DummyVecEnv([_DummyPatchEnv]))
# #             patch_env.patch_vecnorm.training = False
# #             patch_env.patch_vecnorm.norm_reward = False
# #         else:
# #             patch_env.patch_vecnorm = None

# #     # True parameter sharing: each AgentEnv is shared by two AgentView slots.
# #     # Slots come in consecutive pairs [view0, view1, view0, view1, ...] so that
# #     # view0 always submits its action first and view1 triggers the physics step.
# #     env_fns = []
# #     for i in range(NUM_ENVS):
# #         fn0, fn1 = make_agent_views(i, seed=42, patch_env=patch_env)
# #         env_fns.append(_monitored(fn0))
# #         env_fns.append(_monitored(fn1))

# #     # DummyVecEnv required: AgentView pairs share state — SubprocVecEnv would
# #     # fork the process and break the shared reference between view0 and view1.
# #     env = DummyVecEnv(env_fns)

# #     env = VecCheckNan(env, raise_exception=True)
# #     env = VecNormalize(
# #         env,
# #         norm_obs=True,
# #         norm_reward=norm_reward,
# #         clip_obs=10.0,
# #         clip_reward=5.0,
# #         gamma=0.99,
# #     )

# #     callback = TrainingCallback(run_dir, checkpoint_freq, "AGENT")

# #     rollout_steps = 2048
# #     # 2*NUM_ENVS slots (two AgentView per shared env)
# #     n_slots = 2 * NUM_ENVS
# #     batch_size = n_slots * rollout_steps // 4

# #     if resume_from and os.path.exists(resume_from + ".zip"):
# #         model = PPO.load(resume_from, env=env)
# #     else:
# #         model = PPO(
# #             MAPPOPolicy,   # CTDE: decentralized actor (obs[:12]), centralized critic (obs[:24])
# #             env,
# #             tensorboard_log=tb_log_dir,
# #             learning_rate=3e-4,
# #             seed=42,
# #             n_steps=rollout_steps,
# #             batch_size=batch_size,
# #             n_epochs=10,
# #             gamma=0.99,
# #             gae_lambda=0.95,
# #             clip_range=0.2,
# #             ent_coef=0.01,
# #             vf_coef=0.5,
# #             max_grad_norm=0.5,
# #             verbose=1,
# #             use_sde=False,
# #             # net_arch is handled inside MAPPOPolicy._build_mlp_extractor
# #         )

# #     save_best_cb = SaveBestWithVecNormalize(save_dir=run_dir, check_freq=rollout_steps)
# #     callbacks = [callback, save_best_cb]

# #     model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=True)
# #     model.save(os.path.join(run_dir, "final_model"))
# #     env.save(os.path.join(run_dir, "final_vecnormalize.pkl"))
# #     env.close()
# #     if WANDB_AVAILABLE and wandb_run is not None:
# #         wandb.finish()
# #     return model


# # def train_joint_policy(
# #     total_timesteps: int = 2_000_000,
# #     save_path: str = "joint_policy_models",
# #     checkpoint_freq: int = 10000,
# #     NUM_ENVS: int = 4,
# #     phase_steps: int = 4096,
# #     resume_patch: str = "",
# #     resume_agents: str = "",
# #     norm_reward: bool = True,
# #     wandb_project: str = "joint_sempc_training", ):
# #     """Train patch + agent policies jointly with alternating MAPPO.

# #     Each iteration alternates between:
# #       1. Patch training (phase_steps): agents use frozen snapshot of agent policy.
# #       2. Agent training (phase_steps): patch uses frozen snapshot of patch policy.

# #     Both policies co-evolve — the patch learns to keep agents inside;
# #     agents learn to follow whatever pace the patch sets.
# #     """
# #     if not SB3_AVAILABLE:
# #         raise RuntimeError("stable-baselines3 is required.")

# #     os.makedirs(save_path, exist_ok=True)
# #     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
# #     run_dir = os.path.join(save_path, f"run_{run_id}")
# #     os.makedirs(run_dir, exist_ok=True)

# #     wandb_run = None
# #     if WANDB_AVAILABLE:
# #         wandb_run = wandb.init(
# #             project=wandb_project,
# #             sync_tensorboard=True,
# #             save_code=True,
# #         )

# #     def _mon(thunk):
# #         return lambda: Monitor(thunk())

# #     # Build separate JointEnv sets for each training phase to avoid state collision.
# #     # patch_shared_envs[i] is used for patch training only.
# #     # agent_shared_envs[i] is used for agent training only.
# #     patch_shared_envs, patch_fns = [], []
# #     agent_shared_envs, agent_fns = [], []

# #     for i in range(NUM_ENVS):
# #         env_p, pf, _, _ = make_joint_views(i, seed=42)
# #         patch_shared_envs.append(env_p)
# #         patch_fns.append(_mon(pf))

# #         env_a, _, af0, af1 = make_joint_views(i + NUM_ENVS, seed=42)
# #         agent_shared_envs.append(env_a)
# #         agent_fns.extend([_mon(af0), _mon(af1)])

# #     # DummyVecEnv required: shared env state cannot cross process boundaries.
# #     patch_vec = DummyVecEnv(patch_fns)
# #     agent_vec = DummyVecEnv(agent_fns)

# #     patch_vec = VecCheckNan(patch_vec, raise_exception=True)
# #     agent_vec = VecCheckNan(agent_vec, raise_exception=True)
# #     patch_vec = VecNormalize(patch_vec, norm_obs=True, norm_reward=norm_reward,
# #                               clip_obs=10.0, clip_reward=10.0, gamma=0.99)
# #     agent_vec = VecNormalize(agent_vec, norm_obs=True, norm_reward=norm_reward,
# #                               clip_obs=10.0, clip_reward=5.0, gamma=0.99)

# #     rollout_steps = 2048

# #     # --- Patch PPO: standard MlpPolicy (15D obs → 4D action) ---
# #     if resume_patch and os.path.exists(resume_patch + ".zip"):
# #         patch_ppo = PPO.load(resume_patch, env=patch_vec)
# #     else:
# #         patch_ppo = PPO(
# #             "MlpPolicy", patch_vec,
# #             tensorboard_log=os.path.join(run_dir, "tb_patch"),
# #             learning_rate=3e-4, seed=42,
# #             n_steps=rollout_steps,
# #             batch_size=NUM_ENVS * rollout_steps // 4,
# #             n_epochs=10, gamma=0.99, gae_lambda=0.95,
# #             clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
# #             max_grad_norm=0.5, verbose=1,
# #             policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
# #         )

# #     # --- Agent PPO: CTDE MAPPOPolicy (24D obs → 2D action, parameter shared) ---
# #     n_agent_slots = 2 * NUM_ENVS
# #     if resume_agents and os.path.exists(resume_agents + ".zip"):
# #         agent_ppo = PPO.load(resume_agents, env=agent_vec)
# #     else:
# #         agent_ppo = PPO(
# #             MAPPOPolicy, agent_vec,
# #             tensorboard_log=os.path.join(run_dir, "tb_agents"),
# #             learning_rate=3e-4, seed=42,
# #             n_steps=rollout_steps,
# #             batch_size=n_agent_slots * rollout_steps // 4,
# #             n_epochs=10, gamma=0.99, gae_lambda=0.95,
# #             clip_range=0.2, ent_coef=0.01, vf_coef=0.5,
# #             max_grad_norm=0.5, verbose=1,
# #         )

# #     patch_cb = TrainingCallback(run_dir, checkpoint_freq, "JOINT-PATCH")
# #     agent_cb = TrainingCallback(run_dir, checkpoint_freq, "JOINT-AGENT")

# #     # ------------------------------------------------------------------
# #     # Alternating training loop
# #     # ------------------------------------------------------------------
# #     total_steps = 0
# #     iteration = 0

# #     while total_steps < total_timesteps:
# #         iteration += 1
# #         print(f"\n[Joint iter {iteration}] steps={total_steps}/{total_timesteps}")

# #         # --- Phase 1: Train patch — agents use frozen snapshot ---
# #         _agent_snap = agent_ppo.policy
# #         _agent_vec_norm = agent_vec  # VecNormalize for un-normalising agent obs

# #         for env in patch_shared_envs:
# #             def _agent_fn(obs0, obs1, _pol=_agent_snap, _vn=_agent_vec_norm):
# #                 # obs0/obs1 are raw (un-normalised) — normalise before prediction
# #                 obs0_n = _vn.normalize_obs(obs0[np.newaxis])[0]
# #                 obs1_n = _vn.normalize_obs(obs1[np.newaxis])[0]
# #                 a0, _ = _pol.predict(obs0_n[np.newaxis], deterministic=True)
# #                 a1, _ = _pol.predict(obs1_n[np.newaxis], deterministic=True)
# #                 return a0[0], a1[0]
# #             env._agent_action_fn = _agent_fn

# #         patch_ppo.learn(
# #             phase_steps, callback=patch_cb,
# #             reset_num_timesteps=(iteration == 1), progress_bar=False,
# #         )
# #         total_steps += phase_steps

# #         # --- Phase 2: Train agents — patch uses frozen snapshot ---
# #         _patch_snap = patch_ppo.policy
# #         _patch_vec_norm = patch_vec  # VecNormalize for patch obs

# #         for env in agent_shared_envs:
# #             def _patch_fn(obs, _pol=_patch_snap, _vn=_patch_vec_norm):
# #                 obs_n = _vn.normalize_obs(obs[np.newaxis])[0]
# #                 action, _ = _pol.predict(obs_n[np.newaxis], deterministic=True)
# #                 return action[0]
# #             env._patch_action_fn = _patch_fn

# #         agent_ppo.learn(
# #             phase_steps, callback=agent_cb,
# #             reset_num_timesteps=(iteration == 1), progress_bar=False,
# #         )
# #         total_steps += phase_steps

# #         # Log combined metrics to wandb once per iteration
# #         if WANDB_AVAILABLE and wandb_run is not None:
# #             patch_buf = patch_ppo.ep_info_buffer
# #             agent_buf = agent_ppo.ep_info_buffer
# #             log = {"joint/total_steps": total_steps, "joint/iteration": iteration}
# #             if patch_buf:
# #                 log["joint/patch_ep_rew_mean"] = float(np.mean([e["r"] for e in patch_buf]))
# #                 log["joint/patch_ep_len_mean"] = float(np.mean([e["l"] for e in patch_buf]))
# #             if agent_buf:
# #                 log["joint/agent_ep_rew_mean"] = float(np.mean([e["r"] for e in agent_buf]))
# #                 log["joint/agent_ep_len_mean"] = float(np.mean([e["l"] for e in agent_buf]))
# #             wandb.log(log, step=total_steps)

# #     # --- Save ---
# #     patch_ppo.save(os.path.join(run_dir, "patch_final"))
# #     agent_ppo.save(os.path.join(run_dir, "agents_final"))
# #     patch_vec.save(os.path.join(run_dir, "patch_vecnorm_final.pkl"))
# #     agent_vec.save(os.path.join(run_dir, "agents_vecnorm_final.pkl"))
# #     patch_vec.close()
# #     agent_vec.close()
# #     if WANDB_AVAILABLE and wandb_run is not None:
# #         wandb.finish()
# #     print(f"Joint training complete. Models saved to {run_dir}")
# #     return patch_ppo, agent_ppo


# def _save_frozen_agent_npz(policy, vec_norm, path):
#     """Extract agent policy weights + VecNormalize stats as numpy .npz.

#     Only plain numpy arrays cross the process boundary (via filesystem),
#     so SubprocVecEnv workers never need to pickle PyTorch objects.
#     """
#     sd = {k: v.detach().cpu().numpy() for k, v in policy.state_dict().items()}

#     save_kw = {}
#     for i in range(0, 100, 2):
#         wk = f"mlp_extractor.policy_net.{i}.weight"
#         bk = f"mlp_extractor.policy_net.{i}.bias"
#         if wk not in sd:
#             break
#         save_kw[f"pi_w_{i}"] = sd[wk]
#         save_kw[f"pi_b_{i}"] = sd[bk]
#     save_kw["act_w"] = sd["action_net.weight"]
#     save_kw["act_b"] = sd["action_net.bias"]

#     act_fn = getattr(policy, "activation_fn", None)
#     save_kw["use_tanh"] = np.array(
#         act_fn is nn.Tanh if act_fn is not None else True
#     )

#     if vec_norm is not None and hasattr(vec_norm, "obs_rms"):
#         save_kw["vnorm_mean"] = vec_norm.obs_rms.mean.copy()
#         save_kw["vnorm_var"] = vec_norm.obs_rms.var.copy()
#         save_kw["vnorm_clip"] = np.array(vec_norm.clip_obs)
#         save_kw["vnorm_eps"] = np.array(vec_norm.epsilon)

#     np.savez(path, **save_kw)


# def train_joint_sb3(
#     total_timesteps: int = 2_000_000,
#     save_path: str = "joint_sb3_models",
#     checkpoint_freq: int = 2000,
#     num_envs: int = 8,
#     num_cpus: int = 4,
#     phase_steps: int = 8192,
#     patch_solo_timesteps: int = 10_000_000,
#     resume_patch: str = "",
#     resume_agents: str = "",
#     norm_reward: bool = False,
#     agent_only: bool = False,
#     pretrained_patch_path: str = "",
#     pretrained_patch_vecnorm: str = "",
#     wandb_project: str = "joint_sb3_training",
#     wandb_entity: Optional[str] = None,
#     wandb_run_name: Optional[str] = None, ):
#     """Alternating co-evolution using PettingZoo + SuperSuit for agent parallelization.

#     Phase 0 (optional): Patch-car solo on f110 (PatchCarEnv, same 11D obs as PatchEnv) for
#     ``patch_solo_timesteps`` — set to 0 to skip and start directly with joint patch training.

#     Phase A (each iteration): Train patch policy on JointPatchView (2 f110 cars; frozen
#     agent policy from .npz when available). SubprocVecEnv for true parallelism.

#     Phase B: Train single agent policy via SuperSuit (1 PettingZoo agent × num_envs parallel).
#     Patch inside uses a frozen snapshot of the patch policy.
#     """
#     if not SB3_AVAILABLE:
#         raise RuntimeError("stable-baselines3 is required.")
#     if not SUPERSUIT_AVAILABLE:
#         raise RuntimeError("supersuit is required: pip install supersuit")

#     # === JOINT TRAINING DISABLED — only agent_only path is supported ===
#     # Use train_patch_policy() to produce the frozen patch checkpoint, then
#     # call this function with agent_only=True and pretrained_patch_path=...
#     if not agent_only:
#         raise NotImplementedError(
#             "Joint training (Phase A/B alternation) is disabled. "
#             "Run with --agent_only=True and a pretrained_patch_path."
#         )

#     try:
#         from .pz_env import AgentEnv
#     except ImportError:
#         from envs.pz_env import AgentEnv

#     from stable_baselines3.common.vec_env import VecMonitor

#     os.makedirs(save_path, exist_ok=True)
#     run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = os.path.join(save_path, f"run_{run_id}")
#     os.makedirs(run_dir, exist_ok=True)

#     tb_log_dir = os.path.join(run_dir, "tb")
#     os.makedirs(tb_log_dir, exist_ok=True)

#     wandb_run = None
#     if WANDB_AVAILABLE:
#         _patch_cfg = dataclasses.asdict(PatchEnvConfig())
#         wandb_run = wandb.init(
#             project=wandb_project,
#             entity=wandb_entity,
#             name=wandb_run_name or f"joint_sb3_{run_id}",
#             config={
#                 "mode": "joint_sb3",
#                 "total_timesteps": total_timesteps,
#                 "patch_solo_timesteps": patch_solo_timesteps,
#                 "num_envs": num_envs,
#                 "num_cpus": num_cpus,
#                 "phase_steps": phase_steps,
#                 # key reward knobs — visible as flat fields in wandb run table
#                 "reward_steer_bias_weight": _patch_cfg["reward_steer_bias_weight"],
#                 "reward_spin_weight": _patch_cfg["reward_spin_weight"],
#                 "spin_yawrate_threshold": _patch_cfg["spin_yawrate_threshold"],
#                 "collision_penalty": _patch_cfg["collision_penalty"],
#                 "stuck_no_progress_steps": _patch_cfg["stuck_no_progress_steps"],
#                 "stuck_penalty": _patch_cfg["stuck_penalty"],
#                 "reward_progress_scale": _patch_cfg["reward_progress_scale"],
#                 "time_penalty_per_sec": _patch_cfg["time_penalty_per_sec"],
#                 "max_steps": _patch_cfg["max_steps"],
#                 # full config nested for completeness
#                 "patch_env_full": _patch_cfg,
#             },
#             sync_tensorboard=True,
#             save_code=True,
#             reinit=True,
#         )

#     _device = "cuda"

#     TARGET_ROLLOUT = 65536
#     patch_n_steps = max(512, TARGET_ROLLOUT // max(1, num_envs))
#     agent_n_steps = max(512, TARGET_ROLLOUT // max(1, num_envs))
#     patch_batch   = max(256, patch_n_steps * num_envs // 8)
#     agent_batch   = max(256, agent_n_steps * num_envs // 8)

#     min_phase = max(patch_n_steps * num_envs, agent_n_steps * num_envs)
#     if phase_steps < min_phase:
#         print(f"[WARNING] phase_steps={phase_steps} < {min_phase} "
#               f"(min to guarantee 1 PPO update per phase). Rounding up.")
#         phase_steps = min_phase
#     print(f"[joint_sb3] patch n_steps={patch_n_steps} batch={patch_batch} | "
#           f"agent n_steps={agent_n_steps} batch={agent_batch} | "
#           f"phase_steps={phase_steps} | device={_device}")

#     if agent_only:
#         # phase_steps is for alternating joint training; agent_only needs a compact rollout.
#         # With phase_steps=131072 and num_envs=16 the first PPO update requires 2M steps —
#         # the policy never updates within a typical 20M-step budget. Use 4096 instead.
#         agent_n_steps = 4096
#         agent_batch   = max(256, agent_n_steps * num_envs // 8)  # → 8192
#         print(f"[joint_sb3 agent_only] agent n_steps={agent_n_steps} batch={agent_batch}")

#     patch_dir = os.path.join(run_dir, "patch")
#     agent_dir = os.path.join(run_dir, "agents")
#     os.makedirs(patch_dir, exist_ok=True)
#     os.makedirs(agent_dir, exist_ok=True)

#     # ------------------------------------------------------------------
#     # Patch loading: agent_only mode loads the frozen patch from a
#     # pretrained checkpoint produced by train_patch_policy. Joint patch
#     # training (Phase 0 solo + Phase A on JointPatchView) is disabled.
#     # ------------------------------------------------------------------
#     # === JOINT TRAINING DISABLED — JointPatchView/JointEnv imports removed ===
#     # try:
#     #     from .ppo_policy import (
#     #         JointEnv, JointEnvConfig, JointPatchView, PatchCarEnv, PatchEnvConfig,
#     #     )
#     # except ImportError:
#     #     from envs.ppo_policy import (
#     #         JointEnv, JointEnvConfig, JointPatchView, PatchCarEnv, PatchEnvConfig,
#     #     )

#     # === JOINT TRAINING DISABLED — patch_cb/save_patch_best unused ===
#     # patch_cb = TrainingCallback(
#     #     save_dir=patch_dir,
#     #     save_freq=max(1, checkpoint_freq // num_envs),
#     #     policy_name="patch_sb3",
#     # )
#     # save_patch_best = SaveBestWithVecNormalize(save_dir=patch_dir, check_freq=patch_n_steps,
#     #                                            model_name="best_model")

#     # === JOINT TRAINING DISABLED — Phase 0 PatchCarEnv builder unused here ===
#     # def _make_patch_car():
#     #     cfg = PatchEnvConfig(split_mode=True, random_spawn=True)
#     #     env = PatchCarEnv(cfg)
#     #     env.reset()
#     #     return Monitor(env)

#     # === JOINT TRAINING DISABLED — JointPatchView builder ===
#     # def _make_joint_patch_env():
#     #     try:
#     #         print("[DEBUG] Creating JointEnv...")
#     #         shared = JointEnv(JointEnvConfig(render_mode=None))
#     #         print("[DEBUG] JointEnv created, calling reset...")
#     #         shared.reset()
#     #         print("[DEBUG] JointEnv reset successful")
#     #         print("[DEBUG] Creating JointPatchView...")
#     #         view = JointPatchView(shared)
#     #         print("[DEBUG] JointPatchView created")
#     #         return Monitor(view)
#     #     except Exception as e:
#     #         print(f"[ERROR] Failed to create JointEnv: {type(e).__name__}: {e}")
#     #         import traceback
#     #         traceback.print_exc()
#     #         raise

#     patch_ppo = None
#     # patch_car_vec = None
#     # _solo_obs_rms = None

#     # === JOINT TRAINING DISABLED — Phase 0 solo + Phase A patch training removed ===
#     # if not agent_only:
#     #     if patch_solo_timesteps > 0 and not resume_patch:
#     #         if num_envs > 1:
#     #             patch_car_vec = SubprocVecEnv(
#     #                 [_make_patch_car for _ in range(num_envs)],
#     #                 start_method="fork",
#     #             )
#     #         else:
#     #             patch_car_vec = DummyVecEnv([_make_patch_car])
#     #         patch_car_vec = VecMonitor(patch_car_vec)
#     #         if norm_reward:
#     #             patch_car_vec = VecNormalize(
#     #                 patch_car_vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
#     #             )
#     #         patch_ppo = PPO(
#     #             "MlpPolicy", patch_car_vec,
#     #             tensorboard_log=os.path.join(tb_log_dir, "patch_solo"),
#     #             n_steps=patch_n_steps, batch_size=patch_batch, n_epochs=10,
#     #             gamma=0.99, gae_lambda=0.95, clip_range=0.2,
#     #             ent_coef=0.05,
#     #             vf_coef=0.5, max_grad_norm=0.5,
#     #             device=_device, verbose=1,
#     #         )
#     #         print(f"[joint_sb3] Phase 0: PatchCarEnv solo f110 for {patch_solo_timesteps} steps")
#     #         patch_ppo.learn(
#     #             patch_solo_timesteps,
#     #             callback=[patch_cb, save_patch_best],
#     #             reset_num_timesteps=True,
#     #             progress_bar=False,
#     #         )
#     #         patch_ppo.save(os.path.join(patch_dir, "patch_after_solo"))
#     #         _solo_obs_rms = patch_car_vec.obs_rms if isinstance(patch_car_vec, VecNormalize) else None
#     #         if isinstance(patch_car_vec, VecNormalize):
#     #             patch_car_vec.save(os.path.join(patch_dir, "patch_after_solo_vecnormalize.pkl"))
#     #         patch_car_vec.close()
#     #         patch_car_vec = None
#     #
#     #     # TEMPORARY: Use DummyVecEnv for Phase A to see actual worker errors
#     #     # TODO: Switch back to SubprocVecEnv once JointEnv is verified stable
#     #     print(f"[DEBUG] Creating patch_vec with DummyVecEnv (num_envs={num_envs})")
#     #     # patch_vec = DummyVecEnv([_make_joint_patch_env for _ in range(num_envs)])
#     #     if num_envs > 1:
#     #         patch_vec = SubprocVecEnv(
#     #             [_make_joint_patch_env for _ in range(num_envs)],
#     #             start_method="fork",
#     #         )
#     #     else:
#     #         patch_vec = DummyVecEnv([_make_joint_patch_env])
#     #     patch_vec = VecMonitor(patch_vec)
#     #     if norm_reward:
#     #         patch_vec = VecNormalize(patch_vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)
#     #
#     #     if resume_patch:
#     #         patch_ppo = PPO.load(resume_patch, env=patch_vec)
#     #     elif patch_ppo is not None:
#     #         # Phase 0 trained on 267D (scalars + occupancy grid)
#     #         # Phase A now also provides 267D (same format) so obs spaces match
#     #         patch_ppo.set_env(patch_vec)
#     #         # Transfer obs normalization stats from Phase 0
#     #         if _solo_obs_rms is not None and isinstance(patch_vec, VecNormalize):
#     #             patch_vec.obs_rms = _solo_obs_rms
#     #     else:
#     #         patch_ppo = PPO(
#     #             "MlpPolicy", patch_vec,
#     #             tensorboard_log=os.path.join(tb_log_dir, "patch"),
#     #             n_steps=patch_n_steps, batch_size=patch_batch, n_epochs=10,
#     #             gamma=0.99, gae_lambda=0.95, clip_range=0.2,
#     #             ent_coef=0.05,
#     #             vf_coef=0.5, max_grad_norm=0.5,
#     #             device=_device, verbose=1,
#     #         )
#     # else:
#     # agent_only: load frozen patch from checkpoint, no patch_vec
#     if not pretrained_patch_path:
#         raise ValueError("agent_only=True requires pretrained_patch_path")

#     _frozen_patch_ppo = PPO.load(pretrained_patch_path)
#     _frozen_patch_vn = None
#     if pretrained_patch_vecnorm and os.path.exists(pretrained_patch_vecnorm):
#         import pickle
#         with open(pretrained_patch_vecnorm, "rb") as f:
#             _frozen_patch_vn = pickle.load(f)
#         _frozen_patch_vn.training = False
#         _frozen_patch_vn.norm_reward = False

#     print(f"[joint_sb3 agent_only] Loaded frozen patch from {pretrained_patch_path}")
#     patch_vec = None   # signal for final cleanup
#     patch_ppo = None

#     # ------------------------------------------------------------------
#     # Auto-detect the patch observation mode from the frozen checkpoint and
#     # configure JointEnv to match. The patch obs that JointEnv feeds the
#     # frozen policy MUST have the same dimensionality the policy was trained
#     # on, else VecNormalize.normalize_obs() crashes on a broadcast mismatch.
#     #   grid  mode → 86D  (22 scalars + 8x8 occupancy grid)
#     #   lidar mode → 7 + num_lidar_beams  (e.g. 115D for 108 beams)
#     # ------------------------------------------------------------------
#     _GRID_PATCH_OBS_DIM = 86
#     _patch_obs_dim = int(np.prod(_frozen_patch_ppo.observation_space.shape))
#     if _frozen_patch_vn is not None:
#         _vn_dim = int(_frozen_patch_vn.obs_rms.mean.shape[0])
#         if _vn_dim != _patch_obs_dim:
#             raise ValueError(
#                 f"Patch checkpoint / VecNormalize obs-dim mismatch: "
#                 f"policy expects {_patch_obs_dim}D but VecNormalize is {_vn_dim}D. "
#                 f"They must come from the same patch training run."
#             )
#     if _patch_obs_dim == _GRID_PATCH_OBS_DIM:
#         _patch_obs_mode, _patch_n_beams = "grid", 108
#     else:
#         _patch_obs_mode, _patch_n_beams = "lidar", _patch_obs_dim - 7
#         if _patch_n_beams <= 0:
#             raise ValueError(
#                 f"Frozen patch obs-dim {_patch_obs_dim} is neither grid ({_GRID_PATCH_OBS_DIM}D) "
#                 f"nor a valid lidar layout (7 scalars + N beams)."
#             )
#     _patch_env_config = {"obs_mode": _patch_obs_mode, "num_lidar_beams": _patch_n_beams}
#     print(f"[joint_sb3 agent_only] Patch obs-mode='{_patch_obs_mode}' "
#           f"(obs_dim={_patch_obs_dim}, num_lidar_beams={_patch_n_beams}) — "
#           f"JointEnv configured to match.")

#     # ------------------------------------------------------------------
#     # Agent vec env — built ONCE, never rebuilt.
#     # Uses a mutable policy holder so the frozen patch policy can be
#     # swapped in-place each iteration without tearing down envs.
#     # This avoids file-descriptor leaks from repeated subprocess creation
#     # and preserves VecNormalize running stats across iterations.
#     # ------------------------------------------------------------------
#     class _MutablePatchPolicy:
#         """In-process mutable wrapper: all AgentEnvs share this reference."""
#         def __init__(self):
#             self._fn = None
#         def __call__(self, obs):
#             if self._fn is not None:
#                 return self._fn(obs)
#             return np.array([0.0, 3.0, 3.0, 2.5], dtype=np.float32)
#         def update(self, fn):
#             self._fn = fn

#     _patch_holder = _MutablePatchPolicy()

#     from supersuit.vector import MarkovVectorEnv, ConcatVecEnv
#     from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

#     def _make_markov_vec():
#         return MarkovVectorEnv(
#             AgentEnv(env_config=_patch_env_config, patch_policy=_patch_holder)
#         )

#     agent_raw = ConcatVecEnv([_make_markov_vec for _ in range(num_envs)])
#     agent_raw = SB3VecEnvWrapper(agent_raw)
#     agent_vec = VecMonitor(agent_raw)
#     # Always apply obs normalisation (zero-mean, unit-var → helps policy gradient).
#     # NEVER normalise rewards: clip_reward=10 would compress +0.1 inside_reward to ≈ 0
#     # and make repulsion (-100) and collision (-3000) indistinguishable at the clip wall.
#     # The reward structure is intentional — PPO + max_grad_norm=0.5 handles raw magnitudes.
#     agent_vec = VecNormalize(agent_vec, norm_obs=True, norm_reward=False, clip_obs=10.0)

#     agent_cb = TrainingCallback(
#         save_dir=agent_dir,
#         save_freq=max(1, checkpoint_freq // num_envs),
#         policy_name="agent_sb3",
#     )
#     save_agent_best = SaveBestWithVecNormalize(save_dir=agent_dir, check_freq=agent_n_steps,
#                                                model_name="best_model")

#     _agent_tb_log = os.path.join(tb_log_dir, "agent")
#     if resume_agents:
#         agent_ppo = PPO.load(resume_agents, env=agent_vec,
#                              tensorboard_log=_agent_tb_log)
#     else:
#         agent_ppo = PPO(
#             "MlpPolicy", agent_vec,
#             tensorboard_log=_agent_tb_log,
#             n_steps=agent_n_steps, batch_size=agent_batch, n_epochs=10,
#             gamma=0.99, gae_lambda=0.95, clip_range=0.2,
#             ent_coef=0.005, vf_coef=0.5, max_grad_norm=0.5,
#             policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
#             device=_device, verbose=1,
#         )

#     # === JOINT TRAINING DISABLED — frozen-agent .npz handoff was used by Phase A ===
#     # _frozen_agent_npz = os.path.join(run_dir, "_frozen_agent.npz")

#     # ------------------------------------------------------------------
#     # Agent-only training: frozen patch policy is injected once, then PPO
#     # learns the agent for total_timesteps. No alternation, no Phase A.
#     # ------------------------------------------------------------------
#     # Build frozen patch closure and inject once — no alternation needed
#     _frozen_pol = _frozen_patch_ppo.policy
#     _frozen_vn  = _frozen_patch_vn

#     def _frozen_patch_fn(obs, _pol=_frozen_pol, _vn=_frozen_vn):
#         obs_in = obs[np.newaxis]
#         if _vn is not None:
#             obs_in = _vn.normalize_obs(obs_in)
#         action, _ = _pol.predict(obs_in, deterministic=True)
#         return action[0]

#     _patch_holder.update(_frozen_patch_fn)
#     print(f"[joint_sb3 agent_only] Frozen patch injected. Training agents for {total_timesteps} steps.")

#     agent_ppo.learn(
#         total_timesteps,
#         callback=[agent_cb, save_agent_best],
#         reset_num_timesteps=not bool(resume_agents),
#         progress_bar=False,
#     )

#     # Save final agent only
#     agent_ppo.save(os.path.join(agent_dir, "final_model"))
#     if isinstance(agent_vec, VecNormalize):
#         agent_vec.save(os.path.join(agent_dir, "final_vecnormalize.pkl"))
#     agent_vec.close()
#     if WANDB_AVAILABLE and wandb_run is not None:
#         wandb.finish()
#     print(f"agent_only training complete. Models saved to {run_dir}")
#     return None, agent_ppo

#     # === JOINT TRAINING DISABLED — Phase A/B alternating loop ===
#     # else:
#     #     for iteration in range(1, iterations + 1):
#     #         print(f"\n=== Iteration {iteration}/{iterations} ===")
#     #
#     #         # ------------------------------------------------------------------
#     #         # Inject frozen agent policy into patch SubprocVecEnv workers.
#     #         # Workers load a .npz file (only a string crosses the pipe).
#     #         # ------------------------------------------------------------------
#     #         if _has_trained_agents:
#     #             patch_vec.env_method("load_frozen_agent_npz", _frozen_agent_npz)
#     #         else:
#     #             patch_vec.env_method("load_frozen_agent_npz", None)
#     #
#     #         # Phase 1: train patch (do not reset SB3 timestep schedule on iter 1 if solo warmup ran)
#     #         patch_ppo.learn(
#     #             phase_steps,
#     #             callback=[patch_cb, save_patch_best],
#     #             reset_num_timesteps=(iteration == 1 and not solo_done),
#     #             progress_bar=False,
#     #         )
#     #         total_steps     += phase_steps
#     #         patch_ppo_steps += phase_steps
#     #
#     #         # Phase 2: train agent with frozen patch policy snapshot.
#     #         # Just swap the policy inside the mutable holder — no env rebuild needed.
#     #         _patch_snap = patch_ppo.policy
#     #         _patch_vn = patch_vec if isinstance(patch_vec, VecNormalize) else None
#     #
#     #         def _frozen_patch(obs: np.ndarray, _pol=_patch_snap, _vn=_patch_vn) -> np.ndarray:
#     #             obs_in = obs[np.newaxis]
#     #             if _vn is not None:
#     #                 obs_in = _vn.normalize_obs(obs_in)
#     #             action, _ = _pol.predict(obs_in, deterministic=True)
#     #             return action[0]
#     #
#     #         _patch_holder.update(_frozen_patch)
#     #
#     #         agent_ppo.learn(
#     #             phase_steps,
#     #             callback=[agent_cb, save_agent_best],
#     #             reset_num_timesteps=(iteration == 1),
#     #             progress_bar=False,
#     #         )
#     #         total_steps     += phase_steps
#     #         agent_ppo_steps += phase_steps
#     #
#     #         # Save frozen agent weights for next iteration's patch phase
#     #         _agent_vn = agent_vec if isinstance(agent_vec, VecNormalize) else None
#     #         _save_frozen_agent_npz(agent_ppo.policy, _agent_vn, _frozen_agent_npz)
#     #         _has_trained_agents = True
#     #
#     #         if WANDB_AVAILABLE and wandb_run is not None:
#     #             patch_buf = patch_ppo.ep_info_buffer
#     #             agent_buf = agent_ppo.ep_info_buffer
#     #             log = {
#     #                 "joint_sb3/total_steps":     total_steps,
#     #                 "joint_sb3/patch_ppo_steps": patch_ppo_steps,
#     #                 "joint_sb3/agent_ppo_steps": agent_ppo_steps,
#     #                 "joint_sb3/iteration":       iteration,
#     #             }
#     #             if patch_buf:
#     #                 log["joint_sb3/patch_ep_rew_mean"] = float(np.mean([e["r"] for e in patch_buf]))
#     #                 log["joint_sb3/patch_ep_len_mean"] = float(np.mean([e["l"] for e in patch_buf]))
#     #             if agent_buf:
#     #                 log["joint_sb3/agent_ep_rew_mean"] = float(np.mean([e["r"] for e in agent_buf]))
#     #                 log["joint_sb3/agent_ep_len_mean"] = float(np.mean([e["l"] for e in agent_buf]))
#     #             wandb.log(log, step=total_steps)
#     #
#     #     # --- Save final ---
#     #     patch_ppo.save(os.path.join(patch_dir, "final_model"))
#     #     agent_ppo.save(os.path.join(agent_dir, "final_model"))
#     #     if isinstance(patch_vec, VecNormalize):
#     #         patch_vec.save(os.path.join(patch_dir, "final_vecnormalize.pkl"))
#     #     if isinstance(agent_vec, VecNormalize):
#     #         agent_vec.save(os.path.join(agent_dir, "final_vecnormalize.pkl"))
#     #     patch_vec.close()
#     #     agent_vec.close()
#     #     if os.path.exists(_frozen_agent_npz):
#     #         os.remove(_frozen_agent_npz)
#     #     if WANDB_AVAILABLE and wandb_run is not None:
#     #         wandb.finish()
#     #     print(f"joint_sb3 training complete. Models saved to {run_dir}")
#     #     return patch_ppo, agent_ppo


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Train patch or agent policy with modular envs stack.")
#     parser.add_argument("--mode", type=str, default="patch",
#                         choices=["patch", "agent", "joint", "joint_sb3"],
#                         help="Which policy to train")
#     parser.add_argument("--timesteps", type=int, default=500000)
#     parser.add_argument("--checkpoint-freq", type=int, default=10000)
#     parser.add_argument("--num-envs", type=int, default=16)
#     parser.add_argument("--save-path", type=str, default="patch_policy_models")
#     parser.add_argument("--resume", type=str, default=None)
#     parser.add_argument("--domain-randomize", action="store_true")
#     parser.add_argument("--no-norm-reward", action="store_true")
#     parser.add_argument("--base-reset-type", type=str, default="rl_random_static")
#     # Agent-mode args
#     parser.add_argument("--patch-checkpoint", type=str, default="",
#                         help="Path to frozen patch PPO checkpoint (without .zip), required for --mode agent")
#     parser.add_argument("--patch-vecnorm", type=str, default="",
#                         help="Path to VecNormalize .pkl for the patch policy, required for --mode agent")
#     # Joint-mode args
#     parser.add_argument("--phase-steps", type=int, default=131072,
#                         help="Steps per training phase per policy (joint mode)")
#     parser.add_argument(
#         "--patch-solo-timesteps",
#         type=int,
#         default=10_000_000,
#         help="joint_sb3: f110 PatchCarEnv solo warmup before alternating (0 = skip)",
#     )
#     parser.add_argument("--resume-patch", type=str, default="",
#                         help="Resume patch policy from checkpoint (joint mode)")
#     parser.add_argument("--resume-agents", type=str, default="",
#                         help="Resume agent policy from checkpoint (joint mode)")
#     parser.add_argument("--wandb-project", type=str, default="patch_sempc_training")
#     parser.add_argument("--wandb-entity", type=str, default=None)
#     parser.add_argument("--wandb-run-name", type=str, default=None)
#     # joint_sb3-specific
#     parser.add_argument("--num-cpus", type=int, default=4,
#                         help="CPU workers for SuperSuit concat_vec_envs_v1 (joint_sb3 mode)")
#     parser.add_argument("--agent-only", action="store_true",
#                         help="Train agents only against a frozen patch (no Phase A)")
#     parser.add_argument("--pretrained-patch-path", type=str, default="",
#                         help="Path to frozen patch .zip checkpoint (agent_only mode)")
#     parser.add_argument("--pretrained-patch-vecnorm", type=str, default="",
#                         help="Path to patch VecNormalize .pkl (agent_only mode, optional)")
#     args = parser.parse_args()

#     if args.mode == "joint_sb3":
#         train_joint_sb3(
#             total_timesteps=args.timesteps,
#             save_path=args.save_path if args.save_path != "patch_policy_models" else "joint_sb3_models",
#             checkpoint_freq=args.checkpoint_freq,
#             num_envs=args.num_envs,
#             num_cpus=args.num_cpus,
#             phase_steps=args.phase_steps,
#             patch_solo_timesteps=args.patch_solo_timesteps,
#             resume_patch=args.resume_patch,
#             resume_agents=args.resume_agents,
#             norm_reward=not args.no_norm_reward,
#             agent_only=args.agent_only,
#             pretrained_patch_path=args.pretrained_patch_path,
#             pretrained_patch_vecnorm=args.pretrained_patch_vecnorm,
#             wandb_project=args.wandb_project,
#             wandb_entity=args.wandb_entity,
#             wandb_run_name=args.wandb_run_name,
#         )
#     # === JOINT TRAINING DISABLED — joint and agent modes routed through commented-out functions ===
#     # elif args.mode == "joint":
#     #     train_joint_policy(
#     #         total_timesteps=args.timesteps,
#     #         save_path=args.save_path if args.save_path != "patch_policy_models" else "joint_policy_models",
#     #         checkpoint_freq=args.checkpoint_freq,
#     #         NUM_ENVS=args.num_envs,
#     #         phase_steps=args.phase_steps,
#     #         resume_patch=args.resume_patch,
#     #         resume_agents=args.resume_agents,
#     #         norm_reward=not args.no_norm_reward,
#     #         wandb_project=args.wandb_project,
#     #     )
#     # elif args.mode == "agent":
#     #     train_agent_policy(
#     #         total_timesteps=args.timesteps,
#     #         save_path=args.save_path if args.save_path != "patch_policy_models" else "agent_policy_models",
#     #         checkpoint_freq=args.checkpoint_freq,
#     #         NUM_ENVS=args.num_envs,
#     #         resume_from=args.resume,
#     #         patch_checkpoint_path=args.patch_checkpoint,
#     #         patch_vecnorm_path=args.patch_vecnorm,
#     #         norm_reward=not args.no_norm_reward,
#     #         wandb_project=args.wandb_project,
#     #         wandb_entity=args.wandb_entity,
#     #         wandb_run_name=args.wandb_run_name,
#     #     )
#     elif args.mode in ("joint", "agent"):
#         raise SystemExit(
#             f"--mode {args.mode} is disabled. "
#             "Use --mode patch to train the patch, then --mode joint_sb3 --agent-only "
#             "with --pretrained-patch-path to train agents against a frozen patch."
#         )
#     else:
#         train_patch_policy(
#             total_timesteps=args.timesteps,
#             save_path=args.save_path,
#             checkpoint_freq=args.checkpoint_freq,
#             NUM_ENVS=args.num_envs,
#             resume_from=args.resume,
#             domain_randomize=args.domain_randomize,
#             norm_reward=not args.no_norm_reward,
#             base_reset_type=args.base_reset_type,
#             wandb_project=args.wandb_project,
#             wandb_entity=args.wandb_entity,
#             wandb_run_name=args.wandb_run_name,
#         )



# # for training the agents with the patch run this command 

# # python -m envs.training \
# #   --mode agent \
# #   --patch-checkpoint /path/to/patch_model \
# #   --patch-vecnorm /path/to/vecnormalize.pkl \
# #   --timesteps 1000000 \
# #   --num-envs 4