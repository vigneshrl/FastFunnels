# /u/atk9sb/.local/lib/python3.12/site-packages/stable_baselines3/common/vec_env/vec_monitor.py:44: UserWarning: The environment is already wrapped with a `Monitor` wrapperbut you are wrapping it with a `VecMonitor` wrapper, the `Monitor` statistics will beoverwritten by the `VecMonitor` ones.
#   warnings.warn(
# Traceback (most recent call last):
#   File "<frozen runpy>", line 198, in _run_module_as_main
#   File "<frozen runpy>", line 88, in _run_code
#   File "/bigtemp/atk9sb/FastFunnels/envs/training.py", line 1104, in <module>
#     train_joint_sb3(
#   File "/bigtemp/atk9sb/FastFunnels/envs/training.py", line 937, in train_joint_sb3
#     agent_raw = ConcatVecEnv([_make_markov_vec() for _ in range(num_envs)])
#                 ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#   File "/u/atk9sb/.local/lib/python3.12/site-packages/supersuit/vector/concat_vec_env.py", line 23, in __init__
#     self.vec_envs = vec_envs = [vec_env_fn() for vec_env_fn in vec_env_fns]
#                                 ^^^^^^^^^^^^
# TypeError: 'MarkovVectorEnv' object is not callable

# what is this above issue which is not letting me to run the agent policy training??



















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
    from .ppo_policy import (
        make_patch_env, make_agent_env, PatchEnv, PatchEnvConfig, JointEnvConfig,
    )
except ImportError:
    from envs.ppo_policy import (
        make_patch_env, make_agent_env, PatchEnv, PatchEnvConfig, JointEnvConfig,
    )

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecCheckNan, VecNormalize

    SB3_AVAILABLE = True
except ImportError:
    SB3_AVAILABLE = False

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
        self.best_mean_reward = -1e18

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq != 0:
            return True

        if len(self.model.ep_info_buffer) == 0:
            return True

        mean_reward = float(np.mean([ep["r"] for ep in self.model.ep_info_buffer]))
        print(f"Num timesteps: {self.num_timesteps} | Mean train reward: {mean_reward:.2f}")

        if mean_reward > self.best_mean_reward:
            self.best_mean_reward = mean_reward
            print(f"Saving new best: {mean_reward:.2f}")
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
    norm_reward: bool = False,
    base_reset_type: str = "rl_random_static",
    use_base_done_termination: bool = False,
    patch_only_mode: bool = True,
    debug_print_every_n_steps: int = 0,
    debug_print_episode_end: bool = True,
    wandb_project: str = "patch_sempc_training",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None, ):
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
                    # navigation_mode="centerline",
                    # debug_print_every_n_steps=debug_print_every_n_steps,
                    # debug_print_episode_end=debug_print_episode_end,
                    base_reset_type=base_reset_type,
                    # use_base_done_termination=use_base_done_termination,
                    # patch_only_mode=patch_only_mode,
                ))
                for i in range(NUM_ENVS)
            ]
        )
    else:
        env = DummyVecEnv(
            [
                _monitored(make_patch_env(
                    0,
                    seed=42,
                    domain_randomize=domain_randomize,
                    # navigation_mode="centerline",
                    # debug_print_every_n_steps=debug_print_every_n_steps,
                    # debug_print_episode_end=debug_print_episode_end,
                    base_reset_type=base_reset_type,
                    # use_base_done_termination=use_base_done_termination,
                    # patch_only_mode=patch_only_mode,
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

    rollout_steps = 2048
    # total_rollout = rollout_steps * max(1, NUM_ENVS)
    batch_size = NUM_ENVS *  rollout_steps //4 #max(64, total_rollout // 4)
    # if total_rollout % batch_size != 0:
    #     # Keep PPO minibatches exact to avoid partial batches.
    #     batch_size = 256 if total_rollout % 256 == 0 else 128

    if resume_from and os.path.exists(resume_from + ".zip"):
        model = PPO.load(resume_from, env=env, device="cpu")
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
            # device="cpu",
            policy_kwargs= dict(net_arch=dict(pi=[256, 256], vf=[256, 256]))
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
    phase_steps: int = 8192,
    patch_solo_timesteps: int = 10_000_000,
    resume_patch: str = "",
    resume_agents: str = "",
    norm_reward: bool = True,
    wandb_project: str = "joint_sb3_training",
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None, ):
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
    if not SUPERSUIT_AVAILABLE:
        raise RuntimeError("supersuit is required: pip install supersuit")

    try:
        from .pz_env import AgentEnv
    except ImportError:
        from envs.pz_env import AgentEnv

    from stable_baselines3.common.vec_env import VecMonitor

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
            name=wandb_run_name or f"joint_sb3_{run_id}",
            config={
                "total_timesteps": total_timesteps,
                "patch_solo_timesteps": patch_solo_timesteps,
                "num_envs": num_envs,
                "num_cpus": num_cpus,
                "phase_steps": phase_steps,
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

    patch_dir = os.path.join(run_dir, "patch")
    agent_dir = os.path.join(run_dir, "agents")
    os.makedirs(patch_dir, exist_ok=True)
    os.makedirs(agent_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Patch training: optional solo PatchCarEnv (f110 ×1), then JointPatchView (f110 ×3).
    # Frozen agent policy injected via env_method + .npz (joint patch vec only).
    # ------------------------------------------------------------------
    try:
        from .ppo_policy import (
            JointEnv, JointEnvConfig, JointPatchView, PatchCarEnv, PatchEnvConfig,
        )
    except ImportError:
        from envs.ppo_policy import (
            JointEnv, JointEnvConfig, JointPatchView, PatchCarEnv, PatchEnvConfig,
        )

    patch_cb = TrainingCallback(
        save_dir=patch_dir,
        save_freq=max(1, checkpoint_freq // num_envs),
        policy_name="patch_sb3",
    )
    save_patch_best = SaveBestWithVecNormalize(save_dir=patch_dir, check_freq=patch_n_steps,
                                               model_name="best_model")

    def _make_patch_car():
        cfg = PatchEnvConfig(split_mode=True, random_spawn=True)
        env = PatchCarEnv(cfg)
        env.reset()
        return Monitor(env)

    def _make_joint_patch_env():
        try:
            print("[DEBUG] Creating JointEnv...")
            shared = JointEnv(JointEnvConfig(render_mode=None))
            print("[DEBUG] JointEnv created, calling reset...")
            shared.reset()
            print("[DEBUG] JointEnv reset successful")
            print("[DEBUG] Creating JointPatchView...")
            view = JointPatchView(shared)
            print("[DEBUG] JointPatchView created")
            return Monitor(view)
        except Exception as e:
            print(f"[ERROR] Failed to create JointEnv: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            raise

    patch_ppo = None
    patch_car_vec = None
    _solo_obs_rms = None
    if patch_solo_timesteps > 0 and not resume_patch:
        if num_envs > 1:
            patch_car_vec = SubprocVecEnv(
                [_make_patch_car for _ in range(num_envs)],
                start_method="fork",
            )
        else:
            patch_car_vec = DummyVecEnv([_make_patch_car])
        patch_car_vec = VecMonitor(patch_car_vec)
        if norm_reward:
            patch_car_vec = VecNormalize(
                patch_car_vec, norm_obs=True, norm_reward=False, clip_obs=10.0,
            )
        patch_ppo = PPO(
            "MlpPolicy", patch_car_vec,
            tensorboard_log=os.path.join(tb_log_dir, "patch_solo"),
            n_steps=patch_n_steps, batch_size=patch_batch, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5, max_grad_norm=0.5,
            device=_device, verbose=1,
        )
        print(f"[joint_sb3] Phase 0: PatchCarEnv solo f110 for {patch_solo_timesteps} steps")
        patch_ppo.learn(
            patch_solo_timesteps,
            callback=[patch_cb, save_patch_best],
            reset_num_timesteps=True,
            progress_bar=False,
        )
        patch_ppo.save(os.path.join(patch_dir, "patch_after_solo"))
        _solo_obs_rms = patch_car_vec.obs_rms if isinstance(patch_car_vec, VecNormalize) else None
        if isinstance(patch_car_vec, VecNormalize):
            patch_car_vec.save(os.path.join(patch_dir, "patch_after_solo_vecnormalize.pkl"))
        patch_car_vec.close()
        patch_car_vec = None

    # TEMPORARY: Use DummyVecEnv for Phase A to see actual worker errors
    # TODO: Switch back to SubprocVecEnv once JointEnv is verified stable
    print(f"[DEBUG] Creating patch_vec with DummyVecEnv (num_envs={num_envs})")
    # patch_vec = DummyVecEnv([_make_joint_patch_env for _ in range(num_envs)])
    if num_envs > 1:
        patch_vec = SubprocVecEnv(
            [_make_joint_patch_env for _ in range(num_envs)],
            start_method="fork",
        )
    else:
        patch_vec = DummyVecEnv([_make_joint_patch_env])
    patch_vec = VecMonitor(patch_vec)
    if norm_reward:
        patch_vec = VecNormalize(patch_vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    if resume_patch:
        patch_ppo = PPO.load(resume_patch, env=patch_vec)
    elif patch_ppo is not None:
        patch_ppo.set_env(patch_vec)
        # Transfer obs normalization stats so policy sees same obs scale as Phase 0
        if _solo_obs_rms is not None and isinstance(patch_vec, VecNormalize):
            patch_vec.obs_rms = _solo_obs_rms
    else:
        patch_ppo = PPO(
            "MlpPolicy", patch_vec,
            tensorboard_log=os.path.join(tb_log_dir, "patch"),
            n_steps=patch_n_steps, batch_size=patch_batch, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.05,
            vf_coef=0.5, max_grad_norm=0.5,
            device=_device, verbose=1,
        )

    # ------------------------------------------------------------------
    # Agent vec env — built ONCE, never rebuilt.
    # Uses a mutable policy holder so the frozen patch policy can be
    # swapped in-place each iteration without tearing down envs.
    # This avoids file-descriptor leaks from repeated subprocess creation
    # and preserves VecNormalize running stats across iterations.
    # ------------------------------------------------------------------
    class _MutablePatchPolicy:
        """In-process mutable wrapper: all AgentEnvs share this reference."""
        def __init__(self):
            self._fn = None
        def __call__(self, obs):
            if self._fn is not None:
                return self._fn(obs)
            return np.array([0.0, 3.0, 3.0, 2.5], dtype=np.float32)
        def update(self, fn):
            self._fn = fn

    _patch_holder = _MutablePatchPolicy()

    from supersuit.vector import MarkovVectorEnv, ConcatVecEnv
    from supersuit.vector.sb3_vector_wrapper import SB3VecEnvWrapper

    def _make_markov_vec():
        return MarkovVectorEnv(AgentEnv(patch_policy=_patch_holder))

    agent_raw = ConcatVecEnv([_make_markov_vec() for _ in range(num_envs)])
    agent_raw = SB3VecEnvWrapper(agent_raw)
    agent_vec = VecMonitor(agent_raw)
    if norm_reward:
        agent_vec = VecNormalize(agent_vec, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0)

    agent_cb = TrainingCallback(
        save_dir=agent_dir,
        save_freq=max(1, checkpoint_freq // num_envs),
        policy_name="agent_sb3",
    )
    save_agent_best = SaveBestWithVecNormalize(save_dir=agent_dir, check_freq=agent_n_steps,
                                               model_name="best_model")

    if resume_agents:
        agent_ppo = PPO.load(resume_agents, env=agent_vec)
    else:
        agent_ppo = PPO(
            "MlpPolicy", agent_vec,
            tensorboard_log=os.path.join(tb_log_dir, "agent"),
            n_steps=agent_n_steps, batch_size=agent_batch, n_epochs=10,
            gamma=0.99, gae_lambda=0.95, clip_range=0.2,
            ent_coef=0.01, vf_coef=0.5, max_grad_norm=0.5,
            device=_device, verbose=1,
        )

    # Temp file for cross-process frozen agent policy transfer (patch SubprocVecEnv)
    _frozen_agent_npz = os.path.join(run_dir, "_frozen_agent.npz")

    # ------------------------------------------------------------------
    # Alternating co-evolution loop
    # ------------------------------------------------------------------
    solo_done = patch_solo_timesteps > 0 and not resume_patch
    total_steps = patch_solo_timesteps if solo_done else 0
    iterations = max(1, total_timesteps // (2 * phase_steps))
    print(f"Starting joint_sb3 training: {iterations} iterations, "
          f"{phase_steps} steps/phase, {num_envs} envs"
          + (f" (after {patch_solo_timesteps} solo patch steps)" if solo_done else ""))

    _has_trained_agents = False
    patch_ppo_steps = patch_solo_timesteps if solo_done else 0
    agent_ppo_steps = 0

    for iteration in range(1, iterations + 1):
        print(f"\n=== Iteration {iteration}/{iterations} ===")

        # ------------------------------------------------------------------
        # Inject frozen agent policy into patch SubprocVecEnv workers.
        # Workers load a .npz file (only a string crosses the pipe).
        # ------------------------------------------------------------------
        if _has_trained_agents:
            patch_vec.env_method("load_frozen_agent_npz", _frozen_agent_npz)
        else:
            patch_vec.env_method("load_frozen_agent_npz", None)

        # Phase 1: train patch (do not reset SB3 timestep schedule on iter 1 if solo warmup ran)
        patch_ppo.learn(
            phase_steps,
            callback=[patch_cb, save_patch_best],
            reset_num_timesteps=(iteration == 1 and not solo_done),
            progress_bar=False,
        )
        total_steps     += phase_steps
        patch_ppo_steps += phase_steps

        # Phase 2: train agent with frozen patch policy snapshot.
        # Just swap the policy inside the mutable holder — no env rebuild needed.
        _patch_snap = patch_ppo.policy
        _patch_vn = patch_vec if isinstance(patch_vec, VecNormalize) else None

        def _frozen_patch(obs: np.ndarray, _pol=_patch_snap, _vn=_patch_vn) -> np.ndarray:
            obs_in = obs[np.newaxis]
            if _vn is not None:
                obs_in = _vn.normalize_obs(obs_in)
            action, _ = _pol.predict(obs_in, deterministic=True)
            return action[0]

        _patch_holder.update(_frozen_patch)

        agent_ppo.learn(
            phase_steps,
            callback=[agent_cb, save_agent_best],
            reset_num_timesteps=(iteration == 1),
            progress_bar=False,
        )
        total_steps     += phase_steps
        agent_ppo_steps += phase_steps

        # Save frozen agent weights for next iteration's patch phase
        _agent_vn = agent_vec if isinstance(agent_vec, VecNormalize) else None
        _save_frozen_agent_npz(agent_ppo.policy, _agent_vn, _frozen_agent_npz)
        _has_trained_agents = True

        if WANDB_AVAILABLE and wandb_run is not None:
            patch_buf = patch_ppo.ep_info_buffer
            agent_buf = agent_ppo.ep_info_buffer
            log = {
                "joint_sb3/total_steps":     total_steps,
                "joint_sb3/patch_ppo_steps": patch_ppo_steps,
                "joint_sb3/agent_ppo_steps": agent_ppo_steps,
                "joint_sb3/iteration":       iteration,
            }
            if patch_buf:
                log["joint_sb3/patch_ep_rew_mean"] = float(np.mean([e["r"] for e in patch_buf]))
                log["joint_sb3/patch_ep_len_mean"] = float(np.mean([e["l"] for e in patch_buf]))
            if agent_buf:
                log["joint_sb3/agent_ep_rew_mean"] = float(np.mean([e["r"] for e in agent_buf]))
                log["joint_sb3/agent_ep_len_mean"] = float(np.mean([e["l"] for e in agent_buf]))
            wandb.log(log, step=total_steps)

    # --- Save final ---
    patch_ppo.save(os.path.join(patch_dir, "final_model"))
    agent_ppo.save(os.path.join(agent_dir, "final_model"))
    if isinstance(patch_vec, VecNormalize):
        patch_vec.save(os.path.join(patch_dir, "final_vecnormalize.pkl"))
    if isinstance(agent_vec, VecNormalize):
        agent_vec.save(os.path.join(agent_dir, "final_vecnormalize.pkl"))
    patch_vec.close()
    agent_vec.close()
    if os.path.exists(_frozen_agent_npz):
        os.remove(_frozen_agent_npz)
    if WANDB_AVAILABLE and wandb_run is not None:
        wandb.finish()
    print(f"joint_sb3 training complete. Models saved to {run_dir}")
    return patch_ppo, agent_ppo


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
    args = parser.parse_args()

    if args.mode == "joint_sb3":
        train_joint_sb3(
            total_timesteps=args.timesteps,
            save_path=args.save_path if args.save_path != "patch_policy_models" else "joint_sb3_models",
            checkpoint_freq=args.checkpoint_freq,
            num_envs=args.num_envs,
            num_cpus=args.num_cpus,
            phase_steps=args.phase_steps,
            patch_solo_timesteps=args.patch_solo_timesteps,
            resume_patch=args.resume_patch,
            resume_agents=args.resume_agents,
            norm_reward=not args.no_norm_reward,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
        )
    elif args.mode == "joint":
        train_joint_policy(
            total_timesteps=args.timesteps,
            save_path=args.save_path if args.save_path != "patch_policy_models" else "joint_policy_models",
            checkpoint_freq=args.checkpoint_freq,
            NUM_ENVS=args.num_envs,
            phase_steps=args.phase_steps,
            resume_patch=args.resume_patch,
            resume_agents=args.resume_agents,
            norm_reward=not args.no_norm_reward,
            wandb_project=args.wandb_project,
        )
    elif args.mode == "agent":
        train_agent_policy(
            total_timesteps=args.timesteps,
            save_path=args.save_path if args.save_path != "patch_policy_models" else "agent_policy_models",
            checkpoint_freq=args.checkpoint_freq,
            NUM_ENVS=args.num_envs,
            resume_from=args.resume,
            patch_checkpoint_path=args.patch_checkpoint,
            patch_vecnorm_path=args.patch_vecnorm,
            norm_reward=not args.no_norm_reward,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
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