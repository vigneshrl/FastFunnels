import numpy as np

from envs.f110_env import F110EnvAdapter, F110Config
from envs.reset.map_reset import MapResetHelper

def mean_disp(prev_obs, obs):
    ds = []
    n = len(obs["poses_x"])
    for i in range(n):
        dx = float(obs["poses_x"][i] - prev_obs["poses_x"][i])
        dy = float(obs["poses_y"][i] - prev_obs["poses_y"][i])
        ds.append((dx**2 + dy**2) ** 0.5)
    return float(np.mean(ds)), ds

cfg = F110Config(
    map_name="Spielberg",
    num_agents=2,
    timestep=0.05,
    control_input=("speed", "steering_angle"),
    reset_type="rl_random_static",
)
f110 = F110EnvAdapter(cfg, render_mode=None)
f110.ensure_initialized()

print("Base action space low/high:")
print(" low :", f110.base_env.action_space.low)
print(" high:", f110.base_env.action_space.high)

# Build deterministic-ish reset around centerline start
track, occ_map, resolution, origin = f110.get_track_data()
waypoints = np.column_stack([track.centerline.xs, track.centerline.ys]).astype(np.float32)
patch_x, patch_y = float(waypoints[0, 0]), float(waypoints[0, 1])
dx = waypoints[1, 0] - waypoints[0, 0]
dy = waypoints[1, 1] - waypoints[0, 1]
patch_theta = float(np.arctan2(dy, dx))

mr = MapResetHelper()
poses = mr.generate_safe_agent_poses(
    num_agents=2,
    patch_x=patch_x,
    patch_y=patch_y,
    patch_theta=patch_theta,
    patch_a=1.8,
    patch_b=1.2,
    domain_randomize=False,
    np_random=None,
)

# ---------- Test A: [speed, steering] ----------
obs, _ = f110.reset(poses=poses)
actions_a = np.array([[2.0, 0.10], [2.0, -0.10]], dtype=np.float32)
obs_a, _, _, _, _ = f110.step(actions_a)
mean_a, each_a = mean_disp(obs, obs_a)

print("\nTest A actions [[speed, steering]]:")
print(" actions:", actions_a)
print(" mean disp:", mean_a, "each:", each_a)

# ---------- Test B: [steering, speed] ----------
obs, _ = f110.reset(poses=poses)
actions_b = np.array([[0.10, 2.0], [-0.10, 2.0]], dtype=np.float32)
obs_b, _, _, _, _ = f110.step(actions_b)
mean_b, each_b = mean_disp(obs, obs_b)

print("\nTest B actions [[steering, speed]]:")
print(" actions:", actions_b)
print(" mean disp:", mean_b, "each:", each_b)

print("\nConclusion:")
if mean_a > mean_b:
    print(" -> Env likely expects [speed, steering].")
elif mean_b > mean_a:
    print(" -> Env likely expects [steering, speed].")
else:
    print(" -> Both similar; inspect simulator docs/config for exact mapping.")

f110.close()
