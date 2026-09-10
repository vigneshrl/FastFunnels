"""Trace b and lateral offset vs station on neck_escalate, to see HOW it fails
at the first centred triangle (s=41.16): too wide, or wide enough but off-line?"""
import os, sys, glob
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
W = "/p/cral/vignesh/bigtemp_files/FastFunnels/.claude/worktrees/ring-formation"
sys.path.insert(0, W); sys.path.insert(0, W + "/presentation_code")
import numpy as np
from mass_eval import install_variant_map_finder
install_variant_map_finder(W + "/maps/scenario_neck")
from make_legacy_patch_reel import load_legacy_policy
from envs.ppo_policy import PatchCarEnv, PatchEnvConfig

ckpt = sys.argv[1]
act, obs_dim, legacy = load_legacy_policy(ckpt)
env = PatchCarEnv(PatchEnvConfig(
    num_agents=2, render_mode=None, random_spawn=False, obs_mode="lidar",
    num_lidar_beams=obs_dim - 7, map_name="neck_escalate",
    a_cmd_min=1.5, a_cmd_max=3.0, b_cmd_min=1.0, b_cmd_max=4.6,
    wall_filter_enabled=False))
obs, _ = env.reset(seed=0)
rows = []
for i in range(20000):
    a = act(obs, env)
    obs, r, term, trunc, info = env.step(a)
    rows.append((float(info.get("frenet_s", np.nan)),
                 float(info.get("p0_b_now", np.nan)),
                 float(info.get("frenet_ey", np.nan))))
    if term or trunc:
        break
rows = np.array(rows)
s, b, ey = rows[:, 0], rows[:, 1], rows[:, 2]
print(f"{os.path.basename(os.path.dirname(ckpt))}/{os.path.basename(ckpt)}")
print(f"  ended at step {len(rows)}, s={s[-1]:.1f} m ({s[-1]/356:.1%})")
print(f"\n  {'s [m]':>7} {'b':>6} {'ey [m]':>8}   (triangle at s=41.16, lane needs b<=1.56)")
for target in (20, 28, 34, 36, 38, 39, 40, 41, 42):
    k = int(np.argmin(np.abs(s - target)))
    if abs(s[k] - target) > 3:
        continue
    mark = "  <-- triangle" if abs(target - 41) <= 1 else ""
    print(f"  {s[k]:7.1f} {b[k]:6.2f} {ey[k]:8.2f}{mark}")
near = np.where((s >= 36) & (s <= 43))[0]
if len(near):
    print(f"\n  in s=36..43 : b min {b[near].min():.2f} mean {b[near].mean():.2f}"
          f" | |ey| max {np.abs(ey[near]).max():.2f} m")
    print(f"  needed      : b <= 1.56, and ey pushed to ONE side to clear the block")
