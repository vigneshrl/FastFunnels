"""If a feasible arrangement exists at 100% of steps and the solver never fails,
why do followers leave the funnel?

Hypothesis: the REFERENCE is not continuous. The adaptive packing is solved
independently per (a,b) and cached on rounded values, so two consecutive ticks
can get structurally different arrangements -- the slots teleport, and no
follower with real dynamics can track a teleporting target.

Measures, per step of the real episode:
  * how far each slot moves between consecutive ticks (patch frame)
  * the world-frame speed a follower would need to stay on its slot, including
    the funnel's own rotation
"""
import os, sys, math, re
import numpy as np
ROOT = "/p/cral/vignesh/bigtemp_files/FastFunnels/.claude/worktrees/ring-formation"
os.chdir(ROOT); sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/presentation_code")
POOL = f"{ROOT}/maps/scenario_on_obs"
os.environ.update(QT_QPA_PLATFORM="offscreen", MPLBACKEND="Agg", REEL_POOL_DIR=POOL)
os.environ.pop("DISPLAY", None)
import matplotlib; matplotlib.use("Agg")
from mass_eval import install_variant_map_finder
install_variant_map_finder(POOL)
from make_legacy_patch_reel import load_legacy_policy
from envs.ppo_policy import PatchCarEnv, PatchEnvConfig

CK = "/p/cral/vignesh/bigtemp_files/FastFunnels/patch_policy_models/run_20260518_151612/checkpoint_18510000.zip"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 4

# import the packer from the runner
src = open(f"{ROOT}/mpc_follower_native_n.py").read()
g = {"np": np, "math": math, "os": os, "MIN_DIST": 0.80, "CONTAIN_MARGIN": 0.15,
     "_PACK_CACHE": {}, "_PACK_ITERS": 140, "N": N}
for fn in ("_pack_relax", "_pack", "_adaptive_slots"):
    m = re.search(rf"\ndef {fn}\(.*?(?=\ndef |\nclass |\n# )", src, re.S)
    exec(m.group(0), g)

act, obs_dim, _ = load_legacy_policy(CK)
env = PatchCarEnv(PatchEnvConfig(
    num_agents=2, render_mode=None, random_spawn=False, obs_mode="lidar",
    num_lidar_beams=obs_dim - 7, map_name="open_narrow_obs",
    a_cmd_min=1.5, a_cmd_max=3.0, b_cmd_min=1.0, b_cmd_max=3.0,
    wall_filter_enabled=False))
o_, _ = env.reset(seed=0)
traj = []
for t in range(9000):
    o_, r, term, trunc, info = env.step(act(o_, env))
    p = env.active_patches[0]
    traj.append((float(p.a), float(p.b), float(p.theta), float(p.x), float(p.y),
                 float(p.v)))
    if term or trunc:
        break
print(f"episode {len(traj)} steps, N={N}\n")

DT = 0.01
jump_local, need_v, rot_v = [], [], []
prev_local = None
for i, (a, b, th, x, y, v) in enumerate(traj):
    S = np.array([g["_adaptive_slots"](k, a, b) for k in range(N)])
    if prev_local is not None:
        jump_local.append(np.linalg.norm(S - prev_local, axis=1).max())
    prev_local = S
    if i > 0:
        pa, pb, pth, px, py, pv = traj[i - 1]
        Sp = np.array([g["_adaptive_slots"](k, pa, pb) for k in range(N)])
        c, s = math.cos(th), math.sin(th)
        cp, sp = math.cos(pth), math.sin(pth)
        W = np.stack([x + S[:, 0] * c - S[:, 1] * s,
                      y + S[:, 0] * s + S[:, 1] * c], 1)
        Wp = np.stack([px + Sp[:, 0] * cp - Sp[:, 1] * sp,
                       py + Sp[:, 0] * sp + Sp[:, 1] * cp], 1)
        need_v.append(np.linalg.norm(W - Wp, axis=1).max() / DT)
        dth = (th - pth + math.pi) % (2 * math.pi) - math.pi
        rot_v.append(abs(dth) / DT * np.linalg.norm(S, axis=1).max())

jump_local = np.array(jump_local); need_v = np.array(need_v); rot_v = np.array(rot_v)
print("SLOT MOVEMENT IN THE PATCH FRAME (should be ~0: the formation should")
print("only breathe slowly as the funnel changes shape)")
print(f"  max slot jump per tick : mean {jump_local.mean():.3f} m   "
      f"p95 {np.percentile(jump_local,95):.3f} m   MAX {jump_local.max():.3f} m")
for thr in (0.05, 0.1, 0.25, 0.5):
    print(f"    jump > {thr:4.2f} m on {100*np.mean(jump_local>thr):5.1f}% of ticks")

print("\nWORLD-FRAME SPEED A FOLLOWER WOULD NEED to sit on its slot")
print(f"  mean {need_v.mean():6.2f} m/s   p95 {np.percentile(need_v,95):7.2f}   "
      f"MAX {need_v.max():8.2f}   (car limit ~10 m/s)")
for thr in (10, 20, 50):
    print(f"    needs > {thr:3d} m/s on {100*np.mean(need_v>thr):5.1f}% of ticks")

print("\ncontribution from FUNNEL ROTATION alone (patch yaw rate x slot radius)")
print(f"  mean {rot_v.mean():6.2f} m/s   p95 {np.percentile(rot_v,95):7.2f}   "
      f"MAX {rot_v.max():8.2f}")
pv = np.array([t[5] for t in traj])
print(f"\npatch speed: mean {pv.mean():.2f}  max {pv.max():.2f} m/s")
