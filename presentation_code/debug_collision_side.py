"""At the moment of failure, which side kills it -- the centred block or the
outer wall? Prints the last steps with the funnel's inner/outer edge against the
free-lane bounds measured from the occupancy grid."""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
W = "/p/cral/vignesh/bigtemp_files/FastFunnels/.claude/worktrees/ring-formation"
sys.path.insert(0, W); sys.path.insert(0, W + "/presentation_code")
import numpy as np, yaml
from PIL import Image
from PIL.Image import Transpose
from mass_eval import install_variant_map_finder
install_variant_map_finder(W + "/maps/scenario_neck")
from make_legacy_patch_reel import load_legacy_policy
from envs.ppo_policy import PatchCarEnv, PatchEnvConfig

d = W + "/maps/scenario_neck/neck_escalate"
spec = yaml.safe_load(open(f"{d}/neck_escalate_map.yaml"))
occ = np.array(Image.open(f"{d}/{spec['image']}").transpose(Transpose.FLIP_TOP_BOTTOM))
res = float(spec["resolution"]); ox, oy = spec["origin"][0], spec["origin"][1]
H, Wd = occ.shape


def free_spans(cl, s):
    """Free intervals along the normal at station s, left-positive."""
    x, y = cl.position_at(float(s)); yaw = float(cl.yaw_at(float(s)))
    nx, ny = -np.sin(yaw), np.cos(yaw)
    ts = np.arange(-6.0, 6.0 + 1e-6, res)
    cx = np.clip(((x + ts * nx - ox) / res).astype(int), 0, Wd - 1)
    cy = np.clip(((y + ts * ny - oy) / res).astype(int), 0, H - 1)
    free = (occ[cy, cx] > 200).astype(np.int8)
    e = np.flatnonzero(np.diff(np.concatenate(([0], free, [0]))))
    return [(ts[a], ts[b - 1]) for a, b in zip(e[::2], e[1::2])]


ckpt = sys.argv[1]
act, obs_dim, _ = load_legacy_policy(ckpt)
env = PatchCarEnv(PatchEnvConfig(
    num_agents=2, render_mode=None, random_spawn=False, obs_mode="lidar",
    num_lidar_beams=obs_dim - 7, map_name="neck_escalate",
    a_cmd_min=1.5, a_cmd_max=3.0, b_cmd_min=1.0, b_cmd_max=4.6,
    wall_filter_enabled=False))
obs, _ = env.reset(seed=0)
hist = []
for _ in range(20000):
    obs, r, term, trunc, info = env.step(act(obs, env))
    hist.append((float(info.get("frenet_s", np.nan)),
                 float(info.get("p0_b_now", np.nan)),
                 float(info.get("frenet_ey", np.nan))))
    if term or trunc:
        break
print(f"failed at step {len(hist)}, s={hist[-1][0]:.2f}")
print(f"\n{'s':>7} {'b':>5} {'ey':>6} {'funnel span':>15}   free spans (left-positive)")
for s, b, ey in hist[-8:]:
    sp = free_spans(env.cl, s)
    sp_txt = "  ".join(f"[{a:+.2f},{c:+.2f}]" for a, c in sp if c - a > 0.3)
    print(f"{s:7.2f} {b:5.2f} {ey:+6.2f} [{ey-b:+.2f},{ey+b:+.2f}]   {sp_txt}")
