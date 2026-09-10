"""True difficulty of a map: track each CONTINUOUS lane through every obstacle
and report the tightest point of the best one.

check_funnel_traversable.py asks "does the minimum funnel fit SOMEWHERE on this
scan", which counts a lane the patch cannot reach without crossing the block.
A committed vehicle must stay inside one lane, so what binds is the narrowest
point of the lane it picked.
"""
import os, sys
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
W = "/p/cral/vignesh/bigtemp_files/FastFunnels/.claude/worktrees/ring-formation"
sys.path.insert(0, W); sys.path.insert(0, W + "/presentation_code")
import numpy as np, yaml
from PIL import Image
from PIL.Image import Transpose
from mass_eval import install_variant_map_finder

pool, name = sys.argv[1], sys.argv[2]
install_variant_map_finder(pool)
from envs.ppo_policy import PatchCarEnv, PatchEnvConfig

d = f"{pool}/{name}"
spec = yaml.safe_load(open(f"{d}/{name}_map.yaml"))
occ = np.array(Image.open(f"{d}/{spec['image']}").transpose(Transpose.FLIP_TOP_BOTTOM))
res = float(spec["resolution"]); ox, oy = spec["origin"][0], spec["origin"][1]
H, Wd = occ.shape
env = PatchCarEnv(PatchEnvConfig(
    num_agents=2, render_mode=None, random_spawn=False, obs_mode="lidar",
    num_lidar_beams=108, map_name=name, wall_filter_enabled=False))
env.reset(seed=0)
cl = env.cl
L = float(env.track_length)


def spans(s):
    x, y = cl.position_at(float(s)); yaw = float(cl.yaw_at(float(s)))
    nx, ny = -np.sin(yaw), np.cos(yaw)
    ts = np.arange(-6.0, 6.0 + 1e-6, res)
    cx = np.clip(((x + ts * nx - ox) / res).astype(int), 0, Wd - 1)
    cy = np.clip(((y + ts * ny - oy) / res).astype(int), 0, H - 1)
    free = (occ[cy, cx] > 200).astype(np.int8)
    e = np.flatnonzero(np.diff(np.concatenate(([0], free, [0]))))
    return [(ts[a], ts[b - 1]) for a, b in zip(e[::2], e[1::2]) if ts[b - 1] - ts[a] > 0.25]


# walk the course; wherever the scan splits into >1 lane, follow each lane by
# overlap until they merge again, and record the tightest width along each.
S = np.arange(0.0, L, 0.25)
FLOOR = 2.0            # funnel width at the b=1.0 floor
worst = []
i = 0
while i < len(S):
    sp = spans(S[i])
    if len(sp) < 2:
        i += 1
        continue
    j, lanes = i, [[w] for w in sp]
    while j + 1 < len(S):
        j += 1
        nxt = spans(S[j])
        if len(nxt) < 2:
            break
        newl = []
        for tr in lanes:
            lo, hi = tr[-1]
            cand = [w for w in nxt if min(hi, w[1]) - max(lo, w[0]) > 0.05]
            if cand:
                newl.append(tr + [max(cand, key=lambda w: w[1] - w[0])])
        if not newl:
            break
        lanes = newl
    widths = [min(h - l for l, h in tr) for tr in lanes]
    best = max(widths) if widths else 0.0
    worst.append((S[i], S[j], best, sorted(widths, reverse=True)))
    i = j + 1

print(f"{name}: {L:.0f} m, {len(worst)} split sections (a block divides the corridor)\n")
print(f"{'from':>7} {'to':>7} {'best lane':>10} {'all lanes':>22}  verdict")
tight = 0
for s0, s1, best, ws in worst:
    ok = best - FLOOR
    verdict = "OK" if ok > 0.6 else ("TIGHT" if ok > 0.15 else "AT LIMIT")
    if ok <= 0.6:
        tight += 1
    print(f"{s0:7.1f} {s1:7.1f} {best:10.2f} {str([round(w,2) for w in ws]):>22}  {verdict}")
print(f"\nfunnel at the b=1.0 floor is {FLOOR:.1f} m wide")
print(f"sections where the best lane leaves under 0.6 m of total slack: {tight}/{len(worst)}")
