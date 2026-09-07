"""Is the NMPC being asked to do something geometrically impossible?

Replays the patch on open_narrow_obs, and at EVERY step asks: given the funnel's
own (a,b) at that instant, does ANY arrangement of N followers exist that is
  - inside the containment ellipse (a-0.15, b-0.15), and
  - at least min_agent_dist from the patch car and from each other?
If no arrangement exists, no controller can hold containment -- the reference is
infeasible and the failure is the funnel's size, not the NMPC's tracking.
"""
import os, sys, math, itertools
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
MARGIN, MIND = 0.15, 0.80

def feasible(a, b, N, tries=400, seed=0):
    """Can N points fit? Ring search over radii/phases, then a repulsion polish."""
    ae, be = max(a - MARGIN, 0.5), max(b - MARGIN, 0.5)
    rng = np.random.default_rng(seed)
    for _ in range(tries):
        th = rng.uniform(0, 2 * math.pi) + np.arange(N) * 2 * math.pi / N \
             + rng.normal(0, 0.35, N)
        rad = rng.uniform(MIND, max(ae, be), N)
        P = np.stack([rad * np.cos(th), rad * np.sin(th)], 1)
        for _ in range(120):
            for i in range(N):
                f = np.zeros(2)
                for j in range(N):
                    if i == j: continue
                    d = P[i] - P[j]; r = np.linalg.norm(d) + 1e-9
                    if r < MIND + 0.01: f += d / r * (MIND + 0.01 - r)
                r0 = np.linalg.norm(P[i]) + 1e-9
                if r0 < MIND + 0.01: f += P[i] / r0 * (MIND + 0.01 - r0)
                P[i] += f
                g = (P[i, 0] / ae) ** 2 + (P[i, 1] / be) ** 2
                if g > 1.0: P[i] /= math.sqrt(g)
        ok = all(np.linalg.norm(p) >= MIND - 1e-3 for p in P) and \
             all((p[0] / ae) ** 2 + (p[1] / be) ** 2 <= 1.0 + 1e-6 for p in P) and \
             all(np.linalg.norm(P[i] - P[j]) >= MIND - 1e-3
                 for i, j in itertools.combinations(range(N), 2))
        if ok:
            return True
    return False

act, obs_dim, _ = load_legacy_policy(CK)
env = PatchCarEnv(PatchEnvConfig(
    num_agents=2, render_mode=None, random_spawn=False, obs_mode="lidar",
    num_lidar_beams=obs_dim - 7, map_name="open_narrow_obs",
    a_cmd_min=1.5, a_cmd_max=3.0, b_cmd_min=1.0, b_cmd_max=3.0,
    wall_filter_enabled=False))
o_, _ = env.reset(seed=0)
AB = []
for t in range(9000):
    o_, r, term, trunc, info = env.step(act(o_, env))
    p = env.active_patches[0]
    AB.append((round(float(p.a), 2), round(float(p.b), 2)))
    if term or trunc:
        break
print(f"episode: {len(AB)} steps on open_narrow_obs\n")

uniq = sorted(set(AB))
print(f"{len(uniq)} distinct (a,b) states visited")
cache = {}
for N in (1, 2, 3, 4, 6):
    for ab in uniq:
        cache[(ab, N)] = feasible(ab[0], ab[1], N)
    ok = sum(cache[(ab, N)] for ab in AB)
    print(f"  N={N}: funnel can contain the formation on "
          f"{100.0*ok/len(AB):5.1f}% of steps")

b = np.array([x[1] for x in AB])
print(f"\nfunnel b: min {b.min():.2f} max {b.max():.2f} mean {b.mean():.2f}"
      f" | at 1.00 floor {100*np.mean(b <= 1.005):.0f}% of steps")
for thr in (1.0, 1.2, 1.5, 2.0):
    print(f"  b <= {thr}: {100*np.mean(b <= thr + 1e-9):5.1f}% of steps")
