"""
N-follower decentralised NMPC against the patch running in its NATIVE env
(PatchCarEnv) -- the multi-agent version of mpc_follower_native.py.

WHY THIS EXISTS
---------------
Steps the frozen patch in PatchCarEnv exactly as patch_native_eval.py does
(real f110 scan -> the patch actually completes open_narrow) and simulates N
followers against the live funnel ellipse, one SEMPCSolver each.

Both the follower PLANT and the follower MPC use the f1tenth_gym SINGLE-TRACK
DYNAMIC model by DEFAULT (tyre slip, same params / RK4 the real f110 sim uses).

The code answers one question: with a well-behaved patch, can decentralised NMPC hold
N=2 / N=3 followers inside the funnel through the whole corridor?

Deliberately sidesteps "problem 2" (formation assembly): followers SPAWN already
in their slots, spaced >= MPC min_agent_dist.

    PY=/p/cral/vignesh/envs/fastfunnels/bin/python
    $PY mpc_follower_native_n.py --policy patch_policy_models/best_scenario_pool \
        --map open_narrow --n 2 --seed 0
    MPC_SLOTS=abreast $PY mpc_follower_native_n.py --policy ... --n 3

env vars: FOLLOWER_MODEL {st(default),kinematic}  MPC_MODEL {st(default),kinematic}
          MPC_SLOTS {wedge(default),trail,abreast,split}  MPC_GAP  MPC_STAGGER
          MPC_SLOT_D  MPC_SLOT_LAT  MPC_MIN_DIST  MPC_HZ_S  MPC_HZ_N
          MPC_ST_SUBSTEPS  MPC_MAX_ITER  MPC_W_VEL/W_CENTER/W_CONTAIN
          MPC_W_COLL  MPC_COLL_HARD {0,1,patch}  MPC_COLL_R
"""
from __future__ import annotations
import os, sys, math, pickle, argparse
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import imageio.v2 as imageio

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "presentation_code"))
from stable_baselines3 import PPO
from envs.ppo_policy import PatchCarEnv, PatchEnvConfig
from envs.mpc import SEMPCSolver, MPCConfig

ap = argparse.ArgumentParser()
ap.add_argument("--policy", default="patch_policy_models/best_scenario_pool")
ap.add_argument("--map", default="open_narrow")
ap.add_argument("--map-dir", default="maps/map_foundations",
                help="extra map tree to resolve --map against (staged foundation/pool maps)")
ap.add_argument("--n", type=int, default=2, help="number of followers")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--steps", type=int, default=6000)
ap.add_argument("--every", type=int, default=6, help="render every Nth sim step")
ap.add_argument("--mpc-every", type=int, default=1,
                help="re-solve MPC every Nth control tick (1 = every funnel update)")
ap.add_argument("--render", choices=["mpl", "f1tenth", "both"], default="both",
                help="mpl = schematic; f1tenth = the gym's pygame renderer (real car "
                     "sprites); both = side-by-side")
ap.add_argument("--out", default="")
args = ap.parse_args()
N = args.n

# resolve staged maps (mf_*, sp_*) that aren't in the shared f1tenth_gym tree
if args.map_dir and os.path.isdir(os.path.join(args.map_dir, args.map)):
    from mass_eval import install_variant_map_finder
    install_variant_map_finder(args.map_dir)
    print(f"[map] resolving '{args.map}' from {args.map_dir}")

DT = 0.01
WB = 0.33
V_LO, V_HI = 0.5, 12.0
ACCEL_MAX = 9.5

CAR_L, CAR_W = 0.58, 0.31        # f1tenth footprint (m) — patch car + followers

# ---- follower plant model -------------------------------------------------
#   st (default): f1tenth_gym single-track DYNAMIC model (tyre slip), same
#                 params / RK4 / PID the real f110 sim uses
#   kinematic   : the analytic bicycle (override via FOLLOWER_MODEL=kinematic)
FOLLOWER_MODEL = os.environ.get("FOLLOWER_MODEL", "st").lower()
_STP = dict(mu=1.0489, C_Sf=4.718, C_Sr=5.4562, lf=0.15875, lr=0.17145, h=0.074,
            m=3.74, I=0.04712, s_min=-0.4189, s_max=0.4189, sv_min=-3.2, sv_max=3.2,
            v_switch=7.319, a_max=9.51, v_min=-5.0, v_max=20.0)
_ST_ARGS = (_STP["mu"], _STP["C_Sf"], _STP["C_Sr"], _STP["lf"], _STP["lr"], _STP["h"],
            _STP["m"], _STP["I"], _STP["s_min"], _STP["s_max"], _STP["sv_min"],
            _STP["sv_max"], _STP["v_switch"], _STP["a_max"], _STP["v_min"], _STP["v_max"])
if FOLLOWER_MODEL == "st":
    from f1tenth_gym.envs.dynamic_models import vehicle_dynamics_st, pid_steer, pid_accl
    # 7-D single-track state per follower: [x, y, delta, v, psi, psi_dot, beta]
    _FST: list = []

    def _st_rhs(x, u):
        return vehicle_dynamics_st(x, u, *_ST_ARGS)

    def _st_rk4(x, u):
        k1 = _st_rhs(x, u)
        k2 = _st_rhs(x + 0.5 * DT * k1, u)
        k3 = _st_rhs(x + 0.5 * DT * k2, u)
        k4 = _st_rhs(x + DT * k3, u)
        return x + (DT / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

# Slot geometry MUST keep every commanded slot >= ~1 car-length from the patch
# car and from every other slot, or the reference itself asks cars to overlap.
GAP      = float(os.environ.get("MPC_GAP", "0.9"))    # follower standoff behind patch car
STAGGER  = float(os.environ.get("MPC_STAGGER", "0.85"))
SLOTS    = os.environ.get("MPC_SLOTS", "wedge")
SLOT_D   = float(os.environ.get("MPC_SLOT_D", "0.80"))
SLOT_LAT = float(os.environ.get("MPC_SLOT_LAT", "0.62"))
MIN_DIST = float(os.environ.get("MPC_MIN_DIST", "0.8"))   # MPC inter-agent keep-out

zp = args.policy if args.policy.endswith(".zip") else os.path.join(args.policy, "best_model.zip")
tag = os.path.basename(os.path.dirname(zp)) or os.path.basename(args.policy)
OUT_MP4 = args.out or (
    f"videos/mpc_native_n{N}_{SLOTS}_plant-{FOLLOWER_MODEL}"
    f"_mpc-{os.environ.get('MPC_MODEL', 'st').lower()}_{tag}"
    f"_{args.map}_seed{args.seed}.mp4")
os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
COLORS = ["red", "lime", "magenta", "orange", "cyan", "yellow"]


# ---------------------------------------------------------------- frozen patch
def load_patch(zp):
    d = os.path.dirname(zp); b = os.path.basename(zp)[:-4]
    vn = os.path.join(d, b + "_vecnormalize.pkl")
    if not os.path.exists(vn):
        vn = os.path.join(d, "best_vecnormalize.pkl")
    p = PPO.load(zp, device="cpu")
    v = pickle.load(open(vn, "rb"))
    mean = v.obs_rms.mean.astype(np.float32); var = v.obs_rms.var.astype(np.float32)
    clip = float(v.clip_obs)

    def act(o):
        o = np.clip((np.asarray(o, np.float32) - mean) / np.sqrt(var + 1e-8), -clip, clip)
        a, _ = p.predict(o[None], deterministic=True)
        return np.asarray(a, np.float32).flatten()
    return act


patch_act = load_patch(zp)

env = PatchCarEnv(PatchEnvConfig(num_agents=2, render_mode=None, random_spawn=False,
                                 obs_mode="lidar", num_lidar_beams=108, map_name=args.map))
obs, _ = env.reset(seed=args.seed)
p0 = env.active_patches[0]
_, occ, res, origin = env.base_env.get_track_data()
occ = occ / 255.0 if occ is not None else None
# the patch policy may hold each action for K sim steps (PatchEnvConfig.action_repeat).
# one env.step() then advances the patch K*0.01 s -- the follower MPC decides once per
# env.step (== once per funnel update) and the follower PLANT is sub-stepped K x 0.01 s
# against a linearly-interpolated patch pose so the timelines stay aligned.
_KREP = max(1, int(getattr(getattr(env, "cfg", None), "action_repeat", 1)))
_KREP = int(os.environ.get("PATCH_ACTION_REPEAT", _KREP))
print(f"map={args.map} seed={args.seed} N={N} slots={SLOTS}  action_repeat={_KREP}  "
      f"patch spawn=({p0.x:.2f},{p0.y:.2f}) a/b=({p0.a:.2f},{p0.b:.2f})")


# ---- ground-truth progress from the map centerline (env lap_progress can be
#      wrong on serpentine maps; keep an independent honest measure) -----------
_CL = None
for _d in (args.map_dir, "maps", "f1tenth_gym/maps"):
    _c = os.path.join(_d, args.map, f"{args.map}_centerline.csv")
    if os.path.exists(_c):
        _CL = np.loadtxt(_c, delimiter=",", skiprows=1)[:, :2]
        break
_CL_CUM = None
if _CL is not None:
    _seg = np.hypot(np.diff(_CL[:, 0]), np.diff(_CL[:, 1]))
    _CL_CUM = np.concatenate([[0.0], np.cumsum(_seg)])


def true_progress(x, y):
    if _CL is None:
        return float("nan")
    k = int(np.argmin((_CL[:, 0] - x) ** 2 + (_CL[:, 1] - y) ** 2))
    return float(_CL_CUM[k] / _CL_CUM[-1])


# ---- oriented-bounding-box overlap (SAT) for real car-vs-car contact --------
def _corners(x, y, th, L=CAR_L, W=CAR_W):
    c, s = math.cos(th), math.sin(th)
    return np.array([(x + px * c - py * s, y + px * s + py * c)
                     for px, py in ((L/2, W/2), (L/2, -W/2), (-L/2, -W/2), (-L/2, W/2))])


def obb_overlap(a, b):
    """True if the two (x,y,theta) car rectangles intersect."""
    ca, cb = _corners(*a), _corners(*b)
    for poly in (ca, cb):
        for i in range(4):
            edge = poly[(i + 1) % 4] - poly[i]
            axis = np.array([-edge[1], edge[0]])
            axis = axis / (np.hypot(*axis) or 1.0)
            pa, pb = ca @ axis, cb @ axis
            if pa.max() < pb.min() or pb.max() < pa.min():
                return False
    return True


# ---------------------------------------------------------------- slot layout
# "wedge" (default): followers ride behind the patch car in a triangle so no
#   two slots — and no slot and the patch car — are within a car length.
#   N=1: dead astern.  N=2: two abreast.  N=3: two abreast + one centred behind.
#   N=4: two rows of two.
_WEDGE = {
    1: [(-GAP, 0.0)],
    2: [(-GAP, +SLOT_LAT), (-GAP, -SLOT_LAT)],
    3: [(-GAP, +SLOT_LAT), (-GAP, -SLOT_LAT), (-GAP - SLOT_D, 0.0)],
    4: [(-GAP, +SLOT_LAT), (-GAP, -SLOT_LAT),
        (-GAP - SLOT_D, +SLOT_LAT), (-GAP - SLOT_D, -SLOT_LAT)],
}


def _slot(i):
    if SLOTS == "wedge":
        tbl = _WEDGE.get(N, _WEDGE[3])
        return tbl[i] if i < len(tbl) else (-GAP - (i // 2) * SLOT_D,
                                            SLOT_LAT if i % 2 == 0 else -SLOT_LAT)
    if SLOTS == "trail":
        return (-(GAP + i * STAGGER), 0.0)
    if SLOTS == "abreast":
        return (-GAP - (i // 2) * SLOT_D, SLOT_LAT if i % 2 == 0 else -SLOT_LAT)
    k = i // 2 + 1                                # "split": spread fore/aft
    return ((1.0 if i % 2 == 0 else -1.0) * k * SLOT_D, 0.0)


def slot_along(i):
    return _slot(i)[0]


def slot_lat(i):
    return _slot(i)[1]


# ---------------------------------------------------------------- MPC followers
MPC_MODEL = os.environ.get("MPC_MODEL", "st").lower()          # st (default) | kinematic
MPC_IS_ST = MPC_MODEL == "st"
if MPC_IS_ST and FOLLOWER_MODEL != "st":
    sys.exit("MPC_MODEL=st requires FOLLOWER_MODEL=st (st-MPC drives the plant "
             "with steering-rate/accel, only wired for the st plant).")

# the single-track NLP is stiff -- it needs a finer discretisation than the
# kinematic one or IPOPT thrashes (dt<=0.025).  Model-aware horizon defaults:
_HZ_S = float(os.environ.get("MPC_HZ_S", "1.0" if MPC_IS_ST else "1.5"))
_HZ_N = int(os.environ.get("MPC_HZ_N", "40" if MPC_IS_ST else "15"))
_SUBSTEPS = int(os.environ.get("MPC_ST_SUBSTEPS", "1"))
_W_VEL = float(os.environ.get("MPC_W_VEL", "40.0" if MPC_IS_ST else "5.0"))
_W_CENTER = float(os.environ.get("MPC_W_CENTER", "90.0" if MPC_IS_ST else "80.0"))


def mk_solver():
    return SEMPCSolver(MPCConfig(
        v_max=V_HI, v_min=V_LO, accel_max=ACCEL_MAX, steering_max=0.4189,
        num_neighbors=max(1, N),                  # patch car + (N-1) other followers
        horizon_seconds=_HZ_S,
        horizon_steps=_HZ_N,
        w_vel=_W_VEL,
        w_center=_W_CENTER,
        w_contain=float(os.environ.get("MPC_W_CONTAIN", "1500.0")),
        max_iter=int(os.environ.get("MPC_MAX_ITER", "200")),
        min_agent_dist=MIN_DIST,
        model=MPC_MODEL,
        st_substeps=_SUBSTEPS,
        w_collision=float(os.environ.get("MPC_W_COLL", "200.0")),
        collision_hard=os.environ.get("MPC_COLL_HARD", "0") != "0",
        collision_hard_patch_only=os.environ.get("MPC_COLL_HARD", "0") == "patch",
        collision_radius=float(os.environ.get("MPC_COLL_R", "0.55")),
        w_collision_slack=float(os.environ.get("MPC_W_COLL_SLACK", "4000.0")),
    ))


solvers = [mk_solver() for _ in range(N)]


class _PatchShim:
    def __init__(self, p, vx, vy):
        self.x = self.y = 0.0
        self.theta = p.theta
        self.a, self.b, self.v = p.a, p.b, p.v
        self.vx, self.vy = vx, vy


_pth = [None]
_om = [0.0]
_pv = [max(p0.v, 0.5)]
_accel = [0.0]


def patch_kinematics(p, dt=DT):
    """Smoothed funnel yaw-rate + longitudinal accel, finite-differenced over the
    interval `dt` since the last call (one env.step == action_repeat*0.01 s)."""
    th0 = p.theta
    raw = 0.0 if _pth[0] is None else \
        ((th0 - _pth[0] + math.pi) % (2 * math.pi) - math.pi) / dt
    _pth[0] = th0
    _om[0] += 0.02 * (float(np.clip(raw, -2.5, 2.5)) - _om[0])
    a_raw = (p.v - _pv[0]) / dt
    _pv[0] = p.v
    _accel[0] += 0.05 * (float(np.clip(a_raw, -ACCEL_MAX, ACCEL_MAX)) - _accel[0])
    return _om[0], _accel[0]


def center_traj(p, i, n, dt, omega, accel):
    """Funnel-slot path over the horizon in the patch-at-origin frame.
    Propagates the patch's own longitudinal accel so a lagging follower aims at
    where the patch WILL be (same as mpc_follower_eval_n.py)."""
    th0 = p.theta
    v0 = max(p.v, 0.5)
    along_i, lat_i = slot_along(i), slot_lat(i)
    out = np.zeros((n, 2), np.float32)
    cx = cy = 0.0
    for k in range(n):
        h = th0 + omega * k * dt
        ux, uy = math.cos(h), math.sin(h)         # forward; left = (-uy, ux)
        out[k] = (cx + along_i * ux - lat_i * uy,
                  cy + along_i * uy + lat_i * ux)
        vk = min(max(v0 + accel * k * dt, 0.5), V_HI)
        cx += vk * ux * dt
        cy += vk * uy * dt
    return out


_fail = [0] * N
_hold = [[0.0, 2.0] for _ in range(N)]
_hold_k = [0] * N            # consecutive failed solves per agent (open-loop plan index)


def _from_plan(i, kidx):
    """Command taken directly from agent i's last successful MPC plan, kidx steps
    in.  On a failed re-solve the follower keeps executing that plan open-loop --
    there is no regulator, the controller is always the MPC."""
    Us = solvers[i].prev_U_sol
    Xs = solvers[i].prev_X_sol
    if Us is None:
        return None
    k = min(kidx, Us.shape[1] - 1)
    if MPC_IS_ST:
        return (float(np.clip(Us[1, k], -_STP["sv_max"], _STP["sv_max"])),
                float(np.clip(Us[0, k], -ACCEL_MAX, ACCEL_MAX)))
    kl = min(kidx + 3, Xs.shape[1] - 1)
    return (float(np.clip(Us[1, k], -0.4189, 0.4189)),
            float(np.clip(Xs[3, kl], V_LO, V_HI)))


def mpc_command(i, folls, p, omega, accel, do_solve):
    """Returns (u0, u1).  kinematic MPC -> (steering ANGLE, speed setpoint);
    st MPC -> (steering RATE, longitudinal accel) applied straight to the plant."""
    if not do_solve:
        return _hold[i][0], _hold[i][1]
    f = folls[i]
    if MPC_IS_ST:
        s = _FST[i]                                # true 7-D state, patch-at-origin frame
        x0 = np.array([s[0] - p.x, s[1] - p.y, s[2], s[3], s[4], s[5], s[6]], np.float64)
    else:
        x0 = np.array([f[0] - p.x, f[1] - p.y, f[2], f[3]], np.float32)
    pvx, pvy = p.v * math.cos(p.theta), p.v * math.sin(p.theta)
    ct = center_traj(p, i, solvers[i].config.horizon_steps + 1, solvers[i].dt, omega, accel)
    nbr = [[0.0, 0.0]]                             # patch car at origin
    nbr_v = [[pvx, pvy]]                           # ... moving with the funnel
    for j in range(N):
        if j != i:
            fj = folls[j]
            nbr.append([fj[0] - p.x, fj[1] - p.y])
            nbr_v.append([fj[3] * math.cos(fj[2]), fj[3] * math.sin(fj[2])])
    U, ok = solvers[i].solve(x0, _PatchShim(p, pvx, pvy), nbr,
                             center_traj=ct, neighbor_vels=nbr_v)
    if not ok or U is None:
        # NO regulator.  A failed re-solve keeps executing the last successful
        # MPC plan open-loop -- the follower is always MPC-driven, never handed
        # to a pure-pursuit / heuristic controller.
        _fail[i] += 1
        _hold_k[i] += 1
        cmd = _from_plan(i, _hold_k[i])
        if cmd is not None:
            _hold[i][0], _hold[i][1] = cmd
        return _hold[i][0], _hold[i][1]
    _hold_k[i] = 0
    if MPC_IS_ST:
        # U = [accel, steer_vel] -> hand the plant (steer_vel, accel) directly
        _hold[i][0] = float(np.clip(U[1], -_STP["sv_max"], _STP["sv_max"]))
        _hold[i][1] = float(np.clip(U[0], -ACCEL_MAX, ACCEL_MAX))
    else:
        Xs = solvers[i].prev_X_sol
        k_look = min(3, solvers[i].config.horizon_steps)
        delta = float(np.clip(U[1], -0.4189, 0.4189))
        v_cmd = float(Xs[3, k_look]) if Xs is not None else f[3] + float(U[0]) * solvers[i].dt
        _hold[i][0], _hold[i][1] = delta, float(np.clip(v_cmd, V_LO, V_HI))
    return _hold[i][0], _hold[i][1]


def step_follower(f, delta, v_cmd):
    x, y, th, v = f
    v += float(np.clip(v_cmd - v, -ACCEL_MAX * DT, ACCEL_MAX * DT))
    x += v * math.cos(th) * DT
    y += v * math.sin(th) * DT
    th += (v / WB) * math.tan(delta) * DT
    return np.array([x, y, th, v], np.float32)


def step_follower_st(i, c0, c1):
    """Advance follower i one DT with the f1tenth_gym single-track dynamic model.
    st MPC: (c0,c1) = (steer_vel, accel) applied raw.
    kinematic MPC: (c0,c1) = (steer ANGLE, speed) -> mapped through the f110 PIDs."""
    xs = _FST[i]
    if MPC_IS_ST:
        sv, ac = float(c0), float(c1)
    else:
        sv = float(pid_steer(c0, xs[2], _STP["sv_max"]))
        ac = float(pid_accl(c1, xs[3], _STP["a_max"], _STP["v_max"], _STP["v_min"]))
    xs = _st_rk4(xs, np.array([sv, ac], np.float64))
    _FST[i] = xs
    # global heading of travel = yaw + slip; report v as the body-frame speed
    return np.array([xs[0], xs[1], xs[4], max(abs(xs[3]), 1e-3)], np.float32)


def advance_follower(i, f, delta, v_cmd):
    return step_follower_st(i, delta, v_cmd) if FOLLOWER_MODEL == "st" \
        else step_follower(f, delta, v_cmd)


def dnorm(f, p):
    dx, dy = f[0] - p.x, f[1] - p.y
    xr = dx * math.cos(p.theta) + dy * math.sin(p.theta)
    yr = -dx * math.sin(p.theta) + dy * math.cos(p.theta)
    return math.hypot(xr / max(p.a, 1e-3), yr / max(p.b, 1e-3))


# ---------------------------------------------------------------- spawn in-formation
def spawn_pose(i):
    al, la = slot_along(i), slot_lat(i)
    ct, st = math.cos(p0.theta), math.sin(p0.theta)
    return np.array([p0.x + al * ct - la * st,
                     p0.y + al * st + la * ct,
                     p0.theta, max(p0.v, 0.5)], np.float32)


folls = [spawn_pose(i) for i in range(N)]
if FOLLOWER_MODEL == "st":
    # seed the 7-D single-track state from each 4-D spawn pose
    #   [x, y, delta=0, v, psi=heading, psi_dot=0, beta=0]
    _FST[:] = [np.array([f[0], f[1], 0.0, f[3], f[2], 0.0, 0.0], np.float64)
               for f in folls]
print(f"[follower plant] {FOLLOWER_MODEL}"
      + ("  (f1tenth_gym single-track RK4 + PID)" if FOLLOWER_MODEL == "st" else "")
      + f"   [MPC model] {MPC_MODEL}"
      + (f"  ({solvers[0].config.st_substeps} RK4 substeps/step)" if MPC_IS_ST else ""))
# report spawn spacing so we know problem 2 is out of the picture
sp = [np.hypot(folls[a][0] - folls[b][0], folls[a][1] - folls[b][1])
      for a in range(N) for b in range(a + 1, N)]
print(f"spawn pairwise follower dist: {[f'{d:.2f}' for d in sp] or 'n/a'}  "
      f"(MPC min_agent_dist={MIN_DIST})")


# ---------------------------------------------------------------- PHASE A: sim
REC = []
prog = 0.0                                         # env-reported progress
tprog = 0.0                                        # ground-truth centerline progress
reason = "max_steps"
n_all_inside = 0
n_inside = [0] * N
first_exit = None                                  # (step, tprog) of first "someone out"
min_gap = float("inf")                             # min centre-centre over ALL car pairs
min_clear = float("inf")                           # same, minus one car length (rough edge gap)
n_overlap = 0                                      # steps with an actual OBB car-car overlap
overlap_first = None

class _PInterp:
    """Patch pose linearly blended between the pre/post env.step snapshots."""
    __slots__ = ("x", "y", "theta", "a", "b", "v", "_vx", "_vy")

    def __init__(self, p0, p1, frac):
        f = frac
        self.x = p0["x"] + f * (p1["x"] - p0["x"])
        self.y = p0["y"] + f * (p1["y"] - p0["y"])
        d = ((p1["theta"] - p0["theta"] + math.pi) % (2 * math.pi)) - math.pi
        self.theta = p0["theta"] + f * d
        self.a = p0["a"] + f * (p1["a"] - p0["a"])
        self.b = p0["b"] + f * (p1["b"] - p0["b"])
        self.v = p0["v"] + f * (p1["v"] - p0["v"])
        self._vx = self.v * math.cos(self.theta)
        self._vy = self.v * math.sin(self.theta)

    def get_velocity_vector(self):
        return self._vx, self._vy


def _snap(p):
    return dict(x=float(p.x), y=float(p.y), theta=float(p.theta),
               a=float(p.a), b=float(p.b), v=float(p.v))


for step in range(1, args.steps + 1):
    p = env.active_patches[0]
    omega, accel = patch_kinematics(p, dt=_KREP * DT)
    do_solve = (step - 1) % args.mpc_every == 0
    # MPC decides once, against the funnel state at the start of this tick
    cmds = [mpc_command(i, folls, p, omega, accel, do_solve) for i in range(N)]

    p_prev = _snap(p)
    obs, r, term, trunc, info = env.step(patch_act(obs))
    prog = max(prog, float(info.get("lap_progress", prog)))
    p = env.active_patches[0]
    p_now = _snap(p)
    tprog = max(tprog, true_progress(p.x, p.y))

    # sub-step the follower PLANT K x 0.01 s against the interpolated patch,
    # holding the MPC command; score containment / contact at every sub-step
    all_in = True
    hit = False
    dns = [0.0] * N
    for sub in range(1, _KREP + 1):
        pin = _PInterp(p_prev, p_now, sub / _KREP)
        folls = [advance_follower(i, folls[i], cmds[i][0], cmds[i][1]) for i in range(N)]
        dns = [dnorm(folls[i], pin) for i in range(N)]
        sub_all_in = all(d <= 1.0 for d in dns)
        for i in range(N):
            n_inside[i] += int(dns[i] <= 1.0)
        n_all_inside += int(sub_all_in)
        all_in = all_in and sub_all_in
        if not sub_all_in and first_exit is None:
            first_exit = (step, tprog)
        cars = [(pin.x, pin.y, pin.theta)] + \
               [(float(f[0]), float(f[1]), float(f[2])) for f in folls]
        for a in range(len(cars)):
            for b in range(a + 1, len(cars)):
                d = math.hypot(cars[a][0] - cars[b][0], cars[a][1] - cars[b][1])
                min_gap = min(min_gap, d)
                min_clear = min(min_clear, d - CAR_L)
                if obb_overlap(cars[a], cars[b]):
                    hit = True
    n_substeps_total = step * _KREP
    if hit:
        n_overlap += 1
        if overlap_first is None:
            overlap_first = (step, tprog)

    f_steer = ([float(_FST[i][2]) for i in range(N)] if FOLLOWER_MODEL == "st"
               else [float(c[0]) for c in cmds])
    REC.append(dict(px=p.x, py=p.y, pth=p.theta, pa=p.a, pb=p.b, pv=p.v,
                    folls=[f.copy() for f in folls], steers=[c[0] for c in cmds],
                    f_steer=f_steer,
                    prog=prog, tprog=tprog, dns=dns, all_in=all_in, overlap=hit))

    if step % 200 == 0:
        print(f"[{step}] patch v={p.v:4.1f} true_prog={tprog:5.1%} | "
              f"dn={[f'{d:.2f}' for d in dns]} | all-inside "
              f"{100*n_all_inside/max(step*_KREP,1):.0f}% | min car gap {min_gap:.2f}m")
    if term or trunc:
        reason = info.get("termination_reason", "?")
        print(f"\nPATCH EPISODE END @ step {step}: {reason}  "
              f"(env prog {prog:.1%} / true prog {tprog:.1%})")
        break

try: env.close()
except Exception: pass
n_steps = len(REC)
completed = tprog >= 0.98
print(f"\n=== RESULT  N={N}  slots={SLOTS}  patch={tag}  map={args.map} ===")
print(f"steps {n_steps} | patch end reason '{reason}' | "
      f"true progress {tprog:.1%}  ({'COMPLETED' if completed else 'DID NOT COMPLETE'})")
_den = max(n_steps * _KREP, 1)   # containment is scored per 0.01 s sub-step
print(f"all-inside {n_all_inside}/{_den} ({100*n_all_inside/_den:.0f}%) | "
      f"per-follower inside {[f'{100*n_inside[i]/_den:.0f}%' for i in range(N)]}")
print(f"first 'someone out': {'never' if first_exit is None else f'step {first_exit[0]} @ {first_exit[1]:.1%}'}")
print(f"min car-car centre gap {min_gap:.2f} m  (min edge clearance ~{min_clear:+.2f} m; "
      f"car = {CAR_L}x{CAR_W} m)")
if n_overlap:
    print(f"** CAR-CAR OVERLAP on {n_overlap}/{n_steps} steps, first "
          f"step {overlap_first[0]} @ {overlap_first[1]:.1%} **")
else:
    print("no car-car overlap (OBB) at any step")
print(f"MPC solve fails per follower: {_fail}  "
      f"({100*sum(_fail)/max(n_steps*N,1):.0f}% overall)")


# ---------------------------------------------------------------- PHASE B: video
def ellipse_pts(cx, cy, th, a, b, n=40):
    t = np.linspace(0, 2 * math.pi, n)
    ct, st = math.cos(th), math.sin(th)
    ex, ey = a * np.cos(t), b * np.sin(t)
    return np.stack([cx + ex * ct - ey * st, cy + ex * st + ey * ct], axis=1)


fig, ax = plt.subplots(figsize=(8, 7))
PT = []
AT = [[] for _ in range(N)]


def draw_car(x, y, th, facecolor, edgecolor, z=10):
    """Draw a TRUE-footprint (0.58 x 0.31 m) oriented car body centred at (x, y)."""
    L, W = CAR_L, CAR_W
    c, s = math.cos(th), math.sin(th)
    body = [(L / 2, W / 2), (L / 2, -W / 2), (-L / 2, -W / 2), (-L / 2, W / 2)]
    nose = [(L / 2, W / 2), (L / 2 + 0.30 * L, 0.0), (L / 2, -W / 2)]
    for poly, fc in ((body, facecolor), (nose, edgecolor)):
        pts = [(x + px * c - py * s, y + px * s + py * c) for px, py in poly]
        ax.add_patch(plt.Polygon(pts, closed=True, facecolor=fc, edgecolor=edgecolor,
                                 lw=1.4, zorder=z, joinstyle="round"))


def frame(rec, done_reason=None):
    ax.clear()
    cx, cy = rec["px"], rec["py"]; m = max(rec["pa"], rec["pb"]) + 7
    if occ is not None:
        h, w = occ.shape
        a0 = max(0, int((cx - m - origin[0]) / res)); a1 = min(w, int((cx + m - origin[0]) / res))
        b0 = max(0, int((cy - m - origin[1]) / res)); b1 = min(h, int((cy + m - origin[1]) / res))
        if a1 > a0 and b1 > b0:
            reg = occ[b0:b1, a0:a1]
            rgba = np.zeros((*reg.shape, 4), np.uint8)
            rgba[reg < 0.5] = (40, 40, 40, 235); rgba[reg >= 0.5] = (225, 225, 225, 55)
            ax.imshow(rgba, extent=[a0 * res + origin[0], a1 * res + origin[0],
                                    b0 * res + origin[1], b1 * res + origin[1]],
                      origin="lower", zorder=0, interpolation="nearest")
    PT.append((rec["px"], rec["py"]))
    ax.plot(*zip(*PT[-500:]), "-", color="steelblue", lw=1, alpha=.6)
    ins_all = rec["all_in"]
    ax.add_patch(Ellipse((rec["px"], rec["py"]), rec["pa"] * 2, rec["pb"] * 2,
                         angle=np.degrees(rec["pth"]),
                         facecolor="cyan" if ins_all else "yellow",
                         edgecolor="darkblue" if ins_all else "red", alpha=.28, lw=2.5))
    bp = ellipse_pts(rec["px"], rec["py"], rec["pth"], rec["pa"], rec["pb"], 28)
    ax.plot(bp[:, 0], bp[:, 1], "k.", ms=3, alpha=.5)
    # patch car (agent P) — steel-blue body, true footprint
    draw_car(rec["px"], rec["py"], rec["pth"], facecolor="steelblue",
             edgecolor="navy", z=9)
    ax.plot([], [], "s", color="steelblue", mec="navy", ms=10, label="patch car")
    for i in range(N):
        f = rec["folls"][i]; AT[i].append((f[0], f[1]))
        ax.plot(*zip(*AT[i][-500:]), "-", color=COLORS[i % 6], lw=1, alpha=.5)
        ins = rec["dns"][i] <= 1.0
        draw_car(f[0], f[1], f[2], facecolor=COLORS[i % 6],
                 edgecolor="black" if ins else "red", z=11)
        ax.plot([], [], "s", color=COLORS[i % 6], mec="black", ms=10,
                label=f"follower {i}" + ("" if ins else " (OUT)"))
    ax.set_aspect("equal"); ax.set_xlim(cx - m + 2, cx + m - 2); ax.set_ylim(cy - m + 2, cy + m - 2)
    ax.legend(loc="upper right", fontsize=8)
    ttl = (f"native patch + decentralised NMPC   N={N}  slots={SLOTS}\n"
           f"patch v={rec['pv']:.1f} m/s  a/b=({rec['pa']:.2f},{rec['pb']:.2f})  "
           f"true progress {rec['tprog']:.0%}\n"
           f"dn=[{', '.join(f'{d:.2f}' for d in rec['dns'])}]  "
           f"{'ALL INSIDE' if ins_all else 'SOMEONE OUT'}"
           f"{'   CAR OVERLAP' if rec.get('overlap') else ''}")
    if done_reason:
        ttl += f"   |   PATCH {done_reason.upper()}"
    ax.set_title(ttl, fontsize=9)
    fig.canvas.draw()
    return np.ascontiguousarray(
        np.frombuffer(fig.canvas.buffer_rgba(), np.uint8).reshape(
            fig.canvas.get_width_height()[::-1] + (4,))[..., :3])


# ---- f1tenth_gym pygame renderer (real car sprites) ---------------------
#   A render-only F110 env with N+1 agents (0 = patch car, 1..N = followers).
#   We never step its physics -- each frame we push the recorded poses straight
#   into its renderer, exactly like mpc_follower_eval_n.py's f1tenth path.
_f1 = {}


def _f1_setup():
    F1_ZOOM = float(os.environ.get("F1_ZOOM", "3.4"))
    try:                                              # tweak spec before canvases build
        from f1tenth_gym.envs.rendering.renderer import RenderSpec as _RS
        _orig = _RS.__init__
        _pal = ["#1f77ff", "#e6194b", "#3cb44b", "#f58231", "#911eb4", "#ffe119"]

        def _patched(self, *a, **k):
            _orig(self, *a, **k)
            self.zoom_in_factor = F1_ZOOM
            self.show_info = False
            self.car_tickness = 2
            self.vehicle_palette = _pal              # agent 0 = patch (blue), 1.. = followers
        _RS.__init__ = _patched
    except Exception:
        pass
    from envs.f110_env import F110EnvAdapter, F110Config
    ad = F110EnvAdapter(F110Config(map_name=args.map, num_agents=N + 1),
                        render_mode="rgb_array")
    ad.ensure_initialized()
    spawn = np.array([[REC[0]["px"], REC[0]["py"], REC[0]["pth"]]]
                     + [[f[0], f[1], f[2]] for f in REC[0]["folls"]], np.float64)
    ad.reset(poses=spawn)
    u = ad.base_env.unwrapped
    rnd = u.renderer
    try:
        rnd.follow_agent_flag = True
        rnd.agent_to_follow = 0                       # patch car
        rnd.active_map_renderer = "car"
    except Exception:
        pass

    def _funnel_cb(r):
        rc = _f1["rec"]
        pts = ellipse_pts(rc["px"], rc["py"], rc["pth"], rc["pa"], rc["pb"], 60).astype(np.float32)
        col = (0, 170, 190) if rc["all_in"] else (230, 60, 60)
        try:
            r.render_closed_lines(pts, color=col, size=2)
        except Exception:
            pass
    rnd.add_renderer_callback(_funnel_cb)
    _f1["ad"], _f1["u"] = ad, u


def f1_frame(rec):
    if "ad" not in _f1:
        _f1_setup()
    _f1["rec"] = rec
    u = _f1["u"]
    px = np.array([rec["px"]] + [f[0] for f in rec["folls"]], np.float64)
    py = np.array([rec["py"]] + [f[1] for f in rec["folls"]], np.float64)
    pth = np.array([rec["pth"]] + [f[2] for f in rec["folls"]], np.float64)
    steer = np.array([0.0] + list(rec["f_steer"]), np.float64)
    u.render_obs = {
        "ego_idx": 0, "poses_x": px, "poses_y": py, "poses_theta": pth,
        "steering_angles": steer, "collisions": np.zeros(N + 1),
        "lap_times": np.zeros(N + 1), "lap_counts": np.zeros(N + 1),
        "sim_time": float(rec["tprog"]),
    }
    fr = _f1["ad"].base_env.render()
    return np.ascontiguousarray(np.asarray(fr)[..., :3])


def _fit(img, H):
    import numpy as _np
    h, w = img.shape[:2]
    nw = max(1, int(round(w * H / h)))
    ys = (_np.linspace(0, h - 1, H)).astype(int)
    xs = (_np.linspace(0, w - 1, nw)).astype(int)
    return img[ys][:, xs]


writer = imageio.get_writer(OUT_MP4, fps=25, codec="libx264", quality=8,
                            macro_block_size=None)
idxs = list(range(0, n_steps, args.every))
last_reason = reason if reason != "max_steps" else None
fr = None
for j, i in enumerate(idxs):
    dr = last_reason if i >= n_steps - args.every else None
    if args.render == "mpl":
        fr = frame(REC[i], dr)
    elif args.render == "f1tenth":
        fr = f1_frame(REC[i])
    else:
        a_ = frame(REC[i], dr)
        try:
            b_ = f1_frame(REC[i])
            H = max(a_.shape[0], b_.shape[0])
            fr = np.hstack([_fit(a_, H), _fit(b_, H)])
        except Exception as e:
            if j == 0:
                print(f"  [f1tenth panel disabled: {e}]")
            fr = a_
    writer.append_data(fr)
    if j % 50 == 0:
        print(f"  frame {j}/{len(idxs)}")
if fr is not None:
    for _ in range(20):
        writer.append_data(fr)
writer.close(); plt.close(fig)
if "ad" in _f1:
    try: _f1["ad"].close()
    except Exception: pass
print("video:", os.path.abspath(OUT_MP4))
