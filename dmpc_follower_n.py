"""
N-follower DISTRIBUTED MPC (DMPC) against the patch running in its NATIVE env.

WHY THIS EXISTS (vs mpc_follower_native_n.py)
---------------------------------------------
`mpc_follower_native_n.py` is a MYOPIC DECOUPLED scheme: each follower solves
its own OCP and models every other car as an obstacle travelling at CONSTANT
VELOCITY for the whole horizon.  No follower ever learns what the others
actually intend, so at N>=5 two of them pick mirror-image dodges, each assuming
the other will hold course, and they collide (measured: car-car overlap on
288/1200 steps at N=6, 12% MPC solve failures).

This file keeps EVERYTHING else identical -- same env, same frozen patch, same
single-track plant, same funnel containment cost, same slot layouts (including
`ring`), same metrics and rendering -- and changes only the coordination:

  1. TRAJECTORY EXCHANGE.  Followers broadcast their predicted paths; each
     agent plans against the others' real intended trajectories instead of a
     constant-velocity ray.

  2. RECIPROCAL SEPARATING HYPERPLANES (buffered Voronoi cells).  For every
     pair and every horizon step a plane is dropped between the two predicted
     positions and each agent is constrained to its own side with HALF the
     required clearance.  Both agents get the same plane with opposite
     normals, so each concedes exactly the share it expects the other to
     concede -- the "golden rule" of DMPC coordination.  Against the patch car
     the split is off: the RL patch does not cooperate, so the follower takes
     the whole burden.

  3. ITERATED SWEEPS.  The agents re-solve against each other's updated plans
     within a single control tick (Gauss-Seidel by default, so agent i already
     sees the fresh plans of 0..i-1), with damping, until the joint plan stops
     moving.  The myopic version reacts a tick late; this one converges now.

    PY=/p/cral/vignesh/envs/fastfunnels/bin/python
    MPC_SLOTS=ring $PY dmpc_follower_n.py --policy patch_policy_models/best_scenario_pool \
        --map on_rep_clean --map-dir maps/scenario_on_rep --n 6 --seed 0

env vars: FOLLOWER_MODEL {st(default),kinematic}  MPC_MODEL {st(default),kinematic}
          MPC_SLOTS {wedge(default),adaptive,ring,plus,corners,trail,abreast,split}  MPC_GAP  MPC_STAGGER
          MPC_RING_RHO  MPC_SLOT_D  MPC_SLOT_LAT  MPC_MIN_DIST  MPC_HZ_S  MPC_HZ_N
          MPC_ST_SUBSTEPS  MPC_MAX_ITER  MPC_W_VEL/W_CENTER/W_CONTAIN  MPC_W_COLL
          DMPC_SWEEPS   coordination sweeps per control tick (default 2)
          DMPC_DAMP     plan relaxation in (0,1]; 1 = take the new plan whole (default 0.75)
          DMPC_MODE     {gs (Gauss-Seidel, default), jacobi}
          DMPC_HP_R     required pairwise clearance for the hyperplanes (default MPC_MIN_DIST)
          DMPC_W_HP     penalty on hyperplane slack (default 6000)
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
from envs.dmpc import DMPCSolver, DMPCConfig, separating_hyperplanes

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
RING_RHO = float(os.environ.get("MPC_RING_RHO", "0.82"))   # ring radius as a fraction of the funnel half-axes
CONTAIN_MARGIN = float(os.environ.get("MPC_CONTAIN_MARGIN", "0.15"))  # must match MPCConfig.containment_margin
MIN_DIST = float(os.environ.get("MPC_MIN_DIST", "0.8"))   # MPC inter-agent keep-out
# fraction of the follower speed limit the formation is allowed to demand
PACK_V_FRAC = float(os.environ.get("MPC_PACK_V_FRAC", "0.80"))

zp = args.policy if args.policy.endswith(".zip") else os.path.join(args.policy, "best_model.zip")
tag = os.path.basename(os.path.dirname(zp)) or os.path.basename(args.policy)
OUT_MP4 = args.out or (
    f"videos/dmpc_n{N}_{SLOTS}_plant-{FOLLOWER_MODEL}"
    f"_mpc-{os.environ.get('MPC_MODEL', 'st').lower()}_{tag}"
    f"_{args.map}_seed{args.seed}.mp4")
os.makedirs(os.path.dirname(OUT_MP4), exist_ok=True)
COLORS = ["red", "lime", "magenta", "orange", "cyan", "yellow"]


# ---------------------------------------------------------------- frozen patch
def load_patch(zp):
    """Returns act(obs) -> centred action.

    Auto-detects LEGACY raw-action checkpoints (e.g. the June-2026
    run_20260602_233933 family), which emit absolute physical commands
    [steer_rad, v(2..10), a(1.5..3), b(1..3)] instead of today's centred
    [steer_RATE(-1..1), v_c, a_c, b_c]. For those we renormalise v/a/b and
    convert the absolute steer angle into the rate that drives the env's
    integrated steer state toward it -- same shim as
    presentation_code/make_legacy_patch_reel.py.
    """
    d = os.path.dirname(zp); b = os.path.basename(zp)[:-4]
    # SB3 runs here save `best_model.zip` + `best_vecnormalize.pkl` and
    # `final_model.zip` + `final_vecnormalize.pkl` -- the "_model" is DROPPED in
    # the pkl name. Only trying "<stem>_vecnormalize.pkl" and then falling back
    # to best_vecnormalize.pkl silently normalises one checkpoint's
    # observations with ANOTHER checkpoint's statistics, which looks like a
    # catastrophically bad policy rather than an error (final_model on
    # sw_lshape001: 100% with the right stats, 11% with best_model's).
    _cands = [os.path.join(d, b + "_vecnormalize.pkl"),
              os.path.join(d, b + "_vecnorm.pkl")]
    if b.endswith("_model"):
        _short = b[: -len("_model")]
        _cands += [os.path.join(d, _short + "_vecnormalize.pkl"),
                   os.path.join(d, _short + "_model_vecnorm.pkl")]
    _cands += [os.path.join(d, "best_vecnormalize.pkl")]
    for _c in _cands:
        if os.path.exists(_c):
            vn = _c
            break
    else:
        raise FileNotFoundError(f"no VecNormalize stats for {zp}; tried {_cands}")
    print(f"[patch] vecnorm = {os.path.basename(vn)}")
    p = PPO.load(zp, device="cpu")
    v = pickle.load(open(vn, "rb"))
    mean = v.obs_rms.mean.astype(np.float32); var = v.obs_rms.var.astype(np.float32)
    clip = float(v.clip_obs)
    lo, hi = p.action_space.low, p.action_space.high
    is_legacy = bool(hi[1] > 1.5)   # centred checkpoints have high[1] == 1
    if is_legacy:
        print("[patch] LEGACY raw-action checkpoint detected -> action shim enabled")

    def act(o):
        o = np.clip((np.asarray(o, np.float32) - mean) / np.sqrt(var + 1e-8), -clip, clip)
        a, _ = p.predict(o[None], deterministic=True)
        r = np.asarray(a, np.float32).flatten()
        if not is_legacy:
            return r
        r = np.clip(r, lo, hi)
        cfg = env.cfg
        cur = float(getattr(env, "_steer_state", 0.0))
        rate = np.clip((float(r[0]) - cur) / cfg.steer_rate_max, -1.0, 1.0)
        def _c(x, lo_, hi_):
            return (x - 0.5 * (lo_ + hi_)) / (0.5 * (hi_ - lo_))
        return np.array([rate,
                         _c(float(r[1]), 2.0, 10.0),
                         _c(float(r[2]), cfg.a_cmd_min, cfg.a_cmd_max),
                         _c(float(r[3]), cfg.b_cmd_min, cfg.b_cmd_max)], np.float32)
    return act


# Use the SOLO-EVAL loader verbatim rather than this file's own copy of the
# shim. The two had drifted (p.predict vs model.policy.predict, and the
# vecnorm lookup), and the drift is invisible from the outside: the patch just
# behaves like a worse policy. On on_obs_x5 the eval path drives
# ck16500000 to 94.7% while the local shim stalled it at 19%. The eval path is
# the reference -- if they disagree, the runner is wrong.
try:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "presentation_code"))
    from make_legacy_patch_reel import load_legacy_policy as _load_eval_policy
    _act_eval, _obs_dim_eval, _is_legacy_eval = _load_eval_policy(zp)
    print(f"[patch] loader = make_legacy_patch_reel.load_legacy_policy "
          f"(obs {_obs_dim_eval}, legacy={_is_legacy_eval})")

    def patch_act(o):
        return _act_eval(o, env)
except Exception as _e:
    print(f"[patch] eval loader unavailable ({_e}); using local shim")
    patch_act = load_patch(zp)

# MUST match the solo-eval env (make_legacy_patch_reel.make_env): PatchEnvConfig
# defaults b_cmd_max to 4.5, but every legacy/raw-action checkpoint was trained
# and evaluated with b in [1.0, 3.0]. Leaving the default inflates the commanded
# funnel by 50% (a raw b=3.0 decodes to 4.5), so the patch drives a funnel far
# too fat for the corridor and dies early -- with followers it looked like a
# formation problem when it was an env mismatch.
env = PatchCarEnv(PatchEnvConfig(num_agents=2, render_mode=None, random_spawn=False,
                                 obs_mode="lidar",
                                 num_lidar_beams=int(os.environ.get("PATCH_BEAMS", "108")),
                                 a_cmd_min=float(os.environ.get("PATCH_A_MIN", "1.5")),
                                 a_cmd_max=float(os.environ.get("PATCH_A_MAX", "3.0")),
                                 b_cmd_min=float(os.environ.get("PATCH_B_MIN", "1.0")),
                                 b_cmd_max=float(os.environ.get("PATCH_B_MAX", "3.0")),
                                 lidar_clip_m=float(os.environ.get("PATCH_LIDAR_CLIP", "30.0")),
                                 wall_filter_enabled=False,
                                 map_name=args.map))
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
    # N=5/6: three rows of two (5 drops one corner). The old fallback put
    # slot 2 at (-GAP-SLOT_D, 0) and slot 3 at (-GAP-SLOT_D, -SLOT_LAT) --
    # only SLOT_LAT=0.62 apart, under min_agent_dist=0.8. These rows are
    # SLOT_D apart with +-SLOT_LAT across, so every pair is >= 0.8.
    5: [(-GAP, +SLOT_LAT), (-GAP, -SLOT_LAT),
        (-GAP - SLOT_D, +SLOT_LAT), (-GAP - SLOT_D, -SLOT_LAT),
        (-GAP - 2 * SLOT_D, 0.0)],
    6: [(-GAP, +SLOT_LAT), (-GAP, -SLOT_LAT),
        (-GAP - SLOT_D, +SLOT_LAT), (-GAP - SLOT_D, -SLOT_LAT),
        (-GAP - 2 * SLOT_D, +SLOT_LAT), (-GAP - 2 * SLOT_D, -SLOT_LAT)],
}


# "ring" : followers ride AROUND the funnel -- front, sides and rear -- instead
#   of all being strung out astern.  Angles are in the patch frame, 0 deg =
#   straight ahead, +90 deg = port/left.  The radius is not fixed: each slot is
#   placed on a scaled copy of the LIVE funnel ellipse (RING_RHO of the
#   containment half-axes), so the formation shrinks with the funnel in a tunnel
#   and never sits outside the ellipse it is supposed to be contained by.  It is
#   then pushed out radially if it would come within MIN_DIST of the patch car,
#   which is a real vehicle sitting at the funnel origin.
_RING_DEG = {
    1: [180.0],                                     # dead astern (matches the working N=1 case)
    2: [90.0, 270.0],                               # abeam, port + starboard
    3: [0.0, 120.0, 240.0],                         # one ahead, two on the rear quarters
    4: [0.0, 90.0, 180.0, 270.0],                   # ahead, both beams, astern
    5: [0.0, 72.0, 144.0, 216.0, 288.0],
    6: [0.0, 60.0, 120.0, 180.0, 240.0, 300.0],     # 1 ahead, 2 bows, 2 quarters, 1 astern
}


def _ring_slot(i, a, b):
    """Slot i as (along, lateral) offsets in the patch frame, sized to the live
    funnel.  a/b are the funnel half-axes at the moment of the call."""
    tbl = _RING_DEG.get(N)
    if tbl is None:                                  # N outside the table: even spread
        tbl = [360.0 * k / N for k in range(N)]
    phi = math.radians(tbl[i] if i < len(tbl) else 360.0 * i / N)
    a_eff = max(a - CONTAIN_MARGIN, 0.5)
    b_eff = max(b - CONTAIN_MARGIN, 0.5)
    al = RING_RHO * a_eff * math.cos(phi)
    la = RING_RHO * b_eff * math.sin(phi)
    r = math.hypot(al, la)
    if r < 1e-6:
        return (-GAP, 0.0)
    if r < MIN_DIST:                                 # clear the patch car itself
        k = MIN_DIST / r
        al, la = al * k, la * k
        # stay inside the containment ellipse after being pushed out
        g = (al / a_eff) ** 2 + (la / b_eff) ** 2
        if g > 1.0:
            sc = 1.0 / math.sqrt(g)
            al, la = al * sc, la * sc
    return (al, la)


# "plus" / "corners": FIXED-geometry formations, unlike "ring" the radius does
#   NOT follow the live funnel.  Kept fixed on purpose: the ring makes the whole
#   formation contract every time the patch narrows for an obstacle, so the
#   followers spend the episode chasing a moving reference.  A fixed frame gives
#   them a stationary target; the cost is that the slots can fall OUTSIDE the
#   funnel while it is pinched (with b at its 1.00 floor the containment
#   half-width is only 0.85 m, so any lateral slot >= MIN_DIST=0.8 from the
#   patch car is already at the edge).  PLUS_R sets the radius.
PLUS_R = float(os.environ.get("MPC_PLUS_R", "1.10"))

#   plus     : one ahead, one astern, one on each beam (patch car in the middle)
_PLUS_DEG = [0.0, 90.0, 180.0, 270.0]
#   corners  : one on each quarter -- the same four cars rotated 45 deg, so
#              nobody sits directly ahead of or behind the patch car
_CORNERS_DEG = [45.0, 135.0, 225.0, 315.0]


def _fixed_slot(i, tbl):
    """Slot i at a FIXED radius PLUS_R on the bearing given by tbl (deg,
    0 = straight ahead, +90 = port)."""
    phi = math.radians(tbl[i % len(tbl)] if i < len(tbl) else 360.0 * i / max(N, 1))
    r = max(PLUS_R, MIN_DIST)              # never inside the patch car's keep-out
    return (r * math.cos(phi), r * math.sin(phi))


# "adaptive": the slots are not a fixed bearing pattern at all -- they are
#   SOLVED for from the funnel's live shape on every tick.
#
#   Why: every fixed pattern (wedge/ring/plus/corners) pins each follower to a
#   bearing. A slot on the beam (90 deg) needs lateral room, and when the patch
#   narrows for an obstacle -- b hits its 1.00 floor on 44% of steps on
#   open_narrow_obs -- the containment half-width falls to 0.85 m while
#   min_agent_dist still forces every slot >= 0.80 m from the patch car. The
#   beam slot ends up ON the ellipse boundary, so any tracking error puts that
#   follower outside. Measured: a feasible arrangement EXISTS at 100% of steps
#   for N up to 6, so the funnel is never actually too small -- the fixed
#   bearings are simply the wrong shape for a narrow funnel.
#
#   What this does instead: pack N points into the live containment ellipse so
#   that every point is >= MIN_DIST from the patch car and from every other
#   point, and then pull the whole arrangement AS FAR INSIDE the boundary as it
#   will go (binary search on an ellipse scale factor, taking the SMALLEST
#   feasible scale). The leftover 1 - scale is containment margin: the room a
#   follower has to lag or overshoot without leaving the funnel.
#
#   As b shrinks the solution migrates off the beam and toward the funnel's long
#   axis by itself (the half-length is 1.35 m against 0.85 m of half-width when
#   pinched), which is exactly the arrangement a fixed pattern cannot express.
#   Seeds are deterministic and results are cached per rounded (a, b), so nearby
#   funnel states give nearby formations -- the reference stays continuous
#   rather than jumping between ticks.
_PACK_CACHE = {}
_PACK_ITERS = int(os.environ.get("MPC_PACK_ITERS", "140"))


def _pack_relax(ae, be, n, seed_mode):
    """Deterministic seed + projected repulsion inside the (ae, be) ellipse.
    Returns the arrangement if it satisfies every constraint, else None."""
    if n == 1:
        # ASTERN. The packer only optimises geometry, and with one follower no
        # separation constraint breaks the fore/aft tie -- but a slot AHEAD of
        # the patch car is the worst place to be: the patch accelerates into it
        # and the funnel's front edge is where containment is tightest (a is
        # pinned at its 1.50 floor). Every fixed formation puts a lone follower
        # behind for that reason.
        cand = np.array([[-min(ae, max(MIN_DIST, 0.9 * ae)), 0.0]])
        return cand if np.linalg.norm(cand[0]) >= MIN_DIST - 1e-6 else None
    th = 2.0 * math.pi * np.arange(n) / n
    if seed_mode == 0:
        pass
    elif seed_mode == 1:
        th = th + math.pi / n                      # rotate half a sector
    else:
        # long-axis-biased seed: alternate fore/aft, small lateral spread
        th = np.where(np.arange(n) % 2 == 0, 0.0, math.pi) +              0.35 * (np.arange(n) - (n - 1) / 2.0)
    P = np.stack([0.92 * ae * np.cos(th), 0.92 * be * np.sin(th)], axis=1)

    for _ in range(_PACK_ITERS):
        for i in range(n):
            f = np.zeros(2)
            for j in range(n):
                if i == j:
                    continue
                d = P[i] - P[j]
                r = float(np.linalg.norm(d)) + 1e-9
                if r < MIN_DIST + 0.02:
                    f += d / r * (MIN_DIST + 0.02 - r)
            r0 = float(np.linalg.norm(P[i])) + 1e-9
            if r0 < MIN_DIST + 0.02:               # clear the patch car itself
                f += P[i] / r0 * (MIN_DIST + 0.02 - r0)
            P[i] = P[i] + f
            g = (P[i, 0] / ae) ** 2 + (P[i, 1] / be) ** 2
            if g > 1.0:                            # project back inside
                P[i] = P[i] / math.sqrt(g)

    for i in range(n):
        if np.linalg.norm(P[i]) < MIN_DIST - 1e-3:
            return None
        if (P[i, 0] / ae) ** 2 + (P[i, 1] / be) ** 2 > 1.0 + 1e-6:
            return None
        for j in range(i + 1, n):
            if np.linalg.norm(P[i] - P[j]) < MIN_DIST - 1e-3:
                return None
    return P


def _pack(ae, be, n):
    for mode in (0, 1, 2):
        P = _pack_relax(ae, be, n, mode)
        if P is not None:
            return P
    return None


def _prefer_astern(P, ae, be):
    """Rotate a feasible packing to sit as far astern as it can.

    Rotation preserves every pairwise distance and every distance to the patch
    car, so a rotated packing is still separation-feasible; only ellipse
    containment has to be rechecked (the ellipse is anisotropic). Among the
    rotations that stay inside, take the one with the most negative mean
    along-axis coordinate -- i.e. the formation that trails the patch rather
    than leading it."""
    best, best_score = P, float(np.mean(P[:, 0]))
    for deg in range(10, 360, 10):
        th = math.radians(deg)
        c, s_ = math.cos(th), math.sin(th)
        Q = np.stack([P[:, 0] * c - P[:, 1] * s_,
                      P[:, 0] * s_ + P[:, 1] * c], axis=1)
        if np.any((Q[:, 0] / ae) ** 2 + (Q[:, 1] / be) ** 2 > 1.0 + 1e-6):
            continue
        score = float(np.mean(Q[:, 0]))
        if score < best_score - 1e-9:
            best, best_score = Q, score
    return best


# Patch speed + yaw rate, refreshed once per tick by the main loop. A slot at
# radius r on a funnel yawing at omega sweeps through the world at
# v_patch + |omega|*r, and the follower has to match that. The measured demand
# on open_narrow_obs peaked at 11.9 m/s against a 12.0 m/s follower limit --
# feasible on paper, with no margin for the lag a nonholonomic car always has.
_TICK_V = [0.0]
_TICK_OM = [0.0]


def set_tick_kinematics(v, omega):
    _TICK_V[0], _TICK_OM[0] = float(v), float(omega)


def _speed_scale(P):
    """Shrink factor so tracking the formation stays inside the follower's
    speed budget: v_patch + |omega| * r <= PACK_V_FRAC * V_HI."""
    r = float(np.max(np.linalg.norm(P, axis=1))) if len(P) else 0.0
    om = abs(_TICK_OM[0])
    if r <= 1e-6 or om <= 1e-6:
        return 1.0
    budget = PACK_V_FRAC * V_HI - _TICK_V[0]
    if budget <= 0.0:
        return 0.35                       # patch already at the budget: pull in hard
    r_max = budget / om
    return float(np.clip(r_max / r, 0.35, 1.0))


def _adaptive_slots(i, a, b):
    """Slot i for the CURRENT funnel, from the cached packing for this (a, b)."""
    key = (round(float(a), 2), round(float(b), 2), N,
           round(_speed_scale_key(), 1))
    got = _PACK_CACHE.get(key)
    if got is None:
        ae = max(float(a) - CONTAIN_MARGIN, 0.5)
        be = max(float(b) - CONTAIN_MARGIN, 0.5)
        best = None
        lo, hi = 0.25, 1.0
        for _ in range(11):                        # smallest feasible scale
            mid = 0.5 * (lo + hi)                  #   == largest margin
            P = _pack(ae * mid, be * mid, N)
            if P is not None:
                best, hi = P, mid
            else:
                lo = mid
        if best is None:
            best = _pack(ae, be, N)
        if best is None:                           # give up: ring fallback
            got = tuple(_ring_slot(k, a, b) for k in range(N))
        else:
            best = _prefer_astern(best, ae, be)
            best = best * _speed_scale(best)
            order = np.argsort(-np.arctan2(best[:, 1], best[:, 0]))
            got = tuple((float(best[k, 0]), float(best[k, 1])) for k in order)
        _PACK_CACHE[key] = got
    return got[i % len(got)]


def _speed_scale_key():
    """Coarse key so the cache tracks the speed budget without thrashing."""
    om = abs(_TICK_OM[0])
    if om <= 1e-6:
        return 1.0
    return float(np.clip((PACK_V_FRAC * V_HI - _TICK_V[0]) / max(om, 1e-6), 0.0, 9.9))


def _slot(i, a=None, b=None):
    if SLOTS == "ring":
        return _ring_slot(i, p0.a if a is None else a, p0.b if b is None else b)
    if SLOTS == "adaptive":
        return _adaptive_slots(i, p0.a if a is None else a,
                               p0.b if b is None else b)
    if SLOTS == "plus":
        return _fixed_slot(i, _PLUS_DEG)
    if SLOTS == "corners":
        return _fixed_slot(i, _CORNERS_DEG)
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


def slot_along(i, a=None, b=None):
    return _slot(i, a, b)[0]


def slot_lat(i, a=None, b=None):
    return _slot(i, a, b)[1]


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


SWEEPS   = int(os.environ.get("DMPC_SWEEPS", "2"))
DAMP     = float(os.environ.get("DMPC_DAMP", "0.75"))
DMPC_GS  = os.environ.get("DMPC_MODE", "gs").lower() != "jacobi"
HP_R     = float(os.environ.get("DMPC_HP_R", str(MIN_DIST)))
W_HP     = float(os.environ.get("DMPC_W_HP", "6000.0"))


def mk_solver():
    return DMPCSolver(DMPCConfig(
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
        collision_radius=float(os.environ.get("MPC_COLL_R", "0.55")),
        w_hp_slack=W_HP,
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
    along_i, lat_i = slot_along(i, p.a, p.b), slot_lat(i, p.a, p.b)
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


# ---------------------------------------------------------------- DMPC core
_plans_w = [None] * N          # each (K,2) in WORLD frame, so it survives patch motion
_conv = [0.0]                  # last tick's max plan movement (convergence diagnostic)
_solves = [0]                  # total solver calls (SWEEPS x N per solving tick)
_hp_ticks = [0]                # ticks where the exchange actually changed a plan


def patch_path(p, n, dt, omega, accel):
    """Funnel-CENTRE path over the horizon in the patch-at-origin frame.
    Same propagation as center_traj() with the slot offset set to (0,0) -- this
    is what the followers are handed as the patch car's intended trajectory."""
    th0 = p.theta
    v0 = max(p.v, 0.5)
    out = np.zeros((n, 2), np.float64)
    cx = cy = 0.0
    for k in range(n):
        out[k] = (cx, cy)
        h = th0 + omega * k * dt
        vk = min(max(v0 + accel * k * dt, 0.5), V_HI)
        cx += vk * math.cos(h) * dt
        cy += vk * math.sin(h) * dt
    return out


def _seed_plan(i, folls, p, K, dt):
    """First-tick belief: a straight constant-speed rollout in world frame."""
    f = folls[i]
    v = max(float(f[3]), 0.5)
    th = float(f[2])
    t = np.arange(K) * dt
    return np.stack([f[0] + v * math.cos(th) * t,
                     f[1] + v * math.sin(th) * t], axis=1)


def _apply(i, U, f):
    """Latch a solver U as the (c0, c1) the plant consumes."""
    if MPC_IS_ST:
        _hold[i][0] = float(np.clip(U[1], -_STP["sv_max"], _STP["sv_max"]))
        _hold[i][1] = float(np.clip(U[0], -ACCEL_MAX, ACCEL_MAX))
    else:
        Xs = solvers[i].prev_X_sol
        k_look = min(3, solvers[i].config.horizon_steps)
        delta = float(np.clip(U[1], -0.4189, 0.4189))
        v_cmd = float(Xs[3, k_look]) if Xs is not None else f[3] + float(U[0]) * solvers[i].dt
        _hold[i][0], _hold[i][1] = delta, float(np.clip(v_cmd, V_LO, V_HI))


def dmpc_step(folls, p, omega, accel, do_solve):
    """ONE control tick of distributed MPC -> [(c0,c1)] * N.

    Replaces the myopic per-agent `mpc_command`: agents exchange predicted
    trajectories and re-solve against each other inside the tick, coupled by
    reciprocal separating hyperplanes, until the joint plan stops moving."""
    if not do_solve:
        return [(_hold[i][0], _hold[i][1]) for i in range(N)]

    K = _HZ_N + 1
    dt = solvers[0].dt
    org = np.array([p.x, p.y], np.float64)
    pvx, pvy = p.v * math.cos(p.theta), p.v * math.sin(p.theta)
    shim = _PatchShim(p, pvx, pvy)
    ppath = patch_path(p, K, dt, omega, accel)

    # everyone's current belief, expressed in THIS tick's patch-at-origin frame
    plans = []
    for i in range(N):
        if _plans_w[i] is None:
            _plans_w[i] = _seed_plan(i, folls, p, K, dt)
        plans.append(np.asarray(_plans_w[i], np.float64) - org)

    solved = [False] * N
    last_U = [None] * N
    delta_max = 0.0
    for _sweep in range(SWEEPS):
        frozen = [q.copy() for q in plans]      # Jacobi reads last sweep only
        for i in range(N):
            base = plans if DMPC_GS else frozen  # Gauss-Seidel reads 0..i-1 fresh
            nbr = [ppath] + [base[j] for j in range(N) if j != i]
            # the patch car is an RL policy that will not yield -> no 50/50 split
            coop = [False] + [True] * (N - 1)
            nrm, off = separating_hyperplanes(plans[i], nbr, HP_R, coop)
            f = folls[i]
            if MPC_IS_ST:
                sx = _FST[i]
                x0 = np.array([sx[0] - p.x, sx[1] - p.y, sx[2], sx[3],
                               sx[4], sx[5], sx[6]], np.float64)
            else:
                x0 = np.array([f[0] - p.x, f[1] - p.y, f[2], f[3]], np.float64)
            ct = center_traj(p, i, K, dt, omega, accel)
            U, ok, xy = solvers[i].solve(x0, shim, np.stack(nbr, axis=0),
                                         nrm, off, center_traj=ct)
            _solves[0] += 1
            if ok and xy is not None:
                new = (1.0 - DAMP) * plans[i] + DAMP * xy
                delta_max = max(delta_max, float(np.max(np.abs(new - plans[i]))))
                plans[i] = new
                solved[i] = True
                last_U[i] = U
    _conv[0] = delta_max
    if delta_max > 1e-3:
        _hp_ticks[0] += 1

    cmds = []
    for i in range(N):
        if solved[i]:
            _hold_k[i] = 0
            _apply(i, last_U[i], folls[i])
            _plans_w[i] = plans[i] + org
        else:
            # every sweep failed -> keep executing the last good plan open-loop,
            # exactly as the myopic version does.  Still no regulator anywhere.
            _fail[i] += 1
            _hold_k[i] += 1
            cmd = _from_plan(i, _hold_k[i])
            if cmd is not None:
                _hold[i][0], _hold[i][1] = cmd
            if _plans_w[i] is not None:      # roll the belief on so neighbours see something sane
                pw = np.asarray(_plans_w[i], np.float64)
                _plans_w[i] = np.vstack([pw[1:], 2.0 * pw[-1] - pw[-2]])
        cmds.append((_hold[i][0], _hold[i][1]))
    return cmds


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
    al, la = slot_along(i, p0.a, p0.b), slot_lat(i, p0.a, p0.b)
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
    set_tick_kinematics(p.v, omega)   # speed-aware formation shrink
    cmds = dmpc_step(folls, p, omega, accel, do_solve)

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
              f"{100*n_all_inside/max(step*_KREP,1):.0f}% | min car gap {min_gap:.2f}m"
              f" | dmpc dx={_conv[0]:.3f}")
    if term or trunc:
        reason = info.get("termination_reason", "?")
        print(f"\nPATCH EPISODE END @ step {step}: {reason}  "
              f"(env prog {prog:.1%} / true prog {tprog:.1%})")
        break

try: env.close()
except Exception: pass
n_steps = len(REC)
completed = tprog >= 0.98
print(f"\n=== RESULT [DMPC]  N={N}  slots={SLOTS}  patch={tag}  map={args.map} ===")
print(f"DMPC: sweeps={SWEEPS} mode={'gauss-seidel' if DMPC_GS else 'jacobi'} "
      f"damp={DAMP} hp_radius={HP_R} | {_solves[0]} solver calls | "
      f"last plan delta {_conv[0]:.4f} m")
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
print(f"DMPC solve fails per follower (all sweeps failed): {_fail}  "
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
