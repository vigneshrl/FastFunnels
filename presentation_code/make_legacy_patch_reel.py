#!/usr/bin/env python3
"""
Record ONE video of the LEGACY patch policy (raw-action checkpoints such as
patch_policy_models/run_20260602_233933/checkpoint_15000000) running
patch-only, scene by scene, across the obstacle maps in maps/scenario_pool.

The June-2026 checkpoints emit RAW physical commands
    [steer_angle (rad), speed (m/s) in [2,10], a in [1.5,3], b in [1,3]]
while today's PatchCarEnv expects the CENTRED layout
    [steer_RATE in [-1,1], v_c, a_c, b_c in [-1,1]]
so this script wraps the policy in a shim that (a) renormalises v/a/b into the
centred convention and (b) converts the absolute steer angle into the rate that
drives the env's integrated steer state toward it (slew-limited at
steer_rate_max = 3.0 rad/s, i.e. the actuator limit, so the physical command is
essentially the same as what the checkpoint was trained with).

    python presentation_code/make_legacy_patch_reel.py \
        patch_policy_models/run_20260602_233933/checkpoint_15000000 \
        --per-type 5 --out videos/legacy_patch_scenario_pool_reel.mp4

Scenes render in parallel (one process per scene), then get stitched with a
title card (map name + outcome) before each. Successes and failures are both
included -- a status reel, not a highlight reel.
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, pickle, argparse, multiprocessing as mp
import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "presentation_code"))

_TURNBACK_DROP = float(os.environ.get("REEL_TURNBACK_DROP", "0.03"))
POOL_DIR = os.environ.get("REEL_POOL_DIR", os.path.join(_ROOT, "maps", "scenario_pool"))
FPS = 25
TITLE_CARD_SECONDS = 1.6
FRAME_SIZE = (900, 700)
LABELS = {
    "open_narrow": "OPEN_NARROW (tunnel) + obstacles",
    "mf_narrowing": "NARROWING + pinches + obstacles",
    "mf_zigzag": "ZIG-ZAG corridor + obstacles",
    "mf_lshape": "L-SHAPED corridor + obstacles",
    "mf_slalom": "SLALOM (bar gates) + obstacles",
}


def load_legacy_policy(path: str):
    """Return act(obs, env) -> centred action for a raw-action checkpoint."""
    from stable_baselines3 import PPO
    stem = path[:-4] if path.endswith(".zip") else path
    d, base = os.path.dirname(stem), os.path.basename(stem)
    # SB3 runs here save `best_model.zip` + `best_vecnormalize.pkl` and
    # `final_model.zip` + `final_vecnormalize.pkl` -- note the "_model" is
    # dropped in the pkl name. Getting this wrong silently feeds the policy
    # observations normalised by another checkpoint's statistics, which looks
    # like a catastrophically bad policy rather than an error.
    _cands = [f"{stem}_vecnormalize.pkl", f"{stem}_vecnorm.pkl"]
    if base.endswith("_model"):
        _short = base[: -len("_model")]
        _cands += [os.path.join(d, f"{_short}_vecnormalize.pkl"),
                   os.path.join(d, f"{_short}_model_vecnorm.pkl")]
    _cands += [os.path.join(d, "best_vecnormalize.pkl")]
    for _c in _cands:
        if os.path.exists(_c):
            vn_path = _c
            break
    else:
        raise FileNotFoundError(f"no VecNormalize stats for {path}; tried {_cands}")
    print(f"[policy] {path}  vecnorm={os.path.basename(vn_path)}", flush=True)
    model = PPO.load(stem, device="cpu")
    model.policy.set_training_mode(False)
    vn = pickle.load(open(vn_path, "rb"))
    mean = vn.obs_rms.mean.astype(np.float32)
    var = vn.obs_rms.var.astype(np.float32)
    clip = float(vn.clip_obs)
    lo, hi = model.action_space.low, model.action_space.high
    raw_is_legacy = bool(hi[1] > 1.5)   # centred checkpoints have high[1] == 1
    obs_dim = int(np.prod(model.observation_space.shape))

    def act(obs, env):
        o = np.clip((np.asarray(obs, np.float32) - mean) / np.sqrt(var + 1e-8), -clip, clip)
        a, _ = model.policy.predict(o[None], deterministic=True)
        r = np.asarray(a, np.float32).flatten()
        if not raw_is_legacy:
            return r
        r = np.clip(r, lo, hi)
        cfg = env.cfg
        # steer: absolute angle -> rate that drives the integrator toward it
        cur = float(getattr(env, "_steer_state", 0.0))
        rate = np.clip((float(r[0]) - cur) / cfg.steer_rate_max, -1.0, 1.0)
        def c(x, lo_, hi_):
            return (x - 0.5 * (lo_ + hi_)) / (0.5 * (hi_ - lo_))
        return np.array([rate,
                         c(float(r[1]), 2.0, 10.0),
                         c(float(r[2]), cfg.a_cmd_min, cfg.a_cmd_max),
                         c(float(r[3]), cfg.b_cmd_min, cfg.b_cmd_max)], np.float32)

    return act, obs_dim, raw_is_legacy


def _centerline_projector(map_name: str):
    """s(x, y) by nearest waypoint of the map's centreline CSV. The gym's
    periodic spline is unreliable on open tracks (it force-closes the path),
    so the env's lap_progress can jump; this is the ground truth for reporting."""
    import glob
    cands = glob.glob(os.path.join(POOL_DIR, map_name, f"{map_name}_centerline.csv")) + \
            glob.glob(os.path.join(_ROOT, "f1tenth_gym", "maps", map_name, f"{map_name}_centerline.csv"))
    if not cands:
        return None
    cl = np.loadtxt(cands[0], delimiter=",", comments="#")
    cx, cy = cl[:, 0], cl[:, 1]
    S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
    L = float(S[-1])
    def proj(x, y):
        return float(S[int(np.argmin((cx - x) ** 2 + (cy - y) ** 2))]) / L
    return proj


def make_env(map_name: str, render: bool):
    """Build the eval env. Defaults reproduce the June training env (108 beams,
    a in [1.5,3], b in [1,3]); REEL_ENV_CONFIG points at a yaml of
    PatchEnvConfig overrides so a run trained with different settings (e.g. 360
    lidar beams) is evaluated under the settings it was trained with."""
    from envs.ppo_policy import PatchCarEnv, PatchEnvConfig
    kw = dict(num_agents=2, render_mode="human" if render else None,
              random_spawn=False, obs_mode="lidar", num_lidar_beams=108,
              map_name=map_name,
              a_cmd_min=1.5, a_cmd_max=3.0, b_cmd_min=1.0, b_cmd_max=3.0,
              wall_filter_enabled=False)
    _ec = os.environ.get("REEL_ENV_CONFIG", "")
    if _ec:
        import yaml
        kw.update(yaml.safe_load(open(_ec)) or {})
        kw["map_name"] = map_name
        kw["render_mode"] = "human" if render else None
    return PatchCarEnv(PatchEnvConfig(**kw))


def grab_frame(env):
    env.render()
    fig = env._fig
    if fig is None:
        return None
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
    return np.ascontiguousarray(buf[..., :3])


def run_scene(args):
    """Worker: run one map, save strided frames to an .npz, return outcome."""
    policy, map_name, seed, max_steps, target_frames, out_npz, render = args
    import matplotlib
    matplotlib.use("Agg")
    from mass_eval import install_variant_map_finder
    install_variant_map_finder(POOL_DIR)
    act, obs_dim, _ = load_legacy_policy(policy)
    env = make_env(map_name, render)
    obs, _ = env.reset(seed=seed)
    assert obs.shape[0] == obs_dim, f"obs mismatch env {obs.shape[0]} vs policy {obs_dim}"
    # Pass 1: run headless to learn the episode length (cheap), pass 2 renders
    # only ~target_frames evenly spaced steps so long and short episodes get the
    # same playback duration.
    n_steps, best_prog, reason = 0, 0.0, "running"
    proj = _centerline_projector(map_name)
    for t in range(max_steps):
        obs, r, term, trunc, info = env.step(act(obs, env))
        n_steps = t + 1
        if proj is not None:
            p0 = env.active_patches[0]
            prog = proj(float(p0.x), float(p0.y))
        else:
            prog = float(info.get("lap_progress", 0.0))
        best_prog = max(best_prog, prog)
        if term or trunc:
            reason = info.get("termination_reason", "?")
            break
        # The June checkpoint reaches ~97% of open tracks and then turns round
        # and drives back at floor speed. Cut the scene there and say so.
        # NOTE: 0.03 is far too tight for long multi-tunnel courses, where the
        # patch legitimately gives back more than 3% while manoeuvring -- on
        # bw_on_obs_ext000 this fired at step 823 and reported "turned_back 11%"
        # for a policy the env itself runs to "arrived" at step 6541. Raise
        # TURNBACK_DROP (or set it >= 1.0) to disable the cut.
        if _TURNBACK_DROP < 1.0 and best_prog - prog > _TURNBACK_DROP:
            reason = "turned_back"
            break
    else:
        reason = "stalled" if best_prog < 0.95 else "timeout"
    env.close()
    frames = []
    if render:
        stride = max(1, n_steps // target_frames)
        env = make_env(map_name, True)
        obs, _ = env.reset(seed=seed)
        for t in range(n_steps):
            if t % stride == 0:
                f = grab_frame(env)
                if f is not None:
                    frames.append(f)
            obs, r, term, trunc, info = env.step(act(obs, env))
            if term or trunc:
                break
        f = grab_frame(env)
        if f is not None:
            frames.append(f)
        env.close()
        np.savez_compressed(out_npz, frames=np.stack(frames) if frames else np.zeros((0,)+FRAME_SIZE[::-1]+(3,), np.uint8))
    return {"map": map_name, "steps": n_steps, "progress": best_prog,
            "reason": reason, "n_frames": len(frames), "npz": out_npz}


def title_card(text, subtitle, size=FRAME_SIZE, n_frames=int(TITLE_CARD_SECONDS * FPS)):
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", size, (18, 18, 22))
    draw = ImageDraw.Draw(img)
    try:
        f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 34)
        f_sub = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 22)
    except Exception:
        f_title = ImageFont.load_default(); f_sub = f_title
    w, h = size
    # text may be "line1\nline2": map name on top, scenario family below
    lines = text.split("\n")
    y = h / 2 - 40 - 24 * (len(lines) - 1)
    for ln in lines:
        tb = draw.textbbox((0, 0), ln, font=f_title)
        draw.text(((w - (tb[2] - tb[0])) / 2, y), ln, font=f_title, fill=(240, 240, 245))
        y += 46
    sb = draw.textbbox((0, 0), subtitle, font=f_sub)
    color = (110, 220, 140) if "ARRIVED" in subtitle else (235, 130, 90)
    draw.text(((w - (sb[2] - sb[0])) / 2, y + 14), subtitle, font=f_sub, fill=color)
    return [np.array(img)] * n_frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--out", default="videos/legacy_patch_scenario_pool_reel.mp4")
    ap.add_argument("--per-type", type=int, default=5, help="variants per base map (0 = all 80)")
    ap.add_argument("--maps", default="", help="explicit comma-separated map list (overrides --per-type)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pool-dir", default=POOL_DIR, help="map pool dir (also REEL_POOL_DIR env for the workers)")
    ap.add_argument("--env-config", default="", help="yaml of PatchEnvConfig overrides (evaluate a run under its own training settings)")
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--frames-per-scene", type=int, default=175)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-video", action="store_true", help="evaluate only, print the table")
    ap.add_argument("--stitch-only", action="store_true", help="re-stitch from saved frames + summary json")
    ap.add_argument("--tmp", default=os.environ.get("REEL_TMP", os.path.join(_ROOT, "videos", "_reel_tmp")))
    args = ap.parse_args()

    os.environ["REEL_POOL_DIR"] = os.path.abspath(args.pool_dir)
    if args.env_config:
        os.environ["REEL_ENV_CONFIG"] = os.path.abspath(args.env_config)
    _ip = os.path.join(args.pool_dir, "variants_index.json")
    idx = json.load(open(_ip)) if os.path.exists(_ip) else {"base_maps": [], "variants": []}
    if args.maps:
        maps = args.maps.split(",")
    else:
        maps = []
        for base in idx["base_maps"]:
            vs = [v["name"] for v in idx["variants"] if v["base_map"] == base]
            maps += vs if args.per_type <= 0 else vs[:args.per_type]
    base_of = {v["name"]: v["base_map"] for v in idx["variants"]}

    os.makedirs(args.tmp, exist_ok=True)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    jobs = [(args.policy, m, args.seed, args.max_steps, args.frames_per_scene,
             os.path.join(args.tmp, f"{m}.npz"), not args.no_video) for m in maps]
    summary_path = os.path.splitext(args.out)[0] + "_summary.json"
    if args.stitch_only:
        results = json.load(open(summary_path))
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(min(args.workers, len(jobs))) as pool:
            results = []
            for r in pool.imap(run_scene, jobs):
                results.append(r)
                print(f"[reel] {r['map']:16s} {r['reason']:24s} {r['progress']:6.1%} "
                      f"steps={r['steps']:5d} frames={r['n_frames']}", flush=True)

    json.dump(results, open(summary_path, "w"), indent=1)
    if not args.no_video:
        import imageio.v2 as imageio
        writer = imageio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=1)
        for r in results:
            status = "ARRIVED" if r["reason"] == "arrived" else f"{r['reason'].upper()}  ({r['progress']:.0%} progress)"
            label = f"{r['map']}\n{LABELS.get(base_of.get(r['map'], ''), '')}"
            for f in title_card(label, status):
                writer.append_data(f)
            for f in np.load(r["npz"])["frames"]:
                writer.append_data(f)
        writer.close()
        print(f"\n[reel] wrote {args.out}")

    print("\n[reel] summary by base map:")
    for base in idx["base_maps"]:
        rs = [r for r in results if base_of.get(r["map"]) == base]
        if not rs:
            continue
        arr = sum(r["reason"] == "arrived" for r in rs)
        med = float(np.median([r["progress"] for r in rs]))
        print(f"   {base:14s} arrived {arr}/{len(rs)}   median progress {med:.0%}")


if __name__ == "__main__":
    main()
