#!/usr/bin/env python3
"""Screen patch checkpoints against open_narrow_obs.

For each checkpoint: read the policy's observation size, build a PatchCarEnv
whose obs matches it, run one episode, report furthest true progress and why it
stopped. Auto-detects the June-style raw action space via the same shim the
reel uses.
"""
import argparse, glob, json, os, sys, traceback
import multiprocessing as mp

ROOT = "/p/cral/vignesh/bigtemp_files/FastFunnels/.claude/worktrees/ring-formation"
POOL = os.path.join(ROOT, "maps", "scenario_on_obs")
MAP = "open_narrow_obs"


def run_one(job):
    ckpt, max_steps = job
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.pop("DISPLAY", None)
    os.environ["REEL_POOL_DIR"] = POOL
    sys.path.insert(0, ROOT)
    sys.path.insert(0, os.path.join(ROOT, "presentation_code"))
    os.chdir(ROOT)
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    rec = {"ckpt": ckpt, "status": "?", "progress": 0.0, "steps": 0, "obs_dim": None}
    try:
        from mass_eval import install_variant_map_finder
        install_variant_map_finder(POOL)
        from make_legacy_patch_reel import load_legacy_policy
        from envs.ppo_policy import PatchCarEnv, PatchEnvConfig

        act, obs_dim, is_legacy = load_legacy_policy(ckpt)
        rec["obs_dim"] = int(obs_dim)
        rec["legacy_actions"] = bool(is_legacy)

        # obs = 7 scalars + beams (+ 2*len(pass_lookahead_m) when gap lookahead on)
        cand = []
        base = PatchEnvConfig()
        nlook = 2 * len(getattr(base, "pass_lookahead_m", []) or [])
        for beams in (obs_dim - 7, obs_dim - 7 - nlook):
            if beams in (108, 360, 216, 54, 1080):
                cand.append((beams, beams != obs_dim - 7))
        if not cand:
            cand = [(obs_dim - 7, False)]

        last_err = None
        for beams, gap in cand:
            try:
                kw = dict(num_agents=2, render_mode=None, random_spawn=False,
                          obs_mode="lidar", num_lidar_beams=int(beams),
                          map_name=MAP, a_cmd_min=1.5, a_cmd_max=3.0,
                          b_cmd_min=1.0, b_cmd_max=3.0, wall_filter_enabled=False)
                if gap:
                    kw["obs_gap_lookahead"] = True
                env = PatchCarEnv(PatchEnvConfig(**kw))
                obs, _ = env.reset(seed=0)
                if obs.shape[0] != obs_dim:
                    last_err = f"obs {obs.shape[0]} != policy {obs_dim}"
                    continue
                rec["beams"] = int(beams)
                rec["gap_lookahead"] = bool(gap)
                break
            except Exception as e:
                last_err = f"{type(e).__name__}: {e}"
                env = None
        else:
            rec["status"] = f"ENV_MISMATCH ({last_err})"
            return rec

        cl = np.loadtxt(glob.glob(f"{POOL}/{MAP}/{MAP}_centerline.csv")[0],
                        delimiter=",", comments="#")
        cx, cy = cl[:, 0], cl[:, 1]
        S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
        L = float(S[-1])

        best = 0.0
        reason = "max_steps"
        for t in range(max_steps):
            obs, r, term, trunc, info = env.step(act(obs, env))
            p = env.active_patches[0]
            k = int(np.argmin((cx - p.x) ** 2 + (cy - p.y) ** 2))
            best = max(best, S[k] / L)
            rec["steps"] = t + 1
            if term or trunc:
                reason = info.get("termination_reason", "?")
                break
        rec["progress"] = round(float(best), 4)
        rec["status"] = reason
    except Exception as e:
        rec["status"] = f"ERROR {type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-400:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True, help="file of checkpoint paths, one per line")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--max-steps", type=int, default=9000)
    a = ap.parse_args()

    ckpts = [l.strip() for l in open(a.list) if l.strip()]
    print(f"[screen] {len(ckpts)} checkpoints, {a.workers} workers", flush=True)
    ctx = mp.get_context("spawn")
    out = []
    with ctx.Pool(a.workers) as pool:
        for rec in pool.imap_unordered(run_one, [(c, a.max_steps) for c in ckpts]):
            out.append(rec)
            tag = os.path.relpath(rec["ckpt"], os.path.dirname(os.path.dirname(rec["ckpt"])))
            print(f"[screen] {rec['progress']:6.1%} {rec['status'][:34]:34s} {tag}", flush=True)
            json.dump(sorted(out, key=lambda r: -r["progress"]), open(a.out, "w"), indent=1)
    out.sort(key=lambda r: -r["progress"])
    print("\n[screen] TOP 15")
    for r in out[:15]:
        print(f"  {r['progress']:6.1%}  {r['status'][:26]:26s}  {r['ckpt']}")
    print(f"[screen] wrote {a.out}")


if __name__ == "__main__":
    main()
