#!/usr/bin/env python3
"""Screen checkpoints on a map, scoring FUNNEL BEHAVIOUR as well as progress.

Ranking on progress alone rewards a degenerate policy that crawls at the speed
floor with the funnel collapsed to its minimum -- a tiny funnel trivially fits
past obstacles. This records how much the funnel actually deforms and how fast
the patch goes, so those can be required rather than hoped for.
"""
import argparse, glob, json, os, sys, traceback
import multiprocessing as mp

ROOT = "/p/cral/vignesh/bigtemp_files/FastFunnels/.claude/worktrees/ring-formation"


def run_one(job):
    ckpt, map_name, pool, max_steps = job
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.environ.setdefault("MPLBACKEND", "Agg")
    os.environ.pop("DISPLAY", None)
    os.environ["REEL_POOL_DIR"] = pool
    sys.path.insert(0, ROOT); sys.path.insert(0, f"{ROOT}/presentation_code")
    os.chdir(ROOT)
    import numpy as np
    import matplotlib; matplotlib.use("Agg")
    rec = {"ckpt": ckpt, "status": "?", "progress": 0.0}
    try:
        from mass_eval import install_variant_map_finder
        install_variant_map_finder(pool)
        from make_legacy_patch_reel import load_legacy_policy
        from envs.ppo_policy import PatchCarEnv, PatchEnvConfig

        act, obs_dim, legacy = load_legacy_policy(ckpt)
        rec["obs_dim"] = int(obs_dim); rec["legacy_actions"] = bool(legacy)
        beams = obs_dim - 7
        env = PatchCarEnv(PatchEnvConfig(
            num_agents=2, render_mode=None, random_spawn=False, obs_mode="lidar",
            num_lidar_beams=int(beams), map_name=map_name,
            a_cmd_min=1.5, a_cmd_max=3.0, b_cmd_min=1.0, b_cmd_max=3.0,
            wall_filter_enabled=False))
        obs, _ = env.reset(seed=0)
        if obs.shape[0] != obs_dim:
            rec["status"] = f"ENV_MISMATCH obs {obs.shape[0]} vs {obs_dim}"
            return rec

        cl = np.loadtxt(glob.glob(f"{pool}/{map_name}/{map_name}_centerline.csv")[0],
                        delimiter=",", comments="#")
        cx, cy = cl[:, 0], cl[:, 1]
        S = np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(cx), np.diff(cy)))])
        L = float(S[-1])

        A, B, V = [], [], []
        best, reason = 0.0, "max_steps"
        for t in range(max_steps):
            obs, r, term, trunc, info = env.step(act(obs, env))
            p = env.active_patches[0]
            A.append(float(p.a)); B.append(float(p.b)); V.append(float(p.v))
            k = int(np.argmin((cx - p.x) ** 2 + (cy - p.y) ** 2))
            best = max(best, S[k] / L)
            if term or trunc:
                reason = info.get("termination_reason", "?")
                break
        A = np.array(A); B = np.array(B); V = np.array(V)
        rec.update(
            progress=round(float(best), 4), status=reason, steps=int(len(A)),
            a_mean=round(float(A.mean()), 3), a_std=round(float(A.std()), 3),
            a_min=round(float(A.min()), 3), a_max=round(float(A.max()), 3),
            b_mean=round(float(B.mean()), 3), b_std=round(float(B.std()), 3),
            b_min=round(float(B.min()), 3), b_max=round(float(B.max()), 3),
            v_mean=round(float(V.mean()), 3), v_max=round(float(V.max()), 3),
            b_at_floor=round(float((B <= 1.005).mean()), 3),
            a_at_floor=round(float((A <= 1.505).mean()), 3),
            b_range=round(float(B.max() - B.min()), 3),
        )
        # "deforming" = funnel width genuinely varies and is not parked at the floor
        rec["deforms"] = bool(rec["b_std"] > 0.15 and rec["b_at_floor"] < 0.9)
        rec["quality"] = round(rec["progress"] * (1.0 if rec["deforms"] else 0.0)
                               * min(rec["v_mean"] / 3.0, 1.0), 4)
    except Exception as e:
        rec["status"] = f"ERROR {type(e).__name__}: {e}"
        rec["trace"] = traceback.format_exc()[-300:]
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--map", default="open_narrow_obs")
    ap.add_argument("--pool", default=f"{ROOT}/maps/scenario_on_obs")
    ap.add_argument("--workers", type=int, default=18)
    ap.add_argument("--max-steps", type=int, default=9000)
    a = ap.parse_args()

    cks = [l.strip() for l in open(a.list) if l.strip()]
    print(f"[screen2] {len(cks)} checkpoints on {a.map}", flush=True)
    ctx = mp.get_context("spawn")
    out = []
    with ctx.Pool(a.workers) as pool:
        for rec in pool.imap_unordered(
                run_one, [(c, a.map, a.pool, a.max_steps) for c in cks]):
            out.append(rec)
            if "b_std" in rec:
                print(f"[screen2] prog {rec['progress']:6.1%} v {rec['v_mean']:4.1f} "
                      f"b {rec['b_mean']:4.2f}+-{rec['b_std']:4.2f} "
                      f"floor {rec['b_at_floor']:4.0%} "
                      f"{'DEFORMS' if rec['deforms'] else 'flat   '} "
                      f"{rec['ckpt'].split('patch_')[-1][:58]}", flush=True)
            else:
                print(f"[screen2] {rec['status'][:40]:40s} "
                      f"{rec['ckpt'].split('patch_')[-1][:58]}", flush=True)
            json.dump(out, open(a.out, "w"), indent=1)
    good = [r for r in out if r.get("deforms")]
    good.sort(key=lambda r: -r.get("quality", 0))
    print(f"\n[screen2] {len(good)} of {len(out)} actually deform the funnel")
    print("[screen2] TOP 20 among deforming policies (quality = progress x speed factor)")
    for r in good[:20]:
        print(f"  q{r['quality']:6.3f} prog {r['progress']:6.1%} v {r['v_mean']:4.1f} m/s "
              f"b {r['b_min']:.2f}-{r['b_max']:.2f} (sd {r['b_std']:.2f}) "
              f"{r['status'][:20]:20s} {r['ckpt'].split('patch_policy_models/')[-1]}")
    json.dump(out, open(a.out, "w"), indent=1)
    print(f"[screen2] wrote {a.out}")


if __name__ == "__main__":
    main()
