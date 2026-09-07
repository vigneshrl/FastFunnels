#!/usr/bin/env python3
"""
Same map, same N, same patch -- sweep the FORMATION and stitch one video with a
title card per formation. The transpose of make_nmpc_n_sweep_reel.py (which
sweeps N at a fixed formation).

    python presentation_code/make_formation_compare_reel.py \
        --policy <ckpt> --map open_narrow_obs --map-dir maps/scenario_on_obs \
        --n 4 --slots wedge,ring,plus,corners --out videos/foo.mp4
"""
from __future__ import annotations
import argparse, json, os, re, subprocess, sys, tempfile
import imageio.v2 as imageio

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "presentation_code"))
from make_scene_reel import title_card, FPS

RESULT_RE = re.compile(
    r"steps (\d+) \| patch end reason '([\w_]+)' \| true progress ([\d.]+)%")
INSIDE_RE = re.compile(r"all-inside (\d+)/(\d+) \((\d+)%\)")
PERF_RE = re.compile(r"per-follower inside \[([^\]]*)\]")
OVER_RE = re.compile(r"CAR-CAR OVERLAP on (\d+)/(\d+) steps")
FAIL_RE = re.compile(r"solve fails per follower[^:]*: \[([^\]]*)\]")

LABELS = {
    "wedge":   "WEDGE  (2 abreast + 2 astern)",
    "ring":    "RING  (funnel-scaled: ahead / beams / astern)",
    "plus":    "PLUS  (fixed: 1 ahead, 1 astern, 1 each beam)",
    "corners": "CORNERS  (fixed: one on each quarter)",
    "trail":   "TRAIL  (single file astern)",
    "abreast": "ABREAST",
}


def run_one(job):
    policy, map_name, map_dir, n, slots, steps, mpc_every, seed, clip, every = job
    cmd = [sys.executable, "mpc_follower_native_n.py",
           "--policy", policy, "--map", map_name, "--map-dir", map_dir,
           "--n", str(n), "--seed", str(seed), "--steps", str(steps),
           "--mpc-every", str(mpc_every), "--every", str(every),
           "--render", "mpl", "--out", clip]
    env = dict(os.environ, MPC_SLOTS=slots, QT_QPA_PLATFORM="offscreen",
               MPLBACKEND="Agg", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1")
    env.pop("DISPLAY", None)
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=_ROOT)
    log = p.stdout + p.stderr
    m, ins, per = RESULT_RE.search(log), INSIDE_RE.search(log), PERF_RE.search(log)
    ov, fl = OVER_RE.search(log), FAIL_RE.search(log)
    rec = {"slots": slots, "n": n, "map": map_name}
    if m:
        reason, prog = m.group(2), float(m.group(3))
        status = f"{'ARRIVED' if reason == 'arrived' else reason.upper()}  {prog:.0f}%"
        rec.update(reason=reason, progress=prog, steps=int(m.group(1)))
    else:
        status = "RUN FAILED"
        rec.update(reason="run_failed", progress=0.0)
    if ins:
        status += f"   |  all-inside {ins.group(3)}%"
        rec["all_inside_pct"] = int(ins.group(3))
    if per:
        rec["per_follower"] = per.group(1)
    if ov:
        status += f"   |  OVERLAP {ov.group(1)}"
        rec["overlap_steps"] = int(ov.group(1))
    else:
        rec["overlap_steps"] = 0
    if fl:
        rec["solve_fails"] = fl.group(1)
    rec["status"] = status
    return slots, status, clip, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--map", required=True)
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--slots", default="wedge,ring,plus,corners")
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--mpc-every", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()

    forms = a.slots.split(",")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="form_reel_")
    jobs = [(a.policy, a.map, a.map_dir, a.n, f, a.steps, a.mpc_every, a.seed,
             os.path.join(tmp, f"{f}.mp4"), a.every) for f in forms]

    from concurrent.futures import ProcessPoolExecutor
    res, recs = {}, []
    with ProcessPoolExecutor(max_workers=min(a.workers, len(jobs))) as ex:
        for slots, status, clip, rec in ex.map(run_one, jobs):
            res[slots] = (status, clip)
            recs.append(rec)
            print(f"[form] {slots:9s} -> {status}", flush=True)

    w = imageio.get_writer(a.out, fps=FPS, codec="libx264", quality=8,
                           macro_block_size=1)
    for f in forms:
        status, clip = res.get(f, ("MISSING", None))
        size, rd = (800, 700), None
        if clip and os.path.exists(clip):
            rd = imageio.get_reader(clip)
            size = rd.get_meta_data().get("size", size)
        for fr in title_card(f"N = {a.n}   —   {LABELS.get(f, f.upper())}",
                             status, size=size):
            w.append_data(fr)
        if rd is not None:
            for fr in rd:
                w.append_data(fr)
            rd.close()
    w.close()
    sj = os.path.splitext(a.out)[0] + "_summary.json"
    json.dump(recs, open(sj, "w"), indent=1)
    print(f"\n[form] wrote {a.out}\n[form] wrote {sj}")


if __name__ == "__main__":
    main()
