#!/usr/bin/env python3
"""
Run the patch + N NMPC followers across a LIST OF MAPS (fixed N) in parallel and
stitch every clip into one video with a title card per map.

The sibling script `make_nmpc_n_sweep_reel.py` sweeps N over two fixed maps;
this one is the transpose -- fixed N, sweeping the map -- which is what you want
when the question is "how does this policy cope with these scenarios" rather
than "how many followers can it hold".

    python presentation_code/make_map_reel_followers.py \
        --policy patch_policy_models/run_20260602_233933/checkpoint_15000000.zip \
        --map-dir maps/scenario_corners \
        --maps sc_zigzag000,sc_lshape000 --n 3 --out videos/foo.mp4
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
OVERLAP_RE = re.compile(r"CAR-CAR OVERLAP on (\d+)/(\d+) steps")


def run_one(job):
    (policy, map_name, map_dir, n, steps, mpc_every, slots, seed, clip, every) = job
    cmd = [sys.executable, "mpc_follower_native_n.py",
           "--policy", policy, "--map", map_name, "--map-dir", map_dir,
           "--n", str(n), "--seed", str(seed), "--steps", str(steps),
           "--mpc-every", str(mpc_every), "--every", str(every),
           "--render", "mpl", "--out", clip]
    env = dict(os.environ, MPC_SLOTS=slots, QT_QPA_PLATFORM="offscreen",
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=_ROOT)
    log = p.stdout + p.stderr
    m, ins, ov = RESULT_RE.search(log), INSIDE_RE.search(log), OVERLAP_RE.search(log)
    if m:
        reason, prog = m.group(2), float(m.group(3))
        status = f"{'ARRIVED' if reason == 'arrived' else reason.upper()}  {prog:.0f}% progress"
        rec = dict(map=map_name, reason=reason, progress=prog, steps=int(m.group(1)))
    else:
        status, rec = "RUN FAILED", dict(map=map_name, reason="run_failed", progress=0.0)
    if ins:
        status += f"   |  all-inside {ins.group(3)}%"
        rec["all_inside_pct"] = int(ins.group(3))
    if ov:
        status += f"   |  OVERLAP {ov.group(1)} steps"
        rec["overlap_steps"] = int(ov.group(1))
    else:
        rec["overlap_steps"] = 0
    rec["status"] = status
    return map_name, status, clip, rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--maps", required=True, help="comma-separated map names")
    ap.add_argument("--map-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--steps", type=int, default=9000)
    ap.add_argument("--mpc-every", type=int, default=2)
    ap.add_argument("--slots", default="ring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=60)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()

    maps = args.maps.split(",")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="map_reel_")
    jobs = [(args.policy, m, args.map_dir, args.n, args.steps, args.mpc_every,
             args.slots, args.seed, os.path.join(tmp, f"{m}.mp4"), args.every)
            for m in maps]

    from concurrent.futures import ProcessPoolExecutor
    results, recs = {}, []
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for map_name, status, clip, rec in ex.map(run_one, jobs):
            results[map_name] = (status, clip)
            recs.append(rec)
            print(f"[reel] {map_name:18s} -> {status}", flush=True)

    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264",
                                quality=8, macro_block_size=1)
    for m in maps:
        status, clip = results.get(m, ("MISSING", None))
        size, rd = (800, 700), None
        if clip and os.path.exists(clip):
            rd = imageio.get_reader(clip)
            size = rd.get_meta_data().get("size", size)
        for f in title_card(f"{m}   —   N = {args.n}   —   slots = {args.slots}",
                            status, size=size):
            writer.append_data(f)
        if rd is not None:
            for f in rd:
                writer.append_data(f)
            rd.close()
    writer.close()

    sj = os.path.splitext(args.out)[0] + "_summary.json"
    json.dump(recs, open(sj, "w"), indent=1)
    print(f"\n[reel] wrote {args.out}\n[reel] wrote {sj}")


if __name__ == "__main__":
    main()
