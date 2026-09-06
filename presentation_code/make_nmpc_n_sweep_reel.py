#!/usr/bin/env python3
"""
Run the NMPC follower sweep N=1..6 against a frozen patch checkpoint on the
on_rep_* maps, in parallel, then stitch every clip into ONE video with a title
card (map + N + outcome) before each scene.

    python presentation_code/make_nmpc_n_sweep_reel.py \
        --policy patch_policy_models/run_20260602_233933/checkpoint_15000000.zip \
        --out videos/legacy_nmpc_N1-6.mp4
"""
from __future__ import annotations
import argparse, os, re, subprocess, sys, tempfile
import imageio.v2 as imageio

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "presentation_code"))
from make_scene_reel import title_card, FPS

MAPS = [
    ("on_rep_clean", "OPEN_NARROW (clean)"),
    ("on_rep_obs",   "OPEN_NARROW + obstacles"),
]
MAP_DIR = "maps/scenario_on_rep"
RESULT_RE = re.compile(
    r"steps (\d+) \| patch end reason '([\w_]+)' \| true progress ([\d.]+)%")
INSIDE_RE = re.compile(r"all-inside (\d+)/(\d+) \((\d+)%\)")


def run_one(job):
    (policy, map_name, map_dir, n, steps, mpc_every, slots, seed, clip, every) = job
    cmd = [sys.executable, "mpc_follower_native_n.py",
           "--policy", policy, "--map", map_name,
           "--map-dir", map_dir,
           "--n", str(n), "--seed", str(seed), "--steps", str(steps),
           "--mpc-every", str(mpc_every), "--every", str(every),
           "--render", "mpl", "--out", clip]
    env = dict(os.environ, MPC_SLOTS=slots,
               OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    p = subprocess.run(cmd, env=env, capture_output=True, text=True, cwd=_ROOT)
    log = p.stdout + p.stderr
    m = RESULT_RE.search(log)
    ins = INSIDE_RE.search(log)
    if m:
        reason, prog = m.group(2), float(m.group(3))
        status = f"{'ARRIVED' if reason=='arrived' else reason.upper()}  {prog:.0f}% progress"
    else:
        status = "RUN FAILED"
    if ins:
        status += f"   |  all-inside {ins.group(3)}%"
    return map_name, n, status, clip, log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", default="videos/legacy_nmpc_ring_N1-6.mp4")
    ap.add_argument("--ns", default="1,2,3,4,5,6")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--mpc-every", type=int, default=2)
    ap.add_argument("--slots", default="ring")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--every", type=int, default=10, help="render every Nth sim step")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--maps", default="", help="comma-separated map names (default: the on_rep pair)")
    ap.add_argument("--map-dir", default=MAP_DIR)
    ap.add_argument("--labels", default="", help="comma-separated title-card labels, one per map")
    args = ap.parse_args()

    ns = [int(x) for x in args.ns.split(",")]
    global MAPS
    if args.maps:
        _names = args.maps.split(",")
        _labs = args.labels.split(",") if args.labels else _names
        MAPS = list(zip(_names, _labs))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="nmpc_reel_")

    jobs = []
    for map_name, _ in MAPS:
        for n in ns:
            clip = os.path.join(tmp, f"{map_name}_N{n}.mp4")
            jobs.append((args.policy, map_name, args.map_dir, n, args.steps,
                         args.mpc_every, args.slots, args.seed, clip, args.every))

    from concurrent.futures import ProcessPoolExecutor
    results = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, len(jobs))) as ex:
        for map_name, n, status, clip, log in ex.map(run_one, jobs):
            results[(map_name, n)] = (status, clip)
            print(f"[reel] {map_name} N={n} -> {status}", flush=True)

    writer = imageio.get_writer(args.out, fps=FPS, codec="libx264",
                                quality=8, macro_block_size=1)
    summary = []
    for map_name, label in MAPS:
        for n in ns:
            status, clip = results.get((map_name, n), ("MISSING", None))
            summary.append((map_name, n, status))
            size = (800, 700)
            rd = None
            if clip and os.path.exists(clip):
                rd = imageio.get_reader(clip)
                size = rd.get_meta_data().get("size", size)
            for f in title_card(f"{label}   —   N = {n}   —   slots = {args.slots}", status, size=size):
                writer.append_data(f)
            if rd is not None:
                for f in rd:
                    writer.append_data(f)
                rd.close()
    writer.close()
    import json
    sj = os.path.splitext(args.out)[0] + "_summary.json"
    with open(sj, "w") as fh:
        json.dump([{"map": m, "n": n, "slots": args.slots, "status": st}
                   for m, n, st in summary], fh, indent=1)
    print(f"\n[reel] wrote {args.out}\n[reel] wrote {sj}\n[reel] summary:")
    for m, n, st in summary:
        print(f"   {m:14s} N={n}  {st}")


if __name__ == "__main__":
    main()
