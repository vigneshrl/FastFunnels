#!/usr/bin/env python3
"""Run the ORCA baseline across the randomised narrow-zone layouts.

``sweep_runner.py`` evaluates ORCA on a single map (``open_narrow_obs``, the
submitted layout).  This runs the same ``F1TenthORCARunner`` over the
``eval_maps_matched`` / ``eval_maps_heldout`` splits instead, so its clearance
number sits on the same 100 layouts as the funnel policy and the GCBF+ baseline.

Nothing about ORCA itself changes -- the config, the RVO2 wrapper, the approach
waypoint injection and the ``FullRunMetrics`` collector are all
``baselines/orca.py``'s own.  Only ``map_name`` varies.

Usage
-----
    PYTHONPATH=baselines/Python-RVO2/build/lib.linux-x86_64-cpython-39 \\
    /p/cral/vignesh/envs/fastfunnels/bin/python baselines/run_orca_layouts.py \\
        --split train:eval_maps_matched:50 \\
        --split heldout:eval_maps_heldout:50 \\
        --num-cars 4 --out baselines/orca_layouts.jsonl
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import pathlib
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# Order matters twice over.  The repo root holds a bare `f1tenth_gym/` directory
# that shadows the installed package as a namespace package, so the real
# package's parent has to come first.  And both the root and `baselines/` ship
# an `eval_metrics` -- ORCA wants the latter (FullRunMetrics), so `baselines/`
# must precede the root.  Inserting at 0 in reverse order gives
# [f1tenth_gym, baselines, root].
for _p in (_ROOT, _HERE, os.path.join(_ROOT, "f1tenth_gym")):
    while _p in sys.path:
        sys.path.remove(_p)
    sys.path.insert(0, _p)


def patch_track_lookup(roots: list[str]) -> None:
    """Teach f1tenth_gym to find the eval-map variants by name.

    Same trick ``make_rebuttal_figure.py`` uses: the variants live outside the
    gym's own maps directory, so ``find_track_dir`` is redirected first.
    """
    import f1tenth_gym.envs.track.track as _t
    import f1tenth_gym.envs.track.utils as _u

    original = _u.find_track_dir
    paths = [pathlib.Path(r) for r in roots]

    def find(name):
        for root in paths:
            if (root / name).exists():
                return root / name
        return original(name)

    _t.find_track_dir = find
    _u.find_track_dir = find


def _attach_recorder(runner) -> list:
    """Record every pose the sim produces, without touching orca.py.

    Wraps the env adapter's ``step`` so the run itself is unchanged; the list
    fills as ``F1TenthORCARunner.run()`` drives the episode.
    """
    import numpy as np

    poses: list = []
    original = runner.adapter.step

    def recording_step(actions):
        out = original(actions)
        obs = out[0]
        poses.append(np.stack([obs["poses_x"], obs["poses_y"],
                               obs["poses_theta"]], axis=1))
        return out

    runner.adapter.step = recording_step
    return poses


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def parse_split(spec: str):
    parts = spec.split(":")
    if len(parts) == 2:
        return parts[0], parts[1], None
    if len(parts) == 3:
        return parts[0], parts[1], int(parts[2])
    raise argparse.ArgumentTypeError(f"--split wants label:dir[:count], got {spec!r}")


def collect(root: str, limit: int | None) -> list[str]:
    names = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))
        and os.path.exists(os.path.join(root, d, f"{d}_map.yaml"))
    )
    return names[:limit] if limit is not None else names


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", action="append", required=True, type=parse_split,
                    metavar="LABEL:DIR[:N]")
    ap.add_argument("--num-cars", type=int, default=4)
    ap.add_argument("--target-speed", type=float, default=8.0)
    ap.add_argument("--lateral-gap", type=float, default=0.6)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="baselines/orca_layouts.jsonl")
    ap.add_argument("--trace-dir", default=None,
                    help="write a .npy of per-step poses for each layout")
    ap.add_argument("--quiet", action="store_true", default=True,
                    help="swallow the per-run ORCA metric dump")
    args = ap.parse_args()

    splits = [(label, root, collect(os.path.join(_ROOT, root), limit))
              for label, root, limit in args.split]
    patch_track_lookup([os.path.join(_ROOT, root) for _, root, _ in args.split])

    from orca import F1TenthORCARunner, OrcaConfig

    out_path = os.path.join(_ROOT, args.out) if not os.path.isabs(args.out) else args.out
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if args.trace_dir:
        os.makedirs(args.trace_dir, exist_ok=True)

    results: list[dict] = []
    t0 = time.time()
    with open(out_path, "w") as fh:
        for label, root, names in splits:
            for i, name in enumerate(names):
                cfg = OrcaConfig(
                    num_cars=args.num_cars,
                    map_name=name,
                    render=False,
                    max_steps=args.max_steps,
                    seed=args.seed,
                    target_speed=args.target_speed,
                    lateral_gap=args.lateral_gap,
                    obs_yaml_path=os.path.join(
                        _ROOT, root, name, f"{name}_obs_pos.yaml"),
                )
                try:
                    sink = io.StringIO()
                    ctx = (contextlib.redirect_stdout(sink) if args.quiet
                           else contextlib.nullcontext())
                    with ctx:
                        runner = F1TenthORCARunner(cfg)
                        poses = _attach_recorder(runner) if args.trace_dir else None
                        metrics = runner.run()
                    if poses is not None:
                        import numpy as np

                        np.save(os.path.join(args.trace_dir, f"{name}.npy"),
                                np.asarray(poses))
                    row = {"split": label, "name": name, "failed": False, **metrics}
                except Exception as exc:  # a map ORCA cannot even start on
                    row = {"split": label, "name": name, "failed": True,
                           "error": f"{type(exc).__name__}: {exc}",
                           "success": 0.0, "collision": True}
                results.append(row)
                fh.write(json.dumps(row, default=str) + "\n")
                fh.flush()
                print(f"  [{label} {i + 1}/{len(names)}] {name}: "
                      f"success={row.get('success')} "
                      f"cleared={row.get('n_completed')}/{args.num_cars} "
                      f"speed={row.get('avg_speed_mps')} "
                      f"deform={row.get('deformability')}"
                      + (f"  ERROR {row['error']}" if row["failed"] else ""),
                      flush=True)

    print(f"\n[orca] {len(results)} layouts in {time.time() - t0:.1f}s -> {out_path}")
    for label, _, _ in splits:
        rs = [r for r in results if r["split"] == label]
        k = sum(1 for r in rs if float(r.get("success") or 0.0) >= 1.0)
        lo, hi = wilson(k, len(rs))
        print(f"  {label:>8}: {k:3d}/{len(rs):3d} = {100 * k / len(rs):5.1f}%  "
              f"[{lo:.1f}, {hi:.1f}]  "
              f"errors {sum(1 for r in rs if r['failed'])}")


if __name__ == "__main__":
    main()
