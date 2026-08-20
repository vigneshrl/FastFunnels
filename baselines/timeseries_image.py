#!/usr/bin/env python3
"""
Stroboscopic timeseries composite — fixed MAP camera (focus_on: null).

Runs an actual ORCA or LF trajectory, samples agent positions at specified
sim-times, and composites semi-transparent ghosts onto a clean static map
background (no UI text, no timer, no residual agent pixels).

Usage (from baselines/):
    python timeseries_image.py -o vids/ts_lf_2agent.png
    python timeseries_image.py -o vids/ts_orca_2agent.png --algorithm orca
    python timeseries_image.py -o vids/ts.png --sample-times 0 10 20 30 --alpha 0.65
"""

from __future__ import annotations

import argparse
import math
import os
import pathlib
import sys
import tempfile

import cv2
import numpy as np

os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

_HERE = pathlib.Path(__file__).parent.resolve()
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "FastFunnels"))


# ── Helpers ────────────────────────────────────────────────────────────────────

def _nearest_wp_idx(xs: np.ndarray, ys: np.ndarray, x: float, y: float) -> int:
    return int(np.argmin(np.hypot(xs - x, ys - y)))


def _rect_corners(cx: float, cy: float, w: float, h: float, angle_deg: float) -> np.ndarray:
    """Return (4, 2) array of rotated-rectangle corners in world coords."""
    t = math.radians(angle_deg)
    ct, st = math.cos(t), math.sin(t)
    hw, hh = w / 2, h / 2
    return np.array([
        [cx + ct * hw - st * hh, cy + st * hw + ct * hh],
        [cx - ct * hw - st * hh, cy - st * hw + ct * hh],
        [cx - ct * hw + st * hh, cy - st * hw - ct * hh],
        [cx + ct * hw + st * hh, cy + st * hw - ct * hh],
    ])


def _draw_obstacles_on_renderer(adapter, map_name: str = "open_narrow_obs") -> None:
    """Draw obstacle rectangles as persistent plot items on the PyQt canvas."""
    import yaml

    try:
        from f1tenth_gym.envs.track.utils import find_track_dir
        data = yaml.safe_load(
            (find_track_dir(map_name) / f"{map_name}_obs_pos.yaml").read_text()
        )
        obstacles = data.get("obstacles", [])
    except Exception as exc:
        print(f"  [!] Could not load obstacles: {exc}")
        return

    renderer = adapter.base_env.unwrapped.renderer
    if renderer is None:
        return

    for obs in obstacles:
        if obs.get("type") != "rect":
            continue
        corners = _rect_corners(
            float(obs["center_m"][0]), float(obs["center_m"][1]),
            float(obs["width_m"]), float(obs["height_m"]),
            float(obs.get("angle_deg", 0)),
        )
        renderer.render_closed_lines(corners, color=(255, 140, 0), size=2)  # orange outline

    print(f"  [i] Drew {len(obstacles)} obstacle(s) on renderer")


# Path to the installed f1tenth_gym package rendering.yaml
def _pkg_yaml_path() -> pathlib.Path:
    import f1tenth_gym.envs.rendering as _rmod
    return pathlib.Path(_rmod.__file__).parent / "rendering.yaml"


def _patch_pkg_yaml() -> str:
    """Overwrite rendering.yaml with fixed-camera settings; return original text."""
    p = _pkg_yaml_path()
    original = p.read_text()
    p.write_text(
        "window_size: 800\nfocus_on: null\nzoom_in_factor: 1.5\n"
        "show_wheels: True\ncar_tickness: 1\nshow_info: False\n"
        'render_type: "pyqt6"\n'
        "vehicle_palette:\n"
        "  - '#984ea3'\n  - '#e41a1c'\n  - '#ff7f00'\n  - '#a65628'\n"
    )
    return original


def _restore_pkg_yaml(original: str) -> None:
    _pkg_yaml_path().write_text(original)


def _lock_camera_roi(adapter, cx: float, cy: float, pad_x: float, pad_y: float) -> None:
    """Fix the PyQt plot camera to a world-coordinate ROI and disable autoRange."""
    env = adapter.base_env.unwrapped
    renderer = env.renderer
    if renderer is None:
        return
    renderer.canvas.setXRange(cx - pad_x, cx + pad_x, padding=0)
    renderer.canvas.setYRange(cy - pad_y, cy + pad_y, padding=0)
    renderer.canvas.autoRange = lambda: None  # prevent future autoRange from overriding
    renderer.app.processEvents()


def _hide_renderer_ui(renderer) -> list:
    """Hide cars + all UI text items; return list of items to restore."""
    hidden = []
    for car in (renderer.cars or []):
        for item in [car.chassis,
                     getattr(car, "fl_wheel", None),
                     getattr(car, "fr_wheel", None)]:
            if item is not None:
                item.setVisible(False)
                hidden.append(item)
    for txt in [renderer.time_renderer,
                renderer.bottom_info_renderer,
                renderer.top_info_renderer]:
        lbl = getattr(txt, "text_label", None)
        if lbl is not None:
            lbl.setVisible(False)
            hidden.append(lbl)
    return hidden


def _restore_renderer_ui(hidden: list) -> None:
    for item in hidden:
        item.setVisible(True)


def _get_clean_background(adapter) -> np.ndarray:
    """Export map layer only — cars and all UI text hidden."""
    env = adapter.base_env.unwrapped
    renderer = env.renderer
    if renderer is None:
        adapter.render()
        renderer = env.renderer

    hidden = _hide_renderer_ui(renderer)
    renderer.app.processEvents()
    qimage = renderer.exporter.export(toBytes=True)
    w, h = qimage.width(), qimage.height()
    ptr = qimage.bits()
    ptr.setsize(h * w * 4)
    bg = np.array(ptr).reshape(h, w, 4)[:, :, :3].copy()
    _restore_renderer_ui(hidden)
    return bg


def _agent_mask(frame: np.ndarray, bg: np.ndarray, thresh: int) -> np.ndarray:
    diff = np.abs(frame.astype(np.int32) - bg.astype(np.int32))
    return diff.max(axis=2) > thresh


# Distinct colors per time step (colorblind-friendly sequence)
_STEP_COLORS = [
    (213,  94,   0),  # vermilion
    (  0, 114, 178),  # blue
    (  0, 158, 115),  # green
    (204, 121,  33),  # orange
    (230, 159,   0),  # yellow
    ( 86, 180, 233),  # sky blue
]


def _draw_composite(bg: np.ndarray,
                    sampled: list,
                    alpha: float,
                    diff_thresh: int) -> np.ndarray:
    """
    Build final composite image.

    Each time step gets a distinct color, ghost pixels are tinted with that
    color, and labels are placed with staggered offsets + leader lines so
    they don't overlap.
    """
    composite = bg.astype(np.float32)
    step_info: list[tuple[int, int, float, tuple]] = []  # (cx, cy, t_s, color)

    for idx, (t_s, frame) in enumerate(sampled):
        color = _STEP_COLORS[idx % len(_STEP_COLORS)]
        mask = _agent_mask(frame, bg, diff_thresh)
        mask[:50, :] = False   # suppress top UI
        mask[-55:, :] = False  # suppress bottom UI (time + focus text)

        if not mask.any():
            print(f"  [!] No agent pixels at t={t_s:.1f}s")
            continue

        # Blend real sim car pixels (original f1tenth appearance)
        composite[mask] = (
            composite[mask] * (1.0 - alpha)
            + frame[mask].astype(np.float32) * alpha
        )

        ys_px, xs_px = np.where(mask)
        if len(xs_px):
            step_info.append((int(xs_px.mean()), int(ys_px.mean()), t_s, color))

    out = np.clip(composite, 0, 255).astype(np.uint8)

    # ── Staggered labels with leader lines ───────────────────────────────────
    H, W = out.shape[:2]
    LABEL_OFFSET_X = 45   # px right of centroid
    LABEL_OFFSET_Y = 22   # px vertical stagger per step

    for i, (cx, cy, t_s, color) in enumerate(step_info):
        # Alternate labels above / below the centroid row, stepping outward
        sign = 1 if i % 2 == 0 else -1
        step = (i // 2 + 1) * LABEL_OFFSET_Y
        lx = min(W - 70, cx + LABEL_OFFSET_X)
        ly = max(16, min(H - 16, cy + sign * step))

        bgr = (int(color[2]), int(color[1]), int(color[0]))   # RGB→BGR for cv2

        # Leader line from centroid to label anchor
        cv2.line(out, (cx, cy), (lx - 4, ly), bgr, 1, cv2.LINE_AA)

        # Filled rect behind text for readability
        txt = f"t={t_s:.1f}s"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(out, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2),
                      (255, 255, 255), -1)
        cv2.putText(out, txt, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, bgr, 1, cv2.LINE_AA)

    return out


# ── Runner setup ───────────────────────────────────────────────────────────────

def _init_runner(algo: str, n: int, speed: float):
    if algo == "orca":
        from orca import F1TenthORCARunner, OrcaConfig
        cfg = OrcaConfig(num_cars=n, target_speed=speed, render=False, seed=42)
        runner = F1TenthORCARunner(cfg)
    else:
        from leader_follower import F1TenthLeaderFollower, LeaderFollowerConfig
        cfg = LeaderFollowerConfig(num_cars=n, leader_speed=speed, render=False, seed=42)
        runner = F1TenthLeaderFollower(cfg)

    runner.adapter.render_mode = "rgb_array"
    return runner


def _reset_with_custom_spawn(runner, spawn: np.ndarray, algo: str) -> dict:
    """Reset env to custom spawn poses and update per-runner waypoint indices."""
    obs, _ = runner.adapter.reset(poses=spawn.astype(np.float32))
    n = spawn.shape[0]

    if algo == "orca":
        xs, ys = runner.waypoints_x, runner.waypoints_y
        runner.wp_indices = [
            _nearest_wp_idx(xs, ys, float(spawn[i, 0]), float(spawn[i, 1]))
            for i in range(n)
        ]
    else:
        xs, ys = runner._ext_xs, runner._ext_ys
        runner.wp_idx_cars = [
            _nearest_wp_idx(xs, ys, float(spawn[i, 0]), float(spawn[i, 1]))
            for i in range(n)
        ]
    return obs


def _spawn_at_track_start(runner, n: int, algo: str) -> dict:
    """Reset agents side-by-side at wp[0] of section 1 (obstacle 1 at wp≈216, t≈4.3s).

    Matches record_videos.py exactly: shared CL for all agents (no lateral offset
    per-car PurePursuit), constant leader speed to avoid curvature slowdown.
    """
    try:
        track, _, _, _ = runner.adapter.get_track_data()
        xs = np.asarray(track.centerline.xs[:307], dtype=np.float32)
        ys = np.asarray(track.centerline.ys[:307], dtype=np.float32)
    except Exception:
        print("[!] Track data unavailable, using fallback")
        xs = np.array([0.0, 1.0], dtype=np.float32)
        ys = np.zeros(2, dtype=np.float32)

    theta0 = math.atan2(float(ys[1] - ys[0]), float(xs[1] - xs[0]))
    perp_x, perp_y = -math.sin(theta0), math.cos(theta0)
    lat_gap, half = 0.6, (n - 1) / 2.0
    spawn = np.array(
        [[float(xs[0]) + (i - half) * lat_gap * perp_x,
          float(ys[0]) + (i - half) * lat_gap * perp_y,
          theta0]
         for i in range(n)],
        dtype=np.float32,
    )

    if algo == "lf":
        from leader_follower import PurePursuit
        runner._ext_xs = xs
        runner._ext_ys = ys
        runner._track_closed = False
        runner.cfg.min_speed = runner.cfg.leader_speed  # constant speed
        _tx = np.diff(xs, append=xs[-1] - (xs[-2] - xs[-1]))
        _ty = np.diff(ys, append=ys[-1] - (ys[-2] - ys[-1]))
        _seg = np.hypot(_tx, _ty) + 1e-9
        _nx_arr = (-_ty / _seg).astype(np.float32)  # left-normal x
        _ny_arr = ( _tx / _seg).astype(np.float32)  # left-normal y
        cfg = runner.cfg
        for i in range(n):
            off = (i - half) * lat_gap
            la = cfg.lookahead_dist if i == 0 else cfg.follower_lookahead
            runner.car_pp[i] = PurePursuit(
                (xs + off * _nx_arr).astype(np.float32),
                (ys + off * _ny_arr).astype(np.float32),
                la, cfg.wheelbase, cfg.steer_max, closed=False,
            )
        runner.wp_idx_cars = [0] * n
    else:  # orca
        runner.waypoints_x = xs
        runner.waypoints_y = ys
        runner.track_closed = False
        runner.wp_indices = [0] * n
        runner.lane_offsets = [(i - half) * lat_gap for i in range(n)]

    obs, _ = runner.adapter.reset(poses=spawn)
    return obs


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--algorithm", choices=["orca", "lf"], default="lf")
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--map", default="open_narrow_obs")
    p.add_argument("--target-speed", type=float, default=5.0)
    p.add_argument(
        "--sample-times", type=float, nargs="+",
        default=[0.0, 1.2, 2.4, 3.6, 4.8, 6.0],
        help="Sim-times (s) at which to snapshot agent positions. "
             "Narrow approach lasts ~6s; use 0-30+ for spawn-at-start.",
    )
    p.add_argument("--alpha", type=float, default=0.55,
                   help="Ghost opacity: 0=transparent, 1=opaque")
    p.add_argument("--diff-thresh", type=int, default=20,
                   help="Pixel-diff threshold for agent detection")
    p.add_argument("--spawn", type=float, nargs="*", default=None,
                   help="Custom spawn: flat list of x y theta per agent")
    p.add_argument("--spawn-at-start", action="store_true",
                   help="Spawn agents at track start (section 1) for a longer run")
    # ROI: world-coordinate window — default covers narrow zone + 2 southern obstacles
    # Narrow zone: (40.9, -41.3)  |  obstacles: (54.2, -68.5) and (59.8, -79.8)
    p.add_argument("--roi-cx",   type=float, default=50.0,  help="ROI center x (m)")
    p.add_argument("--roi-cy",   type=float, default=-56.0, help="ROI center y (m)")
    p.add_argument("--roi-padx", type=float, default=13.0,  help="ROI half-width in x (m)")
    p.add_argument("--roi-pady", type=float, default=28.0,  help="ROI half-height in y (m)")
    args = p.parse_args()

    n = args.num_agents
    custom_spawn = (
        np.array(args.spawn, dtype=np.float64).reshape(n, 3)
        if args.spawn else None
    )

    sample_times = sorted(set(args.sample_times))
    dt = 0.01  # sim timestep (s) — both ORCA and LF default to 0.01

    _orig_yaml = _patch_pkg_yaml()
    try:
        runner = _init_runner(args.algorithm, n, args.target_speed)
        obs = runner._init_episode()

        if custom_spawn is not None:
            obs = _reset_with_custom_spawn(runner, custom_spawn, args.algorithm)
        elif args.spawn_at_start:
            obs = _spawn_at_track_start(runner, n, args.algorithm)

        # Lock camera and draw obstacles — renderer is created on first render()
        runner.adapter.render()
        _lock_camera_roi(runner.adapter,
                         args.roi_cx, args.roi_cy, args.roi_padx, args.roi_pady)
        _draw_obstacles_on_renderer(runner.adapter, args.map)

        # ── Collect frames at target sim-times ────────────────────────────────
        sampled: list[tuple[float, np.ndarray]] = []
        next_idx = 0

        # t=0 snapshot before any steps
        if sample_times[next_idx] == 0.0:
            f = runner.adapter.render()
            if f is not None:
                sampled.append((0.0, f.copy()))
            next_idx += 1

        max_steps = int(max(sample_times) / dt) + 50
        for step in range(max_steps):
            if next_idx >= len(sample_times):
                break

            actions = runner._compute_actions(obs)
            try:
                obs, _, done, _, _ = runner.adapter.step(actions)
            except Exception as exc:
                print(f"[!] Sim error at step {step}: {exc}")
                break

            sim_time = (step + 1) * dt
            # Collect all sample times that fall within this step
            while (next_idx < len(sample_times) and
                   sim_time >= sample_times[next_idx] - dt * 0.5):
                f = runner.adapter.render()
                if f is not None:
                    sampled.append((sample_times[next_idx], f.copy()))
                next_idx += 1

            if done:
                print(f"[!] Episode ended at t={sim_time:.2f}s "
                      f"({len(sampled)}/{len(sample_times)} samples captured)")
                break

        # Capture clean background while renderer is still alive
        bg = _get_clean_background(runner.adapter)

    finally:
        runner.adapter.close()
        _restore_pkg_yaml(_orig_yaml)

    if not sampled:
        print("[!] No frames captured.")
        return

    ts_str = ", ".join(f"{t:.1f}" for t, _ in sampled)
    print(f"[✓] Captured {len(sampled)} samples at t = [{ts_str}] s")

    out = _draw_composite(bg, sampled, args.alpha, args.diff_thresh)

    out_path = pathlib.Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved {out_path}  ({out.shape[1]}×{out.shape[0]}px, {len(sampled)} samples)")


if __name__ == "__main__":
    main()
