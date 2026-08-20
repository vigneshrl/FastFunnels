#!/usr/bin/env python3
"""Stroboscopic timeseries on fixed MAP camera (focus_on: null) — paper-style figure."""

from __future__ import annotations

import argparse
import os
import tempfile

import cv2
import numpy as np

# Match video_timeseries_overlay.py (BGR)
LABEL_COLORS = [
    (220, 140, 60),
    (50, 140, 230),
    (80, 180, 70),
    (200, 160, 80),
    (80, 80, 220),
    (180, 90, 180),
]


def _centroids(mask: np.ndarray, min_area: int = 35) -> list[tuple[int, int]]:
    n, _lbl, stats, cents = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8) * 255, 8)
    return [
        (int(cents[i, 0]), int(cents[i, 1]))
        for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]


def _draw_orca_labels(
    img: np.ndarray,
    labels: list[tuple[int, int, float]],
    font_scale: float,
    time_decimals: int = 1,
) -> None:
    """ORCA-style labels on the right; y tracks each snapshot (min gap if tight)."""
    if not labels:
        return
    labels = sorted(labels, key=lambda x: x[2])
    h, w = img.shape[:2]
    margin = 14
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale * 2))

    sizes = [cv2.getTextSize(f"t={t:.{max(0, time_decimals)}f}s", font, font_scale, thickness)
             for _, _, t in labels]
    row_h = max(th + bl + 8 for (_, th), bl in sizes)

    # Target y from agent centroid; enforce minimum vertical spacing.
    ly_targets = [int(np.clip(cy + row_h // 2, row_h, h - margin)) for _, cy, _ in labels]
    for i in range(1, len(ly_targets)):
        if ly_targets[i] < ly_targets[i - 1] + row_h:
            ly_targets[i] = ly_targets[i - 1] + row_h
    for i in range(len(ly_targets) - 2, -1, -1):
        if ly_targets[i] > ly_targets[i + 1] - row_h:
            ly_targets[i] = ly_targets[i + 1] - row_h
    ly_targets = [int(np.clip(y, row_h, h - margin)) for y in ly_targets]

    lx_base = w - margin - max(s[0][0] for s in sizes)
    for i, ((cx, cy, t_s), (ly, ((tw, th), baseline))) in enumerate(
        zip(labels, zip(ly_targets, sizes))
    ):
        color = LABEL_COLORS[i % len(LABEL_COLORS)]
        txt = f"t={t_s:.{max(0, time_decimals)}f}s"
        lx = max(cx + 28, lx_base - (i % 2) * (tw + 16))
        anchor = (lx - 6, ly - th // 2)
        cv2.line(img, (cx, cy), anchor, color, 2, cv2.LINE_AA)
        cv2.putText(img, txt, (lx, ly), font, font_scale, color, thickness, cv2.LINE_AA)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--num-agents", type=int, default=2)
    p.add_argument("--map", default="open_narrow_obs")
    p.add_argument(
        "--times", type=float, nargs="+", default=[10.0, 11.0, 12.0, 13.0],
        help="Labels for each ghost step (ORCA-style: 4 samples)",
    )
    p.add_argument("--time-decimals", type=int, default=1)
    p.add_argument("--alpha", type=float, default=0.55)
    p.add_argument("--zoom", type=float, default=1.15,
                   help="Map zoom (higher = closer, default 1.15)")
    p.add_argument("--out-width", type=int, default=1400,
                   help="Upscale cropped image to this width (0 = full map)")
    p.add_argument("--crop-padding", type=int, default=140)
    p.add_argument("--font-scale", type=float, default=0.8)
    p.add_argument(
        "--spawn", type=float, nargs="*", default=[
            40.800, -28.500, -1.5708,
            41.300, -28.820, -1.5708,
        ],
    )
    p.add_argument("--dy", type=float, default=-5.5,
                   help="Southward y shift per step (m); ~4 m ≈ 1 s at narrow speed")
    args = p.parse_args()

    os.environ["SDL_VIDEODRIVER"] = "dummy"
    import gymnasium as gym
    import f1tenth_gym  # noqa: F401

    n = args.num_agents
    spawn = np.array(args.spawn, dtype=np.float64).reshape(n, 3)

    yaml = (
        f"window_size: 800\nfocus_on: null\nzoom_in_factor: {args.zoom}\n"
        "show_wheels: True\ncar_tickness: 1\nshow_info: False\n"
        "vehicle_palette:\n  - '#984ea3'\n  - '#e41a1c'\n  - '#ff7f00'\n  - '#a65628'\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write(yaml)
        ypath = f.name

    env = gym.make(
        "f1tenth_gym:f1tenth-v0",
        config={
            "num_agents": n, "map": args.map, "timestep": 0.01,
            "integrator": "rk4", "model": "st",
            "control_input": ["speed", "steering_angle"],
            "observation_config": {"type": "original"},
            "rendering": ypath,
        },
        render_mode="rgb_array",
    )

    env.reset(options={"poses": spawn})
    bg = env.render()
    composite = bg.astype(np.float32)
    labels: list[tuple[int, int, float]] = []
    agent_union = np.zeros(bg.shape[:2], dtype=bool)

    for step_i, t_s in enumerate(args.times):
        poses = spawn.copy()
        poses[:, 1] += step_i * args.dy
        env.reset(options={"poses": poses})
        frame = env.render()
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        m = (hsv[:, :, 1] > 65) & (hsv[:, :, 2] > 50) & (hsv[:, :, 2] < 245)
        m[:40, :] = False
        m[-50, :] = False
        agent_union |= m
        composite[m] = composite[m] * (1 - args.alpha) + frame[m].astype(np.float32) * args.alpha

        cents = _centroids(m)
        if cents:
            cx = int(np.mean([c[0] for c in cents]))
            cy = int(np.mean([c[1] for c in cents]))
            labels.append((cx, cy, t_s))

    env.close()
    os.unlink(ypath)

    out = np.clip(composite, 0, 255).astype(np.uint8)
    out[:40, :] = bg[:40, :]
    out[-50:, :] = bg[-50:, :]

    pad = args.crop_padding
    h, w = out.shape[:2]
    xs, ys = [], []
    for cx, cy, _ in labels:
        xs.append(cx)
        ys.append(cy)
    if agent_union.any():
        ay, ax = np.where(agent_union)
        xs.extend(ax.tolist())
        ys.extend(ay.tolist())
    if xs:
        x0, x1 = max(0, min(xs) - pad), min(w, max(xs) + pad)
        y0, y1 = max(0, min(ys) - pad), min(h, max(ys) + pad)
        out = out[y0:y1, x0:x1].copy()
        labels = [(cx - x0, cy - y0, t) for cx, cy, t in labels]

    if args.out_width > 0 and out.shape[1] > 0:
        scale = args.out_width / out.shape[1]
        nh = max(1, int(out.shape[0] * scale))
        out = cv2.resize(out, (args.out_width, nh), interpolation=cv2.INTER_LINEAR)
        labels = [(int(cx * scale), int(cy * scale), t) for cx, cy, t in labels]
        args.font_scale *= scale

    _draw_orca_labels(out, labels, args.font_scale, args.time_decimals)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"Saved {args.output}  ({out.shape[1]}x{out.shape[0]}, {len(args.times)} samples)")


if __name__ == "__main__":
    main()
