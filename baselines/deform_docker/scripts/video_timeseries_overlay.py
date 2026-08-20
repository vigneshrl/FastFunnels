#!/usr/bin/env python3
"""
Stroboscopic timeseries image (ORCA / paper style):
  - static background (reference frame)
  - semi-transparent agent ghosts at each sample time
  - color-coded t= labels with leader lines to each snapshot

Uses ECC alignment when the gym camera pans (focus_on: agent_0).
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video")
    p.add_argument("-o", "--output", help="Output PNG path")
    p.add_argument("--times", type=float, nargs="*", help="Sample times (seconds)")
    p.add_argument("--dt", type=float, default=1.0,
                   help="Sample interval (seconds) if --times omitted")
    p.add_argument("--ref-time", type=float, default=120.0,
                   help="Background frame + alignment reference (seconds)")
    p.add_argument("--alpha", type=float, default=0.55, help="Ghost opacity")
    p.add_argument("--max-samples", type=int, default=8)
    p.add_argument("--crop-padding", type=int, default=120,
                   help="Pixels around agent trail for auto-crop")
    p.add_argument("--out-width", type=int, default=1400,
                   help="Upscale cropped image to this width (0 = no upscale)")
    p.add_argument("--font-scale", type=float, default=0.65,
                   help="Timestamp label font scale")
    p.add_argument(
        "--time-decimals", type=int, default=1,
        help="Decimal places in t= labels (ORCA uses 1)",
    )
    return p.parse_args()


def _open_video(path: str) -> tuple[cv2.VideoCapture, float, float]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, n / fps if n else 0.0


def _read_at(cap: cv2.VideoCapture, t_s: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
    ok, f = cap.read()
    return f


def _gray(frame: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return cv2.GaussianBlur(g, (5, 5), 0)


def _align_euclidean(ref_g: np.ndarray, mov_g: np.ndarray) -> np.ndarray:
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            ref_g, mov_g, warp, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
        )
    except cv2.error:
        pass
    return warp


# ORCA-style label colours (BGR): blue, orange, green, teal, red, purple
LABEL_COLORS = [
    (220, 140, 60),   # blue
    (50, 140, 230),   # orange
    (80, 180, 70),    # green
    (200, 160, 80),   # teal
    (80, 80, 220),    # red
    (180, 90, 180),   # purple
]


def _agent_mask(frame: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    # Agent colours only (exclude grey UI text and black walls).
    mask = ((s > 85) & (v > 70) & (v < 235)).astype(np.uint8) * 255
    mask[:48, :] = 0
    mask[-58:, :] = 0
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return mask


def _strip_ui_bands(frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Replace pygame HUD bands with clean background from reference."""
    out = frame.copy()
    out[:48, :] = ref[:48, :]
    out[-58:, :] = ref[-58:, :]
    return out


def _centroids(mask: np.ndarray, min_area: int = 35) -> list[tuple[int, int]]:
    n, _lbl, stats, cents = cv2.connectedComponentsWithStats(mask, 8)
    return [
        (int(cents[i, 0]), int(cents[i, 1]))
        for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area
    ]


def _sample_times(dur: float, dt: float, explicit: list[float] | None, cap: int) -> list[float]:
    if explicit:
        ts = sorted({max(0.0, min(t, dur - 0.05)) for t in explicit})
    else:
        ts = list(np.linspace(0, max(0, dur - 0.05), max(2, int(dur / dt) + 1)))
    if len(ts) > cap:
        idx = np.linspace(0, len(ts) - 1, cap, dtype=int)
        ts = [ts[i] for i in idx]
    return ts


def _crop_box(
    labels: list[tuple[int, int, float]],
    agent_pixels: np.ndarray,
    shape: tuple[int, int],
    pad: int,
) -> tuple[int, int, int, int]:
    h, w = shape
    xs, ys = [], []
    for cx, cy, _ in labels:
        xs.append(cx)
        ys.append(cy)
    if agent_pixels.size:
        ys_px, xs_px = np.where(agent_pixels)
        xs.extend(xs_px.tolist())
        ys.extend(ys_px.tolist())
    if not xs:
        return 0, 0, w, h
    x0 = max(0, min(xs) - pad)
    x1 = min(w, max(xs) + pad)
    y0 = max(0, min(ys) - pad)
    y1 = min(h, max(ys) + pad)
    return x0, y0, x1, y1


def _draw_labels(
    img: np.ndarray,
    labels: list[tuple[int, int, float]],
    font_scale: float,
    time_decimals: int = 1,
) -> None:
    """ORCA-style: colored t= labels on the right, y tracks each snapshot."""
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


def build_overlay(
    video_path: str,
    output_path: str,
    times: list[float] | None = None,
    dt: float = 10.0,
    ref_time: float = 120.0,
    alpha: float = 0.55,
    max_samples: int = 20,
    crop_padding: int = 120,
    out_width: int = 1400,
    font_scale: float = 0.75,
    time_decimals: int = 1,
) -> str:
    cap, _fps, dur = _open_video(video_path)
    ref = _read_at(cap, ref_time)
    if ref is None:
        raise RuntimeError(f"Cannot read ref frame t={ref_time}s")
    ref_g = _gray(ref)
    composite = ref.astype(np.float32)
    labels: list[tuple[int, int, float]] = []
    agent_union = np.zeros(ref.shape[:2], dtype=bool)

    sample_ts = _sample_times(dur, dt, times, max_samples)
    print(f"  dur={dur:.1f}s  ref={ref_time:.1f}s  n={len(sample_ts)}", file=sys.stderr)

    for t_s in sample_ts:
        frame = _read_at(cap, t_s)
        if frame is None:
            continue
        warp = _align_euclidean(ref_g, _gray(frame))
        warped = _strip_ui_bands(
            cv2.warpAffine(
                frame, warp, (ref.shape[1], ref.shape[0]),
                flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245),
            ),
            ref,
        )
        m = cv2.warpAffine(
            _agent_mask(frame), warp, (ref.shape[1], ref.shape[0]),
            flags=cv2.INTER_NEAREST, borderValue=0,
        ) > 0
        if not np.any(m):
            continue
        agent_union |= m
        composite[m] = composite[m] * (1 - alpha) + warped[m].astype(np.float32) * alpha
        cents = _centroids(m.astype(np.uint8) * 255)
        if cents:
            cx = int(np.mean([c[0] for c in cents]))
            cy = int(np.mean([c[1] for c in cents]))
            labels.append((cx, cy, t_s))

    cap.release()
    full = np.clip(composite, 0, 255).astype(np.uint8)
    full[:48, :] = ref[:48, :]
    full[-58:, :] = ref[-58:, :]

    x0, y0, x1, y1 = _crop_box(labels, agent_union, full.shape[:2], crop_padding)
    cropped = full[y0:y1, x0:x1].copy()
    labels_c = [(cx - x0, cy - y0, t) for cx, cy, t in labels]

    if out_width > 0 and cropped.shape[1] > 0:
        scale = out_width / cropped.shape[1]
        nh = max(1, int(cropped.shape[0] * scale))
        cropped = cv2.resize(cropped, (out_width, nh), interpolation=cv2.INTER_LINEAR)
        labels_c = [(int(cx * scale), int(cy * scale), t) for cx, cy, t in labels_c]
        font_scale *= scale

    _draw_labels(cropped, labels_c, font_scale, time_decimals)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cropped)
    print(f"  crop [{x0}:{x1}, {y0}:{y1}] → {cropped.shape[1]}x{cropped.shape[0]}",
          file=sys.stderr)
    return output_path


def main() -> None:
    args = _parse_args()
    video = os.path.abspath(args.video)
    if args.output:
        out = os.path.abspath(args.output)
    else:
        stem = os.path.splitext(os.path.basename(video))[0]
        root = os.path.dirname(os.path.dirname(video))
        if os.path.basename(root) == "videos":
            root = os.path.dirname(root)
        out = os.path.join(root, "timeseries", f"{stem}_overlay.png")

    path = build_overlay(
        video, out,
        times=args.times if args.times else None,
        dt=args.dt,
        ref_time=args.ref_time,
        alpha=args.alpha,
        max_samples=args.max_samples,
        crop_padding=args.crop_padding,
        out_width=args.out_width,
        font_scale=args.font_scale,
        time_decimals=args.time_decimals,
    )
    im = cv2.imread(path)
    print(f"Saved {path}  ({im.shape[1]}x{im.shape[0]})")


if __name__ == "__main__":
    main()
