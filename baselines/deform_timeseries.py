#!/usr/bin/env python3
"""
Stroboscopic timeseries from a deform video — matches timeseries_image.py style.

Uses ECC frame alignment to compensate for the follow-camera (focus_on: agent_0),
then calls the same _draw_composite() used for ORCA/LF so label style is identical.

Usage (from baselines/):
    python deform_timeseries.py \\
        deform_docker/results/videos/deform_4agents_staggered_clear.mp4 \\
        -o vids/timeseries/ts_deform_4agent_narrow.png \\
        --times 95 105 115 125 135 \\
        --ref-time 80
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from timeseries_image import _STEP_COLORS  # reuse color scheme


def _read_rgb(cap: cv2.VideoCapture, t_s: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
    ok, frame = cap.read()
    if not ok:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _to_gray_blur(rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _ecc_align(ref_g: np.ndarray, mov_g: np.ndarray,
               frame: np.ndarray) -> np.ndarray:
    """Euclidean (translation + rotation) ECC warp of frame onto ref."""
    warp = np.eye(2, 3, dtype=np.float32)
    try:
        _, warp = cv2.findTransformECC(
            ref_g, mov_g, warp, cv2.MOTION_EUCLIDEAN,
            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5),
        )
    except cv2.error:
        pass
    h, w = frame.shape[:2]
    return cv2.warpAffine(frame, warp, (w, h),
                          flags=cv2.INTER_LINEAR, borderValue=(245, 245, 245))


def _strip_ui(frame: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Replace top/bottom HUD bands with clean ref pixels."""
    out = frame.copy()
    out[:48, :] = ref[:48, :]
    out[-58:, :] = ref[-58:, :]
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("video")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--times", type=float, nargs="+",
                   default=[95.0, 105.0, 115.0, 125.0, 135.0],
                   help="Video timestamps (seconds) to sample")
    p.add_argument("--ref-time", type=float, default=80.0,
                   help="Reference background frame (seconds)")
    p.add_argument("--alpha", type=float, default=0.60,
                   help="Ghost opacity")
    p.add_argument("--diff-thresh", type=int, default=30,
                   help="Pixel-diff threshold for agent detection")
    p.add_argument("--crop-pad", type=int, default=120,
                   help="Pixels of padding around agent region when cropping")
    p.add_argument("--no-crop", action="store_true",
                   help="Disable auto-crop (keep full frame)")
    args = p.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit(f"Cannot open {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps

    # ── Background / reference frame ─────────────────────────────────────────
    bg = _read_rgb(cap, args.ref_time)
    if bg is None:
        sys.exit(f"Cannot read ref frame at t={args.ref_time}s")
    bg = _strip_ui(bg, np.full_like(bg, 245))  # clean bands → neutral grey
    ref_g = _to_gray_blur(bg)

    # ── Sample and align frames ───────────────────────────────────────────────
    sampled: list[tuple[float, np.ndarray]] = []
    for t_s in sorted(args.times):
        if t_s > dur:
            print(f"  [!] t={t_s:.1f}s exceeds video duration ({dur:.1f}s), skipping")
            continue
        frame = _read_rgb(cap, t_s)
        if frame is None:
            print(f"  [!] Could not read frame at t={t_s:.1f}s")
            continue
        aligned = _ecc_align(ref_g, _to_gray_blur(frame), frame)
        aligned = _strip_ui(aligned, bg)
        sampled.append((t_s, aligned))

    cap.release()

    if not sampled:
        sys.exit("[!] No frames captured.")

    ts_str = ", ".join(f"{t:.1f}" for t, _ in sampled)
    print(f"[✓] Captured {len(sampled)} samples at t = [{ts_str}] s")

    # ── Build composite — HSV masking avoids ECC alignment artifacts ─────────
    # HSV saturation detects colored cars only; black walls + grey BG are excluded
    def _hsv_agent_mask(frame_rgb: np.ndarray) -> np.ndarray:
        bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        m = ((hsv[:, :, 1] > 80) & (hsv[:, :, 2] > 60) & (hsv[:, :, 2] < 240))
        m[:50, :] = False
        m[-55:, :] = False
        m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((3, 3), np.uint8)) > 0
        return m

    composite = bg.astype(np.float32)
    step_info: list[tuple[int, int, float, tuple]] = []
    union = np.zeros(bg.shape[:2], bool)

    for idx, (t_s, frame) in enumerate(sampled):
        color = _STEP_COLORS[idx % len(_STEP_COLORS)]
        mask = _hsv_agent_mask(frame)
        union |= mask
        if not mask.any():
            print(f"  [!] No agent pixels at t={t_s:.1f}s")
            continue
        composite[mask] = (
            composite[mask] * (1.0 - args.alpha)
            + frame[mask].astype(np.float32) * args.alpha
        )
        ys_px, xs_px = np.where(mask)
        if len(xs_px):
            step_info.append((int(xs_px.mean()), int(ys_px.mean()), t_s, color))

    out = np.clip(composite, 0, 255).astype(np.uint8)

    # ── Staggered leader-line labels (same style as timeseries_image.py) ─────
    H, W = out.shape[:2]
    LABEL_OFFSET_X = 45
    LABEL_OFFSET_Y = 22
    for i, (cx, cy, t_s, color) in enumerate(step_info):
        sign = 1 if i % 2 == 0 else -1
        step = (i // 2 + 1) * LABEL_OFFSET_Y
        lx = min(W - 70, cx + LABEL_OFFSET_X)
        ly = max(16, min(H - 16, cy + sign * step))
        bgr = (int(color[2]), int(color[1]), int(color[0]))
        cv2.line(out, (cx, cy), (lx - 4, ly), bgr, 1, cv2.LINE_AA)
        txt = f"t={t_s:.1f}s"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        cv2.rectangle(out, (lx - 2, ly - th - 2), (lx + tw + 2, ly + 2),
                      (255, 255, 255), -1)
        cv2.putText(out, txt, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, bgr, 1, cv2.LINE_AA)

    # ── Auto-crop to agent region ─────────────────────────────────────────────
    if not args.no_crop:
        union[:50, :] = False; union[-55:, :] = False
        ys_u, xs_u = np.where(union)
        if len(xs_u):
            pad = args.crop_pad
            x0 = max(0, xs_u.min() - pad)
            x1 = min(W, xs_u.max() + pad)
            y0 = max(0, ys_u.min() - pad)
            y1 = min(H, ys_u.max() + pad)
            out = out[y0:y1, x0:x1]
            print(f"  cropped [{x0}:{x1}, {y0}:{y1}] → {out.shape[1]}×{out.shape[0]}px")

    out_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print(f"[✓] Saved {out_path}  ({out.shape[1]}×{out.shape[0]}px)")


if __name__ == "__main__":
    main()
