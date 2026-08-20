#!/usr/bin/env python3
"""
Build a 4-row filmstrip from an MP4 using markers t0, t1, t2, t3.

Each row shows *every* video frame in one segment (4 rows cover the full video):
  row t0 : [0,  t0]
  row t1 : [t0, t1]
  row t2 : [t1, t2]
  row t3 : [t2, t3]     ← set t3 = video duration to include the tail

Example:
  python3 scripts/video_segment_filmstrip.py results/videos/foo.mp4 15 60 120 375

Thumbnails are scaled down so the full strip fits within --max-strip-width.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", help="Input MP4 path")
    p.add_argument(
        "markers", nargs=4, type=float, metavar=("T0", "T1", "T2", "T3"),
        help="Segment end times in seconds (must be strictly increasing)",
    )
    p.add_argument(
        "-o", "--output",
        help="Output PNG (default: results/timeseries/<stem>_segments.png)",
    )
    p.add_argument(
        "--max-strip-width", type=int, default=2400,
        help="Wrap each segment to this max width (pixels)",
    )
    p.add_argument(
        "--row-height", type=int, default=100,
        help="Thumbnail height per frame",
    )
    p.add_argument(
        "--gap", type=int, default=1,
        help="Pixels between consecutive frames in a strip",
    )
    p.add_argument(
        "--include-tail", action="store_true",
        help="Add a 5th row [t3, end] (optional)",
    )
    return p.parse_args()


def _video_duration(cap: cv2.VideoCapture) -> tuple[float, float, int]:
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n / fps if n > 0 else 0.0
    return fps, duration, n


def _extract_segment_frames(
    cap: cv2.VideoCapture,
    fps: float,
    t_start: float,
    t_end: float,
) -> list[tuple[float, np.ndarray]]:
    t_start = max(0.0, t_start)
    t_end = max(t_start, t_end)
    i0 = int(round(t_start * fps))
    i1 = max(i0, int(round(t_end * fps)) - 1)

    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    out: list[tuple[float, np.ndarray]] = []
    for idx in range(i0, i1 + 1):
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        t_s = idx / fps
        if t_s > t_end + 1e-6:
            break
        out.append((t_s, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    return out


def _build_strip(
    frames: list[tuple[float, np.ndarray]],
    label: str,
    row_h: int,
    gap: int,
    max_w: int,
) -> np.ndarray:
    """Pack every frame left-to-right, wrapping to new lines within max_w."""
    label_h = 34
    if not frames:
        block = np.full((row_h + label_h, min(max_w, 400), 3), 240, dtype=np.uint8)
        cv2.putText(block, f"{label}  (no frames)", (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 1, cv2.LINE_AA)
        return block

    aspect = frames[0][1].shape[1] / frames[0][1].shape[0]
    thumb_h = row_h
    thumb_w = max(1, int(round(thumb_h * aspect)))

    # Wrap frames into rows of at most max_w pixels.
    rows: list[list[np.ndarray]] = []
    cur: list[np.ndarray] = []
    x = 0
    for _t, frame in frames:
        thumb = cv2.resize(frame, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA)
        need = thumb_w if not cur else gap + thumb_w
        if cur and x + need > max_w:
            rows.append(cur)
            cur = [thumb]
            x = thumb_w
        else:
            cur.append(thumb)
            x += need
    if cur:
        rows.append(cur)

    strip_w = max(
        sum(t.shape[1] for t in r) + gap * max(0, len(r) - 1) for r in rows
    )
    strip_h = label_h + len(rows) * thumb_h + max(0, len(rows) - 1) * gap
    strip = np.full((strip_h, strip_w, 3), 245, dtype=np.uint8)
    cv2.putText(
        strip, f"{label}  ({len(frames)} frames)", (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA,
    )
    y = label_h
    for row in rows:
        x = 0
        for i, thumb in enumerate(row):
            strip[y : y + thumb_h, x : x + thumb_w] = thumb
            x += thumb_w + (gap if i + 1 < len(row) else 0)
        y += thumb_h + gap
    return strip


def build_segment_filmstrip(
    video_path: str,
    output_path: str,
    t0: float,
    t1: float,
    t2: float,
    t3: float,
    max_strip_width: int = 14000,
    row_height: int = 140,
    gap: int = 1,
    include_tail: bool = False,
) -> str:
    markers = [t0, t1, t2, t3]
    if any(markers[i] >= markers[i + 1] for i in range(3)):
        raise ValueError(f"Markers must be strictly increasing: {markers}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps, duration, _ = _video_duration(cap)

    segments = [
        (0.0, t0, "t0  |  0 → t0"),
        (t0, t1, "t1  |  t0 → t1"),
        (t1, t2, "t2  |  t1 → t2"),
        (t2, t3, "t3  |  t2 → t3"),
    ]
    if include_tail:
        segments.append((t3, duration, f"t4  |  t3 → {duration:.1f}s"))

    strips = []
    for t_a, t_b, label in segments:
        frames = _extract_segment_frames(cap, fps, t_a, t_b)
        strips.append(_build_strip(frames, label, row_height, gap, max_strip_width))
        print(f"  {label}: {len(frames)} frames  [{t_a:.2f}s, {t_b:.2f}s]", file=sys.stderr)
    cap.release()

    pad = 8
    w = max(s.shape[1] for s in strips)
    h = sum(s.shape[0] for s in strips) + pad * (len(strips) + 1)
    canvas = np.full((h, w, 3), 230, dtype=np.uint8)
    y = pad
    for strip in strips:
        canvas[y : y + strip.shape[0], : strip.shape[1]] = strip
        y += strip.shape[0] + pad

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
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
        out = os.path.join(root, "timeseries", f"{stem}_segments.png")

    path = build_segment_filmstrip(
        video, out,
        t0=args.markers[0], t1=args.markers[1],
        t2=args.markers[2], t3=args.markers[3],
        max_strip_width=args.max_strip_width,
        row_height=args.row_height,
        gap=args.gap,
        include_tail=args.include_tail,
    )
    img = cv2.imread(path)
    print(f"Saved {path}  ({img.shape[1]}x{img.shape[0]})")


if __name__ == "__main__":
    main()
