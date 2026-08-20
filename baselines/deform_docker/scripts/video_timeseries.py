#!/usr/bin/env python3
"""Extract evenly spaced frames from an MP4 and compose a labeled time-series grid."""

from __future__ import annotations

import argparse
import math
import os
import sys

import cv2
import numpy as np


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("video", help="Input MP4 path")
    p.add_argument(
        "-o", "--output",
        help="Output PNG path (default: results/timeseries/<video_stem>_timeseries.png)",
    )
    p.add_argument(
        "-n", "--num-frames", type=int, default=16,
        help="Number of frames to sample (default: 16)",
    )
    p.add_argument(
        "--cols", type=int, default=0,
        help="Grid columns (0 = auto, roughly sqrt(n))",
    )
    p.add_argument(
        "--thumb", type=int, default=320,
        help="Thumbnail width in pixels (height scales to keep aspect)",
    )
    p.add_argument(
        "--start-s", type=float, default=0.0,
        help="Start sampling time in seconds",
    )
    p.add_argument(
        "--end-s", type=float, default=-1.0,
        help="End sampling time in seconds (-1 = video duration)",
    )
    return p.parse_args()


def _grid_shape(n: int, cols: int) -> tuple[int, int]:
    if cols > 0:
        rows = math.ceil(n / cols)
        return rows, cols
    cols = max(1, round(math.sqrt(n)))
    rows = math.ceil(n / cols)
    return rows, cols


def _read_frame_at(cap: cv2.VideoCapture, t_s: float) -> np.ndarray | None:
    cap.set(cv2.CAP_PROP_POS_MSEC, max(0.0, t_s) * 1000.0)
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _label_frame(frame: np.ndarray, t_s: float, w: int) -> np.ndarray:
    h = int(frame.shape[0] * w / frame.shape[1])
    thumb = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
    bar_h = 28
    canvas = np.full((h + bar_h, w, 3), 255, dtype=np.uint8)
    canvas[bar_h:, :] = thumb
    label = f"t = {t_s:6.1f} s"
    cv2.putText(
        canvas, label, (8, 19),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1, cv2.LINE_AA,
    )
    return canvas


def build_timeseries(
    video_path: str,
    output_path: str,
    num_frames: int = 16,
    cols: int = 0,
    thumb_w: int = 320,
    start_s: float = 0.0,
    end_s: float = -1.0,
) -> str:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_total / fps if n_total > 0 else 0.0
    if duration <= 0:
        duration = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
    cap.release()

    if end_s < 0:
        end_s = duration
    end_s = min(end_s, max(0.0, duration - 0.05))
    start_s = max(0.0, min(start_s, end_s))

    if num_frames < 1:
        raise ValueError("num_frames must be >= 1")
    if num_frames == 1:
        times = [start_s]
    else:
        times = np.linspace(start_s, end_s, num_frames)

    cap = cv2.VideoCapture(video_path)
    cells = []
    for t in times:
        frame = _read_frame_at(cap, float(t))
        if frame is None:
            print(f"  [!] skip t={t:.1f}s (read failed)", file=sys.stderr)
            continue
        cells.append(_label_frame(frame, float(t), thumb_w))
    cap.release()

    if not cells:
        raise RuntimeError("No frames extracted")

    rows, cols = _grid_shape(len(cells), cols)
    cell_h, cell_w = cells[0].shape[:2]
    pad = 6
    grid = np.full(
        (rows * (cell_h + pad) + pad, cols * (cell_w + pad) + pad, 3),
        240, dtype=np.uint8,
    )
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, cols)
        y0 = pad + r * (cell_h + pad)
        x0 = pad + c * (cell_w + pad)
        grid[y0 : y0 + cell_h, x0 : x0 + cell_w] = cell

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    cv2.imwrite(output_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
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
        out = os.path.join(root, "timeseries", f"{stem}_timeseries.png")

    path = build_timeseries(
        video,
        out,
        num_frames=args.num_frames,
        cols=args.cols,
        thumb_w=args.thumb,
        start_s=args.start_s,
        end_s=args.end_s,
    )
    img = cv2.imread(path)
    h, w = img.shape[:2]
    print(f"Saved {path}  ({w}x{h}, {args.num_frames} samples)")


if __name__ == "__main__":
    main()
