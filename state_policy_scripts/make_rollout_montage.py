#!/usr/bin/env python3
"""Tile the rollout videos of one policy into a single montage MP4 (+ contact-sheet PNG).
usage: make_rollout_montage.py <rollouts_dir/task> [--cols 5] [--scale 0.5]
"""
import argparse, glob
from pathlib import Path
import imageio.v2 as iio
import numpy as np
from PIL import Image, ImageDraw

ap = argparse.ArgumentParser()
ap.add_argument("task_dir", type=Path)
ap.add_argument("--cols", type=int, default=5)
ap.add_argument("--scale", type=float, default=0.5)
ap.add_argument("--front-only", action="store_true", help="crop to the left (front) camera")
a = ap.parse_args()

paths = sorted(a.task_dir.glob("rollout_*.mp4"))
clips = []
for p in paths:
    frames = [f for f in iio.get_reader(p)]
    if a.front_only:
        w = frames[0].shape[1] // 2
        frames = [f[:, :w] for f in frames]
    h, w = frames[0].shape[:2]
    nh, nw = int(h * a.scale), int(w * a.scale)
    resized = []
    for i, f in enumerate(frames):
        im = Image.fromarray(f).resize((nw, nh), Image.BILINEAR)
        ImageDraw.Draw(im).text((6, 4), f"{p.stem}  t={i/15:4.1f}s", fill=(255, 255, 0))
        resized.append(np.asarray(im))
    clips.append(resized)
n = len(clips); rows = int(np.ceil(n / a.cols)); T = min(len(c) for c in clips)
h, w = clips[0][0].shape[:2]
out_frames = []
for t in range(T):
    canvas = np.zeros((rows * h, a.cols * w, 3), np.uint8)
    for k, c in enumerate(clips):
        r, col = divmod(k, a.cols)
        canvas[r * h:(r + 1) * h, col * w:(col + 1) * w] = c[t]
    out_frames.append(canvas)
out = a.task_dir / f"montage_{a.task_dir.name}.mp4"
iio.mimwrite(out, out_frames, fps=15, codec="libx264", quality=6, macro_block_size=1)
sheet_idx = np.linspace(0, T - 1, 6).astype(int)
sheet = np.concatenate([out_frames[i] for i in sheet_idx], axis=0)
iio.imwrite(a.task_dir / f"contact_sheet_{a.task_dir.name}.png", sheet)
print(f"{out}: {len(out_frames)} frames {out_frames[0].shape}, {out.stat().st_size/1e6:.1f} MB")
