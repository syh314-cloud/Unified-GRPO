"""Convert sparse NAVSIM probe frames into a 9-frame DriveLaW-Video condition clip."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path

from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output-frames-dir", type=Path, required=True)
    parser.add_argument("--fps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_paths = sorted(args.input_dir.glob("input_[0-9][0-9].jpg"))
    if len(input_paths) != 4:
        raise ValueError(
            f"Expected four NAVSIM input frames in {args.input_dir}, found {len(input_paths)}"
        )
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to create the conditioning MP4")

    # DriveLaW-Video's official condition clip contains 9 frames at 8 FPS,
    # spanning one second. NAVSIM inputs are at 2 Hz, so the last three frames
    # provide samples at t={0.0, 0.5, 1.0}s for temporal interpolation.
    source_frames = [Image.open(path).convert("RGB") for path in input_paths[-3:]]
    if len({frame.size for frame in source_frames}) != 1:
        raise ValueError("All NAVSIM condition frames must have the same resolution")

    args.output_frames_dir.mkdir(parents=True, exist_ok=True)
    for old_path in args.output_frames_dir.glob("frame_*.png"):
        old_path.unlink()

    num_output_frames = args.fps + 1
    for index in range(num_output_frames):
        source_position = index / args.fps * 2.0
        left = min(math.floor(source_position), len(source_frames) - 1)
        right = min(left + 1, len(source_frames) - 1)
        alpha = source_position - left
        frame = Image.blend(source_frames[left], source_frames[right], alpha)
        frame.save(args.output_frames_dir / f"frame_{index:02d}.png")

    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(args.fps),
            "-i",
            str(args.output_frames_dir / "frame_%02d.png"),
            "-c:v",
            "mpeg4",
            "-q:v",
            "2",
            "-pix_fmt",
            "yuv420p",
            str(args.output_video),
        ],
        check=True,
    )
    print(f"Saved {num_output_frames}-frame conditioning video to {args.output_video}")


if __name__ == "__main__":
    main()
