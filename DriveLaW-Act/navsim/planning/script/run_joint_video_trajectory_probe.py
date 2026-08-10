"""Save the complete VAE-decoded DriveLaW Action/Video output without slicing."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Dict

import hydra
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image

from navsim.common.dataloader import SceneLoader
from navsim.planning.script.run_generated_video_idm_probe import (
    _build_features,
    _save_video,
)


CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def _save_decoded_frames(video: torch.Tensor, output_dir: Path) -> int:
    """Save every VAE-decoded frame without assigning temporal semantics."""
    video = video.detach().float().cpu()
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected one decoded video, got shape {tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4 or video.shape[1] not in (1, 3):
        raise ValueError(f"Expected decoded video shaped (T,C,H,W), got {tuple(video.shape)}")
    if video.min() < 0:
        video = (video + 1.0) * 0.5

    output_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in output_dir.glob("frame_*.png"):
        old_frame.unlink()
    for frame_index, frame in enumerate(video):
        frame_uint8 = (
            frame.permute(1, 2, 0).clamp(0, 1).mul(255).round().to(torch.uint8).numpy()
        )
        Image.fromarray(frame_uint8).save(output_dir / f"frame_{frame_index:04d}.png")
    return int(video.shape[0])


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    output_dir = Path(
        str(
            OmegaConf.select(
                cfg,
                "probe.output_dir",
                default="/tmp/drivelaw_joint_video_trajectory_probe",
            )
        )
    )
    video_dir = output_dir / "videos"
    frame_dir = output_dir / "frames"
    raw_decoded_frame_dir = output_dir / "raw_decoded_frames"
    trajectory_dir = output_dir / "trajectories"
    for directory in (video_dir, frame_dir, raw_decoded_frame_dir, trajectory_dir):
        directory.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)

    agent = instantiate(cfg.agent)
    agent.device = device
    agent.initialize()
    agent.eval()

    scene_filter = instantiate(cfg.train_test_split.scene_filter)
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=False,
    )

    limit = int(OmegaConf.select(cfg, "probe.limit", default=4))
    tokens = sorted(scene_loader.tokens)[:limit]
    manifest: Dict[str, object] = {
        "num_samples": len(tokens),
        "world_model_config": str(cfg.agent.config_file),
        "video_num_inference_steps": int(agent.args.video_num_inference_step),
        "action_num_inference_steps": int(agent.args.num_inference_step),
        "freeze_action_video_features": bool(agent.args.freeze_action_video_features),
        "video_fps": int(agent.args.frame_rate),
        "tokens": [],
        "samples": [],
    }
    print(
        "Joint inference: "
        f"Video={manifest['video_num_inference_steps']} steps, "
        f"Action={manifest['action_num_inference_steps']} steps, "
        f"freeze_action_video_features={manifest['freeze_action_video_features']}",
        flush=True,
    )

    with torch.inference_mode():
        for index, token in enumerate(tokens):
            print(f"[{index + 1}/{len(tokens)}] {token}", flush=True)
            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            features = _build_features(agent, agent_input)

            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            predictions = agent.sample_joint_rollouts(
                features,
                sample_time=1,
                build_idm_clip=False,
                extract_future_video=False,
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            inference_seconds = time.perf_counter() - start

            world_trajectory = predictions["actions_tensor"][0].detach().float().cpu()
            decoded_video = predictions["decoded_video"].detach().float().cpu()
            gt_target = agent.get_target_builders()[0].compute_targets(scene)
            gt_trajectory = gt_target["trajectory"]
            if not torch.is_tensor(gt_trajectory):
                gt_trajectory = torch.as_tensor(gt_trajectory)
            gt_trajectory = gt_trajectory.detach().float().cpu()

            print("World model trajectory:", flush=True)
            print(world_trajectory, flush=True)
            print("GT trajectory:", flush=True)
            print(gt_trajectory, flush=True)
            print(
                f"Complete VAE-decoded video frames={decoded_video.shape[1]}, "
                f"inference={inference_seconds:.3f}s",
                flush=True,
            )

            stem = f"{index:04d}_{token}"
            torch.save(decoded_video, frame_dir / f"{stem}_decoded_frames.pt")
            raw_frame_count = _save_decoded_frames(
                decoded_video,
                raw_decoded_frame_dir / stem,
            )
            _save_video(
                decoded_video,
                video_dir / f"{stem}_raw_decoded_all_frames.mp4",
                fps=float(agent.args.frame_rate),
            )
            torch.save(
                {
                    "token": token,
                    "world_model_trajectory": world_trajectory,
                    "gt_trajectory": gt_trajectory,
                    "decoded_video_frames": int(decoded_video.shape[1]),
                    "raw_decoded_png_frames": raw_frame_count,
                    "inference_seconds": inference_seconds,
                },
                trajectory_dir / f"{stem}_trajectories.pt",
            )

            manifest["tokens"].append(token)
            manifest["samples"].append(
                {
                    "token": token,
                    "decoded_video_frames": int(decoded_video.shape[1]),
                    "raw_decoded_png_frames": raw_frame_count,
                    "inference_seconds": inference_seconds,
                }
            )

    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print(f"Saved probe artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
