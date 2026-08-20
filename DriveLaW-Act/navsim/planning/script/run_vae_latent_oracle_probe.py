"""Check whether the joint training latent target is independently VAE-decodable."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import hydra
import torch
from einops import rearrange
from hydra.utils import get_original_cwd, instantiate
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from yaml import Loader, load

# The legacy trainer imports its local helper package as top-level `utils`.
# Put that package ahead of navsim/planning/script/utils.py before importing it.
VIDEODRIVE_DIR = Path(__file__).resolve().parents[2] / "agents" / "videodrive"
sys.path.insert(0, str(VIDEODRIVE_DIR))

from navsim.agents.videodrive.run_training_videodrive import (
    CacheOnlyDatasetLazyIO,
    navsim_genie_collate_fn,
)
from navsim.agents.videodrive.utils.data_utils import get_latents
from navsim.agents.videodrive.utils.model_loader import load_vae


CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


@torch.no_grad()
def _decode_training_latents(vae, latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Invert the exact normalization used by get_latents(), then decode."""
    mean = vae.latents_mean.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    std = vae.latents_std.view(1, -1, 1, 1, 1).to(latents.device, latents.dtype)
    latents = (latents * std + mean).to(dtype=dtype)
    timestep = None
    if vae.config.timestep_conditioning:
        timestep = torch.zeros(latents.shape[0], device=latents.device, dtype=dtype)
    return vae.decode(latents, temb=timestep, return_dict=False)[0].clamp(-1, 1)


def _save_video_and_frames(video: torch.Tensor, root: Path, fps: int) -> int:
    """Save a (B,C,T,H,W) or (C,T,H,W) video as PNG frames and MP4."""
    if video.ndim == 5:
        if video.shape[0] != 1:
            raise ValueError(f"Expected batch size 1, got {tuple(video.shape)}")
        video = video[0]
    if video.ndim != 4 or video.shape[0] not in (1, 3):
        raise ValueError(f"Expected (C,T,H,W), got {tuple(video.shape)}")

    video = video.detach().float().cpu().clamp(-1, 1)
    frames = ((video + 1.0) * 127.5).round().to(torch.uint8)
    frame_dir = root / "frames"
    frame_dir.mkdir(parents=True, exist_ok=True)
    for index in range(frames.shape[1]):
        array = frames[:, index].permute(1, 2, 0).numpy()
        Image.fromarray(array).save(frame_dir / f"frame_{index:04d}.png")

    try:
        import torchvision

        torchvision.io.write_video(
            str(root / "video.mp4"),
            frames.permute(1, 2, 3, 0),
            fps=fps,
        )
    except Exception as exc:
        print(f"Warning: MP4 export failed for {root}: {exc}", flush=True)
    return int(frames.shape[1])


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The VAE oracle probe requires one CUDA GPU")

    output_dir = Path(
        str(
            OmegaConf.select(
                cfg,
                "probe.output_dir",
                default="/tmp/drivelaw_vae_latent_oracle",
            )
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    sample_index = int(OmegaConf.select(cfg, "probe.index", default=0))

    config_file = str(cfg.agent.config_file)
    if not os.path.isabs(config_file):
        config_file = os.path.join(get_original_cwd(), config_file)
    with open(config_file, "r", encoding="utf-8") as stream:
        args = argparse.Namespace(**load(stream, Loader=Loader))

    agent = instantiate(cfg.agent)
    dataset = CacheOnlyDatasetLazyIO(
        cache_path=str(cfg.cache_path),
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        log_names=cfg.train_logs,
        min_hist_frames=8,
    )
    if not 0 <= sample_index < len(dataset):
        raise IndexError(f"probe.index={sample_index} outside dataset size {len(dataset)}")

    token = dataset.tokens[sample_index]
    batch = navsim_genie_collate_fn([dataset[sample_index]])
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    torch.manual_seed(int(getattr(args, "seed", 42)))
    torch.cuda.manual_seed_all(int(getattr(args, "seed", 42)))

    history = batch["video"].to(device=device, dtype=dtype).contiguous()
    history = rearrange(history, "b c v t h w -> (b v) c t h w")
    future = batch["future_video"].to(device=device, dtype=dtype).contiguous()
    future = rearrange(future, "b c v t h w -> (b v) c t h w")
    if history.shape[2] != 4 or future.shape[2] != 8:
        raise ValueError(
            f"Expected 4 history and 8 future frames, got {history.shape[2]} and {future.shape[2]}"
        )
    target_clip = torch.cat([history[:, :, -1:], future], dim=2)

    vae = load_vae(args.pretrained_model_name_or_path, dtype=dtype).to(device).eval()
    vae.enable_slicing()
    vae.enable_tiling()

    with torch.inference_mode():
        memory_latents, future_latents_packed = get_latents(
            vae,
            history,
            target_clip,
            device=device,
            dtype=dtype,
        )
        latent_height = history.shape[-2] // vae.spatial_compression_ratio
        latent_width = history.shape[-1] // vae.spatial_compression_ratio
        memory_latents = rearrange(
            memory_latents,
            "(b m) (h w) c -> b c m h w",
            b=1,
            m=history.shape[2],
            h=latent_height,
            w=latent_width,
        )
        future_latents = rearrange(
            future_latents_packed,
            "b (f h w) c -> b c f h w",
            b=1,
            h=latent_height,
            w=latent_width,
        )
        joined_latents = torch.cat([memory_latents, future_latents], dim=2)

        future_roundtrip = _decode_training_latents(vae, future_latents, dtype)
        joined_roundtrip = _decode_training_latents(vae, joined_latents, dtype)
        history_roundtrip = _decode_training_latents(
            vae, rearrange(memory_latents, "b c m h w -> (b m) c 1 h w"), dtype
        )
        history_roundtrip = rearrange(
            history_roundtrip, "(b m) c 1 h w -> b c m h w", b=1, m=4
        )

    frame_counts = {
        "source_history": _save_video_and_frames(history, output_dir / "source_history", 2),
        "source_target_clip": _save_video_and_frames(
            target_clip, output_dir / "source_target_clip", 2
        ),
        "history_single_frame_roundtrip": _save_video_and_frames(
            history_roundtrip, output_dir / "history_single_frame_roundtrip", 2
        ),
        "future_clip_roundtrip": _save_video_and_frames(
            future_roundtrip, output_dir / "future_clip_roundtrip", 2
        ),
        "joined_six_latent_roundtrip": _save_video_and_frames(
            joined_roundtrip, output_dir / "joined_six_latent_roundtrip", 8
        ),
    }
    metrics = {
        "token": token,
        "sample_index": sample_index,
        "history_shape": list(history.shape),
        "target_clip_shape": list(target_clip.shape),
        "memory_latent_shape": list(memory_latents.shape),
        "future_latent_shape": list(future_latents.shape),
        "joined_latent_shape": list(joined_latents.shape),
        "frame_counts": frame_counts,
        "history_roundtrip_mse": torch.mean(
            (history_roundtrip.float() - history.float()).square()
        ).item(),
        "future_roundtrip_mse": torch.mean(
            (future_roundtrip.float() - target_clip.float()).square()
        ).item(),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2, ensure_ascii=True), flush=True)
    print(f"Saved VAE oracle artifacts to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
