from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
import os
from pathlib import Path
import sys
from typing import Optional, Tuple
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IDMClipMeta:
    """Metadata describing the semantic clip layout used by the IDM."""

    history_frames: int
    future_frames: int
    fps: float


def _to_bthwc_video(video: torch.Tensor) -> torch.Tensor:
    """
    Normalize common video layouts to (B, T, C, H, W).

    Supported layouts:
      - (B, C, T, H, W)
      - (B, T, C, H, W)
      - (T, C, H, W) -> assumed B=1
    """
    if video.ndim == 4:
        video = video.unsqueeze(0)

    if video.ndim != 5:
        raise ValueError(f"Expected a 4D/5D video tensor, got shape {tuple(video.shape)}")

    if video.shape[1] in (3, 6):
        return video.permute(0, 2, 1, 3, 4).contiguous()
    return video.contiguous()


def _normalize_rgb_clip(clip: torch.Tensor) -> torch.Tensor:
    """XVO 的 ViT 需要0~1的 RGB 输入"""
    clip = clip.float()
    if clip.numel() == 0:
        return clip
    if clip.min() < 0:
        clip = (clip + 1.0) * 0.5
    return clip.clamp_(0.0, 1.0)


def build_drivelaw_idm_clip(
    history_video: torch.Tensor,
    future_video: torch.Tensor,
    future_interval_s: float = 0.5,
) -> Tuple[torch.Tensor, IDMClipMeta]:
    """
    Build the semantic IDM input clip for the current DriveLaW training path.

    Inputs are the current DriveLaW tensors:
      - history_video: (B, C, 1, T_hist, H, W) or squeezed variants
      - future_video : (B, C, 1, T_fut,  H, W) or squeezed variants

    Returns:
      clip: (B, 1 + T_fut, C, H, W), ordered as [f0, f1, ..., f8]
      meta: semantic frame counts and effective fps
    """
    if history_video.ndim == 6:
        history_video = history_video.squeeze(2)
    if future_video.ndim == 6:
        future_video = future_video.squeeze(2)

    fut = _to_bthwc_video(future_video)
    dt = float(future_interval_s)
    if dt <= 0.0:
        raise ValueError(f"future_interval_s must be positive, got {dt}")
    hist = _to_bthwc_video(history_video)
    if hist.shape[1] < 1:
        raise ValueError("IDM requires one current frame f0 in history_video.")
    current = hist[:, -1:]
    clip = torch.cat([current, fut], dim=1)
    meta = IDMClipMeta(
        history_frames=1,
        future_frames=fut.shape[1],
        fps=1.0 / dt,
    )
    return clip, meta


def angle_wrap(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi] for stable yaw comparisons."""
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _build_sinusoidal_time_embedding(
    *,
    num_steps: int,
    fps: float,
    dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build fixed time embeddings for future queries q_k at k / fps seconds."""
    if num_steps <= 0:
        raise ValueError(f"num_steps must be positive, got {num_steps}")
    if fps <= 0.0:
        raise ValueError(f"fps must be positive, got {fps}")
    if dim <= 0:
        raise ValueError(f"dim must be positive, got {dim}")

    times = torch.arange(1, num_steps + 1, device=device, dtype=torch.float32) / float(fps) # [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
    half_dim = dim // 2
    if half_dim == 0:
        return times.unsqueeze(-1).to(dtype=dtype)

    freq_idx = torch.arange(half_dim, device=device, dtype=torch.float32)
    freq_divisor = max(half_dim - 1, 1)
    freq = torch.exp(-math.log(10000.0) * freq_idx / freq_divisor)
    phase = times.unsqueeze(-1) * freq.unsqueeze(0)
    embedding = torch.cat([torch.sin(phase), torch.cos(phase)], dim=-1)
    if embedding.shape[-1] < dim:
        embedding = F.pad(embedding, (0, dim - embedding.shape[-1]))
    return embedding.to(dtype=dtype)


def compute_idm_consistency_reward(
    pred_waypoints: torch.Tensor,
    sampled_waypoints: torch.Tensor,
    lambda_pos: float = 1.0,
    lambda_yaw: float = 0.2,
    time_weights: Optional[torch.Tensor] = None,
    reduction: str = "none",
) -> dict:
    """
    Compare IDM-predicted 8-point trajectories against sampled action trajectories.

    Args:
      pred_waypoints:   (B, T, 3)
      sampled_waypoints:(B, T, 3)
      lambda_pos:       position error weight
      lambda_yaw:       yaw error weight
      time_weights:     optional (T,) weights over the waypoint horizon
      reduction:        one of {"none", "mean"}

    Returns:
      dict with:
        - reward:   (B,) or scalar
        - pos_error
        - yaw_error
        - total_cost
    """
    if pred_waypoints.shape != sampled_waypoints.shape:
        raise ValueError(
            "Predicted/sample trajectories must have identical shape, "
            f"got {tuple(pred_waypoints.shape)} and {tuple(sampled_waypoints.shape)}"
        )
    if pred_waypoints.ndim != 3 or pred_waypoints.shape[-1] != 3:
        raise ValueError(
            f"Expected trajectories shaped (B, T, 3), got {tuple(pred_waypoints.shape)}"
        )

    pred_waypoints = pred_waypoints.float()
    sampled_waypoints = sampled_waypoints.float()

    pos_error_steps = torch.norm(
        pred_waypoints[..., :2] - sampled_waypoints[..., :2],
        dim=-1,
    )
    yaw_error_steps = angle_wrap(
        pred_waypoints[..., 2] - sampled_waypoints[..., 2]
    ).abs()

    if time_weights is not None:
        if time_weights.ndim != 1 or time_weights.shape[0] != pred_waypoints.shape[1]:
            raise ValueError(
                "time_weights must have shape (T,), "
                f"got {tuple(time_weights.shape)} for T={pred_waypoints.shape[1]}"
            )
        weights = time_weights.to(pred_waypoints.device, pred_waypoints.dtype)
        weights = weights / weights.sum().clamp_min(1e-8)
        pos_error = (pos_error_steps * weights.unsqueeze(0)).sum(dim=-1)
        yaw_error = (yaw_error_steps * weights.unsqueeze(0)).sum(dim=-1)
    else:
        pos_error = pos_error_steps.mean(dim=-1)
        yaw_error = yaw_error_steps.mean(dim=-1)

    total_cost = lambda_pos * pos_error + lambda_yaw * yaw_error
    reward = -total_cost

    if reduction == "mean":
        reward = reward.mean()
        pos_error = pos_error.mean()
        yaw_error = yaw_error.mean()
        total_cost = total_cost.mean()
    elif reduction != "none":
        raise ValueError(f"Unsupported reduction `{reduction}`")

    return {
        "reward": reward,
        "pos_error": pos_error,
        "yaw_error": yaw_error,
        "total_cost": total_cost,
    }


def _normalize_frame_for_iqa(frame: torch.Tensor) -> torch.Tensor:
    """Convert a frame to BCHW float tensor in [0, 1] for IQA models."""
    if frame.ndim == 3:
        frame = frame.unsqueeze(0)
    if frame.ndim != 4:
        raise ValueError(f"Expected frame tensor shaped (C,H,W) or (B,C,H,W), got {tuple(frame.shape)}")
    frame = frame.float()
    if frame.min() < 0:
        frame = (frame + 1.0) * 0.5
    return frame.clamp_(0.0, 1.0)


def compute_musiq_reward(
    future_video: torch.Tensor,
    musiq_metric: nn.Module,
    reduction: str = "none",
) -> dict:
    """
    Reward 3: score only the middle frame and final frame with MUSIQ, then average.

    Args:
      future_video:  (B, T, C, H, W) or (B, C, T, H, W)
      musiq_metric:  PyIQA MUSIQ module
      reduction:     {"none", "mean"}
    """
    clip = _to_bthwc_video(future_video)
    if clip.ndim != 5 or clip.shape[1] < 1:
        raise ValueError(f"Expected future video shaped (B,T,C,H,W) with T>=1, got {tuple(clip.shape)}")

    mid_index = clip.shape[1] // 2
    last_index = clip.shape[1] - 1

    mid_frame = _normalize_frame_for_iqa(clip[:, mid_index])
    last_frame = _normalize_frame_for_iqa(clip[:, last_index])

    mid_score = musiq_metric(mid_frame).reshape(-1)
    last_score = musiq_metric(last_frame).reshape(-1)
    reward = 0.5 * (mid_score + last_score)

    if reduction == "mean":
        reward = reward.mean()
        mid_score = mid_score.mean()
        last_score = last_score.mean()
    elif reduction != "none":
        raise ValueError(f"Unsupported reduction `{reduction}`")

    return {
        "reward": reward,
        "mid_score": mid_score,
        "last_score": last_score,
    }


def _resolve_optional_path(path_like: Optional[str]) -> Optional[Path]:
    if not path_like:
        return None
    return Path(path_like).expanduser().resolve()


@contextmanager
def _prepend_python_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def _load_checkpoint_state(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
    return checkpoint


class ExternalXVOPairBackbone(nn.Module):
    """
    XVO pair encoder that reuses the original XVO backbone as directly as possible.

    Required external assets:
      - XVO repo root (`XVO_ROOT` or `xvo_root`)
      - MaskFlownet pretrained directory containing:
          * `kitti.yaml`
          * `8caNov12-1532_300000.pth`
        (`XVO_FLOWNET_ROOT` or `flownet_root`)

    Optional:
      - Full XVO checkpoint (`XVO_CHECKPOINT_PATH` or `xvo_checkpoint_path`)
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (384, 640),
        embed_dim: int = 256,
        xvo_root: Optional[str] = None,
        flownet_root: Optional[str] = None,
        xvo_checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.xvo_root = _resolve_optional_path(xvo_root or os.getenv("XVO_ROOT"))
        self.flownet_root = _resolve_optional_path(flownet_root or os.getenv("XVO_FLOWNET_ROOT"))
        self.xvo_checkpoint_path = _resolve_optional_path(
            xvo_checkpoint_path or os.getenv("XVO_CHECKPOINT_PATH")
        )

        if self.xvo_root is None or not self.xvo_root.exists():
            raise FileNotFoundError(
                "XVO repo root not found. Set `xvo_root` or the `XVO_ROOT` environment variable."
            )
        if self.flownet_root is None or not self.flownet_root.exists():
            raise FileNotFoundError(
                "MaskFlownet pretrained directory not found. "
                "Set `flownet_root` or the `XVO_FLOWNET_ROOT` environment variable."
            )
        if not (self.flownet_root / "kitti.yaml").exists():
            raise FileNotFoundError(
                f"Expected `{self.flownet_root / 'kitti.yaml'}` for XVO FlowNet initialization."
            )

        self.encoder, self.transformer, self.token_dim = self._build_external_backbone()
        self.proj = nn.Identity() if self.token_dim == embed_dim else nn.Linear(self.token_dim, embed_dim)

        if freeze_backbone:
            self.freeze_backbone()

    def _build_external_backbone(self) -> Tuple[nn.Module, nn.Module, int]:
        with _prepend_python_path(self.xvo_root):
            params_mod = importlib.import_module("params")
            params_mod.par.pretrained_flownet_path = str(self.flownet_root)

            vomodel_mod = importlib.import_module("model.vomodel")
            # XVO's VOModel wraps this ViT with torchvision's FX feature extractor
            # only to return the final norm tokens. Recent timm versions expose
            # `is_causal`, which torchvision 0.15 traces as a Proxy and cannot pass
            # to scaled_dot_product_attention. Build the two XVO components we use
            # directly, so their checkpoint keys remain encoder.* / transformer.*.
            model = nn.Module()
            model.encoder = vomodel_mod.FlowNet()
            transformer_params = dict(
                img_size=(96, 160),
                patch_size=(12, 16),
                embed_dim=256,
                depth=4,
                num_heads=4,
                num_classes=0,
                in_chans=102,
                class_token=False,
                global_pool="",
            )
            model.transformer = vomodel_mod._create_vision_transformer(
                "vit_base_patch16_224_in21k",
                pretrained=False,
                **transformer_params,
            )

        self._load_pretrained_weights(model)
        return model.encoder, model.transformer, 256

    def _load_pretrained_weights(self, model: nn.Module) -> None:
        model_dict = model.state_dict()
        merged_state = dict(model_dict)

        flownet_ckpt_path = self.flownet_root / "8caNov12-1532_300000.pth"
        if flownet_ckpt_path.exists():
            flownet_state = _load_checkpoint_state(flownet_ckpt_path)
            flowed = {
                f"encoder.maskflownet.{key}": value
                for key, value in flownet_state.items()
                if f"encoder.maskflownet.{key}" in model_dict
                and model_dict[f"encoder.maskflownet.{key}"].shape == value.shape
            }
            merged_state.update(flowed)
        else:
            warnings.warn(
                f"MaskFlownet checkpoint `{flownet_ckpt_path}` was not found. "
                "The external XVO backbone will initialize without optical-flow pretraining."
            )

        if self.xvo_checkpoint_path is not None and self.xvo_checkpoint_path.exists():
            xvo_state = _load_checkpoint_state(self.xvo_checkpoint_path)
            cleaned_state = {}
            for key, value in xvo_state.items():
                clean_key = key[7:] if key.startswith("module.") else key
                if clean_key in model_dict and model_dict[clean_key].shape == value.shape:
                    cleaned_state[clean_key] = value
            merged_state.update(cleaned_state)

        model.load_state_dict(merged_state, strict=False)

    def freeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for param in self.encoder.parameters():
            param.requires_grad = True
        for param in self.transformer.parameters():
            param.requires_grad = True

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        clip = _normalize_rgb_clip(_to_bthwc_video(clip))
        bsz, timesteps, channels, _, _ = clip.shape
        if channels != 3:
            raise ValueError(f"Expected RGB clip with 3 channels, got {channels}")
        if timesteps < 2:
            raise ValueError(f"Need at least 2 frames, got {timesteps}")

        pair_inputs = torch.cat([clip[:, :-1], clip[:, 1:]], dim=2).flatten(0, 1)
        pair_inputs = F.interpolate(
            pair_inputs,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        pair_feats = self.encoder(pair_inputs)
        pair_tokens = self.transformer.forward_features(pair_feats)
        if pair_tokens.ndim == 3:
            pair_tokens = pair_tokens.mean(dim=1)

        return self.proj(pair_tokens).view(bsz, timesteps - 1, -1)


class ReSimStyleIDM(nn.Module):
    """
    ReSim-style inverse dynamics module for DriveLaW.

    The intended configuration is:
      full clip -> original XVO backbone -> temporal attention head -> 8x3 waypoints

    This implementation requires the original XVO backbone and its pretrained
    weights; there is no fallback encoder.
    """

    def __init__(
        self,
        num_waypoints: int = 8,
        waypoint_dim: int = 3,
        embed_dim: int = 256,
        num_temporal_layers: int = 2,
        num_heads: int = 4,
        pair_image_size: Tuple[int, int] = (384, 640),
        xvo_root: Optional[str] = None,
        flownet_root: Optional[str] = None,
        xvo_checkpoint_path: Optional[str] = None,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")

        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim

        self.backbone = ExternalXVOPairBackbone(
            image_size=pair_image_size,
            embed_dim=embed_dim,
            xvo_root=xvo_root,
            flownet_root=flownet_root,
            xvo_checkpoint_path=xvo_checkpoint_path,
            freeze_backbone=freeze_backbone,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_temporal_layers,
        )

        self.waypoint_queries = nn.Parameter(torch.randn(1, num_waypoints, embed_dim) * 0.02)
        self.query_time_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.attn_head = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True,
        )
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, waypoint_dim),
        )

    def freeze_backbone(self) -> None:
        self.backbone.freeze_backbone()

    def unfreeze_backbone(self) -> None:
        self.backbone.unfreeze_backbone()

    def _build_time_aware_queries(
        self,
        *,
        batch_size: int,
        fps: float,
        future_frames: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if future_frames != self.num_waypoints:
            raise ValueError(
                "IDM future frame count must match configured waypoint count, "
                f"got {future_frames} vs {self.num_waypoints}"
            )

        base_queries = self.waypoint_queries.expand(batch_size, -1, -1)
        time_embedding = _build_sinusoidal_time_embedding(
            num_steps=future_frames,
            fps=fps,
            dim=base_queries.shape[-1],
            device=device,
            dtype=dtype,
        )
        return base_queries + self.query_time_proj(time_embedding).unsqueeze(0)

    def forward(self, clip: torch.Tensor, meta: Optional[IDMClipMeta] = None) -> torch.Tensor:
        tokens = self.backbone(clip)
        tokens = self.temporal_encoder(tokens)

        future_frames = self.num_waypoints if meta is None else meta.future_frames
        fps = 2.0 if meta is None else meta.fps
        if tokens.shape[1] != future_frames:
            raise ValueError(
                "IDM requires one current frame plus all future frames so the XVO pair "
                f"tokens are 1:1 with waypoints; got {tokens.shape[1]} tokens for "
                f"{future_frames} future frames."
            )
        queries = self._build_time_aware_queries(
            batch_size=tokens.shape[0],
            fps=fps,
            future_frames=future_frames,
            device=tokens.device,
            dtype=tokens.dtype,
        )
        decoded, _ = self.attn_head(
            queries,
            tokens,
            tokens,
            need_weights=False,
        )
        decoded = self.head_norm(decoded)
        return self.head_mlp(decoded)


def build_resim_idm_for_drivelaw(
    num_waypoints: int = 8,
    waypoint_dim: int = 3,
    xvo_root: Optional[str] = None,
    flownet_root: Optional[str] = None,
    xvo_checkpoint_path: Optional[str] = None,
    freeze_backbone: bool = True,
) -> ReSimStyleIDM:
    """
    Factory aligned with the current DriveLaW target format.

    This function requires XVO pretrained assets. Set either function arguments or:
      - XVO_ROOT
      - XVO_FLOWNET_ROOT
      - XVO_CHECKPOINT_PATH  (optional)
    """
    return ReSimStyleIDM(
        num_waypoints=num_waypoints,
        waypoint_dim=waypoint_dim,
        xvo_root=xvo_root,
        flownet_root=flownet_root,
        xvo_checkpoint_path=xvo_checkpoint_path,
        freeze_backbone=freeze_backbone,
    )
