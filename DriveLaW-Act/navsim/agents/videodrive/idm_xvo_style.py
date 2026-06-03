from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
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
    """Map common DriveLaW video ranges to the [0, 1] range expected by XVO."""
    clip = clip.float()
    if clip.numel() == 0:
        return clip
    if clip.min() < 0:
        clip = (clip + 1.0) * 0.5
    return clip.clamp_(0.0, 1.0)


def build_drivelaw_idm_clip(
    history_video: torch.Tensor,
    future_video: torch.Tensor,
) -> Tuple[torch.Tensor, IDMClipMeta]:
    """
    Build the semantic IDM input clip for the current DriveLaW training path.

    Inputs are the current DriveLaW tensors:
      - history_video: (B, C, 1, T_hist, H, W) or squeezed variants
      - future_video : (B, C, 1, T_fut,  H, W) or squeezed variants

    Returns:
      clip: (B, T_total, C, H, W) using only real frames
      meta: semantic frame counts and effective fps
    """
    if history_video.ndim == 6:
        history_video = history_video.squeeze(2)
    if future_video.ndim == 6:
        future_video = future_video.squeeze(2)

    hist = _to_bthwc_video(history_video)
    fut = _to_bthwc_video(future_video)

    if hist.shape[0] != fut.shape[0] or hist.shape[2:] != fut.shape[2:]:
        raise ValueError(
            "History/future videos must share batch size and frame shape, "
            f"got {tuple(hist.shape)} and {tuple(fut.shape)}"
        )

    clip = torch.cat([hist, fut], dim=1)
    meta = IDMClipMeta(
        history_frames=hist.shape[1],
        future_frames=fut.shape[1],
        fps=2.0,
    )
    return clip, meta


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
            VOModel = vomodel_mod.VOModel
            model = VOModel()

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
        pair_tokens = self.transformer(pair_feats)["feature"]
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

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        tokens = self.backbone(clip)
        tokens = self.temporal_encoder(tokens)

        queries = self.waypoint_queries.expand(tokens.shape[0], -1, -1)
        decoded, _ = self.attn_head(queries, tokens, tokens, need_weights=False)
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
