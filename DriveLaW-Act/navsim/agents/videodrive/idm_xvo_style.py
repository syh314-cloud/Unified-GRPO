from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


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

    # Heuristic: DriveLaW commonly uses (B, C, T, H, W) with C in {3, 6}.
    if video.shape[1] in (3, 6):
        return video.permute(0, 2, 1, 3, 4).contiguous()
    return video.contiguous()


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

    Notes:
      - In the current DriveLaW Act path, history/future supervision is sampled at 2Hz.
      - The training code later repeats the last future frame to satisfy latent padding, but the
        semantic future supervision itself is still 8 real frames.
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


class XVOPairBackbone(nn.Module):
    """
    XVO-style pair encoder adapted for a full video clip.

    Original XVO consumes a pair of adjacent RGB frames concatenated along channels
    (6-channel image). This module applies that idea to every adjacent pair in a clip:

      (f0, f1), (f1, f2), ... , (f_{T-2}, f_{T-1})

    Each pair is resized and encoded independently, yielding one token per pair.
    """

    def __init__(
        self,
        image_size: Tuple[int, int] = (224, 224),
        embed_dim: int = 384,
        backbone_name: str = "vit_small_patch16_224",
    ) -> None:
        super().__init__()
        self.image_size = image_size
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=False,
            in_chans=6,
            num_classes=0,
        )
        self.proj = nn.Linear(self.backbone.num_features, embed_dim)

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
          clip: (B, T, C, H, W)

        Returns:
          pair_tokens: (B, T-1, D)
        """
        clip = _to_bthwc_video(clip)
        bsz, timesteps, channels, _, _ = clip.shape
        if channels != 3:
            raise ValueError(f"Expected RGB clip with 3 channels, got {channels}")
        if timesteps < 2:
            raise ValueError(f"Need at least 2 frames, got {timesteps}")

        pair_inputs = torch.cat([clip[:, :-1], clip[:, 1:]], dim=2)  # (B, T-1, 6, H, W)
        pair_inputs = pair_inputs.flatten(0, 1)  # (B*(T-1), 6, H, W)
        pair_inputs = F.interpolate(
            pair_inputs,
            size=self.image_size,
            mode="bilinear",
            align_corners=False,
        )

        pair_feats = self.backbone.forward_features(pair_inputs)
        if pair_feats.ndim == 3:
            pair_feats = pair_feats.mean(dim=1)

        pair_tokens = self.proj(pair_feats).view(bsz, timesteps - 1, -1)
        return pair_tokens


class ReSimStyleIDM(nn.Module):
    """
    ReSim-style inverse dynamics module for DriveLaW.

    Design:
      1. Use an XVO-style adjacent-pair visual backbone to encode the clip.
      2. Aggregate temporal motion tokens.
      3. Use a lightweight attention head to decode all future waypoints at once.

    This follows the ReSim paper at the interface level:
      full video clip -> XVO-style backbone -> attention head -> 8 future waypoints
    """

    def __init__(
        self,
        num_waypoints: int = 8,
        waypoint_dim: int = 3,
        embed_dim: int = 384,
        num_temporal_layers: int = 2,
        num_heads: int = 6,
        image_size: Tuple[int, int] = (224, 224),
    ) -> None:
        super().__init__()
        self.num_waypoints = num_waypoints
        self.waypoint_dim = waypoint_dim

        self.backbone = XVOPairBackbone(
            image_size=image_size,
            embed_dim=embed_dim,
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

    def forward(self, clip: torch.Tensor) -> torch.Tensor:
        """
        Args:
          clip: (B, T, C, H, W) or (B, C, T, H, W)

        Returns:
          waypoints: (B, num_waypoints, waypoint_dim)
        """
        tokens = self.backbone(clip)
        tokens = self.temporal_encoder(tokens)

        queries = self.waypoint_queries.expand(tokens.shape[0], -1, -1)
        decoded, _ = self.attn_head(queries, tokens, tokens, need_weights=False)
        decoded = self.head_norm(decoded)
        return self.head_mlp(decoded)


def build_resim_idm_for_drivelaw(
    num_waypoints: int = 8,
    waypoint_dim: int = 3,
) -> ReSimStyleIDM:
    """
    Convenience factory aligned with the current DriveLaW action target format.

    Current DriveLaW trajectory targets are 8 future poses with 3 values each:
      [x, y, heading]
    """
    return ReSimStyleIDM(
        num_waypoints=num_waypoints,
        waypoint_dim=waypoint_dim,
    )
