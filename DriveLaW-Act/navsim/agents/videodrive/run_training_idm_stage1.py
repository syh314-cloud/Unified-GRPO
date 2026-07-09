from __future__ import annotations

import argparse
import gzip
import json
import logging
from pathlib import Path
import pickle
import random
import sys
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, Dataset
from yaml import safe_load

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from navsim.agents.videodrive.idm_xvo_style import (
    IDMClipMeta,
    angle_wrap,
    build_resim_idm_for_drivelaw,
)

logger = logging.getLogger("idm_stage1")
DEFAULT_SPLIT_YAML = (
    Path(__file__).resolve().parents[2] / "planning/script/config/training/default_train_val_test_log_split.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stage-1 supervised training for DriveLaW IDM.")
    parser.add_argument("--cache-path", type=str, required=True, help="Path to videodrive cache directory.")
    parser.add_argument(
        "--split-yaml",
        type=str,
        default=str(DEFAULT_SPLIT_YAML),
        help="Optional YAML file containing `train_logs` and `val_logs` lists.",
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Directory for logs and checkpoints.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--image-height", type=int, default=768)
    parser.add_argument("--image-width", type=int, default=1344)
    parser.add_argument("--interval-length", type=float, default=0.5, help="Future trajectory interval in seconds.")
    parser.add_argument("--num-waypoints", type=int, default=8)
    parser.add_argument("--lambda-pos", type=float, default=1.0)
    parser.add_argument("--lambda-yaw", type=float, default=0.2)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--resume", type=str, default="", help="Checkpoint path or `auto`.")
    parser.add_argument("--keep-last", type=int, default=3, help="Keep only the latest N epoch checkpoints.")
    parser.add_argument("--save-every", type=int, default=1, help="Save checkpoint every N epochs.")
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="",
        help="Directory for TensorBoard event files. Defaults to <output-dir>/tensorboard.",
    )
    parser.add_argument("--freeze-backbone", action="store_true", default=True)
    parser.add_argument("--unfreeze-backbone", dest="freeze_backbone", action="store_false")
    parser.add_argument("--xvo-root", type=str, default=None)
    parser.add_argument("--flownet-root", type=str, default=None)
    parser.add_argument("--xvo-checkpoint-path", type=str, default=None)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_log_split(split_yaml: Optional[str]) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    if split_yaml is None:
        return None, None
    with open(split_yaml, "r", encoding="utf-8") as f:
        split_cfg = safe_load(f) or {}
    train_logs = split_cfg.get("train_logs")
    val_logs = split_cfg.get("val_logs")
    if train_logs is not None:
        train_logs = [str(item) for item in train_logs]
    if val_logs is not None:
        val_logs = [str(item) for item in val_logs]
    return train_logs, val_logs


def load_pickle_gz(path: Path) -> Dict[str, object]:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def load_image_as_tensor(path_like: str | Path, height: int, width: int) -> torch.Tensor:
    image = Image.open(path_like).convert("RGB")
    image = image.resize((width, height), resample=Image.BILINEAR)
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = arr * 2.0 - 1.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


class IDMStage1CacheDataset(Dataset):
    """Loads future frames and GT trajectories directly from trajectory_target cache files."""

    def __init__(
        self,
        cache_path: str,
        *,
        log_names: Optional[Sequence[str]],
        image_height: int,
        image_width: int,
        expected_steps: int,
    ) -> None:
        self.cache_path = Path(cache_path)
        if not self.cache_path.is_dir():
            raise FileNotFoundError(f"Cache path does not exist: {self.cache_path}")

        self.image_height = image_height
        self.image_width = image_width
        self.expected_steps = expected_steps
        self.target_file_name = "trajectory_target.gz"
        self.samples = self._collect_samples(log_names)
        if not self.samples:
            raise RuntimeError(f"No valid IDM stage-1 samples found under {self.cache_path}")

    def _collect_samples(self, log_names: Optional[Sequence[str]]) -> List[Path]:
        if log_names is None:
            candidate_logs = sorted(path for path in self.cache_path.iterdir() if path.is_dir())
        else:
            candidate_logs = [self.cache_path / log_name for log_name in log_names if (self.cache_path / log_name).is_dir()]

        sample_paths: List[Path] = []
        for log_path in candidate_logs:
            for token_path in sorted(path for path in log_path.iterdir() if path.is_dir()):
                target_path = token_path / self.target_file_name
                if target_path.is_file():
                    sample_paths.append(target_path)
        return sample_paths

    def __len__(self) -> int:
        return len(self.samples)

    def _load_future_frames(self, target_dict: Dict[str, object]) -> torch.Tensor:
        if "future_frames" in target_dict and isinstance(target_dict["future_frames"], torch.Tensor):
            frames = target_dict["future_frames"]
            if frames.ndim != 4:
                raise ValueError(f"Expected future_frames shaped (T,C,H,W), got {tuple(frames.shape)}")
            return F.interpolate(
                frames,
                size=(self.image_height, self.image_width),
                mode="bilinear",
                align_corners=False,
            )

        future_paths = target_dict.get("future_image_paths")
        if isinstance(future_paths, (list, tuple)) and future_paths:
            frames = [load_image_as_tensor(path, self.image_height, self.image_width) for path in future_paths]
            return torch.stack(frames, dim=0)

        raise KeyError("trajectory_target cache must contain `future_frames` or `future_image_paths`.")

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor | str]:
        target_path = self.samples[index]
        target_dict = load_pickle_gz(target_path)

        trajectory = target_dict["trajectory"]
        if not isinstance(trajectory, torch.Tensor):
            trajectory = torch.tensor(trajectory, dtype=torch.float32)
        trajectory = trajectory.float()

        future_video = self._load_future_frames(target_dict).float()

        if future_video.shape[0] != self.expected_steps:
            raise ValueError(
                f"Future video length must match expected_steps={self.expected_steps}, got {future_video.shape[0]} "
                f"for sample {target_path}"
            )
        if trajectory.shape[0] != self.expected_steps:
            raise ValueError(
                f"Trajectory length must match expected_steps={self.expected_steps}, got {trajectory.shape[0]} "
                f"for sample {target_path}"
            )
        if trajectory.shape[-1] != 3:
            raise ValueError(f"Expected trajectory shape (T,3), got {tuple(trajectory.shape)} for sample {target_path}")

        return {
            "future_video": future_video,
            "trajectory": trajectory,
            "token": target_path.parent.name,
        }


def idm_stage1_collate(batch: Sequence[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    future_video = torch.stack([sample["future_video"] for sample in batch], dim=0)
    trajectory = torch.stack([sample["trajectory"] for sample in batch], dim=0)
    tokens = [str(sample["token"]) for sample in batch]
    return {
        "future_video": future_video,
        "trajectory": trajectory,
        "tokens": tokens,
    }


def compute_idm_supervised_metrics(
    pred_waypoints: torch.Tensor,
    gt_waypoints: torch.Tensor,
    *,
    lambda_pos: float,
    lambda_yaw: float,
) -> Dict[str, torch.Tensor]:
    pos_error_steps = torch.norm(pred_waypoints[..., :2] - gt_waypoints[..., :2], dim=-1)
    yaw_error_steps = angle_wrap(pred_waypoints[..., 2] - gt_waypoints[..., 2]).abs()

    ade = pos_error_steps.mean()
    fde = pos_error_steps[:, -1].mean()
    yaw_mean = yaw_error_steps.mean()
    yaw_final = yaw_error_steps[:, -1].mean()
    total_loss = lambda_pos * ade + lambda_yaw * yaw_mean

    nearest_indices = torch.cdist(pred_waypoints[..., :2], gt_waypoints[..., :2]).argmin(dim=-1)
    diagonal = torch.arange(pred_waypoints.shape[1], device=pred_waypoints.device).unsqueeze(0)
    step_alignment_acc = (nearest_indices == diagonal).float().mean()

    return {
        "loss": total_loss,
        "ade": ade,
        "fde": fde,
        "yaw_mean": yaw_mean,
        "yaw_final": yaw_final,
        "step_alignment_acc": step_alignment_acc,
    }


def optimizer_param_groups(model: nn.Module) -> List[Dict[str, object]]:
    decay_params: List[nn.Parameter] = []
    no_decay_params: List[nn.Parameter] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(".bias"):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
    return [
        {"params": decay_params},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


def save_metrics_jsonl(path: Path, metrics: Dict[str, float]) -> None:
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(metrics, ensure_ascii=True) + "\n")


def build_tensorboard_writer(output_dir: Path, tensorboard_dir_arg: str, resume_global_step: int) -> SummaryWriter:
    tensorboard_dir = Path(tensorboard_dir_arg) if tensorboard_dir_arg else (output_dir / "tensorboard")
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tensorboard_dir), purge_step=resume_global_step)


def list_epoch_checkpoints(checkpoint_dir: Path) -> List[Path]:
    return sorted(checkpoint_dir.glob("epoch_*.pt"))


def prune_old_checkpoints(checkpoint_dir: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    epoch_ckpts = list_epoch_checkpoints(checkpoint_dir)
    if len(epoch_ckpts) <= keep_last:
        return
    for stale_path in epoch_ckpts[:-keep_last]:
        stale_path.unlink(missing_ok=True)


def resolve_resume_checkpoint(checkpoint_dir: Path, resume_arg: str) -> Optional[Path]:
    if not resume_arg:
        return None
    if resume_arg == "auto":
        epoch_ckpts = list_epoch_checkpoints(checkpoint_dir)
        return epoch_ckpts[-1] if epoch_ckpts else None
    path = Path(resume_arg)
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint not found: {path}")
    return path


def save_checkpoint(
    *,
    checkpoint_dir: Path,
    epoch: int,
    global_step: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    best_val_loss: float,
    args: argparse.Namespace,
    is_best: bool,
    keep_last: int,
) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / f"epoch_{epoch:04d}_step_{global_step:08d}.pt"
    state = {
        "epoch": epoch,
        "global_step": global_step,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "scaler_state_dict": None if scaler is None else scaler.state_dict(),
        "args": vars(args),
    }
    torch.save(state, ckpt_path)
    prune_old_checkpoints(checkpoint_dir, keep_last=keep_last)

    if is_best:
        best_path = checkpoint_dir / "best.pt"
        torch.save(state, best_path)

    return ckpt_path


def load_checkpoint(
    *,
    checkpoint_path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    scaler: Optional[torch.amp.GradScaler],
    device: torch.device,
) -> Tuple[int, int, float]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler_state = checkpoint.get("scheduler_state_dict")
    if scheduler is not None and scheduler_state is not None:
        scheduler.load_state_dict(scheduler_state)
    scaler_state = checkpoint.get("scaler_state_dict")
    if scaler is not None and scaler_state is not None:
        scaler.load_state_dict(scaler_state)
    start_epoch = int(checkpoint["epoch"]) + 1
    global_step = int(checkpoint["global_step"])
    best_val_loss = float(checkpoint.get("best_val_loss", float("inf")))
    logger.info("Resumed IDM stage-1 training from %s", checkpoint_path)
    return start_epoch, global_step, best_val_loss


def move_batch_to_device(batch: Dict[str, torch.Tensor | List[str]], device: torch.device) -> Dict[str, torch.Tensor | List[str]]:
    out: Dict[str, torch.Tensor | List[str]] = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            out[key] = value.to(device, non_blocking=True)
        else:
            out[key] = value
    return out


def evaluate(
    *,
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    fps: float,
    lambda_pos: float,
    lambda_yaw: float,
    amp_enabled: bool,
) -> Dict[str, float]:
    model.eval()
    metric_sums = {
        "loss": 0.0,
        "ade": 0.0,
        "fde": 0.0,
        "yaw_mean": 0.0,
        "yaw_final": 0.0,
        "step_alignment_acc": 0.0,
    }
    total_samples = 0

    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"
    with torch.inference_mode():
        for batch in dataloader:
            batch = move_batch_to_device(batch, device)
            future_video = batch["future_video"]
            gt_waypoints = batch["trajectory"]
            meta = IDMClipMeta(history_frames=0, future_frames=future_video.shape[1], fps=fps)

            with torch.amp.autocast(device_type=autocast_device, enabled=amp_enabled):
                pred_waypoints = model(future_video, meta=meta)
                metrics = compute_idm_supervised_metrics(
                    pred_waypoints,
                    gt_waypoints,
                    lambda_pos=lambda_pos,
                    lambda_yaw=lambda_yaw,
                )

            batch_size = future_video.shape[0]
            total_samples += batch_size
            for key in metric_sums:
                metric_sums[key] += float(metrics[key].detach().cpu()) * batch_size

    if total_samples == 0:
        raise RuntimeError("Validation dataloader is empty.")

    return {key: value / total_samples for key, value in metric_sums.items()}


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    metrics_path = output_dir / "metrics.jsonl"

    setup_logging(output_dir)
    set_seed(args.seed)

    device = torch.device(args.device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train_logs, val_logs = load_log_split(args.split_yaml)

    logger.info("Building IDM stage-1 datasets from cache: %s", args.cache_path)
    train_dataset = IDMStage1CacheDataset(
        args.cache_path,
        log_names=train_logs,
        image_height=args.image_height,
        image_width=args.image_width,
        expected_steps=args.num_waypoints,
    )
    val_dataset = IDMStage1CacheDataset(
        args.cache_path,
        log_names=val_logs,
        image_height=args.image_height,
        image_width=args.image_width,
        expected_steps=args.num_waypoints,
    )
    logger.info("Train samples: %d | Val samples: %d", len(train_dataset), len(val_dataset))

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=idm_stage1_collate,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=idm_stage1_collate,
        drop_last=False,
    )

    model = build_resim_idm_for_drivelaw(
        num_waypoints=args.num_waypoints,
        xvo_root=args.xvo_root,
        flownet_root=args.flownet_root,
        xvo_checkpoint_path=args.xvo_checkpoint_path,
        freeze_backbone=args.freeze_backbone,
    ).to(device)
    if args.freeze_backbone:
        logger.info("Stage-1 training freezes the XVO backbone.")
    else:
        logger.info("Stage-1 training will update the XVO backbone.")

    optimizer = torch.optim.AdamW(
        optimizer_param_groups(model),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp and device.type == "cuda"))

    start_epoch = 1
    global_step = 0
    best_val_loss = float("inf")
    resume_ckpt = resolve_resume_checkpoint(checkpoint_dir, args.resume)
    if resume_ckpt is not None:
        start_epoch, global_step, best_val_loss = load_checkpoint(
            checkpoint_path=resume_ckpt,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )

    writer = build_tensorboard_writer(output_dir, args.tensorboard_dir, global_step)

    fps = 1.0 / float(args.interval_length)
    autocast_device = device.type if device.type in {"cuda", "cpu"} else "cuda"

    try:
        for epoch in range(start_epoch, args.epochs + 1):
            model.train()
            running = {
                "loss": 0.0,
                "ade": 0.0,
                "fde": 0.0,
                "yaw_mean": 0.0,
                "yaw_final": 0.0,
                "step_alignment_acc": 0.0,
            }
            seen_samples = 0

            for batch_index, batch in enumerate(train_loader, start=1):
                batch = move_batch_to_device(batch, device)
                future_video = batch["future_video"]
                gt_waypoints = batch["trajectory"]
                meta = IDMClipMeta(history_frames=0, future_frames=future_video.shape[1], fps=fps)

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(device_type=autocast_device, enabled=(args.amp and device.type == "cuda")):
                    pred_waypoints = model(future_video, meta=meta)
                    metrics = compute_idm_supervised_metrics(
                        pred_waypoints,
                        gt_waypoints,
                        lambda_pos=args.lambda_pos,
                        lambda_yaw=args.lambda_yaw,
                    )

                scaler.scale(metrics["loss"]).backward()
                scaler.unscale_(optimizer)
                if args.grad_clip_norm > 0.0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()

                batch_size = future_video.shape[0]
                seen_samples += batch_size
                global_step += 1
                for key in running:
                    running[key] += float(metrics[key].detach().cpu()) * batch_size

                writer.add_scalar("lr/step", optimizer.param_groups[0]["lr"], global_step)
                for key, value in metrics.items():
                    writer.add_scalar(f"train_step/{key}", float(value.detach().cpu()), global_step)

                if batch_index % args.log_every == 0:
                    logger.info(
                        "Epoch %d Step %d | loss=%.4f ade=%.4f fde=%.4f yaw=%.4f align=%.4f",
                        epoch,
                        global_step,
                        running["loss"] / seen_samples,
                        running["ade"] / seen_samples,
                        running["fde"] / seen_samples,
                        running["yaw_mean"] / seen_samples,
                        running["step_alignment_acc"] / seen_samples,
                    )

            scheduler.step()
            train_metrics = {f"train/{key}": value / seen_samples for key, value in running.items()}
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                device=device,
                fps=fps,
                lambda_pos=args.lambda_pos,
                lambda_yaw=args.lambda_yaw,
                amp_enabled=(args.amp and device.type == "cuda"),
            )
            val_metrics = {f"val/{key}": value for key, value in val_metrics.items()}

            current_val_loss = val_metrics["val/loss"]
            is_best = current_val_loss < best_val_loss
            if is_best:
                best_val_loss = current_val_loss

            epoch_summary = {
                "epoch": epoch,
                "global_step": global_step,
                "lr": optimizer.param_groups[0]["lr"],
                "best_val_loss": best_val_loss,
            }
            epoch_summary.update(train_metrics)
            epoch_summary.update(val_metrics)
            save_metrics_jsonl(metrics_path, epoch_summary)

            writer.add_scalar("lr/epoch", optimizer.param_groups[0]["lr"], epoch)
            writer.add_scalar("best/val_loss", best_val_loss, epoch)
            for key, value in train_metrics.items():
                writer.add_scalar(key, value, epoch)
            for key, value in val_metrics.items():
                writer.add_scalar(key, value, epoch)
            writer.flush()

            logger.info(
                "Epoch %d done | train_loss=%.4f val_loss=%.4f val_ADE=%.4f val_FDE=%.4f val_align=%.4f",
                epoch,
                train_metrics["train/loss"],
                val_metrics["val/loss"],
                val_metrics["val/ade"],
                val_metrics["val/fde"],
                val_metrics["val/step_alignment_acc"],
            )

            if epoch % args.save_every == 0:
                ckpt_path = save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    epoch=epoch,
                    global_step=global_step,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    scaler=scaler,
                    best_val_loss=best_val_loss,
                    args=args,
                    is_best=is_best,
                    keep_last=args.keep_last,
                )
                logger.info("Saved checkpoint: %s", ckpt_path)

        logger.info("Finished IDM stage-1 training. Best val loss: %.6f", best_val_loss)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
