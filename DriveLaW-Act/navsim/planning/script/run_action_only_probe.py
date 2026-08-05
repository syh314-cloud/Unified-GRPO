"""Compare official Action-only inference with joint video/action inference."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

from navsim.common.dataloader import SceneLoader


def _build_features(agent: Any, agent_input: Any) -> Dict[str, torch.Tensor]:
    features: Dict[str, torch.Tensor] = {}
    for builder in agent.get_feature_builders():
        features.update(builder.compute_features(agent_input))
    return {key: value.unsqueeze(0) for key, value in features.items()}


def _metrics(prediction: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    position_error = torch.linalg.vector_norm(prediction[:, :2] - target[:, :2], dim=-1)
    heading_error = (prediction[:, 2] - target[:, 2]).abs()
    return {
        "position_ade": float(position_error.mean()),
        "position_fde": float(position_error[-1]),
        "heading_mae": float(heading_error.mean()),
        "heading_final_abs_error": float(heading_error[-1]),
    }


def _mean_metrics(rows: list[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    return {
        key: sum(row[key] for row in rows) / len(rows)
        for key in rows[0]
    }


def run_action_only_probe(cfg: DictConfig) -> None:
    probe = cfg.probe
    joint_output_dir = Path(str(probe.output_dir))
    output_dir = Path(
        os.environ.get("ACTION_ONLY_OUTPUT_DIR", f"{joint_output_dir}_action_only")
    )
    trajectory_dir = output_dir / "trajectories"
    trajectory_dir.mkdir(parents=True, exist_ok=True)

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

    tokens = sorted(scene_loader.tokens)
    limit = int(probe.limit)
    if limit > 0:
        tokens = tokens[:limit]

    action_only_gt_metrics = []
    joint_gt_metrics = []
    action_only_joint_metrics = []
    manifest: Dict[str, Any] = {
        "mode": "action_only",
        "num_samples": len(tokens),
        "num_inference_steps": int(agent.args.num_inference_step),
        "world_model_config": str(cfg.agent.config_file),
        "joint_trajectory_dir": str(joint_output_dir / "trajectories"),
        "tokens": [],
    }

    torch.set_printoptions(precision=4, sci_mode=False, linewidth=160)
    with torch.inference_mode():
        for index, token in enumerate(tokens):
            print(f"[{index + 1}/{len(tokens)}] {token}", flush=True)
            agent_input = scene_loader.get_agent_input_from_token(token)
            scene = scene_loader.get_scene_from_token(token)
            features = _build_features(agent, agent_input)
            predictions = agent.sample_action_rollouts(features, sample_time=1)
            action_only = predictions["actions_tensor"][0]

            gt_target = agent.get_target_builders()[0].compute_targets(scene)
            gt = gt_target["trajectory"]
            if not torch.is_tensor(gt):
                gt = torch.as_tensor(gt)
            gt = gt.detach().float().cpu()

            stem = f"{index:04d}_{token}"
            joint_path = joint_output_dir / "trajectories" / f"{stem}_trajectories.pt"
            if not joint_path.is_file():
                raise FileNotFoundError(f"Joint trajectory not found: {joint_path}")
            joint_data = torch.load(joint_path, map_location="cpu")
            joint = joint_data["world_model_trajectory"].detach().float().cpu()

            action_only_vs_gt = _metrics(action_only, gt)
            joint_vs_gt = _metrics(joint, gt)
            action_only_vs_joint = _metrics(action_only, joint)
            action_only_gt_metrics.append(action_only_vs_gt)
            joint_gt_metrics.append(joint_vs_gt)
            action_only_joint_metrics.append(action_only_vs_joint)

            print("\nAction-only World Model trajectory:")
            print(action_only)
            print("\nJoint World Model trajectory:")
            print(joint)
            print("\nGT trajectory:")
            print(gt)
            print("\nMetrics:")
            print("  Action-only vs GT:", action_only_vs_gt)
            print("  Joint vs GT:", joint_vs_gt)
            print("  Action-only vs Joint:", action_only_vs_joint, flush=True)

            torch.save(
                {
                    "action_only_world_model_trajectory": action_only,
                    "joint_world_model_trajectory": joint,
                    "gt_trajectory": gt,
                    "action_only_minus_joint": action_only - joint,
                    "action_only_vs_gt": action_only_vs_gt,
                    "joint_vs_gt": joint_vs_gt,
                    "action_only_vs_joint": action_only_vs_joint,
                    "action_only_inference_seconds": predictions["timing"]["pipe_infer"],
                },
                trajectory_dir / f"{stem}_comparison.pt",
            )
            manifest["tokens"].append(token)

    manifest["aggregate"] = {
        "action_only_vs_gt": _mean_metrics(action_only_gt_metrics),
        "joint_vs_gt": _mean_metrics(joint_gt_metrics),
        "action_only_vs_joint": _mean_metrics(action_only_joint_metrics),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    print("\nAggregate comparison:")
    print(json.dumps(manifest["aggregate"], indent=2), flush=True)
    print(f"Saved Action-only comparison to {output_dir}", flush=True)
