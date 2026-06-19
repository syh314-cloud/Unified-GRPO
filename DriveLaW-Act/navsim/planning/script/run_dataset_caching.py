from typing import Any, Dict, List, Optional, Union
from pathlib import Path
from collections import Counter
import json
import logging
import uuid
import os

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pytorch_lightning as pl

from nuplan.planning.utils.multithreading.worker_pool import WorkerPool
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.planning.training.dataset import Dataset, CACHE_REPORTS_DIRNAME
from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.agents.abstract_agent import AbstractAgent

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"


def _agent_loads_image_paths(cfg: DictConfig) -> bool:
    """VideoDrive caches image paths; most baselines need decoded sensor arrays."""
    target = str(cfg.agent.get("_target_", ""))
    return "videodrive" in target.lower()


def _merge_cache_stats(into: Dict[str, Any], stats: Dict[str, Any]) -> None:
    into["succeeded"] = into.get("succeeded", 0) + int(stats.get("succeeded", 0))
    into["skipped"] = into.get("skipped", 0) + int(stats.get("skipped", 0))
    into["total"] = into.get("total", 0) + int(stats.get("total", 0))
    for reason, count in stats.get("skip_reasons", {}).items():
        into.setdefault("skip_reasons", Counter())[reason] += int(count)
    for sample in stats.get("skip_samples", []):
        samples = into.setdefault("skip_samples", [])
        if len(samples) < 200:
            samples.append(sample)


def _aggregate_cache_reports(reports_dir: Path) -> Dict[str, Any]:
    totals: Dict[str, Any] = {
        "succeeded": 0,
        "skipped": 0,
        "total": 0,
        "skip_reasons": Counter(),
        "skip_samples": [],
    }
    if not reports_dir.is_dir():
        return totals

    for report_file in sorted(reports_dir.glob("cache_skip_report_*.json")):
        try:
            with open(report_file, encoding="utf-8") as f:
                stats = json.load(f)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable cache report %s: %s", report_file, exc)
            continue
        if isinstance(stats, dict):
            _merge_cache_stats(totals, stats)
    return totals


def cache_features(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Optional[Any]]:
    """
    Helper function to cache features and targets of learnable agent.
    :param args: arguments for caching
    """
    node_id = int(os.environ.get("NODE_RANK", 0))
    thread_id = str(uuid.uuid4())

    log_names = [a["log_file"] for a in args]
    tokens = [t for a in args for t in a["tokens"]]
    cfg: DictConfig = args[0]["cfg"]
    load_image_path = _agent_loads_image_paths(cfg)
    os.environ["NAVSIM_CACHE_ONLY"] = "1"

    agent: AbstractAgent = instantiate(cfg.agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens
    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=load_image_path,
    )
    logger.info(f"Extracted {len(scene_loader.tokens)} scenarios for thread_id={thread_id}, node_id={node_id}.")

    dataset = Dataset(
        scene_loader=scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=cfg.cache_path,
        force_cache_computation=cfg.force_cache_computation,
    )
    # worker_map flattens list returns; a bare dict would be iterated by key (str).
    return [dataset.cache_stats]


@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for dataset caching script.
    :param cfg: omegaconf dictionary
    """
    logger.info("Global Seed set to 0")
    pl.seed_everything(0, workers=True)

    logger.info("Building Worker")
    worker: WorkerPool = instantiate(cfg.worker)

    logger.info("Building SceneLoader")
    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)
    scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=scene_filter,
        sensor_config=SensorConfig.build_no_sensors(),
    )
    logger.info(f"Extracted {len(scene_loader)} scenarios for training/validation dataset")

    data_points = [
        {
            "cfg": cfg,
            "log_file": log_file,
            "tokens": tokens_list,
        }
        for log_file, tokens_list in scene_loader.get_tokens_list_per_log().items()
    ]

    worker_map(worker, cache_features, data_points)

    cache_path = Path(cfg.cache_path)
    cache_path.mkdir(parents=True, exist_ok=True)
    reports_dir = cache_path / CACHE_REPORTS_DIRNAME
    reports_dir.mkdir(parents=True, exist_ok=True)

    aggregated = _aggregate_cache_reports(reports_dir)
    total_succeeded = int(aggregated.get("succeeded", 0))
    total_skipped = int(aggregated.get("skipped", 0))
    total_tokens = int(aggregated.get("total", 0))
    skip_reasons: Counter = aggregated.get("skip_reasons", Counter())
    skip_samples: List[Dict[str, str]] = list(aggregated.get("skip_samples", []))

    summary = {
        "tokens_expected": len(scene_loader),
        "tokens_attempted": total_tokens,
        "succeeded": total_succeeded,
        "skipped": total_skipped,
        "success_rate": round(total_succeeded / total_tokens, 4) if total_tokens else 0.0,
        "skip_reasons": dict(skip_reasons),
        "skip_samples": skip_samples,
    }
    summary_path = reports_dir / "cache_skip_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 60)
    logger.info("GLOBAL CACHE SUMMARY")
    logger.info("  tokens attempted : %d", total_tokens)
    logger.info("  succeeded        : %d", total_succeeded)
    logger.info("  skipped          : %d", total_skipped)
    logger.info("  success rate     : %.2f%%", 100.0 * summary["success_rate"])
    logger.info("  skip reasons     : %s", dict(skip_reasons))
    logger.info("  full report      : %s", summary_path)
    logger.info("=" * 60)
    logger.info(f"Finished caching {len(scene_loader)} scenarios for training/validation dataset")


if __name__ == "__main__":
    main()
