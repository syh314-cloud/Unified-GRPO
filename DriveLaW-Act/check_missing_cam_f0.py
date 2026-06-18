#!/usr/bin/env python3
"""检查 navtrain 每个 token 在 cache 时需要的 CAM_F0 是否存在。

与 run_dataset_caching 对齐：VideoDrive 会加载 history(0-3) + future(4-11) 的 cam_f0，
不能只检查 AgentInput 的 4 帧 history。
"""
import json
import os
from collections import defaultdict
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import Scene, SensorConfig

REPO_ROOT = Path(__file__).resolve().parent
NAVTRAIN_SCENE_FILTER = (
    REPO_ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
)

# VideoDrive cache 与 videodrive_agent.get_sensor_config() 一致
CACHE_SENSOR_ITERS = list(range(12))
# TrajectoryTargetBuilder: time_horizon=4, interval_length=0.5 -> 8 future poses
NUM_FUTURE_POSES = 8

OPENSCENE = Path(os.environ.get(
    "OPENSCENE_DATA_ROOT",
    "/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset",
))
OUT = REPO_ROOT / "missing_cam_f0_report.json"


def _cam_f0_paths_from_scene(scene) -> list[Path]:
    paths: list[Path] = []
    agent_input = scene.get_agent_input()
    for cams in agent_input.cameras:
        img = cams.cam_f0.image
        if img is not None:
            paths.append(Path(img))

    future_cams = scene.get_future_frames(num_trajectory_frames=NUM_FUTURE_POSES)
    for cams in future_cams:
        img = cams.cam_f0.image
        if img is not None:
            paths.append(Path(img))
    return paths


def main():
    scene_filter = instantiate(OmegaConf.load(NAVTRAIN_SCENE_FILTER))
    sensor_config = SensorConfig.build_all_sensors(include=CACHE_SENSOR_ITERS)

    loader = SceneLoader(
        data_path=OPENSCENE / "navsim_logs" / "trainval",
        sensor_blobs_path=OPENSCENE / "sensor_blobs" / "trainval",
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )

    missing_paths: set[str] = set()
    missing_by_log: dict[str, set[str]] = defaultdict(set)
    ok_tokens = 0
    bad_tokens = 0
    error_tokens = 0

    for token in tqdm(loader.tokens, desc="check tokens"):
        try:
            scene = Scene.from_scene_dict_list(
                loader.scene_frames_dicts[token],
                loader._sensor_blobs_path,
                num_history_frames=scene_filter.num_history_frames,
                num_future_frames=scene_filter.num_future_frames,
                sensor_config=sensor_config,
                load_image_path=True,
            )
            token_missing: set[str] = set()
            for path in _cam_f0_paths_from_scene(scene):
                if not path.exists():
                    rel = str(path).split("sensor_blobs/trainval/")[-1]
                    token_missing.add(rel)
                    missing_paths.add(rel)
                    missing_by_log[loader.token_to_log_file[token]].add(rel)
            if token_missing:
                bad_tokens += 1
            else:
                ok_tokens += 1
        except Exception as exc:
            error_tokens += 1
            bad_tokens += 1
            missing_paths.add(f"__ERROR__:{token}:{type(exc).__name__}:{str(exc)[:300]}")

    report = {
        "total_tokens": len(loader.tokens),
        "ok_tokens": ok_tokens,
        "bad_tokens": bad_tokens,
        "error_tokens": error_tokens,
        "unique_missing_files": len([p for p in missing_paths if not p.startswith("__ERROR__")]),
        "checked_frames_per_token": "history 0-3 + future 4-11 cam_f0 (VideoDrive cache)",
        "missing_by_log_top20": {
            k: len(v) for k, v in sorted(missing_by_log.items(), key=lambda x: -len(x[1]))[:20]
        },
        "missing_samples": sorted(p for p in missing_paths if not p.startswith("__ERROR__"))[:500],
        "error_samples": sorted(p for p in missing_paths if p.startswith("__ERROR__"))[:50],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"total tokens       : {report['total_tokens']}")
    print(f"ok tokens          : {ok_tokens}")
    print(f"bad tokens         : {bad_tokens}")
    print(f"  load errors      : {error_tokens}")
    print(f"unique missing jpgs: {report['unique_missing_files']}")
    print(f"check scope        : {report['checked_frames_per_token']}")
    print(f"report saved to    : {OUT.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()