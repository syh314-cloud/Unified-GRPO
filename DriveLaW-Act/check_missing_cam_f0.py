#!/usr/bin/env python3
"""检查 navtrain 每个 token 在 cache 时需要的 CAM_F0 是否存在。

与 run_dataset_caching 对齐：VideoDrive 会加载 history(0-3) + future(4-11) 的 cam_f0。
输出完整场景/缺图清单，便于对照补数据。
"""
import csv
import json
import os
import re
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

CACHE_SENSOR_ITERS = list(range(12))
NUM_FUTURE_POSES = 8

OPENSCENE = Path(os.environ.get(
    "OPENSCENE_DATA_ROOT",
    "/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset",
))
SENSOR_BLOBS_TRAINVAL = Path(os.environ.get(
    "SENSOR_BLOBS_TRAINVAL",
    "/data/home/scyb955/run/sunyuhan/navsim_workspace/navsim/download/trainval_sensor_blobs/trainval",
))
NAVSIM_LOGS_TRAINVAL = Path(os.environ.get(
    "NAVSIM_LOGS_TRAINVAL",
    "/data/home/scyb955/run/sunyuhan/navsim_workspace/navsim/download/trainval_navsim_logs/trainval",
))
OUT_DIR = Path(os.environ.get(
    "MISSING_CAM_F0_OUT_DIR",
    str(REPO_ROOT / "missing_cam_f0_reports"),
))

_SENSOR_PATH_MARKERS = (
    "sensor_blobs/trainval/",
    "trainval_sensor_blobs/trainval/",
)


def _rel_sensor_path(path: Path | str) -> str:
    s = str(path)
    for marker in _SENSOR_PATH_MARKERS:
        if marker in s:
            return s.split(marker, 1)[-1]
    return s


def _extract_missing_path_from_error(exc: Exception) -> str | None:
    msg = str(exc)
    for marker in _SENSOR_PATH_MARKERS:
        if marker in msg:
            return _rel_sensor_path(msg)
    for line in msg.splitlines():
        line = line.strip()
        if re.match(r"^[123]\)", line):
            candidate = line.split(")", 1)[1].strip()
            for marker in _SENSOR_PATH_MARKERS:
                if marker in candidate:
                    return _rel_sensor_path(candidate)
    return None


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


def _write_outputs(
    out_dir: Path,
    ok_tokens: int,
    bad_tokens: int,
    error_tokens: int,
    total_tokens: int,
    missing_by_log: dict[str, set[str]],
    ok_tokens_by_log: dict[str, int],
    bad_tokens_by_log: dict[str, int],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    all_missing_files = sorted({p for paths in missing_by_log.values() for p in paths})
    logs_with_missing = sorted(
        missing_by_log.keys(),
        key=lambda log: (-len(missing_by_log[log]), -bad_tokens_by_log.get(log, 0), log),
    )

    summary = {
        "total_tokens": total_tokens,
        "ok_tokens": ok_tokens,
        "bad_tokens": bad_tokens,
        "error_tokens": error_tokens,
        "unique_missing_files": len(all_missing_files),
        "logs_with_missing_files": len(logs_with_missing),
        "checked_frames_per_token": "history 0-3 + future 4-11 cam_f0 (VideoDrive cache)",
        "output_dir": str(out_dir.resolve()),
        "files": {
            "summary_json": "missing_cam_f0_report.json",
            "scene_summary_csv": "missing_scene_summary.csv",
            "missing_scenes_txt": "missing_scenes.txt",
            "missing_files_txt": "missing_files_all.txt",
            "missing_by_log_json": "missing_by_log_all.json",
            "missing_paths_csv": "missing_paths_by_scene.csv",
        },
    }
    (out_dir / "missing_cam_f0_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with open(out_dir / "missing_scene_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "log_name",
            "ok_tokens",
            "bad_tokens",
            "unique_missing_cam_f0_files",
        ])
        all_logs = set(ok_tokens_by_log) | set(bad_tokens_by_log) | set(missing_by_log)
        for log in sorted(all_logs):
            writer.writerow([
                log,
                ok_tokens_by_log.get(log, 0),
                bad_tokens_by_log.get(log, 0),
                len(missing_by_log.get(log, set())),
            ])

    (out_dir / "missing_scenes.txt").write_text(
        "\n".join(logs_with_missing) + ("\n" if logs_with_missing else ""),
        encoding="utf-8",
    )

    (out_dir / "missing_files_all.txt").write_text(
        "\n".join(all_missing_files) + ("\n" if all_missing_files else ""),
        encoding="utf-8",
    )

    missing_by_log_json = {
        log: sorted(missing_by_log[log]) for log in logs_with_missing
    }
    (out_dir / "missing_by_log_all.json").write_text(
        json.dumps(missing_by_log_json, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with open(out_dir / "missing_paths_by_scene.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["log_name", "relative_cam_f0_path"])
        for log in logs_with_missing:
            for rel_path in sorted(missing_by_log[log]):
                writer.writerow([log, rel_path])


def main():
    scene_filter = instantiate(OmegaConf.load(NAVTRAIN_SCENE_FILTER))
    sensor_config = SensorConfig.build_all_sensors(include=CACHE_SENSOR_ITERS)

    loader = SceneLoader(
        data_path=NAVSIM_LOGS_TRAINVAL,
        sensor_blobs_path=SENSOR_BLOBS_TRAINVAL,
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )

    missing_by_log: dict[str, set[str]] = defaultdict(set)
    ok_tokens_by_log: dict[str, int] = defaultdict(int)
    bad_tokens_by_log: dict[str, int] = defaultdict(int)
    ok_tokens = 0
    bad_tokens = 0
    error_tokens = 0

    for token in tqdm(loader.tokens, desc="check tokens"):
        log_name = loader.token_to_log_file[token]
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
                    rel = _rel_sensor_path(path)
                    token_missing.add(rel)
                    missing_by_log[log_name].add(rel)
            if token_missing:
                bad_tokens += 1
                bad_tokens_by_log[log_name] += 1
            else:
                ok_tokens += 1
                ok_tokens_by_log[log_name] += 1
        except Exception as exc:
            error_tokens += 1
            bad_tokens += 1
            bad_tokens_by_log[log_name] += 1
            rel = _extract_missing_path_from_error(exc)
            if rel:
                missing_by_log[log_name].add(rel)

    _write_outputs(
        OUT_DIR,
        ok_tokens,
        bad_tokens,
        error_tokens,
        len(loader.tokens),
        missing_by_log,
        ok_tokens_by_log,
        bad_tokens_by_log,
    )

    all_missing_files = {p for paths in missing_by_log.values() for p in paths}
    print("=" * 60)
    print(f"total tokens            : {len(loader.tokens)}")
    print(f"ok tokens               : {ok_tokens}")
    print(f"bad tokens              : {bad_tokens}")
    print(f"  load errors           : {error_tokens}")
    print(f"unique missing jpgs     : {len(all_missing_files)}")
    print(f"logs with missing files : {len(missing_by_log)}")
    print(f"output dir              : {OUT_DIR.resolve()}")
    print("  - missing_cam_f0_report.json")
    print("  - missing_scene_summary.csv")
    print("  - missing_scenes.txt")
    print("  - missing_files_all.txt")
    print("  - missing_by_log_all.json")
    print("  - missing_paths_by_scene.csv")
    print("=" * 60)


if __name__ == "__main__":
    main()
