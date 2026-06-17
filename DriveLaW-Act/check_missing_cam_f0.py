#!/usr/bin/env python3
"""检查 navtrain 每个 token 需要的 CAM_F0 是否存在。"""
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

from navsim.common.dataloader import SceneLoader
from navsim.common.dataclasses import SensorConfig

REPO_ROOT = Path(__file__).resolve().parent
NAVTRAIN_SCENE_FILTER = (
    REPO_ROOT / "navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml"
)

# ===== 按你的路径改 =====
OPENSCENE = Path(os.environ.get(
    "OPENSCENE_DATA_ROOT",
    "/data/home/scyb955/run/sunyuhan/navsim_workspace/dataset",
))
OUT = REPO_ROOT / "missing_cam_f0_report.json"


def main():
    scene_filter = instantiate(OmegaConf.load(NAVTRAIN_SCENE_FILTER))
    sensor_config = SensorConfig.build_all_sensors(include=[0, 1, 2, 3])  # navtrain 4 帧 history

    loader = SceneLoader(
        data_path=OPENSCENE / "navsim_logs" / "trainval",
        sensor_blobs_path=OPENSCENE / "sensor_blobs" / "trainval",
        scene_filter=scene_filter,
        sensor_config=sensor_config,
        load_image_path=True,
    )

    missing_paths = set()
    missing_by_log = defaultdict(set)
    ok_tokens = 0
    bad_tokens = 0

    for token in tqdm(loader.tokens, desc="check tokens"):
        try:
            agent_input = loader.get_agent_input_from_token(token)
            token_ok = True
            for i, cams in enumerate(agent_input.cameras):
                p = cams.cam_f0.image
                if p is None:
                    continue
                path = Path(p)
                if not path.exists():
                    token_ok = False
                    rel = str(path).split("sensor_blobs/trainval/")[-1]
                    missing_paths.add(rel)
                    log = loader.token_to_log_file[token]
                    missing_by_log[log].add(rel)
            if token_ok:
                ok_tokens += 1
            else:
                bad_tokens += 1
        except Exception as e:
            bad_tokens += 1
            missing_paths.add(f"__ERROR__:{token}:{type(e).__name__}:{e}")

    report = {
        "total_tokens": len(loader.tokens),
        "ok_tokens": ok_tokens,
        "bad_tokens": bad_tokens,
        "unique_missing_files": len(missing_paths),
        "missing_by_log_top20": {
            k: len(v) for k, v in sorted(missing_by_log.items(), key=lambda x: -len(x[1]))[:20]
        },
        "missing_samples": sorted(missing_paths)[:500],
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"total tokens      : {report['total_tokens']}")
    print(f"ok tokens         : {ok_tokens}")
    print(f"bad tokens        : {bad_tokens}")
    print(f"unique missing jpgs: {len(missing_paths)}")
    print(f"report saved to   : {OUT.resolve()}")
    print("=" * 60)

if __name__ == "__main__":
    main()