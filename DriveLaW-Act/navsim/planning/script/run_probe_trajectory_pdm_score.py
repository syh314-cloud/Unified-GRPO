"""Score trajectories saved by a probe with the official NAVSIM PDM scorer."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from navsim.common.dataloader import MetricCacheLoader
from navsim.common.dataclasses import Trajectory
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import (
    PDMScorer,
    PDMScorerConfig,
)
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import (
    PDMSimulator,
)


TRAJECTORY_SAMPLING = TrajectorySampling(time_horizon=4.0, interval_length=0.5)
PROPOSAL_SAMPLING = TrajectorySampling(num_poses=40, interval_length=0.1)


def _load_probe_record(path: Path) -> Dict[str, Any]:
    try:
        record = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        record = torch.load(path, map_location="cpu")
    if not isinstance(record, dict):
        raise TypeError(f"Expected a dictionary in {path}, got {type(record)}")
    return record


def _as_navsim_trajectory(value: Any, path: Path, key: str) -> Trajectory:
    if torch.is_tensor(value):
        value = value.detach().float().cpu().numpy()
    poses = np.asarray(value, dtype=np.float32)
    if poses.ndim == 3 and poses.shape[0] == 1:
        poses = poses[0]
    expected_shape = (TRAJECTORY_SAMPLING.num_poses, 3)
    if poses.shape != expected_shape:
        raise ValueError(
            f"{path}:{key} has shape {poses.shape}; expected {expected_shape} "
            "for a 4-second trajectory sampled every 0.5 seconds."
        )
    return Trajectory(poses=poses, trajectory_sampling=TRAJECTORY_SAMPLING)


def _average_row(results: pd.DataFrame) -> Dict[str, Any]:
    valid_results = results[results["valid"]]
    average: Dict[str, Any] = {
        column: float(valid_results[column].mean())
        for column in valid_results.select_dtypes(include=[np.number]).columns
    }
    average.update(
        {
            "token": "average",
            "valid": bool(results["valid"].all()),
            "trajectory_file": "",
            "error": "",
        }
    )
    return average


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute official NAVSIM PDM scores for saved probe trajectories."
    )
    parser.add_argument(
        "--trajectory-dir",
        type=Path,
        required=True,
        help="Directory containing *_trajectories.pt probe artifacts.",
    )
    parser.add_argument(
        "--metric-cache-path",
        type=Path,
        required=True,
        help="NAVSIM metric-cache root containing metadata/*.csv.",
    )
    parser.add_argument(
        "--trajectory-key",
        default="world_model_trajectory",
        help="Dictionary key to score (default: world_model_trajectory).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Output CSV path (default: <trajectory-dir>/pdm_scores_<key>.csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trajectory_files = sorted(args.trajectory_dir.glob("*_trajectories.pt"))
    if not trajectory_files:
        raise FileNotFoundError(
            f"No *_trajectories.pt files found in {args.trajectory_dir}"
        )

    records = [(path, _load_probe_record(path)) for path in trajectory_files]
    tokens = [str(record["token"]) for _, record in records]
    if len(tokens) != len(set(tokens)):
        raise ValueError("Probe trajectory directory contains duplicate tokens.")

    metric_cache_loader = MetricCacheLoader(args.metric_cache_path)
    missing_tokens = sorted(set(tokens) - set(metric_cache_loader.tokens))
    if missing_tokens:
        missing_text = "\n".join(f"  {token}" for token in missing_tokens)
        raise FileNotFoundError(
            "Metric cache is missing for these probe tokens:\n"
            f"{missing_text}\n"
            "Create metric caches for the same NAVSIM split before scoring."
        )

    simulator = PDMSimulator(PROPOSAL_SAMPLING)
    scorer = PDMScorer(PROPOSAL_SAMPLING, PDMScorerConfig())
    rows: List[Dict[str, Any]] = []

    for index, (path, record) in enumerate(records, start=1):
        token = str(record["token"])
        row: Dict[str, Any] = {
            "token": token,
            "valid": True,
            "trajectory_file": str(path),
            "error": "",
        }
        try:
            if args.trajectory_key not in record:
                raise KeyError(f"Missing key {args.trajectory_key!r}")
            trajectory = _as_navsim_trajectory(
                record[args.trajectory_key], path, args.trajectory_key
            )
            result = pdm_score(
                metric_cache=metric_cache_loader.get_from_token(token),
                model_trajectory=trajectory,
                future_sampling=PROPOSAL_SAMPLING,
                simulator=simulator,
                scorer=scorer,
            )
            row.update(asdict(result))
            print(
                f"[{index}/{len(records)}] {token}: score={row['score']:.6f}",
                flush=True,
            )
        except Exception as error:
            row["valid"] = False
            row["error"] = f"{type(error).__name__}: {error}"
            print(f"[{index}/{len(records)}] {token}: FAILED: {row['error']}", flush=True)
        rows.append(row)

    results = pd.DataFrame(rows)
    results.loc[len(results)] = _average_row(results)
    output_csv = args.output_csv or (
        args.trajectory_dir / f"pdm_scores_{args.trajectory_key}.csv"
    )
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)

    display_columns = [
        "token",
        "valid",
        "no_at_fault_collisions",
        "drivable_area_compliance",
        "ego_progress",
        "time_to_collision_within_bound",
        "comfort",
        "driving_direction_compliance",
        "score",
    ]
    print("\n" + results[[c for c in display_columns if c in results]].to_string(index=False))
    print(f"\nSaved PDM results to {output_csv}")


if __name__ == "__main__":
    main()
