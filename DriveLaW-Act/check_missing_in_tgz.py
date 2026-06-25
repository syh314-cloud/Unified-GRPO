#!/usr/bin/env python3
"""在 navtrain .tgz 内查找 missing_by_log_all.json 里记录的缺图是否存在。

用法（在含 .tgz 的目录）:
  python check_missing_in_tgz.py \\
    --missing-json ~/missing_by_log_all.json \\
    --tgz navtrain_current_{1..4}.tgz navtrain_history_{1..4}.tgz

先对每个 tgz 建索引（tar -tzf），再与缺图清单交叉比对。
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_REL_PATH_RE = re.compile(
    r"(\d{4}\.\d{2}\.\d{2}\.\d{2}\.\d{2}\.\d{2}_veh-\d+_\d+_\d+/CAM_F0/[0-9a-f]{16}\.jpg)"
)
_TGZ_LINE_RE = _REL_PATH_RE  # tgz 内路径同样用 log/CAM_F0/hash.jpg 匹配


def _clean_rel_path(raw: str) -> str | None:
    m = _REL_PATH_RE.search(raw)
    return m.group(1) if m else None


def load_missing_paths(json_path: Path) -> list[str]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    paths: set[str] = set()
    for entries in data.values():
        for raw in entries:
            rel = _clean_rel_path(raw)
            if rel:
                paths.add(rel)
    return sorted(paths)


def index_tgz(tgz: Path) -> set[str]:
    print(f"indexing {tgz.name} ...", flush=True)
    proc = subprocess.run(
        ["tar", "-tzf", str(tgz)],
        capture_output=True,
        text=True,
        check=True,
    )
    # 统一成 log/CAM_F0/hash.jpg（兼容 current_split_N / history_split_N 等前缀）
    indexed: set[str] = set()
    for line in proc.stdout.splitlines():
        m = _TGZ_LINE_RE.search(line)
        if m:
            indexed.add(m.group(1))
    print(f"  -> {len(indexed)} CAM_F0 jpg paths", flush=True)
    return indexed


def main() -> int:
    parser = argparse.ArgumentParser(description="Check missing CAM_F0 paths inside navtrain tgz files.")
    parser.add_argument("--missing-json", type=Path, required=True)
    parser.add_argument("--tgz", type=Path, nargs="+", required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("missing_in_tgz_report"))
    args = parser.parse_args()

    missing = load_missing_paths(args.missing_json)
    print(f"missing paths (unique): {len(missing)}")

    found_in: dict[str, list[str]] = defaultdict(list)
    all_indexed: set[str] = set()
    tgz_index_sizes: dict[str, int] = {}
    skipped_tgz: list[str] = []
    found_count_by_tgz: dict[str, int] = defaultdict(int)

    for tgz in args.tgz:
        if not tgz.exists():
            skipped_tgz.append(str(tgz))
            print(f"WARN: skip missing file {tgz}", file=sys.stderr)
            continue
        idx = index_tgz(tgz)
        tgz_index_sizes[tgz.name] = len(idx)
        all_indexed |= idx
        for rel in missing:
            if rel in idx:
                found_in[rel].append(tgz.name)
                found_count_by_tgz[tgz.name] += 1

    not_in_any = [p for p in missing if p not in all_indexed]
    in_tgz = [p for p in missing if p in all_indexed]
    found_in_current = sum(
        1 for p in in_tgz if any("current" in n for n in found_in[p])
    )
    found_in_history = sum(
        1 for p in in_tgz if any("history" in n for n in found_in[p])
    )

    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    (out / "missing_found_in_tgz.txt").write_text(
        "\n".join(f"{p}\t{','.join(found_in[p])}" for p in in_tgz) + ("\n" if in_tgz else ""),
        encoding="utf-8",
    )
    (out / "missing_not_in_any_tgz.txt").write_text(
        "\n".join(not_in_any) + ("\n" if not_in_any else ""),
        encoding="utf-8",
    )
    summary = {
        "missing_json": str(args.missing_json.resolve()),
        "tgz_files_checked": [str(p.resolve()) for p in args.tgz if p.exists()],
        "tgz_files_skipped": skipped_tgz,
        "tgz_index_sizes": tgz_index_sizes,
        "found_count_by_tgz": dict(sorted(found_count_by_tgz.items())),
        "unique_missing_paths": len(missing),
        "found_in_tgz": len(in_tgz),
        "found_in_current_only": found_in_current,
        "found_in_history": found_in_history,
        "not_in_any_tgz": len(not_in_any),
        "interpretation": {
            "missing_found_in_tgz.txt": "磁盘缺、但 tgz 里有 → 重解压可补",
            "missing_not_in_any_tgz.txt": "8 个 tgz 都没有 → 官方包可能不含这些帧",
            "if_full_same_as_current_only": (
                "若加 history 后数字不变，先查 tgz_files_skipped、"
                "tgz_index_sizes 里 history 是否为 0，或 history_3/4 是否下完整"
            ),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("=" * 60)
    print(f"found in tgz     : {len(in_tgz)}  (current: {found_in_current}, history: {found_in_history})")
    print(f"not in any tgz   : {len(not_in_any)}")
    if skipped_tgz:
        print(f"skipped tgz      : {len(skipped_tgz)}")
    print("index sizes:")
    for name, size in sorted(tgz_index_sizes.items()):
        hits = found_count_by_tgz.get(name, 0)
        print(f"  {name}: {size} jpgs, {hits} missing hits")
    print(f"report dir       : {out.resolve()}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
