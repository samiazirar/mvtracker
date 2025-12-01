"""Count sizes of H5 and SVO files referenced by cam2base extrinsic superset."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Tuple, Set


def _walk_episode_paths(base_path: Path):
    """Yield (h5_path, recordings_dir) for each episode under base_path."""
    for h5 in base_path.rglob("trajectory.h5"):
        episode_dir = h5.parent
        recordings_dir = episode_dir / "recordings" / "SVO"
        yield h5, recordings_dir


def _load_cam2base(path: Path) -> Dict:
    with open(path, "r") as f:
        return json.load(f)


def _extract_relative_paths(obj) -> Set[Path]:
    """
    Recursively extract relative episode paths from the cam2base JSON.

    Heuristics:
    - Any string containing a path separator ("/") is treated as a path.
    - Keys commonly used: "relative_path", "path".
    - Works whether the JSON is keyed by camera id or relative path.
    """
    paths = set()

    if isinstance(obj, dict):
        for k, v in obj.items():
            # Some formats store the path as a value
            if isinstance(v, str) and "/" in v:
                paths.add(Path(v))
            # Sometimes the key itself is the path (if it's not a pure camera id)
            if isinstance(k, str) and "/" in k:
                paths.add(Path(k))
            paths.update(_extract_relative_paths(v))
    elif isinstance(obj, list):
        for item in obj:
            paths.update(_extract_relative_paths(item))
    elif isinstance(obj, str):
        if "/" in obj:
            paths.add(Path(obj))

    return paths


def _count_sizes(h5_path: Path, recordings_dir: Path) -> Tuple[int, int]:
    h5_size = h5_path.stat().st_size if h5_path.exists() else 0
    svo_total = 0
    if recordings_dir.exists():
        for svo in recordings_dir.glob("*.svo"):
            svo_total += svo.stat().st_size
    return h5_size, svo_total


def human_readable_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    size = float(num_bytes)
    for unit in units:
        if size < 1024.0:
            return f"{size:.2f} {unit}"
        size /= 1024.0
    return f"{size:.2f} PB"


def main():
    parser = argparse.ArgumentParser(description="Count SVO and H5 sizes referenced by cam2base superset.")
    parser.add_argument(
        "--cam2base",
        required=True,
        help="Path to cam2base extrinsic superset JSON (e.g., cam2base_extrinsic_superset.json).",
    )
    parser.add_argument(
        "--data_root",
        required=True,
        help="Root of the DROID data tree that matches the cam2base paths (e.g., /data/droid/data/droid_raw/1.0.1).",
    )
    args = parser.parse_args()

    cam2base = _load_cam2base(Path(args.cam2base))
    data_root = Path(args.data_root)

    # Build a set of relative episode paths referenced in cam2base
    episode_rel_paths = _extract_relative_paths(cam2base)

    total_h5 = 0
    total_svo = 0

    print("[INFO] Counting sizes for episodes referenced in cam2base...")
    for rel in sorted(episode_rel_paths):
        episode_dir = data_root / rel
        h5_path = episode_dir / "trajectory.h5"
        recordings_dir = episode_dir / "recordings" / "SVO"

        h5_size, svo_size = _count_sizes(h5_path, recordings_dir)
        total_h5 += h5_size
        total_svo += svo_size

        print(f"- {rel}: H5={human_readable_size(h5_size)}, SVO total={human_readable_size(svo_size)}")

    print("=== Totals ===")
    print(f"H5 total: {human_readable_size(total_h5)}")
    print(f"SVO total: {human_readable_size(total_svo)}")
    print(f"Combined: {human_readable_size(total_h5 + total_svo)}")


if __name__ == "__main__":
    main()
