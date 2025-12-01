import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]  # repo root
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
UTILS = ROOT / "conversions" / "droid"
if str(UTILS) not in sys.path:
    sys.path.insert(0, str(UTILS))


def _copy_config_with_overrides(config_path: Path, workdir: Path, max_frames: int, out_root: Path) -> Path:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["max_frames"] = max_frames
    cfg["video_output_path"] = str(out_root / "videos")
    cfg["rrd_output_path"] = str(out_root / "rrd" / "output.rrd")
    cfg["tracks_npz_path"] = None
    # Make sure training output root points to same test dir
    cfg["training_output_root"] = str(out_root)
    out_cfg = workdir / "config_test.yaml"
    with open(out_cfg, "w") as f:
        yaml.safe_dump(cfg, f)
    return out_cfg


def _run_create_video(config_path: Path):
    cmd = [
        sys.executable,
        str(ROOT / "conversions" / "droid" / "create_video_with_tracks.py"),
        "--config",
        str(config_path),
    ]
    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT)
    return time.perf_counter() - start


@pytest.fixture(scope="module")
def config_path():
    env_path = os.environ.get("DROID_TEST_CONFIG")
    cfg = Path(env_path) if env_path else Path("conversions/droid/training_data/example_config.yaml")
    if not cfg.exists():
        pytest.fail(f"Config not found at {cfg}")
    return cfg


def _skip_if_data_missing(cfg: dict):
    required = [
        cfg.get("h5_path"),
        cfg.get("recordings_dir"),
        cfg.get("extrinsics_json_path"),
    ]
    missing = [p for p in required if not p or not Path(p).exists()]
    recordings_dir = cfg.get("recordings_dir")
    if recordings_dir and Path(recordings_dir).exists():
        has_svo = any(Path(recordings_dir).glob("*.svo"))
    else:
        has_svo = False
    if missing or not has_svo:
        msg = f"Skipping end-to-end; missing data: {missing}, SVO present={has_svo}"
        pytest.skip(msg)


def test_end_to_end_rrd_and_videos(config_path, tmp_path):
    max_frames_env = os.environ.get("DROID_TEST_E2E_MAX_FRAMES")
    max_frames = int(max_frames_env) if max_frames_env else 3
    out_root = Path("point_clouds") / "tests"
    out_root.mkdir(parents=True, exist_ok=True)

    cfg_copy = _copy_config_with_overrides(config_path, tmp_path, max_frames=max_frames, out_root=out_root)
    with open(cfg_copy, "r") as f:
        cfg = yaml.safe_load(f)
    _skip_if_data_missing(cfg)

    duration = _run_create_video(cfg_copy)

    config_tag = Path(cfg_copy).stem
    video_dir = Path(cfg["video_output_path"]) / config_tag / "tracks_reprojection"
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    assert videos, "No video produced by create_video_with_tracks"

    tracks_npz = Path(cfg["rrd_output_path"].replace(".rrd", "_gripper_tracks.npz"))
    assert tracks_npz.exists(), "tracks NPZ not saved"
    rrd_path = Path(cfg["rrd_output_path"])
    assert rrd_path.exists(), "RRD not saved"

    print(f"[TEST] end-to-end video+rrd runtime (~{max_frames} frames): {duration:.2f}s")
