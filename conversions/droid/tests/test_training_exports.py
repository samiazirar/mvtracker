import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from conversions.droid.training_data.export_tracks import _derive_episode_relative_path


ROOT = Path(__file__).resolve().parents[3]  # repo root


def _copy_config_with_overrides(config_path: Path, tmpdir: Path, max_frames: int = 1, video_out: bool = False):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["max_frames"] = max_frames
    cfg["training_output_root"] = str(tmpdir)
    if video_out:
        cfg["video_output_path"] = str(tmpdir / "videos")
        cfg["rrd_output_path"] = str(tmpdir / "videos" / "dummy.rrd")
    out_cfg = tmpdir / "config.yaml"
    with open(out_cfg, "w") as f:
        yaml.safe_dump(cfg, f)
    return out_cfg


def _run_script(script_rel: str, config_path: Path, output_root: Path, extra_args=None):
    cmd = [
        sys.executable,
        str(ROOT / script_rel),
        "--config",
        str(config_path),
        "--output_root",
        str(output_root),
    ]
    if extra_args:
        cmd.extend(extra_args)
    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT)
    return time.perf_counter() - start


def _run_script_no_output_override(script_rel: str, config_path: Path, extra_args=None):
    cmd = [
        sys.executable,
        str(ROOT / script_rel),
        "--config",
        str(config_path),
    ]
    if extra_args:
        cmd.extend(extra_args)
    start = time.perf_counter()
    subprocess.run(cmd, check=True, cwd=ROOT)
    return time.perf_counter() - start


def _episode_out_path(cfg: dict, output_root: Path):
    rel = _derive_episode_relative_path(cfg["h5_path"])
    return output_root / rel


@pytest.fixture(scope="module")
def config_path():
    env_path = os.environ.get("DROID_TEST_CONFIG")
    if not env_path:
        pytest.skip("DROID_TEST_CONFIG not set; skipping data-dependent tests.")
    return Path(env_path)


def test_export_tracks_runs_and_writes_outputs(config_path, tmp_path):
    cfg_copy = _copy_config_with_overrides(config_path, tmp_path, max_frames=1)
    with open(cfg_copy, "r") as f:
        cfg = yaml.safe_load(f)

    duration = _run_script(
        "conversions/droid/training_data/export_tracks.py",
        cfg_copy,
        tmp_path,
    )

    episode_out = _episode_out_path(cfg, tmp_path)
    tracks_npz = episode_out / "tracks.npz"
    extrinsics_npz = episode_out / "extrinsics.npz"
    quality_json = episode_out / "quality.json"

    assert tracks_npz.exists(), "tracks.npz not written"
    assert extrinsics_npz.exists(), "extrinsics.npz not written"
    assert quality_json.exists(), "quality.json not written"

    data = np.load(tracks_npz)
    for key in ["tracks_3d", "gripper_poses", "gripper_positions", "cartesian_positions", "fps"]:
        assert key in data, f"Missing {key} in tracks.npz"
    assert data["tracks_3d"].shape[0] == 1, "max_frames override not respected"

    print(f"[TEST] export_tracks runtime (1 frame): {duration:.2f}s")


@pytest.mark.skipif(
    os.environ.get("DROID_TEST_EXPORT_RGB_DEPTH") not in ("1", "true", "True"),
    reason="RGB/depth export test disabled; set DROID_TEST_EXPORT_RGB_DEPTH=1 to enable.",
)
def test_export_rgb_depth_pngs(config_path, tmp_path):
    cfg_copy = _copy_config_with_overrides(config_path, tmp_path, max_frames=1)
    with open(cfg_copy, "r") as f:
        cfg = yaml.safe_load(f)

    duration = _run_script(
        "conversions/droid/training_data/export_rgb_depth_pngs.py",
        cfg_copy,
        tmp_path,
    )

    episode_out = _episode_out_path(cfg, tmp_path) / "recordings"
    rgb_dir = episode_out / "rgb"
    depth_dir = episode_out / "depth"

    assert rgb_dir.exists(), "RGB directory missing"
    assert depth_dir.exists(), "Depth directory missing"
    rgb_files = list(rgb_dir.glob("*.png"))
    depth_files = list(depth_dir.glob("*.png"))
    assert rgb_files, "No RGB PNGs written"
    assert depth_files, "No depth PNGs written"

    print(f"[TEST] export_rgb_depth_pngs runtime (1 frame): {duration:.2f}s")


@pytest.mark.skipif(
    os.environ.get("DROID_TEST_FULL_SEQUENCE") not in ("1", "true", "True"),
    reason="Full-sequence tests disabled; set DROID_TEST_FULL_SEQUENCE=1 to enable.",
)
def test_export_tracks_full_sequence(config_path, tmp_path):
    cfg_copy = _copy_config_with_overrides(config_path, tmp_path, max_frames=0)
    with open(cfg_copy, "r") as f:
        cfg = yaml.safe_load(f)
    duration = _run_script(
        "conversions/droid/training_data/export_tracks.py",
        cfg_copy,
        tmp_path,
    )
    episode_out = _episode_out_path(cfg, tmp_path)
    tracks_npz = episode_out / "tracks.npz"
    assert tracks_npz.exists(), "tracks.npz not written"
    data = np.load(tracks_npz)
    assert data["tracks_3d"].shape[0] == data["num_frames"], "full sequence not exported"
    print(f"[TEST] export_tracks runtime (full): {duration:.2f}s")


@pytest.mark.skipif(
    os.environ.get("DROID_TEST_FULL_SEQUENCE") not in ("1", "true", "True"),
    reason="Full-sequence tests disabled; set DROID_TEST_FULL_SEQUENCE=1 to enable.",
)
@pytest.mark.skipif(
    os.environ.get("DROID_TEST_EXPORT_RGB_DEPTH") not in ("1", "true", "True"),
    reason="RGB/depth export test disabled; set DROID_TEST_EXPORT_RGB_DEPTH=1 to enable.",
)
def test_export_rgb_depth_pngs_full_sequence(config_path, tmp_path):
    cfg_copy = _copy_config_with_overrides(config_path, tmp_path, max_frames=0)
    with open(cfg_copy, "r") as f:
        cfg = yaml.safe_load(f)

    duration = _run_script(
        "conversions/droid/training_data/export_rgb_depth_pngs.py",
        cfg_copy,
        tmp_path,
    )

    episode_out = _episode_out_path(cfg, tmp_path) / "recordings"
    rgb_files = list((episode_out / "rgb").glob("*.png"))
    depth_files = list((episode_out / "depth").glob("*.png"))
    assert rgb_files and depth_files, "Full sequence export missing outputs"
    print(f"[TEST] export_rgb_depth_pngs runtime (full): {duration:.2f}s")


@pytest.mark.skipif(
    os.environ.get("DROID_TEST_VIDEO") not in ("1", "true", "True"),
    reason="Video test disabled; set DROID_TEST_VIDEO=1 to enable.",
)
def test_create_video_with_tracks(config_path, tmp_path):
    # Limit frames if env provides a cap
    max_frames_env = os.environ.get("DROID_TEST_VIDEO_MAX_FRAMES")
    max_frames = int(max_frames_env) if max_frames_env else 5
    cfg_copy = _copy_config_with_overrides(config_path, tmp_path, max_frames=max_frames, video_out=True)

    duration = _run_script_no_output_override(
        "conversions/droid/create_video_with_tracks.py",
        cfg_copy,
    )

    with open(cfg_copy, "r") as f:
        cfg = yaml.safe_load(f)
    config_tag = Path(cfg_copy).stem
    video_dir = Path(cfg["video_output_path"]) / config_tag / "tracks_reprojection"
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.avi"))
    assert videos, "No video produced by create_video_with_tracks"

    print(f"[TEST] create_video_with_tracks runtime (~{max_frames} frames): {duration:.2f}s")
