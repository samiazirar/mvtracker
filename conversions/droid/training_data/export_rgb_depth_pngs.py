"""Export RGB and depth frames as lossless PNGs for training, mirroring DROID layout."""

import argparse
import glob
import os
import re
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import pyzed.sl as sl
import yaml

from utils import find_episode_data_by_date, find_svo_for_camera


def _derive_episode_relative_path(h5_path: str) -> str:
    parts = os.path.normpath(os.path.dirname(h5_path)).split(os.sep)
    return os.path.join(*parts[-5:]) if len(parts) >= 5 else os.path.basename(os.path.dirname(h5_path))


def _normalize_episode_id_to_rel_path(episode_id: str, cam2base_path: str = None) -> str:
    if cam2base_path and os.path.exists(cam2base_path):
        with open(cam2base_path, "r") as f:
            cam2base = yaml.safe_load(f) if cam2base_path.endswith((".yaml", ".yml")) else __import__("json").load(f)
        rels = set()
        def collect(obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    if k == "relative_path" and isinstance(v, str):
                        rels.add(v)
                    collect(v)
            elif isinstance(obj, list):
                for i in obj:
                    collect(i)
        collect(cam2base)
        parts = episode_id.split("+")
        ts = parts[-1] if parts else episode_id
        ts_norm = ts.replace("h", ":").replace("m", ":").replace("s", "")
        date_bits = ts.split("-")[0:3]
        date_fragment = "-".join(date_bits) if date_bits else ""
        time_digits = re.sub(r"[^0-9]", "", ts_norm)
        for rel in rels:
            rel_digits = re.sub(r"[^0-9]", "", rel)
            if date_fragment and date_fragment in rel and time_digits and time_digits in rel_digits:
                return rel
    parts = episode_id.split("+")
    if len(parts) >= 3:
        lab = parts[0]
        timestamp = parts[-1]
        date_bits = timestamp.split("-")
        if len(date_bits) >= 4:
            date_folder = "-".join(date_bits[:3])
            time_folder = timestamp.replace("h", ":").replace("m", ":").replace("s", "")
            return os.path.join(lab, date_folder, time_folder)
    return episode_id


def _init_cameras(config: Dict):
    cameras = {}
    # External cams
    ext_data = find_episode_data_by_date(config["h5_path"], config["extrinsics_json_path"]) or {}
    for cam_id in ext_data.keys():
        if not cam_id.isdigit():
            continue
        svo = find_svo_for_camera(config["recordings_dir"], cam_id)
        if svo:
            cameras[cam_id] = {"type": "external", "svo": svo}
    # Wrist
    metadata_path = config.get("metadata_path")
    if metadata_path is None:
        episode_dir = os.path.dirname(config["h5_path"])
        metadata_files = glob.glob(os.path.join(episode_dir, "metadata_*.json"))
        if metadata_files:
            metadata_path = metadata_files[0]
    if metadata_path and os.path.exists(metadata_path):
        import json
        with open(metadata_path, "r") as f:
            meta = json.load(f)
        wrist_serial = str(meta.get("wrist_cam_serial", "") or "")
        if wrist_serial:
            svo = find_svo_for_camera(config["recordings_dir"], wrist_serial)
            if svo:
                cameras[wrist_serial] = {"type": "wrist", "svo": svo}
    return cameras


def _save_png(path: str, image: np.ndarray):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cv2.imwrite(path, image)


def main():
    parser = argparse.ArgumentParser(description="Export RGB/Depth PNGs from SVO recordings.")
    parser.add_argument("--config", default="conversions/droid/config.yaml", help="Path to YAML config file.")
    parser.add_argument("--output_root", default=None, help="Override root for training outputs.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolve episode paths from episode_id + data_root when provided
    if config.get("episode_id") and config.get("data_root"):
        rel_path = _normalize_episode_id_to_rel_path(config["episode_id"], config.get("extrinsics_json_path"))
        base = Path(config["data_root"]) / rel_path
        config["h5_path"] = str(base / "trajectory.h5")
        config["recordings_dir"] = str(base / "recordings" / "SVO")
        if config.get("metadata_path") is None:
            config["metadata_path"] = None

    output_root = args.output_root or config.get("training_output_root", "training_output")
    rel_path = _derive_episode_relative_path(config["h5_path"])
    base_out = os.path.join(output_root, rel_path, "recordings")
    rgb_dir = os.path.join(base_out, "rgb")
    depth_dir = os.path.join(base_out, "depth")

    cameras = _init_cameras(config)
    if not cameras:
        print("[ERROR] No cameras found to export.")
        return

    max_frames = config.get("max_frames", 0)
    for serial, data in cameras.items():
        zed = sl.Camera()
        init = sl.InitParameters()
        init.set_from_svo_file(data["svo"])
        init.svo_real_time_mode = False
        init.coordinate_units = sl.UNIT.METER
        init.depth_mode = sl.DEPTH_MODE.NEURAL

        if zed.open(init) != sl.ERROR_CODE.SUCCESS:
            print(f"[WARN] Failed to open {serial}")
            continue

        runtime = sl.RuntimeParameters()
        frame_idx = 0
        while True:
            if max_frames and frame_idx >= max_frames:
                break
            if zed.grab(runtime) != sl.ERROR_CODE.SUCCESS:
                break

            # RGB
            mat_img = sl.Mat()
            zed.retrieve_image(mat_img, sl.VIEW.LEFT)
            img_bgra = mat_img.get_data()
            frame_rgb = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)

            # Depth (float meters -> uint16 millimeters)
            mat_depth = sl.Mat()
            zed.retrieve_measure(mat_depth, sl.MEASURE.DEPTH)
            depth_m = mat_depth.get_data()
            depth_mm = np.nan_to_num(depth_m, nan=0.0, posinf=0.0, neginf=0.0) * 1000.0
            depth_uint16 = np.clip(depth_mm, 0, np.iinfo(np.uint16).max).astype(np.uint16)

            frame_name = f"{serial}_{frame_idx:06d}.png"
            _save_png(os.path.join(rgb_dir, frame_name), frame_rgb)
            _save_png(os.path.join(depth_dir, frame_name), depth_uint16)

            frame_idx += 1

        zed.close()
        print(f"[INFO] Saved {frame_idx} frames for camera {serial} to {base_out}")


if __name__ == "__main__":
    main()
