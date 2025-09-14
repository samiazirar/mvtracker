#!/usr/bin/env python3
"""
Pack RH20T episodes into single .npz samples.

Outputs (per episode):
  rgbs:  [C, T, 3, H, W] uint8          (channels-first)
  depths:[C, T, 1, H, W] uint16         (explicit depth channel)
  intrs: [C, T, 3, 3]   float32
  extrs: [C, T, 3, 4]   float32
  query_points: [N, 4] float32 (t,x,y,z) (or from --query-points)

Assumptions:
- Episode folder contains camera subfolders named like 'cam_038522062288'
  with 'color/' (jpg) and 'depth/' (png/16-bit) plus optional per-cam timestamps.npy.
- Global episode timestamps at <episode>/timestamps.npy (if missing, union of cams).
- Calibration folders live at <root>/calib/<calib_ts> with intrinsics.npy, extrinsics.npy,
  and (optionally) devices.npy mapping indexes to camera ids.
- If a frame for a camera is missing at time t, we reuse the closest previous (forward-fill).
- If no calib timestamp ≤ t exists, we use the earliest available calib.

This script is defensive: handles dict/array calib .npy formats and pads images if
per-camera resolutions differ (pads to max H,W with zeros).


HOW TO RUN
----------

0) Deps (once):
   pip install --upgrade numpy pillow tqdm

1) Pack **all** episodes:
   python scripts/rh20t/rh20t_pack_episodes.py \
     --root /data/rh20t_api/data/RH20T/RH20T_cfg3 \
     --out  /data/rh20t_api/data/RH20T/packed_npz \
     --episodes-glob "task_*" \
     --compress

2) Pack a **single** episode:
   python scripts/rh20t/rh20t_pack_episodes.py \
     --root /data/rh20t_api/data/RH20T/RH20T_cfg3 \
     --out  /data/rh20t_api/data/RH20T/packed_npz \
     --only-episode task_0032_user_0010_scene_0005_cfg_0003 \
     --compress

3) Optional query points (shape [N,4] (t,x,y,z), [T,N,3], or [N,3]):
   python scripts/rh20t/rh20t_pack_episodes.py ... \
     --query-points /path/to/query_points.npy

4) Load a produced sample (PyTorch):
   import numpy as np, torch
   sample = np.load('/data/rh20t_api/data/RH20T/packed_npz/<episode>.npz')
   rgbs = torch.from_numpy(sample["rgbs"]).float()        # [C,T,3,H,W]
   depths = torch.from_numpy(sample["depths"]).float()    # [C,T,1,H,W]
   intrs = torch.from_numpy(sample["intrs"]).float()      # [C,T,3,3]
   extrs = torch.from_numpy(sample["extrs"]).float()      # [C,T,3,4]
   query_points = torch.from_numpy(sample["query_points"]).float()  # [N,4]

Notes:
- Script expects calib under <root>/calib/<timestamp> with intrinsics.npy, extrinsics.npy,
  and optionally devices.npy (camera id ordering).
- Missing frames per camera are forward-filled (use last available).
- Outputs one .npz per episode in --out.
"""
import argparse
import os
import re
import json
from glob import glob
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
from PIL import Image
from tqdm import tqdm

NUM_RE = re.compile(r"(\d+)")
def _num_in_name(p: Path) -> Optional[int]:
    m = NUM_RE.search(p.stem)
    return int(m.group(1)) if m else None

def read_rgb(path: Path) -> np.ndarray:
    im = Image.open(path).convert("RGB")
    return np.asarray(im, dtype=np.uint8)

def read_depth(path: Path) -> np.ndarray:
    # Depth often in 16-bit PNG; keep raw units as uint16
    im = Image.open(path)
    # If JPEG depth (8-bit), upcast to uint16 for consistent dtype
    arr = np.asarray(im)
    if arr.dtype == np.uint8:
        return arr.astype(np.uint16)
    if arr.dtype == np.int32:  # PIL "I" can be int32
        # Clip into uint16 range
        arr = np.clip(arr, 0, 65535).astype(np.uint16)
    elif arr.dtype != np.uint16:
        arr = arr.astype(np.uint16, copy=False)
    return arr

def load_timestamps(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        ts = np.load(path, allow_pickle=True)
        ts = np.asarray(ts).astype(np.int64)
        ts.sort()
        return ts
    return None

def list_frames(folder: Path) -> Dict[int, Path]:
    out = {}
    if not folder.exists():
        return out
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        for p in folder.glob(ext):
            t = _num_in_name(p)
            if t is not None:
                out[t] = p
    return dict(sorted(out.items()))

def forward_fill_index(sorted_keys: List[int], query_ts: np.ndarray) -> np.ndarray:
    """For each t in query_ts, return index of largest key <= t; if none, use 0."""
    keys = np.array(sorted_keys, dtype=np.int64)
    if len(keys) == 0:
        return np.full_like(query_ts, -1)
    # For each t, pos = rightmost insertion point - 1
    pos = np.searchsorted(keys, query_ts, side="right") - 1
    pos[pos < 0] = 0
    return pos

def pad_to(arr: np.ndarray, H: int, W: int, is_rgb: bool) -> np.ndarray:
    if is_rgb:
        h, w, c = arr.shape
        out = np.zeros((H, W, 3), dtype=arr.dtype)
        out[:h, :w, :] = arr
    else:
        h, w = arr.shape[:2]
        out = np.zeros((H, W), dtype=arr.dtype)
        out[:h, :w] = arr
    return out

def to_4x4(mat: np.ndarray) -> np.ndarray:
    """Ensure extrinsics are 4x4 homogeneous."""
    mat = np.asarray(mat)
    if mat.shape == (4,4):
        return mat.astype(np.float32)
    if mat.shape == (3,4):
        bot = np.array([[0,0,0,1]], dtype=mat.dtype)
        return np.vstack([mat, bot]).astype(np.float32)
    # Fallback: identity
    I = np.eye(4, dtype=np.float32)
    I[:mat.shape[0], :mat.shape[1]] = mat[:min(4,mat.shape[0]), :min(4,mat.shape[1])]
    return I

def pick_key(keys: List[int], t: int) -> int:
    """Largest key <= t, else first."""
    if not keys:
        return None
    idx = np.searchsorted(np.array(keys, dtype=np.int64), t, side="right") - 1
    if idx < 0:
        idx = 0
    return keys[idx]

def load_devices_index(calib_dir: Path) -> Optional[List[str]]:
    # devices.npy may be under this calib dir or parent calib root
    for cand in [calib_dir / "devices.npy", calib_dir.parent / "devices.npy"]:
        if cand.exists():
            val = np.load(cand, allow_pickle=True)
            if isinstance(val, np.ndarray) and val.dtype == object and val.size == 1:
                val = val.item()
            # Accept common shapes: dict with 'cameras' or 'serials'
            if isinstance(val, dict):
                for k in ("cameras", "serials", "device_ids", "rgb"):
                    if k in val:
                        seq = val[k]
                        return [str(x) for x in list(seq)]
            # Or straight list/array of ids
            if isinstance(val, (list, tuple)):
                return [str(x) for x in val]
            if isinstance(val, np.ndarray):
                flat = [str(x) for x in val.tolist()]
                return flat
    return None

def extract_for_cam(obj, cam_id: str, fallback_index: int = 0):
    """Try multiple formats to fetch the matrix for a given camera id."""
    if isinstance(obj, dict):
        # Direct key matches like "cam_123", "123", etc.
        for key in (cam_id, f"cam_{cam_id}", f"rgb_{cam_id}", f"device_{cam_id}"):
            if key in obj:
                return np.asarray(obj[key])
        # Try numeric-only keys
        for k, v in obj.items():
            if isinstance(k, str) and cam_id in k:
                return np.asarray(v)
        # Fallback to first value
        try:
            return np.asarray(next(iter(obj.values())))
        except StopIteration:
            pass
    elif isinstance(obj, (list, tuple)):
        if 0 <= fallback_index < len(obj):
            return np.asarray(obj[fallback_index])
    elif isinstance(obj, np.ndarray):
        if obj.ndim >= 3:  # [N,3,3] or [N,4,4]
            if 0 <= fallback_index < obj.shape[0]:
                return obj[fallback_index]
        elif obj.ndim == 2:
            return obj
    return None

def load_calib_mats(calib_root: Path, all_calib_ts: List[int], cam_ids: List[str]) -> Dict[Tuple[int,str], Tuple[np.ndarray,np.ndarray]]:
    """
    Returns dict keyed by (calib_ts, cam_id) -> (K[3,3], T_wc[4,4])
    """
    cache = {}
    devices_map_cache = {}
    for ts in all_calib_ts:
        cdir = calib_root / str(ts)
        K_path = cdir / "intrinsics.npy"
        E_path = cdir / "extrinsics.npy"
        if not (K_path.exists() and E_path.exists()):
            continue
        K_raw = np.load(K_path, allow_pickle=True)
        E_raw = np.load(E_path, allow_pickle=True)

        # devices index hints
        devs = devices_map_cache.get(ts)
        if devs is None:
            devs = load_devices_index(cdir)  # list of camera ids corresponding to array indices, if any
            devices_map_cache[ts] = devs

        for ci, cam in enumerate(cam_ids):
            # fallback_index: align by devices map if available, else by ci
            idx = ci
            if devs is not None:
                try:
                    idx = devs.index(cam) if cam in devs else devs.index(cam.replace("cam_",""))
                except ValueError:
                    idx = ci
            K = extract_for_cam(K_raw, cam_id=cam.replace("cam_",""), fallback_index=idx)
            E = extract_for_cam(E_raw, cam_id=cam.replace("cam_",""), fallback_index=idx)
            if K is None or E is None:
                continue
            K = np.asarray(K, dtype=np.float32).reshape(3,3)
            E = to_4x4(np.asarray(E, dtype=np.float32))
            cache[(ts, cam)] = (K, E)
    return cache

def collect_episode_timestamps(ep: Path, cam_dirs: List[Path]) -> np.ndarray:
    global_ts = load_timestamps(ep / "timestamps.npy")
    if global_ts is not None and len(global_ts) > 0:
        return global_ts
    # Otherwise union of per-cam timestamps from filenames (color priority)
    ts_set = set()
    for c in cam_dirs:
        for sub in ("color", "depth"):
            mp = list_frames(c / sub)
            ts_set.update(mp.keys())
    ts = np.array(sorted(ts_set), dtype=np.int64)
    return ts

def pack_episode(ep: Path, calib_root: Path, out_dir: Path, query_points_path: Optional[Path] = None, compress: bool = True):
    cam_dirs = sorted([p for p in ep.glob("cam_*") if p.is_dir()])
    cam_ids = [p.name.replace("cam_","") for p in cam_dirs]
    C = len(cam_dirs)
    if C == 0:
        print(f"[WARN] No cameras found in {ep}")
        return

    # Canonical timeline
    ts = collect_episode_timestamps(ep, cam_dirs)  # [T]
    T = len(ts)
    if T == 0:
        print(f"[WARN] No timestamps in {ep}")
        return

    # For each camera, map timestamp->file
    color_maps: List[Dict[int, Path]] = []
    depth_maps: List[Dict[int, Path]] = []
    per_cam_keys: List[List[int]] = []

    for cdir in cam_dirs:
        cmap = list_frames(cdir / "color")
        dmap = list_frames(cdir / "depth")
        color_maps.append(cmap)
        depth_maps.append(dmap)
        per_cam_keys.append(sorted(set(cmap.keys()) | set(dmap.keys())))

    # Precompute forward-fill indices
    cam_indices_for_t: List[np.ndarray] = []
    for keys in per_cam_keys:
        cam_indices_for_t.append(forward_fill_index(keys, ts))

    # Calibration timestamps (dirs named as ints)
    calib_ts = [int(p.name) for p in calib_root.iterdir() if p.is_dir() and p.name.isdigit()]
    calib_ts.sort()
    if len(calib_ts) == 0:
        print(f"[WARN] No calib dirs in {calib_root}; using identity extrinsics and unit intrinsics")
    # Map each frame t -> chosen calib_ts
    chosen_calib_ts = []
    for t in ts:
        chosen_calib_ts.append(pick_key(calib_ts, int(t)) if calib_ts else None)
    chosen_calib_ts = chosen_calib_ts

    # Preload calib matrices
    calib_cache = load_calib_mats(calib_root, sorted(set([x for x in chosen_calib_ts if x is not None])), [f"{x}" for x in cam_ids])

    # Read frames (collect first pass to get max H,W)
    rgb_list: List[List[np.ndarray]] = [[] for _ in range(T)]
    dep_list: List[List[np.ndarray]] = [[] for _ in range(T)]
    intr_list: List[List[np.ndarray]] = [[] for _ in range(T)]
    extr_list: List[List[np.ndarray]] = [[] for _ in range(T)]
    maxH = 0
    maxW = 0

    last_rgb_paths = [None]*C
    last_dep_paths = [None]*C
    last_rgb = [None]*C
    last_dep = [None]*C

    for ti in tqdm(range(T), desc=f"Packing {ep.name}", ncols=100):
        t = int(ts[ti])
        cts = chosen_calib_ts[ti]
        for ci in range(C):
            keys = per_cam_keys[ci]
            if not keys:
                # No frames at all for this camera: create black frame
                rgb = np.zeros((1,1,3), dtype=np.uint8)
                dep = np.zeros((1,1), dtype=np.uint16)
            else:
                idx = cam_indices_for_t[ci][ti]
                k = keys[idx]
                # Resolve color
                cpath = color_maps[ci].get(k)
                if cpath is None and last_rgb[ci] is None:
                    # fallback to any path we can find (first available)
                    cpath = next(iter(color_maps[ci].values()), None)
                if cpath is not None:
                    if last_rgb_paths[ci] != cpath:
                        last_rgb[ci] = read_rgb(cpath)
                        last_rgb_paths[ci] = cpath
                    rgb = last_rgb[ci]
                else:
                    rgb = np.zeros((1,1,3), dtype=np.uint8)

                # Resolve depth
                dpath = depth_maps[ci].get(k)
                if dpath is None and last_dep[ci] is None:
                    dpath = next(iter(depth_maps[ci].values()), None)
                if dpath is not None:
                    if last_dep_paths[ci] != dpath:
                        last_dep[ci] = read_depth(dpath)
                        last_dep_paths[ci] = dpath
                    dep = last_dep[ci]
                else:
                    dep = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint16)

            maxH = max(maxH, rgb.shape[0], dep.shape[0])
            maxW = max(maxW, rgb.shape[1], dep.shape[1])

            # Calib
            if cts is not None and (cts, cam_ids[ci]) in calib_cache:
                K, E = calib_cache[(cts, cam_ids[ci])]
            else:
                K = np.eye(3, dtype=np.float32)
                E = np.eye(4, dtype=np.float32)

            rgb_list[ti].append(rgb)
            dep_list[ti].append(dep)
            intr_list[ti].append(K)
            extr_list[ti].append(E)

    # Pad & stack to fixed tensors (initially HWC / HW)
    rgbs = np.zeros((C, T, maxH, maxW, 3), dtype=np.uint8)
    depths = np.zeros((C, T, maxH, maxW), dtype=np.uint16)
    intrs = np.zeros((C, T, 3, 3), dtype=np.float32)
    extrs = np.zeros((C, T, 3, 4), dtype=np.float32)

    for ti in range(T):
        for ci in range(C):
            rgbs[ci,ti] = pad_to(rgb_list[ti][ci], maxH, maxW, is_rgb=True)
            depths[ci,ti] = pad_to(dep_list[ti][ci], maxH, maxW, is_rgb=False)
            intrs[ci,ti] = intr_list[ti][ci]
            extrs[ci,ti] = extr_list[ti][ci][:3, :]  # Convert 4x4 to 3x4

    # === Layout fix: convert to channels-first for MVTracker-style consumption ===
    # rgbs: (C, T, H, W, 3) -> (C, T, 3, H, W)
    rgbs = np.moveaxis(rgbs, -1, 2)
    # depths: (C, T, H, W) -> (C, T, 1, H, W)
    depths = depths[:, :, None, :, :]

    # Query points
    if query_points_path and Path(query_points_path).exists():
        qp = np.load(query_points_path, allow_pickle=True)
        qp = np.asarray(qp, dtype=np.float32)
        # Convert to [N,4] format with (t,x,y,z)
        if qp.ndim == 2 and qp.shape[-1] == 3:
            # [N,3] -> [N,4] assuming t=0 for all points
            t_col = np.zeros((qp.shape[0], 1), dtype=np.float32)
            qp = np.concatenate([t_col, qp], axis=1)
        elif qp.ndim == 3 and qp.shape[-1] == 3 and qp.shape[0] == T:
            # [T,N,3] -> [T*N,4] with proper time indices
            N = qp.shape[1]
            qp_reshaped = qp.reshape(-1, 3)  # [T*N, 3]
            t_indices = np.repeat(np.arange(T, dtype=np.float32), N).reshape(-1, 1)  # [T*N, 1]
            qp = np.concatenate([t_indices, qp_reshaped], axis=1)  # [T*N, 4]
        elif qp.ndim == 2 and qp.shape[-1] == 4:
            # Already in [N,4] format
            pass
        else:
            raise ValueError("--query-points must be [T,N,3], [N,3], or [N,4].")
    else:
        qp = np.zeros((0,4), dtype=np.float32)

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ep.name}.npz"

    save_fn = np.savez_compressed if compress else np.savez
    save_fn(
        out_path,
        rgbs=rgbs,
        depths=depths,
        intrs=intrs,
        extrs=extrs,
        query_points=qp,
        timestamps=ts.astype(np.int64),
        camera_ids=np.array(cam_ids, dtype=object),
        episode=np.array(ep.name),
    )
    print(f"[OK] Wrote {out_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="RH20T_cfg* root containing episodes and calib/")
    ap.add_argument("--out", required=True, help="Output directory for .npz files")
    ap.add_argument("--episodes-glob", default="task_*", help="Glob under root to find episodes")
    ap.add_argument("--only-episode", default=None, help="If set, pack only this episode folder name")
    ap.add_argument("--query-points", default=None, help="Optional .npy/.npz path -> [T,N,3] or [N,3]")
    ap.add_argument("--compress", action="store_true", help="Use np.savez_compressed")
    args = ap.parse_args()

    root = Path(args.root)
    calib_root = root / "calib"
    out_dir = Path(args.out)

    episodes = []
    if args.only_episode:
        p = root / args.only_episode
        if p.is_dir():
            episodes = [p]
    else:
        episodes = [p for p in root.glob(args.episodes_glob) if p.is_dir() and p.name.startswith("task_")]

    if not episodes:
        print(f"[WARN] No episodes found in {root} with pattern '{args.episodes_glob}'")
        return

    for ep in episodes:
        try:
            pack_episode(ep, calib_root=calib_root, out_dir=out_dir,
                         query_points_path=Path(args.query_points) if args.query_points else None,
                         compress=args.compress)
        except Exception as e:
            print(f"[ERROR] {ep.name}: {e}")

if __name__ == "__main__":
    main()
