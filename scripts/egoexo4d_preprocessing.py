"""
Environment setup:
```bash
cd ..

# Clone the projectaria_tools repository
git clone -b 1.5.0 https://github.com/facebookresearch/projectaria_tools
cd projectaria_tools/

# Install required libraries using Conda
conda install -c conda-forge cmake fmt xxhash libjpeg-turbo gcc_linux-64 gxx_linux-64
conda install -c conda-forge boost-cpp=1.82.0 boost=1.82.0

# Set compiler environment variables
export BOOST_ROOT=$CONDA_PREFIX
export BOOST_INCLUDEDIR=$CONDA_PREFIX/include
export BOOST_LIBRARYDIR=$CONDA_PREFIX/lib

# Clean previous builds and install projectaria_tools
rm -rf build/ dist/ *.egg-info
cmake -S . -B build \
  -DBOOST_ROOT=$BOOST_ROOT \
  -DBoost_NO_SYSTEM_PATHS=ON \
  -DBoost_INCLUDE_DIR=$BOOST_INCLUDEDIR \
  -DBoost_LIBRARY_DIR=$BOOST_LIBRARYDIR \
  -DBUILD_PYTHON_BINDINGS=ON
cmake --build build -j
pip install .

# Additional packages (if required)
pip install av

cd ../mvtracker
```

Download a subset of the data:
```bash
# Install CLI for downloading the data
pip install ego4d --upgrade

# Get an access id and key after filling a form at https://ego4ddataset.com/egoexo-license/
...

# Install AWS CLI from https://aws.amazon.com/cli/ (assuming no sudo)
cd ..
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
./aws/install -i ~/local/aws-cli -b ~/local/bin
# Add to ~/.bashrc: export PATH=$HOME/.local/bin:$PATH
source ~/.bashrc
aws --version
# aws-cli/2.27.49 Python/3.13.4 Linux/6.8.0-57-generic exe/x86_64.ubuntu.24
aws configure
# Now you can enter the access id and key...

# Download a small subset of the data (around 100 GB)
egoexo -o ./datasets/egoexo4d --parts metadata
egoexo -o ./datasets/egoexo4d --parts take_trajectory take_vrs_noimagestream captures annotations metadata takes downscaled_takes/448 --uids ed3ec638-8363-4e1d-9851-c7936cbfad8c 51fc36b3-e769-4617-b087-3826b280cad3 f179e1a2-3265-464a-a106-a08c30d0a2ae 43dca3b5-21d9-4ebf-856e-515a5c417699 c3915dd7-3ac0-40b7-a69b-73b7326bd15c e08856e4-a1c7-4e36-96b6-a233efb27bfd 425d8f94-ed65-49d5-86e7-174f555fda5d ed698f62-ccdb-4601-8a0a-ee89a0a7e1c0 4e5aa06a-7a60-4e23-9853-d55260a9e6e9 001ae9a5-9c8a-4710-9f7f-7dc67597a02f 2aaaca24-108a-437e-ab9a-bc3e8d65fcdf 503cc92d-7052-44ff-a21d-da6c4a5d6927 f32dc6d9-0eb8-4c85-8ab9-7d47b8c4c660 2423c2ff-c85d-4998-afab-29de8d26d263 0e5d13c6-87ba-4c9b-ab2f-1aaac4e0aacb 1a9a21ab-9023-402f-ac64-df08feaabb5b 811ad284-702f-4d38-af99-a2c4006fa298 a261cc1d-7a45-479f-81a9-7c73eb379e6c c2fb62e3-8894-4101-9923-5eedeb1b4282
egoexo -o ./datasets/egoexo4d --parts captures
egoexo -o ./datasets/egoexo4d --parts annotations --benchmarks egopose

```

Running the script: `PYTHONPATH=/local/home/frrajic/xode/duster:$PYTHONPATH python -m scripts.egoexo4d_preprocessing`
Note that you need to set up dust3r first, see docstring of `scripts/estimate_depth_with_duster.py`.
"""
import json
import os
import pickle
import time
from typing import Optional

import av
import cv2
import math
import numpy as np
import pandas as pd
import rerun as rr
import torch
# from projectaria_tools.core import calibration
# from projectaria_tools.core import data_provider
# from projectaria_tools.core import mps
# from projectaria_tools.core.calibration import CameraCalibration, KANNALA_BRANDT_K3
# from projectaria_tools.core.stream_id import StreamId
from tqdm import tqdm

from scripts.estimate_depth_with_duster import run_duster


def main_preprocess_egoexo4d(
        release_dir: str,
        take_name: str,
        outputs_dir: str,
        max_frames: Optional[int] = None,
        frames_downsampling_factor: Optional[int] = None,
        downscaled_longerside: Optional[int] = None,
        save_rerun_viz: bool = True,
        stream_rerun_viz: bool = False,
        skip_if_output_exists: bool = True,
):
    # Skip if output exists
    save_pkl_path = os.path.join(outputs_dir, f"{take_name}.pkl")
    if skip_if_output_exists and os.path.exists(save_pkl_path):
        print(f"Skipping {save_pkl_path} since it already exists")
        print()
        return
    else:
        print(f"Processing {take_name}...")

    # Load necessary metadata files
    egoexo = {
        "takes": os.path.join(release_dir, "takes.json"),
        "captures": os.path.join(release_dir, "captures.json")
    }
    for k, v in egoexo.items():
        egoexo[k] = json.load(open(v))
    takes = egoexo["takes"]
    captures = egoexo["captures"]
    takes_by_name = {x["take_name"]: x for x in takes}

    # Take the take
    take = takes_by_name[take_name]

    # Initialize exo cameras from calibration file
    traj_dir = os.path.join(release_dir, take["root_dir"], "trajectory")
    exo_traj_path = os.path.join(traj_dir, "gopro_calibs.csv")

    exo_traj_df = pd.read_csv(exo_traj_path)
    exo_cam_names = list(exo_traj_df["cam_uid"])
    ego_cam_names = [x["cam_id"] for x in take["capture"]["cameras"] if x["is_ego"] and x["cam_id"].startswith("aria")]
    all_cams = ego_cam_names + exo_cam_names
    ego_cam_name = ego_cam_names[0]
    print("exo cameras: ", exo_cam_names)
    print(" ego camera: ", ego_cam_name)

    go_pro_proxy = {}
    static_calibrations = mps.read_static_camera_calibrations(exo_traj_path)
    for static_calibration in static_calibrations:
        # assert the GoPro was correctly localized
        if static_calibration.quality != 1.0:
            print(f"Camera: {static_calibration.camera_uid} was not localized, ignoring this camera.")
            continue
        proxy = {}
        proxy["name"] = static_calibration.camera_uid
        proxy["pose"] = static_calibration.transform_world_cam
        proxy["camera"] = CameraCalibration(
            static_calibration.camera_uid,
            KANNALA_BRANDT_K3,
            static_calibration.intrinsics,
            static_calibration.transform_world_cam,  # probably extrinsics
            static_calibration.width,
            static_calibration.height,
            None,
            math.pi,
            "")

        go_pro_proxy[static_calibration.camera_uid] = proxy

    # Configure the VRSDataProvider (interface used to retrieve Trajectory data)
    ego_exo_project_path = os.path.join(release_dir, 'takes', take['take_name'])

    aria_dir = os.path.join(release_dir, take["root_dir"])
    aria_path = os.path.join(aria_dir, f"{ego_cam_name}.vrs")
    vrs_data_provider = data_provider.create_vrs_data_provider(aria_path)
    device_calibration = vrs_data_provider.get_device_calibration()

    ego_stream_name = "214-1"
    rgb_stream_id = StreamId(ego_stream_name)
    rgb_stream_label = vrs_data_provider.get_label_from_stream_id(rgb_stream_id)
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)

    mps_data_paths_provider = mps.MpsDataPathsProvider(ego_exo_project_path)
    mps_data_paths = mps_data_paths_provider.get_data_paths()
    mps_data_provider = mps.MpsDataProvider(mps_data_paths)

    # Extract ego extrinsics
    capture_name = take["capture"]["capture_name"]
    timesync = pd.read_csv(os.path.join(release_dir, f"captures/{capture_name}/timesync.csv"))

    start_idx = take["timesync_start_idx"] + 1
    end_idx = take["timesync_end_idx"]
    take_timestamps = []
    for idx in range(start_idx, end_idx):
        ts = timesync.iloc[idx][f"{ego_cam_name}_{ego_stream_name}_capture_timestamp_ns"]
        take_timestamps.append(ts)
    if frames_downsampling_factor is not None:
        take_timestamps = take_timestamps[::frames_downsampling_factor]
    if max_frames is not None:
        take_timestamps = take_timestamps[:max_frames]
    valid_frames = np.array([not np.isnan(ts) for ts in take_timestamps])
    if not valid_frames.all():
        print(f"Number of invalid frames (with nan ego timesync): {(~valid_frames).sum()}")
    take_timestamps = np.array(take_timestamps)[valid_frames].astype(int)
    ego_closed_loop_poses = [mps_data_provider.get_closed_loop_pose(t) for t in take_timestamps]

    ego_extrs = []
    T_device_camera = rgb_camera_calibration.get_transform_device_camera()
    for pose in ego_closed_loop_poses:
        assert pose is not None
        T_world_device = pose.transform_world_device
        T_world_camera = T_world_device @ T_device_camera
        extrinsic_matrix = T_world_camera.inverse().to_matrix()[:3, :]

        # Rotate camera 90° clockwise around Z
        R_z_90 = np.array([
            [0, -1, 0],
            [1, 0, 0],
            [0, 0, 1]
        ])
        extrinsic_matrix[:3, :] = R_z_90 @ extrinsic_matrix[:3, :]

        ego_extrs.append(extrinsic_matrix)

    # Extract videos
    base_directory = os.path.join(release_dir, take["root_dir"])
    videos = {}
    for cam_name in all_cams:
        if cam_name in exo_cam_names:
            stream_name = '0'
        else:
            stream_name = 'rgb'

        local_path = os.path.join(base_directory, take['frame_aligned_videos'][cam_name][stream_name]['relative_path'])
        container = av.open(local_path)

        frames = []
        for frame_idx, frame in enumerate(tqdm(container.decode(video=0))):
            if frame_idx % frames_downsampling_factor != 0:
                continue
            if max_frames is not None and len(frames) >= max_frames:
                break
            frames.append(np.array(frame.to_image()))
        frames = np.stack(frames)[valid_frames]
        videos[cam_name] = frames

    # Undistorted videos
    rgbs = {}
    intrs = {}
    extrs = {}
    for cam_name in all_cams:
        frames = videos[cam_name]
        h, w = frames[0].shape[:2]

        if cam_name in exo_cam_names:
            calib = exo_traj_df[exo_traj_df.cam_uid == cam_name].iloc[0].to_dict()
            D = np.array([calib[f"intrinsics_{i}"] for i in range(4, 8)])
            K = np.array([
                [calib["intrinsics_0"], 0, calib["intrinsics_2"]],
                [0, calib["intrinsics_1"], calib["intrinsics_3"]],
                [0, 0, 1]
            ])
            width, height = calib["image_width"], calib["image_height"]
            scaled_K = K * w / width
            scaled_K[2][2] = 1.0

            new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, (w, h), np.eye(3), balance=0.0)
            map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, (w, h), cv2.CV_16SC2)
            undistorted = []
            for img in tqdm(frames, desc=f"Undistorting {cam_name}"):
                ud = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR)
                undistorted.append(ud)

            intrs[cam_name] = new_K
            extrs[cam_name] = go_pro_proxy[cam_name]["pose"].inverse().to_matrix()[:3, :]
            rgbs[cam_name] = np.stack([f.transpose(2, 0, 1) for f in undistorted])

        else:
            src_calib = rgb_camera_calibration
            dst_calib = calibration.get_linear_camera_calibration(w, h, 450)

            fx, fy = dst_calib.get_focal_lengths()
            cx, cy = dst_calib.get_principal_point()
            K = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

            undistorted = []
            for img in tqdm(frames, desc=f"Undistorting {cam_name}"):
                img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
                ud = calibration.distort_by_calibration(img, dst_calib, src_calib)
                ud = cv2.rotate(ud, cv2.ROTATE_90_CLOCKWISE)
                undistorted.append(ud)
            undistorted = [ud.transpose(2, 0, 1) for ud in undistorted]

            intrs[cam_name] = K
            extrs[cam_name] = np.stack(ego_extrs)
            rgbs[cam_name] = np.stack(undistorted)

    # Check shapes
    n_frames, _, h_exo, w_exo = rgbs[exo_cam_names[0]].shape
    _, _, h_ego, w_ego = rgbs[ego_cam_name].shape
    for cam_name in all_cams:
        if cam_name in exo_cam_names:
            assert rgbs[cam_name].shape == (n_frames, 3, h_exo, w_exo)
            assert intrs[cam_name].shape == (3, 3)
            assert extrs[cam_name].shape == (3, 4)
        else:
            assert rgbs[cam_name].shape == (n_frames, 3, h_ego, w_ego)
            assert intrs[cam_name].shape == (3, 3)
            assert extrs[cam_name].shape == (n_frames, 3, 4)

    # Save downsized version
    if downscaled_longerside is not None:
        print(f"Downscaling to longer side {downscaled_longerside}")
        for cam_name in rgbs:
            _, _, h, w = rgbs[cam_name].shape
            scale = downscaled_longerside / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            resized = []
            for img in rgbs[cam_name]:
                img = img.transpose(1, 2, 0)  # CHW -> HWC
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                resized.append(img.transpose(2, 0, 1))  # HWC -> CHW
            rgbs[cam_name] = np.stack(resized)

            # scale intrinsics
            intrs[cam_name][:2] *= scale

    # Save processed output to a pickle file
    os.makedirs(outputs_dir, exist_ok=True)
    with open(save_pkl_path, "wb") as f:
        pickle.dump(
            dict(
                rgbs=rgbs,
                intrs=intrs,
                extrs=extrs,
                ego_cam_name=ego_cam_name,
            ),
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {save_pkl_path}")

    # Visualize the data sample using rerun
    rerun_modes = []
    if stream_rerun_viz:
        rerun_modes += ["stream"]
    if save_rerun_viz:
        rerun_modes += ["save"]
    for rerun_mode in rerun_modes:
        rr.init(f"3dpt", recording_id="v0.16")
        if rerun_mode == "stream":
            rr.connect_tcp()

        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.set_time_seconds("frame", 0)
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )

        fps = 30
        for frame_idx in range(n_frames):
            rr.set_time_seconds("frame", frame_idx / fps)

            for cam_name in all_cams:
                extr = extrs[cam_name] if cam_name in exo_cam_names else extrs[cam_name][frame_idx]
                intr = intrs[cam_name]
                img = rgbs[cam_name][frame_idx].transpose(1, 2, 0).astype(np.uint8)

                # Camera pose logging
                E = extr if extr.shape == (3, 4) else extr[0]
                T = np.eye(4)
                T[:3, :] = E
                T_world_cam = np.linalg.inv(T)
                rr.log(f"{cam_name}/image", rr.Transform3D(
                    translation=T_world_cam[:3, 3],
                    mat3x3=T_world_cam[:3, :3],
                ))

                # Intrinsics and image
                rr.log(f"{cam_name}/image", rr.Pinhole(
                    image_from_camera=intr,
                    width=img.shape[1],
                    height=img.shape[0]
                ))
                rr.log(f"{cam_name}/image", rr.Image(img))

        if rerun_mode == "save":
            save_rrd_path = os.path.join(outputs_dir, f"rerun__{take_name}.rrd")
            rr.save(save_rrd_path)
            print(f"Saved rerun viz to {os.path.abspath(save_rrd_path)}")


def main_estimate_duster_depth(
        pkl_scene_file,
        depths_output_dir,
        save_rerun_viz=False,
        skip_if_output_already_exists=True,
):
    duster_kwargs = {
        "model_name_or_path": "/workspace/duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "silent": True,
        "output_2d_matches": False,
        "dump_exhaustive_data": False,
        "save_ply": False,
        "save_png_viz": False,
        "show_debug_plots": False,
        "save_rerun_viz": save_rerun_viz,
        "skip_if_output_already_exists": skip_if_output_already_exists,
    }

    print(f"Generating DUSt3R depths to {os.path.abspath(depths_output_dir)}")
    assert os.path.exists(pkl_scene_file)
    with open(pkl_scene_file, "rb") as f:
        scene = pickle.load(f)

    rgbs = scene["rgbs"]
    intrs = scene["intrs"]
    extrs = scene["extrs"]
    ego_cam_name = scene["ego_cam_name"]
    exo_cam_names = sorted([cam_name for cam_name in rgbs.keys() if cam_name != ego_cam_name])

    n_frames, _, h, w = rgbs[exo_cam_names[0]].shape

    fx, fy, cx, cy, extrinsics = [], [], [], [], []
    for cam_name in exo_cam_names:
        intrinsics = intrs[cam_name]
        extrinsics_view = np.eye(4)
        extrinsics_view[:3, :4] = extrs[cam_name]

        assert np.isclose(intrinsics[0, 1], 0)
        assert np.isclose(intrinsics[1, 0], 0)
        assert np.isclose(intrinsics[2, 0], 0)
        assert np.isclose(intrinsics[2, 1], 0)
        assert np.isclose(intrinsics[2, 2], 1)

        fx.append(intrinsics[0, 0])
        fy.append(intrinsics[1, 1])
        cx.append(intrinsics[0, 2])
        cy.append(intrinsics[1, 2])
        extrinsics.append(extrinsics_view)

    fx = torch.tensor(fx).float()
    fy = torch.tensor(fy).float()
    cx = torch.tensor(cx).float()
    cy = torch.tensor(cy).float()
    extrinsics = torch.from_numpy(np.stack(extrinsics)).float()

    start = time.time()
    images_tensor = torch.from_numpy(np.stack([rgbs[cam_name] for cam_name in exo_cam_names]))
    run_duster(images_tensor, depths_output_dir, fx, fy, cx, cy, extrinsics, **duster_kwargs)
    time_elapsed = time.time() - start
    print(f"Time elapsed for DUST3R: {time_elapsed:.2f} seconds")


if __name__ == '__main__':
    release_dir = "datasets/egoexo4d/"
    outputs_dir = "datasets/egoexo4d-processed/"

    num_devices = 1
    device_id = int(os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    device_id = device_id % num_devices
    print(f"Device ID: {device_id} (out of {num_devices}). The devices split the work.")

    for i, take_name in enumerate([
        "fair_cooking_06_4",  # take_uid = "a261cc1d-7a45-479f-81a9-7c73eb379e6c"
        "cmu_bike01_2",  # take_uid = "ed3ec638-8363-4e1d-9851-c7936cbfad8c"
        "georgiatech_cooking_01_01_2",  # take_uid = "51fc36b3-e769-4617-b087-3826b280cad3"
        "iiith_cooking_49_2",  # take_uid = "f179e1a2-3265-464a-a106-a08c30d0a2ae"
        "indiana_bike_12_5",  # take_uid = "43dca3b5-21d9-4ebf-856e-515a5c417699"
        "minnesota_rockclimbing_033_20",  # take_uid = "c3915dd7-3ac0-40b7-a69b-73b7326bd15c"
        "sfu_basketball_09_21",  # take_uid = "425d8f94-ed65-49d5-86e7-174f555fda5d"
        "unc_basketball_03-09-23_02_11",  # take_uid = "ed698f62-ccdb-4601-8a0a-ee89a0a7e1c0"
        "unc_music_04-26-23_02_7",  # take_uid = "4e5aa06a-7a60-4e23-9853-d55260a9e6e9"
        "uniandes_dance_017_57",  # take_uid = "0e5d13c6-87ba-4c9b-ab2f-1aaac4e0aacb"
        "upenn_0331_Guitar_2_4",  # take_uid = "1a9a21ab-9023-402f-ac64-df08feaabb5b"
        "unc_basketball_02-24-23_01_12",  # take_uid = "c2fb62e3-8894-4101-9923-5eedeb1b4282"
    ]):
        if i % num_devices != device_id:
            continue

        for max_frames, frames_downsampling_factor, downscaled_longerside in [(300, 1, 512), (300, 1, 518)]:
            # Extract rgbs, intrs, extrs from EgoExo4D dataset
            outputs_subdir = os.path.join(outputs_dir, f"maxframes-{max_frames}_"
                                                       f"downsample-{frames_downsampling_factor}_"
                                                       f"downscale-{downscaled_longerside}")
            main_preprocess_egoexo4d(release_dir, take_name, outputs_subdir,
                                     max_frames, frames_downsampling_factor, downscaled_longerside)

            # Run Dust3r to estimate depths from rgbs, fix the known intrs and extrs during multi-view stereo optim
            take_pkl = os.path.join(outputs_subdir, f"{take_name}.pkl")
            depth_subdir = os.path.join(outputs_subdir, f"duster_depths__{take_name}")
            main_estimate_duster_depth(
                pkl_scene_file=take_pkl,
                depths_output_dir=depth_subdir,
            )

            # Run VGGT to estimate depths from rgbs, align with the known extrs afterward
            ...