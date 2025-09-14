"""
Set up the environment:
```sh
cd /local/home/frrajic/xode

git clone --recursive git@github.com:ethz-vlg/duster.git
cd duster

# Fix models path, since there are two in the project
sed -i 's/from models/from croco.models/g' croco/*.py
sed -i 's/from models/from croco.models/g' croco/*/*.py
sed -i 's/from models/from croco.models/g' dust3r/*.py
sed -i 's/from models/from croco.models/g' dust3r/*/*.py

# Download the checkpoint
wget https://download.europe.naverlabs.com/ComputerVision/DUSt3R/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth -P checkpoints
md5sum checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth
# c3fab9b455b03f23d20e6bf77f2607bb  checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth

# You should be able to use the same environment as for
# the rest of the project, just install missing packages:
pip install roma==1.5.1
```
Running the script:
```sh
cd /local/home/frrajic/xode/mvtracker
export PYTHONPATH=/local/home/frrajic/xode/duster:$PYTHONPATH

python scripts/estimate_depth_with_duster.py --dataset dexycb
python scripts/estimate_depth_with_duster.py --dataset kubric-val
python scripts/estimate_depth_with_duster.py --dataset kubric-train
```

Running the script on Panoptic Sports from Dynamic 3DGS:
```sh
# Download the data
cd datasets
wget https://omnomnom.vision.rwth-aachen.de/data/Dynamic3DGaussians/data.zip
unzip data.zip
mv data panoptic_d3dgs
cd -

# Run the script
cd /local/home/frrajic/xode/duster/mvtracker
export PYTHONPATH=/local/home/frrajic/xode/duster:$PYTHONPATH

python scripts/estimate_depth_with_duster.py --dataset panoptic_d3dgs
```
"""
import argparse
import json
import os
import random
import time
import warnings
from copy import deepcopy
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import rerun as rr
import torch
import torch.nn.functional as F
import trimesh
from PIL import Image
from PIL.ImageOps import exif_transpose

from dust3r.cloud_opt import PointCloudOptimizer
from dust3r.image_pairs import make_pairs
from dust3r.inference import inference
from dust3r.model import AsymmetricCroCo3DStereo
from dust3r.utils.device import to_numpy
from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
from dust3r.utils.image import load_images
from dust3r.utils.image import rgb, heif_support_enabled, _resize_pil_image, ImgNorm
from mvtracker.datasets import KubricMultiViewDataset

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12


def seed_all(seed):
    """
    Seed all random number generators.

    Parameters
    ----------
    seed : int
        The seed to use.

    Returns
    -------
    None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_view_visibility(scene, pts):
    vis = np.zeros((len(scene.imgs), len(pts)), dtype=bool)
    poses = scene.get_im_poses().detach().cpu().numpy()
    extrinsics = np.linalg.inv(poses)
    focals = scene.get_focals().squeeze(-1).detach().cpu().numpy()
    pps = scene.get_principal_points().detach().cpu().numpy()
    depths = [d.detach().cpu().numpy() for d in scene.get_depthmaps(raw=False)]

    # Apply masks to the depthmaps as to not consider points that have low confidence
    per_view_masks = [m.detach().cpu().numpy() for m in scene.get_masks()]
    for view_idx, mask in enumerate(per_view_masks):
        depths[view_idx] = depths[view_idx] * mask

    for view_idx in range(len(scene.imgs)):
        p_world = pts
        p_world = np.concatenate([p_world, np.ones((len(p_world), 1))], axis=1)
        p_cam = extrinsics[view_idx] @ p_world.T
        z = p_cam[2]
        x = p_cam[0, :] / z[:] * focals[view_idx, 0] + pps[view_idx, 0]
        y = p_cam[1, :] / z[:] * focals[view_idx, 1] + pps[view_idx, 1]
        x_floor = np.floor(x).astype(int)
        y_floor = np.floor(y).astype(int)
        x_ceil = np.ceil(x).astype(int)
        y_ceil = np.ceil(y).astype(int)
        h, w = depths[view_idx].shape[:2]
        out_of_view = (
                (x_floor < 0)
                | (x_ceil >= w)
                | (y_floor < 0)
                | (y_ceil >= h)
                | (z < 0)
        )
        z_from_depthmap_1 = depths[view_idx][y_floor[~out_of_view], x_floor[~out_of_view]]
        z_from_depthmap_2 = depths[view_idx][y_floor[~out_of_view], x_ceil[~out_of_view]]
        z_from_depthmap_3 = depths[view_idx][y_ceil[~out_of_view], x_floor[~out_of_view]]
        z_from_depthmap_4 = depths[view_idx][y_ceil[~out_of_view], x_ceil[~out_of_view]]
        z_from_depthmap = np.stack([z_from_depthmap_1, z_from_depthmap_2, z_from_depthmap_3, z_from_depthmap_4], axis=0)
        vis[view_idx] = ~out_of_view
        vis[view_idx][~out_of_view] = np.isclose(z[~out_of_view], z_from_depthmap.min(axis=0), rtol=0.001, atol=0.1)

        # import pandas as pd
        # x = pd.Series(np.abs(z[~out_of_view] - z_from_depthmap.min(axis=0)))
        # quantiles_to_print = [0.001, 0.01, 0.05, 0.1, 0.5, 0.9, 0.95, 0.99, 0.999]
        # print(f"Quantiles of the difference between the depthmap and the z coordinate of the point in the camera frame")
        # for q in quantiles_to_print:
        #     print(f"{q=}: {x.quantile(q)}")

    return vis


def get_3D_model_from_scene(
        output_file_prefix,
        silent,
        scene,
        min_conf_thr=3,
        mask_sky=False,
        clean_depth=False,
        feats=None,

        dump_exhaustive_data=False,
        save_ply=False,
        save_png_viz=False,
        save_rerun_viz=False,
        rerun_radii=0.01,
        rerun_viz_timestamp=0,
):
    scene = deepcopy(scene)
    if clean_depth:
        scene = scene.clean_pointcloud()
    if mask_sky:
        scene = scene.mask_sky()

    rgbimg = scene.imgs
    pts3d = to_numpy(scene.get_pts3d())
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    msk = to_numpy(scene.get_masks())

    if not silent:
        print(f'Exporting 3D scene to prefix={output_file_prefix}')

    assert len(pts3d) == len(msk) <= len(rgbimg)
    pts3d = to_numpy(pts3d)
    pts3d_view_idx = [view_idx * np.ones_like(p[:, :, 0]) for view_idx, p in enumerate(pts3d)]
    imgs = to_numpy(rgbimg)

    pts_view_idx = np.concatenate([pvi[m] for pvi, m in zip(pts3d_view_idx, msk)])
    pts = np.concatenate([p[m] for p, m in zip(pts3d, msk)])
    col = np.concatenate([p[m] for p, m in zip(imgs, msk)])
    # get_view_visibility(scene, np.stack(pts3d).reshape(-1, 3)[:10], np.stack(pts3d_view_idx).reshape(-1)[:10])  # debug
    vis = get_view_visibility(scene, pts)

    msk = np.stack([m for m in msk])

    depths = to_numpy(scene.get_depthmaps())
    depths = np.stack([d for d in depths])

    confs = to_numpy([c for c in scene.im_conf])
    confs = np.stack([c for c in confs])

    output_dict = {
        "depths": depths,
        "confs": confs,
        "cleaned_mask": msk,
        "min_conf_thr": min_conf_thr,
        "mask_sky": mask_sky,
        "clean_depth": clean_depth,
    }
    if dump_exhaustive_data:
        output_dict.update({
            "pts": pts,
            "pts_view": pts_view_idx,
            "col": col,
            "vis": vis,
            "rgbs": imgs,
        })
    if feats is not None:
        output_dict["feats"] = feats
    np.savez(f"{output_file_prefix}__scene.npz", **output_dict)

    if save_ply:
        pcd = trimesh.PointCloud(vertices=pts, colors=col)
        pcd.export(f"{output_file_prefix}__pc.ply")

    if rerun_viz_timestamp == 0:
        init_pt_cld = np.concatenate([pts, col, np.ones_like(pts[:, :1])], axis=1)
        np.savez(f"{output_file_prefix}__init_pt_cld.npz", data=init_pt_cld)

    if save_png_viz:
        # Results visualization
        rgbimg = scene.imgs
        cmap = plt.get_cmap('jet')
        depths_max = max([d.max() for d in depths])
        depths_viz = [d / depths_max for d in depths]
        confs_max = max([d.max() for d in confs])
        confs_viz = [cmap(d / confs_max) for d in confs]
        assert len(rgbimg) == len(depths_viz) == len(confs)
        H, W = rgbimg[0].shape[:2]
        N = len(rgbimg)
        plt.figure(dpi=100, figsize=(4 * W / 100, N * H / 100))
        for i in range(N):
            a = rgbimg[i]
            b = rgb(depths_viz[i])
            c = rgb(confs_viz[i])
            d = rgb(msk[i])
            plt.subplot(N, 4, 1 + 4 * i)
            plt.imshow(a)
            plt.axis('off')
            plt.subplot(N, 4, 2 + 4 * i)
            plt.imshow(b)
            plt.axis('off')
            plt.subplot(N, 4, 3 + 4 * i)
            plt.imshow(c)
            plt.axis('off')
            plt.subplot(N, 4, 4 + 4 * i)
            plt.imshow(d)
            plt.axis('off')
        plt.tight_layout(pad=0)
        plt.savefig(f"{output_file_prefix}__viz.png")
        plt.close()

    if save_rerun_viz:
        rr.init("reconstruction", recording_id="v0.1")
        # rr.connect_tcp()
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.set_time_seconds("frame", 0)
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        rr.set_time_seconds("frame", rerun_viz_timestamp / 30)
        for v in range(len(rgbimg)):
            h, w = scene.imshape
            fx, fy = scene.get_focals().cpu().numpy()[v]
            cx, cy = scene.get_principal_points().cpu().numpy()[v]
            K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
            c2w = scene.get_im_poses().cpu().numpy()[v]
            rr.log(f"image/view-{v}/rgb", rr.Image(scene.imgs[v]))
            rr.log(f"image/view-{v}/depth", rr.DepthImage(depths[v], point_fill_ratio=0.2))
            rr.log(f"image/view-{v}", rr.Pinhole(image_from_camera=K, width=w, height=h))
            rr.log(f"image/view-{v}", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))
            rr.log(f"point_cloud/duster-cleaned/view-{v}", rr.Points3D(pts, colors=col, radii=rerun_radii))
            rr.log(f"point_cloud/duster-raw/view-{v}", rr.Points3D(positions=np.stack(pts3d).reshape(-1, 3),
                                                                   colors=np.stack(imgs).reshape(-1, 3),
                                                                   radii=rerun_radii))
        rr_rrd_path = f"{output_file_prefix}__rerun_viz.rrd"
        rr.save(rr_rrd_path)
        print(f"Saved Rerun recording to: {os.path.abspath(rr_rrd_path)}")


def get_2D_matches(output_file_prefix, scene, input_views, min_conf_thr, clean_depth, viz_matches=False):
    scene = deepcopy(scene)
    scene.min_conf_thr = float(scene.conf_trf(torch.tensor(min_conf_thr)))
    if clean_depth:
        scene = scene.clean_pointcloud()

    # retrieve useful values from scene:
    imgs = scene.imgs
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    pts2d_list, pts3d_list = {}, {}
    for view_i in range(len(input_views)):
        conf_i = confidence_masks[view_i].cpu().numpy()
        pts2d_list[view_i] = xy_grid(*imgs[view_i].shape[:2][::-1])[conf_i]  # imgs[i].shape[:2] = (H, W)
        pts3d_list[view_i] = pts3d[view_i].detach().cpu().numpy()[conf_i]

    matches = {}
    for view_i in range(len(input_views) - 1):
        for view_j in range(view_i + 1, len(input_views)):

            # find 2D-2D matches between the two images
            reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(pts3d_list[view_i], pts3d_list[view_j])
            assert num_matches == reciprocal_in_P2.sum()
            print(f'view_{view_i}-view_{view_j}: {num_matches} matches')
            matches_i_xy = pts2d_list[view_i][nn2_in_P1][reciprocal_in_P2]
            matches_j_xy = pts2d_list[view_j][reciprocal_in_P2]
            matches_i_xyz = pts3d_list[view_i][nn2_in_P1][reciprocal_in_P2]
            matches_j_xyz = pts3d_list[view_j][reciprocal_in_P2]
            assert len(matches_i_xy) == len(matches_j_xy) == len(matches_i_xyz) == len(matches_j_xyz) == num_matches

            # store the matches
            matches[(view_i, view_j)] = {
                'matches_i_xy': matches_i_xy,
                'matches_j_xy': matches_j_xy,
                'matches_i_xyz': matches_i_xyz,
                'matches_j_xyz': matches_j_xyz,
            }

            # visualize a few matches
            if viz_matches:
                n_viz = 18
                match_idx_to_viz = np.round(np.linspace(0, num_matches - 1, n_viz)).astype(int)
                viz_matches_im0, viz_matches_im1 = matches_i_xy[match_idx_to_viz], matches_j_xy[match_idx_to_viz]
                H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
                img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
                img = np.concatenate((img0, img1), axis=1)
                plt.figure(dpi=200)
                plt.imshow(img)
                cmap = plt.get_cmap('jet')
                for i in range(n_viz):
                    (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
                    plt.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
                plt.savefig(f"{output_file_prefix}__matches__v{view_i}-v{view_j}.png")
                plt.tight_layout(pad=0)
                plt.close()

    # save the matches
    np.savez(f"{output_file_prefix}__matches.npz", matches=matches)


def load_images(folder_or_list, size, square_ok=False, verbose=True):
    """ open and convert all images in a list or folder to proper input format for DUSt3R
    """
    if isinstance(folder_or_list, str):
        if verbose:
            print(f'>> Loading images from {folder_or_list}')
        root, folder_content = folder_or_list, sorted(os.listdir(folder_or_list))

    elif isinstance(folder_or_list, list):
        if verbose:
            print(f'>> Loading a list of {len(folder_or_list)} images')
        root, folder_content = '', folder_or_list

    else:
        raise ValueError(f'bad {folder_or_list=} ({type(folder_or_list)})')

    supported_images_extensions = ['.jpg', '.jpeg', '.png']
    if heif_support_enabled:
        supported_images_extensions += ['.heic', '.heif']
    supported_images_extensions = tuple(supported_images_extensions)

    imgs = []
    for path in folder_content:
        if not path.lower().endswith(supported_images_extensions):
            continue
        img = exif_transpose(Image.open(os.path.join(root, path))).convert('RGB')
        W1, H1 = img.size
        if size == 224:
            # resize short side to 224 (then crop)
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            # resize long side to 512
            img = _resize_pil_image(img, size)

        # W, H = img.size
        # cx, cy = W // 2, H // 2
        # if size == 224:
        #     half = min(cx, cy)
        #     img = img.crop((cx - half, cy - half, cx + half, cy + half))
        # else:
        #     halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
        #     if not (square_ok) and W == H:
        #         halfh = 3 * halfw / 4
        #     img = img.crop((cx - halfw, cy - halfh, cx + halfw, cy + halfh))

        W2, H2 = img.size
        if verbose:
            print(f' - adding {path} with resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(img=ImgNorm(img)[None], true_shape=np.int32(
            [img.size[::-1]]), idx=len(imgs), instance=str(len(imgs))))

    assert imgs, 'no images foud at ' + root
    if verbose:
        print(f' (Found {len(imgs)} images)')
    return imgs, (W1, H1, W2, H2)


def tensor_to_pil(img_tensor):
    """Convert uint8 torch tensor [3, H, W] to PIL.Image"""
    return Image.fromarray(img_tensor.permute(1, 2, 0).cpu().numpy())


def load_tensor_images(tensor_list, size, square_ok=False, verbose=True):
    """Convert torch.Tensor RGB uint8 images to DUSt3R-ready format"""
    imgs = []
    for i, tensor in enumerate(tensor_list):
        if not (isinstance(tensor, torch.Tensor) and tensor.dtype == torch.uint8 and tensor.ndim == 3 and tensor.shape[
            0] == 3):
            raise ValueError(f"Invalid tensor at index {i}")

        img = tensor_to_pil(tensor)
        W1, H1 = img.size

        if size == 224:
            img = _resize_pil_image(img, round(size * max(W1 / H1, H1 / W1)))
        else:
            img = _resize_pil_image(img, size)

        W2, H2 = img.size
        if verbose:
            print(f' - tensor[{i}] resolution {W1}x{H1} --> {W2}x{H2}')
        imgs.append(dict(
            img=ImgNorm(img)[None],
            true_shape=np.int32([img.size[::-1]]),
            idx=i,
            instance=str(i)
        ))

    if not imgs:
        raise ValueError('No valid images in input list.')

    return imgs, (W1, H1, W2, H2)


def global_aligner(dust3r_output, device, **optim_kw):
    view1, view2, pred1, pred2 = [dust3r_output[k] for k in 'view1 view2 pred1 pred2'.split()]
    net = PointCloudOptimizer(view1, view2, pred1, pred2, **optim_kw).to(device)
    return net


def load_known_camera_parameters_from_neus_dataset(dataset_path, input_views):
    fx = []
    fy = []
    cx = []
    cy = []
    extrinsics = []
    for input_view in input_views:
        cameras_sphere_path = os.path.join(dataset_path, input_view, "cameras_sphere.npz")
        assert os.path.exists(cameras_sphere_path)

        cameras_sphere = np.load(cameras_sphere_path)
        world_mat_0 = cameras_sphere['world_mat_0']

        out = cv2.decomposeProjectionMatrix(world_mat_0[:3, :])
        K, R, t = out[:3]
        K = K / K[2, 2]
        t = t[:3].squeeze() / t[3]

        fx.append(K[0, 0])
        fy.append(K[1, 1])
        cx.append(K[0, 2])
        cy.append(K[1, 2])

        pose = np.eye(4)
        pose[:3, :3] = R.T
        pose[:3, 3] = t
        extrinsics_ = np.linalg.inv(pose)
        extrinsics.append(extrinsics_)

    fx = torch.tensor(fx).float()
    fy = torch.tensor(fy).float()
    cx = torch.tensor(cx).float()
    cy = torch.tensor(cy).float()
    extrinsics = torch.from_numpy(np.stack(extrinsics)).float()
    return fx, fy, cx, cy, extrinsics


def run_duster(
        images_tensor_or_image_paths,
        output_path,
        fx,
        fy,
        cx,
        cy,
        extrinsics,

        model_name_or_path="../duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        image_size=512,

        skip_if_output_already_exists=True,
        silent=False,
        output_2d_matches=False,
        dump_exhaustive_data=False,
        save_ply=False,
        save_png_viz=False,
        show_debug_plots=False,
        save_rerun_viz=False,
        rerun_radii=0.01,
        frame_selection=None,

        ga_lr=0.01,
        ga_schedule='linear',  # linear, cosine
        scenegraph_type="complete",  # complete, swin, oneref
        use_known_poses_for_pairwise_pose_init=False,  # True, False
        ga_niter=300,  # from 0 to 5000, default in demo was 300

        min_conf_thr=20,  # from 1 to 20, step 0.1, defualt in demo was 3
        mask_sky=False,  # True, False, default in demo was False
        clean_depth=True,  # True, False, default in demo was True

):
    # Set the random seed
    seed_all(72)
    os.makedirs(output_path, exist_ok=True)
    output_path = Path(output_path)

    # Load the model
    model = AsymmetricCroCo3DStereo.from_pretrained(model_name_or_path).to(device)

    # Load images into a torch tensor
    images_all = []
    n_views, n_frames = None, None
    original_w, original_h, target_w, target_h = None, None, None, None
    if not isinstance(images_tensor_or_image_paths, torch.Tensor):
        n_views = len(images_tensor_or_image_paths)
        n_frames = len(images_tensor_or_image_paths[0])

        for frame_idx in range(n_frames):
            frame_img_paths = [str(images_tensor_or_image_paths[view_idx][frame_idx]) for view_idx in range(n_views)]
            images, shapes = load_images(frame_img_paths, image_size, verbose=not silent)
            if original_w is None:
                original_w, original_h, target_w, target_h = shapes
            images_all.append(images)
    else:
        n_views, n_frames, _, original_h, original_w = images_tensor_or_image_paths.shape
        for frame_idx in range(n_frames):
            frame_imgs = [images_tensor_or_image_paths[view_idx, frame_idx] for view_idx in range(n_views)]
            images, shapes = load_tensor_images(frame_imgs, image_size, verbose=not silent)
            if target_w is None:
                assert (original_w, original_h) == shapes[:2]
                _, _, target_w, target_h = shapes
            images_all.append(images)

    # Check the input data
    assert len(fx) == len(fy) == len(cx) == len(cy) == len(extrinsics) == n_views
    assert all(extrinsics[view_idx].shape == (4, 4) for view_idx in range(n_views))

    # Assume known camera parameters
    known_poses = extrinsics.inverse()
    known_focals = torch.stack([fx, fy], dim=-1)
    known_pp = torch.stack([cx, cy], dim=-1)

    patch_h, patch_w = model.patch_embed.patch_size  # e.g., (16, 16)
    pad_h = (patch_h - (target_h % patch_h)) % patch_h
    pad_w = (patch_w - (target_w % patch_w)) % patch_w
    assert pad_h % 2 == 0, f"pad_h {pad_h} is not divisible by 2"
    assert pad_w % 2 == 0, f"pad_w {pad_w} is not divisible by 2"
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    if pad_h or pad_w:
        for frame_images in images_all:  # images_all[frame_idx] == list of dicts per view
            for im_dict in frame_images:
                # shape: [1, 3, H, W]
                assert im_dict["img"].shape[-2:] == (target_h, target_w)
                # F.pad takes (left, right, top, bottom)
                im_dict["img"] = F.pad(im_dict["img"], (pad_left, pad_right, pad_top, pad_bottom), mode="replicate")
                im_dict["true_shape"] = np.int32([[target_h + pad_h, target_w + pad_w]])

        # shift principal point to the padded image coordinate system
        # (we padded symmetrically, so add half the padding on each axis)
        known_pp = known_pp.clone()
        known_pp[..., 0] = known_pp[..., 0] + pad_left  # cx
        known_pp[..., 1] = known_pp[..., 1] + pad_top  # cy

    if frame_selection is None:
        frame_selection = range(n_frames)
    for frame_idx in frame_selection:
        print(f"Processing frame {frame_idx:05d}/{n_frames:05d}...")
        if skip_if_output_already_exists and os.path.exists(output_path / f"3d_model__{frame_idx:05d}__scene.npz"):
            try:
                np.load(output_path / f"3d_model__{frame_idx:05d}__scene.npz")
                print(f"Skipping frame because the output file already exists.")
                continue
            except Exception as e:
                print(f"Output file already exists but is corrupted: {e}")

        # Load preprocessed input images
        images = images_all[frame_idx]

        assert (target_h + pad_h, target_w + pad_w) == images[0]['img'].shape[-2:]
        assert len(images) == n_views
        print(f"Loaded {len(images)} images. "
              f"Original resolution: {original_w}x{original_h}. "
              f"Target resolution: {target_w}x{target_h}.")

        # Extract encoder features for each image
        feats = []
        for view_idx in range(n_views):
            with torch.no_grad():
                feat, pos_enc, _ = model._encode_image(images[view_idx]["img"].to(device),
                                                       images[view_idx]["true_shape"])
                feats.append(feat)
        feats = torch.concat(feats).detach().cpu().numpy()

        # Run DUSt3R on the pairs
        pairs = make_pairs(images, scene_graph=scenegraph_type, prefilter=None, symmetrize=True)
        output = inference(pairs, model, device, batch_size=1, verbose=not silent)

        # Unpad the output if padding was applied
        if pad_h or pad_w:
            H_pad = target_h + pad_h
            W_pad = target_w + pad_w
            t, l = pad_top, pad_left
            b, r = t + target_h, l + target_w

            assert output["view1"]["img"].shape == (len(pairs), 3, H_pad, W_pad)
            assert output["view2"]["img"].shape == (len(pairs), 3, H_pad, W_pad)
            assert output["pred1"]["conf"].shape == (len(pairs), H_pad, W_pad)
            assert output["pred2"]["conf"].shape == (len(pairs), H_pad, W_pad)
            assert output["pred1"]["pts3d"].shape == (len(pairs), H_pad, W_pad, 3)
            assert output["pred2"]["pts3d_in_other_view"].shape == (len(pairs), H_pad, W_pad, 3)

            output["view1"]["img"] = output["view1"]["img"][:, :, t:b, l:r].contiguous()
            output["view2"]["img"] = output["view2"]["img"][:, :, t:b, l:r].contiguous()
            output["pred1"]["conf"] = output["pred1"]["conf"][:, t:b, l:r].contiguous()
            output["pred2"]["conf"] = output["pred2"]["conf"][:, t:b, l:r].contiguous()
            output["pred1"]["pts3d"] = output["pred1"]["pts3d"][:, t:b, l:r, :].contiguous()
            output["pred2"]["pts3d_in_other_view"] = output["pred2"]["pts3d_in_other_view"][:, t:b, l:r, :].contiguous()
            output["view1"]["true_shape"] = np.int32([[target_h, target_w]])
            output["view2"]["true_shape"] = np.int32([[target_h, target_w]])

        # Set the known camera parameters
        scene = global_aligner(output, device=device, verbose=not silent)
        if not np.isclose(target_w / original_w, target_h / original_h):
            warnings.warn(f"The aspect ratio of the input images is different from the target aspect ratio:\n"
                          f" - rescaling factor x: {target_w}/{original_w} = {target_w / original_w}\n"
                          f" - rescaling factor y: {target_h}/{original_h} = {target_h / original_h}")
        if target_w == 512:
            rescaling_factor = target_w / original_w
        elif target_h == 512:
            rescaling_factor = target_h / original_h
        else:
            raise ValueError(f"Unexpected target resolution: {target_w}x{target_h}")
        print(f"We will use the rescaling factor: {target_w}/{original_w} = {rescaling_factor}")
        scene.preset_focal(known_focals.clone() * rescaling_factor)
        scene.im_pp.requires_grad_(True)
        scene.preset_principal_point(known_pp.clone() * rescaling_factor)
        scene.preset_pose(known_poses.clone())
        # scene.im_pp.requires_grad_(True)

        # Run global alignment to get the global pointcloud and estimated camera parameters
        init = 'mst' if not use_known_poses_for_pairwise_pose_init else 'known_poses'
        try:
            loss = scene.compute_global_alignment(init=init, niter=ga_niter, schedule=ga_schedule, lr=ga_lr)
        except Exception as e:
            other_init = {"mst": "known_poses", "known_poses": "mst"}
            print(f"Error during global alignment: {e}")
            print(f"Trying the other initialization method init={other_init[init]} instead of init={init}")
            loss = scene.compute_global_alignment(init=other_init[init], niter=ga_niter, schedule=ga_schedule, lr=ga_lr)
        print(f"Global alignment loss: {loss}")
        print(f"Poses after global alignment:")
        print(f"{scene.get_im_poses().cpu().tolist()},")
        print(f"Intrinsic after global alignment:")
        print(f"{scene.get_focals().cpu().tolist()}")
        print(f"{scene.get_principal_points().cpu().tolist()}")
        print()

        # Save the scene data, pointclouds, and camera parameters
        if feats is not None and (pad_h or pad_w):
            warnings.warn(f"The saved 'feats' won't take into account the padding (pad_h={pad_h}, pad_w={pad_w}).")
        get_3D_model_from_scene(
            output_file_prefix=output_path / f"3d_model__{frame_idx:05d}",
            silent=silent,
            scene=scene,
            min_conf_thr=min_conf_thr,
            mask_sky=mask_sky,
            clean_depth=clean_depth,
            feats=feats,
            dump_exhaustive_data=dump_exhaustive_data,
            save_ply=save_ply,
            save_png_viz=save_png_viz,
            save_rerun_viz=save_rerun_viz,
            rerun_radii=rerun_radii,
            rerun_viz_timestamp=frame_idx,
        )
        # get_3D_model_from_scene(output_path / f"low_threshold_3d_model__{frame_idx:05d}", silent, scene, 1, mask_sky, clean_depth)
        # get_3D_model_from_scene(output_path / f"non_clean_3d_model__{frame_idx:05d}", silent, scene, 0, mask_sky, False)
        if output_2d_matches:
            output_file_prefix = os.path.join(output_path, f"frame_{frame_idx}")
            get_2D_matches(output_file_prefix, scene, image_paths, min_conf_thr, clean_depth, viz_matches=True)

        if show_debug_plots:
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=3)
            fvec_flat_all = feats.reshape(-1, 1024)
            reducer.fit(fvec_flat_all)
            fvec_reduced = reducer.transform(fvec_flat_all)
            reducer_min = fvec_reduced.min(axis=0)
            reducer_max = fvec_reduced.max(axis=0)

            def fvec_to_rgb(fvec):
                fvec_reduced = reducer.transform(fvec)
                fvec_reduced_rescaled = (fvec_reduced - reducer_min) / (reducer_max - reducer_min)
                fvec_reduced_rgb = (fvec_reduced_rescaled * 255).astype(int)
                return fvec_reduced_rgb

            rgb_with_feat_list = []
            for view_idx in range(n_views):
                fvec_flat = feats[view_idx, :, :].reshape(((target_h + pad_h) // 16) * ((target_w + 16) // 16), 1024)
                fvec_reduced_rgb = fvec_to_rgb(fvec_flat).reshape((target_h + pad_h) // 16, (target_w + pad_w) // 16, 3)
                rgb_img = ((images[view_idx]["img"][0].permute(1, 2, 0).numpy() / 2 + 0.5) * 255).astype(int)
                fvec_img = np.kron(fvec_reduced_rgb, np.ones((16, 16, 1))).astype(int)
                rgb_with_feat = np.concatenate([rgb_img, fvec_img], axis=1)
                rgb_with_feat_list.append(rgb_with_feat)
            rgb_with_feat = np.concatenate(rgb_with_feat_list, axis=0)

            import matplotlib.pyplot as plt;
            plt.figure(figsize=(rgb_with_feat.shape[1] / 100, rgb_with_feat.shape[0] / 100), dpi=100)
            plt.imshow(rgb_with_feat)
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(os.path.join(output_path, f"debug__{frame_idx:05d}__rgb_with_encoder_features.png"))
            # plt.show()
            plt.close()


def main_on_neus_scene(scene_root, views_selection, **duster_kwargs):
    views_selection_str = ''.join(str(v) for v in views_selection)
    output_path = scene_root / f'duster-views-{views_selection_str}'
    view_paths = [scene_root / f"view_{v:02d}" for v in views_selection]

    frame_paths = [sorted((view_path / "rgb").glob("*.png")) for view_path in view_paths]
    n_frames = len(frame_paths[0])
    assert n_frames > 0
    assert all(len(f) == n_frames for f in frame_paths)

    fx, fy, cx, cy, extrinsics = [], [], [], [], []
    for view_path in view_paths:
        camera_params_file = os.path.join(view_path, "intrinsics_extrinsics.npz")
        params = np.load(camera_params_file)
        intrinsics = params["intrinsics"]
        extrinsics_view = params["extrinsics"]

        assert intrinsics[0, 1] == 0
        assert intrinsics[1, 0] == 0
        assert intrinsics[2, 0] == 0
        assert intrinsics[2, 1] == 0
        assert intrinsics[2, 2] == 1

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

    print(f"Processing {output_path}")
    run_duster(frame_paths, output_path, fx, fy, cx, cy, extrinsics, **duster_kwargs)


def main_on_kubric_scene(scene_root, views_selection, **duster_kwargs):
    views_selection_str = ''.join(str(v) for v in views_selection)
    output_path = scene_root / f'duster-views-{views_selection_str}'
    view_paths = [scene_root / f"view_{v:01d}" for v in views_selection]

    frame_paths = [sorted(view_path.glob("rgba_*.png")) for view_path in view_paths]
    n_frames = len(frame_paths[0])
    assert n_frames > 0
    assert all(len(f) == n_frames for f in frame_paths)

    datapoint = KubricMultiViewDataset.getitem_raw_datapoint(scene_root)
    fx, fy, cx, cy, extrinsics = [], [], [], [], []
    for view_idx in views_selection:
        intrinsics = datapoint["views"][view_idx]["intrinsics"]
        extrinsics_view = np.eye(4)
        extrinsics_view[:3, :4] = datapoint["views"][view_idx]["extrinsics"][0]

        assert intrinsics[0, 1] == 0
        assert intrinsics[1, 0] == 0
        assert intrinsics[2, 0] == 0
        assert intrinsics[2, 1] == 0
        assert intrinsics[2, 2] == 1

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
    print(f"Processing {output_path}")
    run_duster(frame_paths, output_path, fx, fy, cx, cy, extrinsics, **duster_kwargs)
    time_elapsed = time.time() - start
    print(f"Time elapsed for DUST3R: {time_elapsed:.2f} seconds")


def main_on_d3dgs_panoptic_scene(
        scene_root,
        views_selection,
        save_rerun_viz=False,
        rerun_radii=0.002,
        **duster_kwargs,
):
    md = json.load(open(os.path.join(scene_root, "train_meta.json"), 'r'))
    n_frames = len(md['fn'])

    # Check that the selected views are in the training set
    view_paths = []
    for view_idx in views_selection:
        view_path = scene_root / "ims" / f"{view_idx}"
        assert view_idx in md["cam_id"][0], f"Camera {view_idx} is not in the training set"
        assert view_path.exists()
        view_paths.append(view_path)
    frame_paths = [sorted(view_path.glob("*.jpg")) for view_path in view_paths]
    assert all(len(frame_paths[v]) == n_frames for v in range(len(views_selection)))

    # Create the output directory
    views_selection_str = '-'.join(str(v) for v in views_selection)
    output_path = scene_root / f'duster-views-{views_selection_str}'
    os.makedirs(output_path, exist_ok=True)

    # Load the camera parameters
    fx, fy, cx, cy, extrinsics = [], [], [], [], []
    for view_idx in views_selection:
        fx_current, fy_current, cx_current, cy_current, extrinsics_current = [], [], [], [], []
        for t in range(n_frames):
            view_idx_in_array = md['cam_id'][t].index(view_idx)
            k = md['k'][t][view_idx_in_array]
            w2c = np.array(md['w2c'][t][view_idx_in_array])

            fx_current.append(k[0][0])
            fy_current.append(k[1][1])
            cx_current.append(k[0][2])
            cy_current.append(k[1][2])
            extrinsics_current.append(w2c)

        assert all(np.equal(fx_current[0], fx_current[t]).all() for t in range(1, n_frames))
        assert all(np.equal(fy_current[0], fy_current[t]).all() for t in range(1, n_frames))
        assert all(np.equal(cx_current[0], cx_current[t]).all() for t in range(1, n_frames))
        assert all(np.equal(cy_current[0], cy_current[t]).all() for t in range(1, n_frames))
        assert all(np.equal(extrinsics_current[0], extrinsics_current[t]).all() for t in range(1, n_frames))

        fx.append(fx_current[0])
        fy.append(fy_current[0])
        cx.append(cx_current[0])
        cy.append(cy_current[0])
        extrinsics.append(extrinsics_current[0])

    fx = torch.tensor(fx).float()
    fy = torch.tensor(fy).float()
    cx = torch.tensor(cx).float()
    cy = torch.tensor(cy).float()
    extrinsics = torch.from_numpy(np.stack(extrinsics)).float()

    # Visualize the initialization point cloud used in D3DGS
    if save_rerun_viz:
        init_pt_cld = np.load(scene_root / "init_pt_cld.npz")["data"]
        xyz = init_pt_cld[:, :3]
        col = init_pt_cld[:, 3:6]
        seg = init_pt_cld[:, 6:7]
        rr.init("reconstruction", recording_id="v0.1")
        # rr.connect_tcp()
        rr.set_time_seconds("frame", 0 / 30)
        rr.log(f"point_cloud/sfm-full", rr.Points3D(xyz, colors=col, radii=rerun_radii))
        rr.log(f"point_cloud/sfm-full-seg", rr.Points3D(xyz, colors=col * seg, radii=rerun_radii))
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        rr.set_time_seconds("frame", 0)
        rr.log(
            "world/xyz",
            rr.Arrows3D(
                vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
                colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]],
            ),
        )
        # moge_depths = []
        # moge_masks = []
        for selected_view_idx, view_idx in enumerate(views_selection):
            rgbs = np.stack([np.array(Image.open(frame_paths[selected_view_idx][t])) for t in range(n_frames)])
            rgbs = torch.from_numpy(rgbs).permute(0, 3, 1, 2).float()
            H, W = rgbs.shape[-2], rgbs.shape[-1]
            K = np.array([
                [fx[selected_view_idx], 0, cx[selected_view_idx]],
                [0, fy[selected_view_idx], cy[selected_view_idx]],
                [0, 0, 1],
            ])
            K_inv = np.linalg.inv(K)
            K_for_moge = np.array([
                [fx[selected_view_idx] / W, 0, 0.5],
                [0, fy[selected_view_idx] / H, 0.5],
                [0, 0, 1],
            ])
            # depths, i, _, _, mask = moge(rgbs[::10], intrinsics=K_for_moge)
            # moge_depths.append(depths)
            # moge_masks.append(mask)
            for t in range(0, n_frames, 10):
                rr.set_time_seconds("frame", t / 30)
                c2w = torch.linalg.inv(extrinsics[selected_view_idx]).numpy()
                rr.log(f"image/view-{view_idx}/rgb", rr.Image(rgbs[t].permute(1, 2, 0).numpy()))
                # rr.log(f"image/view-{view_idx}/depth",
                #        rr.DepthImage(moge_depths[selected_view_idx][t // 10], point_fill_ratio=0.2))
                rr.log(f"image/view-{view_idx}", rr.Pinhole(image_from_camera=K, width=W, height=H))
                rr.log(f"image/view-{view_idx}", rr.Transform3D(translation=c2w[:3, 3], mat3x3=c2w[:3, :3]))

                # # Generate and log point cloud colored by RGB values
                # y, x = np.indices((H, W))
                # homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
                # depth_values = moge_depths[selected_view_idx][t // 10].ravel()
                # cam_coords = (K_inv @ homo_pixel_coords) * depth_values
                # cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                # world_coords = (c2w @ cam_coords)[:3].T
                # valid_mask = (depth_values > 0) & moge_masks[selected_view_idx][t // 10].reshape(-1, )
                # world_coords = world_coords[valid_mask]
                # rgb_colors = rgbs[t].permute(1, 2, 0).reshape(-1, 3).numpy()[valid_mask].astype(np.uint8)
                # rr.log(f"point_cloud/view-{view_idx}", rr.Points3D(world_coords, colors=rgb_colors, radii=rerun_radii))
        rr.save(output_path / "init_pt_cld.rrd")

    # Run DUSt3R
    print(f"Processing {output_path}")
    run_duster(frame_paths, output_path, fx, fy, cx, cy, extrinsics,
               save_rerun_viz=save_rerun_viz, rerun_radii=rerun_radii, **duster_kwargs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='The dataset to process')
    args = parser.parse_args()

    duster_kwargs = {
        "model_name_or_path": "../duster/checkpoints/DUSt3R_ViTLarge_BaseDecoder_512_dpt.pth",
        "silent": False,
        "output_2d_matches": False,
        "dump_exhaustive_data": True,
        "save_ply": True,
        "save_png_viz": True,
        "show_debug_plots": True,
    }

    if args.dataset == "dexycb":
        data_root = Path('./datasets/dex-january-2025/neus_nsubsample-3/')
        views_selections = [
            [0, 1, 2, 3],
            [2, 3, 4, 5],
            [4, 5, 6, 7],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ]
        for scene_root in sorted(data_root.glob("*")):
            for views_selection in views_selections:
                main_on_neus_scene(scene_root, views_selection, **duster_kwargs)

    elif args.dataset == "kubric-val":
        data_root = Path('./datasets/kubric_multiview_003/test/')
        duster_kwargs["save_rerun_viz"] = True
        views_selections = [
            # [0, 1],
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ]
        for scene_root in sorted(data_root.glob("[!.]*")):
            for views_selection in views_selections:
                main_on_kubric_scene(scene_root, views_selection, **duster_kwargs)

    elif args.dataset == "kubric-train":
        # Save space by not saving all logs
        duster_kwargs["dump_exhaustive_data"] = False
        duster_kwargs["save_ply"] = False
        duster_kwargs["save_png_viz"] = False
        duster_kwargs["show_debug_plots"] = False

        data_root = Path('./datasets/kubric_multiview_003/train/')
        views_selections = [
            [0, 1, 2, 3],
            [0, 1, 2, 3, 4, 5, 6, 7],
        ]

        # # Parallelize across a machine with 4 GPUs
        # total_gpus = 4
        # gpu_id = int(os.environ.get("CUDA_VISIBLE_DEVICES"))
        # # Run, e.g., as:
        # # --------------
        # # CUDA_VISIBLE_DEVICES=0 python scripts/estimate_depth_with_duster.py --dataset kubric-train
        # # CUDA_VISIBLE_DEVICES=1 python scripts/estimate_depth_with_duster.py --dataset kubric-train
        # # CUDA_VISIBLE_DEVICES=2 python scripts/estimate_depth_with_duster.py --dataset kubric-train
        # # CUDA_VISIBLE_DEVICES=3 python scripts/estimate_depth_with_duster.py --dataset kubric-train

        # Parallelize across 128 machines with 4 GPUs each
        total_gpus = 128 * 4
        a = int(os.environ.get("CHUNK"))
        b = int(os.environ.get("CUDA_VISIBLE_DEVICES"))
        gpu_id = a * 4 + b
        # Run, e.g., as:
        # --------------
        # CHUNK=0 CUDA_VISIBLE_DEVICES=0 python scripts/estimate_depth_with_duster.py --dataset kubric-train
        # CHUNK=0 CUDA_VISIBLE_DEVICES=1 python scripts/estimate_depth_with_duster.py --dataset kubric-train
        # CHUNK=0 CUDA_VISIBLE_DEVICES=2 python scripts/estimate_depth_with_duster.py --dataset kubric-train
        # CHUNK=0 CUDA_VISIBLE_DEVICES=3 python scripts/estimate_depth_with_duster.py --dataset kubric-train
        # CHUNK=1 CUDA_VISIBLE_DEVICES=1 python scripts/estimate_depth_with_duster.py --dataset kubric-train
        # ...
        # CHUNK=15 CUDA_VISIBLE_DEVICES=3 python scripts/estimate_depth_with_duster.py --dataset kubric-train

        print(f"Running on GPU {gpu_id} (out of {total_gpus})")
        print(f'Total scenes to process: {len(sorted(data_root.glob("[!.]*"))[gpu_id::total_gpus])}')
        for scene_root in sorted(data_root.glob("[!.]*"))[gpu_id::total_gpus]:
            for views_selection in views_selections:
                main_on_kubric_scene(scene_root, views_selection, **duster_kwargs)

    elif args.dataset == "panoptic_d3dgs":
        duster_kwargs["skip_if_output_already_exists"] = True
        duster_kwargs["save_rerun_viz"] = False
        duster_kwargs["frame_selection"] = None  # [0]
        data_root = Path('./datasets/panoptic_d3dgs/')
        views_selections = [
            # [27, 16, 14, 8, 11, 19, 11, 6, 23, 1],  # 10 views
            [27, 16, 14, 8, 11, 19, 11, 6],  # 8 views
            [27, 16, 14, 8],  # 4 views
            [27, 16],  # 2 views

            # [1, 4, 7, 11, 14, 17, 20, 23, 26, 29],  # 10 views
            # # [5, 8, 11, 14, 17, 20, 23, 26, 29],  # 9 views
            [1, 4, 7, 11, 14, 17, 20, 23],  # 8 views
            #
            [1, 4, 7, 11, ],  # 4 views - v1
            [1, 7, 14, 20, ],  # 4 views - v2
            #
            # [1, 4],  # 2 views - v1
            # [1, 14],  # 2 views - v2
        ]
        for scene_root in sorted(data_root.glob("[!.]*")):
            for views_selection in views_selections:
                main_on_d3dgs_panoptic_scene(scene_root, views_selection, **duster_kwargs)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Done.")
