import os
from dataclasses import dataclass
from functools import partial
from typing import Literal, cast

import cv2
import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tyro
from loguru import logger as guru
from roma import roma
from tqdm import tqdm

from flow3d.data.base_dataset import BaseDataset
from flow3d.data.utils import (
    UINT16_MAX,
    SceneNormDict,
    get_tracks_3d_for_query_frame,
    median_filter_2d,
    normal_from_depth_image,
    normalize_coords,
    parse_tapir_track_info,
)
from flow3d.transforms import rt_to_mat4

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

import models.spatracker.datasets.utils as dataset_utils
from models.spatracker.datasets.panoptic_studio_multiview_dataset import PanopticStudioMultiViewDataset

from torch.utils.data import default_collate

@dataclass
class PanopticDataConfig:
    seq_name: str
    root_dir: str
    start: int = 0
    end: int = -1
    res: str = ""
    image_type: str = "images"
    mask_type: str = "masks"
    depth_type: Literal[
        "aligned_depth_anything",
        "aligned_depth_anything_v2",
        "depth_anything",
        "depth_anything_v2",
        "unidepth_disp",
    ] = "aligned_depth_anything"
    camera_type: Literal["droid_recon"] = "droid_recon"
    track_2d_type: Literal["bootstapir", "tapir"] = "bootstapir"
    mask_erosion_radius: int = 7
    scene_norm_dict: tyro.conf.Suppress[SceneNormDict | None] = None
    num_targets_per_frame: int = 4
    load_from_cache: bool = False

class PanopticStudioDatasetSoM(BaseDataset):
    def __init__(
        self,
        seq_name: str,
        root_dir: str,
        res: str = "480p",
        depth_type: Literal[
            "aligned_depth_anything",
            "aligned_depth_anything_v2",
            "depth_anything",
            "depth_anything_v2",
            "unidepth_disp",
        ] = "aligned_depth_anything",
        mask_erosion_radius: int = 0,
        scene_norm_dict: SceneNormDict | None = None,
        num_targets_per_frame: int = 4,
        load_from_cache: bool = False,
        **_,
    ):
        super().__init__()

        self.seq_name = seq_name
        self.root_dir = root_dir
        self.res = res
        self.depth_type = depth_type
        self.num_targets_per_frame = num_targets_per_frame
        self.load_from_cache = load_from_cache
        self.has_validation = False
        self.mask_erosion_radius = mask_erosion_radius

        #######################################################################
        self.views_to_return = [1, 7, 14, 20]

        datasets_root = "/cluster/scratch/egundogdu/datasets/"
        panoptic_kwargs = {
            "data_root": os.path.join(datasets_root, "panoptic_d3dgs"), "traj_per_sample": 384, "seed": 72,
            "max_videos": 1, "perform_sanity_checks": False, "views_to_return": [1, 7, 14, 20],
            "use_duster_depths": False, "clean_duster_depths": False,
        }
        self.panoptic_spatial_dataset = PanopticStudioMultiViewDataset(**panoptic_kwargs)
        
        datapoint = self.panoptic_spatial_dataset.__getitem__(0)

        if isinstance(datapoint, tuple):
            datapoint, gotit = datapoint
            assert gotit
        if torch.cuda.is_available():
            dataset_utils.dataclass_to_cuda_(datapoint)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        self.img_dir_view_1 = os.path.join(datasets_root, "panoptic_d3dgs", "basketball", "ims", "1")
        self.frame_names = [os.path.splitext(p)[0] for p in sorted(os.listdir(self.img_dir_view_1))]

        # Per view data
        self.rgbs = datapoint.video
        self.depths = datapoint.videodepth
        self.image_features = datapoint.feats
        self.intrs = datapoint.intrs
        self.extrs = datapoint.extrs
        self.gt_trajectories_2d_pixelspace_w_z_cameraspace = datapoint.trajectory
        self.gt_visibilities_per_view = datapoint.visibility
        self.query_points_2d = (datapoint.query_points.clone().float().to(device)
                           if datapoint.query_points is not None else None)
        self.query_points_3d = datapoint.query_points_3d.clone().float().to(device)

        # Non-per-view data
        self.gt_trajectories_3d_worldspace = datapoint.trajectory_3d
        self.valid_tracks_per_frame = datapoint.valid
        self.track_upscaling_factor = datapoint.track_upscaling_factor

        print(self.rgbs.shape)
        num_views, num_frames, _, height, width = self.rgbs.shape
        num_points = self.gt_trajectories_2d_pixelspace_w_z_cameraspace.shape[2]

        self.rgbs = self.rgbs.permute(0, 1, 3, 4, 2).cpu()
        self.depths = self.depths.permute(0, 1, 3, 4, 2).cpu()

        # Assert shapes of per-view data
        assert self.depths is not None, "Depth is required for evaluation."
        assert self.rgbs.shape == (num_views, num_frames, height, width, 3)
        assert self.depths.shape == (num_views, num_frames, height, width, 1)
        assert self.intrs.shape == (num_views, num_frames, 3, 3)
        assert self.extrs.shape == (num_views, num_frames, 3, 4)
        assert self.gt_trajectories_2d_pixelspace_w_z_cameraspace.shape == (
            num_views, num_frames, num_points, 3)
        assert self.gt_visibilities_per_view.shape == (num_views, num_frames, num_points)

        # Assert shapes of non-per-view data
        assert self.query_points_3d.shape == (num_points, 4)
        assert self.gt_trajectories_3d_worldspace.shape == (num_frames, num_points, 3)
        assert self.valid_tracks_per_frame.shape == (num_frames, num_points)

        self.w2cs = torch.eye(4).expand(num_views, num_frames, 4, 4).clone()
        self.w2cs[:, :, :3, :] = self.extrs.squeeze(0).cpu()  # (n_views, n_frames, 4, 4)
        self.Ks = self.intrs.squeeze(0).cpu()                 # (n_views, n_frames, 3, 3)



        
        ###### normalization...
        self.scale = 1

        tracks_3d = self.get_tracks_3d(5000, step=num_frames // 10)[0]
        scale, transfm = compute_scene_norm(tracks_3d, self.w2cs)
        scene_norm_dict = SceneNormDict(scale=scale, transfm=transfm)

        # transform cameras
        self.scene_norm_dict = cast(SceneNormDict, scene_norm_dict)
        self.scale = self.scene_norm_dict["scale"]
        transform = self.scene_norm_dict["transfm"]
        guru.info(f"scene norm {self.scale=}, {transform=}")
        for v in range(num_views):
            self.w2cs[v] = torch.einsum("nij,jk->nik", self.w2cs[v], torch.linalg.inv(transform))
            self.w2cs[v, :, :3, 3] /= self.scale


        

    @property
    def num_frames(self) -> int:
        return len(self.frame_names)

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        # return self._keyframe_idcs
        return np.array(range(10,140,10))

    def __len__(self):
        return len(self.frame_names)

    def get_w2cs(self, view_index=0) -> torch.Tensor:
        return self.w2cs[view_index].cpu().to(torch.float32)

    def get_Ks(self, view_index=0) -> torch.Tensor:
        return self.Ks[view_index].cpu().to(torch.float32)

    def get_img_wh(self) -> tuple[int, int]:
        return self.get_image(0).shape[1::-1]

    def get_image(self, index, view_index=0) -> torch.Tensor:
        return self.rgbs[view_index][index].cpu().to(torch.float32) / 255.0

    def get_mask(self, index, view_index=0) -> torch.Tensor:
        view = self.views_to_return[view_index]
        mask = self.load_mask(index, view)
        mask = cast(torch.Tensor, mask)
        return mask.cpu().to(torch.float32)

    def get_depth(self, index, view=0) -> torch.Tensor:
        # return self.load_depth(index, view) / self.scales[view]
        return self.load_depth(index, view).cpu().to(torch.float32) / self.scale

    def load_mask(self, index, view=0) -> torch.Tensor:
        # self.mask_dir = "/cluster/scratch/egundogdu/datasets/panoptic_d3dgs/basketball/seg"
        self.mask_dir = "/cluster/home/egundogdu/projects/vlg-lab/spatialtracker/shape-of-motion/panoptic_masks"
        path = f"{self.mask_dir}/{view}/{self.frame_names[index]}.png"
        r = self.mask_erosion_radius
        mask = imageio.imread(path)
        fg_mask = mask.reshape((*mask.shape[:2], -1)).max(axis=-1) > 0
        bg_mask = ~fg_mask
        fg_mask_erode = cv2.erode(
            fg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        bg_mask_erode = cv2.erode(
            bg_mask.astype(np.uint8), np.ones((r, r), np.uint8), iterations=1
        )
        out_mask = np.zeros_like(fg_mask, dtype=np.float32)
        out_mask[bg_mask_erode > 0] = -1
        out_mask[fg_mask_erode > 0] = 1
        return torch.from_numpy(out_mask).float()

    def load_depth(self, index, view=0) -> torch.Tensor:
        depth = self.depths[view][index]
        depth = depth.permute(2, 0, 1).unsqueeze(0)
        depth = median_filter_2d(depth, 11, 1)[0, 0]
        return depth.squeeze(0)


    #####################################
    def get_foreground_points(
        self,
        num_samples: int,
        use_kf_tstamps: bool = False,
        stride: int = 4,
        down_rate: int = 8,
        min_per_frame: int = 64,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = 0
        end = self.num_frames
        H, W = self.rgbs.shape[2:4]  # Get height & width from rgbs shape

        # Create pixel grid
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )  # Shape: (H, W, 2)

        if use_kf_tstamps:
            query_idcs = self.keyframe_idcs.tolist()
        else:
            num_query_frames = self.num_frames // stride
            query_endpts = torch.linspace(start, end, num_query_frames + 1)
            query_idcs = ((query_endpts[:-1] + query_endpts[1:]) / 2).long().tolist()

        bg_geometry = []
        print(f"{query_idcs=}")
        
        # for v in range(self.rgbs.shape[0]):  # Iterate over views
        for query_idx in tqdm(query_idcs, desc=f"Loading foreground points (view)", leave=False):
            for v in [0, 1, 2, 3]:

                img = self.get_image(query_idx, v).cpu().numpy()  # Shape: (H, W, 3)
                height, width = img.shape[0], img.shape[1]

                depth = self.get_depth(query_idx, v).cpu().numpy()
                mask = self.get_mask(query_idx, v).cpu().numpy() < 0  # Shape: (H, W)
                valid_mask = (~mask * (depth > 0)).ravel()

                w2c = self.w2cs[v, query_idx].cpu().numpy()
                c2w = np.linalg.inv(w2c)
                k = self.Ks[v, query_idx].cpu().numpy()
                k_inv = np.linalg.inv(k)
            

                y, x = np.indices((height, width))
                homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
                cam_coords = (k_inv @ homo_pixel_coords) * depth.ravel()
                cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                world_coords = (c2w @ cam_coords)[:3].T
                world_coords = world_coords[valid_mask]
                rgb_colors = img.reshape(-1, 3)[valid_mask].astype(np.uint8)


                bg_geometry.append((torch.from_numpy(world_coords), torch.from_numpy(world_coords), torch.from_numpy(rgb_colors)))

                rr.set_time_seconds("frame", query_idx / 30)
                rr.log(f"world/points/view_{v}_foreground", rr.Points3D(positions=world_coords, colors=rgb_colors * 255.0))



                # tmp_img = img.clone()
                # tmp_img[~bool_mask] = 1
                # img_8bit = (tmp_img.reshape(self.rgbs[v, query_idx].shape).cpu().numpy() * 255).astype(np.uint8)
                # datasets_root = f"/cluster/scratch/egundogdu/datasets/view{v}_frame{query_idx}.png"
                # cv2.imwrite(datasets_root, img_8bit[..., ::-1])
                # print(f"Saved {datasets_root}")

                # img_8bit = (depth.cpu().numpy() * 255).astype(np.uint8)
                # datasets_root = f"/cluster/scratch/egundogdu/datasets/depth_view{v}_frame{query_idx}.png"
                # cv2.imwrite(datasets_root, img_8bit[..., ::-1])
                # print(f"Saved {datasets_root}")

                # img_8bit = (bool_mask.cpu().numpy() * 255).astype(np.uint8)
                # datasets_root = f"/cluster/scratch/egundogdu/datasets/bool_mask_view{v}_frame{query_idx}.png"
                # cv2.imwrite(datasets_root, img_8bit[..., ::-1])
                # print(f"Saved {datasets_root}")



        bg_points, bg_normals, bg_colors = map(
            partial(torch.cat, dim=0), zip(*bg_geometry)
        )

        # Final downsampling
        # doesnt use texture-based prob sampling
        # TODO: add texture information to sample from a probability
        if len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]

        return bg_points, bg_normals, bg_colors
    

    def get_bkgd_points(
        self,
        num_samples: int,
        use_kf_tstamps: bool = False,
        stride: int = 8,
        down_rate: int = 8,
        min_per_frame: int = 64,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        start = 0
        end = self.num_frames
        H, W = self.rgbs.shape[2:4]  # Get height & width from rgbs shape

        # Create pixel grid
        grid = torch.stack(
            torch.meshgrid(
                torch.arange(0, W, dtype=torch.float32),
                torch.arange(0, H, dtype=torch.float32),
                indexing="xy",
            ),
            dim=-1,
        )  # Shape: (H, W, 2)

        if use_kf_tstamps:
            query_idcs = self.keyframe_idcs.tolist()
        else:
            num_query_frames = self.num_frames // stride
            query_endpts = torch.linspace(start, end, num_query_frames + 1)
            query_idcs = ((query_endpts[:-1] + query_endpts[1:]) / 2).long().tolist()

        bg_geometry = []
        print(f"{query_idcs=}")
        
        view_index_list = [0, 1, 2, 3]

        # for v in range(self.rgbs.shape[0]):  # Iterate over views
        for query_idx in tqdm(query_idcs, desc=f"Loading bkgd points (view)", leave=False):
            for v in view_index_list:

                img = self.get_image(query_idx, v).cpu().numpy()  # Shape: (H, W, 3)
                height, width = img.shape[0], img.shape[1]

                depth = self.get_depth(query_idx, v).cpu().numpy()
                mask = self.get_mask(query_idx, v).cpu().numpy() < 0  # Shape: (H, W)
                valid_mask = (mask * (depth > 0)).ravel()
                # valid_mask = depth.ravel() > 0

                w2c = self.w2cs[v, query_idx].cpu().numpy()
                c2w = np.linalg.inv(w2c)
                k = self.Ks[v, query_idx].cpu().numpy()
                k_inv = np.linalg.inv(k)
            

                y, x = np.indices((height, width))
                homo_pixel_coords = np.stack([x.ravel(), y.ravel(), np.ones_like(x).ravel()], axis=1).T
                cam_coords = (k_inv @ homo_pixel_coords) * depth.ravel()
                cam_coords = np.vstack((cam_coords, np.ones((1, cam_coords.shape[1]))))
                world_coords = (c2w @ cam_coords)[:3].T
                world_coords = world_coords[valid_mask]
                rgb_colors = img.reshape(-1, 3)[valid_mask]


                bg_geometry.append((torch.from_numpy(world_coords).to(torch.float32), torch.from_numpy(world_coords).to(torch.float32), torch.from_numpy(rgb_colors).to(torch.float32)))


        bg_points, bg_normals, bg_colors = map(
            partial(torch.cat, dim=0), zip(*bg_geometry)
        )

        # Final downsampling
        # doesnt use texture-based prob sampling
        # TODO: add texture information to sample from a probability
        if len(bg_points) > num_samples:
            sel_idcs = np.random.choice(len(bg_points), num_samples, replace=False)
            bg_points = bg_points[sel_idcs]
            bg_normals = bg_normals[sel_idcs]
            bg_colors = bg_colors[sel_idcs]

        return bg_points, bg_normals, bg_colors

    #####################################
    def load_target_tracks(
        self, query_index: int, target_indices: list[int], view_index=0, dim: int = 1
    ):
        """
        tracks are 2d, occs and uncertainties
        :param dim (int), default 1: dimension to stack the time axis
        return (N, T, 4) if dim=1, (T, N, 4) if dim=0
        """
        view = self.views_to_return[view_index]

        q_name = self.frame_names[query_index]
        all_tracks = []
        for ti in target_indices:
            t_name = self.frame_names[ti]
            # path = f"/cluster/scratch/egundogdu/datasets/panoptic_d3dgs/basketball/tracks_tapvid_som/{view}/{q_name}_{t_name}.npy"
            path = f"/cluster/home/egundogdu/projects/vlg-lab/spatialtracker/shape-of-motion/panoptic_tracks/{view}/{q_name}_{t_name}.npy"
            tracks = np.load(path).astype(np.float32)
            all_tracks.append(tracks)
        return torch.from_numpy(np.stack(all_tracks, axis=dim))
    
    def get_tracks_3d(
        self, num_samples: int, start: int = 0, end: int = -1, step: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_frames = self.num_frames
        if end < 0:
            end = num_frames + 1 + end
        query_idcs = list(range(start, end, step))
        target_idcs = list(range(start, end, step))
        
        num_per_query_frame = int(np.ceil(num_samples / len(query_idcs) / 8))
        cur_num = 0
        tracks_all_queries = []


        view_index_list = [0, 1, 2, 3]

        precomputed_data = {}
        for v in view_index_list:
            masks = torch.stack([self.get_mask(i, v).cpu() for i in target_idcs], dim=0)
            fg_masks = (masks == 1).float()
            depths = torch.stack([self.get_depth(i, v).cpu() for i in target_idcs], dim=0)
            inv_Ks = torch.linalg.inv(self.Ks[v][target_idcs].cpu())
            c2ws = torch.linalg.inv(self.w2cs[v][target_idcs].cpu())
            
            precomputed_data[v] = (fg_masks, depths, inv_Ks, c2ws)

        for q_idx in tqdm(query_idcs, desc=f"Loading 3d tracks points", leave=False):
            for v in view_index_list:
                # # masks = torch.stack([self.get_mask(i, v) for i in target_idcs], dim=0)
                # # fg_masks = (masks == 1).float()
                # # depths = torch.stack([self.get_depth(i, v) for i in target_idcs], dim=0)
                # inv_Ks = torch.linalg.inv(self.Ks[v][target_idcs])
                # c2ws = torch.linalg.inv(self.w2cs[v][target_idcs])
                fg_masks, depths, inv_Ks, c2ws = precomputed_data[v]

                # (N, T, 4)
                # print(q_idx, len(query_idcs), "cur: ", cur_num)
                tracks_2d = self.load_target_tracks(q_idx, target_idcs, v).cpu()
                num_sel = int(
                    min(num_per_query_frame, num_samples - cur_num, len(tracks_2d))
                )
                if num_sel < len(tracks_2d):
                    sel_idcs = np.random.choice(len(tracks_2d), num_sel, replace=False)
                    tracks_2d = tracks_2d[sel_idcs]
                cur_num += tracks_2d.shape[0]

                img = self.get_image(q_idx, v).cpu()
                tidx = target_idcs.index(q_idx)
                tracks_tuple = get_tracks_3d_for_query_frame(
                    tidx, img, tracks_2d, depths, fg_masks, inv_Ks, c2ws
                )
                tracks_all_queries.append(tracks_tuple)

        tracks_3d, colors, visibles, invisibles, confidences = map(
            partial(torch.cat, dim=0), zip(*tracks_all_queries)
        )
        return tracks_3d, visibles, invisibles, confidences, colors


    
    def train_collate_fn(self, batch):
        """
        Collate function that correctly batches data when each sample consists of multiple views.
        """

        # Step 1: Transpose the batch to group by views
        # If batch contains 4 views per sample, `batch` is a list of lists: [ [view_1, view_2, view_3, view_4], [view_1, view_2, view_3, view_4], ... ]
        # We want to group all view_1's together, all view_2's together, etc.
        num_views = len(batch[0])  # Assumes each sample has the same number of views
        batch_per_view = list(zip(*batch))  # Transposes list-of-lists structure

        collated_views = []
        
        # Step 2: Collate each view separately
        for view_batch in batch_per_view:
            collated = {}
            for k in view_batch[0]:  # Iterate over keys in the dictionary
                if k not in [
                    "query_tracks_2d",
                    "target_ts",
                    "target_w2cs",
                    "target_Ks",
                    "target_tracks_2d",
                    "target_visibles",
                    "target_track_depths",
                    "target_invisibles",
                    "target_confidences",
                ]:
                    collated[k] = default_collate([sample[k] for sample in view_batch])
                else:
                    collated[k] = [sample[k] for sample in view_batch]  # Keep list format
            collated_views.append(collated)

        return collated_views  # List of collated dictionaries, one per view
    

    # def __getitem__(self, index: int, view=0):
    #     index = np.random.randint(0, self.num_frames)
    #     data = {
    #         # ().
    #         "frame_names": self.frame_names[index],
    #         # ().
    #         "ts": torch.tensor(index),
    #         # (4, 4).
    #         "w2cs": self.w2cs[view][index],
    #         # (3, 3).
    #         "Ks": self.Ks[view][index],
    #         # (H, W, 3).
    #         "imgs": self.get_image(index, view),
    #         "depths": self.get_depth(index, view),
    #     }
    #     tri_mask = self.get_mask(index, view)
    #     valid_mask = tri_mask != 0  # not fg or bg
    #     mask = tri_mask == 1  # fg mask
    #     data["masks"] = mask.float()
    #     data["valid_masks"] = valid_mask.float()

    #     # (P, 2)
    #     query_tracks = self.load_target_tracks(index, [index], view_index=view)[:, 0, :2]
    #     target_inds = torch.from_numpy(
    #         np.random.choice(
    #             self.num_frames, (self.num_targets_per_frame,), replace=False
    #         )
    #     )
    #     # (N, P, 4)
    #     target_tracks = self.load_target_tracks(index, target_inds.tolist(), view_index=view, dim=0)
    #     data["query_tracks_2d"] = query_tracks
    #     data["target_ts"] = target_inds
    #     data["target_w2cs"] = self.w2cs[view][target_inds]
    #     data["target_Ks"] = self.Ks[view][target_inds]
    #     data["target_tracks_2d"] = target_tracks[..., :2]
    #     # (N, P).
    #     (
    #         data["target_visibles"],
    #         data["target_invisibles"],
    #         data["target_confidences"],
    #     ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
    #     # (N, H, W)
    #     target_depths = torch.stack([self.get_depth(i, view) for i in target_inds], dim=0)
    #     H, W = target_depths.shape[-2:]
    #     data["target_track_depths"] = F.grid_sample(
    #         target_depths[:, None],
    #         normalize_coords(target_tracks[..., None, :2], H, W),
    #         align_corners=True,
    #         padding_mode="border",
    #     )[:, 0, :, 0]
    #     return data
    
    def get_batches(self, batch_size):
        num_batches = self.num_frames // batch_size  # Determine number of batches
        train_collated_merged_data = []
        
        for _ in range(num_batches):
            train_collated_merged_data.append(self.__getitem_as_batch__(batch_size))
        
        return train_collated_merged_data

    def __getitem_as_batch__(self, batch_size):
        # index = np.random.randint(0, self.num_frames)
        if batch_size > self.num_frames:
            index = np.random.choice(self.num_frames, batch_size, replace=True)  # Sample with replacement
        else:
            index = np.random.choice(self.num_frames, batch_size, replace=False)  # Sample without replacement
        
        merged_data = []
        for i in tqdm(index):
            view_data = []
            for view in [0, 1, 2, 3]:
                view_data.append(self.__getitem_single_view__(i, view))
            merged_data.append(view_data)
        
        return self.train_collate_fn(merged_data)

    def __getitem_single_view__(self, index: int, view: int):
        index = np.random.randint(0, self.num_frames)
    
        data = {
            # ().
            "frame_names": self.frame_names[index],
            # ().
            "ts": torch.tensor(index),
            # (4, 4).
            "w2cs": self.w2cs[view][index],
            # (3, 3).
            "Ks": self.Ks[view][index],
            # (H, W, 3).
            "imgs": self.get_image(index, view),
            "depths": self.get_depth(index, view),
        }
        tri_mask = self.get_mask(index, view)
        valid_mask = tri_mask != 0  # not fg or bg
        mask = tri_mask == 1  # fg mask
        data["masks"] = mask.float()
        data["valid_masks"] = valid_mask.float()

        # (P, 2)
        query_tracks = self.load_target_tracks(index, [index], view_index=view)[:, 0, :2]
        target_inds = torch.from_numpy(
            np.random.choice(
                self.num_frames, (self.num_targets_per_frame,), replace=False
            )
        )
        # (N, P, 4)
        target_tracks = self.load_target_tracks(index, target_inds.tolist(), view_index=view, dim=0)
        data["query_tracks_2d"] = query_tracks
        data["target_ts"] = target_inds
        data["target_w2cs"] = self.w2cs[view][target_inds]
        data["target_Ks"] = self.Ks[view][target_inds]
        data["target_tracks_2d"] = target_tracks[..., :2]
        # (N, P).
        (
            data["target_visibles"],
            data["target_invisibles"],
            data["target_confidences"],
        ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
        # (N, H, W)
        target_depths = torch.stack([self.get_depth(i, view) for i in target_inds], dim=0)
        H, W = target_depths.shape[-2:]
        data["target_track_depths"] = F.grid_sample(
            target_depths[:, None],
            normalize_coords(target_tracks[..., None, :2], H, W),
            align_corners=True,
            padding_mode="border",
        )[:, 0, :, 0] 

        return data


    def __getitem__(self, index: int):
        index = np.random.randint(0, self.num_frames)
        merged_data = []
        for view in [0, 1, 2, 3]:
            data = {
                # ().
                "frame_names": self.frame_names[index],
                # ().
                "ts": torch.tensor(index),
                # (4, 4).
                "w2cs": self.w2cs[view][index],
                # (3, 3).
                "Ks": self.Ks[view][index],
                # (H, W, 3).
                "imgs": self.get_image(index, view),
                "depths": self.get_depth(index, view),
            }
            tri_mask = self.get_mask(index, view)
            valid_mask = tri_mask != 0  # not fg or bg
            mask = tri_mask == 1  # fg mask
            data["masks"] = mask.float()
            data["valid_masks"] = valid_mask.float()

            # (P, 2)
            query_tracks = self.load_target_tracks(index, [index], view_index=view)[:, 0, :2]
            target_inds = torch.from_numpy(
                np.random.choice(
                    self.num_frames, (self.num_targets_per_frame,), replace=False
                )
            )
            # (N, P, 4)
            target_tracks = self.load_target_tracks(index, target_inds.tolist(), view_index=view, dim=0)
            data["query_tracks_2d"] = query_tracks
            data["target_ts"] = target_inds
            data["target_w2cs"] = self.w2cs[view][target_inds]
            data["target_Ks"] = self.Ks[view][target_inds]
            data["target_tracks_2d"] = target_tracks[..., :2]
            # (N, P).
            (
                data["target_visibles"],
                data["target_invisibles"],
                data["target_confidences"],
            ) = parse_tapir_track_info(target_tracks[..., 2], target_tracks[..., 3])
            # (N, H, W)
            target_depths = torch.stack([self.get_depth(i, view) for i in target_inds], dim=0)
            H, W = target_depths.shape[-2:]
            data["target_track_depths"] = F.grid_sample(
                target_depths[:, None],
                normalize_coords(target_tracks[..., None, :2], H, W),
                align_corners=True,
                padding_mode="border",
            )[:, 0, :, 0] 

            merged_data.append(data)
            
        return merged_data


def compute_scene_norm(
    X: torch.Tensor, w2cs: torch.Tensor
) -> tuple[float, torch.Tensor]:
    """
    :param X: [N*T, 3]
    # :param w2cs: [N, 4, 4]
    :param w2cs: [n_views, N, 4, 4]
    """
    X = X.reshape(-1, 3)
    scene_center = X.mean(dim=0)
    X = X - scene_center[None]
    min_scale = X.quantile(0.05, dim=0)
    max_scale = X.quantile(0.95, dim=0)
    scale = (max_scale - min_scale).max().item() / 2.0
    
    original_up = -F.normalize(w2cs[:, :, 1, :3].mean(dim=(0,1)), dim=-1)
    target_up = original_up.new_tensor([0.0, 0.0, 1.0])

    R = roma.rotvec_to_rotmat(
        F.normalize(original_up.cross(target_up), dim=-1)
        * original_up.dot(target_up).acos_()
    )
    transfm = rt_to_mat4(R, torch.einsum("ij,j->i", -R, scene_center))
    return scale, transfm






# import rerun as rr
if __name__ == "__main__":
#     rr.init("3dpt", recording_id="v0.1")
#     rr.connect_tcp("0.0.0.0:9876")
#     rr.set_time_seconds("frame", 0)
#     rr.log("world/xyz", rr.Arrows3D(vectors=[[1, 0, 0], [0, 2, 0], [0, 0, 3]],
#                                             colors=[[255, 0, 0], [0, 255, 0], [0, 0, 255]]))

    d = PanopticStudioDatasetSoM("", "", camera_type="")
    batch = d.__getitem_as_batch__(150)
    import ipdb
    ipdb.set_trace()


    # print(d["imgs"])

#     # Get background points
#     points, normals, colors = d.get_bkgd_points(num_samples=100_000)
#     print(points.dtype)

#     rr.set_time_seconds("frame", 0)
#     rr.log(f"world/points/final_background", rr.Points3D(positions=points, colors=colors * 255.0))
#     print("Done.")

#     # # Get foreground points
#     points, normals, colors = d.get_foreground_points(num_samples=40_000)
#     rr.set_time_seconds("frame", 0)
#     rr.log(f"world/points/final_foreground", rr.Points3D(positions=points, colors=colors * 255.0))
#     print("Done.")

#     # tracks_2d = d.load_target_tracks(0, [0,1,2,3,4], 1)    
#     # print(tracks_2d.dtype)
#     # # tracks_3d, visibles, invisibles, confidences, colors = d.get_tracks_3d(40000)
#     # # colors = (colors * 255.0)
#     # # print(
#     # #     f"{tracks_3d.shape=} {visibles.shape=} "
#     # #     f"{invisibles.shape=} {confidences.shape=} "
#     # #     f"{colors.shape=}"
#     # # )

#     # # # Loop through 150 frames and log the corresponding points
#     # # num_frames = tracks_3d.shape[1]  # 150 frames
#     # # for frame_idx in range(num_frames):
#     # #     rr.set_time_seconds("frame", frame_idx)

#     # #     # Get the 3D positions for the current frame
#     # #     frame_tracks = tracks_3d[:, frame_idx, :]  # Shape: (35418, 3)
#     # #     frame_visibles = visibles[:, frame_idx]  # Visibility mask

#     # #     # Filter only visible points
#     # #     visible_tracks = frame_tracks[frame_visibles > 0]
#     # #     visible_colors = colors[frame_visibles > 0]

#     # #     rr.set_time_seconds("frame", frame_idx / 30)
#     # #     rr.log(f"world/tracks_3d", rr.Points3D(positions=visible_tracks, colors=visible_colors))
