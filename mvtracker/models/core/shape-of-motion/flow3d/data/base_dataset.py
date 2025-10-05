from abc import abstractmethod

import torch
from torch.utils.data import Dataset, default_collate


class BaseDataset(Dataset):
    @property
    @abstractmethod
    def num_frames(self) -> int: ...

    @property
    def keyframe_idcs(self) -> torch.Tensor:
        return torch.arange(self.num_frames)

    @abstractmethod
    def get_w2cs(self) -> torch.Tensor: ...

    @abstractmethod
    def get_Ks(self) -> torch.Tensor: ...

    @abstractmethod
    def get_image(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def get_depth(self, index: int) -> torch.Tensor: ...

    @abstractmethod
    def get_mask(self, index: int) -> torch.Tensor: ...

    def get_img_wh(self) -> tuple[int, int]: ...

    @abstractmethod
    def get_tracks_3d(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns 3D tracks:
            coordinates (N, T, 3),
            visibles (N, T),
            invisibles (N, T),
            confidences (N, T),
            colors (N, 3)
        """
        ...

    @abstractmethod
    def get_bkgd_points(
        self, num_samples: int, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns background points:
            coordinates (N, 3),
            normals (N, 3),
            colors (N, 3)
        """
        ...

    # @staticmethod
    # def train_collate_fn(batch):
    #     collated = {}
    #     for k in batch[0]:
    #         if k not in [
    #             "query_tracks_2d",
    #             "target_ts",
    #             "target_w2cs",
    #             "target_Ks",
    #             "target_tracks_2d",
    #             "target_visibles",
    #             "target_track_depths",
    #             "target_invisibles",
    #             "target_confidences",
    #         ]:
    #             collated[k] = default_collate([sample[k] for sample in batch])
    #         else:
    #             collated[k] = [sample[k] for sample in batch]
    #     return collated

    @staticmethod
    def train_collate_fn(batch):
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