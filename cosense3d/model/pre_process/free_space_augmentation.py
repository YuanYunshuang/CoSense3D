import os
import logging
import pickle

import numpy as np
import torch
import cv2
from torchvision.transforms import ToTensor
from PIL import Image
from torch_scatter import scatter_mean

from cosense3d.utils import pclib, box_utils
from cosense3d.dataset.data_utils import project_points_by_matrix
from cosense3d.model.pre_process import PreProcessorBase



class FreeSpaceAugmentation(PreProcessorBase):
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs:
            free_space_h: float
            free_space_step: int
            free_space_d: float
        """
        super().__init__(**kwargs)

    def __call__(self, batch_dict):
        # transform lidar points from ego to local, and to torch tensor to speed up runtime
        # TODO: currently only support intermediate fusion
        lidar = batch_dict['pcds'][:, 1:].clone()
        lidar_idx = batch_dict['pcds'][:, 0]
        tf_cav2ego = batch_dict['tf_cav2ego']
        if batch_dict.get('projected', False) and tf_cav2ego is not None:
            lidar[:, :3] = self.lidar_transform(lidar[:, :3], lidar_idx, tf_cav2ego)
        # get point lower than z_min=1.5m
        z_min = self.free_space_h
        m = lidar[:, 2] < z_min
        points = lidar[m][:, :3]
        points_idx = lidar_idx[m]

        # generate free space points based on points
        d = torch.norm(points[:, :2], dim=1).reshape(-1, 1)
        free_space_d = self.free_space_d
        free_space_step = self.free_space_step
        delta_d = torch.arange(1, free_space_d + free_space_step,
                               free_space_step,
                               device=lidar.device).reshape(1, -1)
        steps = delta_d.shape[1]
        tmp = (d - delta_d) / d  # Nxsteps
        xyz_new = points[:, None, :] * tmp[:, :, None]  # Nx3x3
        points_idx = torch.tile(points_idx.reshape(-1, 1), (1, steps))
        ixyz = torch.cat([points_idx.reshape(-1, steps, 1), xyz_new], dim=-1)

        # 1.remove free space points with negative distances to lidar center
        # 2.remove free space points higher than z_min
        # 3.remove duplicated points with resolution 1m
        ixyz = ixyz[tmp > 0]
        ixyz = ixyz[(ixyz[..., 3] < z_min)]
        ixyz = ixyz[torch.randperm(len(ixyz))]
        selected = torch.unique(torch.floor(ixyz / 2).long(), return_inverse=True, dim=0)[1]
        ixyz = scatter_mean(src=ixyz, index=selected, dim=0)

        # pad free space point intensity as -1
        ixyz = torch.cat([ixyz, - torch.ones_like(ixyz[:, :2])], dim=-1)
        # project free space points to ego frame
        if batch_dict.get('projected', False) and tf_cav2ego is not None:
            ixyz[:, 1:4] = self.lidar_transform(ixyz[:, 1:4], ixyz[:, 0], tf_cav2ego, ego2local=False)

        # transform augmented points back to ego and numpy
        # lidar = torch.cat([batch_dict['pcds'][:, 1:], ixyz[:, 1:]], dim=0)
        # lidar_idx = torch.cat([lidar_idx, ixyz[:, 0]], dim=0)
        batch_dict['pcds'] = torch.cat([batch_dict['pcds'], ixyz], dim=0)
        # features = batch_dict['features']

    def lidar_transform(self,
                        lidar: torch.Tensor,
                        lidar_idx: torch.Tensor,
                        tf_matrices: torch.Tensor,
                        ego2local=True):
        lidar_tf = []
        lidar_tf_idx = []
        uniq_idx = [int(x.item()) for x in torch.unique(lidar_idx.int())]
        for i in uniq_idx:
            mat = torch.inverse(tf_matrices[i]) if ego2local else tf_matrices[i]
            mask = lidar_idx.int() == i
            lidar[mask] = project_points_by_matrix(lidar[mask], mat)
            # cur_idx = lidar_idx[mask]
            # lidar_tf_idx.append(cur_idx)
            # lidar_tf.append(cur_lidar)

        # lidar_tf_idx = torch.cat(lidar_tf_idx, dim=0)
        # lidar_tf = torch.cat(lidar_tf, dim=0)
        return lidar



