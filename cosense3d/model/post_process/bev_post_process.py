
import logging, os
from typing import Tuple, Any
import importlib

import torch, torch_scatter
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt

from cosense3d.utils import pclib, box_utils, vislib
from cosense3d.model.post_process import PostProcess
from cosense3d.ops.utils import points_in_boxes_gpu_2d
from cosense3d.model.utils.me_utils import *
from cosense3d.model.utils import indices2metric


class BevPostProcess(PostProcess):
    def __init__(self,
                 data_info,
                 stride,
                 edl=True):
        super(BevPostProcess, self).__init__(data_info, stride)
        self.edl = edl

    def __call__(self, batch_dict):
        bev_dict = batch_dict['bev']
        conf_map, unc_map = bev_sparse_to_dense(self, bev_dict)
        gt_bev = self.gen_box_bev(
            batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11]],
            unc_map.shape,
            unc_map.device
        )
        out_dict = {
            'bev': {
                'conf': conf_map,
                'unc': unc_map,
                'gt': gt_bev
            }
        }

        return out_dict

    def gen_box_bev(self, boxes, shape, device='cpu'):
        gt_bev = torch.zeros(shape, device=device).bool()
        B, X, Y = torch.where(torch.logical_not(gt_bev))
        points = torch.stack([B,
                              X + self.offset_sz_x,
                              Y + self.offset_sz_y,
                              torch.zeros_like(X)],
                             dim=-1)
        points = indices2metric(points, self.res)
        box_idx_of_pts = points_in_boxes_gpu_2d(points, boxes, batch_size=shape[0])
        mask = box_idx_of_pts >= 0
        gt_bev[B, X, Y] = mask

        # import matplotlib.pyplot as plt
        # plt.imshow(gt_bev[0].cpu().numpy())
        # plt.show()
        # plt.close()
        return gt_bev