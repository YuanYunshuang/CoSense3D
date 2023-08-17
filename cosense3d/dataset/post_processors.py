import logging, os
from typing import Tuple, Any
import importlib

import torch, torch_scatter
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt

from cosense3d.utils import pclib, box_utils, vislib
from cosense3d.ops.utils import points_in_boxes_gpu_2d
from cosense3d.model.utils.me_utils import *
from cosense3d.model.utils import indices2metric


class Compose:
    """Composes several pre-processing modules together.
        Take care that these functions modify the input data directly.
    """

    def __init__(self, processes):
        self.processes = processes

    def __call__(self, *args):
        out_dict = {}
        for t in self.processes:
            out = t(*args)
            out_dict.update(out)
        return out_dict

    def set_log_dir(self, log_dir):
        for t in self.processes:
            t.set_log_dir(log_dir)


class PostProcess:
    def __init__(self, data_info, stride=None):
        update_me_essentials(self, data_info, stride)

    def __call__(self, batch_dict):
        raise NotImplementedError

    def set_log_dir(self, log_dir):
        self.log_dir = log_dir


class DetectionPostProcess(PostProcess):
    def __init__(self,
                 pc_range,
                 visualization=False
                 ):
        super(DetectionPostProcess, self).__init__()
        self.z_offset = pc_range[2]
        self.pc_range = pc_range
        self.vis = visualization
        self.out = {}

    def __call__(self, batch_dict):
        names = [batch_dict['scenario'][i] + '/' + batch_dict['frame'][i] \
                 for i in range(batch_dict['batch_size'])]
        objects = batch_dict['objects']
        gt_boxes = objects[:, [3, 4, 5, 6, 7, 8, 11]]
        # correct z offset
        gt_boxes[:, 2] += self.z_offset
        obj_idx = objects[:, 0]
        gt_boxes = [gt_boxes[obj_idx==b] \
                    for b in range(batch_dict['batch_size'])]
        preds = batch_dict['det_s2'] if 'det_s2' in batch_dict else batch_dict['det_s1']

        # correct z offset
        for i in range(len(preds)):
            preds[i]['box'][:, 2] += self.z_offset
        if self.vis:
            self.visualization(batch_dict)
        return {
            'name': names,
            'pred_boxes': preds,
            'gt_boxes': gt_boxes,
        }

    def set_log_dir(self, log_dir):
        self.log_dir = os.path.join(log_dir, 'Detection')
        os.makedirs(self.log_dir, exist_ok=True)

    def visualization(self, batch_dict):
        """
        Plot lidar0 points, gt_boxes and pred_boxes of the first sample in the batch.
        """
        ss = batch_dict['scenario'][0] + '_' + batch_dict['frame'][0]
        img_file = os.path.join(self.log_dir,  f"{ss}.png" )
        points = batch_dict['pcds'][batch_dict['pcds'][:, 0]==0, 1:]
        gt_boxes = batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11]]
        gt_boxes = gt_boxes[gt_boxes[:, 0] == 0, 1:]
        if 'det_s2' in batch_dict:
            pred_boxes = batch_dict['det_s2'][0]['box']
        else:
            pred_boxes = batch_dict['det_s1'][0]['box']
        vislib.draw_points_boxes_plt(
            pc_range=self.pc_range,
            points=points.cpu().numpy(),
            boxes_gt=gt_boxes.cpu().numpy(),
            boxes_pred=pred_boxes.cpu().numpy(),
            marker_size=0.5,
            filename=img_file
        )


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


class TrackingPostProcess:
    def __init__(self, **kwargs):
        super().__init__()
        m = importlib.import_module("scipy.optimize")
        self.assign_fn = getattr(m, 'linear_sum_assignment')
        self.out = {}

    def __call__(self, batch_dict):
        similarity = batch_dict['center_similarity']
        assignments = []
        for s in similarity:
            cost = (s[0].max() - s[0]).detach().cpu().numpy()
            inds = self.assign_fn(cost)
            assignments.append(inds)
        self.out['assignments'] = assignments

    def set_log_dir(self, log_dir):
        self.log_dir = os.path.join(log_dir, 'MOT')
        os.makedirs(self.log_dir, exist_ok=True)




