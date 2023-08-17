
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


class DetectionPostProcess(PostProcess):
    def __init__(self,
                 data_info,
                 stride=None,
                 visualization=False
                 ):
        super(DetectionPostProcess, self).__init__(data_info, stride)
        self.z_offset = self.lidar_range[2]
        self.vis = visualization

    def __call__(self, batch_dict):
        detections = self.formatting_detection_result(batch_dict)
        detections = self.get_confidence_for_boxes(batch_dict, detections)
        if self.vis:
            self.visualization(batch_dict, detections)
        return {'detections': detections}

    def formatting_detection_result(self, batch_dict):
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
        return {
            'name': names,
            'pred_boxes': preds,
            'gt_boxes': gt_boxes,
        }

    def get_confidence_for_boxes(self, batch_dict, detections):
        conf = batch_dict['bev']['conf']
        centers = batch_dict['bev']['centers']
        pred_boxes = detections['pred_boxes']

        confidences = []
        for b in range(batch_dict['batch_size']):
            ctr_mask = centers[:, 0] == b
            cur_conf = conf[ctr_mask]
            cur_centers = F.pad(centers[ctr_mask], (0, 1, 0, 0), mode="constant", value=0)
            cur_centers[:, 0] = 0
            cur_boxes = F.pad(pred_boxes[b]['box'], (1, 0, 0, 0), mode="constant", value=0)
            normalizer = (cur_boxes[:, 4] * cur_boxes[:, 5]) / (self.res[0] * self.res[1])

            box_idx_of_pts_a = points_in_boxes_gpu_2d(
                cur_centers, cur_boxes, batch_size=1
            )
            ma = box_idx_of_pts_a >= 0
            cur_boxes[:, 4:6] *= 1.5
            box_idx_of_pts_b = points_in_boxes_gpu_2d(
                cur_centers, cur_boxes, batch_size=1
            )
            mb = box_idx_of_pts_b >= 0

            conf_a = torch.zeros_like(cur_boxes[:, :2])
            conf_b = torch.zeros_like(cur_boxes[:, :2])
            torch_scatter.scatter(cur_conf[ma], box_idx_of_pts_a[ma], dim=0, reduce='sum', out=conf_a)
            torch_scatter.scatter(cur_conf[mb], box_idx_of_pts_b[mb], dim=0, reduce='sum', out=conf_b)
            conf_box = conf_a[:, 1] / torch.maximum(conf_b[:, 1], normalizer)
            detections['pred_boxes'][b]['avg_conf'] = conf_box

            if self.vis:
                detections['pred_boxes'][b]['conf_points'] = cur_centers[mb]
                detections['pred_boxes'][b]['conf_values'] = cur_conf[mb]
        return detections

    def set_log_dir(self, log_dir):
        self.log_dir = os.path.join(log_dir, 'Detection')
        os.makedirs(self.log_dir, exist_ok=True)

    def visualization(self, batch_dict, out_dict):
        """
        Plot lidar points, gt_boxes and pred_boxes of the first sample in the batch.
        """
        ss = batch_dict['scenario'][0] + '_' + batch_dict['frame'][0]
        img_file = os.path.join(self.log_dir,  f"{ss}.png" )
        if batch_dict['pcds'][:, 0].max() + 1 > batch_dict['batch_size']:
            points = batch_dict['pcds'][batch_dict['pcds'][:, 0] < batch_dict['num_cav'][0], 1:]
        else:
            points = batch_dict['pcds'][batch_dict['pcds'][:, 0]==0, 1:]
        gt_boxes = out_dict['gt_boxes'][0].cpu().numpy()
        pred_boxes = out_dict['pred_boxes'][0]['box'].cpu().numpy()
        pred_conf = out_dict['pred_boxes'][0]['avg_conf'].cpu().numpy()
        conf_points = out_dict['pred_boxes'][0]['conf_points'].cpu().numpy()
        conf_values = out_dict['pred_boxes'][0]['conf_values'].cpu().numpy()[:, 1]
        box_labels = [f'{int(conf * 100):d}' for conf in pred_conf]

        ax = vislib.draw_points_boxes_plt(
            pc_range=self.lidar_range,
            points=points.cpu().numpy(),
            marker_size=0.5,
            return_ax=True,
        )
        ax.scatter(conf_points[:, 1], conf_points[:, 2],
                   c=conf_values, cmap='cool', s=1, vmin=0, vmax=1)
        vislib.draw_points_boxes_plt(
            pc_range=self.lidar_range,
            boxes_gt=gt_boxes,
            boxes_pred=pred_boxes,
            bbox_pred_label=box_labels,
            marker_size=0.5,
            ax=ax,
            filename=img_file
        )
