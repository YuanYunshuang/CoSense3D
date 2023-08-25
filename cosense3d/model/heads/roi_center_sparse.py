import copy, math
import numpy as np
import importlib
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from einops import rearrange

from cosense3d.model.utils import get_conv2d_layers
from cosense3d.model.submodules import centernet_utils
from cosense3d.model.losses import instantiate_losses
from cosense3d.model.utils import linear_layers, linear_last, indices2metric
from cosense3d.model.losses.common import weighted_smooth_l1_loss
from cosense3d.ops.iou3d_nms_utils import nms_gpu
from cosense3d.model.utils.me_utils import *


class RoiCenterSparse(nn.Module):
    def __init__(self, cfgs):
        super(RoiCenterSparse, self).__init__()
        for k, v in cfgs.items():
            setattr(self, k, v)
        # self.det_r = cfgs['data_info']['det_r']
        # self.voxel_size = cfgs['data_info']['voxel_size']
        update_me_essentials(self, cfgs['data_info'], cfgs['stride'])

        self.roi_layer = linear_last(self.input_channels, self.input_channels // 2, 2)

        if 'target_assigner' in cfgs:
            # update configs
            from cosense3d.model.utils.target_assigner import TargetAssigner
            self.tgt_assigner = TargetAssigner(cfgs['target_assigner'],
                                               batch_dict_key=self.__class__.__name__)
        instantiate_losses(self, self.loss_cfg)

        self.temp = 1.

    def forward(self, batch_dict):
        self.temp += 1.
        stensor = batch_dict[self.feature_src][f'p{self.stride}']
        coor = stensor['coor']
        feat = stensor['feat']
        centers = indices2metric(coor, self.voxel_size)

        out_dict = {
            f'p{self.stride}': {
                'centers': centers,
                'coor': coor,
                'feat': feat,
                'cls': self.roi_layer(feat),
            }
        }
        batch_dict[self.__class__.__name__] = out_dict

        # from tools.vis_tools import draw_boxes
        # draw_boxes(batch_dict, self.det_r)
        # pass

    def loss(self, batch_dict):
        tgt = self.tgt_assigner(batch_dict)
        src = batch_dict[self.__class__.__name__][f'p{self.stride}']

        # center loss
        cur_cls_src = rearrange(src['cls'], 'n d ... -> n ... d').contiguous()
        cur_cls_tgt = rearrange(tgt['center_cls'], 'n d ... -> n ... d').contiguous().float().squeeze(-2)
        cared = (cur_cls_tgt >= 0).any(dim=-1)
        cur_cls_src = cur_cls_src[cared]
        cur_cls_tgt = cur_cls_tgt[cared]

        tgt_pos = cur_cls_tgt
        tgt_neg = 1 - cur_cls_tgt

        cur_cls_tgt_onehot = torch.cat([tgt_neg, tgt_pos], dim=-1).contiguous()  # b, h, w, n_cls+1

        center_loss, _ = self.loss_center(
            cur_cls_src,
            cur_cls_tgt_onehot,
            2,
            temp=self.temp
        )
        center_loss = center_loss * self.loss_center_weight

        loss_dict = {
            'roi_center': center_loss,
        }
        return center_loss, loss_dict


