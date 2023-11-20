"""
Seg head for bev understanding
"""

import torch
import torch.nn as nn
from einops import rearrange

from cosense3d.modules import BaseModule
from cosense3d.modules.losses import build_loss


class BevSegHead(BaseModule):
    def __init__(self, target, input_dim, output_class, loss_cls, **kwargs):
        super(BevSegHead, self).__init__(**kwargs)
        self.target = target
        if 'dynamic' in self.target:
            self.dynamic_head = nn.Conv2d(input_dim,
                                          output_class,
                                          kernel_size=3,
                                          padding=1)
        if 'static' in self.target:
            self.static_head = nn.Conv2d(input_dim,
                                         output_class,
                                         kernel_size=3,
                                         padding=1)
        self.loss_cls = build_loss(**loss_cls)

    def forward(self,  x, **kwargs):
        x = self.stack_data_from_list(x)
        out_dict = {}
        if 'dynamic' in self.target:
            out_dict['dynamic_seg'] = self.dynamic_head(x)

        if 'static' in self.target:
            out_dict['static_seg'] = self.static_head(x)

        output_list = self.compose_result_list(out_dict, len(x))
        return {self.scatter_keys[0]: output_list}

    def loss(self, preds, dynamic_bev, **kwargs):
        dynamic_bev_preds = self.stack_data_from_list(preds, 'dynamic_seg')
        dynamic_bev_gt = torch.stack(dynamic_bev, dim=0)
        loss_dict = self.loss_cls(
            dynamic_pred=dynamic_bev_preds,
            dynamic_gt=dynamic_bev_gt
        )
        return loss_dict


