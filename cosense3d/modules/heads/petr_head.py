from typing import List

import torch
from torch import nn

from cosense3d.modules import BaseModule
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.utils.common import inverse_sigmoid
from cosense3d.modules.utils.misc import SELayer_Linear, MLN
from cosense3d.modules.utils.positional_encoding import pos2posemb3d


class PETRHead(BaseModule):
    def __init__(self,
                 embed_dims,
                 pc_range,
                 code_weights,
                 num_classes,
                 num_reg_fcs=2,
                 num_pred=3,
                 use_logits=True,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.code_size = 10
        self.num_classes = num_classes
        self.num_reg_fcs = num_reg_fcs
        self.num_pred = num_pred
        self.use_logits = use_logits

        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.code_weights = nn.Parameter(torch.tensor(code_weights), requires_grad=False)

        self._init_layers()

    def _init_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(nn.Linear(self.embed_dims, self.num_classes))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, petr_feat, **kwargs):
        outs_dec = self.stack_data_from_list(petr_feat, 'outs_dec').permute(1, 0, 2, 3)
        reference_points = self.stack_data_from_list(petr_feat, 'ref_pts')
        outputs_classes = []
        outputs_coords = []
        for lvl in range(len(outs_dec)):
            out_dec = outs_dec[lvl]
            out_dec = torch.nan_to_num(out_dec)

            # assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](out_dec)
            tmp = self.reg_branches[lvl](out_dec)

            if self.use_logits:
                reference = reference_points.clone()
                reference[..., :3] = inverse_sigmoid(reference[..., :3])
                tmp[..., :reference.shape[-1]] = tmp[..., :reference.shape[-1]] + reference
                tmp[..., 0:3] = tmp[..., 0:3].sigmoid()
            else:
                reference = reference_points.clone()
                reference[..., :3] = reference[..., :3] * (
                        self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3]
                tmp[..., :reference.shape[-1]] = tmp[..., :reference.shape[-1]] + reference

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        if self.use_logits:
            all_bbox_preds[..., :3] = (all_bbox_preds[..., :3] * (
                    self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3])

        outs = [
            {
                'all_cls_scores': all_cls_scores[:, i],
                'all_bbox_preds': all_bbox_preds[:, i],
            } for i in range(len(petr_feat))
        ]

        return {self.scatter_keys[0]: outs}



    def loss(self, petr_out, local_boxes, local_labels, **kwargs):
        pass

