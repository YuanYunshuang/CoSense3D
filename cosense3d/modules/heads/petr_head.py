from typing import List

import torch
from torch import nn

from cosense3d.modules import BaseModule
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.utils.common import inverse_sigmoid
from cosense3d.utils.misc import multi_apply
from cosense3d.utils.box_utils import normalize_bbox
from cosense3d.modules.losses import build_loss


class PETRHead(BaseModule):
    def __init__(self,
                 embed_dims,
                 pc_range,
                 code_weights,
                 num_classes,
                 box_assigner,
                 loss_cls,
                 loss_bbox,
                 loss_iou=None,
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

        self.box_assigner = build_plugin_module(box_assigner)

        self.loss_cls = build_loss(**loss_cls)
        self.loss_bbox = build_loss(**loss_bbox)
        if loss_iou is not None:
            self.loss_iou = build_loss(**loss_iou)

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

    def forward(self, feat_in, **kwargs):
        outs_dec = self.stack_data_from_list(feat_in, 'outs_dec').permute(1, 0, 2, 3)
        reference_points = self.stack_data_from_list(feat_in, 'ref_pts')
        pos_dim = reference_points.shape[-1]
        outputs_classes = []
        outputs_coords = []
        for lvl in range(len(outs_dec)):
            out_dec = outs_dec[lvl]
            out_dec = torch.nan_to_num(out_dec)

            pred_cls = self.cls_branches[lvl](out_dec)
            pred_reg = self.reg_branches[lvl](out_dec)

            if self.use_logits:
                reference = inverse_sigmoid(reference_points.clone())
                pred_reg[..., :pos_dim] += reference
                pred_reg[..., :3] = pred_reg[..., :3].sigmoid()
            else:
                reference = reference_points.clone()
                reference[..., :pos_dim] = (reference[..., :pos_dim] * (
                        self.pc_range[3:3+pos_dim] - self.pc_range[0:pos_dim])
                                            + self.pc_range[0:pos_dim])
                pred_reg[..., :pos_dim] = pred_reg[..., :pos_dim] + reference

            outputs_classes.append(pred_cls)
            outputs_coords.append(pred_reg)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        if self.use_logits:
            all_bbox_preds[..., :3] = (all_bbox_preds[..., :3] * (
                    self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3])

        outs = [
            {
                'all_cls_scores': all_cls_scores[:, i],
                'all_bbox_preds': all_bbox_preds[:, i],
            } for i in range(len(feat_in))
        ]

        return {self.scatter_keys[0]: outs}

    def loss(self, petr_out, gt_boxes, gt_labels, **kwargs):
        cls_scores = self.stack_data_from_list(petr_out, 'all_cls_scores').flatten(0, 1)
        bbox_preds = self.stack_data_from_list(petr_out, 'all_bbox_preds').flatten(0, 1)
        gt_boxes = [boxes for boxes in gt_boxes for _ in range(self.num_pred)]
        gt_labels = [labels for labels in gt_labels for _ in range(self.num_pred)]
        code_weights = [self.code_weights] * len(gt_labels)

        num_gts, assigned_gt_inds, assigned_labels = multi_apply(
            self.box_assigner.assign,
            bbox_preds,
            cls_scores,
            gt_boxes,
            gt_labels,
            code_weights
        )

        cared_pred_boxes = []
        aligned_bboxes_gt = []
        aligned_labels = []
        mask = []
        for i in range(len(cls_scores)):
            pos_mask = assigned_gt_inds[i] > 0
            mask.append(pos_mask)
            pos_inds = assigned_gt_inds[i][pos_mask] - 1
            boxes = bbox_preds[i][pos_mask]
            cared_pred_boxes.append(boxes)
            aligned_bboxes_gt.append(gt_boxes[i][pos_inds])
            labels = pos_mask.new_full((len(pos_mask), ), self.num_classes, dtype=torch.long)
            labels[pos_mask] = gt_labels[i][pos_inds]
            aligned_labels.append(labels)

        cared_pred_boxes = torch.cat(cared_pred_boxes, dim=0)
        aligned_bboxes_gt = torch.cat(aligned_bboxes_gt, dim=0)
        aligned_labels = torch.cat(aligned_labels, dim=0)
        mask = torch.cat(mask, dim=0)

        cls_avg_factor = max(sum(num_gts), 1)
        loss_cls = self.loss_cls(cls_scores.reshape(-1, cls_scores.shape[-1]),
                                 aligned_labels,  avg_factor=cls_avg_factor)

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))[mask]
        normalized_bbox_targets = normalize_bbox(aligned_bboxes_gt)
        isnotnan = torch.isfinite(bbox_preds).all(dim=-1)
        bbox_weights = torch.ones_like(cared_pred_boxes) * self.code_weights
        loss_box = self.loss_bbox(cared_pred_boxes[isnotnan],
                                  normalized_bbox_targets[isnotnan],
                                  bbox_weights[isnotnan])
        return {
            'petr_cls_loss': loss_cls,
            'petr_box_loss': loss_box
        }



