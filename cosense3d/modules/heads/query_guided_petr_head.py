from typing import List
import os
import torch
from torch import nn

from cosense3d.modules import BaseModule
from cosense3d.modules.plugin import build_plugin_module
from cosense3d.modules.utils.common import inverse_sigmoid
from cosense3d.utils.misc import multi_apply
from cosense3d.utils.box_utils import normalize_bbox, denormalize_bbox
from cosense3d.modules.losses import build_loss
from cosense3d.modules.losses.edl import pred_to_conf_unc


class QueryGuidedPETRHead(BaseModule):
    def __init__(self,
                 embed_dims,
                 pc_range,
                 code_weights,
                 num_classes,
                 cls_assigner,
                 box_assigner,
                 loss_cls,
                 loss_box,
                 num_reg_fcs=1,
                 num_pred=1,
                 use_logits=False,
                 reg_channels=None,
                 **kwargs):
        super().__init__(**kwargs)
        self.embed_dims = embed_dims
        self.reg_channels = {}
        if reg_channels is None:
            self.code_size = 10
        else:
            for c in reg_channels:
                name, channel = c.split(':')
                self.reg_channels[name] = int(channel)
            self.code_size = sum(self.reg_channels.values()) + 2
        self.num_classes = num_classes
        self.num_reg_fcs = num_reg_fcs
        self.num_pred = num_pred
        self.use_logits = use_logits

        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.code_weights = nn.Parameter(torch.tensor(code_weights), requires_grad=False)

        self.box_assigner = build_plugin_module(box_assigner)
        self.cls_assigner = build_plugin_module(cls_assigner)

        self.loss_cls = build_loss(**loss_cls)
        self.loss_box = build_loss(**loss_box)

        self._init_layers()
        self.init_weights()

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
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, 2.0)
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.init.xavier_uniform_(m.weight)
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

            outputs_classes.append(pred_cls)
            outputs_coords.append(pred_reg)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_reg = torch.stack(outputs_coords)
        if self.use_logits:
            all_bbox_reg[..., :3] = (all_bbox_reg[..., :3] * (
                    self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3])

        reference_points = reference_points * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
        pred_boxes = self.get_pred_boxes(all_bbox_reg, reference_points)

        outs = [
            {
                'all_cls_scores': all_cls_scores[:, i],
                'all_bbox_reg': all_bbox_reg[:, i],
                'ref_pts': reference_points[i],
                'all_bbox_preds': pred_boxes[:, i],
            } for i in range(len(feat_in))
        ]

        if not self.training:
            dets = self.get_predictions(all_cls_scores, pred_boxes)
            for i, out in enumerate(outs):
                out['preds'] = dets[i]

        return {self.scatter_keys[0]: outs}

    def loss(self, petr_out, gt_boxes, gt_labels, det, **kwargs):
        epoch = kwargs.get('epoch', 0)
        cls_scores = self.stack_data_from_list(petr_out, 'all_cls_scores').flatten(0, 1)
        bbox_reg = self.stack_data_from_list(petr_out, 'all_bbox_reg').flatten(0, 1)
        ref_pts = self.stack_data_from_list(petr_out, 'ref_pts').unsqueeze(1).repeat(
            1, self.num_pred, 1, 1).flatten(0, 1)
        gt_boxes = [boxes for boxes in gt_boxes for _ in range(self.num_pred)]
        # gt_velos = [boxes[:, 7:] for boxes in gt_boxes for _ in range(self.num_pred)]
        gt_labels = [labels for labels in gt_labels for _ in range(self.num_pred)]

        # cls loss
        cls_tgt = multi_apply(self.cls_assigner.assign,
                              ref_pts, gt_boxes, gt_labels, **kwargs)
        cls_src = cls_scores.view(-1, self.num_classes)

        if kwargs['itr'] % 500 == 0:
            from cosense3d.utils.vislib import draw_points_boxes_plt, plt
            points = ref_pts[0].detach().cpu().numpy()
            boxes = gt_boxes[0][:, :7].detach().cpu().numpy()
            scores = pred_to_conf_unc(cls_scores[0], 'exp')[0]
            scores = scores[:, 1].detach().cpu().numpy()
            ax = draw_points_boxes_plt(
                pc_range=self.pc_range.tolist(),
                points=points,
                boxes_gt=boxes,
                return_ax=True
            )
            ax = draw_points_boxes_plt(
                pc_range=self.pc_range.tolist(),
                points=points[cls_tgt[0].squeeze().detach().cpu().numpy() > 0],
                points_c="green",
                ax=ax,
                return_ax=True
            )
            ax = draw_points_boxes_plt(
                pc_range=self.pc_range.tolist(),
                points=points[scores > 0.5],
                points_c="magenta",
                ax=ax,
                return_ax=True
            )
            plt.savefig(f"{os.environ['HOME']}/Downloads/tmp.jpg")
            plt.close()

        cls_tgt = torch.cat(cls_tgt, dim=0)
        cared = (cls_tgt >= 0).any(dim=-1)
        cls_src = cls_src[cared]
        cls_tgt = cls_tgt[cared]

        # convert one-hot to labels
        cur_labels = torch.zeros_like(cls_tgt[..., 0]).long()
        lbl_inds, cls_inds = torch.where(cls_tgt)
        cur_labels[lbl_inds] = cls_inds + 1

        avg_factor = max((cur_labels > 0).sum(), 1)
        loss_cls = self.loss_cls(
            cls_src,
            cur_labels,
            temp=epoch,
            avg_factor=avg_factor
        )

        # box loss
        # pad ref pts with batch index
        box_tgt = self.box_assigner.assign(
            self.cat_data_from_list(ref_pts, pad_idx=True),
            self.cat_data_from_list(gt_boxes, pad_idx=True),
            self.cat_data_from_list(gt_labels)
        )
        ind = box_tgt['idx'][0]  # only one head
        loss_box = 0
        bbox_reg = bbox_reg.view(-1, self.code_size)
        if ind.shape[1] > 0:
            ptr = 0
            for reg_name, reg_dim in self.reg_channels.items():
                pred_reg = bbox_reg[:, ptr:ptr+reg_dim].contiguous()
                if reg_name == 'scr':
                    pred_reg = pred_reg.sigmoid()
                cur_reg_src = pred_reg[box_tgt['valid_mask'][0]]
                if reg_name == 'vel':
                    cur_reg_tgt = box_tgt['aux'][0] * 0.1
                else:
                    cur_reg_tgt = box_tgt[reg_name][0]  # N, C
                cur_loss = self.loss_box(cur_reg_src, cur_reg_tgt)

                loss_box = loss_box + cur_loss
                ptr += reg_dim

        return {
            'petr_cls_loss': loss_cls,
            'petr_box_loss': loss_box,
            'petr_cls_max': pred_to_conf_unc(cls_src, 'exp')[0][..., 1].max()
        }

    def get_pred_boxes(self, bbox_preds, ref_pts):
        reg = {}

        ptr = 0
        for reg_name, reg_dim in self.reg_channels.items():
            reg[reg_name] = bbox_preds[..., ptr:ptr + reg_dim].contiguous()
            ptr += reg_dim

        boxes = self.box_assigner.box_coder.decode(ref_pts[None], reg)
        return torch.cat([boxes, reg['vel']], dim=-1)

    def get_predictions(self, cls_scores, bbox_preds):
        l, b, n = cls_scores.shape[:3]
        conf = pred_to_conf_unc(cls_scores[-1], 'exp')[0]
        scores = conf[..., 1:].sum(dim=-1)
        labels = conf[..., 1:].argmax(dim=-1)
        pos = scores > self.box_assigner.center_threshold

        dets = []
        for i in range(b):
            dets.append({
                'box': bbox_preds[-1][i][pos[i]],
                'scr': scores[i][pos[i]],
                'lbl': labels[i][pos[i]],
                'idx': torch.ones_like(labels[i][pos[i]]) * i
            })
        return dets


