from typing import List

import torch
from torch import nn
from cosense3d.modules import BaseModule
from cosense3d.modules import plugin
from cosense3d.modules.losses import build_loss
from cosense3d.utils.misc import multi_apply


class DetAnchorDense(BaseModule):
    def __init__(self,
                 in_channels,
                 loss_cls,
                 loss_box,
                 num_classes=1,
                 target_assigner=None,
                 **kwargs):
        super(DetAnchorDense, self).__init__(**kwargs)
        assert num_classes == 1, 'currently only support binary classification.'
        self.num_classes = num_classes
        self.target_assigner = plugin.build_plugin_module(target_assigner)
        self.num_anchors = self.target_assigner.num_anchors
        self.code_size = self.target_assigner.box_coder.code_size
        self.cls_head = nn.Conv2d(in_channels, self.num_anchors, kernel_size=1)
        self.reg_head = nn.Conv2d(in_channels, self.code_size * self.num_anchors, kernel_size=1)
        self.loss_cls = build_loss(**loss_cls)
        self.loss_box = build_loss(**loss_box)

    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                nn.utils.init.xavier_uniform_(m)
        self._is_init = True

    def forward(self, bev_feat_list, **kwargs):
        bev_feat = torch.stack(bev_feat_list, dim=0)
        cls = self.cls_head(bev_feat)
        reg = self.reg_head(bev_feat)

        out = {'cls': cls, 'reg': reg}
        return self.format_output(out, len(bev_feat))

    def format_output(self, out_dict, B):
        out_list = []
        for i in range(B):
            out_list.append({k: v[i] for k, v in out_dict.items()})
        return {self.scatter_keys[0]: out_list}

    def loss(self, preds, gt_boxes, gt_labels, **kwargs):
        pred_cls = self.stack_data_from_list(preds, 'cls')
        pred_reg = self.stack_data_from_list(preds, 'reg')
        pred_cls = pred_cls.permute(0, 3, 2, 1).reshape(-1)
        pred_reg = pred_reg.permute(0, 3, 2, 1).reshape(-1, 7)
        cls_tgt, reg_tgt, _ = multi_apply(
            self.target_assigner.assign, gt_boxes)
        cls_tgt = torch.cat(cls_tgt, dim=0)
        reg_tgt = torch.cat(reg_tgt, dim=0)

        pos_mask = cls_tgt > 0
        cared = cls_tgt >= 0
        avg_factor = max(pos_mask.sum(), 1)
        # downsample negative
        # neg_inds = torch.where(cls_tgt == 0)[0]
        # neg_inds = neg_inds[torch.randperm(len(neg_inds))[:avg_factor * 5]]
        # cared[neg_inds] = True

        labels = pos_mask.new_full((len(pos_mask), ), self.num_classes, dtype=torch.long)
        labels[pos_mask] = 0

        loss_cls = self.loss_cls(pred_cls[cared].view(-1, 1), labels[cared],
                                 avg_factor=avg_factor)

        reg_preds_sin, reg_tgts_sin = self.add_sin_difference(pred_reg[pos_mask], reg_tgt)
        loss_box = self.loss_box(reg_preds_sin, reg_tgts_sin,
                                 avg_factor=avg_factor / reg_preds_sin.shape[-1])

        return {
            'cls_loss': loss_cls,
            'box_loss': loss_box
        }

    @staticmethod
    def add_sin_difference(boxes1, boxes2, dim=6):
        assert dim != -1
        rad_pred_encoding = torch.sin(boxes1[..., dim:dim + 1]) * \
                            torch.cos(boxes2[..., dim:dim + 1])
        rad_tg_encoding = torch.cos(boxes1[..., dim:dim + 1]) * \
                          torch.sin(boxes2[..., dim:dim + 1])

        boxes1 = torch.cat([boxes1[..., :dim], rad_pred_encoding,
                            boxes1[..., dim + 1:]], dim=-1)
        boxes2 = torch.cat([boxes2[..., :dim], rad_tg_encoding,
                            boxes2[..., dim + 1:]], dim=-1)
        return boxes1, boxes2



