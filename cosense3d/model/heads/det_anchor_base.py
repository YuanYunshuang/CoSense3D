import importlib
import functools
import math
import torch_scatter
from cosense3d.model import utils
from cosense3d.model.utils import *

from cosense3d.ops.iou3d_nms_utils import nms_gpu, boxes_iou_bev, \
    aligned_boxes_iou3d_gpu,  boxes_iou3d_gpu
import cosense3d.model.losses.common as loss_fns


class DetAnchorBase(nn.Module):
    def __init__(self, cfgs):
        super(DetAnchorBase, self).__init__()
        for k, v in cfgs.items():
            setattr(self, k, v)
        self.device = getattr(self, 'device', 'cuda')

        self.target_assigner = self.get_target_assigner(cfgs['target_assigner'])
        self.num_cls = self.target_assigner.num_cls
        # loss functions
        self.cls_loss_fn = self.get_loss_fn(cfgs['loss_cfg']['cls'])
        self.reg_loss_fn = self.get_loss_fn(cfgs['loss_cfg']['reg'])
        self.scr_loss_fn = self.get_loss_fn(cfgs['loss_cfg']['scr'])
        # intermediate result
        self.out = {}

    def get_target_assigner(self, cfg):
        module = importlib.import_module(
            'model.utils.anchor_target_assigner'
        )
        target_assigner_inst = getattr(module, cfg['name'])(
            cfg, self.det_r, self.voxel_size, self.stride)
        return target_assigner_inst

    def get_loss_fn(self, cfg):
        return functools.partial(
            getattr(loss_fns, cfg['name']),
            **cfg.get('args', {})
        )

    def forward(self, batch_dict):
        raise NotImplementedError

    def reg_to_bbx(self, batch_dict):
        batch_size = batch_dict['batch_size']
        cls = self.out['cls'].sigmoid()
        scores = self.out['scores'].sigmoid()
        ious_pred = scores[..., :self.num_cls].view(-1, self.num_cls)
        cls_scores = cls.view(-1, self.num_cls)

        anchors = self.target_assigner.gen_batch_anchors(
            batch_size, self.xy
        )
        indices = [torch.ones(len(acr)) * i for i, acr in enumerate(anchors)]
        indices = torch.cat(indices, dim=0).to(cls_scores.device)
        anchors = torch.cat(anchors, dim=0).to(cls_scores.device)
        boxes_enc = self.out['reg'].view(len(anchors), self.num_cls, 10)

        # cls_tgt, iou_tgt, dir_tgt, reg_tgt = \
        #     self.target_assigner(
        #     batch_dict['objects'],
        #     batch_dict['batch_size'],
        #     self.xy
        # )
        # boxes_enc[cls_tgt>0] = reg_tgt + 1e-3

        dir_scores = scores[..., self.num_cls:].view(
            len(anchors), self.num_cls, 2)
        # dir_scores[cls_tgt>0] = dir_tgt
        dec_boxes = self.target_assigner.box_coder.decode(
            anchors, boxes_enc, dir_scores
        )
        boxes = []
        ious = []
        scrs = []
        for b in range(batch_size):
            mask = indices == b
            cur_boxes = dec_boxes[mask].view(-1, 7)
            cur_scores = cls_scores[mask].view(-1)
            # cur_scores = cls_tgt[mask].view(-1)
            cur_ious = ious_pred[mask].view(-1)
            # remove abnormal boxes
            mask = torch.logical_and(
                cur_boxes[:, 3:6] > 1,
                cur_boxes[:, 3:6] < 10
            ).all(dim=-1)

            keep = torch.logical_and(cur_scores > 0.5, mask)
            if keep.sum() == 0:
                boxes.append(torch.empty((0, 7), device=cur_boxes.device))
                scrs.append(torch.empty((0,), device=cur_boxes.device))
                ious.append(torch.empty((0,), device=cur_boxes.device))
                continue
            cur_boxes = cur_boxes[keep]
            cur_scores = cur_scores[keep]
            cur_ious = cur_ious[keep]

            cur_scores_rectified = cur_scores * cur_ious ** 4
            keep = nms_gpu(cur_boxes, cur_scores_rectified,
                           thresh=0.01, pre_maxsize=500)
            boxes.append(cur_boxes[keep])
            scrs.append(cur_scores[keep])
            ious.append(cur_ious[keep])

        self.out['pred_box'] = boxes
        self.out['pred_scr'] = scrs
        self.out['pred_iou'] = ious

    def loss(self, batch_dict):
        cls_tgt, iou_tgt, dir_tgt, reg_tgt = \
            self.target_assigner(
            batch_dict['objects'],
            batch_dict['batch_size'],
            self.xy
        )
        cared = cls_tgt >= 0
        batch_size = batch_dict['batch_size']
        # cls loss
        cls_pred = self.out['cls'].reshape(-1, self.num_cls)
        loss_cls = self.cls_loss_fn(cls_pred[cared],
                                    cls_tgt[cared]).mean()
        loss_dict = {'bx_cls': loss_cls}
        loss = loss_cls
        pos = cls_tgt > 0
        if pos.sum() > 0:
            # reg loss
            reg_src = self.out['reg'].reshape(-1, self.num_cls, 10)[pos]
            loss_reg = self.reg_loss_fn(reg_src, reg_tgt).mean()
            # score loss
            scores = self.out['scores'].sigmoid().reshape(-1, self.num_cls * 3)
            ious = scores[..., :self.num_cls][cared]
            dirs = scores[..., self.num_cls:].view(-1, self.num_cls, 2)[pos]
            loss_iou = self.scr_loss_fn(ious, iou_tgt[cared]).mean()
            loss_dir = self.scr_loss_fn(dirs, dir_tgt).mean()

            loss = loss + loss_iou + loss_dir + loss_reg
            loss_dict.update({
                'bx_iou': loss_iou,
                'bx_dir': loss_dir,
                'bx_reg': loss_reg,
            })
        return loss, loss_dict





