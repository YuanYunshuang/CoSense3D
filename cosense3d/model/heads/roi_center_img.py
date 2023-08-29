import torch
from torch import nn

from cosense3d.model.heads.sequence_runner import SequenceRunner
from mmdet3d_plugin.models.dense_heads.focal_head import FocalHead
from mmdet3d_plugin.models.utils.misc import locations
from cosense3d.py_cfg.stream_petr import img_roi_head


class RoiCenterImg(SequenceRunner):
    def __init__(self, cfgs):
        super().__init__(**cfgs)
        self.focal_head = FocalHead(**img_roi_head)
        self.stride = self.focal_head.stride

    def prepare_location(self, img_metas=None, **data):
        """Patch centers in img ratio metric"""
        pad_h, pad_w, _ = img_metas[0]['pad_shape'][0]
        bs, n = data['img_feats'].shape[:2]
        x = data['img_feats'].flatten(0, 1)
        location = locations(x, self.stride, pad_h, pad_w)[None].repeat(bs*n, 1, 1, 1)
        return location

    def frame_forward(self,
                      skip=False,
                      seq_len=None,
                      batch_size=None,
                      **data):
        location = self.prepare_location(**data)
        out = None
        if not skip:
            out = self.focal_head(location, **data)

        return {
            'location': location,
            'roi': out
        }

    def _loss(self, out_dict,
              gt_bboxes=None,
              gt_labels=None,
              centers2d=None,
              depths=None,
              img_metas=None,
              **data):
        outs_roi = out_dict['roi']
        loss_dict = self.focal_head.loss(gt_bboxes, gt_labels, centers2d, depths, outs_roi, img_metas)

        return loss_dict

    def get_frame_gt(self, seq_idx,
                     gt_bboxes=None,
                     gt_labels=None,
                     centers2d=None,
                     depths=None,
                     img_metas=None,
                     **batch_dict):
        return (gt_bboxes[seq_idx], gt_labels[seq_idx], centers2d[seq_idx],
                depths[seq_idx], img_metas[seq_idx])



