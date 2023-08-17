import copy, math
import numpy as np
import importlib
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_

from cosense3d.model.utils import get_conv2d_layers
from cosense3d.model.submodules import centernet_utils
from cosense3d.model.losses.centernet_loss import neg_loss_cornernet
from cosense3d.model.losses.common import weighted_smooth_l1_loss
from cosense3d.ops.iou3d_nms_utils import nms_gpu


def get_head_conv(in_channel, out_channel, use_bias):
    seq = get_conv2d_layers(
        'Conv2d', in_channel, in_channel // 2,
        n_layers=1, kernel_size=[3], stride=[1], padding=[1], bias=use_bias,
        sequential=False
    )
    seq.append(nn.Conv2d(in_channel // 2, out_channel,
                         kernel_size=3, stride=1, padding=1, bias=use_bias))
    return nn.Sequential(*seq)


class UnitedClsHead(nn.Module):
    def __init__(self,
                 class_names_each_head,
                 in_channel,
                 one_hot_encoding=True,
                 use_bias=False,
                 **kwargs):
        super().__init__()
        n_cls = sum([len(c) for c in class_names_each_head])
        out_channel = n_cls + 1 if one_hot_encoding else n_cls
        self.head = get_head_conv(in_channel, out_channel, use_bias)

    def forward(self, x):
        return [self.head(x)]


class SeparatedClsHead(nn.Module):
    def __init__(self,
                 class_names_each_head,
                 in_channel,
                 one_hot_encoding=True,
                 use_bias=False,
                 **kwargs):
        super().__init__()
        self.n_head = len(class_names_each_head)
        for i, cls_names in enumerate(class_names_each_head):
            out_channel = len(cls_names)
            if one_hot_encoding:
                out_channel += 1
            setattr(self, f'head_{i}',
                    get_head_conv(in_channel, out_channel, use_bias))

    def forward(self, x):
        out = []
        for i in range(self.n_head):
            out.append(getattr(self, f'head_{i}')(x))
        return out


class UnitedRegHead(nn.Module):
    def __init__(self,
                 reg_channels,
                 in_channel,
                 combine_channels=True,
                 sigmoid_keys=None,
                 use_bias=False,
                 **kwargs):
        super().__init__()
        self.combine_channels = combine_channels
        self.sigmoid_keys = [] if sigmoid_keys is None else sigmoid_keys
        self.reg_channels = {}
        for c in reg_channels:
            name, channel = c.split(':')
            self.reg_channels[name] = int(channel)

        if combine_channels:
            out_channel = sum(list(self.reg_channels.values()))
            self.head = get_head_conv(in_channel, out_channel, use_bias)
        else:
            for name, channel in self.reg_channels.items():
                setattr(self, f'head_{name}',
                        get_head_conv(in_channel, int(channel), use_bias))

    def forward(self, x):
        out_dict = {}
        if self.combine_channels:
            out_tensor = self.head(x)
            ptr = 0
            for k, v in self.reg_channels.items():
                out = out_tensor[:, ptr:ptr+v]
                if k in self.sigmoid_keys:
                    out = out.sigmoid()
                out_dict[k] = [out] # list compatible with separated head
                ptr += v
        else:
            for k in self.reg_channels.keys():
                out_dict[k] = [getattr(self, f'head_{k}')(x)]
        return out_dict


class DetCenterDense(nn.Module):
    def __init__(self, cfgs):
        super(DetCenterDense, self).__init__()
        for k, v in cfgs.items():
            setattr(self, k, v)
        self.det_r = cfgs['data_info']['det_r']
        self.voxel_size = cfgs['data_info']['voxel_size']
        self.n_heads = len(cfgs['class_names_each_head'])
        self.reg_heads = []

        self.shared_conv = get_conv2d_layers(
            'Conv2d', cfgs['input_channels'], cfgs['shared_conv_channel'],
            n_layers=1, kernel_size=[3], stride=[1], padding=[1], bias=False
        )

        self.cls_head = globals()[self.cls_head_cfg['name']](
            cfgs['class_names_each_head'],
            cfgs['shared_conv_channel'],
            one_hot_encoding=self.cls_head_cfg.get('one_hot_encoding', True)
        )
        self.reg_head = globals()[self.reg_head_cfg['name']](
            cfgs['reg_channels'],
            cfgs['shared_conv_channel'],
            combine_channels=self.reg_head_cfg['combine_channels'],
            sigmoid_keys=self.reg_head_cfg['sigmoid_keys'],
        )

        if 'target_assigner' in cfgs:
            from cosense3d.model.utils.target_assigner import TargetAssigner
            self.tgt_assigner = TargetAssigner(cfgs['target_assigner'],
                                               cfgs['class_names_each_head'])
        for k, v in self.loss_cfg.items():
            if isinstance(v['_target_'], str):
                m_name, cls_name = v['_target_'].rsplit('.', 1)
                v['_target_'] = getattr(importlib.import_module(f'cosense.{m_name}'), cls_name)

        self.out_dict = {'cls': []}
        for name in self.reg_heads:
            self.out_dict[f'reg_{name}'] = []

        self.temp = 1

    def forward(self, batch_dict):
        self.temp += 1
        feature = batch_dict[self.last_module][f'p{self.stride}']
        x = self.shared_conv(feature)

        out_dict = {}
        out_dict['cls'] = self.cls_head(x)
        out_dict['reg'] = self.reg_head(x)
        batch_dict['det_center_head'] = out_dict

        if self.get_rois:
            batch_dict['roi'] = self.rois(batch_dict)

        if self.get_predictions:
            batch_dict['det_s1'] = self.predictions(batch_dict)

        # from tools.vis_tools import draw_boxes
        # draw_boxes(batch_dict, self.det_r)
        # pass

    def loss(self, batch_dict):
        tgt = self.tgt_assigner(batch_dict)
        src = batch_dict['det_center_head']
        n_classes = [len(n) for n in self.class_names_each_head]

        center_loss = 0
        reg_loss = 0
        ptr = 0
        for h in range(self.n_heads):
            # center loss
            cur_cls_src = src['cls'][h].permute(0, 2, 3, 1).contiguous()  # b, h, w, n_cls+1
            cur_cls_tgt = tgt['center'][:, ptr:ptr+n_classes[h]].permute(0, 2, 3, 1).contiguous()
            ptr += n_classes[h]
            # tgt_pos = cur_cls_tgt.sum(dim=-1, keepdim=True) # b, h, w, 1
            # tgt_neg = 1 - torch.clamp(tgt_pos, min=0, max=1.0)
            # cur_cls_tgt = torch.cat([tgt_neg, cur_cls_tgt], dim=-1)
            # cur_cls_tgt = torch.div(cur_cls_tgt, cur_cls_tgt.sum(dim=-1, keepdim=True))
            # cur_cls_tgt = cur_cls_tgt.permute(0, 2, 3, 1).contiguous()  # b, h, w, n_cls+1
            max_scrs, max_inds = cur_cls_tgt.max(dim=-1, keepdim=True)
            tgt_pos = max_scrs > 0.1
            tgt_neg = torch.logical_not(tgt_pos)
            cur_cls_tgt_onehot = torch.zeros_like(cur_cls_tgt).view(-1, n_classes[h])
            cur_cls_tgt_onehot[torch.arange(max_inds.numel()), max_inds.view(-1)] = 1
            cur_cls_tgt_onehot[tgt_neg.view(-1)] = 0
            cur_cls_tgt_onehot = torch.cat([tgt_neg.view(-1, 1),
                                            cur_cls_tgt_onehot],
                                           dim=-1).contiguous()  # b, h, w, n_cls+1


            # from tools.vis_tools import vis_centernet
            # vis_centernet(cur_cls_src, cur_cls_tgt, batch_dict, h, self.voxel_size,
            #               self.stride, self.det_r)

            # only sample half of the negative samples to speed up training
            ignore = torch.rand_like(cur_cls_tgt[..., 0]) < 0.5
            ignore[tgt_pos.squeeze(-1)] = False
            cur_cls_tgt_onehot[ignore.view(-1)] = -1

            lcenter, _ = self.loss_cfg['center']['_target_'](
                cur_cls_src.view(-1, n_classes[h] + 1),
                cur_cls_tgt_onehot,
                n_classes[h] + 1,
                self.temp // 250,
                self.loss_cfg['center']['args']['annealing_step'] // 250,
                f"cls_h{h}"
            )
            center_loss = center_loss + lcenter

            # reg loss
            ind = tgt['reg']['idx'][h]
            if ind.shape[1] > 0:
                for reg_name in self.reg_head.reg_channels.keys():
                    cur_reg_src = src['reg'][reg_name][h].permute(0, 2, 3, 1).contiguous()
                    cur_reg_src = cur_reg_src[ind[0], ind[1], ind[2]]  # N, C
                    cur_reg_tgt = tgt['reg'][reg_name][h]  # N, C
                    cur_loss = weighted_smooth_l1_loss(cur_reg_src, cur_reg_tgt).mean()

                    reg_loss = reg_loss + cur_loss

        loss_dict = {
            'center': center_loss,
            'reg': reg_loss
        }
        loss = center_loss + reg_loss
        return loss, loss_dict

    def rois(self, batch_dict):
        return self.tgt_assigner.decode_box_dense(batch_dict)

    def predictions(self, batch_dict):
        roi = batch_dict['roi']
        boxes = torch.cat(roi['box'])
        scores = torch.cat(roi['scr'])
        labels = torch.cat(roi['lbl'])
        indices = torch.cat(roi['idx'])  # map index for retrieving features

        # nms
        preds = []
        l = batch_dict['batch_size'] * batch_dict.get('seq_len', 1)
        for b in range(l):
            mask = boxes[:, 0] == b
            if mask.sum() == 0:
                preds.append({
                    'box': torch.zeros((0, 7), device=boxes.device),
                    'scr': torch.zeros((0,), device=scores.device),
                    'lbl': torch.zeros((0,), device=labels.device),
                    'idx': torch.zeros((3, 0), device=indices.device),
                })
            else:
                keep = nms_gpu(
                    boxes[mask][:, 1:],
                    scores[mask],
                    thresh=self.post_processing['nms_thr'],
                    pre_maxsize=self.post_processing['nms_premaxsize']
                )
                preds.append({
                    'box': boxes[mask][keep][:, 1:],
                    'scr': scores[mask][keep],
                    'lbl': labels[mask][keep],
                    'idx': indices.T[mask][keep].T,
                })

        return preds
