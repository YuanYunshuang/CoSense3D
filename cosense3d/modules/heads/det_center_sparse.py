
import importlib
from einops import rearrange

from cosense3d.modules import BaseModule
from cosense3d.modules.utils.common import linear_last
from cosense3d.modules.losses.common import weighted_smooth_l1_loss
from cosense3d.ops.iou3d_nms_utils import nms_gpu
from cosense3d.modules.utils.me_utils import *


class UnitedClsHead(nn.Module):
    def __init__(self,
                 class_names_each_head,
                 in_channel,
                 one_hot_encoding=True,
                 use_bias=False,
                 norm='BN',
                 **kwargs):
        super().__init__()
        n_cls = sum([len(c) for c in class_names_each_head])
        out_channel = n_cls + 1 if one_hot_encoding else n_cls
        self.head = linear_last(in_channel, in_channel, out_channel, use_bias, norm)

    def forward(self, x):
        return [self.head(x)]


class SeparatedClsHead(nn.Module):
    def __init__(self,
                 class_names_each_head,
                 in_channel,
                 one_hot_encoding=True,
                 use_bias=False,
                 norm='BN',
                 **kwargs):
        super().__init__()
        self.n_head = len(class_names_each_head)
        for i, cls_names in enumerate(class_names_each_head):
            out_channel = len(cls_names)
            if one_hot_encoding:
                out_channel += 1
            setattr(self, f'head_{i}',
                    linear_last(in_channel, in_channel, out_channel, use_bias, norm))

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
                 norm='BN',
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
            self.head = linear_last(in_channel, in_channel, out_channel, use_bias, norm)
        else:
            for name, channel in self.reg_channels.items():
                setattr(self, f'head_{name}',
                        linear_last(in_channel, in_channel, int(channel), use_bias, norm))

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


class DetCenterSparse(BaseModule):
    def __init__(self,
                 data_info,
                 stride,
                 class_names_each_head,
                 shared_conv_channel,
                 cls_head_cfg,
                 reg_head_cfg,
                 reg_channels,
                 loss_cfg,
                 target_assigner=None,
                 **kwargs):
        super(DetCenterSparse, self).__init__(**kwargs)
        update_me_essentials(self, data_info, stride)

        self.n_heads = len(class_names_each_head)
        self.class_names_each_head = class_names_each_head
        self.loss_cfg = loss_cfg
        self.reg_heads = []

        self.cls_head = globals()[cls_head_cfg['name']](
            class_names_each_head,
            shared_conv_channel,
            one_hot_encoding=cls_head_cfg.get('one_hot_encoding', True)
        )
        self.reg_head = globals()[reg_head_cfg['name']](
            reg_channels,
            shared_conv_channel,
            combine_channels=reg_head_cfg['combine_channels'],
            sigmoid_keys=reg_head_cfg['sigmoid_keys'],
        )

        if target_assigner is not None:
            from cosense3d.modules.utils.target_assigner import TargetAssigner
            self.tgt_assigner = TargetAssigner(target_assigner,
                                               class_names_each_head=class_names_each_head,
                                               batch_dict_key=self.__class__.__name__)
        for k, v in self.loss_cfg.items():
            if isinstance(v['_target_'], str):
                m_name, cls_name = v['_target_'].rsplit('.', 1)
                v['_target_'] = getattr(importlib.import_module(f'cosense3d.{m_name}'), cls_name)

        self.out_dict = {'cls': []}
        for name in self.reg_heads:
            self.out_dict[f'reg_{name}'] = []

        self.temp = 1

    def forward(self, stensor_list):
        self.temp += 1
        coor, feat = self.format_input(stensor_list)
        centers = indices2metric(coor, self.voxel_size)

        out_dict = {
            'center': centers,
            'cls': self.cls_head(feat),
            'reg': self.reg_head(feat)
        }

        if getattr(self, 'get_rois', False):
            out_dict['roi'] = self.rois(out_dict)

        elif getattr(self, 'vis_training', False) or not self.training:
            out_dict['det_s1'] = self.predictions(out_dict)

        # from tools.vis_tools import draw_boxes
        # draw_boxes(batch_dict, self.det_r)
        # pass
        return self.format_output(out_dict, len(stensor_list))

    def format_input(self, stensor_list):
        return self.compose_stensor(stensor_list, self.stride)

    def format_output(self, output, B=None):
        # decompose batch
        output_new = {k: [] for k in output.keys()}
        for i in range(B):
            mask = output['center'][:, 0] == i
            output_new['center'].append(output['center'][mask, 1:])
            output_new['cls'].append([h_cls[mask] for h_cls in output['cls']])
            output_new['reg'].append({k:[vi[mask] for vi in v] for k, v in output['reg'].items()})
        output = {self.scatter_keys[0]: self.compose_result_list(output_new, B)}
        return output

    def loss(self, batch_dict):
        tgt = self.tgt_assigner(batch_dict)
        src = batch_dict['det_center_head']
        n_classes = [len(n) for n in self.class_names_each_head]

        # peudo_src = copy.copy(src)
        # mask = tgt['reg']['valid_mask'][0]
        # peudo_src['centers'] = peudo_src['centers'][mask]
        # class_logits = torch.zeros_like(src['cls'][0][mask])
        # tgt_cls = tgt['center_cls'][mask].squeeze()
        # tgt_cls[tgt_cls < 0] = 0
        # class_logits[:, 0] = 1 - tgt_cls
        # class_logits[:, 1] = tgt_cls * 50
        # peudo_src['cls'][0] = class_logits
        # peudo_src['reg'] = tgt['reg']
        # batch = copy.copy(batch_dict)
        # batch['det_center_head'] = peudo_src
        # self.predictions(batch)

        center_loss = 0
        reg_loss = 0
        ptr = 0
        for h in range(self.n_heads):
            # center loss
            cur_cls_src = rearrange(src['cls'][h], 'n d ... -> n ... d').contiguous()
            cur_cls_tgt = rearrange(tgt['center_cls'][:, ptr:ptr+n_classes[h]
                                    ], 'n d ... -> n ... d').contiguous().float().squeeze(-2)
            cared = (cur_cls_tgt >= 0).any(dim=-1)
            cur_cls_src = cur_cls_src[cared]
            cur_cls_tgt = cur_cls_tgt[cared]
            ptr += n_classes[h]
            # tgt_pos = cur_cls_tgt.sum(dim=-1, keepdim=True) # b, h, w, 1
            # tgt_neg = 1 - torch.clamp(tgt_pos, min=0, max=1.0)
            # cur_cls_tgt = torch.cat([tgt_neg, cur_cls_tgt], dim=-1)
            # cur_cls_tgt = torch.div(cur_cls_tgt, cur_cls_tgt.sum(dim=-1, keepdim=True))
            # cur_cls_tgt = cur_cls_tgt.permute(0, 2, 3, 1).contiguous()  # b, h, w, n_cls+1
            max_scrs, max_inds = cur_cls_tgt.max(dim=-1, keepdim=True)
            tgt_pos = max_scrs > 0.1
            tgt_neg = torch.logical_not(tgt_pos)

            if cur_cls_tgt.shape[-1] == n_classes[h]:
                cur_cls_tgt_onehot = cur_cls_tgt
            elif cur_cls_tgt.shape[-1] == 1:
                cur_cls_tgt_onehot = torch.zeros_like(cur_cls_src).view(-1, n_classes[h])
                cur_cls_tgt_onehot[torch.arange(max_inds.numel()), max_inds.view(-1)] = 1
            else:
                raise NotImplementedError
            cur_cls_tgt_onehot = torch.cat([tgt_neg.view(-1, 1).float(),
                                            cur_cls_tgt_onehot],
                                           dim=-1).contiguous()  # b, h, w, n_cls+1

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
                    cur_reg_src = rearrange(src['reg'][reg_name][h], 'n d ... -> n ... d').contiguous()
                    if cur_reg_src.ndim > 2:
                        cur_reg_src = cur_reg_src[ind[0], ind[1], ind[2]]  # N, C
                    else:
                        cur_reg_src = cur_reg_src[tgt['reg']['valid_mask'][h]]
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
        return self.tgt_assigner.decode_box(batch_dict)

    def predictions(self, batch_dict):
        roi = self.rois(batch_dict)
        batch_dict['roi'] = roi
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

        if getattr(self, 'vis_training', False):
            from cosense3d.tools.vis_tools import vis_detection
            pcds = batch_dict['pcds'][batch_dict['pcds'][:, 0] < batch_dict['num_cav'][0], 1:]
            gt_boxes = batch_dict['objects']
            gt_boxes = gt_boxes[gt_boxes[:, 0] == 0][:, [3, 4, 5, 6, 7, 8, 11]]
            vis_detection(preds[0], pcds,
                          pc_range=self.lidar_range,
                          gt_boxes=gt_boxes)
        return preds
