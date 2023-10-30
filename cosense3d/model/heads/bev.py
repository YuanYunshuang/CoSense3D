import numpy as np
import torch
from torch import nn
from cosense3d.model.utils.me_utils import *
from cosense3d.model.utils import pad_r, indices2metric, linear_last, metric2indices
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.model.losses.edl import edl_mse_loss, evidence_to_conf_unc


class Bev(nn.Module):
    def __init__(self, cfgs=None, **kwargs):
        if cfgs is None:
            cfgs = kwargs
        super(Bev, self).__init__()
        for k, v in cfgs.items():
            setattr(self, k, v)
        for k, v in cfgs['data_info'].items():
            setattr(self, k, v)
        update_me_essentials(self, cfgs['data_info'], self.stride)

        # self.res = (self.stride * self.voxel_size[0], self.stride * self.voxel_size[1])
        # if getattr(self, 'det_r', False):
        #     lr = [-self.det_r, -self.det_r, 0, self.det_r, self.det_r, 0]
        # elif getattr(self, 'lidar_range', False):
        #     lr = self.lidar_range
        # else:
        #     raise NotImplementedError
        # self.lidar_range = lr
        # self.mink_xylim = mink_coor_limit(lr, self.voxel_size, self.stride)
        # self.size_x = round((lr[3] - lr[0]) / self.res[0])
        # self.size_y = round((lr[4] - lr[1]) / self.res[1])
        # self.offset_sz_x = round(lr[0] / self.res[0])
        # self.offset_sz_y = round(lr[1] / self.res[1])

        self.reg_layer = linear_last(cfgs['in_dim'], 32, 2, bias=True)

        if 'target_assigner' in cfgs:
            from cosense3d.model.utils.target_assigner import TargetAssigner
            self.tgt_assigner = TargetAssigner(cfgs['target_assigner'],
                                               cfgs['class_names_each_head'])

    def forward(self, batch_dict):
        conv_out = batch_dict[self.feature_src][f'p{self.stride}']
        coor = conv_out['coor']
        feat = conv_out['feat']
        if self.training:
            coor, feat = self.down_sample(coor, feat)

        centers = indices2metric(coor, self.voxel_size)
        reg = self.reg_layer(feat)
        conf, unc = evidence_to_conf_unc(reg.relu())

        out = {
            'centers': centers,
            'reg': reg,
            'conf': conf,
            'unc': unc
        }
        
        batch_dict['bev'] = out

        if getattr(self, 'visualize_training', False):
            import matplotlib.pyplot as plt
            conf_map, unc_map = bev_sparse_to_dense(self, out)
            img = conf_map[0, ..., 1].detach().cpu().numpy()
            plt.imshow(img.T[::-1], vmin=0, vmax=1)
            plt.savefig(self.visualize_training)

    def down_sample(self, coor, feat):
        keep = torch.rand_like(feat[:, 0]) > 0.5
        coor = coor[keep]
        feat = feat[keep]

        return coor, feat

    def loss(self, batch_dict):
        tgt_pts, tgt_label, valid = self.get_tgt(batch_dict)
        preds = batch_dict['bev']
        epoch_num = batch_dict.get('epoch', 0)
        loss, loss_dict = edl_mse_loss(preds=preds['reg'][valid],
                                       tgt=tgt_label,
                                       n_cls=2,
                                       temp=epoch_num,
                                       annealing_step=self.annealing_step,
                                       model_label='bev')
        return loss, loss_dict

    @torch.no_grad()
    def get_tgt(self, batch_dict):
        epoch_num = batch_dict.get('epoch', 0)
        preds = batch_dict['bev']
        tgt_pts = preds['centers'].clone()
        boxes = batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11]].clone()
        boxes[:, 3] = 0
        pts = pad_r(tgt_pts)
        try:
            _, box_idx_of_pts = points_in_boxes_gpu(
                pts, boxes, batch_size=batch_dict['batch_size']
            )
            boxes[:, 4:6] *= 2
            _, box_idx_of_pts2 = points_in_boxes_gpu(
                pts, boxes, batch_size=batch_dict['batch_size']
            )
        except:
            print(boxes.shape)
            print(pts.shape)
        # set area B: dense neg as -1 for down-sampling, differentiate from area C: sparse neg.
        tgt_label = - (box_idx_of_pts2 >= 0).int()
        tgt_label[box_idx_of_pts >= 0] = 1

        n_sam = len(boxes) * 50
        if self.sampling['annealing']:
            annealing_ratio = epoch_num / self.annealing_step
            n_sam = n_sam + annealing_ratio * len(tgt_label) / 50
            # down-sample
            mask = self.downsample_tgt_pts(tgt_label, max_sam=n_sam)
            tgt_label[tgt_label == -1] = 0  # set area B to 0

            # positive sample annealing
            conf = preds['conf']
            labeled_pos = tgt_label == 1
            potential_pos = (conf[..., 1] > (1 - annealing_ratio * 0.5))
            unlabeled_potential_pos = torch.logical_and(potential_pos,
                                                        torch.logical_not(labeled_pos))
            if self.sampling['topk']:
                k = int(labeled_pos.sum().item() * (1 + 30 * annealing_ratio))
                topk = torch.topk(conf[..., 1], k)
                is_topk = torch.zeros_like(labeled_pos)
                is_topk[topk.indices] = 1
                topk_potential_pos = torch.logical_and(is_topk, unlabeled_potential_pos)
                unlabeled_potential_pos = topk_potential_pos

            # set potential positive samples label to ignore
            tgt_label[unlabeled_potential_pos] = -1
        else:
            mask = self.downsample_tgt_pts(tgt_label, max_sam=n_sam)
            # mask = torch.ones_like(tgt_label).bool()
            tgt_label[tgt_label == -1] = 0  # set area B to 0

        # get final tgt
        tgt_pts = tgt_pts[mask]
        tgt_label = tgt_label[mask]

        # from cosense3d.utils.vislib import draw_points_boxes_plt
        # boxes_src = batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11]]
        # ax = draw_points_boxes_plt(
        #     pc_range=self.lidar_range,
        #     points=tgt_pts[tgt_pts[:, 0] == 0, 1:].cpu().numpy(),
        #     boxes_gt=boxes_src[boxes_src[:, 0] == 0, 1:],
        #     return_ax=True
        # )
        # pts_ = tgt_pts[tgt_label==1]
        # ax = draw_points_boxes_plt(
        #     points=pts_[pts_[:, 0] == 0, 1:].cpu().numpy(),
        #     points_c='r',
        #     ax=ax,
        #     return_ax=True,
        # )
        # pts_ = tgt_pts[tgt_label==-1]
        # draw_points_boxes_plt(
        #     points=pts_[pts_[:, 0] == 0, 1:].cpu().numpy(),
        #     points_c='orange',
        #     filename='/home/yuan/Downloads/tmp1.png',
        #     ax=ax
        # )

        return tgt_pts, tgt_label, mask

    @torch.no_grad()
    def downsample_tgt_pts(self, tgt_label, max_sam):
        selected = torch.ones_like(tgt_label.bool())
        pos = tgt_label == 1
        if pos.sum() > max_sam:
            mask = torch.rand_like(tgt_label[pos].float()) < max_sam / pos.sum()
            selected[pos] = mask

        neg = tgt_label == 0
        if neg.sum() > max_sam:
            mask = torch.rand_like(tgt_label[neg].float()) < max_sam / neg.sum()
            selected[neg] = mask
        return selected



