from cosense3d.model.utils import *
from cosense3d.model.heads.det_anchor_base import DetAnchorBase
from cosense3d.ops.iou3d_nms_utils import nms_gpu


class DetectionS1Sparse(DetAnchorBase):
    def __init__(self, cfgs):
        super(DetectionS1Sparse, self).__init__(cfgs)
        self.coor_lim = self.target_assigner.coor_lim
        # intermediate result
        self.xy = None
        self.coor = None

        ks = int(0.8 / self.voxel_size[0] / self.stride) * 2 + 1
        self.convs = nn.Sequential(
            minkconv_conv_block(self.in_dim, 64, ks, 1, d=2, bn_momentum=0.1,
                                expand_coordinates=True),
            minkconv_conv_block(64, 64, ks, 1, d=2, bn_momentum=0.1,
                                expand_coordinates=True),
            minkconv_conv_block(64, 64, ks, 1, d=2, bn_momentum=0.1,
                                expand_coordinates=True)
        )
        self.cls = linear_last(64, 32, self.num_cls)
        self.scores = linear_last(64, 32, 3 * self.num_cls)         # iou, dir1, dir2
        self.reg = linear_last(64, 32, 10 * self.num_cls)           # xyzlwh, ct1, st1, ct2, st2

    def forward(self, batch_dict):
        # self.points = batch_dict['pcds'][
        #         batch_dict['in_data'].C[:, 0] == 0, 1:4].cpu().numpy()
        # self.gt_boxes = batch_dict['objects'][
        #     batch_dict['objects'][:, 0] == 0,
        # [3, 4, 5, 6, 7, 8, 11]].cpu().numpy()
        stensor3d = batch_dict['compression'].get(f'p{self.stride}')
        stensor2d = ME.SparseTensor(
            coordinates=stensor3d.C[:, :3].contiguous(),
            features=stensor3d.F,
            tensor_stride=[self.stride] * 2
        )
        self.update_coords(stensor2d.C)
        feat = stensor2d.F

        self.out['cls'] = self.cls(feat)
        self.out['scores'] = self.scores(feat)
        self.out['reg'] = self.reg(feat)

        # self.get_stage_one_boxes()
        # # boxes_fused, scores_fused = self.box_fusion(batch_dict['num_cav'])
        # batch_dict['detection']['boxes_fused'] = self.out['pred_box']
        # return self.out['pred_box'], self.out['pred_scr']

    def update_coords(self, coor):
        xy = coor.float()
        xy[:, 1] = (xy[:, 1] - self.coor_lim[0]) / self.stride
        xy[:, 2] = (xy[:, 2] - self.coor_lim[1]) / self.stride
        self.xy = xy.long()
        coor_ = coor.float()
        coor_[:, 1] = coor_[:, 1] * self.voxel_size[0]
        coor_[:, 2] = coor_[:, 2] * self.voxel_size[0]
        self.coor = coor_.T

    def get_stage_one_boxes(self):
        dec_boxes = self.decode_boxes()
        ious_pred = self.out['scores'][:, :2].sigmoid()
        cls_scores = self.out['cls'].sigmoid()
        boxes = []
        ious = []
        scrs = []
        for b in self.xy[0].unique():
            mask = self.xy[0] == b
            cur_boxes = dec_boxes[mask].view(-1, 7)
            cur_scores = cls_scores[mask].view(-1)
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

            # #####
            # mask = self.xy[0]==0
            # vis_boxes = self.anchors[self.xy[1, mask], self.xy[2, mask]].view(-1, 7).cpu().numpy()
            # # vis_boxes = cur_boxes.detach().cpu().numpy()
            # from utils.vislib import draw_points_boxes_plt
            # if b==0:
            #     draw_points_boxes_plt(
            #         pc_range=50,
            #         points=self.centers[1:, self.centers[0]==0].T.cpu().numpy(),
            #         boxes_pred=vis_boxes,
            #         boxes_gt=self.gt_boxes
            #     )
            #     print('d')
            # #####

            cur_scores_rectified = cur_scores * cur_ious ** 4
            keep = nms_gpu(cur_boxes, cur_scores_rectified,
                           thresh=0.01, pre_maxsize=500)
            boxes.append(cur_boxes[keep])
            scrs.append(cur_scores[keep])
            ious.append(cur_ious[keep])

        self.out['pred_box'] = boxes
        self.out['pred_scr'] = scrs
        self.out['pred_iou'] = ious

    def box_fusion(self, num_cavs):
        boxes = self.out['pred_box']
        scores = self.out['pred_scr']
        ious = self.out['pred_iou']
        boxes_fused = []
        scores_fused = []
        idx_start = 0
        for b, num in enumerate(num_cavs):
            idx_end = idx_start + num
            cur_boxes = torch.cat(boxes[idx_start:idx_end], dim=0)
            cur_scores = torch.cat(scores[idx_start:idx_end], dim=0)
            cur_ious = torch.cat(ious[idx_start:idx_end], dim=0)
            cur_scores_rectified = cur_scores * cur_ious ** 4
            idx_start = idx_end
            if len(cur_boxes) == 0:
                continue
            keep = nms_gpu(cur_boxes, cur_scores_rectified,
                           thresh=0.01, pre_maxsize=100)
            bf = cur_boxes[keep]
            sf = cur_scores[keep]
            boxes_fused.append(torch.cat(
                [torch.ones_like(sf).view(-1, 1) * b, bf], dim=-1)
            )
            scores_fused.append(sf)
        boxes_fused = torch.cat(boxes_fused, dim=0)
        scores_fused = torch.cat(scores_fused, dim=0)
        return boxes_fused, scores_fused

    def get_vehicle_points(self, batch_dict):
        raw_points = torch.cat([batch_dict['in_data'].C[:, :1],
                                batch_dict['xyz']], dim=-1)
        raw_points = raw_points[batch_dict['target_semantic'] == 1]
        if len(raw_points[:, 0].unique()) > batch_dict['batch_size'] \
                and 'num_cav' in batch_dict:
            for i, c in enumerate(batch_dict['num_cav']):
                idx_start = sum(batch_dict['num_cav'][:i])
                mask = torch.logical_and(
                    raw_points[:, 0] >= idx_start,
                    raw_points[:, 0] < idx_start + c
                )
                raw_points[mask, 0] = i
        return raw_points
