import importlib

import torch
from cosense3d.ops.iou3d_nms_utils import nms_gpu, boxes_iou_bev, \
    aligned_boxes_iou3d_gpu,  boxes_iou3d_gpu
from cosense3d.ops.utils import points_in_boxes_gpu


class AnchorTargetAssigner(object):
    def __init__(self, cfg, det_r, voxel_size, stride):
        self.cfg = cfg
        self.det_r = det_r
        self.stride = stride
        self.voxel_size = voxel_size
        self.sample_size = cfg['sample_size']
        self.coor_lim = self.get_mink_min_max_coor()
        self.anchor_generator = self.get_anchor_generator(cfg['anchor_generator'])
        self.box_coder = self.get_box_coder(cfg['box_coder'])
        self.num_cls = self.anchor_generator.num_cls
        # intermediate result
        self.out = {}

    def get_anchor_generator(self, anchor_cfg):
        anchor_generator_cls = getattr(
            importlib.import_module("model.utils.anchor_generator"),
            anchor_cfg['name']
        )
        anchor_generator_inst = anchor_generator_cls(
            anchor_cfg.get('cfg', None),
            self.coor_lim,
            self.stride,
            self.voxel_size
        )
        return anchor_generator_inst

    def get_box_coder(self, box_coder_cfg):
        box_coder_cls = getattr(
            importlib.import_module("model.utils.box_coder"),
            box_coder_cfg['name']
        )
        box_coder_inst = box_coder_cls()
        return box_coder_inst

    def get_mink_min_max_coor(self):
        """Calculate input min and max coordinate of ME Sparsetensors"""
        if 'r' in self.det_r:
            lr = self.det_r['r']
            self.det_r['x'] = [-lr, lr]
            self.det_r['y'] = [-lr, lr]
            grid_size = int(lr / self.voxel_size[0] / self.stride * 2)
            self.grid_size = (grid_size, grid_size)
        else:
            assert 'x' in self.det_r
            assert 'y' in self.det_r
            self.grid_size = (int((self.det_r['x'][1] - self.det_r['x'][0]) \
                                  / self.voxel_size[0] / self.stride),
                              int((self.det_r['y'][1] - self.det_r['y'][0]) \
                                  / self.voxel_size[1] / self.stride))
        x_max = (self.det_r['x'][1] / self.voxel_size[0] - 1) // self.stride * self.stride
        x_min = (self.det_r['x'][0] / self.voxel_size[0]) // self.stride * self.stride
        y_max = (self.det_r['y'][1] / self.voxel_size[1] - 1) // self.stride * self.stride
        y_min = (self.det_r['y'][0] / self.voxel_size[1]) // self.stride * self.stride

        return [x_min, y_min, x_max, y_max]

    def axis_aligned_target_assign(self, gt_boxes, coords=None):
        """
        Assign targets to anchors according to the predefined thresholds.

        Parameters
        ----------
        gt_boxes: (N, 8) [cls, x, y, z, l, w, h, heading]

        Returns
        -------

        """
        anchors = self.anchor_generator.anchors(coords).view(
            -1, self.num_cls, 7
        ).to(gt_boxes.device)
        iou_match = self.anchor_generator.iou_match.view(
            1, self.num_cls
        ).to(gt_boxes.device)
        iou_unmatch = self.anchor_generator.iou_unmatch.view(
            1, self.num_cls
        ).to(gt_boxes.device)
        assert len(gt_boxes) > 0
        assert len(anchors) > 0
        ious = boxes_iou_bev(anchors.view(-1, 7), gt_boxes[:, 1:]). \
            view(-1, self.num_cls, len(gt_boxes))  # na, 2, nb
        ious_max, max_idx = ious.max(dim=-1)  # na, 2
        pos = ious_max > iou_match
        neg = ious_max < iou_unmatch
        # down sample neg samples
        s = min(self.sample_size // self.num_cls * 2, pos.sum())
        if neg.all(dim=1).sum() > self.sample_size:
            perm = torch.randperm(neg.all(dim=1).sum())[:self.sample_size]
            neg = torch.where(neg.all(dim=1))[0][perm]
        cls = torch.ones_like(ious_max) * -1
        # set positive and negative cls labels
        try:
            cls[neg] = 0
        except:
            print(cls.shape)
            print(neg.shape)
        cls[pos] = 1
        boxes = gt_boxes[max_idx[pos], 1:]
        anchors = anchors[pos]
        return cls, ious_max, boxes, anchors

    def atss_target_assign(self, gt_boxes, coords=None):
        """
        Adaptive Training Sample Selection (ATSS).
        Reference: https://arxiv.org/abs/1912.02424

        Parameters
        ----------
        gt_boxes

        Returns
        -------

        """
        anchors = self.anchor_generator.dense_anchors().view(
            -1, self.num_cls, 7
        ).to(gt_boxes.device)
        iou_match = self.anchor_generator.iou_match.view(
            1, self.num_cls
        ).to(gt_boxes.device)
        iou_unmatch = self.anchor_generator.iou_unmatch.view(
            1, self.num_cls
        ).to(gt_boxes.device)
        assert len(gt_boxes) > 0
        assert len(anchors) > 0
        ious = boxes_iou_bev(anchors.view(-1, 7), gt_boxes[:, 1:])  # (na, nb)
        dist = torch.norm(
            anchors.view(-1, 7)[:, None, :3] - gt_boxes[None, :, 1:4],
            dim=-1
        )  # (na, nb)
        topk = self.cfg.get('topk', 10)
        _, topk_idxs = dist.topk(topk, dim=0, largest=False)  # (k, m)
        candidate_ious = ious[topk_idxs, torch.arange(len(gt_boxes))]  # (k, m)
        iou_mean_per_gt = candidate_ious.mean(dim=0)
        iou_std_per_gt = candidate_ious.std(dim=0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt + 1e-6

        ious_max, max_idx = ious.max(dim=-1)  # na, 2
        pos = (ious > iou_thresh_per_gt).any(dim=1)
        neg = (ious < iou_thresh_per_gt * 0.6).any(dim=1)
        # down sample neg samples
        s = min(self.sample_size // self.num_cls * 2, pos.sum())
        if neg.sum() > self.sample_size:
            perm = torch.randperm(neg.sum())[:self.sample_size]
            neg = torch.where(neg)[0][perm]
        cls = torch.ones_like(ious_max) * -1
        # set positive and negative cls labels
        cls[neg, :] = 0
        cls[pos] = gt_boxes[max_idx[pos], 0] + 1
        boxes = gt_boxes[max_idx[pos], 1:]
        anchors = anchors[pos]
        return cls, ious_max, boxes, anchors

    def atss_target_assign_(self, gt_boxes, coords=None):
        """code from open pcdet"""
        anchors = self.anchor_generator.dense_anchors().view(-1, 7).to(gt_boxes.device)
        ious = boxes_iou_bev(anchors, gt_boxes[:, 1:])
        dist = torch.norm(
            anchors.view(-1, 7)[:, None, :3] - gt_boxes[None, :, 1:4],
            dim=-1
        )  # (n, m)
        topk = self.cfg.get('topk', 10)
        _, topk_idxs = dist.topk(topk, dim=0, largest=False)  # (k, m)
        candidate_ious = ious[topk_idxs, torch.arange(len(gt_boxes))]  # (k, m)
        iou_mean_per_gt = candidate_ious.mean(dim=0)
        iou_std_per_gt = candidate_ious.std(dim=0)
        iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt + 1e-6
        is_pos = candidate_ious >= iou_thresh_per_gt[None, :]  # (k, m)

        # check whether anchor_center in gt_boxes, only check BEV x-y axes
        candidate_anchors = anchors[topk_idxs.view(-1)]  # (k*m, 7)
        gt_boxes_of_each_anchor = gt_boxes[:, :].repeat(topk, 1)  # (k*m, 7)
        anchor_in_gt = points_in_boxes_gpu(
            candidate_anchors[:, None, :3],
            gt_boxes_of_each_anchor[:, None, 1:]
        ).view(topk, -1) >= 0
        is_pos = is_pos & anchor_in_gt  # (k, m)

        for ng in range(len(gt_boxes)):
            topk_idxs[:, ng] += ng * len(anchors)
        # select the highest IoU if an anchor box is assigned with multiple gt_boxes
        INF = -0x7FFFFFFF
        ious_inf = torch.full_like(ious, INF).t().contiguous().view(-1)  # (m*n)
        index = topk_idxs.view(-1)[is_pos.view(-1)]
        ious_inf[index] = ious.t().contiguous().view(-1)[index]
        ious_inf = ious_inf.view(len(gt_boxes), -1).t()  # (N, M)

        anchors_to_gt_values, anchors_to_gt_indices = ious_inf.max(dim=1)

        # match the gt_boxes to the anchors which have maximum iou with them
        max_iou_of_each_gt, argmax_iou_of_each_gt = ious.max(dim=0)
        anchors_to_gt_indices[argmax_iou_of_each_gt] = torch.arange(0, len(gt_boxes), device=ious.device)
        anchors_to_gt_values[argmax_iou_of_each_gt] = max_iou_of_each_gt

        cls_labels = gt_boxes[anchors_to_gt_indices, 0] + 1
        cls_labels[anchors_to_gt_values == INF] = 0
        matched_gts = gt_boxes[anchors_to_gt_indices]

        pos_mask = cls_labels > 0
        reg_tgts = matched_gts.new_zeros(((pos_mask > 0).sum(), self.box_coder.code_size))
        if pos_mask.sum() > 0:
            reg_tgts = self.box_coder.encode_torch(matched_gts[pos_mask > 0], anchors[pos_mask > 0])

        return cls_labels, None, reg_tgts, None

    def gen_batch_anchors(self, batch_size, coords=None):
        """Generate anchors during inference."""
        anchors = []
        for b in range(batch_size):
            if coords is not None:
                cur_coords = coords[coords[:, 0] == b, 1:]
            else:
                cur_coords = None
            anchors.append(
                self.anchor_generator.anchors(cur_coords).view(
                    -1, self.num_cls, 7
                )
            )
        return anchors

    def __call__(self, gt_boxes_in, batch_size, coords=None):
        gt_boxes = gt_boxes_in[:, [2, 3, 4, 5, 6, 7, 8, 11]]
        obj_idx = gt_boxes_in[:, 0]

        iou_tgts = []
        boxes_aligned = []
        anchors_aligned = []
        cls_tgts = []
        for b in range(batch_size):
            cur_boxes = gt_boxes[obj_idx == b]
            if coords is not None:
                cur_coords = coords[coords[:, 0] == b, 1:]
            else:
                cur_coords = None
            cls_tgt, iou_tgt, box_aligned, anchor_aligned = \
            getattr(
                self, f"{self.cfg['assign_method']}_target_assign"
            )(cur_boxes, cur_coords)

            # #########
            # from utils.vislib import draw_box_plt
            # import matplotlib.pyplot as plt
            # fig = plt.figure(figsize=(8, 8))
            # ax = fig.add_subplot()
            # xy = cur_coords.cpu().numpy() * 0.8 - 100
            # ax.plot(xy[:, 0], xy[:, 1], '.', markersize=1)
            # anchors = self.anchor_generator.anchors(cur_coords).view(-1, 7)
            # ax = draw_box_plt(cur_boxes[:, 1:].detach().cpu().numpy(), ax, color='g')
            # ax = draw_box_plt(anchor_aligned.detach().cpu().numpy(), ax, color='k')
            #
            # plt.show()
            # plt.close()
            # #########

            cls_tgts.append(cls_tgt)
            iou_tgts.append(iou_tgt)
            boxes_aligned.append(box_aligned)
            anchors_aligned.append(anchor_aligned)

        cls_tgts = torch.cat(cls_tgts, dim=0)
        iou_tgts = torch.cat(iou_tgts, dim=0)
        boxes_aligned = torch.cat(boxes_aligned, dim=0)
        anchors_aligned = torch.cat(anchors_aligned, dim=0)
        reg_tgts, dir_tgts = self.box_coder.encode(anchors_aligned,
                                                   boxes_aligned)
        return cls_tgts, iou_tgts, dir_tgts, reg_tgts

