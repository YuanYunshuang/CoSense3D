import copy
import importlib

import torch
from einops import rearrange
from scipy.optimize import linear_sum_assignment

from cosense3d.dataset.const import CoSenseBenchmarks as csb
from cosense3d.modules.utils.gaussian_utils import draw_gaussian_map, gaussian_radius
from cosense3d.modules.utils.edl_utils import logit_to_edl
from cosense3d.modules.utils.me_utils import update_me_essentials
from cosense3d.ops.utils import points_in_boxes_gpu_2d
from cosense3d.modules.utils.common import cat_coor_with_idx


class Hungarian3D:
    def __init__(self, pc_range, code_weights=None, with_velo=False, **kwargs):
        self.pc_range = pc_range
        self.code_weights = code_weights
        self.with_velo = with_velo
        if self.code_weights is not None:
            self.code_weights = torch.nn.Parameter(torch.tensor(
            self.code_weights).float(), requires_grad=False)
        match_cost_modules = importlib.import_module(f'cosense3d.model.utils.match_cost')
        for k, v in kwargs.items():
            args = {a: b for a, b in v.items() if a!='type'}
            cost_inst = getattr(match_cost_modules, v['type'])(**args)
            setattr(self, k, cost_inst)

    def __call__(self, bbox_pred, cls_pred, gt_boxes):
        """
        Match pred and gt boxes using Hungarian alignment in a single batch.
        The cost for the alignment is the weighted sum of classification and normalized bbox parameters.

        Parameters
        ----------
        bbox_pred: Tensor(N, 10), columns = (x, y, z, log(l), log(w), log(h), sin(r), cos(r), vx, vy)
        cls_pred: Tensor(N, n_cls)
        gt_boxes: Tensor(N, 10), columns = (cls, x, y, z, l, w, h, r, vx, vy)

        Returns
        -------
        dict:
            num_gts: int, number of gt boxes
            assigned_gt_inds: LongTensor(N,), assigned gt indices of each pred sample, index 0 is background
            aligned_tgt_labels: LongTensor(N,), assinged class labels, -1 indicates background
            aligned_pred_boxes: FloatTensor(N, 10), weighted and aligned
            aligned_tgt_boxes: FloatTensor(N, 10), normalized, weighted and aligned
        """
        gt_labels = gt_boxes[:, 0].long()
        gt_bboxes = gt_boxes[:, 1:]
        num_gts, num_bboxes = gt_boxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes,), -1, dtype=torch.long)

        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0

            return dict(num_gts=num_gts,
                        assigned_gt_inds=assigned_gt_inds,
                        aligned_tgt_boxes=bbox_pred,
                        aligned_tgt_labels=assigned_labels,
                        aligned_pred_boxes=bbox_pred)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalized_gt_bboxes = self.normalize_bbox(gt_bboxes, self.pc_range)

        if self.code_weights is not None:
            code_weights = self.code_weights.to(bbox_pred.device)
            bbox_pred = bbox_pred * code_weights
            normalized_gt_bboxes = normalized_gt_bboxes * code_weights

        if self.with_velo:
            reg_cost = self.reg_cost(bbox_pred, normalized_gt_bboxes)
        else:
            reg_cost = self.reg_cost(bbox_pred[:, :8], normalized_gt_bboxes[:, :8])

        # weighted sum of above two costs
        cost = cls_cost + reg_cost

        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        matched_row_inds, matched_col_inds = linear_sum_assignment(cost)
        matched_row_inds = torch.from_numpy(matched_row_inds).to(
            bbox_pred.device)
        matched_col_inds = torch.from_numpy(matched_col_inds).to(
            bbox_pred.device)

        # 4. assign backgrounds and foregrounds
        # assign all indices to backgrounds first
        assigned_gt_inds[:] = 0
        # assign foregrounds based on matching results
        assigned_gt_inds[matched_row_inds] = matched_col_inds + 1
        assigned_labels[matched_row_inds] = gt_labels[matched_col_inds]

        # 5. align matched pred and gt
        aligned_tgt_boxes = torch.zeros_like(bbox_pred)
        assign_mask = assigned_gt_inds > 0
        aligned_tgt_boxes[assign_mask] = normalized_gt_bboxes[assigned_gt_inds[assign_mask] - 1]

        # from cosense3d.utils.vislib import draw_points_boxes_plt
        # vis_boxes_pred = self.denormalize_bbox(bbox_pred[assign_mask], self.pc_range)[:, :-2]
        # vis_boxes_pred[:, :2] /= code_weights[:2]
        # vis_boxes_gt = self.denormalize_bbox(aligned_tgt_boxes[assign_mask], self.pc_range)[:, :-2]
        # vis_boxes_gt[:, :2] /= code_weights[:2]
        # draw_points_boxes_plt(
        #     pc_range=self.pc_range,
        #     boxes_pred=vis_boxes_pred.detach().cpu().numpy(),
        #     bbox_pred_label=[str(i) for i in range(vis_boxes_pred.shape[0])],
        #     boxes_gt=vis_boxes_gt.detach().cpu().numpy(),
        #     bbox_gt_label=[str(i) for i in range(vis_boxes_gt.shape[0])],
        #     filename='/home/yuan/Downloads/tmp.png'
        # )

        return dict(num_gts=num_gts,
                    assigned_gt_inds=assigned_gt_inds,
                    aligned_tgt_boxes=aligned_tgt_boxes,
                    aligned_tgt_labels=assigned_labels,
                    aligned_pred_boxes=bbox_pred)

    @staticmethod
    def normalize_bbox(bboxes, pc_range):
        cx = bboxes[..., 0:1]
        cy = bboxes[..., 1:2]
        cz = bboxes[..., 2:3]
        l = bboxes[..., 3:4].log()
        w = bboxes[..., 4:5].log()
        h = bboxes[..., 5:6].log()

        rot = bboxes[..., 6:7]
        if bboxes.size(-1) > 7:
            vx = bboxes[..., 7:8]
            vy = bboxes[..., 8:9]
            normalized_bboxes = torch.cat(
                (cx, cy, cz, l, w, h, rot.sin(), rot.cos(), vx, vy), dim=-1
            )
        else:
            normalized_bboxes = torch.cat(
                (cx, cy, cz, l, w, h, rot.sin(), rot.cos()), dim=-1
            )
        return normalized_bboxes

    @staticmethod
    def denormalize_bbox(normalized_bboxes, pc_range):
        # rotation
        rot_sine = normalized_bboxes[..., 6:7]

        rot_cosine = normalized_bboxes[..., 7:8]
        rot = torch.atan2(rot_sine, rot_cosine)

        # center in the bev
        cx = normalized_bboxes[..., 0:1]
        cy = normalized_bboxes[..., 1:2]
        cz = normalized_bboxes[..., 2:3]

        # size
        l = normalized_bboxes[..., 3:4]
        w = normalized_bboxes[..., 4:5]
        h = normalized_bboxes[..., 5:6]

        l = l.exp()
        w = w.exp()
        h = h.exp()
        if normalized_bboxes.size(-1) > 8:
            # velocity
            vx = normalized_bboxes[:, 8:9]
            vy = normalized_bboxes[:, 9:10]
            denormalized_bboxes = torch.cat([cx, cy, cz, l, w, h, rot, vx, vy], dim=-1)
        else:
            denormalized_bboxes = torch.cat([cx, cy, cz, l, w, h, rot], dim=-1)
        return denormalized_bboxes


class TargetAssigner(object):
    def __init__(self, cfg, class_names_each_head=None, batch_dict_key=None):
        # must be set if multi-head classification used
        self.class_names_each_head = class_names_each_head
        # key to retrieve data from batch_dict
        self.batch_dict_key = batch_dict_key
        update_me_essentials(self, cfg['data_info'], cfg['stride'])
        self.meter_per_pixel = (self.stride * self.voxel_size[0],
                                self.stride * self.voxel_size[1])
        self.csb = csb.get(cfg['detection_benchmark'])
        self.n_cls = len(self.csb)

        # import modules if exist
        self.assigners = []
        if isinstance(cfg['assigners'], list):
            # cfg from yaml
            for assigner in cfg['assigners']:
                k, v = list(assigner.items())[0]
                self.get_assigner(k, v)
        elif isinstance(cfg['assigners'], dict):
            # cfg from pycfg
            for k, v in cfg['assigners'].items():
                self.get_assigner(k, v)
        else:
            raise NotImplementedError

    def get_assigner(self, k, v):
        self.assigners.append(k)
        if '_target_' in v and isinstance(v['_target_'], str):
            if '.' in v['_target_']:
                m_name, cls_name = v['_target_'].rsplit('.', 1)
                v['_target_'] = getattr(importlib.import_module(f'cosense3d.{m_name}'),
                                        cls_name)(**v.get('args', {}))
            else:
                v['_target_'] = globals().get(v['_target_'])(**v.get('args', {}))
        setattr(self, f"{k}_args", v)

    def __call__(self, batch_list, *args, **kwargs):
        tgt_dict = {}
        for assigner in self.assigners:
            kwargs.update(getattr(self, f"{assigner}_args"))
            tgt = getattr(self, assigner)(batch_list, *args, **kwargs)
            tgt_dict.update(tgt)
        return tgt_dict

    def get_gt_boxes(self, batch_dict):
        gt_boxes = batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11, 2]]
        if self.n_cls == 1:
            box_cls = torch.zeros_like(gt_boxes[:, -1])
        else:
            box_cls = gt_boxes[:, -1]
        return gt_boxes, box_cls

    def get_bev_pts(self, batch_size):
        x = torch.arange(self.lidar_range[0], self.lidar_range[3], self.meter_per_pixel[0]) \
                         + self.meter_per_pixel[0] * 0.5
        y = torch.arange(self.lidar_range[1], self.lidar_range[4], self.meter_per_pixel[1]) \
                         + self.meter_per_pixel[1] * 0.5
        bev_pts = torch.stack(torch.meshgrid(x, y, indexing='ij'), dim=-1).view(-1, 2)
        bev_pts = [torch.cat([torch.ones_like(bev_pts[:, :1]) * b,
                              bev_pts,
                              torch.zeros_like(bev_pts[:, :1])], dim=-1) \
                   for b in range(batch_size)]
        bev_pts = torch.cat(bev_pts, dim=0)
        return bev_pts

    def pts_to_indices(self, bev_pts):
        """
        Args:
            bev_pts: (N, 3+), 1st column is batch idx
        """
        x = (bev_pts[:, 1] - self.meter_per_pixel[0] * 0.5 - self.lidar_range[0]) \
                  / self.meter_per_pixel[0]
        y = (bev_pts[:, 2] - self.meter_per_pixel[1] * 0.5 - self.lidar_range[1]) \
                  / self.meter_per_pixel[1]
        indices = torch.stack([bev_pts[:, 0].long(), x.long(), y.long()], dim=1)
        return indices

    def indices_to_pts(self, indices):
        """
        Args:
            indices: (N, 3), 1st column is batch idx
        """
        x = indices[:, 1].float() * self.meter_per_pixel[0] + self.lidar_range[0] \
                 + self.meter_per_pixel[0] * 0.5
        y = indices[:, 2].float() * self.meter_per_pixel[1] + self.lidar_range[1] \
                 + self.meter_per_pixel[1] * 0.5
        points = torch.stack([indices[:, 0].float(), x, y], dim=1)
        return points

    def sample_mining(self, scores, labels, ratio, **kwargs):
        """
        Mine potential positive samples and ignore a portion of negative samples for training.
        Args:
            scores: (N, ..., nhead, C), classification scores/confidences.
            labels: (N, ..., nhead, 1) or (N, ..., nhead, C), class labels, or one-hot class labels.
            ratio: float, `n_neg_sample` / `n_pos_sample`
        """
        assert scores.ndim == labels.ndim
        assert scores.shape[-1] == labels.shape[-1] or labels.shape[-1] == 1
        xdims = {f'x{i}': x for i, x in enumerate(scores.shape[1:][:-2])}
        xdims_str = ' '.join(list(xdims.keys()))
        scores = rearrange(scores, f'N {xdims_str} n C -> (N {xdims_str}) n C')
        labels = rearrange(labels, f'N {xdims_str} n C -> (N {xdims_str}) n C')
        if labels.shape[-1] == 1:
            pos = (labels > 0).squeeze(-1)
        else:
            pos = labels.argmax(dim=-1) > 0
        neg = torch.logical_not(pos)
        labels_out = []
        for i in range(self.n_cls):
            cur_pos = pos[:, i]
            cur_neg = neg[:, i]
            cur_labels = labels[:, i]
            n_ignore = cur_neg.sum() - cur_pos.sum() * ratio
            neg_inds = torch.where(neg)[0]
            rand_inds = torch.randperm(len(neg_inds))[:n_ignore]
            ignore_inds = neg_inds[rand_inds]
            cur_labels[ignore_inds] = -1
        labels = rearrange(labels, f'(N {xdims_str}) n C -> N {xdims_str} n C', **xdims)
        return labels

    def pos_neg_sampling(self, labels, ratio):
        """
        Parameters
        ----------
        labels: Tensor, (N, n_cls, n_head)
        ratio: float
        """
        assert len(labels.shape) == 3
        _, n_cls, n_head = labels.shape
        pos = (labels > 0).any(dim=1)
        neg = (labels == 0).all(dim=1)
        n_neg_sample = pos.sum(dim=0) * ratio
        for n in range(n_head):
            if neg[:, n].sum() > n_neg_sample[n]:
                neg_inds = torch.where(neg[:, n])[0]
                perm = torch.randperm(len(neg_inds))[n_neg_sample:]
                labels[neg_inds[perm], :, n] = -1
        return labels

    def gaussian_center_map(self, batch_dict, args, tgt):
        gt_boxes, box_cls = self.get_gt_boxes(batch_dict)

        bs = batch_dict['batch_size'] * batch_dict.get('seq_len', 1)
        center_maps = gt_boxes.new_zeros((bs, self.n_cls, self.map_size, self.map_size))
        for n in range(self.n_cls):
            cur_boxes = gt_boxes[box_cls==n, :-1]
            radius = gaussian_radius(cur_boxes[:, 4:6],
                                     self.meter_per_pixel,
                                     args['gaussian_overlap'],
                                     args['min_radius'])
            center_maps[:, n] = draw_gaussian_map(
                cur_boxes, self.lidar_range,
                self.meter_per_pixel, bs, radius=radius,
                min_radius=args['min_radius']
            )
        tgt['center'] = center_maps

    def bev_map(self, batch_dict, args, tgt):
        gt_boxes, box_cls = self.get_gt_boxes(batch_dict)
        bs = l = batch_dict['batch_size'] * batch_dict.get('seq_len', 1)
        bev_maps = gt_boxes.new_zeros((bs, self.n_cls, self.size_x, self.size_y))
        bev_pts = self.get_bev_pts(bs).to(gt_boxes.device)

        for n in range(self.n_cls):
            cur_boxes = gt_boxes[box_cls == n, :-1]
            box_idx_of_pts = points_in_boxes_gpu_2d(bev_pts, cur_boxes, batch_size=bs)
            in_box_mask = box_idx_of_pts >= 0
            bev_maps[:, n] = in_box_mask.view(bs, self.size_x, self.size_y)

        tgt['bev'] = bev_maps

    def points_centerness(self, batch_list, gt_boxes, gt_cls, **kwargs):
        B = len(batch_list)
        center_cls = []
        n_cls = 1 if kwargs.get('merge_all_classes', False) else self.n_cls
        for n in range(n_cls):
            labels = []
            for b in range(B):
                cur_centers = batch_list[b]['center']
                cur_boxes = gt_boxes[b]
                if len(cur_boxes) == 0:
                    labels.append(torch.zeros_like(cur_centers[:, :1]))
                    continue
                dists = torch.norm(cur_centers.unsqueeze(1) - cur_boxes[:, :2].unsqueeze(0), dim=-1)
                dists_min = dists.min(dim=1, keepdim=True).values
                labels.append((dists_min < kwargs['min_radius']).float())
            center_cls.append(labels)

        # cat batch list
        center_cls = [torch.cat(x, dim=0).float() for x in center_cls]
        # stack classes
        center_cls = torch.stack(center_cls, dim=1).float()
        if 'pos_neg_ratio' in kwargs:
            center_cls = self.pos_neg_sampling(center_cls, kwargs['pos_neg_ratio'])

        tgt = {'centerness': center_cls}

        pred_cls = torch.cat([x['cls'][0] for x in batch_list], dim=0)
        cur_cls_src = rearrange(pred_cls, 'n d ... -> n ... d').contiguous()
        cur_cls_tgt = rearrange(tgt['centerness'], 'n d ... -> n ... d').contiguous().float().squeeze(-2)

        # mask = centers[:, 0] == 0
        # label = labels[mask].squeeze()
        # ctrs = centers[centers[:, 0] == 0, 1:]
        # boxes = gt_boxes[gt_boxes[:, 0] == 0, 1:]
        # dists = torch.norm(ctrs[label==1].unsqueeze(1) - boxes[:, :2].unsqueeze(0), dim=-1)
        # dists_min = dists.min(dim=1, keepdim=True).values
        # min_d = dists_min.max()

        # from cosense3d.utils import vislib
        # mask = centers[:, 0].int() == 0
        # label = center_cls[mask].int().detach().cpu().numpy()
        # label = label[:, 0, 0]
        # points = centers[mask, 1:].detach().cpu().numpy()
        # boxes = gt_boxes[gt_boxes[:, 0].int() == 0, 1:].cpu().numpy()
        # ax = vislib.draw_points_boxes_plt(
        #     pc_range=self.lidar_range,
        #     points=points,
        #     return_ax=True
        # )
        # ax = vislib.draw_points_boxes_plt(
        #     pc_range=self.lidar_range,
        #     points=points[label == 0],
        #     points_c='k',
        #     ax=ax,
        #     return_ax=True
        # )
        # vislib.draw_points_boxes_plt(
        #     pc_range=self.lidar_range,
        #     points=points[label == 1],
        #     points_c='r',
        #     boxes_gt=boxes,
        #     ax=ax,
        #     filename='/home/yuan/Downloads/tmp.png'
        # )
        #
        # pass
        return tgt

    def hungarian3d(self, batch_dict, args, tgt):
        hungarian_assigner = args['_target_']
        tgts = []
        for i, gt_boxes in enumerate(batch_dict['gt_boxes']):
            tgt_dict = hungarian_assigner(batch_dict['pred_boxes'][i],
                                     batch_dict['pred_scores'][i],
                                     gt_boxes)
            tgts.append(tgt_dict)

        keys = tgts[0].keys()
        tgts = {k: [tgt[k] for tgt in tgts] for k in keys}
        tgt['hungarian3d'] = tgts

    def get_valid_centers(self, center_map, thresh=0.1):
        conf, _ = logit_to_edl(center_map.permute(0, 2, 3, 1))
        center_mask = conf[..., 1:].max(dim=-1).values > thresh # b, h, w
        center_indices = torch.stack(torch.where(center_mask), dim=0)
        centers = self.indices_to_pts(center_indices[1:]).T
        centers = torch.cat([center_indices[0].unsqueeze(-1), centers], dim=-1)
        return centers, center_indices

    def encode_box(self, batch_list, gt_boxes, gt_cls,
                   center_thresh, **kwargs):
        box_names = [self.csb[c.item()][0] for x in gt_cls for c in x]
        gt_boxes = cat_coor_with_idx(gt_boxes)

        # consider unified head as multi-head with one head
        if not isinstance(batch_list[0]['cls'], list):
            centers_cls = [torch.cat([x['cls'] for x in batch_list], dim=0)]
            cls_names = [[n for cn in self.class_names_each_head for n in cn]]
        else:
            n_head = len(batch_list[0]['cls'])
            centers_cls = [torch.cat([x['cls'][n] for x in batch_list], dim=0) for n in range(n_head)]
            cls_names = self.class_names_each_head

        is_dense = True if centers_cls[0].ndim > 2 else False
        # cal regression targets
        reg_tgt = {'box': [], 'dir': [], 'scr': [], 'idx': [], 'valid_mask': []}
        for h, center_map in enumerate(centers_cls):
            if is_dense:
                centers, center_indices = self.get_valid_centers(centers_cls,
                                                                 center_thresh)
            else:
                centers = cat_coor_with_idx([x['center'] for x in batch_list])
                center_indices = self.pts_to_indices(centers).T
            cur_cls_names = cls_names[h]
            box_mask = [n in cur_cls_names for n in box_names]
            cur_boxes = gt_boxes[box_mask]
            reg_box, reg_dir, dir_score, valid = kwargs['_target_'].encode(
                centers, cur_boxes, self.meter_per_pixel)
            reg_tgt['idx'].append(center_indices[:, valid])
            reg_tgt['valid_mask'].append(valid)
            reg_tgt['box'].append(reg_box)
            reg_tgt['dir'].append(reg_dir)
            reg_tgt['scr'].append(dir_score)
        return {'reg': reg_tgt}

    def decode_box(self, preds):
        """
        Decode the center and regression maps into BBoxes.
        Args:
            preds:
                cls: list[Tensor], each tensor is the result from a cls head with shape (B or N, Ncls, ...).
                reg:
                    box: list[Tensor], one tensor per reg head with shape (B or N, 6, ...).
                    dir: list[Tensor], one tensor per reg head with shape (B or N, 8, ...).
                    scr: list[Tensor], one tensor per reg head with shape (B or N, 4, ...).

        Returns:
            roi:
                box: list[Tensor], one tensor per head with shape (N, 8).
                scr: list[Tensor], one tensor per head with shape (N,).
                lbl: list[Tensor], one tensor per head with shape (N,).
                idx: list[Tensor], one tensor per head with shape (3, N), center map indices of the boxes.

        """
        box_coder = self.encode_box_args['_target_']
        roi = {'box': [], 'scr': [], 'lbl': [], 'idx': []}
        lbl_cnt = torch.cumsum(torch.Tensor([0] + [m.shape[1] for m in preds['cls']]), dim=0)
        confs = []
        for h, center_cls in enumerate(preds['cls']):
            if center_cls.ndim > 2:
                conf, _ = logit_to_edl(center_cls.permute(0, 2, 3, 1))
                center_mask = conf[..., 1:].max(dim=-1).values > self.encode_box_args['center_thresh'] # b, h, w
                center_indices = torch.stack(torch.where(center_mask), dim=0)
                centers = self.indices_to_pts(center_indices[1:]).T
                cur_centers = torch.cat([center_indices[0].unsqueeze(-1), centers], dim=-1)
                cur_reg = {k: preds['reg'][k][h].permute(0, 2, 3, 1)[center_mask]
                           for k in ['box', 'dir', 'scr']}
            else:
                conf, _ = logit_to_edl(center_cls)
                centers = preds['center']
                center_mask = conf[..., 1:].max(dim=-1).values > self.encode_box_args['center_thresh']  # b, h, w
                cur_centers = centers[center_mask]
                center_indices = self.pts_to_indices(cur_centers)
                cur_reg = {k: preds['reg'][k][h][center_mask]
                           for k in ['box', 'dir', 'scr']}

                # from cosense3d.utils import vislib
                # mask = cur_centers[:, 0].int() == 0
                # confs = conf[center_mask][mask, 1].detach().cpu().numpy()
                # points = cur_centers[mask, 1:].detach().cpu().numpy()
                # fig = vislib.plt.figure(figsize=(6, 6))
                # vislib.plt.scatter(points[:, 0], points[:, 1], c=confs, s=1)
                # vislib.plt.show()
                # vislib.plt.close()

            cur_box = box_coder.decode(cur_centers, cur_reg)
            cur_scr, cur_lbl = conf[center_mask].max(dim=-1)
            cur_lbl = cur_lbl + lbl_cnt[h]
            roi['box'].append(cur_box)
            roi['scr'].append(cur_scr)
            roi['lbl'].append(cur_lbl)
            roi['idx'].append(center_indices)
            confs.append(conf)

        return roi, torch.stack(confs, dim=1)