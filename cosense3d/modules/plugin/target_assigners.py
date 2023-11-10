from abc import ABCMeta, abstractmethod
from functools import partial
from typing import List, Dict, Optional

import torch
from scipy.optimize import linear_sum_assignment

from cosense3d.utils.box_utils import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from cosense3d.utils.iou2d_calculator import bbox_overlaps
from cosense3d.modules.utils.gaussian_utils import gaussian_2d


class BaseAssigner(metaclass=ABCMeta):
    """Base assigner."""

    @abstractmethod
    def assign(self, *args, **kwargs):
        """Assign preds to targets."""


class MatchCost:
    """This class is modified from mmdet."""
    @staticmethod
    def classification(cls_pred, gt_labels, weight=1.0):
        """
        Args:
            cls_pred (Tensor): Predicted classification logits, shape
                (num_query, num_class).
            gt_labels (Tensor): Label of `gt_bboxes`, shape (num_gt,).
            weight (int | float, optional): loss_weight.

        Returns:
            torch.Tensor: cls_cost value with weight
        """
        # Following the official DETR repo, contrary to the loss that
        # NLL is used, we approximate it in 1 - cls_score[gt_label].
        # The 1 is a constant that doesn't change the matching,
        # so it can be omitted.
        cls_score = cls_pred.softmax(-1)
        cls_cost = -cls_score[:, gt_labels]
        return cls_cost * weight

    @staticmethod
    def bboxl1(bbox_pred, gt_bboxes, weight=1., box_format='xyxy'):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
            weight (int | float, optional): loss_weight.
            box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN.

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        else:
            raise NotImplementedError
        bbox_cost = torch.cdist(bbox_pred, gt_bboxes, p=1)
        return bbox_cost * weight

    @staticmethod
    def giou(bboxes, gt_bboxes, weight=1.0):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape (num_query, 4).
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape (num_gt, 4).
            weight (int | float, optional): loss weight.

        Returns:
            torch.Tensor: giou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode="giou", is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * weight

    @staticmethod
    def iou(bboxes, gt_bboxes, weight=1.0):
        """See giou"""
        # overlaps: [num_bboxes, num_gt]
        overlaps = bbox_overlaps(
            bboxes, gt_bboxes, mode="iou", is_aligned=False)
        # The 1 is a constant that doesn't change the matching, so omitted.
        iou_cost = -overlaps
        return iou_cost * weight

    @staticmethod
    def l1(pred, gt, weight=1.0):
        """L1 distance between pred and gt Tensors"""
        cost = torch.cdist(pred, gt, p=1)
        return cost * weight

    @staticmethod
    def binary_focal_loss(cls_pred, gt_labels, weight=1., alpha=0.25, gamma=2, eps=1e-12,):
        cls_pred = cls_pred.flatten(1)
        gt_labels = gt_labels.flatten(1).float()
        n = cls_pred.shape[1]
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + eps).log() * (
            1 - alpha) * cls_pred.pow(gamma)
        pos_cost = -(cls_pred + eps).log() * alpha * (
            1 - cls_pred).pow(gamma)

        cls_cost = torch.einsum('nc,mc->nm', pos_cost, gt_labels) + \
            torch.einsum('nc,mc->nm', neg_cost, (1 - gt_labels))
        return cls_cost / n * weight

    @staticmethod
    def focal_loss(cls_pred, gt_labels, weight=1., alpha=0.25, gamma=2, eps=1e-12,):
        cls_pred = cls_pred.sigmoid()
        neg_cost = -(1 - cls_pred + eps).log() * (
            1 - alpha) * cls_pred.pow(gamma)
        pos_cost = -(cls_pred + eps).log() * alpha * (
            1 - cls_pred).pow(gamma)

        cls_cost = pos_cost[:, gt_labels] - neg_cost[:, gt_labels]
        return cls_cost * weight

    def build(self, type, **kwargs):
        return partial(getattr(self, type), **kwargs)


class HungarianAssigner2D(BaseAssigner):
    """Computes one-to-one matching between predictions and ground truth.

    This class computes an assignment between the targets and the predictions
    based on the costs. The costs are weighted sum of three components:
    classification cost, regression L1 cost, regression iou cost and center2d l1 cost.
    The assignment is done in the following steps, the order matters.

    1. assign every prediction to -1
    2. compute the weighted costs
    3. do Hungarian matching on CPU based on the costs
    4. assign all to 0 (background) first, then for each matched pair
       between predictions and gts, treat this prediction as foreground
       and assign the corresponding gt index (plus 1) to it.
    """

    def __init__(self,
                 cls_cost=dict(type='classification', weight=1.),
                 reg_cost=dict(type='bboxl1', weight=1.0),
                 iou_cost=dict(type='giou', weight=1.0),
                 centers2d_cost=dict(type='l1', weight=1.0)):
        cost_builder = MatchCost()
        self.cls_cost = cost_builder.build(**cls_cost)
        self.reg_cost = cost_builder.build(**reg_cost)
        self.iou_cost = cost_builder.build(**iou_cost)
        self.centers2d_cost = cost_builder.build(**centers2d_cost)

    def assign(self,
               bbox_pred,
               cls_pred,
               pred_centers2d,
               gt_bboxes,
               gt_labels,
               centers2d,
               img_size,
               eps: float = 1e-7
               ):
        """Computes one-to-one matching based on the weighted costs.

        This method assign each query prediction to a ground truth or
        background. The `assigned_gt_inds` with -1 means don't care,
        0 means negative sample, and positive number is the index (1-based)
        of assigned gt.
        The assignment is done in the following steps, the order matters.

        1. assign every prediction to -1
        2. compute the weighted costs
        3. do Hungarian matching on CPU based on the costs
        4. assign all to 0 (background) first, then for each matched pair
           between predictions and gts, treat this prediction as foreground
           and assign the corresponding gt index (plus 1) to it.

        Args:
            bbox_pred: Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            cls_pred: Predicted classification logits, shape
                [num_query, num_class].
            gt_bboxes: Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].
            gt_labels: Label of `gt_bboxes`, shape (num_gt,).
            img_size: input image size.
            eps: A value added to the denominator for
                numerical stability. Default 1e-7.
        """
        num_gts, num_bboxes = gt_bboxes.size(0), bbox_pred.size(0)

        # 1. assign -1 by default
        assigned_gt_inds = bbox_pred.new_full((num_bboxes, ),
                                              -1,
                                              dtype=torch.long)
        assigned_labels = bbox_pred.new_full((num_bboxes, ),
                                             -1,
                                             dtype=torch.long)
        if num_gts == 0 or num_bboxes == 0:
            # No ground truth or boxes, return empty assignment
            if num_gts == 0:
                # No ground truth, assign all to background
                assigned_gt_inds[:] = 0
            return num_gts, assigned_gt_inds, assigned_labels
        img_h, img_w = img_size
        factor = gt_bboxes.new_tensor([img_w, img_h, img_w,
                                       img_h]).unsqueeze(0)

        # 2. compute the weighted costs
        # classification and bboxcost.
        cls_cost = self.cls_cost(cls_pred, gt_labels)
        # regression L1 cost
        normalize_gt_bboxes = gt_bboxes / factor
        reg_cost = self.reg_cost(bbox_pred, normalize_gt_bboxes)
        # regression iou cost, defaultly giou is used in official DETR.
        bboxes = bbox_cxcywh_to_xyxy(bbox_pred) * factor
        iou_cost = self.iou_cost(bboxes, gt_bboxes)

        # center2d L1 cost
        normalize_centers2d = centers2d / factor[:, 0:2]
        centers2d_cost = self.centers2d_cost(pred_centers2d, normalize_centers2d)

        # weighted sum of above four costs
        cost = cls_cost + reg_cost + iou_cost + centers2d_cost
        cost = torch.nan_to_num(cost, nan=100.0, posinf=100.0, neginf=-100.0)
        # 3. do Hungarian matching on CPU using linear_sum_assignment
        cost = cost.detach().cpu()
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
        return num_gts, assigned_gt_inds, assigned_labels


class HeatmapAssigner(BaseAssigner):

    @staticmethod
    def draw_heatmap_gaussian(heatmap, center, radius, k=1):
        """Get gaussian masked heatmap.

        Args:
            heatmap (torch.Tensor): Heatmap to be masked.
            center (torch.Tensor): Center coord of the heatmap.
            radius (int): Radius of gaussian.
            K (int, optional): Multiple of masked_gaussian. Defaults to 1.

        Returns:
            torch.Tensor: Masked heatmap.
        """
        diameter = 2 * radius + 1
        gaussian = gaussian_2d((diameter, diameter), sigma=diameter / 6)

        x, y = int(center[0]), int(center[1])

        height, width = heatmap.shape[0:2]

        left, right = min(x, radius), min(width - x, radius + 1)
        top, bottom = min(y, radius), min(height - y, radius + 1)

        masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
        masked_gaussian = torch.from_numpy(
            gaussian[radius - top:radius + bottom,
                     radius - left:radius + right]).to(heatmap.device,
                                                       torch.float32)
        if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:
            torch.max(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
        return heatmap

    def assign(self, obj_centers2d, obj_bboxes, img_shape, stride):
        img_h, img_w = img_shape[:2]
        heatmap = torch.zeros(img_h // stride, img_w // stride, device=obj_centers2d.device)
        if len(obj_centers2d) != 0:
            l = obj_centers2d[..., 0:1] - obj_bboxes[..., 0:1]
            t = obj_centers2d[..., 1:2] - obj_bboxes[..., 1:2]
            r = obj_bboxes[..., 2:3] - obj_centers2d[..., 0:1]
            b = obj_bboxes[..., 3:4] - obj_centers2d[..., 1:2]
            bound = torch.cat([l, t, r, b], dim=-1)
            radius = torch.ceil(torch.min(bound, dim=-1)[0] / 16)
            radius = torch.clamp(radius, 1.0).cpu().numpy().tolist()
            for center, r in zip(obj_centers2d, radius):
                heatmap = self.draw_heatmap_gaussian(heatmap, center / 16, radius=int(r), k=1)
        return heatmap