import torch
from torch import nn
from cosense3d.modules import BaseModule
from cosense3d.modules.utils.init import bias_init_with_prob
from cosense3d.modules.utils.common import inverse_sigmoid


class ImgFocal(BaseModule):
    def __init__(self, in_channels, embed_dims, num_classes,
                 with_reg=False, with_depth=False, **kwargs):
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.embed_dims = embed_dims
        self.num_classes = num_classes
        self.with_reg = with_reg
        self.with_depth = with_depth

        self._init_layers()

    def _init_layers(self):
        self.cls = nn.Conv2d(self.embed_dims, self.num_classes, kernel_size=1)
        self.shared_cls = nn.Sequential(
                                 nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                                 nn.GroupNorm(32, num_channels=self.embed_dims),
                                 nn.ReLU(),)
        self.centerness = nn.Conv2d(self.embed_dims, 1, kernel_size=1)
        bias_init = bias_init_with_prob(0.01)
        nn.init.constant_(self.cls.bias, bias_init)
        nn.init.constant_(self.centerness.bias, bias_init)

        if self.with_reg:
            self.shared_reg = nn.Sequential(
                nn.Conv2d(self.in_channels, self.embed_dims, kernel_size=(3, 3), padding=1),
                nn.GroupNorm(32, num_channels=self.embed_dims),
                nn.ReLU(), )
            self.ltrb = nn.Conv2d(self.embed_dims, 4, kernel_size=1)
            self.center2d = nn.Conv2d(self.embed_dims, 2, kernel_size=1)
        if self.with_depth:
            self.depth = nn.Conv2d(self.embed_dims, 1, kernel_size=1)

    def forward(self, img_feat, img_coor, **kwargs):
        out_dict = {}
        x = self.data_from_list(img_feat)
        N, c, h, w = x.shape
        n_pixels = h * w

        cls_feat = self.shared_cls(x)
        cls = self.cls(cls_feat)
        centerness = self.centerness(cls_feat)
        cls_logits = cls.permute(0,2,3,1).reshape(-1, n_pixels, self.num_classes)
        centerness = centerness.permute(0,2,3,1).reshape(-1, n_pixels, 1)
        cls_score = cls_logits.topk(1, dim=2).values[..., 0].view(-1, n_pixels, 1)
        sample_weight = cls_score.detach().sigmoid() * centerness.detach().view(-1, n_pixels, 1).sigmoid()

        out_dict.update({
            'centerness': centerness,
            'cls_score': cls_score,
            'sample_weight': sample_weight
        })

        if self.with_reg:
            reg_feat = self.shared_reg(x)
            ltrb = self.ltrb(reg_feat).permute(0, 2, 3, 1).contiguous()
            ltrb = ltrb.sigmoid()
            centers2d_offset = self.center2d(reg_feat).permute(0, 2, 3, 1).contiguous()
            centers2d = self.apply_center_offset(img_coor, centers2d_offset)
            bboxes = self.apply_ltrb(img_coor, ltrb)

            pred_bboxes = bboxes.view(-1, n_pixels, 4)
            pred_centers2d = centers2d.view(-1, n_pixels, 2)
            out_dict.update({
                'pred_boxes': pred_bboxes,
                'pred_centers2d': pred_centers2d
            })

        if self.with_depth:
            raise NotImplementedError

    @staticmethod
    def apply_center_offset(locations, center_offset):
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4)
        """
        centers_2d = torch.zeros_like(center_offset)
        locations = inverse_sigmoid(locations)
        centers_2d[..., 0] = locations[..., 0] + center_offset[..., 0]  # x1
        centers_2d[..., 1] = locations[..., 1] + center_offset[..., 1]  # y1
        centers_2d = centers_2d.sigmoid()

        return centers_2d

    @staticmethod
    def apply_ltrb(locations, pred_ltrb):
        """
        :param locations:  (1, H, W, 2)
        :param pred_ltrb:  (N, H, W, 4)
        """
        pred_boxes = torch.zeros_like(pred_ltrb)
        pred_boxes[..., 0] = (locations[..., 0] - pred_ltrb[..., 0])  # x1
        pred_boxes[..., 1] = (locations[..., 1] - pred_ltrb[..., 1])  # y1
        pred_boxes[..., 2] = (locations[..., 0] + pred_ltrb[..., 2])  # x2
        pred_boxes[..., 3] = (locations[..., 1] + pred_ltrb[..., 3])  # y2
        min_xy = pred_boxes[..., 0].new_tensor(0)
        max_xy = pred_boxes[..., 0].new_tensor(1)
        pred_boxes = torch.where(pred_boxes < min_xy, min_xy, pred_boxes)
        pred_boxes = torch.where(pred_boxes > max_xy, max_xy, pred_boxes)
        pred_boxes = bbox_xyxy_to_cxcywh(pred_boxes)

        return pred_boxes


