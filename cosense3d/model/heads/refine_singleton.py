import torch
import torch_scatter

from cosense3d.model.utils import *
from cosense3d.model.submodules.attention import CrossAttention
from cosense3d.ops.iou3d_nms_utils import aligned_boxes_iou3d_gpu, boxes_iou3d_gpu
from cosense3d.model.losses.common import weighted_smooth_l1_loss


class RefineSingleton(nn.Module):
    def __init__(self, cfgs):
        super(RefineSingleton, self).__init__()

        self.in_channels = cfgs['in_channels']
        self.attention = CrossAttention(self.in_channels, 4, 32, True)

        self.iou_head = nn.Linear(64, 1, bias=False)
        self.reg_head = nn.Linear(64, 8, bias=False)

        self.out = {}

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                roi:
                    box: (N, 8), 1st column is batch index
                box_features: (N, C)
                gt_boxes: (N, 8)
        Returns:

        """
        bs = batch_dict['batch_size']
        sl = batch_dict['seq_len']
        box_features = batch_dict['box_features'].view(bs, sl, -1, 1)
        q = box_features[:, :1]
        k = box_features
        v = box_features
        x = self.attention(q, q, q).view(bs, -1)
        ious = self.iou_head(x)
        regs = self.reg_head(x)

        box_src = batch_dict['roi']['box'].view(bs, sl, -1)[:, 0]
        box_src[:, 0] = torch.div(box_src[:, 0], sl, rounding_mode='floor')
        box_s2 = self.dec_box(box_src[:, 1:], regs)

        self.out = {
            'rois': box_src,
            'reg': regs,
            'iou': ious,
            'box_s2': box_s2
        }
        # bs = batch_dict['batch_size']
        # sl = batch_dict['seq_len']
        # tgt_boxes = batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11]].view(bs, sl, -1)[:, 0]
        # tgt_reg, tgt_iou, pos, src_iou = self.get_tgt(tgt_boxes)
        # box_s2 = self.dec_box(box_src[:, 1:], tgt_reg)

        preds = []
        for b in range(bs):
            preds.append(
                {
                    'pred_boxes': box_s2[b],
                    'pred_scores': ious[b],
                    'pred_labels': None
                }
            )
        batch_dict['det_s2'] = preds
        # from cosense3d.tools.vis_tools import vis_singleton_track
        # vis_singleton_track(batch_dict)
        return batch_dict

    def loss(self, batch_dict):
        bs = batch_dict['batch_size']
        sl = batch_dict['seq_len']
        tgt_boxes = batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11]].view(bs, sl, -1)[:, 0]
        tgt_reg, tgt_iou, pos, src_iou = self.get_tgt(tgt_boxes)
        if pos.sum() == 0:
            return 0, {}
        # selected = pos.float()
        # mask1 = src_iou > 0.7
        # mask2 = torch.logical_and(src_iou < 0.3, pos)
        loss_reg = weighted_smooth_l1_loss(self.out['reg'][pos], tgt_reg[pos]).mean()
        loss_iou = weighted_smooth_l1_loss(self.out['iou'].sigmoid(), tgt_iou).mean()
        loss = loss_reg + loss_iou
        loss_dict = {'roi': loss, 'roi_reg': loss_reg, 'roi_iou': loss_iou}
        return loss, loss_dict

    def get_tgt(self, target_boxes):
        rois = self.out['rois']
        boxes = self.out['box_s2']
        if boxes.shape[1] == 8:
            boxes = boxes[:, 1:]
        tgt_reg, tgt_boxes_aligned, pos_mask = self.enc_box(rois, target_boxes)
        tgt_iou = aligned_boxes_iou3d_gpu(boxes, tgt_boxes_aligned[:, 1:])
        src_iou = aligned_boxes_iou3d_gpu(rois[:, 1:], tgt_boxes_aligned[:, 1:])
        return tgt_reg, tgt_iou, pos_mask, src_iou

    @staticmethod
    @torch.no_grad()
    def enc_box(rois, gt_bbox):
        """
        Args:
            rois: (N, 8)
            gt_bbox: (N, 8)
        """
        rois = rois.detach()

        xa, ya, za, wa, la, ha, ra = torch.split(rois[:, 1:], 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(gt_bbox[:, 1:], 1, dim=-1)

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xt = (xg - xa) / diagonal
        yt = (yg - ya) / diagonal
        zt = (zg - za) / ha

        lt = torch.log(lg / la)
        wt = torch.log(wg / wa)
        ht = torch.log(hg / ha)

        ct = torch.cos(rg) - torch.cos(ra)
        st = torch.sin(rg) - torch.sin(ra)

        ret = torch.cat([xt, yt, zt, wt, lt, ht, ct, st], dim=-1)
        return ret, gt_bbox, torch.ones_like(gt_bbox[:, 0]).bool()

    @staticmethod
    @torch.no_grad()
    def dec_box(anchors, reg):
        xa, ya, za, wa, la, ha, ra = torch.split(anchors, 1, dim=-1)
        xt, yt, zt, wt, lt, ht, ct, st = torch.split(reg, 1, dim=-1)

        diagonal = torch.sqrt(la ** 2 + wa ** 2)
        xg = xt * diagonal + xa
        yg = yt * diagonal + ya
        zg = zt * ha + za

        lg = torch.exp(lt) * la
        wg = torch.exp(wt) * wa
        hg = torch.exp(ht) * ha

        cg = ct + torch.cos(ra)
        sg = st + torch.sin(ra)
        rg = torch.atan2(sg, cg)

        ret = torch.cat([xg, yg, zg, wg, lg, hg, rg], dim=-1)
        return ret