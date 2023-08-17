import torch
import torch_scatter

from cosense3d.model.utils import *
from cosense3d.ops.utils import points_in_boxes_gpu
from cosense3d.ops.iou3d_nms_utils import aligned_boxes_iou3d_gpu, boxes_iou3d_gpu
from cosense3d.model.losses.common import weighted_smooth_l1_loss


class DetRefine(nn.Module):
    def __init__(self, cfgs):
        super(DetRefine, self).__init__()

        self.in_channels = cfgs['in_channels']
        self.grid_size = cfgs.get('rcnn_grid_size', 6)
        grid = (self.grid_size, ) * 3 + (self.in_channels,)
        self.grid_emb = nn.Parameter(torch.randn(grid))
        self.pos_emb_layer = linear_layers([3, 32, 32])
        self.attn_weight = linear_layers([self.in_channels, 32, 1],
                                         ['ReLU', 'Sigmoid'])

        self.proj = linear_layers([self.in_channels, 32])
        self.fc_layer = linear_layers([64, 64])
        self.fc_out = linear_layers([64, 32])

        self.iou_head = nn.Linear(32, 1, bias=False)
        self.reg_head = nn.Linear(32, 8, bias=False)

        self.out = {}
        self.empty = False

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                roi:
                    box: (N, 8), 1st column is batch index
                    scr: (N), optional
                gt_boxes: (N, 8)
        Returns:

        """
        coords = self.get_coords(batch_dict)
        assert 'roi' in batch_dict, "no ROI result from stage 1."
        assert 'p0' in batch_dict['backbone'], "no p0 features in found."
        preds = batch_dict['roi']
        boxes = torch.cat(preds['box'][:-1], dim=0).clone()
        if len(boxes) == 0:
            self.empty = True
            return batch_dict
        else:
            self.empty = False

        boxes[:, 4:7] *= 1.5
        boxes_decomposed, box_idxs_of_pts = points_in_boxes_gpu(
            coords[:, :4], boxes, batch_dict['batch_size']
        )
        in_box_mask = box_idxs_of_pts >= 0
        new_idx = box_idxs_of_pts[in_box_mask]

        new_xyz = coords[in_box_mask, 1:4]
        features = batch_dict['backbone']['p0'][in_box_mask]
        mapped_boxes = boxes_decomposed[new_idx]

        # canonical transformation
        new_xyz = new_xyz - mapped_boxes[:, 1:4]
        # new_xyz will be modified during transformation, so make a copy here
        xyz = new_xyz.clone()
        st = torch.sin(-mapped_boxes[:, -1])
        ct = torch.cos(-mapped_boxes[:, -1])
        new_xyz[:, 0] = xyz[:, 0] * ct - xyz[:, 1] * st
        new_xyz[:, 1] = xyz[:, 0] * st + xyz[:, 1] * ct

        # grid size minus 1e-4 to ensure positive coords
        new_tfield = torch.div(new_xyz, mapped_boxes[:, 4:7]) * self.grid_size + 3
        new_tfield = torch.clamp(new_tfield, 0, self.grid_size - 1e-4)

        new_tfield = ME.TensorField(
            coordinates=torch.cat([new_idx.view(-1, 1), new_tfield], dim=1),
            features=features,
            quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
        )
        voxel_embs = self.voxelize_with_centroids(new_tfield, new_xyz)
        voxel_idx = voxel_embs.C[:, 0].long()
        num_box = len(boxes_decomposed)
        pos = voxel_embs.C.T[1:].long()
        # if pos.min() < 0 or pos.max() > self.grid_size - 1:
        #     print('d')
        pos_emb = self.grid_emb[pos[0], pos[1], pos[2]]
        voxel_embs = voxel_embs.F.contiguous() + pos_emb
        weights = self.attn_weight(voxel_embs)
        weighted_voxel_features = weights * voxel_embs
        out = torch.zeros_like(weighted_voxel_features[:num_box])
        torch_scatter.scatter_add(weighted_voxel_features,
                                  voxel_idx, dim=0, out=out)

        out = self.fc_out(out)
        ious = self.iou_head(out)
        regs = self.reg_head(out)

        box_src = boxes_decomposed
        box_src[:, 4:7] /= 1.5
        box_s2 = self.dec_box(box_src[:, 1:], regs)

        box_s2 = torch.cat([box_src[:, :1], box_s2], dim=-1)

        self.out = {
            'rois': box_src,
            'reg': regs,
            'iou': ious,
            'box_s2': box_s2
        }

        preds = []
        for b in range(batch_dict['batch_size']):
            cur_iou = ious[box_s2[:, 0]==b].squeeze()
            mask = cur_iou > 0.
            preds.append(
                {
                    'pred_boxes': box_s2[box_s2[:, 0]==b, 1:][mask],
                    'pred_scores': cur_iou[mask],
                    'pred_labels': batch_dict['det_s1']['roi_labels'][
                                   b, :(box_s2[:, 0]==b).sum()][mask]
                }
            )
        batch_dict['det_s2'] = preds
        return batch_dict

    def get_coords(self, batch_dict):
        coords = batch_dict['pcds']
        if len(coords[:, 0].unique()) > batch_dict['batch_size']:
            for i, c in enumerate(batch_dict['num_cav']):
                idx_start = sum(batch_dict['num_cav'][:i])
                mask = torch.logical_and(
                    coords[:, 0] >= idx_start,
                    coords[:, 0] < idx_start + c
                )
                coords[mask, 0] = i
        return coords

    def voxelize_with_centroids(self, x: ME.TensorField, coords: torch.Tensor):
        cm = x.coordinate_manager
        features = x.F
        # coords = x.C[:, 1:]

        out = x.sparse()
        size = torch.Size([len(out), len(x)])
        tensor_map, field_map = cm.field_to_sparse_map(x.coordinate_key, out.coordinate_key)
        coords_p1, count_p1 = downsample_points(coords, tensor_map, field_map, size)
        norm_coords = normalize_points(coords, coords_p1, tensor_map)
        pos_emb = self.pos_emb_layer(norm_coords)
        feat_enc = self.proj(features)

        voxel_embs = self.fc_layer(torch.cat([feat_enc, pos_emb], dim=1))
        down_voxel_embs = downsample_embeddings(voxel_embs, tensor_map, size, mode="max")
        out = ME.SparseTensor(down_voxel_embs,
                              coordinate_map_key=out.coordinate_key,
                              coordinate_manager=cm)
        return out

    def loss(self, batch_dict):
        if self.empty:
            return 0.0, {'roi': 0.0}
        tgt_boxes = batch_dict['objects'][:, [0, 3, 4, 5, 6, 7, 8, 11]]
        tgt_reg, tgt_iou, pos, src_iou = self.get_tgt(tgt_boxes)
        if pos.sum() == 0:
            return 0, {}
        # selected = pos.float()
        # mask1 = src_iou > 0.7
        # mask2 = torch.logical_and(src_iou < 0.3, pos)
        loss_reg = weighted_smooth_l1_loss(self.out['reg'][pos], tgt_reg[pos])
        loss_iou = weighted_smooth_l1_loss(self.out['iou'].sigmoid(), tgt_iou)
        loss = loss_reg.mean() + loss_iou.mean()
        loss_dict = {'roi': loss}
        return loss, loss_dict

    def get_tgt(self, target_boxes):
        rois = self.out['rois']
        boxes = self.out['box_s2']
        tgt_reg, tgt_boxes_aligned, pos_mask = self.enc_box(rois, target_boxes)
        tgt_iou = aligned_boxes_iou3d_gpu(boxes[:, 1:], tgt_boxes_aligned[:, 1:])
        src_iou = aligned_boxes_iou3d_gpu(rois[:, 1:], tgt_boxes_aligned[:, 1:])
        return tgt_reg, tgt_iou, pos_mask, src_iou

    @staticmethod
    @torch.no_grad()
    def enc_box(rois, gt_bbox):
        rois = rois.detach()
        ious = boxes_iou3d_gpu(rois[:, 1:], gt_bbox[:, 1:])

        ious_aligned = []
        boxes_aligned = []
        for i in rois[:, 0].unique():
            mask1 = rois[:, 0] == i
            mask2 = gt_bbox[:, 0] == i
            idx1, idx2 = torch.where(mask1)[0], torch.where(mask2)[0]
            cur_ious_max = ious[idx1.min(): idx1.max() + 1,
                           idx2.min(): idx2.max() + 1].max(dim=1)
            ious_aligned.append(cur_ious_max[0])
            boxes = gt_bbox[mask2][cur_ious_max[1]]
            boxes_aligned.append(boxes)

        boxes_aligned = torch.cat(boxes_aligned, dim=0)
        ious_aligned = torch.cat(ious_aligned, dim=0)
        valid = ious_aligned > 0.1

        xa, ya, za, wa, la, ha, ra = torch.split(rois[valid, 1:], 1, dim=-1)
        xg, yg, zg, wg, lg, hg, rg = torch.split(boxes_aligned[valid, 1:], 1, dim=-1)

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
        reg_boxes = torch.zeros_like(rois)
        reg_boxes[valid] = ret
        return reg_boxes, boxes_aligned, valid

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