import torch
from torch import nn
from scipy.optimize import linear_sum_assignment

from cosense3d.model.utils import linear_layers, linear_last
from cosense3d.ops.iou3d_nms_utils import boxes_iou_bev
from cosense3d.model.losses.common import cross_entroy_with_logits


class TrackBase(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        for name, value in cfgs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)

        self.projection_layers = linear_layers(cfgs['projection_layers'])
        self.out_layer = linear_last(cfgs['projection_layers'][-1], 32, 1)
        self.tracker = None

    def forward(self, batch_dict):
        feature = batch_dict[self.input_feature_name]
        if isinstance(feature, dict):
            feature = feature[f"p{self.stride}"]

        feature = feature.permute(0, 2, 3, 1)
        # extract features for center points
        pred_boxes = batch_dict['det_s1']  # list, B*seq_len

        seq_len = batch_dict['seq_len']
        similarities = []
        # loop over batch
        for i in range(len(batch_dict['frame'])):
            center_features = []
            center_locs = []
            cur_frame_indices = torch.arange(seq_len) + i * seq_len
            assert len(cur_frame_indices) == 2, 'Only support sequence length of 2.'
            for fidx in cur_frame_indices:
                if self.use_gt_boxes and fidx != cur_frame_indices[-1]:
                    midx = self.center_indices_from_gt_boxes(batch_dict, fidx)
                else:
                    midx = pred_boxes[fidx]['idx']  # feature map indices
                center_locs.append(midx[1:])
                # print(midx.shape)
                center_features.append(
                    self.projection_layers(feature[midx[0], midx[1], midx[2]])
                )
            # calculate similarity
            if self.training or self.tracker is None or self.use_gt_boxes:
                similarity = self.similarity(center_features, center_locs)
            else:
                similarity = self.track_phase_similarity(center_features, center_locs)
                self.tracker.update(center_features[0], center_locs[0], batch_dict)

            similarities.append(similarity)

        batch_dict['center_similarity'] = similarities

    def center_indices_from_gt_boxes(self, batch_dict, frame_idx):
        gt_boxes = batch_dict['objects']
        gt_boxes = gt_boxes[gt_boxes[:, 0]==frame_idx]
        boxes = gt_boxes[:, [3, 4, 5, 6, 7, 8, 11]]
        # ids = gt_boxes[:, 1]
        # lbls = gt_boxes[:, 2]
        # convert box centers to map indices
        indices = torch.floor((boxes[:, :2] + self.data_info['det_r']) / self.meter_per_pixel).long()
        batch_indices = torch.ones_like(indices[:, :1]) * frame_idx
        indices = torch.cat([batch_indices, indices], dim=-1).T
        return indices

    def similarity(self, center_features, center_locs):
        similarity = []
        for center_feat, center_loc in zip(center_features[:-1], center_locs[:-1]):
            # compare the last and previous frames
            s = self.get_similarity(
                center_feat, center_features[-1],
                center_loc, center_locs[-1],
                neg_gate=False
            )
            similarity.append(s)
        return similarity[-1]

    def track_similarity_to_last_state(self, center_features, center_locs):
        assert len(center_features) == 1
        last_state = self.tracker.state
        if last_state is not None:
            similarity = self.get_similarity(
                last_state['features'], center_features[0],
                last_state['centers'], center_locs[0]
            )
        else:
            similarity = None

        return similarity

    def get_similarity(self, feat1, feat2, ctr1=None, ctr2=None, neg_gate=False):
        diff = self.laplacian_dists(feat1, feat2)  # N1, N2, d
        if neg_gate:
            assert ctr1 is not None and ctr2 is not None
            euc_dists = self.euclidian_dists(ctr1.T, ctr2.T)
            diff[euc_dists * self.meter_per_pixel > 10] = 0
        out = self.out_layer(diff.view(-1, diff.shape[-1])).view(diff.shape[:-1])
        return out

    def get_tgt(self, batch_dict):
        gt_boxes = batch_dict['objects']
        pred_boxes = batch_dict['det_s1']
        # match pred and gt boxes
        pred_ids = []
        for i, boxes in enumerate(pred_boxes):
            cur_pred_boxes = boxes['box']
            cur_gt_boxes = gt_boxes[gt_boxes[:, 0] == i]
            ious = boxes_iou_bev(cur_pred_boxes, cur_gt_boxes[:, [3, 4, 5, 6, 7, 8, 11]])
            iou_max, max_idx = ious.max(dim=1)
            box_ids_of_preds = cur_gt_boxes[max_idx, 1].int()
            box_ids_of_preds[iou_max <= 0.01] = -1
            pred_ids.append(box_ids_of_preds)

        match_matrix = []
        seq_len = batch_dict['seq_len']
        for b, frames in enumerate(batch_dict['frame']):
            mats = []
            for fi in range(len(frames[:-1])):
                a = pred_ids[fi + b * seq_len].unsqueeze(1)
                b = pred_ids[(b + 1) * seq_len - 1].unsqueeze(0)
                mat = (a == b) * (a >= 0) * (b >= 0)
                mats.append(mat)
            match_matrix.append(mats[-1])

        return match_matrix

    def loss(self, batch_dict):
        tgt_match = self.get_tgt(batch_dict)
        similarity = batch_dict['center_similarity']

        # batch_dict['match_matrix'] = tgt_match
        # batch_dict.pop('in_data')
        # batch_dict.pop('backbone')
        # torch.save(batch_dict, "/media/hdd/yuan/Downloads/batch_dict.pth")
        # from tools.vis_tools import vis_matching_matrix
        # vis_matching_matrix(batch_dict)

        loss = 0
        for src, tgt in zip(similarity, tgt_match):
            pos = tgt.sum(dim=0) == 1
            src_pos = src.T[pos]
            tgt_pos = tgt.T[pos]
            lbl = torch.where(tgt_pos)[1].float()
            loss += cross_entroy_with_logits(src_pos, lbl, n_cls=tgt_pos.shape[1]).mean()
        loss = loss * 0.1 / batch_dict['batch_size']
        return loss, {'track': loss}

    def get_assignments(self, batch_dict):
        """Use Hungarian algorithm to find assigments based on center similarity"""
        similarity = batch_dict['center_similarity']
        assignments = []
        for s in similarity:
            cost = (s.max() - s).detach().cpu().numpy()
            inds = linear_sum_assignment(cost)
            assignments.append(inds)
        return assignments

    @staticmethod
    def laplacian_dists(a, b):
        a = a.float().unsqueeze(1)
        b = b.float().unsqueeze(0)
        return a - b

    @staticmethod
    def euclidian_dists(a, b):
        a = a.float().unsqueeze(1)
        b = b.float().unsqueeze(0)
        return torch.norm(a - b, dim=-1)