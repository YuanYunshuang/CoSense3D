import functools

import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from queue import Queue

from cosense3d.model.utils import indices2metric, linear_last, inverse_sigmoid, topk_gather
from cosense3d.utils.module_utils import instantiate_target_module
from cosense3d.ops.iou3d_nms_utils import boxes_iou_bev
from cosense3d.model.losses import instantiate_losses
from cosense3d.model.utils.me_utils import update_me_essentials
from cosense3d.model.utils.transformer_utils import pos2posemb2d, pos2posemb1d, pos2posemb3d, nerf_positional_encoding
from cosense3d.model.heads.det_center_sparse import *
from cosense3d.utils.misc import multi_apply
from cosense3d.model.utils import bias_init_with_prob
from cosense3d.model.submodules.mln import MLN


class TrackQueryBasedCam(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        for name, value in cfgs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)
        update_me_essentials(self, cfgs['data_info'], stride=cfgs['stride'])

        self.num_query = 300
        self.memory_len = 256
        self.num_propagated = 256
        self.pre_topk_proposals = 1024
        self.post_topk_proposals = 256
        self.embed_dims = 128
        self.cls_out_channels = 1
        self.num_reg_fcs = 2
        self.num_pred = 6
        self.code_size = 10
        self.bg_cls_weight = 0
        self.n_cls = 1

        self.pc_range = nn.Parameter(torch.tensor(self.lidar_range), requires_grad=False)

        self.transformer = instantiate_target_module(cfgs['transformer']['_target_'],
                                                     cfgs['transformer']['args'])

        if 'target_assigner' in cfgs:
            # update configs
            from cosense3d.model.utils.target_assigner import TargetAssigner
            self.tgt_assigner = TargetAssigner(cfgs['target_assigner'],
                                               batch_dict_key=self.__class__.__name__)

        self._get_layers()
        self.init_weights()
        self.reset_memory()

        # calculate loss directly after each frame forward
        self.loss_dict = {}

        # cls_head = globals()[self.cls_head_cfg['name']](
        #     cfgs['class_names_each_head'],
        #     cfgs['embed_dims'],
        #     one_hot_encoding=self.cls_head_cfg.get('one_hot_encoding', True),
        #     norm='LN'
        # )
        # reg_head = globals()[self.reg_head_cfg['name']](
        #     cfgs['reg_channels'],
        #     cfgs['embed_dims'],
        #     combine_channels=self.reg_head_cfg['combine_channels'],
        #     sigmoid_keys=self.reg_head_cfg['sigmoid_keys'],
        #     norm='LN'
        # )
        #
        # self.cls_heads = nn.ModuleList([cls_head for _ in range(self.num_pred)])
        # self.reg_heads = nn.ModuleList([reg_head for _ in range(self.num_pred)])
        #
        # if 'target_assigner' in cfgs:
        #     from cosense3d.model.utils.target_assigner import TargetAssigner
        #     self.tgt_assigner = TargetAssigner(cfgs['target_assigner'],
        #                                        cfgs['class_names_each_head'])
        instantiate_losses(self, self.loss_cfg)

        # self.temp = 1

    def _get_layers(self):
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))

        cls_branch.append(nn.Linear(self.embed_dims, self.cls_out_channels))
        fc_cls = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(nn.Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(nn.Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        self.cls_branches = nn.ModuleList(
            [fc_cls for _ in range(self.num_pred)])
        self.reg_branches = nn.ModuleList(
            [reg_branch for _ in range(self.num_pred)])

        self.bev_pos_encoder = nn.Sequential(
            nn.Linear(2, self.embed_dims * 4),
            nn.ReLU(),
            nn.Linear(self.embed_dims * 4, self.embed_dims),
        )

        self.memory_embed = nn.Sequential(
            nn.Linear(self.input_channels, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )
        self.featurized_pe = MLN(self.embed_dims, self.embed_dims)
        self.reference_points = nn.Embedding(self.num_query, 3)
        if self.num_propagated > 0:
            self.pseudo_reference_points = nn.Embedding(self.num_propagated, 3)

        self.query_embedding = nn.Sequential(
            nn.Linear(self.embed_dims * 3, self.embed_dims),
            nn.ReLU(),
            nn.Linear(self.embed_dims, self.embed_dims),
        )

        self.spatial_alignment = MLN(2, self.embed_dims)

        self.time_embedding = nn.Sequential(
            nn.Linear(self.embed_dims, self.embed_dims),
            nn.LayerNorm(self.embed_dims)
        )

    def init_weights(self):
        nn.init.uniform_(self.reference_points.weight.data, 0, 1)
        if self.num_propagated > 0:
            nn.init.uniform_(self.pseudo_reference_points.weight.data, 0, 1)
            self.pseudo_reference_points.weight.requires_grad = False

        self.transformer.init_weights()
        bias_init = bias_init_with_prob(0.01)
        for m in self.cls_branches:
            nn.init.constant_(m[-1].bias, bias_init)

    def reset_memory(self):
        self.memory_embedding = None
        self.memory_reference_point = None

    def pre_update_memory(self, prev_exists):
        """
        Parameters
        ----------
        prev_exists: Tensor(B,) whether previous frame exists
        """
        x = prev_exists
        B = prev_exists.size(0)
        # refresh the memory when the scene changes
        if self.memory_embedding is None:
            self.memory_embedding = x.new_zeros(B, self.memory_len, self.embed_dims)
            self.memory_reference_point = x.new_zeros(B, self.memory_len, 3)
            self.memory_velo = x.new_zeros(B, self.memory_len, 2)
        else:
            self.memory_reference_point = self.memory_reference_point[:, :self.memory_len]
            self.memory_embedding = self.memory_embedding[:, :self.memory_len]
            self.memory_velo = self.memory_velo[:, :self.memory_len]

        # for the first frame, padding pseudo_reference_points (non-learnable)
        if self.num_propagated > 0:
            pseudo_reference_points = (self.pseudo_reference_points.weight *
                                       (self.pc_range[3:6] - self.pc_range[0:3]) + self.pc_range[0:3])
            self.memory_reference_point[:, :self.num_propagated] = (self.memory_reference_point[:, :self.num_propagated]
                                                                    + (1 - x).view(B, 1, 1) * pseudo_reference_points)

    def post_update_memory(self, all_cls_scores, all_bbox_preds, outs_dec):
        rec_reference_points = all_bbox_preds[..., :3][-1]
        rec_velo = all_bbox_preds[..., -2:][-1]
        rec_memory = outs_dec[-1]
        rec_score = all_cls_scores[-1].sigmoid().topk(1, dim=-1).values[..., 0:1]

        # topk proposals
        _, topk_indexes = torch.topk(rec_score, self.post_topk_proposals, dim=1)
        rec_reference_points = topk_gather(rec_reference_points, topk_indexes).detach()
        rec_memory = topk_gather(rec_memory, topk_indexes).detach()
        rec_velo = topk_gather(rec_velo, topk_indexes).detach()

        self.memory_embedding = torch.cat([rec_memory, self.memory_embedding], dim=1)
        self.memory_reference_point = torch.cat([rec_reference_points, self.memory_reference_point], dim=1)
        self.memory_velo = torch.cat([rec_velo, self.memory_velo], dim=1)
            
    def encode_bev_points(self, points2d):
        """
        Parameters
        ----------
        points2d: Tensor(N, 3), ME coords, 1st column indicates batch indices.

        Returns:
            pos_emb: encoded bev points
        -------
        """
        coords = points2d[..., 1:] / self.stride
        coords[..., 0] = (coords[..., 0] - self.offset_sz_x) / self.size_x
        coords[..., 1] = (coords[..., 1] - self.offset_sz_y) / self.size_y
        pos_emb = self.bev_pos_encoder(inverse_sigmoid(coords))
        
        return pos_emb, coords

    def prepare_for_dn(self, batch_size, reference_points, gt_boxes=None):
        if gt_boxes is not None:
            # TODO gt guided training
            raise NotImplementedError
        else:
            padded_reference_points = reference_points.unsqueeze(0).repeat(batch_size, 1, 1)
            attn_mask = None
            mask_dict = None

        return padded_reference_points, attn_mask, mask_dict

    def temporal_alignment(self, query_pos, tgt, reference_points):
        """
        1. Load memory embeddings and reference points.
        TODO:
            2. Encode new tgt and query_pos with ego motion models
            3. Encode memory embeddings and memory_pos with ego memory ego motion models
            4. Encode new query pos and memory query pos with corresponding timestamps
        5. Put the propagated historical query pos and memory into current query pos and tgt
        """
        B = query_pos.size(0)

        temp_reference_point = (self.memory_reference_point - self.pc_range[:3]) / (
                    self.pc_range[3:6] - self.pc_range[:3])
        temp_pos = self.query_embedding(pos2posemb3d(temp_reference_point))
        temp_memory = self.memory_embedding

        if self.num_propagated > 0:
            tgt = torch.cat([tgt, temp_memory[:, :self.num_propagated]], dim=1)
            query_pos = torch.cat([query_pos, temp_pos[:, :self.num_propagated]], dim=1)
            reference_points = torch.cat([reference_points, temp_reference_point[:, :self.num_propagated]], dim=1)
            temp_memory = temp_memory[:, self.num_propagated:]
            temp_pos = temp_pos[:, self.num_propagated:]

        return tgt, query_pos, reference_points, temp_memory, temp_pos

    def forward(self, batch_dict):
        roi_dict = batch_dict[self.feature_src][f'p{self.stride}']

        B = batch_dict['batch_size']
        seq_len = batch_dict['seq_len']
        out_dict = {}

        if self.training:
            # seq: 0-->history_len = new --> old
            for i in range(seq_len-1, -1, -1):
                require_grad = True if i < self.num_frame_grad else False
                return_loss = True if i < self.num_frame_loss else False

                if not require_grad:
                    self.eval()
                    with torch.no_grad():
                        self.frame_forward(roi_dict, seq_len, i, B, batch_dict)
                    self.train()
                else:
                    out_dict = self.frame_forward(roi_dict, seq_len, i, B, batch_dict)

                if return_loss:
                    loss_dict = self._loss(out_dict, self.get_gt_boxes(batch_dict, i))
                    self.loss_dict[i] = loss_dict

        batch_dict['track_query_based'] = out_dict

    def frame_forward(self, roi_dict, seq_len, seq_idx, B, batch_dict):
        coor = roi_dict['coor']
        feat = roi_dict['feat']
        rois = roi_dict['cls'][..., 1]
        topk_coors = []
        topk_feats = []
        for b in range(B):
            mask = coor[:, 0] == (b * (seq_len) + seq_idx)
            topk = torch.topk(rois[mask].view(-1), k=self.pre_topk_proposals, dim=0)
            topk_coor = coor[mask][topk.indices]
            topk_feats.append(feat[mask][topk.indices])
            topk_coors.append(topk_coor)

            # if b == 0 and seq_idx == 0:
            #     from cosense3d.utils.vislib import draw_points_boxes_plt
            #     points = roi_dict['centers'][mask, 1:]
            #     topk_points = points[topk.indices]
            #     boxes = batch_dict['objects']
            #     boxes = boxes[boxes[:, 0] == (b * (seq_len) + seq_idx)][:, [3, 4, 5, 6, 7, 8, 11]]
            #     ax = draw_points_boxes_plt(
            #         pc_range=self.lidar_range,
            #         points=points.detach().cpu().numpy(),
            #         points_c='gray',
            #         return_ax=True
            #     )
            #     draw_points_boxes_plt(
            #         points=topk_points.detach().cpu().numpy(),
            #         points_c='r',
            #         boxes_gt=boxes.detach().cpu().numpy(),
            #         ax=ax,
            #         filename=f'/home/yuan/Downloads/tmp{b}_{seq_idx}.png'
            #     )
            #     pass

        topk_coors = torch.stack(topk_coors, dim=0)
        topk_feats = torch.stack(topk_feats, dim=0)
        memory = self.memory_embed(topk_feats)

        topk_pos_emb, coords = self.encode_bev_points(topk_coors)
        memory = self.spatial_alignment(memory, coords)
        topk_pos_emb = self.featurized_pe(topk_pos_emb, memory)
        self.pre_update_memory(prev_exists=torch.ones_like(topk_feats[:, 0, 0]))

        ref_points = self.reference_points.weight
        ref_points, attn_mask, mask_dict = self.prepare_for_dn(B, ref_points)
        query_pos = self.query_embedding(pos2posemb3d(ref_points))
        tgt = torch.zeros_like(query_pos)

        tgt, query_pos, ref_points, temp_memory, temp_pos = \
            self.temporal_alignment(query_pos, tgt, ref_points)

        # transformer here is from StreamPETR
        attn_mask = None
        # TODO: use gt boxes to discriminate fore- and background to speed up training
        outs_dec, _ = self.transformer(
            memory, tgt, query_pos, topk_pos_emb, attn_mask, temp_memory, temp_pos)
        # outs_dec = memory.unsqueeze(0)

        outs_dec = torch.nan_to_num(outs_dec)
        outputs_classes = []
        outputs_coords = []
        for lvl in range(outs_dec.shape[0]):
            reference = inverse_sigmoid(ref_points.clone())
            assert reference.shape[-1] == 3
            outputs_class = self.cls_branches[lvl](outs_dec[lvl])
            # print(outputs_class.mean())
            tmp = self.reg_branches[lvl](outs_dec[lvl])

            tmp[..., 0:3] += reference[..., 0:3]
            tmp[..., 0:3] = tmp[..., 0:3].sigmoid()

            outputs_coord = tmp
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)

        all_cls_scores = torch.stack(outputs_classes)
        all_bbox_preds = torch.stack(outputs_coords)
        all_bbox_preds[..., 0:3] = (
                all_bbox_preds[..., 0:3] * (self.pc_range[3:6] - self.pc_range[0:3]
                                            ) + self.pc_range[0:3])

        # update the memory bank
        self.post_update_memory(all_cls_scores, all_bbox_preds, outs_dec)

        outs = {
            'all_cls_scores': all_cls_scores,
            'all_bbox_preds': all_bbox_preds,
            'dn_mask_dict': None,
        }

        return outs

    def get_gt_boxes(self, batch_dict, seq_idx):
        objects = batch_dict['objects']
        velo = batch_dict['objects_velo']
        seq_len = batch_dict['seq_len']
        B = batch_dict['batch_size']

        gt_boxes = []
        for b in range(B):
            mask = objects[:, 0] == seq_idx + b * seq_len
            boxes = torch.cat([objects[mask, 2:9], objects[mask, 11:], velo[mask]], dim=-1)
            gt_boxes.append(boxes)
        return gt_boxes

    def _loss(self, pred_dict, gt_boxes):
        all_cls_scores = pred_dict['all_cls_scores']
        all_bbox_preds = pred_dict['all_bbox_preds']

        # vis_box_pred = all_bbox_preds[-1][0][all_cls_scores[-1][0].squeeze().sigmoid() > 0.5]
        cls_res = all_cls_scores[-1][0].squeeze().sigmoid().view(-1)
        # print(torch.topk(cls_res, 10)[0].detach().cpu().numpy().tolist())
        print(cls_res.min().item(), cls_res.max().item(), cls_res.mean().item())

        num_dec_layers = len(all_cls_scores)
        gt_boxes_list = [gt_boxes for _ in range(num_dec_layers)]

        losses_cls, losses_box = multi_apply(
            self._layerwise_loss, all_bbox_preds, all_cls_scores, gt_boxes_list
        )

        loss = 0
        loss_dict = {}
        # loss from the last decoder layer
        loss_dict['loss_cls'] = losses_cls[-1]
        loss_dict['loss_box'] = losses_box[-1]
        # get total loss
        for i in range(num_dec_layers):
            loss = loss + losses_cls[i] + losses_box[i]
            # loss_dict[f'd{num_dec_layer}.loss_cls'] = loss_cls_i
            # loss_dict[f'd{num_dec_layer}.loss_bbox'] = loss_bbox_i

        loss_dict['petr'] = loss

        return loss_dict

    def _layerwise_loss(self, pred_boxes, cls_scores, gt_boxes_list):
        batch_dict = dict(
            pred_boxes=pred_boxes,
            pred_scores=cls_scores,
            gt_boxes=gt_boxes_list
        )
        tgt = self.tgt_assigner(batch_dict)['hungarian3d']

        # (labels_list, label_weights_list, bbox_targets_list,
        #  bbox_weights_list, pos_inds_list, neg_inds_list) = multi_apply(
        #     self.reformat_tgt, pred_boxes, gt_boxes_list,
        #     tgt['assigned_gt_inds'], tgt['assigned_labels']
        # )

        labels = torch.cat(tgt['aligned_tgt_labels'], 0)
        num_total_pos = sum(tgt['num_gts'])
        num_total_neg = (labels < 0).sum()

        # neg_indices = torch.where(labels < 0)[0]
        # sampled_neg = torch.randperm(num_total_neg)[:num_total_pos * 5]
        # sampled_neg = neg_indices[sampled_neg]
        # cls_cared = (labels >= 0)
        # cls_cared[sampled_neg] = True

        pos = labels >= 0
        labels[labels == -1] = self.n_cls
        aligned_bbox_targets = torch.cat(tgt['aligned_tgt_boxes'], 0)[pos]
        aligned_bbox_preds = torch.cat(tgt['aligned_pred_boxes'], 0)[pos]

        # classification loss
        cls_scores = cls_scores.reshape(-1, self.cls_out_channels)
        # construct weighted avg_factor to match with the official DETR repo
        cls_avg_factor = num_total_pos * 1.0 + \
                         num_total_neg * self.bg_cls_weight

        cls_avg_factor = max(cls_avg_factor, 1)
        # loss_cls = self.loss_cls(cls_scores[cls_cared].squeeze(), labels[cls_cared].float())
        loss_cls = self.loss_cls(cls_scores, labels)
        eps = torch.finfo(torch.float32).eps
        loss_cls = loss_cls.sum() / (cls_avg_factor + eps) * self.loss_cls_weight
        # loss_cls = loss_cls.mean() * self.loss_cls_weight

        # regression L1 loss
        isnotnan = torch.isfinite(aligned_bbox_targets).all(dim=-1)

        loss_bbox = self.loss_box(
            aligned_bbox_preds[isnotnan, :10], aligned_bbox_targets[isnotnan, :10])
        loss_bbox = loss_bbox.sum() / (num_total_pos + eps) * self.loss_box_weight

        loss_cls = torch.nan_to_num(loss_cls)
        loss_bbox = torch.nan_to_num(loss_bbox)
        return loss_cls, loss_bbox

    def reformat_tgt(self, pred_boxes, gt_bboxes, assigned_gt_inds, assigned_labels):
        gt_labels = gt_bboxes[:, 0].long()
        gt_bboxes = gt_bboxes[:, 1:]
        pos_inds = torch.nonzero(
            assigned_gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assigned_gt_inds == 0, as_tuple=False).squeeze(-1).unique()

        # gt_flags = pred_boxes.new_zeros(pred_boxes.shape[0], dtype=torch.uint8)
        # pos_boxes = pred_boxes[pos_inds]
        # neg_boxes = pred_boxes[neg_inds]
        # pos_is_gt = gt_flags[pos_inds]

        num_gts = gt_bboxes.shape[0]
        num_bboxes = pred_boxes.size(0)
        pos_assigned_gt_inds = assigned_gt_inds[pos_inds] - 1

        if gt_bboxes.numel() == 0:
            # hack for index error case
            assert pos_assigned_gt_inds.numel() == 0
            pos_gt_bboxes = torch.empty_like(gt_bboxes).view(-1, 4)
        else:
            if len(gt_bboxes.shape) < 2:
                gt_bboxes = gt_bboxes.view(-1, 4)

            pos_gt_bboxes = gt_bboxes[pos_assigned_gt_inds.long(), :]

        # if assigned_labels is not None:
        #     pos_gt_labels = assigned_labels[pos_inds]
        # else:
        #     pos_gt_labels = None

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.cls_out_channels,
                                    dtype=torch.long)
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        code_size = gt_bboxes.size(1)
        bbox_targets = torch.zeros_like(pred_boxes)[..., :code_size]
        bbox_weights = torch.zeros_like(pred_boxes)
        # print(gt_bboxes.size(), bbox_pred.size())
        # DETR
        if num_gts > 0:
            bbox_targets[pos_inds] = pos_gt_bboxes
            bbox_weights[pos_inds] = 1.0
            labels[pos_inds] = gt_labels[pos_assigned_gt_inds]

        return (labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def loss(self, batch_dict):
        loss_dict = {}
        total_loss = 0
        for seq_idx, ldict in self.loss_dict.items():
            loss_dict[f'petr_seq{seq_idx}'] = ldict['petr']
            loss_dict[f'petr_cls_seq{seq_idx}'] = ldict['loss_cls']
            loss_dict[f'petr_box_seq{seq_idx}'] = ldict['loss_box']
            total_loss = total_loss + ldict['petr']
        return total_loss, loss_dict
