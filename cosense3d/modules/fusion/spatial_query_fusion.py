from typing import Mapping, Any

import torch
import torch.nn as nn

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.common import cat_coor_with_idx
from cosense3d.modules.plugin.attn import ScaledDotProductAttention
from cosense3d.modules.utils.localization_utils import register_points
from cosense3d.modules.utils.common import pad_r
from cosense3d.modules.utils.misc import MLN
import cosense3d.modules.utils.positional_encoding as PE


class SpatialQueryFusion(BaseModule):
    def __init__(self,
                 in_channels,
                 pc_range,
                 resolution,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.resolution = resolution
        self.attn = ScaledDotProductAttention(in_channels)

    def forward(self, ego_feats, coop_feats, **kwargs):
        fused_feat = []
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            coor = [ego_feat['ref_pts']]
            feat = [ego_feat['outs_dec'][-1]]
            if len(coop_feat) == 0:
                fused_feat.append({
                        'ref_pts': coor[0],
                        'outs_dec': feat[0].unsqueeze(1)
                })
                continue

            # fuse coop to ego
            for cpfeat in coop_feat.values():
                coor.append(cpfeat[self.gather_keys[0]]['ref_pts'])
                feat.append(cpfeat[self.gather_keys[0]]['outs_dec'][-1])
            coor_cat = cat_coor_with_idx(coor)
            feat_cat = torch.cat(feat, dim=0)
            # coor_int = coor_cat[:, 1:] * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
            # coor_int = (coor_int * (1 / self.resolution)).int()
            uniq_coor, reverse_inds = torch.unique(coor_cat[:, 1:], dim=0,
                                                   return_inverse=True)

            feats_pad = []
            for i, c in enumerate(coor):
                feat_pad = feat_cat.new_zeros(len(uniq_coor), feat_cat.shape[-1])
                feat_pad[reverse_inds[coor_cat[:, 0] == i]] = feat[i]
                feats_pad.append(feat_pad)
            q = feats_pad[0].unsqueeze(1)  # num_pts, 1, d
            kv = torch.stack(feats_pad, dim=1)  # num_pts, num_coop_cav, d
            out = self.attn(q, kv, kv).squeeze(1)
            fused_feat.append({
                'ref_pts': uniq_coor,
                'outs_dec': out.unsqueeze(1)
            })
        return self.format_output(fused_feat)

    def format_output(self, output):
        return {self.scatter_keys[0]: output}


class SpatialQueryAlignFusionRL(BaseModule):
    def __init__(self,
                 in_channels,
                 pc_range,
                 resolution,
                 num_pose_feat=64,
                 **kwargs):
        super().__init__(**kwargs)
        self.pc_range = nn.Parameter(torch.tensor(pc_range), requires_grad=False)
        self.resolution = resolution
        self.emb_dim = in_channels
        self.attn = ScaledDotProductAttention(in_channels)
        self.pose_pe = MLN(4 * 12, f_dim=self.emb_dim)
        self.num_pose_feat = num_pose_feat
        self.position_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )
        self.query_embedding = nn.Sequential(
            nn.Linear(self.num_pose_feat * 2, self.emb_dim),
            nn.ReLU(),
            nn.Linear(self.emb_dim, self.emb_dim),
        )

    def forward(self, det_local, roadline, roadline_preds, ego_queries,
                ego_pose_corrected, ego_poses, ego_poses_aug,
                cpms, **kwargs):
        fused_feat = []
        for i, cpm in enumerate(cpms):
            det = det_local[i]
            ego_rl, ego_rl_pred, ego_query = roadline[i], roadline_preds[i], ego_queries[i]
            ego_pose_corr, ego_pose, pose_aug2g = ego_pose_corrected[i], ego_poses[i], ego_poses_aug[i]
            # augment-frame to ego-aligned-world frame
            Taug2eaw = ego_pose_corr @ ego_pose.inverse() @ pose_aug2g
            ego_bctr = det['preds']['box'][:, :2]
            ego_coor = ego_query['ref_pts']
            ego_coor_emb = self.query_embedding(PE.pos2posemb2d(ego_coor[:, :2], self.num_pose_feat))
            ego_feat = ego_query['outs_dec'][-1] + ego_coor_emb
            ego_coor = ego_coor * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
            coor = [ego_coor] # in augment-frame
            feat = [ego_feat]
            if len(cpm) == 0:
                fused_feat.append({
                        'ref_pts': coor[0],
                        'outs_dec': feat[0].unsqueeze(1)
                })
                continue

            # fuse coop to ego
            for cpfeat in cpm.values():
                if len(cpfeat['box_ctrs']) == 0:
                    continue
                # transformation matrix coop-aligned-world frame to ego-aligned-world frame
                if self.training:
                    # during training, ground-truth poses are used, caw-frame==eaw-frame
                    Tcaw2aug = Taug2eaw.inverse()
                else:
                    Tcaw2eaw = self.align_coordinates(ego_bctr, ego_rl, ego_rl_pred, Taug2eaw, cpfeat)
                    Tcaw2aug = Taug2eaw.inverse() @ Tcaw2eaw
                T = Tcaw2aug @ cpfeat['Taug2caw']

                # encode the transformation matrix that transforms feature points
                # from erroneous ego-frame to the corrected ego-frame
                ref_pts = (T @ pad_r(cpfeat['ref_pts'], 1.0).T)[:3].T
                ref_pts_norm = (ref_pts - self.pc_range[:3]) / (self.pc_range[3:] - self.pc_range[:3])
                rot_emb = PE.nerf_positional_encoding(T[:2, :2].flatten(-2)).repeat(len(ref_pts), 1)
                pos_emb = self.position_embedding(PE.pos2posemb2d(ref_pts_norm[:, :2], self.num_pose_feat))
                transform_emb = self.pose_pe(pos_emb, rot_emb)
                coor.append(ref_pts)
                feat.append(cpfeat['feat'][-1] + transform_emb)

                # inplace transformation for coop point cloud: only for visualization in GLViewer
                cpfeat['points'][:, :3] = (T @ pad_r(cpfeat['points'][:, :3], 1.0).T)[:3].T

            coor_cat = cat_coor_with_idx(coor)
            feat_cat = torch.cat(feat, dim=0)
            # coor_int = coor_cat[:, 1:] * (self.pc_range[3:] - self.pc_range[:3]) + self.pc_range[:3]
            coor_int = (coor_cat[:, 1:] * (1 / self.resolution)).int()
            uniq_coor, reverse_inds = torch.unique(coor_int, dim=0, return_inverse=True)
            uniq_coor = (uniq_coor * self.resolution - self.pc_range[:3]) / (self.pc_range[3:] - self.pc_range[:3])

            feats_pad = []
            for i, c in enumerate(coor):
                feat_pad = feat_cat.new_zeros(len(uniq_coor), feat_cat.shape[-1])
                feat_pad[reverse_inds[coor_cat[:, 0] == i]] = feat[i]
                feats_pad.append(feat_pad)
            q = feats_pad[0].unsqueeze(1)  # num_pts, 1, d
            kv = torch.stack(feats_pad, dim=1)  # num_pts, num_coop_cav, d
            out = self.attn(q, kv, kv).squeeze(1)
            fused_feat.append({
                'ref_pts': uniq_coor,
                'outs_dec': out.unsqueeze(1)
            })

            # from cosense3d.utils.vislib import draw_points_boxes_plt, plt
            # ax = draw_points_boxes_plt(pc_range=self.pc_range.tolist(), return_ax=True)
            # for pts in coor:
            #     pts = pts.detach().cpu().numpy()
            #     ax.plot(pts[:, 0], pts[:, 1], '.', markersize=1)
            # plt.savefig("/home/yuan/Downloads/tmp.png")
            # plt.close()
        return self.format_output(fused_feat)

    def format_output(self, output, **kwargs):
        return {self.scatter_keys[0]: output}

    def align_coordinates(self, ego_bctr, ego_rl, ego_rl_pred, ego_pose, cpfeat):
        coop_bctr = cpfeat['box_ctrs']
        coop_rl = cpfeat['roadline']

        # transform ego points from aug-frame to ego-aligned world-frame
        ego_bctr = (ego_pose @ pad_r(pad_r(ego_bctr, 0.0), 1.0).T).T
        ego_rl_pred = (ego_pose @ pad_r(pad_r(ego_rl_pred, 0.0), 1.0).T).T
        coop_pts = pad_r(torch.cat([coop_rl, coop_bctr], dim=0))
        ego_pts = torch.cat([pad_r(ego_rl[:, :3]), ego_bctr[:, :3]], dim=0)

        transform, coop_pts_tf = register_points(coop_pts, ego_pts, thr=0.8)

        # import matplotlib.pyplot as plt
        # ego_bctr_vis = ego_bctr.detach().cpu().numpy()
        # ego_rl_pred_vis = ego_rl_pred.detach().cpu().numpy()
        # ego_rl_vis = ego_rl.detach().cpu().numpy()
        # coop_bctr_vis = coop_bctr.detach().cpu().numpy()
        # coop_rl_vis = coop_rl.detach().cpu().numpy()
        #
        # plt.plot(ego_rl_vis[:, 0], ego_rl_vis[:, 1], 'g.', markersize=1)
        # plt.plot(ego_rl_pred_vis[:, 0], ego_rl_pred_vis[:, 1], 'y.', markersize=1)
        # plt.plot(ego_bctr_vis[:, 0], ego_bctr_vis[:, 1], 'yo', markersize=5, markerfacecolor='none')
        # plt.plot(coop_rl_vis[:, 0], coop_rl_vis[:, 1], 'r.', markersize=1)
        # plt.plot(coop_bctr_vis[:, 0], coop_bctr_vis[:, 1], 'ro', markersize=5, markerfacecolor='none', alpha=0.5)
        # # plt.plot(coop_pts_tf[:, 0], coop_pts_tf[:, 1], 'b.', markersize=1)
        # plt.savefig("/home/yys/Downloads/tmp.png")
        # plt.close()

        return torch.from_numpy(transform).float().to(ego_pose.device)













