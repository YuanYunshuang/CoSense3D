import warnings
from typing import Dict

import torch

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.plugin.attn import ScaledDotProductAttention
from cosense3d.modules.utils.me_utils import update_me_essentials
from cosense3d.modules.utils.common import cat_coor_with_idx


class SparseAttentionFusion(BaseModule):
    def __init__(self, data_info, stride, in_channels, **kwargs):
        super(SparseAttentionFusion, self).__init__(**kwargs)
        update_me_essentials(self, data_info, stride=stride)
        self.d = len(self.voxel_size)
        self.attn = ScaledDotProductAttention(in_channels)

    def forward(self, ego_feats, coop_feats=None, **kwargs):
        fused_feat = []
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            coor = [ego_feat[f'p{self.stride}']['coor']]
            feat = [ego_feat[f'p{self.stride}']['feat']]
            # fuse coop to ego
            for cpfeat in coop_feat.values():
                if 'pts_feat' not in cpfeat:
                    continue
                coor.append(cpfeat['pts_feat'][f'p{self.stride}']['coor'])
                feat.append(cpfeat['pts_feat'][f'p{self.stride}']['feat'])
            coor_cat = cat_coor_with_idx(coor)
            feat_cat = torch.cat(feat, dim=0)
            uniq_coor, reverse_inds = torch.unique(coor_cat[:, 1:], eturn_inverse=True)

            feats_pad = []
            for i, c in enumerate(coor):
                feat_pad = feat_cat.new_zeros(len(uniq_coor), feat.shape[-1])
                feat_pad[reverse_inds[coor_cat[:, 0] == 0]] = feat[i]
                feats_pad.append(feat_pad)
            q = feats_pad[0].unsqueeze(1)  # num_pts, 1, d
            kv = torch.stack(feats_pad[1:], dim=1)  # num_pts, num_coop_cav, d
            out = self.attn(q, kv, kv)
            fused_feat.append({
                f'p{self.stride}': {
                    'coor': uniq_coor,
                    'feat': out
                }
            })
        return self.format_output(fused_feat)

    def format_output(self, output):
        return {self.scatter_keys[0]: output}


class DenseAttentionFusion(BaseModule):
    def __init__(self, feature_dim, neck=None, **kwargs):
        super(DenseAttentionFusion, self).__init__(**kwargs)
        self.attn = ScaledDotProductAttention(feature_dim)
        if neck is not None:
            self.neck = plugin.build_plugin_module(neck)

    def forward(self, ego_feats, coop_feats=None, **kwargs):
        out = []
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            feat = [ego_feat]
            for cpfeat in coop_feat.values():
                if 'bev_feat' not in cpfeat:
                    continue
                feat.append(cpfeat['bev_feat'])
            xx = torch.stack(feat, dim=0)
            N, C, H, W = xx.shape
            xx = xx.view(N, C, -1).permute(2, 0, 1)
            h = self.attn(xx, xx, xx)
            h = h.permute(1, 2, 0).view(N, C, H, W)[0, ...]
            out.append(h)
        out = torch.stack(out)
        if hasattr(self, 'neck'):
            out = self.neck(out)
        return {self.scatter_keys[0]: out}

