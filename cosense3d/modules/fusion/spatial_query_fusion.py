from typing import Mapping, Any

import torch
import torch.nn as nn

from cosense3d.modules import BaseModule, plugin
from cosense3d.modules.utils.common import cat_coor_with_idx
from cosense3d.modules.plugin.attn import ScaledDotProductAttention


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











