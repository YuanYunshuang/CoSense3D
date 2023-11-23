import torch
import torch.nn as nn

from cosense3d.modules import BaseModule


class BEVMaxoutFusion(BaseModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, ego_feats, coop_feats):
        out_feat = []
        for ego_feat, coop_feat in zip(ego_feats, coop_feats):
            feat = [ego_feat]
            for cpfeat in coop_feat.values():
                if 'pts_feat' not in cpfeat:
                    continue
                feat.append(cpfeat['bev_feat'])
            feat = torch.stack(feat, dim=0).max(dim=0).values
            out_feat.append(feat)

        return {self.scatter_keys[0]: out_feat}