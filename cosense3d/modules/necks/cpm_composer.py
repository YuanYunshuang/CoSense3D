import torch
from torch import nn

from cosense3d.modules import BaseModule, plugin


class KeypointComposer(BaseModule):
    def __init__(self, vsa, **kwargs):
        super().__init__(**kwargs)
        self.vsa = plugin.build_plugin_module(vsa)

    def forward(self, preds, bev_feat, voxel_feat, points, **kwargs):
        res = self.vsa(preds, bev_feat, voxel_feat, points)
        res = self.compose_result_list(res, len(preds))
        return {self.scatter_keys[0]: res}

