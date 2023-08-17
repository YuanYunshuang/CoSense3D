import torch
from torch import nn
from scipy.optimize import linear_sum_assignment
from queue import Queue

from cosense3d.model.utils import indices2metric, linear_last
from cosense3d.model.submodules.attention import CrossAttention
from cosense3d.ops.iou3d_nms_utils import boxes_iou_bev
from cosense3d.model.losses.common import cross_entroy_with_logits
from cosense3d.model.utils.me_utils import update_me_essentials


class TrackQueryBased(nn.Module):
    def __init__(self, cfgs):
        super().__init__()
        for name, value in cfgs.items():
            if name not in ["model", "__class__"]:
                setattr(self, name, value)
        update_me_essentials(self, cfgs['data_info'], stride=cfgs['stride'])

        # object classification layers : get topk queries
        self.objectness_layer = linear_last(
            self.input_channels, self.shared_conv_channel, 1)

        # query layers
        self.pos_emb = nn.Linear()
        self.cross_attn = CrossAttention(self.input_channels, 4,
                                         self.input_channels // 4, qkv_bias=True)
        # memory queue
        self.queue = Queue(self.history_len)

    def forward(self, batch_dict):
        stensor = batch_dict[self.feature_src][f'p{self.stride}']
        coor = stensor['coor']
        feat = stensor['feat']
        centers = indices2metric(coor, self.voxel_size)

        objectness = self.objectness_layer(feat)
        for i in range(self.history_len):
            topk = torch.topk(objectness.view(-1), k=self.topk, dim=0)

            topk_coor = coor[topk.indices]
            topk_feat = feat[topk.indices]
