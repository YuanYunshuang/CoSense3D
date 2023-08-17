import torch
from torch import nn
from cosense3d.model.utils import minkconv_conv_block


class Formatting(nn.Module):
    def __init__(self, cfgs):
        super(Formatting, self).__init__()
        self.functions = cfgs['functions']

    def forward(self, batch_dict):
        for func in self.functions:
            getattr(self, func)(batch_dict)

    def singleton_to_refine(self, batch_dict):
        rois = batch_dict['objects'].clone()[:, [0, 3, 4, 5, 6, 7, 8, 11]]
        sl = batch_dict['seq_len']
        bs = batch_dict['batch_size']
        # 0-->sl: new-->old
        for b in range(bs):
            rois[b*sl:b*sl+sl, 0] = torch.arange(sl) + sl * b
            # accurate center for newest frame not known
            rois[b*sl, 1:3] = 0
            # copy z, l, w, h and yaw from previous frame
            rois[b*sl, 3:] = rois[b*sl + 1, 3:]

        batch_dict['roi'] = {'box': rois}
