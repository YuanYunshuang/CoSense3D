from torch import nn

from cosense3d.modules import BaseModule
from cosense3d.modules import build_module


class MultiTaskHead(BaseModule):
    def __init__(self,
                 heads,
                 strides,
                 losses,
                 **kwargs):
        super().__init__(**kwargs)
        self.losses = losses
        modules = []
        for i, h in enumerate(heads):
            h.update(dict(
                stride=strides[i],
                gather_keys=self.gather_keys,
                scatter_keys=[self.scatter_keys[i]],
                gt_keys=self.gt_keys,
            ))
            modules.append(build_module(h))
        self.heads = nn.ModuleList(modules)

    def forward(self, stensor_list, *args, **kwargs):
        out = {}
        for i, h in enumerate(self.heads):
            out.update(h(stensor_list, *args, **kwargs))

        return out

    def loss(self, *args, **kwargs):
        kl = len(self.scatter_keys)
        heads_out = args[:kl]
        gt_boxes, gt_labels = args[kl:kl + 2]
        loss_dict = {}
        for i, h in enumerate(self.heads):
            if self.losses[i]:
                loss_dict.update(h.loss(heads_out[i], gt_boxes, gt_labels, **kwargs))
        return loss_dict