import functools
from cosense3d.model.utils import *
from cosense3d.model.losses.edl import edl_mse_loss


class PointSemseg(nn.Module):
    def __init__(self, cfgs):
        super(PointSemseg, self).__init__()
        for k, v in cfgs.items():
            setattr(self, k, v)

        self.cls_layer = linear_last(64, 32, self.n_cls)

        self.semseg_loss = functools.partial(
            edl_mse_loss,
            n_cls=self.n_cls,
            annealing_step=self.annealing_step)

        self.out = {}

    def forward(self, batch_dict):
        p0 = batch_dict['backbone']
        self.out['evidence'] = self.cls_layer(p0).relu()

    def loss(self, batch_dict):
        loss, loss_dict = self.semseg_loss(
            'unet', self.out['evidence'],
            batch_dict['target_semantic'],
            batch_dict['epoch']
        )
        return loss, loss_dict