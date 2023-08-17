import importlib
from cosense3d.model.utils import *
from cosense3d.model.utils.target_assigner import TargetAssigner


class BevCvt(nn.Module):
    def __init__(self, cfgs):
        super(BevCvt, self).__init__()
        for k, v in cfgs.items():
            setattr(self, k, v)

        dim_total = 0
        dim_max = 0

        for _, (start, stop) in self.outputs.items():
            assert start < stop

            dim_total += stop - start
            dim_max = max(dim_max, stop)

        assert dim_max == dim_total

        self.to_logits = nn.Sequential(
            nn.Conv2d(self.in_dim, self.dim_last, 3, padding=1, bias=False),
            nn.BatchNorm2d(self.dim_last),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.dim_last, dim_max, 1))

        if 'target_assigner' in cfgs:
            self.tgt_assigner = TargetAssigner(cfgs['target_assigner'])

        self.loss_fns = {}
        for loss, loss_args in cfgs['losses'].items():
            loss_module, loss_name = loss.split('.')
            loss_module = importlib.import_module(f"model.losses.{loss_module}")
            self.loss_fns[loss_name] = {
                'weight': loss_args['weight'],
                'func': getattr(loss_module, loss_args['target'])(**loss_args['args'])
            }

        self.cnt = 0
        from cosense.tools.vis_tools import vis_cvt_pred
        self.visulizer = vis_cvt_pred

    def forward(self, batch_dict):
        x = batch_dict['decoder']
        x = self.to_logits(x)

        batch_dict['bev_semseg'] = \
            {k: x[:, start:stop] for k, (start, stop) in self.outputs.items()}

    def loss(self, batch_dict):
        # load or create target
        if hasattr(self, 'tgt_assigner'):
            tgt = self.tgt_assigner(batch_dict)
        else:
            tgt = {
                'bev': batch_dict['map_bev'],
                'center': batch_dict['map_center'],
                'visibility': batch_dict['map_visibility'],
            }

        pred = batch_dict['bev_semseg']

        if self.cnt % 20 == 0:
            self.visulizer(pred, tgt, batch_dict,
                           label_indices=self.loss_fns['bev']['func'].label_indices)
            self.cnt = 1
        else:
            self.cnt += 1

        loss_dict = {}
        loss = 0
        for k, v in self.loss_fns.items():
            loss_dict[k] = v['func'](pred, tgt)
            loss += loss_dict[k] * v['weight']
        return loss, loss_dict

