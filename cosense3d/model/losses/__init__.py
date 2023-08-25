import functools
from cosense3d.model.losses import *


def instantiate_losses(self, loss_cfg):
    for k, v in loss_cfg.items():
        if isinstance(v['_target_'], str):
            parts = v['_target_'].split('.')
            if len(parts) == 1:
                loss_fn = globals().get(parts[-1])
            else:
                loss_fn = getattr(globals().get(parts[-2]), parts[-1])
            if v.get('args', False):
                loss_fn = functools.partial(loss_fn, **v['args'])

            setattr(self, f'loss_{k}', loss_fn)
            setattr(self, f'loss_{k}_weight', v.get('weight', 1.0))