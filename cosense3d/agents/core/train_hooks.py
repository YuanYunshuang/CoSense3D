import os
import torch


class TrainHooks:
    def __init__(self, cfg):
        self.hooks = {}
        if cfg is None:
            return
        for hook_name, hook_list in cfg.items():
            self.hooks[hook_name] = []
            for hook_cfg in hook_list:
                self.hooks[hook_name].append(
                    globals()[hook_cfg['type']](**hook_cfg))

    def __call__(self, runner, hook_name, **kwargs):
        if hook_name in self.hooks:
            for hook in self.hooks[hook_name]:
                hook(runner, **kwargs)


class BaseHook:
    def __init__(self, **kwargs):
        pass

    def __call__(self, runner, **kwargs):
        raise NotImplementedError


class CheckPointsHook(BaseHook):
    def __init__(self, max_ckpt=3, epoch_every=None, iter_every=None, **kwargs):
        self.max_ckpt = max_ckpt
        self.epoch_every = epoch_every
        self.iter_every = iter_every

    def __call__(self, runner, **kwargs):
        if ((self.iter_every is not None and runner.iter % self.iter_every == 0) or
                (self.epoch_every is not None and runner.epoch % self.epoch_every == 0)):
            self.save(runner)
        else:
            if runner.epoch > self.max_ckpt:
                os.remove(os.path.join(
                    runner.logger.log_path,
                    f'epoch{runner.epoch - self.max_ckpt}.pth'))
            self.save(runner)

    def save(self, runner):
        torch.save({
            'epoch': runner.epoch,
            'model': runner.forward_runner.state_dict(),
            'optimizer': runner.optimizer.state_dict(),
            'lr_scheduler': runner.lr_scheduler.state_dict(),
        }, os.path.join(runner.logger.log_path, f'epoch{runner.epoch}.pth'))