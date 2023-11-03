import os
import torch


class Hooks:
    def __init__(self, cfg):
        self.hooks = {}
        if cfg is None:
            return
        for hook_name, hook_list in cfg.items():
            self.hooks[hook_name] = []
            for hook_cfg in hook_list:
                self.hooks[hook_name].append(
                    globals()[hook_cfg['type']](**hook_cfg))

    def __call__(self, runner, hook_name):
        if hook_name in self.hooks:
            for hook in self.hooks[hook_name]:
                hook(runner)


class CheckPointsHook:
    def __init__(self, epoch_every=10, iter_every=None, **kwargs):
        self.epoch_every = epoch_every
        self.iter_every = iter_every

    def __call__(self, runner):
        if (self.iter_every is not None and runner.iter % self.iter_every == 0) or \
            runner.epoch % self.epoch_every == 0:
            torch.save({
                'epoch': runner.epoch,
                'iteration': runner.iter,
                'model_state_dict': runner.forward_runner.state_dict(),
                'optimizer_state_dict': runner.optimizer.state_dict(),
            }, os.path.join(runner.logger.log_path, f'epoch{runner.epoch}.pth'))