import os
import torch


class ModuleHooks:
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
    def __init__(self, iter_every=10, **kwargs):
        self.iter_every = iter_every

    def __call__(self, runner, **kwargs):
        raise NotImplementedError


class VisCenterness(BaseHook):

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


class VisDetection(BaseHook):

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