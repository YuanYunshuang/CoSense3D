import os
import torch
from importlib import import_module


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
        super().__init__(**kwargs)
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


class BEVSparseToDenseHook(BaseHook):
    def __init__(self, lidar_range, voxel_size, stride, **kwargs):
        super().__init__(**kwargs)
        self.stride = stride
        self.lidar_range = lidar_range
        self.voxel_size = voxel_size

    def __call__(self, runner, **kwargs):
        pass


class EvalDenseBEVHook(BaseHook):
    def __init__(self, thr, **kwargs):
        super().__init__(**kwargs)
        self.thr = thr

    def __call__(self, runner, **kwargs):
        pass


class DetectionNMSHook(BaseHook):
    def __init__(self, nms_thr, pre_max_size, **kwargs):
        super().__init__(**kwargs)
        self.nms_thr = nms_thr
        self.pre_max_size = pre_max_size
        self.nms = import_module('cosense3d.ops.iou3d_nms_utils').nms_gpu

    def __call__(self, runner, **kwargs):
        detection_out = runner.controller.data_manager.gather_ego_data('detection')
        preds = []
        cav_ids = []
        for cav_id, values in detection_out.items():
            cav_ids.append(cav_id)
            boxes = torch.cat(values['preds']['box'])
            scores = torch.cat(values['preds']['scr'])
            labels = torch.cat(values['preds']['lbl'])
            indices = torch.cat(values['preds']['idx'])  # map index for retrieving features

            if len(boxes) == 0:
                preds.append({
                    'box': torch.zeros((0, 7), device=boxes.device),
                    'scr': torch.zeros((0,), device=scores.device),
                    'lbl': torch.zeros((0,), device=labels.device),
                    'idx': torch.zeros((3, 0), device=indices.device),
                })
            else:
                keep = self.nms(
                    boxes,
                    scores,
                    thresh=self.nms_thr,
                    pre_maxsize=self.pre_max_size
                )
                preds.append({
                    'box': boxes[keep],
                    'scr': scores[keep],
                    'lbl': labels[keep],
                    'idx': indices[keep],
                })

        runner.controller.data_manager.scatter(cav_ids, {'detection': preds})
