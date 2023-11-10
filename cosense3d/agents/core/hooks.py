import os
import time

import torch
from importlib import import_module


class Hooks:
    def __init__(self, cfg):
        self.hooks = []
        if cfg is None:
            return
        for hook_cfg in cfg:
            self.hooks.append(
                globals()[hook_cfg['type']](**hook_cfg)
            )

    def __call__(self, runner, hook_stage, **kwargs):
        for hook in self.hooks:
            getattr(hook, hook_stage)(runner, **kwargs)


class BaseHook:
    def __init__(self, **kwargs):
        pass

    def pre_iter(self, runner, **kwargs):
        pass

    def post_iter(self, runner, **kwargs):
        pass

    def pre_epoch(self, runner, **kwargs):
        pass

    def post_epoch(self, runner, **kwargs):
        pass


class MemoryUsageHook(BaseHook):
    def __init__(self, device='cuda:0', **kwargs):
        super().__init__(**kwargs)
        self.device = device

    def post_iter(self, runner, **kwargs):
        memory = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
        torch.cuda.empty_cache()
        runner.logger.update(memory=memory)


class TrainTimerHook(BaseHook):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elapsed_time = 0
        self.start_time = None

    def pre_epoch(self, runner, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()

    def post_iter(self, runner, **kwargs):
        self.elapsed_time = (time.time() - self.start_time) / 3600
        total_run_iter = (runner.total_iter * (runner.epoch - 1)) + runner.iter
        time_per_iter = self.elapsed_time / total_run_iter
        estimated_time = time_per_iter * runner.total_iter * runner.total_epochs
        time_remain = estimated_time - self.elapsed_time
        runner.logger.update(time_remain=time_remain)


class CheckPointsHook(BaseHook):
    def __init__(self, max_ckpt=3, epoch_every=None, iter_every=None, **kwargs):
        super().__init__(**kwargs)
        self.max_ckpt = max_ckpt
        self.epoch_every = epoch_every
        self.iter_every = iter_every

    def post_epoch(self, runner, **kwargs):
        if self.epoch_every is not None and runner.epoch % self.epoch_every == 0:
            self.save(runner, f'epoch{runner.epoch}.pth')
        else:
            if runner.epoch > self.max_ckpt:
                os.remove(os.path.join(
                    runner.logger.log_path,
                    f'epoch{runner.epoch - self.max_ckpt}.pth'))
            self.save(runner, f'epoch{runner.epoch}.pth')

    def post_iter(self, runner, **kwargs):
        if self.iter_every is not None and runner.iter % self.iter_every == 0:
            self.save(runner, f'latest.pth')

    def save(self, runner, name):
        save_path = os.path.join(runner.logger.log_path, name)
        print(f'Saving checkpoint to {save_path}.')
        torch.save({
            'epoch': runner.epoch,
            'model': runner.forward_runner.state_dict(),
            'optimizer': runner.optimizer.state_dict(),
            'lr_scheduler': runner.lr_scheduler.state_dict(),
        }, save_path)


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

    def post_iter(self, runner, **kwargs):
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
                    'ctr': values['center'],
                    'conf': values['conf']
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
                    'ctr': values['center'],
                    'conf': values['conf']
                })

        runner.controller.data_manager.scatter(cav_ids, {'detection': preds})


class EvalOPV2VDetectionHook(BaseHook):
    def __init__(self, iou_thr=[0.5, 0.7], range=[""], **kwargs):
        super().__init__(**kwargs)
        self.iou_thr = iou_thr
        self.eval_funcs = import_module('cosense3d.utils.eval_detection_utils')
        # Create the dictionary for evaluation
        self.result_stat = {iou: {'tp': [], 'fp': [], 'gt': 0} for iou in iou_thr}

    def post_iter(self, runner, **kwargs):
        detection = runner.controller.data_manager.gather_ego_data('detection')
        gt_boxes = runner.controller.data_manager.gather_ego_data('global_bboxes_3d')
        for cav_id, preds in detection.items():
            for iou in self.iou_thr:
                self.eval_funcs.caluclate_tp_fp(
                    preds['box'], preds['scr'], gt_boxes[cav_id], self.result_stat, iou
                )

    def post_epoch(self, runner, **kwargs):
        self.eval_funcs.eval_final_results(self.result_stat, self.iou_thr)

