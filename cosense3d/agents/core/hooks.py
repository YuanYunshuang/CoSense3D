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

    def set_logger(self, logger):
        for hook in self.hooks:
            hook.set_logger(logger)


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

    def set_logger(self, logger):
        self.logger = logger


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
        self.mean_time_per_itr = None
        self.observations = 0

    def pre_epoch(self, runner, **kwargs):
        if self.start_time is None:
            self.start_time = time.time()
            self.last_time = time.time()

    def post_iter(self, runner, **kwargs):
        cur_time = time.time()
        self.elapsed_time = (cur_time - self.start_time) / 3600
        # total_run_iter = (runner.total_iter * (runner.epoch - runner.start_epoch)) + runner.iter
        # time_per_iter = self.elapsed_time / total_run_iter
        time_per_iter = (cur_time - self.last_time) / 3600
        m = self.observations
        self.mean_time_per_itr = m / (m + 1) * self.mean_time_per_itr + 1 / (m + 1) * time_per_iter
        iter_remain = runner.total_iter * (runner.total_epoch - runner.epoch + 1) - runner.iter
        time_remain = self.mean_time_per_itr * iter_remain
        runner.logger.update(t_remain=time_remain, t_used=self.elapsed_time)
        self.observations += 1


class CheckPointsHook(BaseHook):
    def __init__(self, max_ckpt=3, epoch_every=None, iter_every=None, **kwargs):
        super().__init__(**kwargs)
        self.max_ckpt = max_ckpt
        self.epoch_every = epoch_every
        self.iter_every = iter_every

    def post_epoch(self, runner, **kwargs):
        if runner.gpu_id != 0:
            return
        self.save(runner, f'epoch{runner.epoch}.pth')
        if runner.epoch > self.max_ckpt:
            if (self.epoch_every is None or not
            (runner.epoch - self.max_ckpt) % self.epoch_every == 0):
                filename = os.path.join(
                    runner.logger.logdir,
                    f'epoch{runner.epoch - self.max_ckpt}.pth')
                if os.path.exists(filename):
                    os.remove(filename)

    def post_iter(self, runner, **kwargs):
        if runner.gpu_id != 0:
            return
        if self.iter_every is not None and runner.iter % self.iter_every == 0:
            self.save(runner, f'latest.pth')

    def save(self, runner, name):
        save_path = os.path.join(runner.logger.logdir, name)
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
    def __init__(self, nms_thr, pre_max_size,
                 det_key='detection',
                 **kwargs):
        super().__init__(**kwargs)
        self.nms_thr = nms_thr
        self.pre_max_size = pre_max_size
        self.nms = import_module('cosense3d.ops.iou3d_nms_utils').nms_gpu
        self.det_key = det_key
        self.defual_pred_keys = ['box', 'scr', 'lbl', 'idx']
    def post_iter(self, runner, **kwargs):
        detection_out = runner.controller.data_manager.gather_ego_data(self.det_key)
        preds = []
        cav_ids = []
        for cav_id, values in detection_out.items():
            cav_ids.append(cav_id)

            boxes =   values['preds']['box']
            scores =  values['preds']['scr']
            labels =  values['preds']['lbl']
            indices = values['preds']['idx']  # map index for retrieving features

            out = {}
            if 'center' in values:
                out['ctr'] = values['center']
            if 'conf' in values:
                out['conf'] = values['conf']

            if len(values['preds']['box']) == 0:
                out.update({
                    'box': torch.zeros((0, 7), device=boxes.device),
                    'scr': torch.zeros((0,), device=scores.device),
                    'lbl': torch.zeros((0,), device=labels.device),
                    'idx': torch.zeros(indices.shape[0] if isinstance(indices, torch.Tensor) else (0,),
                                       device=indices.device),
                })
            else:
                keep = self.nms(
                    boxes[..., :7],
                    scores,
                    thresh=self.nms_thr,
                    pre_maxsize=self.pre_max_size
                )
                out.update({
                    'box': boxes[keep],
                    'scr': scores[keep],
                    'lbl': labels[keep],
                    'idx': indices[keep],
                })
            preds.append(out)

            # from cosense3d.utils.vislib import draw_points_boxes_plt
            # points = out['ctr'].detach().cpu().numpy()
            # boxes = out['box'].detach().cpu().numpy()
            # draw_points_boxes_plt(
            #     pc_range=[-140.8, -38.4, -3.0, 140.8, 38.4, 1.0],
            #     boxes_pred=boxes,
            #     points=points,
            #     filename="/home/yuan/Pictures/tmp.png"
            # )

        runner.controller.data_manager.scatter(cav_ids, {self.det_key: preds})


class EvalOPV2VDetectionHook(BaseHook):
    def __init__(self, iou_thr=[0.5, 0.7], range=[""], save_result=False,
                 det_key='detection', gt_key='global_bboxes_3d', **kwargs):
        super().__init__(**kwargs)
        self.iou_thr = iou_thr
        self.save_result = save_result
        self.det_key = det_key
        self.gt_key = gt_key
        self.eval_funcs = import_module('cosense3d.utils.eval_detection_utils')
        # Create the dictionary for evaluation
        self.result_stat = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in iou_thr}

    def set_logger(self, logger):
        super().set_logger(logger)
        logdir = os.path.join(logger.logdir, 'detection_eval')
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir

    def post_iter(self, runner, **kwargs):
        detection = runner.controller.data_manager.gather_ego_data(self.det_key)
        gt_boxes = runner.controller.data_manager.gather_ego_data(self.gt_key)
        points = runner.controller.data_manager.gather_batch(0, 'points')
        if self.save_result:
            ego_key = list(detection.keys())[0]
            senario = runner.controller.data_manager.gather_ego_data('scenario')[ego_key]
            frame = runner.controller.data_manager.gather_ego_data('frame')[ego_key]
            filename = f"{senario}.{frame}.{ego_key.split('.')[1]}.pth"
            result = {'detection': detection[ego_key],
                      'gt_boxes': gt_boxes[ego_key],
                      'points': points}
            torch.save(result, os.path.join(self.logdir, filename))
        for cav_id, preds in detection.items():
            for iou in self.iou_thr:
                self.eval_funcs.caluclate_tp_fp(
                    preds['box'], preds['scr'], gt_boxes[cav_id], self.result_stat, iou
                )

    def post_epoch(self, runner, **kwargs):
        result = self.eval_funcs.eval_final_results(self.result_stat, self.iou_thr)
        msg = "##### DETECTION OPV2V AP"


class EvalDetectionHook(BaseHook):
    def __init__(self, pc_range, iou_thr=[0.5, 0.7], metrics=['CoSense3D'], save_result=False,
                 det_key='detection', gt_key='global_bboxes_3d', **kwargs):
        super().__init__(**kwargs)
        self.iou_thr = iou_thr
        self.pc_range = pc_range
        self.save_result = save_result
        self.det_key = det_key
        self.gt_key = gt_key
        for m in metrics:
            assert m in ['OPV2V', 'CoSense3D']
            setattr(self, f'{m.lower()}_result',
                    {iou: {'tp': [], 'fp': [], 'gt': 0, 'scr': []} for iou in iou_thr})
        self.metrics = metrics
        self.eval_funcs = import_module('cosense3d.utils.eval_detection_utils')

    def set_logger(self, logger):
        super().set_logger(logger)
        logdir = os.path.join(logger.logdir, 'detection_eval')
        os.makedirs(logdir, exist_ok=True)
        self.logdir = logdir

    def post_iter(self, runner, **kwargs):
        detection = runner.controller.data_manager.gather_ego_data(self.det_key)
        gt_boxes = runner.controller.data_manager.gather_ego_data(self.gt_key)

        for i, (cav_id, preds) in enumerate(detection.items()):
            if 'preds' in preds:
                preds = preds['preds']
            preds['box'], preds['scr'], preds['lbl'], preds['idx'], preds['time'] = \
            self.filter_box_ranges(preds['box'], preds['scr'], preds['lbl'], preds['idx'], preds.get('time', None))
            cur_gt_boxes = self.filter_box_ranges(gt_boxes[cav_id])[0]
            cur_points = runner.controller.data_manager.gather_batch(i, 'points')

            if self.save_result:
                ego_key = cav_id
                senario = runner.controller.data_manager.gather_ego_data('scenario')[ego_key]
                frame = runner.controller.data_manager.gather_ego_data('frame')[ego_key]
                filename = f"{senario}.{frame}.{ego_key.split('.')[1]}.pth"
                result = {'detection': preds,
                          'gt_boxes': cur_gt_boxes,
                          'points': cur_points}
                torch.save(result, os.path.join(self.logdir, filename))

            for iou in self.iou_thr:
                if 'OPV2V' in self.metrics:
                    result_dict = getattr(self, f'opv2v_result')
                    self.eval_funcs.caluclate_tp_fp(
                        preds['box'][..., :7], preds['scr'], cur_gt_boxes[..., :7], result_dict, iou
                    )
                if 'CoSense3D' in self.metrics:
                    result_dict = getattr(self, f'cosense3d_result')
                    tp = self.eval_funcs.ops_cal_tp(
                        preds['box'][..., :7].detach(), cur_gt_boxes[..., :7].detach(), IoU_thr=iou
                    )
                    result_dict[iou]['tp'].append(tp.cpu())
                    result_dict[iou]['gt'] += len(cur_gt_boxes)
                    result_dict[iou]['scr'].append(preds['scr'].detach().cpu())

    def filter_box_ranges(self, boxes, scores=None, labels=None, indices=None, times=None):
        mask = boxes.new_ones((len(boxes),)).bool()
        if boxes.ndim == 3:
            centers = boxes.mean(dim=1)
        else:
            centers = boxes[:, :3]
        for i in range(3):
            mask = mask & (centers[:, i] > self.pc_range[i]) & (centers[:, i] < self.pc_range[i + 3])
        boxes = boxes[mask]
        if scores is not None:
            scores = scores[mask]
        if labels is not None:
            labels = labels[mask]
        if indices is not None:
            try:
                indices = indices[mask]
            except:
                print("Number of boxes doesn't match the number of indices")
        if times is not None:
            times = times[mask]
        return boxes, scores, labels, indices, times

    def post_epoch(self, runner, **kwargs):
        fmt_str = ("################\n"
                   "DETECTION RESULT\n"
                   "################\n")
        if 'OPV2V' in self.metrics:
            result_dict = getattr(self, f'opv2v_result')
            out_dict = self.eval_funcs.eval_final_results(
                result_dict,
                self.iou_thr,
                global_sort_detections=True
            )
            fmt_str += "OPV2V BEV Global sorted:\n"
            fmt_str += self.format_final_result(out_dict)
            fmt_str += "----------------\n"

            out_dict = self.eval_funcs.eval_final_results(
                result_dict,
                self.iou_thr,
                global_sort_detections=False
            )
            fmt_str += "OPV2V BEV Local sorted:\n"
            fmt_str += self.format_final_result(out_dict)
            fmt_str += "----------------\n"
        if 'CoSense3D' in self.metrics:
            out_dict = self.eval_cosense3d_final()
            fmt_str += "CoSense3D Global sorted:\n"
            fmt_str += self.format_final_result(out_dict)
            fmt_str += "----------------\n"
        print(fmt_str)
        self.logger.log(fmt_str)

    def format_final_result(self, out_dict):
        fmt_str = ""
        for iou in self.iou_thr:
            iou_str = f"{int(iou * 100)}"
            fmt_str += f"AP@{iou_str}: {out_dict[f'ap_{iou_str}']:.3f}\n"
            # fmt_str += f"Precision@{iou_str}: {out_dict[f'mpre_{iou_str}']:.3f}\n"
            # fmt_str += f"Recall@{iou_str}: {out_dict[f'mrec_{iou_str}']:.3f}\n"
        return fmt_str

    def eval_cosense3d_final(self):
        out_dict = {}
        result_dict = getattr(self, f'cosense3d_result')
        for iou in self.iou_thr:
            scores = torch.cat(result_dict[iou]['scr'], dim=0)
            tps = torch.cat(result_dict[iou]['tp'], dim=0)
            n_pred = len(scores)
            n_gt = result_dict[iou]['gt']

            ap, mpre, mrec, _ = self.eval_funcs.cal_ap_all_point(scores, tps, n_pred, n_gt)
            iou_str = f"{int(iou * 100)}"
            out_dict.update({f'ap_{iou_str}': ap,
                             f'mpre_{iou_str}': mpre,
                             f'mrec_{iou_str}': mrec})
        return out_dict


