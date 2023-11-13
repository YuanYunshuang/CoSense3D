import os, glob, tqdm
import torch
from cosense3d.utils.eval_detection_utils import *


def eval_detection_opv2v(test_dir, iou_thr=[0.5, 0.7]):
    result_stat = {iou: {'tp': [], 'fp': [], 'gt': 0} for iou in iou_thr}
    filenames = glob.glob(os.path.join(test_dir, '*.pth'))
    for f in tqdm.tqdm(filenames):
        data = torch.load(f)
        preds = data['detection']
        gt_boxes = data['gt_boxes']
        for iou in iou_thr:
            caluclate_tp_fp(
                preds['box'], preds['scr'], gt_boxes, result_stat, iou
            )
    eval_final_results(result_stat, iou_thr)


def eval_detection_cosense3d(test_dir, iou_thr=[0.5, 0.7], mode='bev'):
    result_stat = {iou: {'tp': [], 'gt': 0, 'scr': []} for iou in iou_thr}
    filenames = glob.glob(os.path.join(test_dir, '*.pth'))
    for f in tqdm.tqdm(filenames):
        data = torch.load(f)
        preds = data['detection']
        gt_boxes = data['gt_boxes']
        for iou in iou_thr:
            tp = ops_cal_tp(
                preds['box'].cpu(), gt_boxes.cpu(), IoU_thr=iou, iou_mode=mode
            )
            result_stat[iou]['tp'].append(tp.cpu())
            result_stat[iou]['gt'] += len(gt_boxes)
            result_stat[iou]['scr'].append(preds['scr'].cpu())

    result = {}
    for iou in iou_thr:
        scores = torch.cat(result_stat[iou]['scr'], dim=0)
        tps = torch.cat(result_stat[iou]['tp'], dim=0)
        n_pred = len(scores)
        n_gt = result_stat[iou]['gt']

        ap, mpre, mrec, _ = cal_ap_all_point(scores, tps, n_pred, n_gt)
        iou_str = f"{int(iou * 100)}"
        result[iou] = {f'ap_{iou_str}': ap,
                       f'mpre_{iou_str}': mpre,
                       f'mrec_{iou_str}': mrec,
                       }
        print(f"AP@{iou}: {ap:.3f}")



if __name__=="__main__":
    eval_detection_cosense3d(
        "/media/yuan/luna/cosense3d/default/epoch50/detection_eval",
    )