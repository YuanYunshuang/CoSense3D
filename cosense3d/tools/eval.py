import copy
import os, glob, tqdm
import shutil

import torch
from cosense3d.utils.eval_detection_utils import *
from cosense3d.utils.box_utils import corners_to_boxes_3d
from cosense3d.utils.vislib import draw_points_boxes_plt,plt
from cosense3d.utils.pclib import load_pcd

lidar_range_opv2v = [-140.8, -40, -3, 140.8, 40, 1]


def filter_box_ranges(boxes, lidar_range):
    mask = boxes.new_ones((len(boxes),)).bool()
    if boxes.ndim == 3:
        centers = boxes.mean(dim=1)
    else:
        centers = boxes[:, :3]
    for i in range(3):
        mask = mask & (centers[:, i] > lidar_range[1]) & (centers[:, i] < lidar_range[i+3])
    return mask


def eval_detection_opv2v(test_dir, iou_thr=[0.5, 0.7], global_sort_detections=True):
    result_stat = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in iou_thr}
    filenames = sorted(glob.glob(os.path.join(test_dir, '*.pth')))
    for i in tqdm.tqdm(range(len(filenames))):
        if os.path.exists(os.path.join(test_dir, f"{i}.pth")):
            data = torch.load(os.path.join(test_dir, f"{i}.pth"))
        else:
            data = torch.load(filenames[i])

        if 'pred' in data:
            pred_boxes = data['pred']
            pred_scores = data['score']
            gt_boxes = data['gt']
        else:
            pred_boxes = data['detection']['box']
            pred_scores = data['detection']['scr']
            gt_boxes = data['gt_boxes']

        gt_boxes = gt_boxes[filter_box_ranges(gt_boxes, lidar_range_opv2v)]
        mask = filter_box_ranges(pred_boxes, lidar_range_opv2v)
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        for iou in iou_thr:
            caluclate_tp_fp(
                pred_boxes, pred_scores, gt_boxes, result_stat, iou
            )
    result = eval_final_results(result_stat, iou_thr, global_sort_detections=global_sort_detections)


def eval_detection_opv2v_with_opencood_gt(test_dir_opencood, test_dir_cosense3d, iou_thr=[0.5, 0.7], global_sort_detections=True):
    result_stat = {iou: {'tp': [], 'fp': [], 'gt': 0, 'score': []} for iou in iou_thr}
    filenames_opencood = sorted(glob.glob(os.path.join(test_dir_opencood, '*.pth')))
    filenames_cosense3d = sorted(glob.glob(os.path.join(test_dir_cosense3d, '*.pth')))
    for i in tqdm.tqdm(range(len(filenames_opencood))):
        data_opencood = torch.load(os.path.join(test_dir_opencood, f"{i}.pth"))
        data_cosense3d = torch.load(filenames_cosense3d[i])

        gt_boxes = data_opencood['gt']
        pred_boxes = data_cosense3d['detection']['box']
        pred_scores = data_cosense3d['detection']['scr']

        gt_boxes = gt_boxes[filter_box_ranges(gt_boxes, lidar_range_opv2v)]
        mask = filter_box_ranges(pred_boxes, lidar_range_opv2v)
        pred_boxes = pred_boxes[mask]
        pred_scores = pred_scores[mask]
        for iou in iou_thr:
            caluclate_tp_fp(
                pred_boxes, pred_scores, gt_boxes, result_stat, iou
            )
    eval_final_results(result_stat, iou_thr, global_sort_detections=global_sort_detections)


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


def compare_detection(test_dir1, test_dir2, out_dir):
    filenames = sorted(glob.glob(os.path.join(test_dir2, '*.pth')))
    pc_range = lidar_range_opv2v
    for i, f in enumerate(filenames):
        if not i % 10 == 0:
            continue
        data1 = torch.load(os.path.join(test_dir1, f"{i}.pth"))
        data2 = torch.load(f)

        pred_boxes1 = corners_to_boxes_3d(data1['pred'], 7)
        pred_scores1 = data1['score']
        gt_boxes1 = corners_to_boxes_3d(data1['gt'], 7)
        points1 = data1['points'][:, :3].cpu().numpy()

        pred_boxes2 = data2['detection']['box']
        pred_scores2 = data2['detection']['scr']
        gt_boxes2 = data2['gt_boxes']
        centers = data2['detection']['ctr']

        fig = plt.figure(figsize=(20, 14))
        axs = fig.subplots(2, 1)
        draw_points_boxes_plt(
            pc_range=pc_range,
            points=points1,
            boxes_pred=pred_boxes1.cpu().numpy(),
            boxes_gt=gt_boxes1.cpu().numpy(),
            ax=axs[0]
        )
        draw_points_boxes_plt(
            pc_range=pc_range,
            points=points1,
            boxes_pred=pred_boxes2.detach().cpu().numpy(),
            boxes_gt=gt_boxes2.detach().cpu().numpy(),
            ax=axs[1]
        )


        scenario, frame, _, _ = os.path.basename(f).split('.')
        plt.savefig(f"{out_dir}/{scenario}_{frame}.jpg")
        plt.close()
        # data_path = "/koko/OPV2V/test"
        # cavs = os.listdir(f"{data_path}/{scenario}")
        # pcds = []
        # for cav in cavs:
        #     if 'yaml' in cav:
        #         continue
        #     points = load_pcd(f"{data_path}/{scenario}/{cav}/{frame}.pcd")['xyz']


def format_final_result(out_dict, iou_thr):
    fmt_str = ""
    for iou in iou_thr:
        iou_str = f"{int(iou * 100)}"
        fmt_str += f"AP@{iou_str}: {out_dict[f'ap_{iou_str}']:.3f}\n"
    return fmt_str


def eval_cosense_detection_with_pth(result_path, pc_range, iou_thr=[0.5, 0.7], metrics=['OPV2V', 'CoSense3D']):
    iou_thr = iou_thr
    res_dict = {}
    for m in metrics:
        assert m in ['OPV2V', 'CoSense3D']
        res_dict[f'{m.lower()}_result'] = \
            {iou: {'tp': [], 'fp': [], 'gt': 0, 'scr': []} for iou in iou_thr}
    files = glob.glob(os.path.join(result_path, "*.pth"))
    for f in tqdm.tqdm(files[::30]):
        res = torch.load(f)
        preds = res['detection']
        cur_gt_boxes = res['gt_boxes']

        for iou in iou_thr:
            if 'OPV2V' in metrics:
                result_dict = res_dict['opv2v_result']
                caluclate_tp_fp(
                    preds['box'][..., :7], preds['scr'], cur_gt_boxes[..., :7], result_dict, iou
                )
            if 'CoSense3D' in metrics:
                result_dict = res_dict['cosense3d_result']
                tp = ops_cal_tp(
                    preds['box'][..., :7].detach(), cur_gt_boxes[..., :7].detach(), IoU_thr=iou
                )
                result_dict[iou]['tp'].append(tp.cpu())
                result_dict[iou]['gt'] += len(cur_gt_boxes)
                result_dict[iou]['scr'].append(preds['scr'].detach().cpu())

    fmt_str = ("################\n"
               "DETECTION RESULT\n"
               "################\n")
    if 'OPV2V' in metrics:
        result_dict = res_dict['opv2v_result']
        out_dict = eval_final_results(
            result_dict,
            iou_thr,
            global_sort_detections=True
        )
        fmt_str += "OPV2V BEV Global sorted:\n"
        fmt_str += format_final_result(out_dict, iou_thr)
        fmt_str += "----------------\n"

        out_dict = eval_final_results(
            result_dict,
            iou_thr,
            global_sort_detections=False
        )
        fmt_str += "OPV2V BEV Local sorted:\n"
        fmt_str += format_final_result(out_dict, iou_thr)
        fmt_str += "----------------\n"
    if 'CoSense3D' in metrics:
        out_dict = {}
        result_dict = res_dict['cosense3d_result']
        for iou in iou_thr:
            scores = torch.cat(result_dict[iou]['scr'], dim=0)
            tps = torch.cat(result_dict[iou]['tp'], dim=0)
            n_pred = len(scores)
            n_gt = result_dict[iou]['gt']

            ap, mpre, mrec, _ = cal_ap_all_point(scores, tps, n_pred, n_gt)
            iou_str = f"{int(iou * 100)}"
            out_dict.update({f'ap_{iou_str}': ap,
                             f'mpre_{iou_str}': mpre,
                             f'mrec_{iou_str}': mrec})
        fmt_str += "CoSense3D Global sorted:\n"
        fmt_str += format_final_result(out_dict, iou_thr)
        fmt_str += "----------------\n"
    print(fmt_str)
    with open(os.path.join(os.path.dirname(result_path), "test.log"), 'a') as fh:
        fh.writelines(fmt_str)


def tmp(ckpt_path):
    ckpt = torch.load(ckpt_path)
    reg_key = 'reg_branches.0.2'
    for k, v in ckpt['model'].items():
        if reg_key in k:
            ckpt['model'][k] = v[:20]
    torch.save(ckpt, ckpt_path)



if __name__=="__main__":
    # compare_detection(
    #     "/media/yuan/luna/official_proj/OpenCOOD/ckpt/voxelnet_attentive_fusion/voxelnet_attentive_fusion/result",
    #     "/media/yuan/luna/cosense3d/default/epoch50/detection_eval",
    #     "/media/yuan/luna/tmp"
    # )
    # compare_detection(
    #     "/home/projects/OpenCOOD/ckpt/voxelnet_attentive_fusion/voxelnet_attentive_fusion/result",
    #     "/koko/logs/cosense3d/epoch50/detection_eval",
    # )
    # eval_detection_opv2v(
    #     "/media/yuan/luna/cosense3d/voxelnet_all_grad/epoch50/detection_eval",
    #     global_sort_detections=True,
    # )
    # eval_detection_opv2v_with_opencood_gt(
    #     "/media/yuan/luna/official_proj/OpenCOOD/ckpt/voxelnet_attentive_fusion/voxelnet_attentive_fusion/result",
    #     "/media/yuan/luna/cosense3d/score_sampling_11-16-17-04-22/epoch37/detection_eval",
    #     global_sort_detections=False,
    # )
    # eval_cosense_detection_with_pth(
    #     "/koko/train_out/StreamLTS_fcooper_dairv2x_02-21-18-40-44/epoch50/detection_eval",
    #     [-100, -38.4, -3.0, 100, 38.4, 1.0],
    # )
    for i in range(10, 51, 10):
        shutil.copy(f"/media/yuan/luna/streamLTS/streamLTS_opv2v/epoch{i}.pth",
                    f"/media/yuan/luna/streamLTS/streamLTS_opv2v/epoch{i}.bak.pth")
        shutil.copy(f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_reg/epoch{i}.pth",
                    f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_reg/epoch{i}.bak.pth")
        shutil.copy(f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_t/epoch{i}.pth",
                    f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_t/epoch{i}.bak.pth")
        shutil.copy(f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_global_attn/epoch{i}.pth",
                    f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_global_attn/epoch{i}.bak.pth")
        tmp(f"/media/yuan/luna/streamLTS/streamLTS_opv2v/epoch{i}.pth")
        tmp(f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_reg/epoch{i}.pth")
        tmp(f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_t/epoch{i}.pth")
        tmp(f"/media/yuan/luna/streamLTS/streamLTS_opv2v_no_global_attn/epoch{i}.pth")