import copy
import os, glob, tqdm
import shutil

import torch
from matplotlib import colormaps
from cosense3d.utils.eval_detection_utils import *
from cosense3d.utils.box_utils import corners_to_boxes_3d
from cosense3d.utils.vislib import draw_points_boxes_plt,plt
from cosense3d.utils.pclib import load_pcd

lidar_range_opv2v = [-140.8, -40, -3, 140.8, 40, 1]
lidar_range_dairv2x = [-100, -38.4, -3.0, 100, 38.4, 1.0]

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


def compare_detection(test_dir1, test_dir2, out_dir, pc_range):
    filenames = sorted(glob.glob(os.path.join(test_dir2, '*.pth')))
    for i, f in enumerate(filenames):
        if not i % 10 == 0: # or '35_1.018951' not in f:
            continue
        data1 = torch.load(os.path.join(test_dir1, os.path.basename(f)))
        data2 = torch.load(f)

        pred_boxes1 = data1['detection']['box'][:, :7]
        pred_scores1 = data1['detection']['scr']
        gt_boxes1 = data1['gt_boxes'][:, :7]
        points1 = torch.cat([v for v in data1['points'].values()], dim=0).cpu().numpy()
        points1 = points1[points1[:, 3] > 0]
        time_norm = (points1[:, -1] - points1[:, -1].min()) / (points1[:, -1].max() - points1[:, -1].min())
        colors = colormaps['jet'](time_norm)
        colors[:, -1] = 0.1

        pred_boxes2 = data2['detection']['box'][:, :7]
        pred_scores2 = data2['detection']['scr']
        gt_boxes2 = data2['gt_boxes'][:, :7]

        fig = plt.figure(figsize=(24, 8))
        axs = fig.subplots(1, 2)
        axs[0].scatter(points1[:, 0], points1[:, 1], c=colors, s=1, alpha=0.5, edgecolors='none')
        axs[1].scatter(points1[:, 0], points1[:, 1], c=colors, s=1, alpha=0.5, edgecolors='none')
        axs[0].set_title('StreamLTS', fontsize=20)
        axs[1].set_title('StreamLTS without RoI regression',fontsize=20)

        draw_points_boxes_plt(
            pc_range=pc_range,
            boxes_pred=pred_boxes1.cpu().numpy(),
            boxes_gt=gt_boxes1.cpu().numpy(),
            linewidth_scale=1.5,
            ax=axs[0],
        )
        draw_points_boxes_plt(
            pc_range=pc_range,
            boxes_pred=pred_boxes2.detach().cpu().numpy(),
            boxes_gt=gt_boxes2.detach().cpu().numpy(),
            linewidth_scale=1.5,
            ax=axs[1]
        )

        scenario, frame, _, _ = os.path.basename(f).split('.')
        plt.savefig(f"{out_dir}/{scenario}_{frame}.png", bbox_inches='tight')
        plt.tight_layout()
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


def tmp(ckpt_path, ckpt_path_template, out_path):
    ckpt = torch.load(ckpt_path)
    ckpt_template = torch.load(ckpt_path_template)
    ckpt_template['model'] = ckpt['model']
    ckpt_template['optimizer']['param_groups'][0]['params'] = ckpt['optimizer']['param_groups'][0]['params']
    step = ckpt_template['optimizer']['state'][0]['step']
    ckpt_template['optimizer']['state'] = ckpt['optimizer']['state']
    for k, v in ckpt_template['optimizer']['state'].items():
        ckpt_template['optimizer']['state'][k]['step'] = step

    torch.save(ckpt_template, out_path)


def plot_model_efficiency():
    import matplotlib.ticker as ticker
    data = {
        'fcooper_opv2vt': {'0.5': 54.4, '0.7': 17.0, 'mem': 16.132, 'time': 0.7904},
        'fcooper_dairv2xt': {'0.5': 41.8, '0.7': 17.7, 'mem': 11.811, 'time': 0.5920},
        'fpvrcnn_opv2vt': {'0.5': 70.8, '0.7': 41.2, 'mem': 17.678, 'time': 1.6071},
        'fpvrcnn_dairv2xt': {'0.5': 51.8, '0.7': 23.9, 'mem': 11.971, 'time': 0.9908},
        'attnfusion_opv2vt': {'0.5': 78.7, '0.7': 41.4, 'mem': 20.021, 'time': 0.8335},
        'attnfusion_dairv2xt': {'0.5': 62.1, '0.7': 34.0, 'mem': 15.224, 'time': 0.5890},
        'streamlts_opv2vt': {'0.5': 81.2, '0.7': 59.5, 'mem': 12.587, 'time': 1.3012},
        'streamlts_dairv2xt': {'0.5': 61.2, '0.7': 33.4, 'mem': 10.420, 'time': 0.8400},
    }

    models = ['Fcooper', 'FPVRCNN', 'AttnFusion', 'StreamLTS']
    markers = ['^', '*', 'o', 's']
    fig, axs = plt.subplots(1, 4, figsize=(10, 3))
    # opv2vt memory
    for i in range(4):
        axs[0].plot(data[f'{models[i].lower()}_opv2vt']['mem'], data[f'{models[i].lower()}_opv2vt']['0.5'],
                    color='green', marker=markers[i], markersize=12, label=f'{models[i]}@IoU0.5', linestyle='')
        axs[0].plot(data[f'{models[i].lower()}_opv2vt']['mem'], data[f'{models[i].lower()}_opv2vt']['0.7'],
                    color='orange', marker=markers[i], markersize=12, label=f'{models[i]}@IoU0.7', linestyle='')
        axs[0].set_title('OPV2Vt: AP vs Memory', fontsize=11)
        axs[0].set_xlabel('Memory usage peak (GB)')
        axs[0].set_ylabel('AP (%)')
        axs[0].set_xlim(11, 21)
        axs[0].set_ylim(10, 90)
        axs[0].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))

    # dairv2xt memory
    for i in range(4):
        axs[1].plot(data[f'{models[i].lower()}_dairv2xt']['mem'], data[f'{models[i].lower()}_dairv2xt']['0.5'],
                    color='green', marker=markers[i], markersize=12, linestyle='')
        axs[1].plot(data[f'{models[i].lower()}_dairv2xt']['mem'], data[f'{models[i].lower()}_dairv2xt']['0.7'],
                    color='orange', marker=markers[i], markersize=12, linestyle='')
        axs[1].set_title('DairV2Xt: AP vs Memory', fontsize=11)
        axs[1].set_xlabel('Memory usage peak (GB)')
        axs[1].set_xlim(9.5, 16)
        axs[1].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.0f'))
        axs[1].set_ylim(10, 70)

    # opv2vt time
    for i in range(4):
        axs[2].plot(data[f'{models[i].lower()}_opv2vt']['time'], data[f'{models[i].lower()}_opv2vt']['0.5'],
                    color='green', marker=markers[i], markersize=12, linestyle='')
        axs[2].plot(data[f'{models[i].lower()}_opv2vt']['time'], data[f'{models[i].lower()}_opv2vt']['0.7'],
                    color='orange', marker=markers[i], markersize=12, linestyle='')
        axs[2].set_xlim(0.7, 1.7)
        axs[2].set_ylim(10, 90)
        axs[2].xaxis.set_major_formatter(ticker.FormatStrFormatter('%0.1f'))
        axs[2].set_title('OPV2Vt: AP vs Time', fontsize=11)
        axs[2].set_xlabel('Epochal training time (Hour)')

    # dairv2xt time
    for i in range(4):
        axs[3].plot(data[f'{models[i].lower()}_dairv2xt']['time'], data[f'{models[i].lower()}_dairv2xt']['0.5'],
                    color='green', marker=markers[i], markersize=12, linestyle='')
        axs[3].plot(data[f'{models[i].lower()}_dairv2xt']['time'], data[f'{models[i].lower()}_dairv2xt']['0.7'],
                    color='orange', marker=markers[i], markersize=12, linestyle='')
        axs[3].set_xlim(0.5, 1.1)
        axs[3].set_ylim(10, 70)
        axs[3].set_title('DairV2Xt: AP vs Time', fontsize=11)
        axs[3].set_xlabel('Epochal training time (Hour)')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=4)
    plt.subplots_adjust(bottom=0.35)
    # plt.tight_layout()
    plt.savefig("/home/yys/Pictures/streamLTS/LTS_mem_time_compare.pdf")
    plt.close()


if __name__=="__main__":
    # compare_detection(
    #     "/media/yuan/luna/streamLTS/LTS_opv2v/epoch3/detection_eval",
    #     "/media/yuan/luna/streamLTS/LTS_opv2v_no_reg/epoch50_v1/detection_eval",
    #     "/media/yuan/luna/images/opv2v",
    #     [-20, -38.4, -3.0, 80, 38.4, 1.0]
    # )
    plot_model_efficiency()