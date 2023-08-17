import matplotlib.pyplot as plt
import numpy as np
import torch

from cosense3d.utils import vislib
from cosense3d.model.utils.edl_utils import logit_to_edl


def parse_gt_boxes(batch_dict):
    gt_boxes = batch_dict['objects']
    gt_boxes = gt_boxes[gt_boxes[:, 0] == 0][:, [3, 4, 5, 6, 7, 8, 11, 2]].cpu().numpy()
    return gt_boxes


def parse_pcd(batch_dict):
    points = batch_dict['pcds'][batch_dict['pcds'][:, 0] == 0]
    points = points.cpu().numpy()
    return points[:, 1:]


def vis_heatmap(batch_dict, target_dicts, pred_dicts, head_idx, voxel_size, stride, det_r):
    fig = plt.figure(figsize=(12, 4))
    axs = fig.subplots(1, 3)
    points = batch_dict['pcds'][batch_dict['pcds'][:, 0] == 0]
    points = points.cpu().numpy()
    hm_preds = pred_dicts[head_idx]['hm'][0].sigmoid()
    if hm_preds.shape[0] > 0:
        hm_preds = hm_preds.max(dim=0)[0]
    hm_preds = hm_preds.detach().cpu().numpy()
    hm_gt = target_dicts['heatmaps'][head_idx][0].sum(dim=0).cpu().numpy()
    gt_boxes = batch_dict['objects']
    gt_boxes = gt_boxes[gt_boxes[:, 0] == 0][:, [3, 4, 5, 6, 7, 8, 11, 2]].cpu().numpy()
    
    img = np.zeros(hm_preds.shape)
    pixel_sz = voxel_size[0] * stride
    x = np.clip((points[:, 1] + det_r) / pixel_sz, a_min=0, a_max=img.shape[0] - 1)
    y = np.clip((points[:, 2] + det_r) / pixel_sz, a_min=0, a_max=img.shape[1] - 1)
    img[x.astype(int), y.astype(int)] = 0.5
    cx = np.clip((gt_boxes[:, 0] + det_r) / pixel_sz, a_min=0, a_max=img.shape[0] - 1)
    cy = np.clip((gt_boxes[:, 1] + det_r) / pixel_sz, a_min=0, a_max=img.shape[1] - 1)
    img[x.astype(int), y.astype(int)] = 0.25
    img[cx.astype(int), cy.astype(int)] = 1
    
    axs[0].imshow(img.T, vmin=0, vmax=1)
    axs[1].imshow(hm_preds, vmin=0, vmax=1)
    axs[2].imshow(hm_gt, vmin=0, vmax=1)
    plt.savefig('/media/hdd/yuan/Downloads/tmp.png')
    plt.close()


def vis_centernet(pred, tgt, batch_dict, head_idx, voxel_size, stride, det_r):
    fig = plt.figure(figsize=(12, 4))
    axs = fig.subplots(1, 3)
    points = parse_pcd(batch_dict)
    # hm_preds = pred['cls'][head_idx][0][1:].sigmoid()
    # if hm_preds.shape[0] > 0:
    #     hm_preds = hm_preds.max(dim=0)[0]
    hm_preds = pred.sigmoid()[0, :, :, 1]
    hm_preds = hm_preds.detach().cpu().numpy()
    # cnts = [t.shape[1] for t in pred['cls']]
    # hm_gt = tgt['center'][0][sum(cnts[:head_idx]):sum(cnts[:head_idx + 1])].sum(dim=0).cpu().numpy()
    hm_gt = tgt[0, :, :, 1:].max(dim=-1).values.cpu().numpy()
    gt_boxes = parse_gt_boxes(batch_dict)

    img = np.zeros(hm_preds.shape)
    pixel_sz = voxel_size[0] * stride
    x = np.clip((points[:, 0] + det_r) / pixel_sz, a_min=0, a_max=img.shape[0] - 1)
    y = np.clip((points[:, 1] + det_r) / pixel_sz, a_min=0, a_max=img.shape[1] - 1)
    img[x.astype(int), y.astype(int)] = 0.5
    cx = np.clip((gt_boxes[:, 0] + det_r) / pixel_sz, a_min=0, a_max=img.shape[0] - 1)
    cy = np.clip((gt_boxes[:, 1] + det_r) / pixel_sz, a_min=0, a_max=img.shape[1] - 1)
    img[x.astype(int), y.astype(int)] = 0.25
    img[cx.astype(int), cy.astype(int)] = 1

    axs[0].imshow(img, vmin=0, vmax=1)
    axs[1].imshow(hm_preds, vmin=0, vmax=1)
    axs[2].imshow(hm_gt, vmin=0, vmax=1)
    plt.savefig('/media/hdd/yuan/Downloads/centernet.png')
    plt.close()


def vis_centernet_sparse(pred, tgt, batch_dict, head_idx, voxel_size, stride, det_r):
    centers = pred['centers'][pred['centers'][:, 0] == 0, 1:]


def vis_detection(preds, pcds, pc_range, gt_boxes):
    boxes = preds['box'].detach().cpu().numpy()
    pcds = pcds.cpu().numpy()
    gt_boxes = gt_boxes.detach().cpu().numpy()
    vislib.draw_points_boxes_plt(
        pc_range=pc_range,
        boxes_pred=boxes,
        boxes_gt=gt_boxes,
        points=pcds,
        filename='/home/yuan/Downloads/tmp.png'
    )


def vis_cvt_pred(pred, tgt, batch_dict, label_indices=None):
    bev = tgt['bev']
    if label_indices is not None:
        bev = [bev[:, idx].max(1, keepdim=True).values for idx in label_indices]
        bev = torch.cat(bev, 1)
    if 'visibility' in tgt:
        mask = tgt['visibility'] < 2
        bev[mask[:, None]] = 0.25
    bev_pred = pred['bev'][0][0].sigmoid().detach().cpu().numpy()
    bev_tgt = bev[0][0].detach().cpu().numpy()

    center_pred = pred['center'][0][0].sigmoid().detach().cpu().numpy()
    center_tgt = tgt['center'][0][0].detach().cpu().numpy()

    fig = plt.figure(figsize=(8, 8))
    axs = fig.subplots(2, 2)
    axs[0, 0].imshow(bev_pred, vmin=0, vmax=1)
    axs[0, 1].imshow(bev_tgt, vmin=0, vmax=1)
    axs[1, 0].imshow(center_pred, vmin=0, vmax=1)
    axs[1, 1].imshow(center_tgt, vmin=0, vmax=1)

    plt.savefig('/media/hdd/yuan/Downloads/tmp.png')
    plt.close()


def draw_boxes(batch_dict, r):
    pc_range = [-r, -r, r, r]
    gt_boxes = parse_gt_boxes(batch_dict)
    pcd = parse_pcd(batch_dict)
    valid_boxes = [k for k in ['roi', 'det_s1' 'det_s2'] if k in batch_dict]
    n_plot = len(valid_boxes)
    pred_colors = ['blue', 'magenta', 'purple', 'orange', 'red']

    fig = plt.figure(figsize=(n_plot * 6, 6))
    axs = fig.subplots(1, n_plot)

    for i, k in enumerate(valid_boxes):
        if n_plot == 1:
            ax = axs
        else:
            ax = axs[i]
        pred_boxes = batch_dict[k]['box']
        ax = vislib.draw_points_boxes_plt(
            pc_range=pc_range,
            boxes_gt=gt_boxes,
            points=pcd,
            ax=ax,
            return_ax=True
        )
        for j, boxes in enumerate(pred_boxes):
            boxes = boxes[boxes[:, 0]==0][:, 1:].detach().cpu().numpy()
            ax = vislib.draw_points_boxes_plt(
                pc_range=pc_range,
                boxes_pred=boxes,
                bbox_pred_c=pred_colors[j],
                ax=ax,
                return_ax=True
            )

    plt.savefig('/media/hdd/yuan/Downloads/draw_boxes.png')
    plt.close()


def vis_matching_boxes(batch_dict):
    gt_boxes = batch_dict['objects']
    pcds = batch_dict['pcds']
    match = batch_dict['match_matrix'][0][0].cpu().numpy()
    match = np.where(match)

    pcda = pcds[pcds[:, 0] == 0, 1:].cpu().numpy()
    pcdb = pcds[pcds[:, 0] == 1, 1:].cpu().numpy()
    pcdb[:, 0] += 110

    bidx = [1, 2, 3, 4, 5, 6, 7, 8, 11]
    bbxa_gt = gt_boxes[gt_boxes[:, 0] == 0][:, bidx].cpu().numpy()
    bbxb_gt = gt_boxes[gt_boxes[:, 0] == 1][:, bidx].cpu().numpy()
    bbxb_gt[:, 2] += 110

    bbxa_pr = batch_dict['det_s1'][0]['box'].detach().cpu().numpy()
    bbxb_pr = batch_dict['det_s1'][1]['box'].detach().cpu().numpy()
    bbxb_pr[:, 0] += 110

    pc_range = [-55, -55, 165, 55]

    ax = vislib.draw_points_boxes_plt(
        pc_range,
        pcda,
        bbxa_pr,
        bbxa_gt[:, 2:],
        return_ax=True
    )
    ax = vislib.draw_points_boxes_plt(
        pc_range,
        pcdb,
        bbxb_pr,
        bbxb_gt[:, 2:],
        ax=ax,
        return_ax=True
    )

    pointa = bbxa_pr[match[0]][:, :2]
    pointb = bbxb_pr[match[1]][:, :2]
    for pa, pb in zip(pointa, pointb):
        ax.plot([pa[0], pb[0]], [pa[1], pb[1]])

    plt.savefig('/media/hdd/yuan/Downloads/matching.png')


def vis_matching_matrix(batch_dict):
    similarity = batch_dict['center_similarity'][0][0].sigmoid().detach().cpu().numpy()
    assignments = batch_dict['assignments'][0]
    matching_gt = batch_dict['match_matrix'][0][0].cpu().numpy()
    matching_preds = np.zeros_like(similarity)
    matching_preds[assignments[0], assignments[1]] = 1

    fig = plt.figure(figsize=(3 * 6, 6))
    axs = fig.subplots(1, 3)

    axs[0].matshow(similarity)
    axs[0].title.set_text('Pred')
    axs[1].matshow(matching_preds)
    axs[1].title.set_text('Pred Match')
    axs[2].matshow(matching_gt)
    axs[2].title.set_text('GT Match')

    plt.savefig('/media/hdd/yuan/Downloads/matching_matrix.png')
    plt.close()


def vis_singleton_track(batch_dict):
    seq_len = batch_dict['seq_len']
    fig = plt.figure(figsize=(3*seq_len, 3))
    axs = fig.subplots(1, seq_len)
    for i in range(seq_len):
        pcd = batch_dict['pcds']
        pcd = pcd[pcd[:, 0]==i][:, 1:4].cpu().numpy()
        box_gt = batch_dict['objects']
        box_gt = box_gt[box_gt[:, 0]==0][:, [3, 4, 5, 6, 7, 8, 11]]
        box_gt = box_gt[i].cpu().numpy()[None, :]
        vislib.draw_points_boxes_plt(
            pc_range=5,
            points=pcd,
            boxes_gt=box_gt,
            ax=axs[i]
        )
        if i == 0:
            box_pred = batch_dict['det_s2'][0]['pred_boxes'].cpu().numpy()[None, :]
            vislib.draw_points_boxes_plt(
                boxes_pred=box_pred,
                ax=axs[i]
            )
    plt.savefig("/home/yuan/Downloads/tmp.png")
    plt.close()
    pass


if __name__=="__main__":
    vis_matching_boxes(
        torch.load("/media/hdd/yuan/Downloads/matching_matrix.pth")
    )







