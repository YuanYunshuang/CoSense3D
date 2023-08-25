import sys
import os
import time

import torch
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d

from cosense3d.utils import pclib
from cosense3d.utils.box_utils import corners_to_boxes_3d, boxes_to_corners_3d


COLOR_PALETTES = {
    'pastels_rock': {
        'DesertSand': [238, 185, 161],
        'DeepChampagne': [241, 213, 170],
        'Champagne': [242, 237, 207],
        'JetStream': [186, 224, 195],
        'LightPeriwinkle':[190, 198, 225],
    },
    'calm_afternoon': {
        'MiddleBlueGreen': [137, 204, 202],
        'Khaki': [245, 222, 145],
        'MacaroniAndCheese': [245, 193, 129],
        'Middle Red': [232, 132, 107],
        'Rose Gold': [189, 93, 115],
        'Rackley': [101, 135, 168],
    },
    'objects': {
        'vehicle': [0, 0, 142],
        'cyclist': [200, 100, 0],
        'pedestrian': [220, 20, 60],
        'truck': [0, 0, 0],
        'motorcycle': [100, 200, 0],
        'bus': [100, 100, 0]
    }
}


def get_palette_colors(palette):
    return np.array(
        list(COLOR_PALETTES[palette].values())
    ) / 255


def visualization(func_list, batch_data):
    for func_str in func_list:
        getattr(sys.modules[__name__], func_str)(batch_data)


def draw_box_plt(boxes_dec, ax, color=None, linewidth_scale=2.0, linestyle='solid'):
    """
    draw boxes in a given plt ax
    :param boxes_dec: (N, 5) or (N, 7) in metric
    :param ax:
    :return: ax with drawn boxes
    """
    if not len(boxes_dec)>0:
        return ax
    boxes_np= boxes_dec
    if isinstance(boxes_np, torch.Tensor):
        boxes_np = boxes_np.cpu().detach().numpy()
    elif isinstance(boxes_np, list):
        boxes_np = np.array(boxes_np)
    if boxes_np.shape[-1]>5:
        boxes_np = boxes_np[:, [0, 1, 3, 4, 6]]
    x = boxes_np[:, 0]
    y = boxes_np[:, 1]
    dx = boxes_np[:, 2]
    dy = boxes_np[:, 3]

    x1 = x - dx / 2
    y1 = y - dy / 2
    x2 = x + dx / 2
    y2 = y + dy / 2
    theta = boxes_np[:, 4:5]
    # bl, fl, fr, br
    corners = np.array([[x1, y1],[x1,y2], [x2,y2], [x2, y1]]).transpose(2, 0, 1)
    new_x = (corners[:, :, 0] - x[:, None]) * np.cos(theta) + (corners[:, :, 1]
              - y[:, None]) * (-np.sin(theta)) + x[:, None]
    new_y = (corners[:, :, 0] - x[:, None]) * np.sin(theta) + (corners[:, :, 1]
              - y[:, None]) * (np.cos(theta)) + y[:, None]
    corners = np.stack([new_x, new_y], axis=2)
    for corner in corners:
        ax.plot(corner[[0,1,2,3,0], 0], corner[[0,1,2,3,0], 1], color=color,
                linewidth=linewidth_scale, linestyle=linestyle)
        # draw direction
        # front = corner[[2, 3]].mean(axis=0)
        # center = corner.mean(axis=0)
        # ax.plot([front[0], center[0]], [front[1], center[1]], color=color,
        #         linewidth=linewidth_scale)
        ax.plot(corner[[2, 3], 0], corner[[2, 3], 1], color=color, linewidth=1.5*linewidth_scale)
    return ax


def draw_points_boxes_plt(pc_range=None, points=None, boxes_pred=None, boxes_gt=None, wandb_name=None,
                          points_c='gray', bbox_gt_c='green', bbox_pred_c='red',
                          bbox_pred_label=None, bbox_gt_label=None,
                          return_ax=False, ax=None, marker_size=2.0, filename=None):
    if pc_range is not None:
        if isinstance(pc_range, int) or isinstance(pc_range, float):
            pc_range = [-pc_range, -pc_range, pc_range, pc_range]
        elif isinstance(pc_range, list) and len(pc_range)==6:
            pc_range = [pc_range[i] for i in [0, 1, 3, 4]]
        else:
            assert isinstance(pc_range, list) and len(pc_range)==4, \
                "pc_range should be a int, float or list of lenth 6 or 4"
    if ax is None:
        ax = plt.figure(figsize=((pc_range[2] - pc_range[0]) / 20,
                                 (pc_range[3] - pc_range[1]) / 20)).add_subplot(1, 1, 1)
        ax.set_aspect('equal', 'box')
    if pc_range is not None:
        ax.set(xlim=(pc_range[0], pc_range[2]),
               ylim=(pc_range[1], pc_range[3]))
    if points is not None:
        ax.plot(points[:, 0], points[:, 1], '.',
                color=points_c, markersize=marker_size)
    if (boxes_pred is not None) and len(boxes_pred) > 0:
        ax = draw_box_plt(boxes_pred, ax, color=bbox_pred_c, linewidth_scale=0.75)
        if bbox_pred_label is not None:
            assert len(boxes_pred) == len(bbox_pred_label)
            for box, label in zip(boxes_pred, bbox_pred_label):
                ax.annotate(label, (box[0], box[1]), textcoords="offset points", xytext=(0, 10), ha='center', color='r')
    if (boxes_gt is not None) and len(boxes_gt) > 0:
        ax = draw_box_plt(boxes_gt, ax, color=bbox_gt_c, linewidth_scale=0.75)
        if bbox_gt_label is not None:
            assert len(boxes_gt) == len(bbox_gt_label)
            for box, label in zip(boxes_gt, bbox_gt_label):
                ax.annotate(label, (box[0], box[1]), textcoords="offset points", xytext=(0, 10), ha='center', color='g')
    plt.xlabel('x')
    plt.ylabel('y')

    if return_ax:
        return ax
    if filename is not None:
        plt.savefig(filename)
        plt.close()


def update_axis_linset(line_set, axis_len=5):
    points = [
        [0, 0, 0],
        [axis_len, 0, 0],
        [0, axis_len, 0],
        [0, 0, axis_len]
    ]
    lines = [[0, 1], [0, 2], [0, 3]]
    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)
    return line_set


def bbx2linset(bbx, color=(0, 1, 0)):
    """
    Convert the bounding box to o3d lineset for visualization.

    Parameters
    ----------
    bbx : np.ndarray
        shape: (n, 7) or (n, 8, 3).
    color : tuple
        The bounding box color.

    Returns
    -------
    line_set : open3d.LineSet
    """
    if len(bbx) > 0 and len(bbx[0]) == 11:
        bbx = bbx[:, 2:]
        bbx_corner = boxes_to_corners_3d(bbx, 'lwh')
    elif len(bbx) > 0 and len(bbx[0]) == 7:
        bbx_tmp = np.zeros((len(bbx), 9))
        bbx_tmp[:, :6] = bbx[:, :6]
        bbx_tmp[:, -1] = bbx[:, -1]
        bbx_corner = boxes_to_corners_3d(bbx_tmp, 'lwh')
    else:
        bbx_corner = bbx
    bbx_corner = np.array(bbx_corner)
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]

    # Use the same color for all lines
    colors = [list(color) for _ in range(len(lines))]
    bbx_linset = []

    for i in range(len(bbx_corner)):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]

        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector(bbx)
        line_set.lines = o3d.utility.Vector2iVector(lines)
        line_set.colors = o3d.utility.Vector3dVector(colors)
        bbx_linset.append(line_set)

    return bbx_linset


def update_lineset_vbo(vbo, bbx, color=None):
    if len(bbx) > 0 and len(bbx[0]) == 9:
        bbx = bbx[:, 2:]
        bbx_corner = boxes_to_corners_3d(bbx, 'lwh')
    else:
        bbx_corner = bbx
    bbx_corner = np.array(bbx_corner)
    # Our lines span from points 0 to 1, 1 to 2, 2 to 3, etc...
    lines = [[0, 1], [1, 2], [2, 3], [0, 3],
             [4, 5], [5, 6], [6, 7], [4, 7],
             [0, 4], [1, 5], [2, 6], [3, 7]]
    lines = np.array(lines)
    if isinstance(color, np.ndarray):
        color = color.squeeze().tolist()
    
    points_all = []
    lines_all = []
    colors_all = []
    for i in range(len(bbx_corner)):
        bbx = bbx_corner[i]
        # o3d use right-hand coordinate
        bbx[:, :1] = - bbx[:, :1]
        points_all.extend(bbx)
        lines_all.extend((lines + 8 * i).tolist())
        # if no color given, use green for all lines
        if color is None:
            box_color = [[0, 1, 0] for _ in range(len(lines))]
        elif isinstance(color[0], float):
            box_color = [color for _ in range(len(lines))]
        else:
            box_color = [color[i] for _ in range(len(lines))]

        colors_all.extend(box_color)
    vbo.points = o3d.utility.Vector3dVector(points_all)
    vbo.lines = o3d.utility.Vector2iVector(lines_all)
    vbo.colors = o3d.utility.Vector3dVector(colors_all)
    return vbo


def o3d_draw_pcds_bbxs(pcds: list,
                       bbxs: list,
                       bbxs_colors: list=None):
    """
    Parameters
    ----------
    pcds: list of np array
    bbxs: list of np array,
        bounding boxes in corner format
    bbxs_colors: list of tuples
    """
    pcds_vis = []
    linsets = []
    for i, bbx in enumerate(bbxs):
        bbx_color = (0, 1, 0)
        if bbxs_colors is not None:
            assert len(bbxs_colors) == len(bbxs)
            bbx_color = bbxs_colors[i]
        linset = bbx2linset(bbx, bbx_color)
        linsets.extend(linset)
    for i, points in enumerate(pcds):
        points[:, 0] *= -1
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        colors = get_palette_colors('calm_afternoon')
        pcd.paint_uniform_color(colors[i])
        pcds_vis.append(pcd)
    o3d.visualization.draw_geometries(pcds_vis + linsets)


def o3d_draw_frame_data(frame_dict, data_path):
    pcds = []
    bbx_global = frame_dict['meta']['bbx_center_global']
    bbx_corners = boxes_to_corners_3d(np.array(bbx_global[:, 2:]))
    linsets = []
    bbx_colors = get_palette_colors('objects')
    for l in np.unique(bbx_global[:, 1]):
        assert l < 3
        linsets.extend(bbx2linset(bbx_corners, bbx_colors[int(l)]))
    for ai, acontent in frame_dict['agents'].items():
        for li, lidar_dict in acontent['lidar0'].items():
            lidar_file = os.path.join(data_path, lidar_dict['filename'])
            points = pclib.load_pcd(lidar_file)[:, :3]
            points = pclib.rotate3d(points, lidar_dict['pose'][3:])
            points = points + np.array(lidar_dict['pose'][:3]).reshape(1, 3)
            # o3d use right hand: left -> right hand
            points[:, 0] *= -1
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            colors = get_palette_colors('calm_afternoon')
            pcd.paint_uniform_color(colors[ai])
            pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds + linsets)


def o3d_draw_agent_data(agent_dict, data_path):
    pcds = []
    bbx_lensets = []
    for li, lidar_dict in agent_dict['lidar0'].items():
        lidar_file = os.path.join(data_path, lidar_dict['filename'])
        points = pclib.load_pcd(lidar_file)[:, :3]
        # o3d use right hand: left -> right hand
        points[:, 0] *= -1
        bbx = np.array(agent_dict['bbx_center'])

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.paint_uniform_color([0.5] * 3)
        linsets = bbx2linset(bbx, (0, 1, 0))
        pcds.append(pcd)
        bbx_lensets.extend(linsets)
    o3d.visualization.draw_geometries(pcds + bbx_lensets)


def o3d_play_sequence(meta_dict, data_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().show_coordinate_frame = True

    vbo_pcd = o3d.geometry.PointCloud()
    vbo_lineset = o3d.geometry.LineSet()
    painter = get_palette_colors('pastels_rock')

    idx = 0
    while True:
        for scenario, scenario_dict in meta_dict.items():
            for frame, frame_dict in scenario_dict.items():
                pcds = []
                colors = []
                for i, (ai, agent_dict) in enumerate(frame_dict['agents'].items()):
                    for li, lidar_dict in agent_dict['lidar0'].items():
                        points = pclib.load_pcd(os.path.join(
                            data_path,
                            lidar_dict['filename'])
                        )[:, :3]
                        points = pclib.rotate3d(points, lidar_dict['pose'][3:])
                        points = points + np.array(lidar_dict['pose'][:3]).reshape(1, 3)
                        pcds.append(points)
                        colors.append(np.ones_like(points) * 
                                      np.array(painter[i]).reshape(1, 3))
                pcds = np.concatenate(pcds, axis=0)
                pcds[:, 0] *= -1
                colors = np.concatenate(colors, axis=0)
                vbo_pcd.points = o3d.utility.Vector3dVector(pcds)
                vbo_pcd.colors = o3d.utility.Vector3dVector(colors)

                # add boxes
                bbxs = frame_dict['meta']['bbx_center_global']
                if len(bbxs) > 0:
                    bbxs = boxes_to_corners_3d(np.array(bbxs)[:, 2:])
                    vbo_lineset = update_lineset_vbo(vbo_lineset, bbxs)
                    if idx == 0:
                        vis.add_geometry(vbo_lineset)
                    else:
                        vis.update_geometry(vbo_lineset)
                # add pcds
                if idx == 0:
                    vis.add_geometry(vbo_pcd)
                else:
                    vis.update_geometry(vbo_pcd)
    
                vis.poll_events()
                vis.update_renderer()
                time.sleep(0.1)
                idx += 1
    
    
def plt_draw_frame_data(frame_dict, data_path):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot()
    for ai, acontent in frame_dict.items():
        for li, lidar_dict in acontent['lidar0'].items():
            lidar_file = os.path.join(data_path, lidar_dict['filename'])
            points = pclib.load_pcd(lidar_file)[:, :3]
            points = pclib.rotate3d(points, lidar_dict['pose'])
            points = points + np.array(lidar_dict['pose'][:3]).reshape(1, 3)
            # points = np.r_[points, [np.ones(points.shape[1])]]
            # points = np.dot(lidar_dict['pose'], points).T[:, :3]
            bbx = np.array(acontent['objects'])
            assert len(bbx.shape) == 2
            bbx = bbx[:, 2:]
            
            ax.plot(points[:, 0], points[:, 1], '.', markersize=.5)
            ax = draw_box_plt(bbx, ax)
    plt.show()
    plt.close()
