import copy
import os
import glob

import numpy as np
import tqdm

from cosense3d.utils import pclib
from cosense3d.utils.misc import load_json, save_json
from cosense3d.utils.box_utils import corners_to_boxes_3d, boxes_to_corners_3d
from cosense3d.utils.vislib import o3d_draw_frame_data, o3d_play_sequence,\
                            o3d_draw_pcds_bbxs
from cosense3d.dataset.toolkit.sustech import obj_from_sustech_to_cosense, \
                                    obj_from_cosense_to_sustech
from cosense3d.dataset.toolkit.cosense import OBJ_ID_MAP, load_meta
from cosense3d.ops.utils import points_in_boxes_cpu


def convert(root_dir, meta_out_dir):
    meta_file = os.path.join(root_dir, 'meta_final.json')
    meta = load_json(meta_file)
    static_bbxes_center =\
        obj_from_sustech_to_cosense(os.path.join(
            root_dir, 'label', 'eight.json'
        )) + obj_from_sustech_to_cosense(os.path.join(
            root_dir, 'label', 'triangle.json'
        ))
    centroids = load_json(os.path.join(root_dir, 'merged_centroid.json'))
    static_bbxes_center = np.array(static_bbxes_center)
    meta_dict = {}
    for frame, fdict in tqdm.tqdm(meta.items()):
        scenario = frame.split('_')[0]
        if scenario not in meta_dict:
            meta_dict[scenario] = {}
        frame_bbx = []
        frame_dict = {'agents': {}, 'meta': {}}
        agent_id = {'gih':0, 'ife':1, 'ikg':2}

        # get current point cloud
        points = pclib.load_pcd(os.path.join(
            root_dir, 'merged_sustech', 'lidar0', f"{frame}.bin"
        ))

        for agent, adict in fdict.items():
            lp = adict['pose_icp']
            lidar_pose = np.array([lp[0], lp[1], lp[2],
                                   lp[5], lp[4], lp[3]])
            timestamp = adict['timestamp']

            # get static boxes in the range 150m of current cav's location
            mask = np.linalg.norm(
                static_bbxes_center[:, 2:5] - lidar_pose[:3].reshape(1, 3),
            axis=1) < 100
            cur_static_bbx = static_bbxes_center[mask]

            # dynamic boxes in local coords
            dynamic_bbx = obj_from_sustech_to_cosense(
                os.path.join(root_dir, 'label', agent, f'{frame}.json')
            )
            if len(dynamic_bbx) > 0:
                dynamic_bbx = np.array(dynamic_bbx)
                dynamic_bbx_center = dynamic_bbx[:, 2:]
                # transform boxes to gloabal coords
                dynamic_bbx_corner = boxes_to_corners_3d(dynamic_bbx_center)
                dynamic_bbx_corner = pclib.rotate_points_batch(
                    dynamic_bbx_corner,
                    lidar_pose[3:].reshape(1, 3)
                    .repeat(len(dynamic_bbx_corner), axis=0)
                )
                dynamic_bbx_corner = dynamic_bbx_corner \
                                     + lidar_pose[:3].reshape(1, 1, 3)
                dynamic_bbx_center = corners_to_boxes_3d(dynamic_bbx_corner)
                cur_dynamic_bbx = np.concatenate(
                    [dynamic_bbx[:, :2], dynamic_bbx_center],
                    axis=1
                )

            # merge all boxes
            if len(dynamic_bbx) > 0:
                cur_bbx = np.concatenate([cur_dynamic_bbx,
                                          cur_static_bbx], axis=0)
            else:
                cur_bbx = cur_static_bbx

            frame_bbx.append(cur_bbx)
            frame_dict['agents'][agent_id[agent]] = \
                {
                    'type': 'cav',
                    'pose': None,
                    'time': None,  # timestamp for the current vehicle pose
                    'lidar0': {
                        0: {
                            'pose': lidar_pose.tolist(),
                            'time': timestamp,  # timestamp for the current lidar0 triggering round
                            'filename': os.path.join('lidar0', agent, f'{frame}.pcd')
                        }
                    },
                    'camera': {},  # TODO API for cameras
                }
        frame_bbx = np.concatenate(frame_bbx, axis=0)
        frame_bbx = np.unique(frame_bbx, axis=0)
        frame_bbx_reduced = copy.deepcopy(frame_bbx)
        frame_bbx_reduced[:, 2:5] -= np.array(centroids[frame]).reshape(1, 3)
        point_indices = points_in_boxes_cpu(points[:, :3],
                                            frame_bbx_reduced[:, [2, 3, 4, 5, 6, 7, 10]])
        mask = np.sum(point_indices, axis=1) > 1
        frame_bbx_reduced = frame_bbx_reduced[mask]
        obj_from_cosense_to_sustech(
            frame_bbx_reduced,
            os.path.join(root_dir, 'merged_sustech', 'label', f'{frame}.json')
        )
        # o3d_draw_pcds_bbxs([points], frame_bbx_reduced)
        frame_dict['meta']['bbx_center_global'] = frame_bbx[mask].tolist()
        # o3d_draw_frame_data(frame_dict, root_dir)
        meta_dict[scenario][frame] = frame_dict

    os.makedirs(os.path.join(meta_out_dir, 'lucoop'), exist_ok=True)
    for scenario, scenario_dict in meta_dict.items():
        meta_file = os.path.join(meta_out_dir, 'lucoop', f'{scenario}.json')
        save_json(scenario_dict, meta_file)


def merge_static_labels():
    eight_bbx_files = glob.glob(r"D:\data\mapathon\label\eight\*.json")
    bbxs = []
    for bbx_file in eight_bbx_files:
        bbx = load_json(bbx_file)
        if len(bbx) > 0:
            bbxs.extend(bbx)
    save_json(bbxs, r"D:\data\mapathon\label\eight.json")


def merge_dynamic_labels(meta_dir, data_path):
    meta_dict = load_meta(meta_dir)
    id2cav = {0: 'gih', 1: 'ife', 2: 'ikg'}
    centroids = {}
    for scenario, scenario_dict in meta_dict.items():
        for frame, frame_dict in tqdm.tqdm(scenario_dict.items()):
            pcds = []
            dynamic_labels = []
            for i, (ai, agent_dict) in enumerate(frame_dict['agents'].items()):
                for li, lidar_dict in agent_dict['lidar0'].items():
                    points = pclib.load_pcd(os.path.join(
                        data_path,
                        lidar_dict['filename'])
                    )
                    points[:, :3] = pclib.rotate3d(points[:, :3], lidar_dict['pose'][3:])
                    points[:, :3] = points[:, :3] + np.array(lidar_dict['pose'][:3]).reshape(1, 3)
                    pcds.append(points)

                    labels = obj_from_sustech_to_cosense(os.path.join(
                        data_path, 'label', id2cav[int(ai)], f"{frame}.json"
                    ))
                    bbx_corner = boxes_to_corners_3d(labels[:, 2:])
                    dynamic_labels.extend(labels)
            dynamic_labels = np.array(dynamic_labels)
            pcds = np.concatenate(pcds, axis=0)
            pcd_mean = pcds[:, :3].mean(axis=0, keepdims=True)
            pcds[:, :3] = pcds[:, :3] - pcd_mean
            pcds.tofile(os.path.join(root_dir, 'lidar0', 'merged_sustech',
                                     'lidar0', f'{frame}.bin'), format='%f')
            centroids[frame] = pcd_mean.squeeze().tolist()
    save_json(centroids, os.path.join(root_dir, 'lidar0', 'merged_centroid.json'))


def merge_frame_pcds(meta_dir, data_path):
    meta_dict = load_meta(meta_dir)
    centroids = {}
    for scenario, scenario_dict in meta_dict.items():
        for frame, frame_dict in tqdm.tqdm(scenario_dict.items()):
            pcds = []
            for i, (ai, agent_dict) in enumerate(frame_dict['agents'].items()):
                for li, lidar_dict in agent_dict['lidar0'].items():
                    points = pclib.load_pcd(os.path.join(
                        data_path,
                        lidar_dict['filename'])
                    )
                    points[:, :3] = pclib.rotate3d(points[:, :3], lidar_dict['pose'][3:])
                    points[:, :3] = points[:, :3] + np.array(lidar_dict['pose'][:3]).reshape(1, 3)
                    pcds.append(points)
            pcds = np.concatenate(pcds, axis=0)
            pcd_mean = pcds[:, :3].mean(axis=0, keepdims=True)
            pcds[:, :3] = pcds[:, :3] - pcd_mean
            pcds.tofile(os.path.join(root_dir, 'lidar0', 'merged_sustech',
                                     'lidar0', f'{frame}.bin'), format='%f')
            centroids[frame] = pcd_mean.squeeze().tolist()
    save_json(centroids, os.path.join(root_dir, 'lidar0', 'merged_centroid.json'))


def tmp():
    files = glob.glob("/media/cav/LAVANDER21/mapathon/label/*/*.json")
    all_types = []
    for f in files:
        try:
            data = load_json(f)
            types = [d['obj_type'] for d in data]
            all_types.extend(types)
        except:
            print(f)
    print(np.unique(all_types))



if __name__=="__main__":
    # data = np.fromfile(r"D:\data\mapathon\lidar0\merged_sustech\lidar0\53_0000.bin", dtype=np.float64).reshape(-1, 3)
    # root_dir = r"D:\data\mapathon"
    # meta_out_dir = r".\dataset\metas"
    # convert(root_dir, meta_out_dir)
    # meta_dict = load_meta(os.path.join(meta_out_dir, 'lucoop'))
    # scenarios = sorted(meta_dict, reverse=True)
    # meta_dict = {s: meta_dict[s] for s in scenarios}
    # o3d_play_sequence(meta_dict, root_dir)

    root_dir = "/media/cav/LAVANDER21/mapathon"
    meta_out_dir = "/home/cav/yunshuang/CoSense/dataset/metas"
    convert(root_dir, meta_out_dir)
    # merge_dynamic_labels(os.path.join(meta_out_dir, 'lucoop'), root_dir)