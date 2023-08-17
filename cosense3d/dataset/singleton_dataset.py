import glob
import os
import logging
import time
from typing import List

import open3d as o3d
import cv2
import tqdm
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import cosense3d.dataset.pre_processors as PreP
import cosense3d.dataset.post_processors as PostP

from cosense3d.dataset.data_utils import load_meta
from cosense3d.dataset.base_dataset import BaseDataset
from cosense3d.utils.pclib import load_pcd
from cosense3d.utils.misc import load_json, save_json
from cosense3d.dataset.const import CoSenseBenchmarks as csb
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs

obj_ranges = {
    'vehicle.car': 6,  # 0
    'vehicle.van': 8,  # 1
    'vehicle.truck': 10,  # 2
    'vehicle.bus': 15,  # 3
    'vehicle.tram': 20,  # 4
    'vehicle.motorcycle': 2,  # 5
    'vehicle.cyclist': 2,  # 6
    'vehicle.scooter': 1,  # 7
    'vehicle.other': 2,  # 8
    'human.pedestrian': 1,  # 9
    'human.wheelchair': 1,  # 10
    'human.sitting': 1,  # 11
    'unknown': 2,  # 12
}


class SingletonDataset(Dataset):
    """
    __Task__:
    For a singleton object O, assume its boxes of previous L frames are known,
    predict the box of O for the current frame. The initial box center of current
    frame f is either given or copied from the last frame.
    __Loaded Data__:
    1. Points of object O in frame [f-L, f], distance(points, box_center) < r.
    2. Current rough box center.
    3. Current ground truth box.
    """

    def __init__(self, cfgs, mode, reload_from_meta=False):
        self.cfgs = cfgs
        self.mode = mode
        self.history_len = cfgs.get('history_len', 2)
        self.init_dataset(reload_from_meta=reload_from_meta)

        pre_processes = []
        if mode in cfgs['preprocessors']:
            for name in cfgs['preprocessors'][mode]:
                pre_processes.append(getattr(PreP, name)(**cfgs['preprocessors'][name]))
            self.pre_processes = PreP.Compose(pre_processes)
        post_processes = []
        if 'postprocessors' in cfgs and mode in cfgs['postprocessors']:
            for name in cfgs['postprocessors'][mode]:
                post_processes.append(getattr(PostP, name)(**cfgs['postprocessors'][name]))
            self.post_processes = PostP.Compose(post_processes)

    def __len__(self):
        return self.n_samples

    def __getitem__(self, item):
        data = self.load_one_sample(item)
        if self.pre_processes is not None:
            self.pre_processes(data)
        return data

    def init_dataset(self, reload_from_meta=False):
        """Load all necessary meta information"""
        self.samples = []
        self.meta = {}
        if self.cfgs['data_path'] == '':
            return
        datasets_paths = os.listdir(os.path.join(self.cfgs['data_path'], self.mode))
        for dataset, info_dict in self.cfgs['datasets'].items():
            if dataset not in datasets_paths or reload_from_meta:
                print(f'Loading dataset {dataset} from meta...')
                dataset_path = os.path.join(self.cfgs['data_path'], self.mode, dataset)
                os.makedirs(dataset_path, exist_ok=True)
                meta_dict = load_meta(info_dict['meta_path'])
                # TODO: remove following line
                meta_dict = {'measurement4_0': meta_dict['measurement4_0']}
                self.dataset_meta_to_singleton(meta_dict, info_dict['data_path'], dataset)
            else:
                print(f'Loading dataset {dataset}...')
                meta_files = glob.glob(os.path.join(self.cfgs['data_path'], 'meta', dataset, '*.json'))
                self.meta[dataset] = {}
                for filename in meta_files:
                    scenario = os.path.basename(filename)[:-5]
                    self.meta[dataset][scenario] = load_json(filename)

        print('Loading Done.')
        self.get_ptr_keys()

    def dataset_meta_to_singleton(self, meta_dict, data_path, dataset):
        for s, sdict in meta_dict.items():
            print(f'Parsing scenario: {s}')
            s_path = os.path.join(self.cfgs['data_path'], self.mode, dataset, s)
            obj_meta_path = os.path.join(self.cfgs['data_path'], 'meta', dataset)
            os.makedirs(s_path, exist_ok=True)
            os.makedirs(obj_meta_path, exist_ok=True)
            obj_meta = {}
            for f, fdict in tqdm.tqdm(sdict.items()):
                gt_boxes = np.array(fdict['meta']['bbx_center_global'])
                lidar = []
                for a, adict in fdict['agents'].items():
                    # load lidar data
                    for l, ldict in adict['lidar'].items():
                        lidar_file = os.path.join(data_path, ldict['filename'])
                        # TODO: except lumpi dataset, pcds might need to be transformed into the same coords.
                        lidar.append(load_pcd(lidar_file))
                        # TODO?: load camera data.
                lidar = np.concatenate(lidar, axis=0)
                # lidar_torch = torch.from_numpy(lidar).cuda()
                # gt_boxes_torch = torch.from_numpy(gt_boxes).cuda()
                # dists = torch.norm(lidar_torch[:, :2].unsqueeze(0) -
                #                    gt_boxes_torch[:, 2:4].unsqueeze(1), dim=-1)
                dists = np.linalg.norm(lidar[:, :2][None, :, :] -
                                       gt_boxes[:, 2:4][:, None, :], axis=-1)
                # get points for gt_boxes
                for box, dist in zip(gt_boxes, dists):
                    box = np.array(box)
                    box_id = int(box[0])
                    box_type = cs.OBJ_ID2NAME[int(box[1])]

                    points = lidar[(dist < obj_ranges.get(box_type, 2))]
                    if points.shape[0] < 5:
                        continue
                    # save points
                    obj_dir = os.path.join(s_path, f'{box_id}')
                    os.makedirs(obj_dir, exist_ok=True)
                    points_file = os.path.join(obj_dir, f'{f}.bin')
                    points.tofile(points_file)
                    # update obj meta
                    if box_id not in obj_meta:
                        obj_meta[box_id] = {
                            'type': box_type,
                            'frames': [],
                            'boxes': []
                        }
                    obj_meta[box_id]['frames'].append(f)
                    obj_meta[box_id]['boxes'].append(box[2:].tolist())
            save_json(obj_meta, os.path.join(obj_meta_path, f'{s}.json'))

    def load_one_sample(self, item):
        ptr_l1, ptr_l2 = self.get_ptr(item)
        keys = self.ptr_keys[ptr_l1]
        obj_key = keys['objs'][ptr_l2]
        obj = self.meta[keys['dataset']][keys['scenario']][obj_key]
        obj_type = obj['type']
        idx = item - ptr_l2
        frame = int(obj['frames'][idx])
        gt_box = np.array([0, 0] + obj['boxes'][idx])
        offset = min(obj_ranges[obj_type] * 0.5, 1)
        error_xyz = (np.random.random(3) * 2 - 1) * offset
        init_center = gt_box[2:5] + error_xyz

        pcds = []
        boxes = []
        lidar_poses = []
        for di in range(self.history_len + 1):
            cur_frame = f'{frame - di:06d}'
            if cur_frame not in obj['frames']:
                cur_frame = f'{frame:06d}'
                cur_box = gt_box
            else:
                cur_box = np.array([0, 0] + obj['boxes'][obj['frames'].index(cur_frame)])
                # in ground truth, when two subsequent boxes have large distance,
                # there might be errors in the label, i.e, caused by label exchange.
                # we remove this by replacing it with the newest frame.
                if np.linalg.norm(cur_box[2:4] - gt_box[2:4]) > 1:
                    cur_frame = f'{frame:06d}'
                    cur_box = gt_box
            # load points
            points_file = os.path.join(self.cfgs['data_path'],
                                       self.mode,
                                       keys['dataset'],
                                       keys['scenario'],
                                       obj_key,
                                       cur_frame + '.bin')
            pcd = load_pcd(points_file)
            if len(pcd) == 0:
                pcd = np.copy(cur_box[2:6])
                pcd[-1] = 0
                pcd = pcd.reshape(1, 4)
            if pcd[:, -1].max() > 100:
                pcd[:, -1] /= 255
            pcd_ = np.ones((len(pcd), 6))
            pcd_[:, 0] = abs(di)
            pcd_[:, 1:5] = pcd
            # TODO last column is reserved for point_cls of semseg
            pcd_[:, -1] = -1

            pcds.append(pcd_)
            boxes.append(cur_box)
            lidar_poses.append(np.zeros(6))

        # to local coords
        pcds = np.concatenate(pcds, axis=0)
        pcds[:, 1:3] = pcds[:, 1:3] - init_center[:2].reshape(1, 2)
        boxes = np.array(boxes)
        boxes[:, 2:4] = boxes[:, 2:4] - init_center[:2].reshape(1, 2)

        return {
            'dataset': keys['dataset'],
            'scenario': keys['scenario'],
            'object_id': obj_key,
            'object_type': obj_type,
            'frame': frame,
            'lidar_poses': lidar_poses,
            'objects': boxes,
            'pcds': pcds,
            'imgs': None,
            'cam_params': None,
            'bev_maps': None
        }

    def gen_online_sample(self,
                          pcds: List[np.ndarray],
                          boxes: List[np.ndarray],
                          center: np.ndarray):
        """
        Args:
            pcds: history_len + 1 arrays with shape (N, 3+)
                point clouds in order of newest --> oldest
            boxes: history_len arrays with shape (B, 11)
                box of the previous frames in order of newest --> oldest
            center: (3,)
                box center of the current frame. This can be clicked by user in Qt interface
                or copied from the last frame.
        """
        for i in range(len(pcds)):
            pcd_ = np.ones((len(pcds[i]), 6))
            pcd_[:, 0] = i
            pcd_[:, 1:5] = pcds[i]
            # TODO last column is reserved for point_cls of semseg
            pcd_[:, -1] = -1
            pcds[i] = pcd_
        pcds = np.concatenate(pcds, axis=0)
        pcds[:, 1:3] = pcds[:, 1:3] - center[:2].reshape(1, 2)
        boxes = np.array(boxes)
        boxes[:, 2:4] = boxes[:, 2:4] - center[:2].reshape(1, 2)
        data = {
            'dataset': 'online',
            'scenario': 'online',
            'object_id': boxes[1][0],
            'object_type': 'unknown',
            'frame': 'online',
            'lidar_poses': np.zeros((3, 6)),
            'objects': boxes,
            'pcds': pcds,
            'imgs': None,
            'cam_params': None,
            'bev_maps': None
        }
        if self.pre_processes is not None:
            self.pre_processes(data)
        return data

    def get_ptr(self, index):
        offsets = self.ptrs_l1 - index
        ptr_l1 = self.ptrs_l1[offsets <= 0][-1]
        ptrs_l2 = np.array(list(self.ptr_keys[ptr_l1]['objs'].keys()))
        offsets = ptrs_l2 - index
        ptr_l2 = ptrs_l2[offsets <= 0][-1]
        return ptr_l1, ptr_l2

    def get_ptr_keys(self):
        self.ptr_keys = {}
        ptr1 = 0
        ptr2 = 0
        for dataset, ddict in self.meta.items():
            for scenario, sdict in ddict.items():
                self.ptr_keys[ptr1] = {
                    'dataset': dataset,
                    'scenario': scenario,
                    'objs': {}
                }
                for obj, odict in sdict.items():
                    self.ptr_keys[ptr1]['objs'][ptr2] = obj
                    ptr2 += len(odict['frames'])
                ptr1 = ptr2
        self.ptrs_l1 = np.array(list(self.ptr_keys.keys()))
        self.n_samples = ptr2

    @staticmethod
    def collate_batch(batch_list):
        ret = {
            'batch_size': len(batch_list),
            # data
            'pcds': [],
            'features': [],
            'coords': [],
            'imgs': [],
            # meta
            'scenario': [],
            'frame': [],
            'obj_ids': [],
            'objects': [],
            'cam_intrinsics': [],
            'cam_extrinsics': [],
            'lidar_poses': None,
            'map_anchors': None,
            'anchor_offsets': None,
        }
        seq_len = len(batch_list[0]['objects'])
        ret['seq_len'] = seq_len

        for i in range(len(batch_list)):
            for j in range(seq_len):
                # all data in the same frame have the same batch index for early fusion
                pcds = batch_list[i]['pcds']
                if batch_list[i]['pcds'] is not None:
                    mask = pcds[:, 0] == j
                    pcd = pcds[mask]
                    pcd[:, 0] = i * seq_len + j
                    features = batch_list[i]['features'][mask]
                    coords = batch_list[i]['coords'][mask]
                    coords = torch.from_numpy(coords).float()
                    coords = F.pad(coords, (1, 0, 0, 0), mode="constant", value=i * seq_len + j)
                    ret['pcds'].append(torch.from_numpy(pcd).float())
                    ret['features'].append(torch.from_numpy(features).float())
                    ret['coords'].append(coords)

            objects = batch_list[i]['objects']
            objects = torch.from_numpy(objects).float()
            objects = F.pad(objects, (1, 0, 0, 0), mode="constant", value=i)
            ret['objects'].append(objects)

            ret['scenario'].append([x['scenario'] for x in batch_list])
            ret['frame'].append([x['frame'] for x in batch_list])
            ret['obj_ids'].append([x['object_id'] for x in batch_list])

        BaseDataset.cat_data_dict_tensors(ret)

        return ret


if __name__ == '__main__':
    import argparse, pathlib
    from cosense3d.config import load_config
    import matplotlib.pyplot as plt
    from cosense3d.utils.vislib import draw_points_boxes_plt
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./config/config.yaml")
    args = parser.parse_args()
    args.config = str(pathlib.Path(__file__).parents[1] / 'config' / 'singleton_detector.yaml')
    cfgs = load_config(args)

    dataset = SingletonDataset(cfgs['DATASET'], mode='train')

    # seq_len = 2
    # for data in tqdm.tqdm(dataset):
        # print(data['object_type'])
        # if data['object_type'] == 'vehicle.car':
        #     fig = plt.figure(figsize=(9, 3))
        #     axs = fig.subplots(1, 3)
        #     for i in range(-2, 1):
    #             box = data['objects'][i+seq_len][[2, 3, 4, 5, 6, 7, 10]]
    #             draw_points_boxes_plt(
    #                 pc_range=6,
    #                 points=data['pcds'][data['pcds'][:, 0] == abs(i), 1:],
    #                 boxes_gt=[box],
    #                 ax=axs[i + seq_len],
    #             )
    #         plt.savefig('/home/yuan/Downloads/tmp.png')
    #         plt.close()

    # dataloader = torch.utils.data.DataLoader(dataset,
    #                                          batch_size=2,
    #                                          sampler=None, num_workers=4,
    #                                          shuffle=False,
    #                                          collate_fn=dataset.collate_batch)
    # for batch in dataloader:
    #     print(batch.keys())