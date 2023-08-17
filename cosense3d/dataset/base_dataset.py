import os
import logging
import time

import open3d as o3d
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import cosense3d.dataset.pre_processors as PreP
import cosense3d.dataset.post_processors as PostP

from cosense3d.dataset.data_utils import project_points_by_matrix
from cosense3d.utils.pclib import rotate_points_along_z_np as \
    rotate_points_along_z
from cosense3d.utils.pclib import load_pcd, rotate3d, pose2tf
from cosense3d.utils.box_utils import limit_period, boxes_to_corners_3d
from cosense3d.utils.vislib import draw_points_boxes_plt, \
    get_palette_colors, update_lineset_vbo, update_axis_linset
from cosense3d.utils.misc import load_json
from cosense3d.dataset.const import CoSenseBenchmarks as csb
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as cs


class BaseDataset(Dataset):
    LABEL_COLORS = {}
    VALID_CLS = []

    def __init__(self, cfgs, mode):
        self.cfgs = cfgs
        self.mode = mode
        self.visualize = cfgs['visualize']
        self.load_lidar = cfgs.get('load_lidar', True)
        self.load_img = cfgs.get('load_img', False)
        self.load_lidar_time = cfgs.get('load_lidar_time', False)
        if self.visualize:
            vis = o3d.visualization.Visualizer()
            vis.create_window()
            vis.get_render_option().background_color = [1, 1, 1]# [0.05, 0.05, 0.05]
            vis.get_render_option().point_size = 3.0
            vis.get_render_option().show_coordinate_frame = True
            # vis.get_render_option().line_width = 10.0
            self.visualizer = vis
            self.vbo_pcd = o3d.geometry.PointCloud()
            self.vbo_lineset = o3d.geometry.LineSet()
            self.vbo_lineset_axis = o3d.geometry.LineSet()
            self.vbo_lineset_axis = update_axis_linset(self.vbo_lineset_axis, axis_len=10)
            self.visualizer.get_render_option().line_width = 20.0
            self.visualizer.add_geometry(self.vbo_lineset_axis)
            # self.visualizer.get_render_option().line_width = 1.0
            self.painter = get_palette_colors('pastels_rock')
            self.painter_box = get_palette_colors('objects')
            self.vis_idx = 0

        self.init_dataset()

        pre_processes = []
        if mode in cfgs['preprocessors']:
            for name in cfgs['preprocessors'][mode]:
                pre_processes.append(getattr(PreP, name)(**cfgs['preprocessors'][name]))
            self.pre_processes = PreP.Compose(pre_processes)

        if 'postprocessors' in cfgs:
            post_processes = []
            if mode in cfgs['postprocessors']:
                for name in cfgs['postprocessors'][mode]:
                    post_processes.append(getattr(PostP, name)(**cfgs['postprocessors'][name]))
                self.post_processes = PostP.Compose(post_processes)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        data = self.load_one_sample(item)
        if self.pre_processes is not None:
            self.pre_processes(data)
        return data

    def init_dataset(self):
        """Load all necessary meta information"""
        self.load_meta()
        self.parse_samples()

    def parse_samples(self):
        # list all frames, each frame as a sample
        self.samples = []
        for scenario, scontent in self.meta_dict.items():
            self.samples.extend(sorted([[scenario, frame] for frame in scontent.keys()]))
        self.samples = sorted(self.samples)

        print(f"{self.mode} : {len(self.samples)} samples.")

    def load_meta(self):
        self.meta_dict = {}
        meta_dir = self.cfgs['meta_path']
        if meta_dir == '':
            return
        if 'split' in self.cfgs:
            scenarios = self.cfgs['split'][self.mode]
        elif os.path.exists(os.path.join(self.cfgs['meta_path'], f"{self.mode}.txt")):
            with open(os.path.join(self.cfgs['meta_path'], f"{self.mode}.txt"), 'r') as fh:
                scenarios = [l.strip() for l in fh.readlines() if len(l.strip()) > 0]
        else:
            scenarios = [d[:-5] for d in os.listdir(meta_dir) if 'json' in d]
        for scenario in scenarios:
            meta_file = os.path.join(meta_dir, f"{scenario}.json")
            scenario_dict = load_json(meta_file)
            # scenario_dict = {s: scenario_dict[s] for s in list(scenario_dict.keys())[:1]}
            self.meta_dict[scenario] = scenario_dict

    def load_one_sample(self, item):
        """
        Load data of the ```item```'th sample.
        Parameters
        ----------
        item : int
            sample index

        Returns
        -------
        batch_dict: dict
            - scenario: str
            - frame: str
            - pcds: np.ndarray [N, 6],
                columns are (x, y, z, intensity, lidar_id, point_cls).
            - imgs: camera data
            - maps: np.ndarray
                HD maps.
            - metas: dict
                all meta info for the current scenario.
        """
        # load meta info
        scenario, frame = self.samples[item]
        meta_in = self.meta_dict[scenario][frame]
        agents_dict = {k: meta_in['agents'][k] for k in sorted(meta_in['agents'])}

        # load ground truth
        bbx_global = None
        if 'bbx_center_global' in self.cfgs['gt']:
            bbx_global = np.array(meta_in['meta']['bbx_center_global'])
            # select and remap box cls
            bbx_global = self.remap_box_cls(bbx_global)
        bev_maps = None
        # TODO: currently cvt loads with PIL.Image, should be standardized
        if 'bev_maps' in self.cfgs['gt']:
            bev_maps = {}
            for k, v in meta_in['meta']['bev_maps'].items():
                file = os.path.join(self.cfgs[f'data_path_{self.mode}'], v)
                if file.split('.')[-1] == 'png':
                    bev_maps[k] = Image.open(file)
                else:
                    bev_maps[k] = np.load(file)

        # load data
        device_ids = {'lidar': [], 'cam': []}
        pcds = []
        imgs = []
        tf_matrices = []
        cam_params = {'intrinsic': [], 'extrinsic': []}
        for i, (ai, adict) in enumerate(agents_dict.items()):
            # load lidar0 data
            if self.load_lidar:
                for li, ldict in adict['lidar'].items():
                    device_ids['lidar'].append(f"{ai}.{li}")
                    pcd_file = os.path.join(self.cfgs[f'data_path_{self.mode}'], ldict['filename'])
                    lidar_dict = load_pcd(pcd_file)
                    xyz = lidar_dict['xyz']
                    pcd = np.zeros((len(xyz), 6))
                    pcd[:, 0] = i
                    pcd[:, 1:4] = xyz
                    if 'intensity' in lidar_dict:
                        pcd[:, 4] = lidar_dict['intensity'].squeeze()
                    if 'label' in lidar_dict:
                        pcd[:, -1] = lidar_dict['label'].squeeze()
                    else:
                        # TODO last column is reserved for point_cls of semseg
                        pcd[:, -1] = -1
                    if self.load_lidar_time and 'time' in lidar_dict:
                        pcd = np.concatenate([pcd, lidar_dict['time']], axis=1)
                    pcds.append(pcd)
                    tf_matrices.append(pose2tf(ldict['pose']))
            # load cam data
            if self.load_img:
                for ci, cdict in adict['camera'].items():
                    device_ids['cam'].append(f"{ai}.{ci}")
                    cam_params['intrinsic'].append(cdict['intrinsic'])
                    cam_params['extrinsic'].append(cdict['extrinsic'])
                    img_files = [os.path.join(self.cfgs[f'data_path_{self.mode}'], f) for f in cdict['filenames']]
                    # img_seq = np.stack([cv2.imread(f.replace('png', 'jpg')) for f in img_files], axis=0)
                    # only read the mid. file
                    try:
                        img = Image.open(img_files[len(img_files) // 2])
                    except:
                        img = Image.open(img_files[len(img_files) // 2].replace('png', 'jpg'))
                    imgs.append(img)

        return {
            'scenario': scenario,
            'frame': frame,
            'device_ids': device_ids,
            'tf_cav2ego': tf_matrices if self.load_lidar else None,
            'projected': False,
            'objects': bbx_global,
            'pcds': np.concatenate(pcds, axis=0) if self.load_lidar else None,
            'imgs': imgs if self.load_img else None,
            'cam_params': cam_params if self.load_img else None,
            'bev_maps': bev_maps
        }

    def remap_box_cls(self, bbx):
        if len(bbx) == 0:
            return None
        bm = csb.get(self.cfgs['DetectionBenchmark'])
        valid_cls = []
        for _, names in bm.items():
            valid_cls.append([cs.OBJ_NAME2ID[n] for n in names])

        new_cls = -1 * np.ones_like(bbx[:, 1])
        for i, cls in enumerate(valid_cls):
            mat = bbx[:, 1:2] == np.array(cls).reshape(1, -1)
            mask = np.any(mat, axis=-1)
            new_cls[mask] = i
        # mask out class -1: not cared objects
        bbx[:, 1] = new_cls
        bbx = bbx[new_cls >= 0]
        return bbx

    def update_vbos(self, data):
        pcds = data['pcds']
        bbxs = data['objects']
        tfs = data['tf_cav2ego']
        colors = []
        pcds_global = []
        for i in sorted(np.unique(pcds[:, 0])):
            points = pcds[pcds[:, 0] == i, 1:4]
            if not data['projected'] and tfs is not None:
                points = (tfs[int(i)][:3, :3] @ points.T).T
                points = points + tfs[int(i)][:3, 3].reshape(1, 3)
                # points = rotate3d(points, lidar_poses[int(i)][3:])
                # points = points + lidar_poses[int(i)][:3].reshape(1, 3)
            pcds_global.append(points)
            colors.append(np.ones_like(points) *
                          np.array(self.painter[int(i)]).reshape(1, 3))

        pcds_global = np.concatenate(pcds_global, axis=0)
        pcds_global[:, 0] *= -1
        colors = np.concatenate(colors, axis=0)
        self.vbo_pcd.points = o3d.utility.Vector3dVector(pcds_global)
        self.vbo_pcd.colors = o3d.utility.Vector3dVector(colors)

        bbxs_corner = boxes_to_corners_3d(np.array(bbxs[:, 2:]))
        self.vbo_lineset = update_lineset_vbo(self.vbo_lineset, bbxs_corner,
                                              self.painter_box[bbxs[:, 1].astype(int)])

        # from cosense3d.utils.vislib import draw_points_boxes_plt
        # draw_points_boxes_plt(
        #     pc_range=100,
        #     points=pcds_global,
        #     boxes_gt=bbxs[:, [2, 3, 4, 5, 6, 7, 10]],
        #     filename='/home/yuan/Downloads/tmp.png'
        # )
        # pass

    def visualize_seq(self, data):
        self.update_vbos(data)
        if self.vis_idx == 0:
            self.visualizer.add_geometry(self.vbo_pcd)
            self.visualizer.add_geometry(self.vbo_lineset)
        else:
            self.visualizer.update_geometry(self.vbo_pcd)
            self.visualizer.update_geometry(self.vbo_lineset)

        self.visualizer.poll_events()
        self.visualizer.update_renderer()
        time.sleep(0.1)
        self.vis_idx += 1

    def visualize_frame(self, data):
        self.update_vbos(data)
        o3d.visualization.draw_geometries(
            [self.vbo_pcd, self.vbo_lineset, self.vbo_lineset_axis],
        )

    def crop_points_by_features(self, features):
        mask = np.ones_like(features[:, 1])
        for k, v in self.cfgs['crop'].items():
            if k == 'd' or k == 'z':
                continue
            symbols = [s.strip() for s in self.cfgs['voxel']['features'].split(',')]
            assert k in symbols
            coor_idx = ''.join(symbols).find(k)
            mask = np.logical_and(
                features[:, coor_idx] > v[0],
                features[:, coor_idx] < v[1]
            ) * mask
        return mask.astype(bool)

    def crop_points_range(self, data_dict):
        lidar = data_dict['lidar0']
        lidar_id = data_dict['lidar_id']
        for s in self.cfgs['crop'].keys():
            mask = self.get_crop_mask(lidar, s)
            lidar = lidar[mask]
            lidar_id = lidar_id[mask]

        data_dict['lidar0'] = lidar
        data_dict['lidar_id'] = lidar_id

        return data_dict

    def crop_boxes_range(self, data_dict):
        boxes = data_dict['boxes']
        for s in self.cfgs['crop'].keys():
            mask = self.get_crop_mask(boxes[:, :3], s)
            if mask.sum() == 0:
                data_dict['boxes'] = np.zeros((0, 7))
                return data_dict
            else:
                boxes = boxes[mask].reshape(-1, 7)
        data_dict['boxes'] = boxes
        return data_dict

    def get_crop_mask(self, points, symbol):
        """
        Get crop mask for points
        :param points: np.ndarray [N, 3+c], column 1-3 must be x, y, z
        :param symbol: one of
        - 'x'(coordinate),
        - 'y'(coordinate),
        - 'z'(coordinate),
        - 't'(theta in degree),
        - 'c'(cos(t)),
        - 's'(sin(t)).
        :return: mask
        """
        points = getattr(self, f'get_feature_{symbol}')(points).squeeze()
        mask = np.logical_and(
            points > self.cfgs['crop'][symbol][0],
            points < self.cfgs['crop'][symbol][1]
        )
        return mask

    @staticmethod
    def cat_data_dict_tensors(data_dict):
        for k, v in data_dict.items():
            if isinstance(v, torch.Tensor):
                data_dict[k] = v.cuda()
            elif isinstance(v, list) and len(v) > 0 and \
                    isinstance(v[0], torch.Tensor):
                data_dict[k] = torch.cat(v, dim=0)
            else:
                data_dict[k] = v



