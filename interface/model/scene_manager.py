import copy
import logging, json, codecs
import os
from copy import deepcopy
from collections import deque
from glob import glob
import numpy as np
from interface.config import FILE, SCENE, POINTCLOUD, COLORS
from interface.model.utils import *
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as csd


# decorator
def _reset_pcds_and_registers(view_ids):
    def decorate(func):
        def wrapper(*args, **kwargs):
            self = args[0]
            # save labels
            self.save_labels(view_ids)

            # execute core func
            func(*args)

            # load pcd1 and pcd2 of the new frame
            self.load_pcds(view_ids)
            # clear registers
            self.reset_registers(view_ids)
            # load label if exist
            self.load_labels(view_ids)
        return wrapper
    return decorate


class Scene:
    def __init__(self, sensor_cfg=None):
        self.sensor_cfg = sensor_cfg
        self.root_dir = None
        self.batch_size = 20
        self.pcd_range = 75.2
        self.batch_data = {}
        self.scenes = []
        self.meta_dict = {}
        self.edit_box_id = None
        self.id_ptr = 0
        self.uniq_ids = set()
        self.frame_viewer_info = {}
        self.snapshots = deque(maxlen=8) #[{1: {}, 2: {}}] * 5

    @property
    def scene(self):
        """Current activate pcd"""
        return self.scenes[self.scene_idx]

    @property
    def pcd(self):
        """Current activate pcd"""
        return self.batch_data[self.frame_idx]['pcd']

    @property
    def label(self):
        """Current activate label"""
        return self.batch_data[self.frame_idx]['label']

    @property
    def local_label(self):
        """Current activate local label"""
        return self.batch_data[self.frame_idx]['local_label']


    @property
    def label_predecessor(self, center_only=True):
        """Predecessor in the last frame of each box in the current frame."""
        predecessor = {}
        last_frame_idx = max((self.frame_idx - 1), 0)
        if last_frame_idx in self.batch_data:
            label_last_frame = self.batch_data[last_frame_idx]['label']
        else:
            label_last_frame = self.load_label(last_frame_idx)
        for i, label in self.label.items():
            last_label = label_last_frame.get(i, None)
            if last_label is not None and center_only:
                last_label = last_label[1:4]
            predecessor[i] = last_label
        return predecessor

    @property
    def frame(self):
        """Current activate frame"""
        return self.frames[self.frame_idx]

    def gen_id(self):
        while True:
            if self.id_ptr not in self.uniq_ids:
                self.uniq_ids.add(self.id_ptr)
                return self.id_ptr
            self.id_ptr += 1

    def load_meta(self, scene):
        self.meta_dict = {}
        meta_files = glob(os.path.join(self.meta_dir, f'{scene}*.json'))
        if len(meta_files) == 0:
            # if no meta files found, generate new files
            sensor_id = list(self.sensor_cfg['agents'].keys())[0]
            self.frames = sorted([x.split('.')[0] for x in os.listdir(
                os.path.join(self.root_dir, scene, str(sensor_id)))])
            for f in self.frames:
                fdict = csd.fdict_template()
                for ai, ainfo in self.sensor_cfg['agents'].items():
                    for lidar_id, sensor_id in enumerate(ainfo['lidar']):
                        lidar_file = os.path.join(scene, str(sensor_id), f'{f}.bin')
                        csd.update_agent_lidar(fdict, ai,
                                               lidar_id=lidar_id,
                                               lidar_file=lidar_file)
                self.meta_dict[f] = fdict
            for i in range(0, len(self.frames), 500):
                cur_meta = {f: self.meta_dict[f] for f in self.frames[i:i + 500]}
                fn = f"{self.scenes[self.scene_idx]}_{i // 500}.json"
                with open(os.path.join(self.meta_dir, fn), 'w') as fh:
                    json.dump(cur_meta, fh, indent=3)
                logging.info('Saved to ' + fn)
        else:
            for f in meta_files:
                with open(f, 'r') as fh:
                    self.meta_dict.update(json.load(fh))
            # sort order
            self.meta_dict = {k: self.meta_dict[k] for k in self.meta_dict.keys()}
        # update uniq ids from meta
        for fdict in self.meta_dict.values():
            if 'meta' in fdict:
                boxes = fdict['meta']['bbx_center_global']
                if len(boxes) > 0:
                    self.uniq_ids.add(set([int(b[0]) for b in boxes]))

    def load_dataset(self, path):
        self.root_dir = path / 'data'
        self.meta_dir = path / 'meta'

        # list scene folders
        self.scenes = sorted([x for x in os.listdir(path) if 'meta' not in x])
        self.meta_dir.mkdir(exist_ok=True)

        # for s in self.scenes:
        #     if os.path.isdir(os.path.join(path, s)) and \
        #        os.path.exists(os.path.join(path, s, 'lidar0')):
        #         self.scenes.append(s)
        #         label_path = os.path.join(path, s, 'label')
        #         if not os.path.exists(label_path):
        #             os.mkdir(label_path)

    def load_dataset_by_meta(self, meta_folder):
        self.meta_dir = meta_folder
        self.root_dir = meta_folder.parent / 'data'
        # parse scenarios
        meta_files = meta_folder.glob('*.json')
        self.scenes = list(set([x.name.split('_')[0] for x in meta_files]))
        self.scenes.sort()

    def set_scene(self, index):
        self.scene_idx = index - 1
        scene = self.scenes[self.scene_idx]
        self.load_meta(scene)

        # update frames
        self.frames = sorted(list(self.meta_dict.keys()))
        # self.lidarfile_extension = files[0].split('.')[-1]
        # self.frames = sorted([f[:-4] for f in os.listdir(self.pcd_folder) \
        #                if f.split('.')[-1] in ['pcd', 'bin']])
        # update frame view info
        lidar_ids = [f'{a}.{l}' for k, v in self.meta_dict.items()
                     for a, adict in v['agents'].items()
                     for l, ldict in adict['lidar'].items()]
        lidar_ids = set(lidar_ids)
        self.frame_viewer_info['lidar_ids'] = lidar_ids
        self.frame_idx = -1

    def set_frame(self, index):
        self.frame_idx = index - 1
        self.load_batch_data()
        self.frame_viewer_info['frame'] = self.frame

    def load_pcd(self, index):
        pcds = {}
        for id, f in self.pcd_files(index).items():
            pcd = load_pcd(f)
            in_range_mask = np.linalg.norm(pcd[:, :2], axis=-1) < self.pcd_range
            pcds[id] = pcd[in_range_mask]
        return pcds

    def load_label(self, index):
        labels = self.meta_dict[self.frames[index]]['meta']['bbx_center_global']
        label_dict = {}
        for label in labels:
            label_dict[int(label[0])] = label[1:]
        return label_dict

    def load_local_label(self, index):
        agents = self.meta_dict[self.frames[index]]['agents']
        label_dict = {}
        for a, adict in agents.items():
            if 'gt_boxes' in adict:
                label_dict[a] = {}
                for i, label in enumerate(adict['gt_boxes']):
                    box = label[1:]
                    box_id = int(label[0])
                    if box_id < 0:
                        box_id = self.gen_id()
                    if 'box_confidences' in adict:
                        box = box + [adict['box_confidences'][i]]
                    label_dict[a][box_id] = box
        return label_dict

    def load_batch_data(self):
        end_index = min(len(self.frames) - 1,
                        self.frame_idx + self.batch_size)
        oldset = set(self.batch_data.keys())
        newset = set(np.arange(self.frame_idx, end_index))
        to_load = newset - oldset
        to_remove = oldset - newset

        # load new
        for i in to_load:
            self.batch_data[i] = \
                {
                    'pcd': self.load_pcd(i),
                    'label': self.load_label(i),
                    'local_label': self.load_local_label(i)
                }
        # pop old
        for i in to_remove:
            self.batch_data.pop(i)

    def pcd_files(self, index=None):
        if index is None:
            index = self.frame_idx
        assert index >= 0 and index < len(self.frames), "index out of bound."
        files = {}
        for a, adict in self.meta_dict[self.frames[index]]['agents'].items():
            if 'lidar' in adict:
                for l, ldict in adict['lidar'].items():
                    files[f'{a}.{l}'] = os.path.join(self.root_dir, ldict['filename'])

        return files

    def batch_data_for_box(self, box_id):
        batch_view_data = {}
        box_in_activate_frame = \
            self.batch_data[self.frame_idx]['label'][box_id]
        for frame, data in self.batch_data.items():
            if box_id in data['label']:
                box = data['label'][box_id]
                status = 'draw'
            else:
                box = copy.deepcopy(box_in_activate_frame)
                status = 'copy'
            points = []
            for pcd in data['pcd']:
                mask = np.linalg.norm(pcd[:, :2] - box[:2], axis=1) < 5
                points.append(pcd[mask])
            batch_view_data[frame] = {
                'pcd': points,
                'label': box,
                'status': status,
            }
        return batch_view_data

    def update_frame_labels(self, labels=None, frame_idx=None):
        if labels is None:
            labels = self.label
        if frame_idx is None:
            frame_idx = self.frame_idx

        if isinstance(labels, list):
            # update labels from gl graphics
            label_list = []
            label_dict = {}
            # labels from GL view
            for label in labels:
                cur_label = [label.id, label.typeid] + label.to_center().tolist()
                label_list.append(cur_label)
                label_dict[label.id] = [label.id,] + label.to_center().tolist()
            self.batch_data[frame_idx] = label_dict
            self.meta_dict[self.frames[frame_idx]]['meta']['bbx_center_global'] = label_list
        elif isinstance(labels, dict):
            # check if ids for new tracklets should be generated
            updated_labels = {}
            for k, v in labels.items():
                if k < 0:
                    updated_labels[self.gen_id()] = v
                else:
                    updated_labels[k] = v
            # update from object dialog
            self.batch_data[frame_idx]['label'] = updated_labels
            self.meta_dict[self.frames[frame_idx]]['meta']['bbx_center_global'] = \
                self.label_dict_to_list(updated_labels)

    def add_label(self, label):
        assert isinstance(label, dict)
        for k, v in label.items():
            self.batch_data[self.frame_idx]['label'][k] = v
            self.meta_dict[self.frames[self.frame_idx]]['meta']['bbx_center_global'].extend(
                self.label_dict_to_list(label)
            )

    def label_dict_to_list(self, labels):
        label_list = []
        for k, v in labels.items():
            label_list.append([k,] + v)
        return label_list

    def save_labels(self):
        batch_inds = list(self.batch_data.keys())
        file_inds = set()
        file_idx_min = int(self.frames[min(batch_inds)]) // 500
        file_inds.add(file_idx_min)
        file_idx_max = int(self.frames[max(batch_inds)]) // 500
        file_inds.add(file_idx_max)
        for fidx in file_inds:
            cur_meta = {f: self.meta_dict[f] for f in self.frames[fidx:fidx+500]}
            fn = f"{self.scenes[self.scene_idx]}_{fidx%500}.json"
            with open(os.path.join(self.meta_dir, fn), 'w') as fh:
                json.dump(cur_meta, fh, indent=3)
            logging.info('Saved to ' + fn)

    def update_type(self, obj_id, obj_type):
        # self.batch_data[self.frame_idx]['label']
        self.label[obj_id][0] = obj_type
        self.update_frame_labels(self.label)




