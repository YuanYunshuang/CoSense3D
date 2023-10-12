import os, random
import logging
import time

import open3d as o3d
import cv2
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from cosense3d.dataset.pipeline import Pipeline
from cosense3d.dataset.cosense_dataset import CosenseDataset

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


class TemporalCosenseDataset(CosenseDataset):
    def __init__(self, cfgs, mode):
        super().__init__(cfgs, mode)
        self.seq_len = cfgs['seq_len']
        self.rand_len = cfgs.get('rand_len', 0)
        self.seq_mode = cfgs.get('seq_mode', False)

    def __getitem__(self, index):
        queue = []
        index_list = list(range(index - self.seq_len - self.rand_len + 1, index))
        random.shuffle(index_list)
        index_list = sorted(index_list[self.rand_len:])
        index_list.append(index)
        prev_scene_token = None
        prev_agents = None
        prev_i = None
        num_cav = None

        for i in index_list:
            i = max(0, i)
            input_dict = self.load_frame_data(i, prev_agents, prev_i)
            prev_i = i

            if not self.seq_mode:  # for sliding window only
                prev_exists = []
                prev_agents = []
                for tk in input_dict['scene_tokens']:
                    prev_agents.append(tk.split('.')[-1])
                    if prev_scene_token is not None and tk in prev_scene_token:
                        prev_exists.append(True)
                    else:
                        prev_exists.append(False)
                input_dict.update(dict(prev_exists=np.array(prev_exists)))
                prev_scene_token = input_dict['scene_tokens']

            queue.append(input_dict)
        queue = {k: [q[k] for q in queue] for k in queue[0].keys()}
        return queue


if __name__=="__main__":
    from cosense3d.utils.misc import load_yaml
    from torch.utils.data import DataLoader
    cfgs = load_yaml("/mars/projects20/CoSense3D/cosense3d/config/petr.yaml")
    cosense_dataset = TemporalCosenseDataset(cfgs['DATASET'], 'train')
    cosense_dataloader = DataLoader(dataset=cosense_dataset, collate_fn=cosense_dataset.collate_batch)
    for data in cosense_dataloader:
        print(data.keys())