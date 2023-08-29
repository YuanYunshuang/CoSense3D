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

from cosense3d.dataset.pipeline import Pipeline
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


class CosenseDataset(Dataset):
    LABEL_COLORS = {}
    VALID_CLS = []

    def __init__(self, cfgs, mode):
        self.cfgs = cfgs
        self.mode = mode

        self.init_dataset()

        self.pipeline = Pipeline(cfgs['pipeline'].get(mode, []))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        sample_info = self.load_sample_info(item)
        return self.pipeline(sample_info)

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

    def load_sample_info(self, item):
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
            - sample_info: dict
        """
        # load meta info
        scenario, frame = self.samples[item]
        sample_info = self.meta_dict[scenario][frame]

        return {
            'scenario': scenario,
            'frame': frame,
            'sample_info': sample_info
        }

    @staticmethod
    def collate_batch(batch_dict):
        return batch_dict