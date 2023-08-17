import argparse
import os, tqdm, time, glob

import numpy as np
import torch
import open3d as o3d

from cosense3d.model import get_model
from cosense3d.dataset import get_dataloader
from cosense3d.utils import misc, metrics
from cosense3d.utils.train_utils import *
from cosense3d.config import load_config
from cosense3d.utils.box_utils import boxes_to_corners_3d
from cosense3d.dataset.const import CoSenseBenchmarks as csbm
from cosense3d.dataset.toolkit.cosense import CoSenseDataConverter as csd


class SingletonTracker:
    def __init__(self, log_dir):
        cfgs = load_config(log_dir)
        self.cfgs = cfgs
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # set paths
        self.log_dir = cfgs['TRAIN']['log_dir']
        # self.logger = open(os.path.join(self.log_dir, 'test', 'tracker.log'), mode='w')

        # load modules
        self.history_len = 2
        self.cfgs['DATASET']['batch_size_val'] = 1
        self.cfgs['DATASET']['history_len'] = self.history_len
        self.cfgs['DATASET']['data_path'] = ''
        # self.cfgs['DATASET']['gt'] = []
        self.test_dataloader = get_dataloader(self.cfgs['DATASET'], mode='test',)
        self.dataset = self.test_dataloader.dataset
        self.model = get_model(cfgs['MODEL']).to(self.device)

        # load checkpoint
        ckpt = torch.load(os.path.join(self.log_dir, 'last.pth'))
        load_model_dict(self.model, ckpt['model_state_dict'])
        self.model.eval()

    def inf(self, pcds, boxes, center):
        data = self.dataset.gen_online_sample(pcds, boxes, center)
        data = self.dataset.collate_batch([data])
        load_tensors_to_gpu(data)
        self.model(data)
        box = data['det_s2'][0]['pred_boxes'].cpu().numpy()
        box[:2] += center[:2]
        # box[2] -= boxes[1][2]
        box = box.tolist()
        res = {boxes[1][0]: [boxes[1][1]] + box[:6] + [0,] * 2 + box[6:]}
        return res
