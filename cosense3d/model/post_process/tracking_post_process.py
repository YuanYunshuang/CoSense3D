
import logging, os
from typing import Tuple, Any
import importlib

import torch, torch_scatter
import torch.nn.functional as F
import numpy as np
from torch import Tensor
import matplotlib.pyplot as plt

from cosense3d.utils import pclib, box_utils, vislib
from cosense3d.model.post_process import PostProcess
from cosense3d.ops.utils import points_in_boxes_gpu_2d
from cosense3d.model.utils.me_utils import *
from cosense3d.model.utils import indices2metric


class TrackingPostProcess:
    def __init__(self, **kwargs):
        super().__init__()
        m = importlib.import_module("scipy.optimize")
        self.assign_fn = getattr(m, 'linear_sum_assignment')
        self.out = {}

    def __call__(self, batch_dict):
        similarity = batch_dict['center_similarity']
        assignments = []
        for s in similarity:
            cost = (s[0].max() - s[0]).detach().cpu().numpy()
            inds = self.assign_fn(cost)
            assignments.append(inds)
        self.out['assignments'] = assignments

    def set_log_dir(self, log_dir):
        self.log_dir = os.path.join(log_dir, 'MOT')
        os.makedirs(self.log_dir, exist_ok=True)