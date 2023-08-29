import os
import logging
import pickle

from cosense3d.dataset.pipeline.img import *
from cosense3d.dataset.pipeline.lidar import *
from cosense3d.dataset.pipeline.gt import *
from cosense3d.dataset.pipeline.random import *


class Pipeline(object):
    """Composes several processing modules together.
        Take care that these functions modify the input data directly.
    """

    def __init__(self, cfg_list):
        self.processes = []
        for cfg in cfg_list:
            for k, v in cfg.items():
                cls = globals().get(k, None)
                assert cls is not None, f"Pipeline process node {k} not found."
                self.processes.append(cls(**v))

    def __call__(self, data_dict):
        for p in self.processes:
            p(data_dict)
        return data_dict


