import os
from datetime import datetime

from cosense3d.utils.train_utils import *
from cosense3d.agents.core.hooks import Hooks


class BaseRunner:
    def __init__(self,
                 dataloader,
                 controller,
                 gpus=1,
                 log_every=10,
                 hooks=None,
                 **kwargs
                 ):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.total_iter = len(dataloader)
        self.iter = 1
        self.epoch = 1

        self.controller = controller
        self.forward_runner = controller.forward_runner
        self.hooks = Hooks(hooks)

        self.gpus = gpus
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.log_every = log_every

        self.init()

    def init(self):
        self.forward_runner.to(self.device)

    def setup_logger(self, *args, **kwargs):
        raise NotImplementedError

    def set_logdir(self, logdir):
        self.logger.log_path = logdir

    def run(self):
        raise NotImplementedError

    def next_batch(self):
        if self.iter >= self.total_iter:
            self.iter = 1
            self.epoch += 1
            self.data_iter = iter(self.dataloader)
        batch = next(self.data_iter)
        return batch

    def vis_data(self,
                 with_input=True,
                 with_detection=True,
                 with_bev=True):
        data = {}
        if with_input:
            data['input'] = self.controller.data_manager.get_vis_data_input()
        if with_detection:
            data['detection'] = self.controller.data_manager.get_vis_data_detection()
        if with_bev:
            data['bev'] = self.controller.data_manager.get_vis_data_bev()
        return data


