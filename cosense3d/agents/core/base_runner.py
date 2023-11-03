import os
from datetime import datetime

from cosense3d.utils.train_utils import *
from cosense3d.utils.logger import LogMeter
from cosense3d.utils.misc import ensure_dir, setup_logger


class BaseRunner:
    def __init__(self,
                 dataloader,
                 controller,
                 gpus=1,
                 log_every=2,
                 logger=None,
                 run_name='default',
                 log_dir='work_dir',
                 use_wandb=False,
                 **kwargs
                 ):
        self.dataloader = dataloader
        self.data_iter = iter(dataloader)
        self.total_iter = len(dataloader)
        self.iter = 0
        self.epoch = 1

        self.controller = controller
        self.forward_runner = controller.forward_runner

        self.gpus = gpus
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.log_every = log_every
        self.setup_logger(logger, run_name, log_dir, use_wandb)

        self.init()

    def init(self):
        self.forward_runner.to(self.device)

    def setup_logger(self, logger, run_name, log_dir, use_wandb):
        if logger is None:
            self.logger = None
            self.wandb_logger = None
        else:
            now = datetime.now().strftime('%m-%d-%H-%M-%S')
            run_name = run_name + '_' + now
            log_path = os.path.join(log_dir, run_name)
            ensure_dir(log_path)
            wandb_project_name = run_name if use_wandb else None
            self.logger = LogMeter(self.total_iter, log_path, log_every=self.log_every,
                                   wandb_project=wandb_project_name)

    def run(self):
        raise NotImplementedError

    def next_batch(self):
        if self.iter < self.total_iter:
            self.iter += 0
        else:
            self.iter = 0
            self.data_iter = iter(self.dataloader)
        batch = next(self.data_iter)
        return batch


