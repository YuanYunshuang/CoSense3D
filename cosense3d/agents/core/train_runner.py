import os
from datetime import datetime

from cosense3d.utils.train_utils import *
from cosense3d.utils.logger import LogMeter
from cosense3d.utils.misc import ensure_dir, setup_logger
from cosense3d.agents.core.base_runner import BaseRunner


class TrainRunner(BaseRunner):
    def __init__(self,
                 max_epoch,
                 optimizer,
                 lr_scheduler,
                 resume=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.optimizer = build_optimizer(self.forward_runner, optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, lr_scheduler,
                                               len(self.dataloader))
        self.total_epochs = max_epoch
        self.start_epoch = 1

    def run(self):
        with torch.autograd.set_detect_anomaly(True):
            for i in range(self.start_epoch, self.total_epochs + 1):
                self.run_epoch(i)
                self.epoch = i
                self.iter = 0

    def step(self):
        data = self.next_batch()
        self.run_itr(data)

    def run_epoch(self, epoch):
        for data in self.dataloader:
            self.run_itr(data)
            self.lr_scheduler.step(epoch)

    def run_itr(self, data):
        load_tensors_to_gpu(data)
        self.optimizer.zero_grad()

        total_loss, loss_dict = self.controller.train_forward(data)
        total_loss.backward()
        # grad_norm = clip_grads(self.controller.parameters)
        # Updating parameters
        self.optimizer.step()

        if self.logger is not None:
            rec_lr = self.lr_scheduler.optimizer.param_groups[0]['lr']
            self.logger.log(self.epoch, self.iter, rec_lr, **loss_dict)

        del data
        torch.cuda.empty_cache()

        self.iter += 1




