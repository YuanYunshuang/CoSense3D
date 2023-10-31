import os
from datetime import datetime

from cosense3d.utils.train_utils import *
from cosense3d.utils.logger import LogMeter
from cosense3d.utils.misc import ensure_dir, setup_logger


class TrainRunner:
    def __init__(self,
                 dataloader,
                 controller,
                 max_epoch,
                 optimizer,
                 lr_scheduler,
                 gpus=1,
                 log_every=2,
                 resume=False,
                 logger=None,
                 run_name='default',
                 log_dir='work_dir',
                 use_wandb=False,
                 **kwargs
                 ):
        self.dataloader = dataloader
        self.controller = controller
        self.forward_runner = controller.forward_runner
        self.optimizer = build_optimizer(self.forward_runner, optimizer)
        self.lr_scheduler = build_lr_scheduler(self.optimizer, lr_scheduler,
                                               len(dataloader))
        self.total_epochs = max_epoch
        self.total_iter = len(dataloader)
        self.start_epoch = 1
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
        for i in range(self.start_epoch, self.total_epochs + 1):
            self.forward_runner.train()
            self.run_epoch()

    def run_epoch(self):
        for data in self.dataloader:
            self.run_itr(data)
            self.lr_scheduler.step()

    def run_itr(self, data):
        load_tensors_to_gpu(data)
        self.optimizer.zero_grad()

        out = self.controller.run_seq(data)

        total_loss, loss_dict = self.controller.loss(out)
        total_loss.backward()
        grad_norm = clip_grads(self.controller.parameters)
        # Updating parameters
        self.optimizer.step()

        del data
        torch.cuda.empty_cache()

