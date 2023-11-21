from torch.optim import lr_scheduler as torch_lr


def build_lr_scheduler(optimizer, cfg, total_iter):
    return LRUpdater(optimizer, total_iter, **cfg)


class LRUpdater:
    """
    Unified API for updating LR with different LR schedulers.
    """
    def __init__(self, optimizer, total_iter, policy, **kwargs):
        self.policy = policy
        self.total_itr = total_iter
        if policy == 'MultiStepLR':
            # construct a learning rate scheduler
            self.lr_scheduler = torch_lr.MultiStepLR(optimizer, **kwargs)
        elif policy == 'CosineAnnealingWarm':
            from timm.scheduler.cosine_lr import CosineLRScheduler
            num_steps = kwargs['epochs'] * total_iter
            warmup_lr = kwargs['warmup_lr']
            warmup_steps = kwargs['warmup_epochs'] * total_iter
            lr_min = kwargs['lr_min']

            self.lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=lr_min,
                warmup_lr_init=warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
            )
        else:
            raise NotImplementedError

        self.optimizer = self.lr_scheduler.optimizer

    def step(self, epoch):
        if self.policy in ['CosineAnnealingWarm',]:
            self.lr_scheduler.step(epoch)
        else:
            self.lr_scheduler.step()

    def state_dict(self):
        return self.lr_scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.lr_scheduler.load_state_dict(state_dict)
