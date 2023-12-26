from torch.optim import lr_scheduler as torch_lr
from torch.optim import Optimizer


def build_lr_scheduler(optimizer, cfg, total_iter):
    return LRUpdater(optimizer, total_iter, **cfg)


class TransformerAdaptiveScheduler(torch_lr._LRScheduler):
    def __init__(self,
                 optimizer: Optimizer,
                 dim_embed: int,
                 warmup_steps: int,
                 last_epoch: int = -1,
                 verbose: bool = False) -> None:
        self.dim_embed = dim_embed
        self.warmup_steps = warmup_steps
        self.num_param_groups = len(optimizer.param_groups)

        super().__init__(optimizer, last_epoch, verbose)

    def get_lr(self) -> float:
        lr = self.calc_lr(self._step_count, self.dim_embed, self.warmup_steps)
        return [lr] * self.num_param_groups

    def calc_lr(self, step, dim_embed, warmup_steps):
        return dim_embed ** (-0.5) * min(step ** (-0.5), step * warmup_steps ** (-1.5))


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
            decay_rate = kwargs.get('decay_rate', 0.5)

            self.lr_scheduler = CosineLRScheduler(
                optimizer,
                t_initial=num_steps,
                lr_min=lr_min,
                warmup_lr_init=warmup_lr,
                warmup_t=warmup_steps,
                cycle_limit=1,
                t_in_epochs=False,
                cycle_decay=decay_rate
            )
        elif policy == 'TransformerAdaptiveScheduler':
            TransformerAdaptiveScheduler(optimizer, **kwargs)
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
