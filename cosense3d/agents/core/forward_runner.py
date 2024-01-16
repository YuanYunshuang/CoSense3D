import torch
from torch import nn

from cosense3d.modules import build_module


class ForwardRunner(nn.Module):
    def __init__(self, shared_modules, data_manager, dist=False):
        super().__init__()
        self.lidar_range = torch.tensor(data_manager.lidar_range)
        self.data_manager = data_manager
        self.dist = dist

        module_dict = {}
        self.module_keys = []
        for k, v in shared_modules.items():
            v['dist'] = dist
            module_dict[k] = build_module(v)
            self.module_keys.append(k)

        self.shared_modules = nn.ModuleDict(module_dict)

    def gather_cav_ids(self, tasks):
        return [t[0] for t in tasks]

    def forward(self, tasks, with_grad=True, **kwargs):
        if with_grad:
            self._forward(tasks, **kwargs)
        else:
            with torch.no_grad():
                self._forward(tasks, **kwargs)

    def _forward(self, tasks, **kwargs):
        for task_name, task_list in tasks.items():
            module = getattr(self.shared_modules, task_name)
            task_ids = self.gather_cav_ids(task_list)
            data = self.data_manager.gather(task_ids, module.gather_keys)
            res = module(*data, **kwargs)
            self.data_manager.scatter(task_ids, res)

    def loss(self, tasks, **kwargs):
        loss_dict = {}
        loss = 0
        for task_name, task_list in tasks.items():
            module = getattr(self.shared_modules, task_name)
            cav_ids = self.gather_cav_ids(task_list)
            data = self.data_manager.gather(cav_ids, module.scatter_keys + module.gt_keys)
            ldict = module.loss(*data, **kwargs)
            for k, v in ldict.items():
                prefix = task_name.replace('_head', '')
                loss_dict[f'{prefix}.{k}'] = v
                loss = loss + v
        loss_dict['total_loss'] = loss
        return loss, loss_dict

    def frame_loss(self, tasks, **kwargs):
        loss_dict = {}
        for task_name, task_list in tasks.items():
            module = getattr(self.shared_modules, task_name)
            cav_ids = self.gather_cav_ids(task_list)
            data = self.data_manager.gather(cav_ids, module.scatter_keys + module.gt_keys)
            ldict = module.loss(*data, **kwargs)
            for k, v in ldict.items():
                prefix = task_name.replace('_head', '')
                loss_dict[f'{prefix}.{k}'] = v
        return loss_dict





