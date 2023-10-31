import torch
from torch import nn

from cosense3d.model import build_module


class ForwardRunner(nn.Module):
    def __init__(self, cfg, data_manager):
        super().__init__()
        self.data_info = data_manager.data_info
        self.lidar_range = torch.tensor(self.data_info['lidar_range'])
        self.data_manager = data_manager

        module_dict = {}
        self.module_keys = []
        for k, v in cfg.items():
            module_dict[k] = build_module(v)
            self.module_keys.append(k)

        self.shared_modules = nn.ModuleDict(module_dict)

    def forward(self, tasks):
        for task_name, task_list in tasks.items():
            func = getattr(self, task_name)
            func(task_list)

    def filter_range(self, tasks):
        for task in tasks:
            cav_id = task[0]
            points = self.data_manager.gather([cav_id], 'points')[0]
            lr = self.lidar_range.to(points.device)
            mask = (points[:, :3] > lr[:3].view(1, 3)) & (points[:, :3] < lr[3:].view(1, 3))
            self.data_manager.update(cav_id, 'points', points[mask.all(dim=-1)])

    def pts_backbone(self, tasks):
        cav_ids = [t[0] for t in tasks]
        point_list = self.data_manager.gather(cav_ids, 'points')
        res = self.shared_modules.pts_backbone(point_list, pad_idx=True)

        for i, cav_id in enumerate(cav_ids):
            cur_res = {}
            for k, v in res.items():
                mask = v.C[:, 0] == i
                cur_res[k] = {
                        'coor': v.C[mask, 1:],
                        'feat': v.F[mask]
                    }
            self.data_manager.update(cav_id, 'pts_feat', cur_res)

