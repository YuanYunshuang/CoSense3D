import torch
from torch import nn

from cosense3d.modules import build_module


class ForwardRunner(nn.Module):
    def __init__(self, cfg, data_manager):
        super().__init__()
        self.lidar_range = torch.tensor(data_manager.lidar_range)
        self.data_manager = data_manager

        module_dict = {}
        self.module_keys = []
        for k, v in cfg.items():
            module_dict[k] = build_module(v)
            self.module_keys.append(k)

        self.shared_modules = nn.ModuleDict(module_dict)

    def gather_cav_ids(self, tasks):
        return [t[0] for t in tasks]

    def forward(self, tasks):
        for task_name, task_list in tasks.items():
            module = getattr(self.shared_modules, task_name)
            cav_ids = self.gather_cav_ids(task_list)
            data = self.data_manager.gather(cav_ids, module.gather_keys)
            res = module(*data)
            self.data_manager.scatter(cav_ids, res)

    def loss(self, tasks):
        for task_name, task_list in tasks.items():
            module = getattr(self.shared_modules, task_name)
            cav_ids = self.gather_cav_ids(task_list)
            data = self.data_manager.gather(cav_ids, module.scatter_keys)
            res = module.loss(*data)
            self.data_manager.scatter(cav_ids, res)

    def filter_range(self, tasks):
        for task in tasks:
            cav_id = task[0]
            points = self.data_manager.gather([cav_id], 'points')[0]
            lr = self.lidar_range.to(points.device)
            mask = (points[:, :3] > lr[:3].view(1, 3)) & (points[:, :3] < lr[3:].view(1, 3))
            self.data_manager.update(cav_id, 'points', points[mask.all(dim=-1)])

    def pts_backbone(self, tasks):
        cav_ids = self.gather_cav_ids(tasks)
        point_list = self.data_manager.gather(cav_ids, 'points')
        res = self.shared_modules.pts_backbone(point_list, pad_idx=True, to_list=True)
        self.data_manager.scatter(cav_ids, 'pts_feat', res)

    def fusion(self, tasks):
        cav_ids = self.gather_cav_ids(tasks)
        ego_feat = self.data_manager.gather(cav_ids, 'pts_feat')
        coop_feat = self.data_manager.gather(cav_ids, 'received_response')
        res = self.shared_modules.fusion(ego_feat, coop_feat)
        self.data_manager.scatter(cav_ids, 'fused_feat', res)

    def fusion_neck(self, tasks):
        cav_ids = self.gather_cav_ids(tasks)
        fused_feat = self.data_manager.gather(cav_ids, 'fused_feat')
        res = self.shared_modules.fusion_neck(fused_feat)
        self.data_manager.scatter(cav_ids, 'fused_neck_feat', res)

    def bev_head(self, tasks):
        cav_ids = self.gather_cav_ids(tasks)
        fused_neck_feat = self.data_manager.gather(cav_ids, 'fused_neck_feat')
        res = self.shared_modules.bev_head(fused_neck_feat)
        self.data_manager.scatter(cav_ids, 'bev_out', res)

    def detection_head(self, tasks):
        cav_ids = self.gather_cav_ids(tasks)
        fused_neck_feat = self.data_manager.gather(cav_ids, 'fused_neck_feat')
        res = self.shared_modules.detection_head(fused_neck_feat)
        self.data_manager.scatter(cav_ids, 'detection_out', res)


