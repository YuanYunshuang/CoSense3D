import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from .multi_modal_cav import BaseCAV


class LidarDetCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_global']

    def prepare_data(self):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def get_response_cpm(self):
        cpm = {}
        for k in ['bev_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.require_grad) and training_mode:
            tasks['with_grad'].append((self.id, '01:pts_backbone', {}))
        else:
            tasks['no_grad'].append((self.id, '01:pts_backbone', {}))

    def forward_fusion(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '11:fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '12:detection_head', {}))
        return tasks

    def loss(self, tasks):
        if self.is_ego:
            tasks['loss'].append((self.id, '22:detection_head', {}))
        return tasks

    def reset_data(self):
        del self.data
        self.data = {}


class LidarDetCAVLateFusion(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_local', 'annos_global']

    def prepare_data(self):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def get_response_cpm(self):
        cpm = {}
        for k in ['detection']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.require_grad) and training_mode:
            tasks['with_grad'].append((self.id, '01:pts_backbone', {}))
            tasks['with_grad'].append((self.id, '02:detection_head', {}))
        else:
            tasks['no_grad'].append((self.id, '01:pts_backbone', {}))
            tasks['no_grad'].append((self.id, '02:detection_head', {}))

    def forward_fusion(self, tasks, training_mode):
        if self.is_ego and not training_mode:
            tasks['no_grad'].append((self.id, '11:fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode):
        return tasks

    def loss(self, tasks):
        if (self.is_ego or self.require_grad):
            tasks['loss'].append((self.id, '22:detection_head', {}))
        return tasks

    def reset_data(self):
        del self.data
        self.data = {}


