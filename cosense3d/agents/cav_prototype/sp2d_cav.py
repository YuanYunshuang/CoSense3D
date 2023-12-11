import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from .multi_modal_cav import BaseCAV


class Sp2DCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_global']

    def prepare_data(self):
        DOP.adaptive_free_space_augmentation(self.data)
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def get_response_cpm(self):
        cpm = {}
        for k in ['pts_feat']:
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
            tasks['with_grad'].append((self.id, '12:fusion_neck', {}))
        return tasks

    def forward_head(self, tasks, training_mode):
        if self.is_ego:
            # tasks['with_grad'].append((self.id, '13:bev_head', {}))
            tasks['with_grad'].append((self.id, '14:detection_head', {}))
        return tasks

    def loss(self, tasks):
        if self.is_ego:
            # tasks['loss'].append((self.id, '21:bev_head', {}))
            tasks['loss'].append((self.id, '22:detection_head', {}))
        return tasks

    def reset_data(self):
        del self.data
        self.data = {}

    def pre_update_memory(self):
        """Update memory before each forward run of a single frame."""
        pass

    def post_update_memory(self):
        """Update memory after each forward run of a single frame."""
        pass
        # if self.is_ego:
        #     update_keys = ['bev', 'detection']
        #     self.memory.append({k: self.data[k] for k in update_keys})
        #     if len(self.memory) > self.memory_len:
        #         self.memory = self.memory[1:]

