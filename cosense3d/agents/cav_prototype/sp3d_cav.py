import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from .base_cav import BaseCAV


class Sp3DCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward_local(self, tasks, training_mode):
        self.prepare_data(keys=['points', 'annos_global'])
        if self.is_ego and training_mode:
            tasks['with_grad'].append((self.id, '3:pts_backbone', {}))
        else:
            tasks['no_grad'].append((self.id, '3:pts_backbone', {}))

    def forward_fusion(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '4:fusion', {}))
            tasks['with_grad'].append((self.id, '5:fusion_neck', {}))
        return tasks

    def forward_head(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '6:bev_head', {}))
            tasks['with_grad'].append((self.id, '7:detection_head', {}))
        return tasks

    def loss(self, tasks):
        if self.is_ego:
            tasks['loss'].append((self.id, '1:bev_head', {}))
            tasks['loss'].append((self.id, '2:detection_head', {}))
        return tasks

    def reset_data(self):
        del self.data
        self.data = {}

    def pre_update_memory(self):
        """Update memory before each forward run of a single frame."""
        pass

    def post_update_memory(self):
        """Update memory after each forward run of a single frame."""
        if self.is_ego:
            update_keys = ['bev', 'detection']
            self.memory.append({k: self.data[k] for k in update_keys})
            if len(self.memory) > self.memory_len:
                self.memory = self.memory[1:]

