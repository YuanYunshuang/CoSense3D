import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from cosense3d.agents.cav_prototype.base_cav import BaseCAV


class MultiModalCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['img', 'points', 'annos_global']
        self.img_norm_mean = torch.tensor([123.675, 116.28, 103.53])
        self.img_norm_std_inv = 1 / torch.tensor([58.395, 57.12, 57.375])

    def normalize_imgs(self):
        for i in range(len(self.data['img'])):
            # inplace operation
            self.data['img'][i] -= self.img_norm_mean
            self.data['img'][i] *= self.img_norm_std_inv

    def prepare_data(self):
        self.normalize_imgs()
        DOP.adaptive_free_space_augmentation(self.data, min_h=-1.5)
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def forward_local(self, tasks, training_mode):
        if self.is_ego and training_mode:
            tasks['with_grad'].append((self.id, '1:img_backbone', {}))
            tasks['with_grad'].append((self.id, '2:img_roi', {}))
            tasks['with_grad'].append((self.id, '3:pts_backbone', {}))
            tasks['with_grad'].append((self.id, '4:pts_roi', {}))
        else:
            tasks['no_grad'].append((self.id, '1:img_backbone', {}))
            tasks['no_grad'].append((self.id, '2:img_roi', {}))
            tasks['no_grad'].append((self.id, '3:pts_backbone', {}))
            tasks['no_grad'].append((self.id, '4:pts_roi', {}))

    def forward_fusion(self, tasks, training_mode):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '5:fusion', {}))
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

