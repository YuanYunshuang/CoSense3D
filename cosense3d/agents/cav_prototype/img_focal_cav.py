import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from .multi_modal_cav import BaseCAV


class ImgFocalCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['img', 'annos_global']
        self.img_norm_mean = torch.tensor([123.675, 116.28, 103.53]).cuda()
        self.img_norm_std_inv = 1 / torch.tensor([58.395, 57.12, 57.375]).cuda()

    def normalize_imgs(self):
        for i in range(len(self.data['img'])):
            # inplace operation
            self.data['img'][i] -= self.img_norm_mean
            self.data['img'][i] *= self.img_norm_std_inv

    def prepare_data(self):
        self.normalize_imgs()
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)
        self.data['img_size'] = [x.shape[:2] for x in self.data['img']]

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.require_grad) and training_mode:
            tasks['with_grad'].append((self.id, '01:img_backbone', {}))
            tasks['with_grad'].append((self.id, '02:img_roi', {}))
            tasks['with_grad'].append((self.id, '03:img2bev', {}))
        else:
            tasks['no_grad'].append((self.id, '01:img_backbone', {}))
            tasks['no_grad'].append((self.id, '02:img_roi', {}))
            tasks['no_grad'].append((self.id, '03:img2bev', {}))

    def forward_fusion(self, tasks, training_mode):
        pass
        return tasks

    def forward_head(self, tasks, training_mode):
        pass
        return tasks

    def loss(self, tasks):
        if self.is_ego:
            tasks['loss'].append((self.id, '21:img_roi', {}))
        return tasks


