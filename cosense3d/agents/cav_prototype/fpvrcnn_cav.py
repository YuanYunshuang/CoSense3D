import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from .multi_modal_cav import BaseCAV


class FpvrcnnCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_global']

    def prepare_data(self):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def get_response_cpm(self):
        cpm = {}
        for k in ['keypoint_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.all_grad) and training_mode:
            tasks['with_grad'].append((self.id, '01:pts_backbone', {}))
            tasks['with_grad'].append((self.id, '02:detection_head_local', {}))
            # tasks['with_grad'].append((self.id, '03:keypoint_composer', {}))
        else:
            tasks['no_grad'].append((self.id, '01:pts_backbone', {}))
            tasks['no_grad'].append((self.id, '02:detection_head_local', {}))
            # tasks['no_grad'].append((self.id, '03:keypoint_composer', {}))

    def forward_fusion(self, tasks, training_mode):
        # if self.is_ego:
        #     tasks['with_grad'].append((self.id, '11:fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode):
        # if self.is_ego:
        #     tasks['with_grad'].append((self.id, '12:detection_head_global', {}))
        return tasks

    def loss(self, tasks):
        if self.is_ego:
            tasks['loss'].append((self.id, '22:detection_head_local', {}))
            # tasks['loss'].append((self.id, '22:detection_head_global', {}))
        return tasks

    def reset_data(self):
        del self.data
        self.data = {}


