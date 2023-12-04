import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP
from .multi_modal_cav import BaseCAV


class StreamLidarCAV(BaseCAV):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prepare_data_keys = ['points', 'annos_local', 'annos_global']
        self.data['memory'] = {k: [] for k in ['embedding', 'ref_pts', 'time', 'pose', 'velo']}

    def prepare_data(self):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def get_response_cpm(self):
        cpm = {}
        # for k in ['bev_feat']:
        #     if k in self.data:
        #         cpm[k] = self.data[k]
        return cpm

    def forward_local(self, tasks, training_mode):
        if (self.is_ego or self.all_grad) and training_mode:
            grad_mode = 'with_grad'
        else:
            grad_mode = 'no_grad'
        tasks[grad_mode].append((self.id, '01:pts_backbone', {}))
        tasks[grad_mode].append((self.id, '02:backbone_neck', {}))
        tasks[grad_mode].append((self.id, '03:roi_head', {}))
        tasks[grad_mode].append((self.id, '04:memory_updater', {}))

    def forward_fusion(self, tasks, training_mode):
        # if self.is_ego:
        #     tasks['with_grad'].append((self.id, '11:fusion', {}))
        return tasks

    def forward_head(self, tasks, training_mode):
        # if self.is_ego:
        #     tasks['with_grad'].append((self.id, '12:detection_head', {}))
        return tasks

    def loss(self, tasks):
        if self.is_ego:
            tasks['loss'].append((self.id, '22:detection_head', {}))
        return tasks

    def pre_update_memory(self):
        """Update memory before each forward run of a single frame."""
        pass

    def post_update_memory(self):
        """Update memory after each forward run of a single frame."""
        pass




