import torch
from cosense3d.agents.utils.transform import DataOnlineProcessor as DOP


class BaseCAV:
    def __init__(self, id, mapped_id, is_ego, lidar_pose, lidar_range, memery_len):
        self.id = id
        self.mapped_id = mapped_id
        self.is_ego = is_ego
        self.lidar_pose = lidar_pose
        self.lidar_range = lidar_range
        self.memory_len = memery_len
        self.data = {}
        self.memory = []  # FIFO

    def update(self, lidar_pose):
        self.lidar_pose = lidar_pose

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(id={self.id}, '
        repr_str += f'is_ego={self.is_ego}, '
        repr_str += f'data={self.data.keys()})'
        return repr_str

    def apply_transform(self, apply_to=['points', 'annos_global']):
        if self.is_ego:
            transform = torch.eye(4).to(self.lidar_pose.device)
        else:
            # cav to ego
            request = self.data['received_request']
            transform = request['lidar_pose'].inverse() @ self.lidar_pose
        DOP.cav_aug_transform(self.data, transform, self.data['augment_params'], apply_to=apply_to)

    def prepare_data(self, keys=['points', 'annos_global']):
        self.apply_transform(keys)
        DOP.filter_range(self.data, self.lidar_range, apply_to=keys)

    def has_request(self):
        if 'received_request' in self.data and self.data['received_request'] is not None:
            return True
        else:
            return False

    def get_request_cpm(self):
        return {'lidar_pose': self.lidar_pose}

    def get_response_cpm(self):
        cpm = {}
        for k in ['pts_feat']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def receive_request(self, request):
        self.data['received_request'] = request

    def receive_response(self, response):
        self.data['received_response'] = response

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

