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
        self.prepare_data_keys = ['points', 'annos_global', 'annos_local']

    def update(self, lidar_pose):
        self.lidar_pose = lidar_pose

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(id={self.id}, '
        repr_str += f'is_ego={self.is_ego}, '
        repr_str += f'data={self.data.keys()})'
        return repr_str

    def apply_transform(self):
        if self.is_ego:
            transform = torch.eye(4).to(self.lidar_pose.device)
        else:
            # cav to ego
            request = self.data['received_request']
            transform = request['lidar_pose'].inverse() @ self.lidar_pose
        DOP.cav_aug_transform(self.data, transform, self.data['augment_params'],
                              apply_to=self.prepare_data_keys)

    def prepare_data(self):
        self.apply_transform()
        DOP.filter_range(self.data, self.lidar_range, apply_to=self.prepare_data_keys)

    def has_request(self):
        if 'received_request' in self.data and self.data['received_request'] is not None:
            return True
        else:
            return False

    def get_request_cpm(self):
        return {'lidar_pose': self.lidar_pose}

    def get_response_cpm(self):
        cpm = {}
        for k in ['points']:
            if k in self.data:
                cpm[k] = self.data[k]
        return cpm

    def receive_request(self, request):
        self.data['received_request'] = request

    def receive_response(self, response):
        self.data['received_response'] = response

    def forward(self, tasks, training_mode):
        self.prepare_data()
        self.forward_local(tasks, training_mode)
        self.forward_fusion(tasks, training_mode)
        self.forward_head(tasks, training_mode)
        return tasks

    def forward_local(self, tasks, training_mode):
        """To be overloaded."""
        return tasks

    def forward_fusion(self, tasks, training_mode):
        """To be overloaded."""
        return tasks

    def forward_head(self, tasks, training_mode):
        """To be overloaded."""
        return tasks

    def loss(self, tasks):
        """To be overloaded."""
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

