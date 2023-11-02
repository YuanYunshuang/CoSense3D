import torch
from queue import Queue


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

    def apply_data_transform(self):
        if self.is_ego:
            transform = torch.eye(4).to(self.lidar_pose.device)
        else:
            # cav to ego
            request = self.data['received_request']
            transform = request['lidar_pose'].inverse() @ self.lidar_pose
        # augmentation
        if 'rot' in self.data['augment_params']:
            transform = self.data['augment_params']['rot'].to(transform.device) @ transform

        C = self.data['points'].shape[-1]
        points = self.data['points'][:, :3]
        points_homo = torch.cat([points, torch.ones_like(points[:, :1])], dim=-1).T
        points_homo = transform @ points_homo

        if 'scale' in self.data['augment_params']:
            points_homo[:2] *= self.data['augment_params']['scale'].item()

        if C > 3:
            self.data['points'] = torch.cat([points_homo[:3].T,
                                             self.data['points'][:, 3:]], dim=-1)
        else:
            self.data['points'] = points_homo[:3].T

    def filter_data_range(self):
        points = self.data['points']
        lr = self.lidar_range.to(points.device)
        mask = (points[:, :3] > lr[:3].view(1, 3)) & (points[:, :3] < lr[3:].view(1, 3))
        self.data['points'] = points[mask.all(dim=-1)]

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

    def forward_local(self, tasks):
        self.apply_data_transform()  # 1
        self.filter_data_range()  # 2
        if self.is_ego:
            tasks['with_grad'].append((self.id, '3:pts_backbone', {}))
        else:
            tasks['no_grad'].append((self.id, '3:pts_backbone', {}))

    def forward_fusion(self, tasks):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '4:fusion', {}))
            tasks['with_grad'].append((self.id, '5:fusion_neck', {}))
        return tasks

    def forward_head(self, tasks):
        if self.is_ego:
            tasks['with_grad'].append((self.id, '6:bev_head', {}))
            tasks['with_grad'].append((self.id, '7:detection_head', {}))
        return tasks

    def loss(self):
        tasks = []
        if self.is_ego:
            tasks.append((self.id, '1: bev_head', {}))
            tasks.append((self.id, '2: detection_head', {}))
        return tasks

    def reset_data(self):
        del self.data
        self.data = {}

    def pre_update_memory(self):
        """Update memory before each forward run of a single frame."""
        pass

    def post_update_memory(self):
        """Update memory after each forward run of a single frame."""
        update_keys = ['bev_out', 'detection_out']
        self.memory.append({self.data[k] for k in update_keys})
        if len(self.memory) > self.memory_len:
            self.memory = self.memory[1:]